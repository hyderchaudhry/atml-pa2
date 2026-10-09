from __future__ import annotations

import argparse

import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import score_reward_pairs
from common.logging_utils import append_jsonl, load_json, save_json, set_seed
from common.metrics import mean_response_length, sample_entropy, sampled_kl
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from task2_ppo.continue_train import (
    generate_rollout, prompt_identity, rollout_logprobs, run_metadata,
)


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def evaluate(config_path: str, adapter: str, name: str = "standard"):
    cfg = load_yaml(config_path)
    result_dir = repo_path(cfg["results_dir"]) / name
    if any((result_dir / f).exists() for f in ("metrics.json", "generations.jsonl")):
        raise FileExistsError(f"Refusing to overwrite held-out results: {result_dir}")
    training_path = result_dir / "training_manifest.json"
    training = load_json(training_path) if training_path.exists() else None
    if training and repo_path(training["output"]).resolve() != repo_path(adapter).resolve():
        raise ValueError("Evaluation adapter does not match this run's training manifest")
    bundle = load_evaluation_bundle(config_path, adapter)
    # Read the fork's effective configuration rather than labeling every run with default beta/epsilon.
    metadata_cfg = training["config"] if training else cfg
    set_seed(int(cfg["seed"]))
    policy = bundle["policy"]
    policy.eval()
    reward_model, reward_tokenizer = bundle["reward"]
    save_json(result_dir / "evaluation_manifest.json", {
        **run_metadata(metadata_cfg), "adapter": adapter, "name": name,
        "evaluation_config": cfg,
        "prompt_ids": [prompt_identity(row, i) for i, row in enumerate(bundle["rows"])],
        "generation_cap": int(cfg["eval_max_response_length"]),
        "training_manifest": str(training_path) if training else None,
    })
    records = []
    # Singleton batches keep padding and memory use fixed across all held-out conditions.
    for index, row in enumerate(bundle["rows"]):
        prompts = [prompt_messages(row)]
        rollout = generate_rollout(policy, bundle["tokenizer"], prompts, cfg, evaluation=True)
        mask = rollout["response_mask"]
        with torch.no_grad():
            policy_logp = rollout_logprobs(policy, rollout)
            with reference_mode(policy):
                ref_logp = rollout_logprobs(policy, rollout)
            reward = score_reward_pairs(
                reward_model, reward_tokenizer, prompts, rollout["responses"],
                max_length=int(cfg["reward_max_length"]),
            )
        record = {
            **prompt_identity(row, index), "prompt": prompts[0],
            "response": rollout["responses"][0],
            "response_token_ids": rollout["response_ids"][0, mask[0].bool()].tolist(),
            "learned_reward": float(reward[0]),
            "kl": float(sampled_kl(policy_logp, ref_logp, mask)),
            "entropy": float(sample_entropy(policy_logp, mask)),
            "response_length": mean_response_length(mask),
            "terminated_with_eos": rollout["terminated_with_eos"][0],
            "truncated": rollout["truncated"][0],
        }
        append_jsonl(result_dir / "generations.jsonl", record)
        records.append(record)
    if not records:
        raise ValueError("The fixed held-out prompt pool is empty")
    lengths = torch.tensor([r["response_length"] for r in records], dtype=torch.float64)
    tokens = float(lengths.sum())
    metrics = {
        "num_prompts": len(records), "valid_response_tokens": int(tokens),
        "learned_reward": sum(r["learned_reward"] for r in records) / len(records),
        # Weight sequence helper outputs by valid tokens to preserve the released token mean.
        "kl": sum(r["kl"] * r["response_length"] for r in records) / tokens,
        "entropy": sum(r["entropy"] * r["response_length"] for r in records) / tokens,
        "response_length": float(lengths.mean()), "response_length_std": float(lengths.std(unbiased=False)),
        "eos_rate": sum(r["terminated_with_eos"] for r in records) / len(records),
        "truncation_rate": sum(r["truncated"] for r in records) / len(records),
        "epsilon": float(metadata_cfg["clip_epsilon"]), "beta_kl": float(metadata_cfg["kl_beta"]),
        "manifest": str(result_dir / "evaluation_manifest.json"),
    }
    save_json(result_dir / "metrics.json", metrics)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    evaluate(args.config, args.adapter, args.name)


if __name__ == "__main__":
    main()
