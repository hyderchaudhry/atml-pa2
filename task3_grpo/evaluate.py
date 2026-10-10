from __future__ import annotations

import argparse

import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import score_reward_pairs
from common.logging_utils import append_jsonl, load_json, save_json, set_seed
from common.metrics import sample_entropy, sampled_kl
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from task3_grpo.continue_train import generate_rollout, prompt_identity, rollout_logprobs, run_metadata


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    print(f"Loading held-out policy: {adapter}; tokenizer and reward model", flush=True)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def evaluate(config_path: str, adapter: str, name: str = "standard"):
    print(f"Beginning held-out GRPO evaluation: {name}", flush=True)
    cfg = load_yaml(config_path)
    result_dir = repo_path(cfg["results_dir"]) / name
    if any((result_dir / f).exists() for f in ("metrics.json", "generations.jsonl")):
        raise FileExistsError(f"Refusing to overwrite evaluation: {result_dir}")
    manifest_path = result_dir / "training_manifest.json"
    training = load_json(manifest_path) if manifest_path.exists() else None
    if training and repo_path(training["output"]).resolve() != repo_path(adapter).resolve():
        raise ValueError("Evaluation adapter does not match the training manifest")
    bundle = load_evaluation_bundle(config_path, adapter)
    if not bundle["rows"]:
        raise ValueError("The fixed held-out prompt pool is empty")
    policy = bundle["policy"]
    policy.eval()
    reward_model, reward_tokenizer = bundle["reward"]
    set_seed(int(cfg["seed"]))
    save_json(result_dir / "evaluation_manifest.json", {
        **run_metadata(training["config"] if training else cfg,
                       training["loss_type"] if training else "grpo"),
        "evaluation_config": cfg, "adapter": adapter, "name": name,
        "prompt_ids": [prompt_identity(row, i) for i, row in enumerate(bundle["rows"])],
        "generation_cap": int(cfg["max_completion_length"]),
        "training_manifest": str(manifest_path) if training else None,
        "mask": "all generated nonpadding response tokens; training truncation masking is not used in evaluation",
    })
    records = []
    for index, row in enumerate(bundle["rows"]):
        print(f"Evaluation {name}: generating/scoring prompt {index + 1}/{len(bundle['rows'])}", flush=True)
        prompts = [prompt_messages(row)]
        rollout = generate_rollout(policy, bundle["tokenizer"], prompts, cfg)
        mask = rollout["response_mask"]
        with torch.no_grad():
            logp = rollout_logprobs(policy, rollout)
            with reference_mode(policy):
                ref_logp = rollout_logprobs(policy, rollout)
            rewards = score_reward_pairs(reward_model, reward_tokenizer, prompts, rollout["responses"])
        record = {
            **prompt_identity(row, index), "prompt": prompts[0],
            "response": rollout["responses"][0],
            "response_token_ids": rollout["response_ids"][0, mask[0].bool()].tolist(),
            "learned_reward": float(rewards[0]), "kl": float(sampled_kl(logp, ref_logp, mask)),
            "entropy": float(sample_entropy(logp, mask)),
            "response_length": rollout["response_lengths"][0],
            "length_bin": "short" if rollout["response_lengths"][0] <= int(cfg["max_completion_length"]) / 2 else "long",
            "terminated_with_eos": rollout["terminated_with_eos"][0],
            "truncated": rollout["truncated"][0],
        }
        append_jsonl(result_dir / "generations.jsonl", record)
        records.append(record)
        print(f"Evaluation {name}: {index + 1}/{len(bundle['rows'])}", flush=True)
    metrics = summarize_generations(records)
    metrics["by_length"] = {
        label: summarize_generations([r for r in records if r["length_bin"] == label])
        for label in ("short", "long")
    }
    save_json(result_dir / "metrics.json", metrics)
    print(f"Completed evaluation {name} | saved evidence: {result_dir}", flush=True)
    return metrics


def summarize_generations(records):
    if not records:
        return {"num_prompts": 0, "learned_reward": None, "kl": None, "entropy": None,
                "response_length": None, "response_length_std": None}
    lengths = torch.tensor([r["response_length"] for r in records], dtype=torch.float64)
    tokens = float(lengths.sum())
    return {
        "num_prompts": len(records), "valid_response_tokens": int(tokens),
        "learned_reward": sum(r["learned_reward"] for r in records) / len(records),
        "kl": sum(r["kl"] * r["response_length"] for r in records) / max(tokens, 1),
        "entropy": sum(r["entropy"] * r["response_length"] for r in records) / max(tokens, 1),
        "response_length": float(lengths.mean()), "response_length_std": float(lengths.std(unbiased=False)),
        "eos_rate": sum(r["terminated_with_eos"] for r in records) / len(records),
        "truncation_rate": sum(r["truncated"] for r in records) / len(records),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    evaluate(args.config, args.adapter, args.name)


if __name__ == "__main__":
    main()
