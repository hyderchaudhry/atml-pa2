from __future__ import annotations

import argparse
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import save_json, set_seed
from common.metrics import masked_mean
from common.models import load_policy, load_tokenizer
from task2_ppo.continue_train import prompt_identity, rollout_logprobs, run_forks, run_metadata
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "prompt_id", "response", "response_tokens", "terminated_with_eos",
                "old_logprobs", "ref_logprobs", "values", "effective_terminal_reward"}
    if any(not required.issubset(row) for row in normalized):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def cached_tensors(rows, cfg):
    """Use cached critic/reward data unchanged; right padding is excluded from every statistic."""
    width = max(int(row["response_tokens"]) for row in rows)
    tensors = {k: torch.zeros(len(rows), width) for k in ("old", "ref", "values", "mask")}
    terminal = []
    for i, row in enumerate(rows):
        n = int(row["response_tokens"])
        for dest, source in (("old", "old_logprobs"), ("ref", "ref_logprobs"), ("values", "values")):
            value = torch.as_tensor(row[source], dtype=torch.float32)
            if value.shape != (n,) or n < 1:
                raise ValueError(f"Cache row {i}: {source} is not aligned with response_tokens")
            tensors[dest][i, :n] = value
        tensors["mask"][i, :n] = 1
        terminal.append(float(row["effective_terminal_reward"]))
    rewards = shaped_rewards(torch.tensor(terminal), tensors["old"], tensors["ref"],
                             tensors["mask"], cfg["kl_beta"])
    advantages, returns = compute_gae(rewards, tensors["values"], tensors["mask"],
                                     gamma=cfg["gamma"], lam=cfg["gae_lambda"])
    tensors["advantages"] = normalize_advantages(advantages, tensors["mask"])
    tensors["returns"] = returns
    return tensors


def reconstruct_cached_response(tokenizer, row, prompt_row, cfg):
    if row["prompt_id"] != prompt_row["prompt_id"] or row["source_index"] != prompt_row["source_index"]:
        raise ValueError("Cached prompt identity does not match the released prompt pool")
    rendered = tokenizer.apply_chat_template(prompt_messages(prompt_row), tokenize=False, add_generation_prompt=True)
    prompt = tokenizer(rendered, truncation=True, max_length=int(cfg["max_prompt_length"]),
                       return_tensors="pt")
    ids = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
    if row["terminated_with_eos"]:
        ids.append(tokenizer.eos_token_id)
    if len(ids) != int(row["response_tokens"]):
        raise ValueError("Cached response cannot be aligned exactly; refusing to change cached token statistics")
    response = torch.tensor([ids], dtype=torch.long)
    return {
        "sequences": torch.cat((prompt["input_ids"], response), dim=1),
        "attention_mask": torch.cat((prompt["attention_mask"], torch.ones_like(response)), dim=1),
        "prompt_width": prompt["input_ids"].shape[1], "response_ids": response,
    }


def clipping_statistics(new_logp, tensors, epsilon):
    loss, ratio, clip_fraction = ppo_policy_loss(
        new_logp, tensors["old"], tensors["advantages"], tensors["mask"], eps=epsilon,
    )
    outside = (ratio < 1 - epsilon) | (ratio > 1 + epsilon)
    # A ratio can be outside the interval without changing the pessimistic surrogate.
    constrained = ((ratio > 1 + epsilon) & (tensors["advantages"] > 0)) | (
        (ratio < 1 - epsilon) & (tensors["advantages"] < 0)
    )
    return {
        "epsilon": epsilon, "policy_loss": float(loss), "clipped_surrogate": float(-loss),
        "unclipped_surrogate": float(masked_mean(ratio * tensors["advantages"], tensors["mask"])),
        "clip_fraction": float(clip_fraction),
        "affected_token_fraction": float(masked_mean(outside.float(), tensors["mask"])),
        "surrogate_constrained_fraction": float(masked_mean(constrained.float(), tensors["mask"])),
    }


def analyze_cached(config_path, adapter=None):
    print("[cached] Starting cached-rollout clipping diagnostic", flush=True)
    cfg = load_yaml(config_path)
    adapter = adapter or cfg["output"]
    result_path = repo_path(cfg["results_dir"]) / "clipping" / "cached_rollout.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite cached analysis: {result_path}")
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    tensors = cached_tensors(rows, cfg)
    prompt_rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    prompts = {row["prompt_id"]: row for row in prompt_rows}
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    policy.eval()
    device = next(policy.parameters()).device
    new_logp = torch.zeros_like(tensors["old"])
    with torch.no_grad():
        for i, row in enumerate(rows):
            rollout = reconstruct_cached_response(tokenizer, row, prompts[row["prompt_id"]], cfg)
            rollout = {k: v.to(device) if torch.is_tensor(v) else v for k, v in rollout.items()}
            new_logp[i, :int(row["response_tokens"])] = rollout_logprobs(policy, rollout)[0].cpu()
    conditions = []
    for eps in cfg["clip_values"]:
        print(f"[cached] Processing epsilon={eps}", flush=True)
        conditions.append(clipping_statistics(new_logp, tensors, float(eps)))
        print(f"[cached] Completed epsilon={eps}", flush=True)
    save_json(result_path, {
        **run_metadata(cfg), "cached_rollouts": cfg["cached_rollouts"],
        "candidate_adapter": adapter,
        "ratio_definition": "candidate policy / cached behavior policy on the unchanged cached responses",
        "affected_token_definition": "ratio outside [1-epsilon, 1+epsilon], same mask as clip_fraction",
        "advantage_definition": "cached values and effective rewards; GAE then normalization over all valid cache tokens",
        "prompt_ids": [prompt_identity(row, i) for i, row in enumerate(rows)],
        "valid_response_tokens": int(tensors["mask"].sum()),
        "conditions": conditions,
    })
    print(f"[cached] Complete | saved results: {result_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", help="Candidate adapter for cached scoring (default: final standard PPO)")
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--cached-only", action="store_true")
    modes.add_argument("--forks-only", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if not args.forks_only:
        analyze_cached(args.config, args.adapter)
    if not args.cached_only:
        run_forks(args.config, "--clip-epsilon", cfg["clip_values"], "clipping")


if __name__ == "__main__":
    main()
