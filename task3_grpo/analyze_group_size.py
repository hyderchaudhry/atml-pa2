from __future__ import annotations

import argparse
from collections import defaultdict
import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json, set_seed
from task3_grpo.grpo import ADVANTAGE_EPS, group_relative_advantages


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) != 8}
    if bad:
        raise ValueError(f"Expected exactly K=8 cached completions per prompt: {bad}")
    if not by_prompt:
        raise ValueError("The supplied completion cache is empty")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
        if [int(row["generation_index"]) for row in group] != list(range(8)):
            raise ValueError("Expected unique generation indices 0..7")
        if len({row["prompt_id"] for row in group}) != 1:
            raise ValueError("A cache source_index must identify one prompt")
        if not all(np.isfinite(float(row["reward"])) for row in group):
            raise ValueError("Cache rewards must be finite")
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int):
    """Return K-sized groups while keeping total cached completions fixed.

    Partition each prompt's ordered eight samples into disjoint contiguous chunks.
    Every condition uses every cached completion exactly once and never mixes prompts.
    """
    if k not in (2, 4, 8):
        raise ValueError("Expected K in {2,4,8}")
    groups = []
    for rows in by_prompt.values():
        if len(rows) != 8:
            raise ValueError("Regrouping requires exactly eight completions per prompt")
        groups.extend([rows[start:start + k] for start in range(0, 8, k)])
    return groups


def group_statistics(groups):
    if not groups:
        return {"num_groups": 0, "num_completions": 0, "informative_group_rate": None,
                "uninformative_group_fraction": None, "mean_reward_std": None,
                "relative_signal_variance": None}
    stds, signals = [], []
    for rows in groups:
        rewards = torch.tensor([row["reward"] for row in rows], dtype=torch.float64)
        stds.append(float(rewards.std(unbiased=False)))
        signals.extend(group_relative_advantages(rewards, torch.zeros(len(rows), dtype=torch.long)).tolist())
    informative = float(np.mean(np.asarray(stds) > ADVANTAGE_EPS))
    return {"num_groups": len(groups), "num_completions": len(signals),
            "informative_group_rate": informative, "uninformative_group_fraction": 1 - informative,
            "mean_reward_std": float(np.mean(stds)), "relative_signal_variance": float(np.var(signals))}


def analyze_group_size(config_path):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    result_dir = repo_path(cfg["results_dir"]) / "group_size"
    if result_dir.exists():
        raise FileExistsError(f"Refusing to overwrite group-size evidence: {result_dir}")
    print(f"Loading fixed K=8 cache: {cfg['group_cache']}", flush=True)
    by_prompt = load_k8_cache(cfg["group_cache"])
    print(f"Loaded {len(by_prompt)} prompts, {8 * len(by_prompt)} completions", flush=True)
    if len(by_prompt) < 2:
        raise ValueError("At least two prompts are required for two difficulty bins")
    # Reward-model score is a difficulty proxy, not an externally verified success rate.
    means = {pid: float(np.mean([row["reward"] for row in rows])) for pid, rows in by_prompt.items()}
    order = sorted(by_prompt, key=lambda pid: (means[pid], pid))
    midpoint = len(order) // 2
    bins = {pid: "lower_reward" if rank < midpoint else "higher_reward" for rank, pid in enumerate(order)}
    prompt_rows = {str(row["source_index"]): row for row in read_jsonl(cfg["paths"]["rl_prompt_eval"])}
    for pid, rows in by_prompt.items():
        if pid not in prompt_rows or prompt_rows[pid]["prompt_id"] != rows[0]["prompt_id"]:
            raise ValueError("Cached prompt identity does not match the fixed held-out pool")
    conditions, evidence = [], []
    for k in cfg["group_sizes"]:
        print(f"Analyzing K={k} at the same fixed generation budget", flush=True)
        groups = regroup_equal_generation_budget(by_prompt, int(k))
        conditions.append({"k": int(k), **group_statistics(groups), "by_difficulty": {
            label: group_statistics([group for group in groups if bins[str(group[0]["source_index"])] == label])
            for label in ("lower_reward", "higher_reward")
        }})
        for group_index, group in enumerate(groups):
            pid = str(group[0]["source_index"])
            rewards = torch.tensor([row["reward"] for row in group])
            advantages = group_relative_advantages(rewards, torch.zeros(len(group), dtype=torch.long))
            std = float(rewards.std(unbiased=False))
            evidence.append({
                "k": int(k), "group_index": group_index, "source_index": group[0]["source_index"],
                "prompt_id": group[0]["prompt_id"], "difficulty_bin": bins[pid],
                "difficulty_proxy": means[pid], "prompt": prompt_messages(prompt_rows[pid]),
                "reward_std": std, "informative": std > ADVANTAGE_EPS,
                "completions": [{**row, "advantage": float(adv)} for row, adv in zip(group, advantages)],
            })
    write_jsonl(result_dir / "groups.jsonl", evidence)
    summary = {
        "config": cfg, "cache": cfg["group_cache"], "cache_generation_cap": cfg["cache_generation_cap"],
        "partition_rule": "sort by generation_index; split each prompt's eight samples into disjoint contiguous K-sized groups; use all samples once per K",
        "generation_budget": sum(len(rows) for rows in by_prompt.values()),
        "generated_token_budget": sum(int(row["completion_tokens"]) for rows in by_prompt.values() for row in rows),
        "difficulty_binning": "rank prompts by mean reward over all original eight cached completions; bottom floor(N/2) lower_reward, remainder higher_reward; source_index string breaks ties; bins fixed across K",
        "difficulty_caveat": "fixed reward-score proxy for prompt difficulty; not verified correctness",
        "signal_definition": "population variance over all completion advantages, (r - within-group mean)/(population std + 1e-6); no training-length masking in this cached study",
        "informative_tolerance": ADVANTAGE_EPS,
        "prompt_bins": [{"source_index": by_prompt[pid][0]["source_index"],
                         "prompt_id": by_prompt[pid][0]["prompt_id"],
                         "difficulty_bin": bins[pid], "mean_k8_reward": means[pid]} for pid in order],
        "conditions": conditions,
    }
    save_json(result_dir / "comparison.json", summary)
    print(f"Saved equal-generation group-size evidence: {result_dir}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    analyze_group_size(args.config)


if __name__ == "__main__":
    main()
