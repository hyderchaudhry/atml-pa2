from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import load_json, save_json


def length_conditioned_statistics(rows):
    """Use the same fixed cap/2 length bins for both loss normalizations."""
    statistics = {}
    for label in ("short", "long"):
        group = [row for row in rows if row["length_bin"] == label]
        keys = ("response_length", "learned_reward", "surrogate_logp_gradient_l1",
                "surrogate_logp_gradient_l2", "mean_absolute_token_gradient")
        statistics[label] = {
            "num_completions": len(group),
            "num_masked": sum(row["valid_training_tokens"] == 0 for row in group),
            **{key: sum(row[key] for row in group) / len(group) if group else None for key in keys},
        }
    return statistics


def compare_normalization(config_path):
    cfg = load_yaml(config_path)
    result_root = repo_path(cfg["results_dir"])
    study_dir = result_root / "normalization"
    output_root = Path(cfg["output"]).parent
    names = {loss: f"normalization_{loss}" for loss in ("grpo", "dr_grpo")}
    if study_dir.exists() or any((result_root / name).exists() or repo_path(output_root / name).exists()
                                 for name in names.values()):
        raise FileExistsError("Refusing to overwrite normalization study evidence")
    conditions = []
    shared_rollouts = str(output_root / names["grpo"] / "rollouts")
    print(f"Beginning normalization study | {cfg['fork_updates']} updates per fork | "
          "identical sampled-token budget", flush=True)
    # Fresh processes reload the identical midpoint and optimizer and release model memory.
    # Canonical GRPO samples online; the second fork uses those exact samples/behavior logp.
    # This holds actual generated tokens/rewards fixed, not only the maximum-length cap.
    for loss_type, name in names.items():
        print(f"Starting fork {name} from the supplied midpoint", flush=True)
        output = str(output_root / name)
        command = [sys.executable, "-m", "task3_grpo.continue_train", "--config", config_path,
                   "--output", output, "--updates", str(cfg["fork_updates"]),
                   "--loss-type", loss_type, "--run-name", name]
        command.extend(["--save-rollouts"] if loss_type == "grpo" else ["--rollout-source", shared_rollouts])
        subprocess.run(command, cwd=repo_path("."), check=True)
        print(f"Evaluating fork {name} on the fixed held-out prompts", flush=True)
        subprocess.run([sys.executable, "-m", "task3_grpo.evaluate", "--config", config_path,
                        "--adapter", output, "--name", name], cwd=repo_path("."), check=True)
        directory = result_root / name
        summary = load_json(directory / "training_summary.json")
        conditions.append({
            "loss_type": loss_type, "run_name": name, "updates": summary["completed_updates"],
            "generated_tokens": summary["generated_tokens"],
            "valid_training_tokens": summary["valid_training_tokens"],
            "mean_gradient_norm": summary["mean_gradient_norm"],
            "skipped_steps": summary["skipped_steps"],
            "held_out": load_json(directory / "metrics.json"),
            "length_conditioned": length_conditioned_statistics(read_jsonl(directory / "training_completions.jsonl")),
            "training_manifest": str(directory / "training_manifest.json"),
            "trajectory": str(directory / "trajectory.jsonl"),
            "training_completions": str(directory / "training_completions.jsonl"),
            "held_out_examples": str(directory / "generations.jsonl"),
        })
        print(f"Completed fork {name} | aggregating length-conditioned evidence", flush=True)
    if conditions[0]["generated_tokens"] != conditions[1]["generated_tokens"] or \
            conditions[0]["valid_training_tokens"] != conditions[1]["valid_training_tokens"]:
        raise ValueError("Normalization forks did not use the same generated/training tokens")
    # Save paired held-out responses for the student's own qualitative quality audit.
    canonical = read_jsonl(result_root / names["grpo"] / "generations.jsonl")
    dr = read_jsonl(result_root / names["dr_grpo"] / "generations.jsonl")
    if [row["prompt_id"] for row in canonical] != [row["prompt_id"] for row in dr]:
        raise ValueError("Normalization forks used different held-out prompts")
    write_jsonl(study_dir / "paired_generations.jsonl", [{
        "prompt_id": a["prompt_id"], "source_index": a["source_index"], "prompt": a["prompt"],
        "grpo": a, "dr_grpo": b,
        "reward_difference_dr_minus_grpo": b["learned_reward"] - a["learned_reward"],
        "length_difference_dr_minus_grpo": b["response_length"] - a["response_length"],
    } for a, b in zip(canonical, dr)])
    comparison = {
        "config": cfg, "conditions": conditions,
        "matched_rollout_protocol": "canonical fork samples online at every update; both forks use identical saved tokens, raw rewards, within-prompt advantages, masks and canonical behavior log probabilities; Dr-GRPO uses importance ratios relative to that shared behavior policy",
        "controlled_change": "policy surrogate denominator only: realized completion length vs configured max_completion_length; unchanged reward normalization, clipping and KL loss",
        "generation_budget": "actual sampled nonpadding tokens are identical; shared samples generated once, with the same effective token budget for both forks",
        "length_bins": "short <= configured max_completion_length/2; long > that threshold; includes zero-gradient masked completions",
        "gradient_statistic": "L1/L2 norm of policy-surrogate derivative w.r.t. each completion's sampled-token log probabilities; exact normalization diagnostic, not a model parameter-gradient norm; KL excluded",
        "quality_evidence": "paired held-out responses for manual inspection; learned reward is not ground-truth quality",
        "paired_examples": str(study_dir / "paired_generations.jsonl"),
    }
    save_json(study_dir / "comparison.json", comparison)
    print(f"Completed normalization study | saved evidence: {study_dir}", flush=True)
    return comparison


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    compare_normalization(args.config)


if __name__ == "__main__":
    main()
