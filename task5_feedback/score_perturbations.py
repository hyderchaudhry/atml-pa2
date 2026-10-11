from __future__ import annotations

import argparse
from collections import defaultdict

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json, set_seed
from common.models import clear_gpu
from task5_feedback.evaluate_math import preference
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}

# Clean is the reference for all interventions. The clean-vs-itself pair is a
# negative control, for which only a tie is expected (not a better response).
CONTROLLED_PAIRS = {
    "clean_correct": "negative_control",
    "corrupt_reasoning_correct_final": "reasoning_sensitivity",
    "good_reasoning_wrong_final": "outcome_sensitivity",
    "persuasive_filler_correct": "filler_susceptibility",
    "gold_distractor_wrong_final": "distractor_robustness",
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        pid, variant = str(row["problem_id"]), row["variant_type"]
        if variant not in EXPECTED_VARIANTS or variant in by_problem[pid]:
            raise ValueError(f"Problem {pid}: unknown or duplicate variant {variant}")
        by_problem[pid][variant] = row
    if not by_problem:
        raise ValueError("Diagnostic set is empty")
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
        clean = variants["clean_correct"]
        for row in variants.values():
            if row["question"] != clean["question"] or str(row["gold_final"]) != str(clean["gold_final"]):
                raise ValueError(f"Problem {pid}: inconsistent question/gold final")
            if exact_reward(row["response"], row["gold_final"]) != row["expected_exact_reward"]:
                raise ValueError(f"Problem {pid}: supplied verifier differs from expected diagnostic reward")
    return by_problem


def rate_summary(pairs, key):
    n = len(pairs)
    counts = {label: sum(row[key] == label for row in pairs) for label in ("A", "B", "TIE")}
    return {"n": n, "better_response_count": counts["A"], "tie_count": counts["TIE"],
            "wrong_preference_count": counts["B"], "better_response_rate": counts["A"] / n,
            "tie_rate": counts["TIE"] / n, "wrong_preference_rate": counts["B"] / n}


def score_perturbations(config_path: str):
    cfg = load_yaml(config_path)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback" / "diagnostics"
    set_seed(int(cfg["seed"]))
    save_json(outdir / "evaluation_manifest.json", {
        "config": cfg, "data_path": cfg["paths"]["task5_diagnostics"],
        "problem_ids": list(groups), "variants": list(CONTROLLED_PAIRS),
        "pair_protocol": "A = clean_correct; B = named variant. Clean-vs-clean is a negative control with no preferred response.",
        "filler_protocol": "Prefer the relevant clean response; count B preferences as filler susceptibility and report ties separately.",
        "reasoning_protocol": "RLVR reasoning-only ties express outcome invariance, not accuracy errors.",
        "outcome_protocol": "Clean vs good_reasoning_wrong_final; distractor robustness is reported separately.",
        "judge": "supplied fixed PairwiseAIJudge, with persistent cache and orientation balancing",
    })
    judge = PairwiseAIJudge(cfg, outdir / "judge_cache.json")
    pairs, responses = [], []
    try:
        for index, (pid, variants) in enumerate(groups.items(), 1):
            print(f"Scoring diagnostic problem {pid}: {index}/{len(groups)}", flush=True)
            clean = variants["clean_correct"]
            rewards = [exact_reward(variants[v]["response"], clean["gold_final"]) for v in CONTROLLED_PAIRS]
            # Score all five responses by the supplied direct-RLAIF group rule.
            group_rewards = judge.group_rewards(clean["question"], [variants[v]["response"] for v in CONTROLLED_PAIRS])
            if len(group_rewards) != len(CONTROLLED_PAIRS):
                raise ValueError("Judge returned the wrong number of group rewards")
            for variant, reward, group_reward in zip(CONTROLLED_PAIRS, rewards, group_rewards):
                responses.append({**variants[variant], "rlvr_reward": reward, "rlaif_group_reward": group_reward})
                result = judge.compare(clean["question"], clean["response"], variants[variant]["response"])
                if result not in {"A", "B", "TIE"}:
                    raise ValueError(f"Invalid judge preference: {result}")
                pairs.append({
                    "problem_id": pid, "variant_type": variant, "diagnostic": CONTROLLED_PAIRS[variant],
                    "question": clean["question"], "gold_final": str(clean["gold_final"]),
                    "response_a": clean["response"], "response_b": variants[variant]["response"],
                    "rlvr_reward_a": rewards[0], "rlvr_reward_b": reward,
                    "rlvr_preference": preference(rewards[0], reward), "rlaif_preference": result,
                    "expected_preference": "TIE" if variant == "clean_correct" else "A",
                })
    finally:
        del judge
        clear_gpu()
    breakdown = {}
    for variant in CONTROLLED_PAIRS:
        selected = [row for row in pairs if row["variant_type"] == variant]
        breakdown[variant] = {mechanism: rate_summary(selected, f"{mechanism}_preference") for mechanism in ("rlvr", "rlaif")}
        if variant == "clean_correct":
            # A/B labels are both spurious distinctions on this negative control.
            breakdown[variant] = {
                mechanism: {"n": len(selected), "better_response_rate": None,
                            "tie_rate": value["tie_rate"], "wrong_preference_rate": 1 - value["tie_rate"],
                            "tie_count": value["tie_count"], "wrong_preference_count": len(selected) - value["tie_count"]}
                for mechanism, value in breakdown[variant].items()
            }
    metrics = {"problem_count": len(groups), "response_count": len(responses), "by_variant": breakdown,
               "sensitivities": {mechanism: {
                   "S_reason": breakdown["corrupt_reasoning_correct_final"][mechanism]["better_response_rate"],
                   "S_outcome": breakdown["good_reasoning_wrong_final"][mechanism]["better_response_rate"],
                   "filler_susceptibility": breakdown["persuasive_filler_correct"][mechanism]["wrong_preference_rate"],
                   "distractor_robustness": breakdown["gold_distractor_wrong_final"][mechanism]["better_response_rate"],
               } for mechanism in ("rlvr", "rlaif")}}
    write_jsonl(outdir / "scored_responses.jsonl", responses)
    write_jsonl(outdir / "controlled_pairs.jsonl", pairs)
    write_jsonl(outdir / "qualitative_evidence.jsonl", [
        {**row, "feedback_disagreement": row["rlvr_preference"] != row["rlaif_preference"]}
        for row in pairs if row["variant_type"] != "clean_correct"
    ])
    save_json(outdir / "metrics.json", metrics)
    print(f"Saved Task 5 diagnostics: {outdir}", flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    score_perturbations(args.config)


if __name__ == "__main__":
    main()
