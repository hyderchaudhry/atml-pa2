from __future__ import annotations

import argparse
import csv
from collections import defaultdict

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import load_json, save_json
from task5_feedback.evaluate_math import (
    dataset_path, policy_metrics, policy_specs, preference, preference_summary,
    validate_generations, validate_math_rows,
)
from task5_feedback.score_perturbations import CONTROLLED_PAIRS, load_diagnostic_groups, rate_summary


def compare_feedback(config_path: str):
    cfg = load_yaml(config_path)
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    datasets, generated, evidence = {}, {}, []
    for dataset in ("gsm", "transfer"):
        directory = outdir / dataset
        manifest = load_json(directory / "evaluation_manifest.json")
        if manifest["config"] != cfg or manifest["dataset"] != dataset:
            raise ValueError(f"{dataset}: evaluation config/dataset mismatch")
        rows = read_jsonl(dataset_path(cfg, dataset))
        validate_math_rows(rows)
        generated[dataset] = {}
        metrics = load_json(directory / "metrics.json")
        for name in policy_specs(cfg):
            records = read_jsonl(directory / f"generated_{name}.jsonl")
            validate_generations(rows, records, name)
            if policy_metrics(records) != metrics["policies"][name]:
                raise ValueError(f"{dataset}/{name}: stale policy metrics")
            generated[dataset][name] = records
            evidence.extend({"dataset": dataset, **row} for row in records if row["failure_type"] != "correct")
        pairwise = read_jsonl(directory / "pairwise.jsonl")
        for key, summary in metrics["pairwise"].items():
            a, b = key.split("_vs_")
            selected = [r for r in pairwise if r["policy_a"] == a and r["policy_b"] == b]
            if len(selected) != len(rows):
                raise ValueError(f"{dataset}/{key}: pairwise count mismatch")
            for pair, left, right in zip(selected, generated[dataset][a], generated[dataset][b]):
                if pair["prompt_id"] != left["prompt_id"] or pair["ai_preference"] not in {"A", "B", "TIE"} or (
                    pair["exact_reward_a"], pair["exact_reward_b"], pair["exact_preference"]
                ) != (left["exact_reward"], right["exact_reward"], preference(left["exact_reward"], right["exact_reward"])):
                    raise ValueError(f"{dataset}/{key}: invalid pairwise alignment/rewards")
            if preference_summary(selected) != summary:
                raise ValueError(f"{dataset}/{key}: stale pairwise metrics")
        if set(metrics["pairwise"]) != {"sft_vs_rlvr", "sft_vs_rlaif", "rlvr_vs_rlaif"} or len(pairwise) != 3 * len(rows):
            raise ValueError(f"{dataset}: expected all three matched policy comparisons")
        datasets[dataset] = metrics
    gsm_manifest = load_json(outdir / "gsm/evaluation_manifest.json")
    transfer_manifest = load_json(outdir / "transfer/evaluation_manifest.json")
    # Prompt lengths may differ, but every decoding setting must remain fixed.
    gsm_decoding = {k: v for k, v in gsm_manifest["decoding"].items() if k != "max_prompt_length"}
    transfer_decoding = {k: v for k, v in transfer_manifest["decoding"].items() if k != "max_prompt_length"}
    if gsm_decoding != transfer_decoding:
        raise ValueError("In-domain and transfer decoding settings differ")
    diagnostics_dir = outdir / "diagnostics"
    diagnostic_manifest = load_json(diagnostics_dir / "evaluation_manifest.json")
    if diagnostic_manifest["config"] != cfg:
        raise ValueError("Diagnostic config differs from policy evaluation config")
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    diagnostic_metrics = load_json(diagnostics_dir / "metrics.json")
    diagnostic_pairs = read_jsonl(diagnostics_dir / "controlled_pairs.jsonl")
    if len(diagnostic_pairs) != len(groups) * len(CONTROLLED_PAIRS):
        raise ValueError("Incomplete controlled diagnostic pairs")
    for variant in CONTROLLED_PAIRS:
        selected = [r for r in diagnostic_pairs if r["variant_type"] == variant]
        if len(selected) != len(groups) or {r["problem_id"] for r in selected} != set(groups):
            raise ValueError(f"Incomplete/duplicate diagnostic pairs: {variant}")
        for row in selected:
            fixed = groups[row["problem_id"]]
            if row["response_a"] != fixed["clean_correct"]["response"] or row["response_b"] != fixed[variant]["response"]:
                raise ValueError(f"Diagnostic responses differ from fixed data: {variant}")
            a, b = fixed["clean_correct"]["expected_exact_reward"], fixed[variant]["expected_exact_reward"]
            if (row["rlvr_reward_a"], row["rlvr_reward_b"], row["rlvr_preference"]) != (a, b, preference(a, b)):
                raise ValueError(f"Diagnostic rewards differ from supplied verifier: {variant}")
        for mechanism in ("rlvr", "rlaif"):
            if any(r[f"{mechanism}_preference"] not in {"A", "B", "TIE"} for r in selected):
                raise ValueError("Invalid diagnostic preference")
            summary = rate_summary(selected, f"{mechanism}_preference")
            saved = diagnostic_metrics["by_variant"][variant][mechanism]
            expected_tie, expected_wrong = summary["tie_rate"], summary["wrong_preference_rate"]
            if variant == "clean_correct":
                expected_wrong = 1 - expected_tie
            expected_better = None if variant == "clean_correct" else summary["better_response_rate"]
            if saved["n"] != len(selected) or saved["tie_rate"] != expected_tie or saved["wrong_preference_rate"] != expected_wrong or saved["better_response_rate"] != expected_better:
                raise ValueError("Stale diagnostic metrics")
    table = []
    for name in policy_specs(cfg):
        gsm = datasets["gsm"]["policies"][name]
        transfer = datasets["transfer"]["policies"][name]
        # SFT self-comparison is a defined tie baseline, not an inferred judge decision.
        gsm_win = 0.5 if name == "sft" else datasets["gsm"]["pairwise"][f"sft_vs_{name}"]["win_rate_b"]
        transfer_win = 0.5 if name == "sft" else datasets["transfer"]["pairwise"][f"sft_vs_{name}"]["win_rate_b"]
        table.append({
            "policy": name, "gsm_exact_accuracy": gsm["exact_accuracy"],
            "transfer_exact_accuracy": transfer["exact_accuracy"],
            "exact_accuracy_drop": gsm["exact_accuracy"] - transfer["exact_accuracy"],
            "gsm_ai_win_rate_vs_sft": gsm_win, "transfer_ai_win_rate_vs_sft": transfer_win,
            "ai_win_rate_drop": gsm_win - transfer_win,
            "gsm_format_compliance": gsm["format_compliance"], "transfer_format_compliance": transfer["format_compliance"],
            "gsm_response_tokens_mean": gsm["response_tokens_mean"], "gsm_response_tokens_std": gsm["response_tokens_std"],
            "transfer_response_tokens_mean": transfer["response_tokens_mean"], "transfer_response_tokens_std": transfer["response_tokens_std"],
            "response_tokens_mean_change": transfer["response_tokens_mean"] - gsm["response_tokens_mean"],
        })
    categories = {}
    for name, records in generated["transfer"].items():
        by_category = defaultdict(list)
        for row in records:
            by_category[row["category"] or "unspecified"].append(row)
        categories[name] = {key: policy_metrics(values) for key, values in by_category.items()}
    comparison = {"config": cfg, "datasets": datasets, "controlled_diagnostics": diagnostic_metrics,
                  "transfer_comparison": table, "transfer_categories": categories,
                  "conventions": {
                      "drop": "in-domain minus transfer; negative means transfer improved; datasets have different difficulty",
                      "sft_self_win_rate": "defined 0.5 tie baseline",
                      "judge_ties": "supplied parser also maps unparseable output to TIE; ambiguous outputs cannot be separated",
                      "failure_types": "parser-observable missing_final/wrong_final; no inferred reasoning-quality labels",
                      "reward_scales": "binary rewards and normalized group win rates are saved separately, not treated as calibrated utilities",
                  },
                  "feedback_cost": {
                      "math_pairwise_decisions": sum(3 * data["policies"]["sft"]["n"] for data in datasets.values()),
                      "diagnostic_group_pairwise_decisions": 10 * len(groups),
                      "diagnostic_controlled_pair_requests": 5 * len(groups),
                      "note": "Judge cache reuses repeated pairs; these are requested decisions, not measured inference latency or cache misses. Exact verification uses no model inference.",
                  }}
    save_json(outdir / "comparison.json", comparison)
    with (outdir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    write_jsonl(outdir / "failure_examples.jsonl", evidence)
    print(f"Saved Task 5 comparison: {outdir}", flush=True)
    return comparison


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    compare_feedback(args.config)


if __name__ == "__main__":
    main()
