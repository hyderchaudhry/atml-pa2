from __future__ import annotations

import argparse
import itertools
from collections import Counter

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import load_json, save_json, set_seed
from common.models import clear_gpu, load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def example_id(row):
    return str(row.get("prompt_id", row["source_index"]))


def validate_math_rows(rows):
    if not rows or len({example_id(row) for row in rows}) != len(rows):
        raise ValueError("Expected a nonempty math dataset with unique fixed IDs")
    for row in rows:
        if not row["question"] or not prompt_messages(row) or row["gold_final"] is None:
            raise ValueError("Missing math question, prompt, or gold final")


def response_record(row, policy, response, tokens, terminated, truncated):
    parsed = extract_designated_final(response)
    return {
        "prompt_id": example_id(row), "source_index": row["source_index"],
        "source_split": row.get("source_split"), "question": row["question"],
        "messages": prompt_messages(row), "gold_final": str(row["gold_final"]),
        "policy": policy, "response": response, "response_tokens": int(tokens),
        "terminated_with_eos": bool(terminated), "truncated": bool(truncated),
        "parsed_final": parsed, "format_compliance": parsed is not None,
        "exact_reward": exact_reward(response, row["gold_final"]),
        "failure_type": "missing_final" if parsed is None else (
            "correct" if exact_reward(response, row["gold_final"]) else "wrong_final"),
        "category": row.get("svamp_type"),
    }


def validate_generations(rows, generated, policy):
    if len(rows) != len(generated):
        raise ValueError(f"{policy}: generated count differs from fixed dataset")
    for source, saved in zip(rows, generated):
        if any(saved[key] != value for key, value in {
            "prompt_id": example_id(source), "source_index": source["source_index"],
            "source_split": source.get("source_split"), "question": source["question"],
            "messages": prompt_messages(source), "gold_final": str(source["gold_final"]),
            "policy": policy, "category": source.get("svamp_type"),
        }.items()):
            raise ValueError(f"{policy}: generated metadata/order differs from fixed dataset")
        if not isinstance(saved["response"], str) or not isinstance(saved["response_tokens"], int) or saved["response_tokens"] < 0:
            raise ValueError(f"{policy}: invalid generated text/token count")
        expected = response_record(source, policy, saved["response"], saved["response_tokens"],
                                   saved["terminated_with_eos"], saved["truncated"])
        if saved != expected:
            raise ValueError(f"{policy}: saved verifier metrics differ from supplied verifier")


def preference(a, b):
    return "A" if a > b else "B" if b > a else "TIE"


def preference_summary(records):
    counts = Counter(row["ai_preference"] for row in records)
    n = len(records)
    differing = [row for row in records if row["exact_preference"] != "TIE"]
    tied = [row for row in records if row["exact_preference"] == "TIE"]
    return {
        "n": n, "wins_a": counts["A"], "wins_b": counts["B"], "ties": counts["TIE"],
        "win_rate_a": (counts["A"] + 0.5 * counts["TIE"]) / n if n else None,
        "win_rate_b": (counts["B"] + 0.5 * counts["TIE"]) / n if n else None,
        "tie_rate": counts["TIE"] / n if n else None,
        "verifier_judge_agreement": sum(r["ai_preference"] == r["exact_preference"] for r in records) / n if n else None,
        "confusion": {exact: {ai: sum(r["exact_preference"] == exact and r["ai_preference"] == ai for r in records)
                              for ai in ("A", "B", "TIE")} for exact in ("A", "B", "TIE")},
        "different_outcome_n": len(differing),
        "different_outcome_agreement": sum(r["ai_preference"] == r["exact_preference"] for r in differing) / len(differing) if differing else None,
        "different_outcome_wrong_preference_rate": sum(r["ai_preference"] not in (r["exact_preference"], "TIE") for r in differing) / len(differing) if differing else None,
        "different_outcome_tie_rate": sum(r["ai_preference"] == "TIE" for r in differing) / len(differing) if differing else None,
        "same_outcome_n": len(tied),
        "same_outcome_ai_distinction_rate": sum(r["ai_preference"] != "TIE" for r in tied) / len(tied) if tied else None,
    }


def policy_metrics(records):
    n = len(records)
    lengths = [row["response_tokens"] for row in records]
    mean = sum(lengths) / n
    return {
        "n": n, "exact_accuracy": sum(row["exact_reward"] for row in records) / n,
        "format_compliance": sum(row["format_compliance"] for row in records) / n,
        "response_tokens_mean": mean,
        "response_tokens_std": (sum((x - mean) ** 2 for x in lengths) / n) ** 0.5,
        "truncation_rate": sum(row["truncated"] for row in records) / n,
        "failure_counts": dict(Counter(row["failure_type"] for row in records)),
    }


def evaluate_math(config_path: str, dataset: str, batch_size: int = 4, evaluate_only: bool = False):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    validate_math_rows(rows)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback" / dataset
    specs = policy_specs(cfg)
    decoding = {**cfg["generation"], "max_new_tokens": int(cfg["math_max_new_tokens"])}
    if decoding["max_new_tokens"] < 1:
        raise ValueError("math_max_new_tokens must be positive")
    if evaluate_only:
        manifest = load_json(outdir / "evaluation_manifest.json")
        if manifest["config"] != cfg or manifest["dataset"] != dataset:
            raise ValueError("Saved evaluation config/dataset differs from this run")
        generated = {name: read_jsonl(outdir / f"generated_{name}.jsonl") for name in specs}
    else:
        if any((outdir / f"generated_{name}.jsonl").exists() for name in specs):
            raise FileExistsError(f"Refusing to overwrite generations; use --evaluate-only: {outdir}")
        tokenizer = load_tokenizer(cfg["base_model"])
        prompts = [prompt_messages(row) for row in rows]
        # Preserve every question token rather than silently truncating fixed prompts.
        max_prompt_length = max(len(tokenizer.apply_chat_template(p, tokenize=True, add_generation_prompt=True)) for p in prompts)
        save_json(outdir / "evaluation_manifest.json", {
            "config": cfg, "dataset": dataset, "data_path": dataset_path(cfg, dataset),
            "policies": specs, "seed": int(cfg["seed"]), "batch_size": batch_size,
            "prompt_ids": [example_id(row) for row in rows],
            "decoding": {**decoding, "max_prompt_length": max_prompt_length},
            "response_length": "generated tokens including EOS, excluding prompt and padding; population standard deviation",
            "agreement": "A/B/TIE equality; additionally condition on unequal binary outcomes. Same-outcome distinctions need not be judge errors.",
            "judge": "supplied PairwiseAIJudge, including its fixed rubric, orientation balancing, parser and cache",
        })
        generated = {}
        for name in specs:
            set_seed(int(cfg["seed"]))
            model = load_frozen_policy(cfg, name)
            # Adapter construction can consume RNG; reset again for matched decoding.
            set_seed(int(cfg["seed"]))
            records = []
            try:
                for start in range(0, len(rows), batch_size):
                    print(f"Generating {dataset}/{name}: {start + 1}-{min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
                    gen = batch_generate(model, tokenizer, prompts[start:start + batch_size],
                                         max_prompt_length=max_prompt_length, **decoding)
                    for source, response, tokens, eos, truncated in zip(
                        rows[start:start + batch_size], gen["responses"], gen["response_lengths"],
                        gen["terminated_with_eos"], gen["truncated"],
                    ):
                        records.append(response_record(source, name, response, tokens, eos, truncated))
                    del gen
                validate_generations(rows, records, name)
                write_jsonl(outdir / f"generated_{name}.jsonl", records)
                generated[name] = records
            finally:
                del model
                clear_gpu()
    for name, records in generated.items():
        validate_generations(rows, records, name)
    set_seed(int(cfg["seed"]))
    judge = PairwiseAIJudge(cfg, outdir / "judge_cache.json")
    pairs, summaries = [], {}
    try:
        for a, b in itertools.combinations(specs, 2):
            comparisons = []
            for index, (left, right) in enumerate(zip(generated[a], generated[b]), 1):
                print(f"Judging {dataset}/{a} vs {b}: {index}/{len(rows)}", flush=True)
                result = judge.compare(left["question"], left["response"], right["response"])
                if result not in {"A", "B", "TIE"}:
                    raise ValueError(f"Invalid judge preference: {result}")
                comparisons.append({
                    "prompt_id": left["prompt_id"], "policy_a": a, "policy_b": b,
                    "exact_reward_a": left["exact_reward"], "exact_reward_b": right["exact_reward"],
                    "exact_preference": preference(left["exact_reward"], right["exact_reward"]),
                    "ai_preference": result,
                })
            pairs.extend(comparisons)
            summaries[f"{a}_vs_{b}"] = preference_summary(comparisons)
    finally:
        del judge
        clear_gpu()
    metrics = {"dataset": dataset, "policies": {name: policy_metrics(records) for name, records in generated.items()},
               "pairwise": summaries}
    write_jsonl(outdir / "pairwise.jsonl", pairs)
    save_json(outdir / "metrics.json", metrics)
    print(f"Saved Task 5 evaluation: {outdir}", flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--evaluate-only", action="store_true", help="Reuse saved generations with their original config")
    args = ap.parse_args()
    evaluate_math(args.config, args.dataset, args.batch_size, args.evaluate_only)


if __name__ == "__main__":
    main()
