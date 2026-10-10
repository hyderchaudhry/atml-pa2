from __future__ import annotations

import argparse
import pandas as pd

from common.data import load_yaml, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json, set_seed
from common.models import clear_gpu, load_policy, load_tokenizer


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    set_seed(int(cfg["seed"]))
    print(f"Loading {policy_name}: {adapter or cfg['base_model']}", flush=True)
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    df = load_xstest(cfg)
    records = []
    for start in range(0, len(df), batch_size):
        print(f"Generating {policy_name}: prompts {start + 1}-{min(start + batch_size, len(df))}/{len(df)}", flush=True)
        chunk = df.iloc[start:start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for (_, row), response, n_tok in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"]):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": response,
                "response_tokens": int(n_tok),
            })
    return records


def validate_responses(cfg, rows, policy_name):
    """Require exactly one response per fixed prompt, in the released order."""
    fixed = load_xstest(cfg).to_dict("records")
    if not fixed or len(rows) != len(fixed):
        raise ValueError(f"{policy_name}: response count must match the fixed XSTest CSV")
    if len({row["xstest_id"] for row in rows}) != len(rows):
        raise ValueError(f"{policy_name}: duplicate XSTest IDs")
    for row, source in zip(rows, fixed):
        if row["policy"] != policy_name or any(row[key] != source[key] for key in
                ("xstest_id", "prompt", "benchmark_class", "type")):
            raise ValueError(f"{policy_name}: response metadata/order differs from the fixed CSV")
        if not isinstance(row["response"], str) or not isinstance(row["response_tokens"], int) or row["response_tokens"] < 0:
            raise ValueError(f"{policy_name}: invalid response text/token count")


def generate_responses(config_path: str, batch_size: int = 4):
    cfg = load_yaml(config_path)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    specs = policy_specs(cfg)
    if any((outdir / f"generated_{name}.jsonl").exists() for name in specs):
        raise FileExistsError(f"Refusing to overwrite generated responses: {outdir}")
    df = load_xstest(cfg)
    if df.empty or not df["xstest_id"].is_unique or set(df["benchmark_class"]) != {"SAFE", "UNSAFE"}:
        raise ValueError("Expected a nonempty fixed XSTest CSV with unique IDs and SAFE/UNSAFE classes")
    save_json(outdir / "generation_manifest.json", {
        "config": cfg, "policies": specs, "seed": int(cfg["seed"]),
        "xstest_ids": [int(value) for value in df["xstest_id"]],
        "decoding": {"do_sample": False, "temperature": 0.0, "top_p": 1.0,
                     "max_prompt_length": 256, "max_new_tokens": int(cfg["safety_max_new_tokens"])},
        "batch_size": batch_size,
        "response_length": "generated response tokens including EOS, excluding padding",
    })
    print(f"Task 4: {len(df)} fixed prompts for each of {list(specs)}", flush=True)
    for name in specs:
        records = generate_for_policy(cfg, name, batch_size)
        validate_responses(cfg, records, name)
        path = outdir / f"generated_{name}.jsonl"
        write_jsonl(path, records)
        print(f"Saved {name} responses: {path}", flush=True)
        clear_gpu()
    return outdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    generate_responses(args.config, args.batch_size)


if __name__ == "__main__":
    main()
