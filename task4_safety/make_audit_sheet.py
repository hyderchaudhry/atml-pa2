from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task4_safety.generate_responses import policy_specs, validate_responses


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def make_audit_sheet(config_path: str):
    cfg = load_yaml(config_path)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    baseline = read_jsonl(src)
    validate_responses(cfg, baseline, "sft")
    ids = fixed_audit_ids(baseline, int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    destinations = [outdir / "manual_audit_ids.csv"] + [
        outdir / f"manual_audit_{name}.csv" for name in policy_specs(cfg)
    ]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Refusing to overwrite audit sheets or human labels")
    sheets = {}
    for name in policy_specs(cfg):
        rows = read_jsonl(outdir / f"generated_{name}.jsonl")
        validate_responses(cfg, rows, name)
        by_id = {row["xstest_id"]: row for row in rows}
        # Use generated files only: no AI label or confidence can leak into the blind audit.
        sheets[name] = [{key: by_id[index][key] for key in
                        ("xstest_id", "policy", "benchmark_class", "type", "prompt", "response")}
                        for index in ids]
    pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)}).to_csv(outdir / "manual_audit_ids.csv", index=False)
    for name, rows in sheets.items():
        sheet = pd.DataFrame(rows)
        sheet["manual_label"] = ""
        sheet["manual_notes"] = ""
        path = outdir / f"manual_audit_{name}.csv"
        sheet.to_csv(path, index=False)
        print(f"Saved blind {name} audit sheet ({len(ids)} fixed prompts): {path}", flush=True)
    save_json(outdir / "audit_manifest.json", {
        "seed": int(cfg["seed"]), "per_class": int(cfg["manual_audit_per_class"]),
        "xstest_ids": ids, "selection": "released fixed_audit_ids helper on SFT rows",
        "scope": f"same {len(ids)} prompt IDs for each policy; assign one human label per policy response",
        "labels": ["SAFE_ANSWER", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "OVER_REFUSAL", "AMBIGUOUS"],
        "instructions": "fill manual_label in each manual_audit_<policy>.csv without inspecting AI labels; then rerun evaluate_safety",
    })
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    make_audit_sheet(args.config)


if __name__ == "__main__":
    main()
