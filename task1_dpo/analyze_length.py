from __future__ import annotations

import argparse
import subprocess
import sys

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json
from task1_dpo.runtime import output_directory, result_directory


def main():
    ap = argparse.ArgumentParser(description="Train supplied length-balanced set and compare with standard DPO")
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--evaluate-only", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    args = ap.parse_args()
    if args.evaluate_only and args.train_only:
        ap.error("Choose only one of --evaluate-only and --train-only")
    cfg = load_yaml(args.config)
    if not args.train_only and not (repo_path(cfg["standard_output"]) / "training_metadata.json").exists():
        ap.error("Train standard DPO first, or use --train-only to train the length-balanced condition")
    if not args.evaluate_only:
        subprocess.run([sys.executable, "-m", "task1_dpo.train", "--config", args.config,
                        "--run-name", "length_balanced", "--dataset", cfg["paths"]["dpo_length_train"],
                        "--output", str(output_directory(cfg, "length_balanced"))], check=True)
    if args.train_only:
        return
    summaries, manifests = {}, {}
    for name in ("standard", "length_balanced"):
        subprocess.run([sys.executable, "-m", "task1_dpo.evaluate", "--config", args.config,
                        "--adapter", str(output_directory(cfg, name)), "--name", name], check=True)
        directory = result_directory(cfg, name)
        metrics = load_json(directory / "metrics.json")
        manifests[name] = load_json(directory / "evaluation_manifest.json")
        summaries[name] = {"metrics_path": str(directory / "metrics.json"),
                           "training_budget": metrics["training_budget"],
                           "length_stratified": metrics["length_stratified"],
                           "common_word_limit": metrics["common_word_limit"]}
    # A comparison is only valid when prompt identities, seeds and decoding agree.
    first, second = manifests["standard"], manifests["length_balanced"]
    for key in ("seed", "settings", "datasets"):
        if first[key] != second[key]:
            raise ValueError(f"Length comparison mismatch: {key}")
    save_json(result_directory(cfg, "length_study") / "comparison.json", summaries)


if __name__ == "__main__":
    main()
