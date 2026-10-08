from __future__ import annotations

import argparse
import subprocess
import sys

from common.data import load_yaml
from common.logging_utils import load_json, save_json
from task1_dpo.runtime import beta_name, output_directory, result_directory


def main():
    ap = argparse.ArgumentParser(description="Independent, matched short DPO forks (train + evaluate)")
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--beta", type=float, help="Run only this released beta; default: all three")
    ap.add_argument("--evaluate-only", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    args = ap.parse_args()
    if args.evaluate_only and args.train_only:
        ap.error("Choose only one of --evaluate-only and --train-only")
    cfg = load_yaml(args.config)
    betas = [float(b) for b in cfg["betas"]]
    if args.beta is not None:
        if args.beta not in betas:
            ap.error(f"--beta must be one of {betas}")
        betas = [args.beta]
    summaries = {}
    for beta in betas:
        name = beta_name(beta)
        output = output_directory(cfg, name)
        # Separate processes guarantee that adapters, optimizer state and CUDA
        # allocations cannot leak from one beta condition into the next.
        if not args.evaluate_only:
            subprocess.run([sys.executable, "-m", "task1_dpo.train", "--config", args.config,
                            "--run-name", name, "--beta", str(beta),
                            "--max-examples", str(cfg["short_ablation_examples"]),
                            "--output", str(output)], check=True)
        if not args.train_only:
            subprocess.run([sys.executable, "-m", "task1_dpo.evaluate", "--config", args.config,
                            "--adapter", str(output), "--name", name], check=True)
            summaries[name] = load_json(result_directory(cfg, name) / "metrics.json")
    if summaries:
        # Separate --beta invocations accumulate the comparison from completed
        # on-disk results, rather than replacing it with the most recent fork.
        manifests = []
        for beta in cfg["betas"]:
            name = beta_name(beta)
            directory = result_directory(cfg, name)
            if (directory / "metrics.json").exists() and (directory / "evaluation_manifest.json").exists():
                manifest = load_json(directory / "evaluation_manifest.json")
                if manifest["status"] != "complete":
                    continue
                if manifest["config"] != cfg:
                    raise ValueError(f"Beta comparison config mismatch: {name}")
                manifests.append(manifest)
                summaries[name] = load_json(directory / "metrics.json")
        for manifest in manifests[1:]:
            for key in ("seed", "settings", "datasets", "training_budget"):
                if manifest[key] != manifests[0][key]:
                    raise ValueError(f"Beta comparison mismatch: {key}")
        save_json(result_directory(cfg, "beta_study") / "comparison.json", summaries)


if __name__ == "__main__":
    main()
