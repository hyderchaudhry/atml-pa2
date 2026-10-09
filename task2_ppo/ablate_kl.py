from __future__ import annotations

import argparse
from common.data import load_yaml
from task2_ppo.continue_train import run_forks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    run_forks(args.config, "--kl-beta", cfg["kl_values"], "kl")


if __name__ == "__main__":
    main()
