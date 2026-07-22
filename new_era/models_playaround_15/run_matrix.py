from __future__ import annotations

import argparse

from new_era.models_playaround_15.backtest_val_sweep import backtest_val_sweep_then_test
from new_era.models_playaround_15.dataset_generation import generate_dataset_npz
from new_era.models_playaround_15.evaluate_tcn_full import evaluate_full
from new_era.models_playaround_15.experiment_matrix import iter_experiment_configs
from new_era.models_playaround_15.paths import build_experiment_paths, experiment_id
from new_era.models_playaround_15.train_tcn import train_tcn


def run_one(cfg) -> None:
    paths = build_experiment_paths(cfg)
    print(f"\n=== {experiment_id(cfg)} ===")
    generate_dataset_npz(cfg, paths)
    train_tcn(cfg, paths)
    evaluate_full(cfg, paths)
    backtest_val_sweep_then_test(cfg, paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1, choices=[1, 2])
    parser.add_argument("--feature-counts", type=str, default="6,7,8,9")
    args = parser.parse_args()

    feature_counts = [int(x) for x in args.feature_counts.split(",") if x.strip()]
    for cfg in iter_experiment_configs(batch=int(args.batch), feature_counts=feature_counts):
        run_one(cfg)


if __name__ == "__main__":
    main()
