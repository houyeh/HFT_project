from common_05a import parse_common_args, run_matched_random


if __name__ == "__main__":
    args = parse_common_args()
    run_matched_random(n_trials=args.trials)
