from common_05a import parse_common_args, train_one_seed


if __name__ == "__main__":
    args = parse_common_args()
    train_one_seed(
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        mtl_steps=args.mtl_steps,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        overwrite_main=True,
    )
