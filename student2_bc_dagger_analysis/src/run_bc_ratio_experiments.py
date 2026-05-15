import argparse
import csv
from pathlib import Path

from bc_train import build_parser as build_bc_parser
from bc_train import train as run_bc_train


RATIO_SPECS = [
    ("noise0", 0.0),
    ("noise10", 0.1),
    ("noise20", 0.2),
    ("noise30", 0.3),
    ("noise50", 0.5),
    ("noise80", 0.8),
    ("noise100", 1.0),
]


def build_parser():
    project_new_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run BC experiments for the dual-arm noisy-ratio mixed datasets.")
    parser.add_argument(
        "--mixed-root",
        type=str,
        default=str(project_new_root / "experiments" / "mixed_data" / "dual_arm"),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(project_new_root / "experiments" / "bc_ratio" / "dual_arm"),
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        default=[name for name, _ in RATIO_SPECS],
        choices=[name for name, _ in RATIO_SPECS],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rollout-episodes", type=int, default=20)
    parser.add_argument("--rollout-max-steps", type=int, default=600)
    parser.add_argument("--force-cpu", action="store_true")
    return parser


def build_bc_args(args, data_path: Path, output_dir: Path):
    bc_args = build_bc_parser().parse_args(
        [
            "--data",
            str(data_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    bc_args.seed = args.seed
    bc_args.epochs = args.epochs
    bc_args.batch_size = args.batch_size
    bc_args.rollout_episodes = args.rollout_episodes
    bc_args.rollout_max_steps = args.rollout_max_steps
    bc_args.rollout_render = False
    bc_args.force_cpu = args.force_cpu
    return bc_args


def main():
    args = build_parser().parse_args()
    mixed_root = Path(args.mixed_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    ratio_lookup = dict(RATIO_SPECS)
    summary_rows = []

    for ratio_name in args.ratios:
        ratio_value = ratio_lookup[ratio_name]
        data_path = mixed_root / ratio_name / "dual_arm_mixed.hdf5"
        manifest_path = mixed_root / ratio_name / "dual_arm_mixed.manifest.json"
        if not data_path.exists():
            raise FileNotFoundError(f"Missing mixed HDF5 for {ratio_name}: {data_path}")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing mixed manifest for {ratio_name}: {manifest_path}")

        output_dir = output_root / ratio_name / f"seed_{args.seed}"
        print(f"Running BC for {ratio_name} ({ratio_value:.1f}) using {data_path}")
        metrics = run_bc_train(build_bc_args(args, data_path=data_path, output_dir=output_dir))
        rollout_metrics = metrics.get("rollout_metrics") or {}

        summary_rows.append(
            {
                "ratio": ratio_value,
                "train_samples": metrics["train_samples"],
                "val_samples": metrics["val_samples"],
                "best_val_loss": metrics["best_val_loss"],
                "success_rate": rollout_metrics.get("success_rate"),
                "mean_steps": rollout_metrics.get("mean_steps"),
                "mean_reward": rollout_metrics.get("mean_reward"),
            }
        )

    summary_csv_path = output_root / "summary.csv"
    with summary_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ratio",
                "train_samples",
                "val_samples",
                "best_val_loss",
                "success_rate",
                "mean_steps",
                "mean_reward",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary CSV to: {summary_csv_path}")


if __name__ == "__main__":
    main()
