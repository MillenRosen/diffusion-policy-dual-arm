import argparse
from pathlib import Path

from mix_hdf5 import mix_hdf5


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
    project_root = project_new_root.parent
    parser = argparse.ArgumentParser(description="Generate the fixed-total dual-arm mixed HDF5 datasets.")
    parser.add_argument(
        "--clean-hdf5",
        type=str,
        default=str(project_root / "demonstrations" / "dual_arm" / "expert" / "dual_arm_clean.hdf5"),
    )
    parser.add_argument(
        "--noisy-hdf5",
        type=str,
        default=str(project_root / "demonstrations" / "dual_arm" / "noisy" / "dual_arm_noisy.hdf5"),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(project_new_root / "experiments" / "mixed_data" / "dual_arm"),
    )
    parser.add_argument("--total-trajs", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    return parser


def main():
    args = build_parser().parse_args()
    clean_hdf5 = Path(args.clean_hdf5)
    noisy_hdf5 = Path(args.noisy_hdf5)
    output_root = Path(args.output_root)

    for ratio_name, ratio_value in RATIO_SPECS:
        ratio_dir = output_root / ratio_name
        output_hdf5 = ratio_dir / "dual_arm_mixed.hdf5"
        output_manifest = ratio_dir / "dual_arm_mixed.manifest.json"
        manifest = mix_hdf5(
            clean_hdf5_path=clean_hdf5,
            noisy_hdf5_path=noisy_hdf5,
            noisy_ratio=ratio_value,
            total_trajs=args.total_trajs,
            seed=args.seed,
            output_hdf5_path=output_hdf5,
            manifest_path=output_manifest,
            task="dual_arm",
            train_ratio=args.train_ratio,
        )
        print(
            f"{ratio_name}: clean={manifest['clean_count']} noisy={manifest['noisy_count']} "
            f"-> {output_hdf5}"
        )


if __name__ == "__main__":
    main()
