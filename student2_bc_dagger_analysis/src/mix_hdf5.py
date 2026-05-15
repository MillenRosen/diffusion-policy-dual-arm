import argparse
import json
import random
from pathlib import Path

import h5py


DEFAULT_TRAIN_RATIO = 0.8


def parse_ratio(value: str) -> float:
    ratio = float(value)
    if ratio < 0.0 or ratio > 1.0:
        raise argparse.ArgumentTypeError("noisy ratio must be within [0, 1]")
    return ratio


def demo_sort_key(name: str):
    try:
        return int(name.split("_")[-1])
    except ValueError:
        return name


def list_demo_names(hdf5_path: Path):
    with h5py.File(hdf5_path, "r") as f:
        return sorted(f["data"].keys(), key=demo_sort_key)


def get_demo_lengths(hdf5_path: Path):
    with h5py.File(hdf5_path, "r") as f:
        return {
            demo_name: int(f["data"][demo_name]["states"].shape[0])
            for demo_name in sorted(f["data"].keys(), key=demo_sort_key)
        }


def normalize_attr_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def read_top_level_attrs(hdf5_path: Path):
    with h5py.File(hdf5_path, "r") as f:
        return {key: normalize_attr_value(value) for key, value in f["data"].attrs.items()}


def validate_compatible_sources(clean_path: Path, noisy_path: Path):
    clean_attrs = read_top_level_attrs(clean_path)
    noisy_attrs = read_top_level_attrs(noisy_path)
    for key in ("env", "env_info", "repository_version"):
        if clean_attrs.get(key) != noisy_attrs.get(key):
            raise ValueError(
                f"Source HDF5 mismatch for attr '{key}': "
                f"{clean_path.name}={clean_attrs.get(key)!r}, {noisy_path.name}={noisy_attrs.get(key)!r}"
            )
    return clean_attrs


def compute_counts(total_trajs: int, noisy_ratio: float):
    noisy_count = int(round(total_trajs * noisy_ratio))
    clean_count = total_trajs - noisy_count
    if clean_count < 0 or noisy_count < 0:
        raise ValueError("invalid clean / noisy counts")
    if clean_count + noisy_count != total_trajs:
        raise ValueError("clean / noisy counts do not sum to total")
    return clean_count, noisy_count


def copy_demo_group(src_group, dst_group, source_name: str, source_demo: str):
    for key, value in src_group.attrs.items():
        dst_group.attrs[key] = value

    for dataset_name in src_group.keys():
        dst_group.create_dataset(dataset_name, data=src_group[dataset_name][:])

    dst_group.attrs["source"] = source_name
    dst_group.attrs["source_demo"] = source_demo


def build_manifest(
    task: str,
    clean_path: Path,
    noisy_path: Path,
    output_hdf5_path: Path,
    total_trajs: int,
    noisy_ratio: float,
    clean_count: int,
    noisy_count: int,
    clean_num_steps: int,
    noisy_num_steps: int,
    seed: int,
    selected_clean_demos,
    selected_noisy_demos,
    train_ratio: float,
    split_seed: int,
    train_demo_ids,
    val_demo_ids,
    split_type: str,
    subset_type: str,
    train_source_counts,
    val_source_counts,
):
    actual_ratio = float(noisy_count / total_trajs) if total_trajs > 0 else 0.0
    total_steps = clean_num_steps + noisy_num_steps
    actual_timestep_ratio = float(noisy_num_steps / total_steps) if total_steps > 0 else 0.0
    return {
        "task": task,
        "clean_hdf5_path": str(clean_path),
        "noisy_hdf5_path": str(noisy_path),
        "output_hdf5_path": str(output_hdf5_path),
        "total_trajectories": int(total_trajs),
        "target_noisy_ratio": float(noisy_ratio),
        "actual_noisy_ratio_by_trajectory": actual_ratio,
        "clean_count": int(clean_count),
        "noisy_count": int(noisy_count),
        "clean_num_steps": int(clean_num_steps),
        "noisy_num_steps": int(noisy_num_steps),
        "actual_noisy_ratio_by_timestep": actual_timestep_ratio,
        "seed": int(seed),
        "selected_clean_demos": selected_clean_demos,
        "selected_noisy_demos": selected_noisy_demos,
        "train_ratio": float(train_ratio),
        "split_seed": int(split_seed),
        "train_demo_ids": train_demo_ids,
        "val_demo_ids": val_demo_ids,
        "split_type": split_type,
        "subset_type": subset_type,
        "train_source_counts": train_source_counts,
        "val_source_counts": val_source_counts,
    }


def split_one_source(demo_ids, train_ratio: float, rng: random.Random):
    demo_ids = list(demo_ids)
    if not demo_ids:
        return [], []
    rng.shuffle(demo_ids)
    train_count = int(len(demo_ids) * train_ratio)
    if len(demo_ids) > 1:
        train_count = max(1, min(train_count, len(demo_ids) - 1))
    else:
        train_count = 1
    return demo_ids[:train_count], demo_ids[train_count:]


def build_stratified_train_val_split(clean_demo_ids, noisy_demo_ids, train_ratio: float, split_seed: int):
    rng = random.Random(split_seed)
    clean_train, clean_val = split_one_source(clean_demo_ids, train_ratio, rng)
    noisy_train, noisy_val = split_one_source(noisy_demo_ids, train_ratio, rng)
    train_demo_ids = clean_train + noisy_train
    val_demo_ids = clean_val + noisy_val
    rng.shuffle(train_demo_ids)
    rng.shuffle(val_demo_ids)
    if not val_demo_ids:
        val_demo_ids = [train_demo_ids.pop()]
    return train_demo_ids, val_demo_ids


def count_sources(demo_ids, source_by_demo_id):
    counts = {"clean": 0, "noisy": 0}
    for demo_id in demo_ids:
        counts[source_by_demo_id[demo_id]] += 1
    return counts


def mix_hdf5(
    clean_hdf5_path: Path,
    noisy_hdf5_path: Path,
    noisy_ratio: float,
    total_trajs: int,
    seed: int,
    output_hdf5_path: Path,
    manifest_path: Path | None = None,
    task: str = "dual_arm",
    train_ratio: float = DEFAULT_TRAIN_RATIO,
):
    clean_hdf5_path = clean_hdf5_path.expanduser().resolve()
    noisy_hdf5_path = noisy_hdf5_path.expanduser().resolve()
    output_hdf5_path = output_hdf5_path.expanduser().resolve()
    manifest_path = (
        manifest_path.expanduser().resolve()
        if manifest_path is not None
        else output_hdf5_path.with_suffix(".manifest.json")
    )

    top_level_attrs = validate_compatible_sources(clean_hdf5_path, noisy_hdf5_path)
    clean_demo_names = list_demo_names(clean_hdf5_path)
    noisy_demo_names = list_demo_names(noisy_hdf5_path)
    clean_demo_lengths = get_demo_lengths(clean_hdf5_path)
    noisy_demo_lengths = get_demo_lengths(noisy_hdf5_path)
    clean_count, noisy_count = compute_counts(total_trajs, noisy_ratio)

    if clean_count > len(clean_demo_names):
        raise ValueError(
            f"Requested {clean_count} clean trajectories but only found {len(clean_demo_names)} in {clean_hdf5_path}"
        )
    if noisy_count > len(noisy_demo_names):
        raise ValueError(
            f"Requested {noisy_count} noisy trajectories but only found {len(noisy_demo_names)} in {noisy_hdf5_path}"
        )

    rng = random.Random(seed)
    clean_shuffled = list(clean_demo_names)
    noisy_shuffled = list(noisy_demo_names)
    rng.shuffle(clean_shuffled)
    rng.shuffle(noisy_shuffled)
    selected_clean_demos = sorted(clean_shuffled[:clean_count], key=demo_sort_key)
    selected_noisy_demos = sorted(noisy_shuffled[:noisy_count], key=demo_sort_key)

    output_hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    selection_plan = [("clean", demo_name) for demo_name in selected_clean_demos]
    selection_plan.extend(("noisy", demo_name) for demo_name in selected_noisy_demos)
    plan_rng = random.Random(seed)
    plan_rng.shuffle(selection_plan)

    clean_num_steps = int(sum(clean_demo_lengths[demo_name] for demo_name in selected_clean_demos))
    noisy_num_steps = int(sum(noisy_demo_lengths[demo_name] for demo_name in selected_noisy_demos))
    clean_new_ids = []
    noisy_new_ids = []
    source_by_demo_id = {}

    with h5py.File(clean_hdf5_path, "r") as clean_f, h5py.File(noisy_hdf5_path, "r") as noisy_f, h5py.File(
        output_hdf5_path, "w"
    ) as out_f:
        out_data = out_f.create_group("data")

        for key, value in top_level_attrs.items():
            out_data.attrs[key] = value
        out_data.attrs["total"] = int(total_trajs)
        out_data.attrs["mixed_clean_hdf5_path"] = str(clean_hdf5_path)
        out_data.attrs["mixed_noisy_hdf5_path"] = str(noisy_hdf5_path)
        out_data.attrs["mixed_target_noisy_ratio"] = float(noisy_ratio)
        out_data.attrs["mixed_seed"] = int(seed)

        for new_index, (source_name, source_demo) in enumerate(selection_plan, start=1):
            src_root = clean_f["data"] if source_name == "clean" else noisy_f["data"]
            src_group = src_root[source_demo]
            new_demo_id = f"demo_{new_index}"
            dst_group = out_data.create_group(new_demo_id)
            copy_demo_group(src_group, dst_group, source_name=source_name, source_demo=source_demo)
            source_by_demo_id[new_demo_id] = source_name
            if source_name == "clean":
                clean_new_ids.append(new_demo_id)
            else:
                noisy_new_ids.append(new_demo_id)

    train_demo_ids, val_demo_ids = build_stratified_train_val_split(
        clean_new_ids,
        noisy_new_ids,
        train_ratio=train_ratio,
        split_seed=seed,
    )
    train_source_counts = count_sources(train_demo_ids, source_by_demo_id)
    val_source_counts = count_sources(val_demo_ids, source_by_demo_id)

    manifest = build_manifest(
        task=task,
        clean_path=clean_hdf5_path,
        noisy_path=noisy_hdf5_path,
        output_hdf5_path=output_hdf5_path,
        total_trajs=total_trajs,
        noisy_ratio=noisy_ratio,
        clean_count=clean_count,
        noisy_count=noisy_count,
        clean_num_steps=clean_num_steps,
        noisy_num_steps=noisy_num_steps,
        seed=seed,
        selected_clean_demos=selected_clean_demos,
        selected_noisy_demos=selected_noisy_demos,
        train_ratio=train_ratio,
        split_seed=seed,
        train_demo_ids=train_demo_ids,
        val_demo_ids=val_demo_ids,
        split_type="stratified",
        subset_type="nested",
        train_source_counts=train_source_counts,
        val_source_counts=val_source_counts,
    )

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description="Mix clean / noisy demonstration HDF5 files at trajectory level.")
    parser.add_argument("--clean-hdf5", type=str, required=True)
    parser.add_argument("--noisy-hdf5", type=str, required=True)
    parser.add_argument("--noisy-ratio", type=parse_ratio, required=True)
    parser.add_argument("--total-trajs", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", type=str, default="dual_arm")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--output-hdf5", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    manifest = mix_hdf5(
        clean_hdf5_path=Path(args.clean_hdf5),
        noisy_hdf5_path=Path(args.noisy_hdf5),
        noisy_ratio=args.noisy_ratio,
        total_trajs=args.total_trajs,
        seed=args.seed,
        output_hdf5_path=Path(args.output_hdf5),
        manifest_path=Path(args.manifest) if args.manifest else None,
        task=args.task,
        train_ratio=args.train_ratio,
    )
    print(
        json.dumps(
            {
                "output_hdf5_path": manifest["output_hdf5_path"],
                "total_trajectories": manifest["total_trajectories"],
                "clean_count": manifest["clean_count"],
                "noisy_count": manifest["noisy_count"],
                "actual_noisy_ratio_by_trajectory": manifest["actual_noisy_ratio_by_trajectory"],
                "actual_noisy_ratio_by_timestep": manifest["actual_noisy_ratio_by_timestep"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
