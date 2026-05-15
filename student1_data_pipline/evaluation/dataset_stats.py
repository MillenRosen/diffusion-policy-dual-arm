from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import h5py
import numpy as np


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute dataset-level statistics from an HDF5 dataset (official robosuite or legacy custom format)."
    )
    parser.add_argument("--hdf5_path", type=str, required=True, help="Path to the HDF5 file.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="analysis_outputs/dataset_stats",
        help="Directory to save the statistics JSON and TXT summary.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Optional output filename prefix. Defaults to the HDF5 stem.",
    )
    return parser


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8")
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_serializable(v) for v in obj]
    return obj


def parse_attr(value: Any) -> Any:
    value = to_serializable(value)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def detect_format(root: h5py.Group) -> str:
    demos = sorted(list(root.keys()))
    if not demos:
        return "unknown"
    first = root[demos[0]]
    keys = set(first.keys())
    if {"states", "actions"}.issubset(keys):
        return "official"
    if "actions_exec" in keys or "obs_json" in keys:
        return "legacy"
    return "unknown"


def collect_action_stats(actions: np.ndarray) -> Dict[str, Any]:
    if actions.size == 0:
        return {
            "shape": list(actions.shape),
            "mean_abs": None,
            "std_abs": None,
            "max_abs": None,
            "per_dim_mean_abs": None,
            "per_dim_std_abs": None,
        }
    abs_actions = np.abs(actions)
    return {
        "shape": list(actions.shape),
        "mean_abs": float(np.mean(abs_actions)),
        "std_abs": float(np.std(abs_actions)),
        "max_abs": float(np.max(abs_actions)),
        "per_dim_mean_abs": np.mean(abs_actions, axis=0).astype(np.float32).tolist(),
        "per_dim_std_abs": np.std(abs_actions, axis=0).astype(np.float32).tolist(),
    }


def collect_state_stats(states: np.ndarray) -> Dict[str, Any]:
    if states.size == 0:
        return {"shape": list(states.shape), "mean_norm": None, "std_norm": None}
    norms = np.linalg.norm(states, axis=1)
    return {
        "shape": list(states.shape),
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms)),
    }


def compute_stats(hdf5_path: str) -> Dict[str, Any]:
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            raise RuntimeError("Expected top-level group 'data' in HDF5 file.")
        root = f["data"]
        fmt = detect_format(root)
        demo_keys = sorted(list(root.keys()))

        lengths: List[int] = []
        success_flags: List[bool] = []
        success_available = False
        all_actions: List[np.ndarray] = []
        all_states: List[np.ndarray] = []
        final_stage_counts: Dict[str, int] = {}

        for key in demo_keys:
            ep_group = root[key]

            if fmt == "official":
                actions = ep_group["actions"][()] if "actions" in ep_group else np.zeros((0, 0), dtype=np.float32)
                states = ep_group["states"][()] if "states" in ep_group else np.zeros((0, 0), dtype=np.float32)
                lengths.append(int(actions.shape[0]))
                if actions.size > 0:
                    all_actions.append(actions)
                if states.size > 0:
                    all_states.append(states)
                # Official robosuite export usually stores only saved demos and often lacks per-demo success attrs.
                if "success" in ep_group.attrs:
                    success_available = True
                    success = parse_attr(ep_group.attrs["success"])
                    success_flags.append(bool(success))

            elif fmt == "legacy":
                attrs = {k: parse_attr(ep_group.attrs[k]) for k in ep_group.attrs.keys()}
                actions = ep_group["actions_exec"][()] if "actions_exec" in ep_group else np.zeros((0, 0), dtype=np.float32)
                lengths.append(int(actions.shape[0]) if actions.size > 0 else int(attrs.get("length", 0)))
                if actions.size > 0:
                    all_actions.append(actions)

                if "success" in attrs:
                    success_available = True
                    success_flags.append(bool(attrs.get("success", False)))

                stage = attrs.get("final_stage", None)
                if stage not in [None, "null"]:
                    stage_key = str(stage)
                    final_stage_counts[stage_key] = final_stage_counts.get(stage_key, 0) + 1
            else:
                raise RuntimeError("Unsupported HDF5 structure. Could not detect official or legacy format.")

        actions_cat = np.concatenate(all_actions, axis=0) if all_actions else np.zeros((0, 0), dtype=np.float32)
        states_cat = np.concatenate(all_states, axis=0) if all_states else np.zeros((0, 0), dtype=np.float32)

        root_attrs = {k: parse_attr(root.attrs[k]) for k in root.attrs.keys()}
        total_attr = root_attrs.get("total", None)

        stats: Dict[str, Any] = {
            "hdf5_path": hdf5_path,
            "format": fmt,
            "num_episodes": int(len(demo_keys)),
            "root_total_attr": int(total_attr) if isinstance(total_attr, (int, np.integer)) else total_attr,
            "avg_length": float(np.mean(lengths)) if lengths else None,
            "std_length": float(np.std(lengths)) if lengths else None,
            "min_length": int(np.min(lengths)) if lengths else None,
            "max_length": int(np.max(lengths)) if lengths else None,
            "actions_stats": collect_action_stats(actions_cat),
            "states_stats": collect_state_stats(states_cat),
            "root_attrs": to_serializable(root_attrs),
            "final_stage_counts": final_stage_counts if final_stage_counts else None,
        }

        if success_available:
            stats["num_success"] = int(sum(success_flags))
            stats["success_rate"] = float(sum(success_flags) / len(success_flags)) if success_flags else 0.0
        else:
            stats["num_success"] = None
            stats["success_rate"] = None
            stats["success_note"] = (
                "Per-demo success is not stored in this HDF5 format. "
                "For official robosuite exports, success is often implicit because only saved demos are kept."
            )

        return to_serializable(stats)


def format_stats_text(stats: Dict[str, Any]) -> str:
    lines = [
        f"hdf5_path: {stats['hdf5_path']}",
        f"format: {stats['format']}",
        f"num_episodes: {stats['num_episodes']}",
        f"avg_length: {stats['avg_length']}",
        f"std_length: {stats['std_length']}",
        f"min_length: {stats['min_length']}",
        f"max_length: {stats['max_length']}",
        f"actions.mean_abs: {stats['actions_stats']['mean_abs']}",
        f"actions.max_abs: {stats['actions_stats']['max_abs']}",
        f"states.mean_norm: {stats['states_stats']['mean_norm']}",
        f"states.std_norm: {stats['states_stats']['std_norm']}",
    ]
    if stats.get("success_rate") is not None:
        lines += [
            f"num_success: {stats['num_success']}",
            f"success_rate: {stats['success_rate']:.4f}",
        ]
    else:
        lines.append(f"success_rate: unavailable ({stats.get('success_note', 'not stored')})")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.out_dir)

    prefix = args.prefix or os.path.splitext(os.path.basename(args.hdf5_path))[0]
    stats = compute_stats(args.hdf5_path)

    json_path = os.path.join(args.out_dir, f"{prefix}_stats.json")
    txt_path = os.path.join(args.out_dir, f"{prefix}_stats.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(format_stats_text(stats))

    print("Dataset statistics saved.")
    print(f"JSON: {json_path}")
    print(f"TXT : {txt_path}")


if __name__ == "__main__":
    main()
