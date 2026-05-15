from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import h5py
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate dataset visualizations from an HDF5 file (official robosuite or legacy custom format)."
    )
    parser.add_argument("--hdf5_path", type=str, required=True, help="Path to the HDF5 file.")
    parser.add_argument("--out_dir", type=str, default="analysis_outputs/visualization", help="Directory to save figures.")
    parser.add_argument("--prefix", type=str, default=None, help="Optional prefix for figure filenames.")
    return parser


def parse_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
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
    if "obs_json" in keys or "actions_exec" in keys:
        return "legacy"
    return "unknown"


def load_dataset_info(hdf5_path: str) -> Dict[str, Any]:
    lengths: List[int] = []
    success_flags: List[bool] = []
    success_available = False
    final_stages: List[str] = []
    mean_abs_actions: List[float] = []
    mean_state_norms: List[float] = []

    with h5py.File(hdf5_path, "r") as f:
        root = f["data"]
        fmt = detect_format(root)
        demos = sorted(list(root.keys()))

        for demo_key in demos:
            ep_group = root[demo_key]

            if fmt == "official":
                if "actions" in ep_group:
                    actions = ep_group["actions"][()]
                    lengths.append(int(actions.shape[0]))
                    mean_abs_actions.append(float(np.mean(np.abs(actions))))
                if "states" in ep_group:
                    states = ep_group["states"][()]
                    if states.size > 0:
                        mean_state_norms.append(float(np.mean(np.linalg.norm(states, axis=1))))
                if "success" in ep_group.attrs:
                    success_available = True
                    success_flags.append(bool(parse_attr(ep_group.attrs["success"])))

            elif fmt == "legacy":
                attrs = {k: parse_attr(ep_group.attrs[k]) for k in ep_group.attrs.keys()}
                success = attrs.get("success", False)
                if isinstance(success, str):
                    success = success.lower() == "true"
                success_available = True
                success_flags.append(bool(success))

                stage = attrs.get("final_stage", None)
                if stage not in [None, "null"]:
                    final_stages.append(str(stage))

                if "actions_exec" in ep_group:
                    actions = ep_group["actions_exec"][()]
                    lengths.append(int(actions.shape[0]))
                    mean_abs_actions.append(float(np.mean(np.abs(actions))))
                else:
                    lengths.append(int(attrs.get("length", 0)))
            else:
                raise RuntimeError("Unsupported HDF5 structure.")

    return {
        "lengths": lengths,
        "success_flags": success_flags,
        "success_available": success_available,
        "final_stages": final_stages,
        "mean_abs_actions": mean_abs_actions,
        "mean_state_norms": mean_state_norms,
    }


def plot_success_bar(success_flags: List[bool], out_path: str) -> None:
    success = int(sum(success_flags))
    fail = int(len(success_flags) - success)
    plt.figure(figsize=(5, 4))
    plt.bar(["success", "fail"], [success, fail])
    plt.ylabel("count")
    plt.title("Success / Failure Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_length_hist(lengths: List[int], out_path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(lengths, bins=min(20, max(5, len(lengths))))
    plt.xlabel("trajectory length")
    plt.ylabel("count")
    plt.title("Trajectory Length Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_stage_bar(final_stages: List[str], out_path: str) -> None:
    counts: Dict[str, int] = {}
    for s in final_stages:
        counts[s] = counts.get(s, 0) + 1
    labels = sorted(counts.keys(), key=lambda x: str(x))
    values = [counts[k] for k in labels]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.xlabel("final stage")
    plt.ylabel("count")
    plt.title("Final Stage Counts")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_action_magnitude(mean_abs_actions: List[float], out_path: str) -> None:
    plt.figure(figsize=(6, 4))
    x = np.arange(len(mean_abs_actions))
    plt.plot(x, mean_abs_actions)
    plt.xlabel("episode index")
    plt.ylabel("mean |action|")
    plt.title("Per-Episode Mean Action Magnitude")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_state_norm(mean_state_norms: List[float], out_path: str) -> None:
    plt.figure(figsize=(6, 4))
    x = np.arange(len(mean_state_norms))
    plt.plot(x, mean_state_norms)
    plt.xlabel("episode index")
    plt.ylabel("mean ||state||")
    plt.title("Per-Episode Mean State Norm")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.out_dir)
    prefix = args.prefix or os.path.splitext(os.path.basename(args.hdf5_path))[0]

    info = load_dataset_info(args.hdf5_path)

    success_plot = os.path.join(args.out_dir, f"{prefix}_success_bar.png")
    length_plot = os.path.join(args.out_dir, f"{prefix}_length_hist.png")
    stage_plot = os.path.join(args.out_dir, f"{prefix}_final_stage_bar.png")
    action_plot = os.path.join(args.out_dir, f"{prefix}_action_magnitude.png")
    state_plot = os.path.join(args.out_dir, f"{prefix}_state_norm.png")

    plot_length_hist(info["lengths"], length_plot)
    if info["success_available"] and info["success_flags"]:
        plot_success_bar(info["success_flags"], success_plot)
    if len(info["final_stages"]) > 0:
        plot_stage_bar(info["final_stages"], stage_plot)
    if len(info["mean_abs_actions"]) > 0:
        plot_action_magnitude(info["mean_abs_actions"], action_plot)
    if len(info["mean_state_norms"]) > 0:
        plot_state_norm(info["mean_state_norms"], state_plot)

    summary = {
        "hdf5_path": args.hdf5_path,
        "num_episodes": len(info["lengths"]),
        "num_success": int(sum(info["success_flags"])) if info["success_available"] else None,
        "success_rate": float(sum(info["success_flags"]) / len(info["success_flags"]))
        if info["success_available"] and info["success_flags"] else None,
        "plots": {
            "success_bar": success_plot if info["success_available"] and info["success_flags"] else None,
            "length_hist": length_plot,
            "final_stage_bar": stage_plot if len(info["final_stages"]) > 0 else None,
            "action_magnitude": action_plot if len(info["mean_abs_actions"]) > 0 else None,
            "state_norm": state_plot if len(info["mean_state_norms"]) > 0 else None,
        },
    }

    json_path = os.path.join(args.out_dir, f"{prefix}_visualization_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Visualization finished.")
    print(f"Summary JSON : {json_path}")
    print(f"Length plot  : {length_plot}")
    if summary["plots"]["success_bar"]:
        print(f"Success plot : {success_plot}")
    if summary["plots"]["final_stage_bar"]:
        print(f"Stage plot   : {stage_plot}")
    if summary["plots"]["action_magnitude"]:
        print(f"Action plot  : {action_plot}")
    if summary["plots"]["state_norm"]:
        print(f"State plot   : {state_plot}")


if __name__ == "__main__":
    main()
