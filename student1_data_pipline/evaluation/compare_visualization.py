from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare clean and noisy HDF5 datasets and generate overlaid figures."
    )
    parser.add_argument("--clean_hdf5", type=str, required=True, help="Path to clean HDF5 file.")
    parser.add_argument("--noisy_hdf5", type=str, required=True, help="Path to noisy HDF5 file.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="analysis_outputs/compare_visualization",
        help="Directory to save comparison figures.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="comparison",
        help="Prefix for output figure names.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of bins for histogram plots.",
    )
    parser.add_argument(
        "--max_episodes_for_line",
        type=int,
        default=200,
        help="Maximum number of episodes to display in line plots.",
    )
    return parser


def parse_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def detect_format(root: h5py.Group) -> str:
    demos = sorted(list(root.keys()))
    if not demos:
        return "unknown"
    first = root[demos[0]]
    keys = set(first.keys())
    if "states" in keys and "actions" in keys:
        return "official"
    if "actions_exec" in keys or "obs_json" in keys:
        return "legacy"
    return "unknown"


def load_dataset_info(hdf5_path: str) -> Dict[str, Any]:
    lengths: List[int] = []
    mean_abs_actions: List[float] = []
    mean_state_norms: List[float] = []
    success_flags: List[bool] = []
    fmt = "unknown"

    with h5py.File(hdf5_path, "r") as f:
        root = f["data"]
        fmt = detect_format(root)
        demos = sorted(list(root.keys()))

        for demo_key in demos:
            ep_group = root[demo_key]

            if fmt == "official":
                if "actions" in ep_group:
                    actions = np.asarray(ep_group["actions"][()], dtype=np.float32)
                    lengths.append(int(actions.shape[0]))
                    mean_abs_actions.append(float(np.mean(np.abs(actions))) if actions.size > 0 else 0.0)
                if "states" in ep_group:
                    states = np.asarray(ep_group["states"][()], dtype=np.float32)
                    if states.ndim == 1:
                        states = states.reshape(1, -1)
                    state_norm = float(np.mean(np.linalg.norm(states, axis=1))) if states.size > 0 else 0.0
                    mean_state_norms.append(state_norm)
                # official robosuite HDF5 usually stores only successful demos, success attr often absent
                success_attr = ep_group.attrs.get("success", None)
                if success_attr is not None:
                    success_flags.append(bool(parse_attr(success_attr)))
            elif fmt == "legacy":
                attrs = {k: parse_attr(ep_group.attrs[k]) for k in ep_group.attrs.keys()}
                if "actions_exec" in ep_group:
                    actions = np.asarray(ep_group["actions_exec"][()], dtype=np.float32)
                    lengths.append(int(actions.shape[0]))
                    mean_abs_actions.append(float(np.mean(np.abs(actions))) if actions.size > 0 else 0.0)
                else:
                    lengths.append(int(attrs.get("length", 0)))
                success = attrs.get("success", None)
                if success is not None:
                    if isinstance(success, str):
                        success = success.lower() == "true"
                    success_flags.append(bool(success))
            else:
                # best-effort fallback
                if "actions" in ep_group:
                    actions = np.asarray(ep_group["actions"][()], dtype=np.float32)
                    lengths.append(int(actions.shape[0]))
                    mean_abs_actions.append(float(np.mean(np.abs(actions))) if actions.size > 0 else 0.0)
                elif "actions_exec" in ep_group:
                    actions = np.asarray(ep_group["actions_exec"][()], dtype=np.float32)
                    lengths.append(int(actions.shape[0]))
                    mean_abs_actions.append(float(np.mean(np.abs(actions))) if actions.size > 0 else 0.0)

    return {
        "format": fmt,
        "lengths": lengths,
        "mean_abs_actions": mean_abs_actions,
        "mean_state_norms": mean_state_norms,
        "success_flags": success_flags,
        "num_episodes": len(lengths),
    }


def _shared_bins(a: List[float], b: List[float], bins: int) -> np.ndarray:
    arr = np.asarray(list(a) + list(b), dtype=np.float32)
    if arr.size == 0:
        return np.linspace(0.0, 1.0, bins + 1)
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if abs(mx - mn) < 1e-8:
        mx = mn + 1.0
    return np.linspace(mn, mx, bins + 1)


def plot_overlaid_hist(
    clean_vals: List[float],
    noisy_vals: List[float],
    out_path: str,
    xlabel: str,
    title: str,
    bins: int,
) -> None:
    plt.figure(figsize=(6.5, 4.5))
    shared_bins = _shared_bins(clean_vals, noisy_vals, bins)
    plt.hist(clean_vals, bins=shared_bins, alpha=0.55, label="clean")
    plt.hist(noisy_vals, bins=shared_bins, alpha=0.55, label="noisy")
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_compare_line(
    clean_vals: List[float],
    noisy_vals: List[float],
    out_path: str,
    ylabel: str,
    title: str,
    max_points: int,
) -> None:
    clean_arr = np.asarray(clean_vals[:max_points], dtype=np.float32)
    noisy_arr = np.asarray(noisy_vals[:max_points], dtype=np.float32)
    n = max(len(clean_arr), len(noisy_arr))
    plt.figure(figsize=(7.0, 4.5))
    if len(clean_arr) > 0:
        plt.plot(np.arange(len(clean_arr)), clean_arr, label="clean")
    if len(noisy_arr) > 0:
        plt.plot(np.arange(len(noisy_arr)), noisy_arr, label="noisy")
    plt.xlabel("episode index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_success_bar(clean_flags: List[bool], noisy_flags: List[bool], out_path: str) -> Optional[Dict[str, Any]]:
    if len(clean_flags) == 0 and len(noisy_flags) == 0:
        return None

    clean_success = int(sum(clean_flags)) if clean_flags else None
    clean_fail = int(len(clean_flags) - sum(clean_flags)) if clean_flags else None
    noisy_success = int(sum(noisy_flags)) if noisy_flags else None
    noisy_fail = int(len(noisy_flags) - sum(noisy_flags)) if noisy_flags else None

    labels = []
    success_vals = []
    fail_vals = []
    if clean_flags:
        labels.append("clean")
        success_vals.append(clean_success)
        fail_vals.append(clean_fail)
    if noisy_flags:
        labels.append("noisy")
        success_vals.append(noisy_success)
        fail_vals.append(noisy_fail)

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(6.0, 4.5))
    plt.bar(x - width / 2, success_vals, width=width, label="success")
    plt.bar(x + width / 2, fail_vals, width=width, label="fail")
    plt.xticks(x, labels)
    plt.ylabel("count")
    plt.title("Success / Failure Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

    return {
        "clean_success": clean_success,
        "clean_fail": clean_fail,
        "noisy_success": noisy_success,
        "noisy_fail": noisy_fail,
    }


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.out_dir)

    clean_info = load_dataset_info(args.clean_hdf5)
    noisy_info = load_dataset_info(args.noisy_hdf5)

    length_hist_path = os.path.join(args.out_dir, f"{args.prefix}_length_hist_compare.png")
    action_hist_path = os.path.join(args.out_dir, f"{args.prefix}_action_hist_compare.png")
    action_line_path = os.path.join(args.out_dir, f"{args.prefix}_action_line_compare.png")
    state_hist_path = os.path.join(args.out_dir, f"{args.prefix}_state_norm_hist_compare.png")
    success_bar_path = os.path.join(args.out_dir, f"{args.prefix}_success_bar_compare.png")

    plot_overlaid_hist(
        clean_info["lengths"],
        noisy_info["lengths"],
        length_hist_path,
        xlabel="trajectory length",
        title="Clean vs Noisy Trajectory Length Distribution",
        bins=args.bins,
    )

    plot_overlaid_hist(
        clean_info["mean_abs_actions"],
        noisy_info["mean_abs_actions"],
        action_hist_path,
        xlabel="per-episode mean |action|",
        title="Clean vs Noisy Action Magnitude Distribution",
        bins=args.bins,
    )

    plot_compare_line(
        clean_info["mean_abs_actions"],
        noisy_info["mean_abs_actions"],
        action_line_path,
        ylabel="mean |action|",
        title="Per-Episode Mean Action Magnitude",
        max_points=args.max_episodes_for_line,
    )

    state_hist_written = False
    if len(clean_info["mean_state_norms"]) > 0 or len(noisy_info["mean_state_norms"]) > 0:
        plot_overlaid_hist(
            clean_info["mean_state_norms"],
            noisy_info["mean_state_norms"],
            state_hist_path,
            xlabel="per-episode mean state norm",
            title="Clean vs Noisy State Norm Distribution",
            bins=args.bins,
        )
        state_hist_written = True

    success_stats = plot_success_bar(clean_info["success_flags"], noisy_info["success_flags"], success_bar_path)

    summary = {
        "clean_hdf5": args.clean_hdf5,
        "noisy_hdf5": args.noisy_hdf5,
        "clean_format": clean_info["format"],
        "noisy_format": noisy_info["format"],
        "clean_num_episodes": clean_info["num_episodes"],
        "noisy_num_episodes": noisy_info["num_episodes"],
        "plots": {
            "length_hist_compare": length_hist_path,
            "action_hist_compare": action_hist_path,
            "action_line_compare": action_line_path,
            "state_norm_hist_compare": state_hist_path if state_hist_written else None,
            "success_bar_compare": success_bar_path if success_stats is not None else None,
        },
        "success_stats": success_stats,
    }

    summary_path = os.path.join(args.out_dir, f"{args.prefix}_comparison_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Comparison visualization finished.")
    print(f"Summary JSON         : {summary_path}")
    print(f"Length hist compare  : {length_hist_path}")
    print(f"Action hist compare  : {action_hist_path}")
    print(f"Action line compare  : {action_line_path}")
    if state_hist_written:
        print(f"State norm hist comp : {state_hist_path}")
    if success_stats is not None:
        print(f"Success bar compare  : {success_bar_path}")


if __name__ == "__main__":
    main()
