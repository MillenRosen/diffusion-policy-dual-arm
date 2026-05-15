from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run coverage analysis with PCA projections from one or more HDF5 datasets (official or legacy)."
    )
    parser.add_argument("--hdf5_paths", type=str, nargs="+", required=True, help="One or more HDF5 files.")
    parser.add_argument("--labels", type=str, nargs="*", default=None, help="Optional labels matching the HDF5 paths.")
    parser.add_argument("--out_dir", type=str, default="analysis_outputs/coverage", help="Directory to save outputs.")
    parser.add_argument("--max_points_per_dataset", type=int, default=5000, help="Maximum vectors to sample from each dataset.")
    parser.add_argument(
        "--feature_mode",
        type=str,
        default="states",
        choices=["states", "actions", "legacy_obs"],
        help="Feature source for PCA. Use 'states' for official robosuite HDF5.",
    )
    parser.add_argument(
        "--sample_mode",
        type=str,
        default="all",
        choices=["first", "all"],
        help="'first' uses only the first vector per episode, 'all' uses all vectors.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling.")
    return parser


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


def parse_obs_json_item(item: Any) -> Dict[str, Any]:
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    if isinstance(item, str):
        return json.loads(item)
    raise TypeError(f"Unsupported obs_json item type: {type(item)}")


def obs_dict_to_vector(obs: Dict[str, Any]) -> np.ndarray:
    parts = []
    for k in sorted(obs.keys()):
        arr = np.asarray(obs[k])
        if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
            parts.append(arr.astype(np.float32).reshape(-1))
    if not parts:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)


def subsample_rows(arr: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if arr.shape[0] <= max_points:
        return arr
    idx = rng.choice(arr.shape[0], size=max_points, replace=False)
    return arr[idx]


def load_vectors(hdf5_path: str, feature_mode: str, sample_mode: str, max_points: int, rng: np.random.Generator) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as f:
        root = f["data"]
        fmt = detect_format(root)
        demos = sorted(list(root.keys()))
        vectors: List[np.ndarray] = []

        for demo_key in demos:
            ep_group = root[demo_key]

            if fmt == "official":
                if feature_mode == "states":
                    if "states" not in ep_group:
                        continue
                    arr = ep_group["states"][()].astype(np.float32)
                elif feature_mode == "actions":
                    if "actions" not in ep_group:
                        continue
                    arr = ep_group["actions"][()].astype(np.float32)
                else:
                    raise RuntimeError("legacy_obs mode is only valid for legacy HDF5 files.")

                if arr.ndim != 2 or arr.shape[0] == 0:
                    continue
                arr = arr[:1] if sample_mode == "first" else arr
                vectors.append(arr)

            elif fmt == "legacy":
                if feature_mode == "legacy_obs":
                    if "obs_json" not in ep_group:
                        continue
                    obs_json = ep_group["obs_json"][()]
                    if len(obs_json) == 0:
                        continue
                    items = [obs_json[0]] if sample_mode == "first" else list(obs_json)
                    rows = []
                    for item in items:
                        vec = obs_dict_to_vector(parse_obs_json_item(item))
                        if vec.size > 0:
                            rows.append(vec)
                    if rows:
                        vectors.append(np.stack(rows, axis=0))
                else:
                    key = "actions_exec" if feature_mode == "actions" else None
                    if key is None or key not in ep_group:
                        continue
                    arr = ep_group[key][()].astype(np.float32)
                    if arr.ndim != 2 or arr.shape[0] == 0:
                        continue
                    arr = arr[:1] if sample_mode == "first" else arr
                    vectors.append(arr)
            else:
                raise RuntimeError(f"Unsupported HDF5 structure: {hdf5_path}")

    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)

    min_dim = min(v.shape[1] for v in vectors)
    arr = np.concatenate([v[:, :min_dim] for v in vectors], axis=0).astype(np.float32)
    return subsample_rows(arr, max_points, rng)


def pca_project(X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}")
    mean = np.mean(X, axis=0, keepdims=True)
    Xc = X - mean
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:n_components]
    Z = Xc @ components.T
    var = (S ** 2) / max(1, X.shape[0] - 1)
    explained_ratio = var[:n_components] / np.sum(var) if np.sum(var) > 0 else np.zeros(n_components)
    return Z.astype(np.float32), mean.squeeze(0), explained_ratio.astype(np.float32)


def compute_pairwise_center_distances(projections: Dict[str, np.ndarray]) -> Dict[str, float]:
    labels = list(projections.keys())
    centers = {k: np.mean(v, axis=0) for k, v in projections.items() if v.shape[0] > 0}
    out: Dict[str, float] = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            if a in centers and b in centers:
                out[f"{a}__vs__{b}"] = float(np.linalg.norm(centers[a] - centers[b]))
    return out


def scatter_plot(projections: Dict[str, np.ndarray], out_path: str, feature_mode: str) -> None:
    plt.figure(figsize=(7, 6))
    for label, Z in projections.items():
        if Z.shape[0] == 0:
            continue
        plt.scatter(Z[:, 0], Z[:, 1], s=8, alpha=0.5, label=label)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"Coverage PCA ({feature_mode})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.out_dir)
    rng = np.random.default_rng(args.seed)

    hdf5_paths = args.hdf5_paths
    labels = args.labels or [os.path.splitext(os.path.basename(p))[0] for p in hdf5_paths]
    if len(labels) != len(hdf5_paths):
        raise ValueError("Number of labels must match number of hdf5_paths.")

    vectors_per_dataset: Dict[str, np.ndarray] = {}
    for path, label in zip(hdf5_paths, labels):
        vectors_per_dataset[label] = load_vectors(
            hdf5_path=path,
            feature_mode=args.feature_mode,
            sample_mode=args.sample_mode,
            max_points=args.max_points_per_dataset,
            rng=rng,
        )

    valid = [v for v in vectors_per_dataset.values() if v.size > 0]
    if not valid:
        raise RuntimeError("No valid vectors were found in the provided datasets.")

    min_dim = min(v.shape[1] for v in valid)
    trimmed = {k: v[:, :min_dim] for k, v in vectors_per_dataset.items() if v.size > 0}
    X_all = np.concatenate(list(trimmed.values()), axis=0)
    Z_all, _, explained_ratio = pca_project(X_all, n_components=2)

    projections: Dict[str, np.ndarray] = {}
    cursor = 0
    for label in labels:
        X = trimmed.get(label, np.zeros((0, min_dim), dtype=np.float32))
        n = X.shape[0]
        projections[label] = Z_all[cursor:cursor + n]
        cursor += n

    plot_path = os.path.join(args.out_dir, f"coverage_pca_{args.feature_mode}.png")
    scatter_plot(projections, plot_path, args.feature_mode)

    stats = {
        "datasets": {
            label: {
                "num_points": int(projections[label].shape[0]),
                "center": np.mean(projections[label], axis=0).astype(np.float32).tolist()
                if projections[label].shape[0] > 0 else None,
            }
            for label in labels
        },
        "explained_variance_ratio": explained_ratio.tolist(),
        "pairwise_center_distances": compute_pairwise_center_distances(projections),
        "plot_path": plot_path,
        "feature_mode": args.feature_mode,
        "sample_mode": args.sample_mode,
        "max_points_per_dataset": args.max_points_per_dataset,
    }

    json_path = os.path.join(args.out_dir, f"coverage_stats_{args.feature_mode}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("Coverage analysis finished.")
    print(f"Stats JSON: {json_path}")
    print(f"PCA Plot  : {plot_path}")


if __name__ == "__main__":
    main()
