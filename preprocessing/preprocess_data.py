from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils import load_robosuite_hdf5, normalize_data


def create_diffusion_dataset(demos, norm_obs, norm_act, obs_horizon=2, action_chunk=4):
    obs_list, act_list = [], []
    for demo in demos:
        obs = demo["obs"]
        acts = demo["actions"]
        length = obs.shape[0]
        if length < obs_horizon + action_chunk:
            continue
        for t in range(length - obs_horizon - action_chunk + 1):
            obs_window = obs[t : t + obs_horizon]
            act_window = acts[t + obs_horizon - 1 : t + obs_horizon - 1 + action_chunk]
            obs_list.append(norm_obs(obs_window))
            act_list.append(norm_act(act_window))
    return np.asarray(obs_list, dtype=np.float32), np.asarray(act_list, dtype=np.float32)


def split_train_val(demos, train_ratio=0.8, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(demos))
    rng.shuffle(indices)
    train_count = max(1, int(len(indices) * train_ratio))
    train_idx = set(indices[:train_count].tolist())
    train_demos = [demos[i] for i in range(len(demos)) if i in train_idx]
    val_demos = [demos[i] for i in range(len(demos)) if i not in train_idx]
    if not val_demos:
        val_demos = [train_demos.pop()]
    return train_demos, val_demos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_path",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "script_collect_demo2.hdf5"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "processed_scripted"),
    )
    parser.add_argument("--obs_horizon", type=int, default=2)
    parser.add_argument("--action_chunk", type=int, default=4)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    demos = load_robosuite_hdf5(args.raw_path)
    train_demos, val_demos = split_train_val(demos, train_ratio=args.train_ratio, seed=args.seed)

    all_train_obs = np.concatenate([d["obs"] for d in train_demos], axis=0)
    all_train_act = np.concatenate([d["actions"] for d in train_demos], axis=0)
    norm_obs, norm_act, _, stats = normalize_data(all_train_obs, all_train_act)

    obs_train, act_train = create_diffusion_dataset(
        train_demos, norm_obs, norm_act, args.obs_horizon, args.action_chunk
    )
    obs_val, act_val = create_diffusion_dataset(
        val_demos, norm_obs, norm_act, args.obs_horizon, args.action_chunk
    )

    np.savez(
        output_dir / f"diffusion_h{args.obs_horizon}_c{args.action_chunk}_data.npz",
        obs_train=obs_train,
        act_train=act_train,
        obs_val=obs_val,
        act_val=act_val,
    )
    np.savez(output_dir / "norm_stats.npz", **stats)

    meta = {
        "raw_path": args.raw_path,
        "num_demos": len(demos),
        "train_demos": len(train_demos),
        "val_demos": len(val_demos),
        "obs_horizon": args.obs_horizon,
        "action_chunk": args.action_chunk,
        "obs_dim": int(all_train_obs.shape[1]),
        "act_dim": int(all_train_act.shape[1]),
        "train_windows": int(obs_train.shape[0]),
        "val_windows": int(obs_val.shape[0]),
    }
    (output_dir / "preprocess_summary.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in meta.items()),
        encoding="utf-8",
    )
    print(meta)


if __name__ == "__main__":
    main()
