from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def load_robosuite_hdf5(path: str | Path):
    path = Path(path)
    demos = []
    with h5py.File(path, "r") as f:
        data_group = f["data"]
        demo_names = sorted(
            [name for name in data_group.keys() if name.startswith("demo_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        for demo_name in demo_names:
            demo = data_group[demo_name]
            states = np.asarray(demo["states"][()], dtype=np.float32)
            actions = np.asarray(demo["actions"][()], dtype=np.float32)
            if states.shape[0] != actions.shape[0]:
                raise ValueError(f"{demo_name} has mismatched state/action lengths: {states.shape} vs {actions.shape}")
            demos.append(
                {
                    "name": demo_name,
                    "obs": states,
                    "actions": actions,
                }
            )
    return demos


def normalize_data(train_obs: np.ndarray, train_act: np.ndarray):
    obs_mean = np.mean(train_obs, axis=0).astype(np.float32)
    obs_std = (np.std(train_obs, axis=0) + 1e-8).astype(np.float32)
    # Keep per-action-dimension statistics instead of collapsing all 14 action dims into one scalar.
    act_mean = np.mean(train_act, axis=0).astype(np.float32)
    act_std = (np.std(train_act, axis=0) + 1e-8).astype(np.float32)

    def norm_obs(obs):
        return (obs - obs_mean) / obs_std

    def norm_act(act):
        return (act - act_mean) / act_std

    def denorm_act(act):
        return act * act_std + act_mean

    stats = {
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "act_mean": act_mean,
        "act_std": act_std,
    }
    return norm_obs, norm_act, denorm_act, stats

