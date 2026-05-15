import argparse
import json
import math
import os
import platform
import random
from pathlib import Path


def prepare_windows_runtime():
    if platform.system() != "Windows":
        return
    tmp_dir = Path(r"C:\tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(tmp_dir)
    os.environ["TMP"] = str(tmp_dir)
    if os.environ.get("MUJOCO_GL", "").strip().lower() in {"", "egl"}:
        os.environ["MUJOCO_GL"] = "wgl"


prepare_windows_runtime()

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def demo_sort_key(name: str):
    try:
        return int(name.split("_")[-1])
    except ValueError:
        return name


def normalize_attr_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def resolve_data_paths(args):
    if not args.data:
        raise ValueError("Please provide at least one --data HDF5 path.")

    unique_paths = []
    seen = set()
    for path_text in args.data:
        path = Path(path_text).expanduser().resolve()
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)

    missing = [str(path) for path in unique_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset file(s): {missing}")
    return unique_paths


def read_env_metadata(hdf5_path: Path):
    with h5py.File(hdf5_path, "r") as f:
        attrs = f["data"].attrs
        env_name = normalize_attr_value(attrs.get("env", "TwoArmLift"))
        env_info_raw = normalize_attr_value(attrs.get("env_info"))
        env_info = json.loads(env_info_raw) if env_info_raw else {}
    return env_name, env_info


def load_trajectories(hdf5_paths):
    trajectories = []
    dataset_summaries = []

    for hdf5_path in hdf5_paths:
        lengths = []
        source_counts = {}
        with h5py.File(hdf5_path, "r") as f:
            demo_names = sorted(f["data"].keys(), key=demo_sort_key)
            for demo_name in demo_names:
                group = f["data"][demo_name]
                states = group["states"][:].astype(np.float32)
                actions = group["actions"][:].astype(np.float32)
                if len(states) != len(actions):
                    raise ValueError(
                        f"{hdf5_path.name}/{demo_name} has mismatched lengths: {len(states)} vs {len(actions)}"
                    )

                source = normalize_attr_value(group.attrs.get("source", hdf5_path.stem))
                source_demo = normalize_attr_value(group.attrs.get("source_demo", demo_name))
                source_counts[source] = source_counts.get(source, 0) + 1

                trajectories.append(
                    {
                        "name": demo_name,
                        "source": source,
                        "source_demo": source_demo,
                        "source_path": str(hdf5_path),
                        "states": states,
                        "actions": actions,
                    }
                )
                lengths.append(len(states))

        dataset_summaries.append(
            {
                "path": str(hdf5_path),
                "num_trajectories": len(lengths),
                "total_steps": int(sum(lengths)),
                "traj_len_min": int(min(lengths)),
                "traj_len_mean": float(np.mean(lengths)),
                "traj_len_max": int(max(lengths)),
                "trajectory_sources": source_counts,
            }
        )

    return trajectories, dataset_summaries


def split_trajectories(trajectories, train_ratio: float, seed: int):
    indices = list(range(len(trajectories)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_count = max(1, int(len(indices) * train_ratio))
    train_idx = set(indices[:train_count])
    train_trajs = [trajectories[i] for i in range(len(trajectories)) if i in train_idx]
    val_trajs = [trajectories[i] for i in range(len(trajectories)) if i not in train_idx]
    if not val_trajs:
        val_trajs = [train_trajs.pop()]
    return train_trajs, val_trajs


def stack_xy(trajectories):
    states = np.concatenate([traj["states"] for traj in trajectories], axis=0)
    actions = np.concatenate([traj["actions"] for traj in trajectories], axis=0)
    return states, actions


def count_by_source(trajectories):
    counts = {}
    for traj in trajectories:
        counts[traj["source"]] = counts.get(traj["source"], 0) + 1
    return counts


class BehaviorCloningDataset(Dataset):
    def __init__(self, states, actions, state_mean, state_std, action_mean, action_std):
        self.states = ((states - state_mean) / state_std).astype(np.float32)
        self.actions = ((actions - action_mean) / action_std).astype(np.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return torch.tensor(self.states[idx], dtype=torch.float32), torch.tensor(
            self.actions[idx], dtype=torch.float32
        )


class BCMLP(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layers = []
        in_dim = state_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, criterion, device, action_std):
    model.eval()
    total_loss = 0.0
    total_count = 0
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for states, actions in loader:
            states = states.to(device)
            actions = actions.to(device)
            preds = model(states)
            loss = criterion(preds, actions)
            batch_size = states.shape[0]
            total_loss += loss.item() * batch_size
            total_count += batch_size
            preds_all.append(preds.cpu().numpy())
            targets_all.append(actions.cpu().numpy())

    avg_loss = total_loss / max(1, total_count)
    preds_np = np.concatenate(preds_all, axis=0)
    targets_np = np.concatenate(targets_all, axis=0)
    mae_norm = float(np.mean(np.abs(preds_np - targets_np)))
    mae_raw = float(np.mean(np.abs((preds_np - targets_np) * action_std)))
    return avg_loss, mae_norm, mae_raw


def plot_losses(train_losses, val_losses, output_path: Path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Behavior Cloning Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_dataset_summary(dataset_summaries):
    print("Loaded datasets:")
    for summary in dataset_summaries:
        print(
            f"  - {summary['path']}: "
            f"traj={summary['num_trajectories']}, "
            f"steps={summary['total_steps']}, "
            f"len[min/mean/max]={summary['traj_len_min']}/"
            f"{summary['traj_len_mean']:.1f}/{summary['traj_len_max']}, "
            f"sources={summary['trajectory_sources']}"
        )


def try_import_robosuite():
    try:
        import robosuite as suite
        from robosuite.controllers import load_controller_config

        return suite, load_controller_config, None
    except Exception as exc:
        return None, None, exc


def make_rollout_env(env_name, env_info, horizon, render):
    suite, load_controller_config, import_error = try_import_robosuite()
    if import_error is not None:
        raise ImportError(f"Rollout evaluation requires robosuite: {import_error}") from import_error

    controller_config = env_info.get("controller_configs")
    if (
        controller_config is None
        or not isinstance(controller_config, dict)
        or "body_parts" in controller_config
        or controller_config.get("type") == "BASIC"
    ):
        controller_config = load_controller_config(default_controller="OSC_POSE")

    env_configuration = env_info.get("env_configuration", "parallel")
    if env_configuration == "parallel":
        env_configuration = "single-arm-parallel"
    elif env_configuration == "opposed":
        env_configuration = "single-arm-opposed"

    config = {
        "env_name": env_name or env_info.get("env_name", "TwoArmLift"),
        "robots": env_info.get("robots", ["Panda", "Panda"]),
        "controller_configs": controller_config,
        "env_configuration": env_configuration,
    }

    make_kwargs = {
        **config,
        "has_renderer": render,
        "has_offscreen_renderer": False,
        "render_camera": "agentview",
        "ignore_done": True,
        "use_camera_obs": False,
        "use_object_obs": True,
        "reward_shaping": True,
        "control_freq": 20,
        "horizon": horizon,
    }
    if render:
        make_kwargs["renderer"] = "nvisii"

    env = suite.make(**make_kwargs)
    return env


def extract_policy_state(env):
    return env.sim.get_state().flatten().astype(np.float32)


def policy_action(model, state, state_mean, state_std, action_mean, action_std, device):
    state_norm = (state - state_mean) / state_std
    state_tensor = torch.tensor(state_norm, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action_norm = model(state_tensor).squeeze(0).cpu().numpy()
    action = action_norm * action_std + action_mean
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def run_rollout_evaluation(
    model,
    checkpoint,
    source_hdf5_path,
    rollout_episodes,
    rollout_max_steps,
    rollout_render,
    device,
):
    env_name, env_info = read_env_metadata(source_hdf5_path)
    env = make_rollout_env(env_name, env_info, rollout_max_steps, rollout_render)

    state_dim = int(checkpoint["state_dim"])
    action_dim = int(checkpoint["action_dim"])
    state_mean = checkpoint["state_mean"].astype(np.float32)
    state_std = checkpoint["state_std"].astype(np.float32)
    action_mean = checkpoint["action_mean"].astype(np.float32)
    action_std = checkpoint["action_std"].astype(np.float32)

    results = []
    success_count = 0

    try:
        for episode_idx in range(rollout_episodes):
            obs = env.reset()
            total_reward = 0.0
            success = False

            for step_idx in range(rollout_max_steps):
                _ = obs
                state = extract_policy_state(env)
                if state.shape[0] != state_dim:
                    raise ValueError(
                        f"Rollout state dim mismatch: env gives {state.shape[0]}, checkpoint expects {state_dim}."
                    )

                action = policy_action(model, state, state_mean, state_std, action_mean, action_std, device)
                if action.shape[0] != action_dim:
                    raise ValueError(
                        f"Rollout action dim mismatch: policy outputs {action.shape[0]}, checkpoint expects {action_dim}."
                    )

                obs, reward, done, info = env.step(action)
                _ = done, info
                total_reward += float(reward)

                if rollout_render:
                    env.render()

                if env._check_success():
                    success = True
                    results.append(
                        {
                            "episode": episode_idx,
                            "success": True,
                            "steps": step_idx + 1,
                            "reward": total_reward,
                        }
                    )
                    success_count += 1
                    break

            if not success:
                results.append(
                    {
                        "episode": episode_idx,
                        "success": False,
                        "steps": rollout_max_steps,
                        "reward": total_reward,
                    }
                )
    finally:
        env.close()

    mean_steps = float(np.mean([item["steps"] for item in results])) if results else None
    mean_reward = float(np.mean([item["reward"] for item in results])) if results else None
    success_rate = float(success_count / rollout_episodes) if rollout_episodes > 0 else None

    return {
        "source_hdf5_path": str(source_hdf5_path),
        "env_name": env_name,
        "episodes": int(rollout_episodes),
        "max_steps": int(rollout_max_steps),
        "success_count": int(success_count),
        "success_rate": success_rate,
        "mean_steps": mean_steps,
        "mean_reward": mean_reward,
        "per_episode": results,
    }


def infer_manifest_paths(data_paths):
    manifest_paths = []
    for path in data_paths:
        candidates = [
            path.with_suffix(".manifest.json"),
            path.parent / f"{path.stem}.manifest.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                manifest_paths.append(str(candidate))
                break
    return manifest_paths


def load_fixed_split(data_paths, trajectories):
    manifest_paths = infer_manifest_paths(data_paths)
    if not manifest_paths:
        return None, None, None

    split_train_ids = []
    split_val_ids = []
    used_manifest_paths = []
    seen_any_split = False

    for manifest_path in manifest_paths:
        with Path(manifest_path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        train_ids = payload.get("train_demo_ids")
        val_ids = payload.get("val_demo_ids")
        if train_ids is None or val_ids is None:
            continue
        seen_any_split = True
        split_train_ids.extend(train_ids)
        split_val_ids.extend(val_ids)
        used_manifest_paths.append(manifest_path)

    if not seen_any_split:
        return None, None, None

    traj_by_key = {(traj["source_path"], traj["name"]): traj for traj in trajectories}
    train_trajs = []
    val_trajs = []

    for data_path in data_paths:
        data_path_str = str(data_path)
        path_train_ids = [demo_id for demo_id in split_train_ids if (data_path_str, demo_id) in traj_by_key]
        path_val_ids = [demo_id for demo_id in split_val_ids if (data_path_str, demo_id) in traj_by_key]
        train_trajs.extend(traj_by_key[(data_path_str, demo_id)] for demo_id in path_train_ids)
        val_trajs.extend(traj_by_key[(data_path_str, demo_id)] for demo_id in path_val_ids)

    if not train_trajs or not val_trajs:
        raise ValueError(
            "Fixed split was found in manifest(s), but could not map it cleanly onto the loaded trajectories."
        )

    split_info = {
        "type": "manifest",
        "manifest_paths": used_manifest_paths,
        "train_demo_ids": [traj["name"] for traj in train_trajs],
        "val_demo_ids": [traj["name"] for traj in val_trajs],
    }
    return train_trajs, val_trajs, split_info


def train(args):
    set_seed(args.seed)
    data_paths = resolve_data_paths(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories, dataset_summaries = load_trajectories(data_paths)
    print_dataset_summary(dataset_summaries)

    train_trajs, val_trajs, split_info = load_fixed_split(data_paths, trajectories)
    if split_info is None:
        train_trajs, val_trajs = split_trajectories(trajectories, args.train_ratio, args.seed)
        split_info = {
            "type": "random",
            "seed": int(args.seed),
            "train_ratio": float(args.train_ratio),
            "train_demo_ids": [traj["name"] for traj in train_trajs],
            "val_demo_ids": [traj["name"] for traj in val_trajs],
        }

    train_states, train_actions = stack_xy(train_trajs)
    val_states, val_actions = stack_xy(val_trajs)

    state_mean = train_states.mean(axis=0).astype(np.float32)
    state_std = (train_states.std(axis=0) + 1e-6).astype(np.float32)
    action_mean = train_actions.mean(axis=0).astype(np.float32)
    action_std = (train_actions.std(axis=0) + 1e-6).astype(np.float32)

    train_dataset = BehaviorCloningDataset(
        train_states, train_actions, state_mean, state_std, action_mean, action_std
    )
    val_dataset = BehaviorCloningDataset(
        val_states, val_actions, state_mean, state_std, action_mean, action_std
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    model = BCMLP(
        state_dim=train_states.shape[1],
        action_dim=train_actions.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_decay, patience=args.lr_patience
    )
    criterion = nn.MSELoss()

    best_val_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_path = output_dir / "bc_best.pt"
    train_losses = []
    val_losses = []

    print(
        f"Train split: traj={len(train_trajs)}, samples={len(train_dataset)}, sources={count_by_source(train_trajs)}"
    )
    print(f"Val split  : traj={len(val_trajs)}, samples={len(val_dataset)}, sources={count_by_source(val_trajs)}")
    print(f"Split mode : {split_info['type']}")
    print(f"Device     : {device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for states, actions in train_loader:
            states = states.to(device)
            actions = actions.to(device)

            optimizer.zero_grad(set_to_none=True)
            preds = model(states)
            loss = criterion(preds, actions)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            batch_size = states.shape[0]
            running_loss += loss.item() * batch_size
            sample_count += batch_size

        train_loss = running_loss / max(1, sample_count)
        val_loss, val_mae_norm, val_mae_raw = evaluate(model, val_loader, criterion, device, action_std)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"lr={current_lr:.2e} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_mae_norm={val_mae_norm:.6f} | "
            f"val_mae_raw={val_mae_raw:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "state_dim": train_states.shape[1],
                    "action_dim": train_actions.shape[1],
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "state_mean": state_mean,
                    "state_std": state_std,
                    "action_mean": action_mean,
                    "action_std": action_std,
                    "data_paths": [str(path) for path in data_paths],
                },
                best_model_path,
            )
        else:
            epochs_without_improvement += 1

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    plot_losses(train_losses, val_losses, output_dir / "loss_curve.png")

    rollout_metrics = None
    if args.rollout_episodes > 0:
        print(f"Starting rollout evaluation for {args.rollout_episodes} episode(s)...")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        try:
            rollout_metrics = run_rollout_evaluation(
                model=model,
                checkpoint=checkpoint,
                source_hdf5_path=data_paths[0],
                rollout_episodes=args.rollout_episodes,
                rollout_max_steps=args.rollout_max_steps,
                rollout_render=args.rollout_render,
                device=device,
            )
            print(
                f"Rollout success_rate={rollout_metrics['success_rate']:.3f} "
                f"({rollout_metrics['success_count']}/{rollout_metrics['episodes']}) | "
                f"mean_steps={rollout_metrics['mean_steps']:.2f} | "
                f"mean_reward={rollout_metrics['mean_reward']:.3f}"
            )
        except Exception as exc:
            rollout_metrics = {
                "requested": int(args.rollout_episodes),
                "status": "failed",
                "error": str(exc),
            }
            print(f"Rollout evaluation skipped/failed: {exc}")

    metrics = {
        "data_paths": [str(path) for path in data_paths],
        "data_manifest_paths": infer_manifest_paths(data_paths),
        "dataset_summaries": dataset_summaries,
        "num_trajectories": len(trajectories),
        "train_trajectories": len(train_trajs),
        "val_trajectories": len(val_trajs),
        "train_samples": int(len(train_dataset)),
        "val_samples": int(len(val_dataset)),
        "train_sources": count_by_source(train_trajs),
        "val_sources": count_by_source(val_trajs),
        "state_dim": int(train_states.shape[1]),
        "action_dim": int(train_actions.shape[1]),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "device": str(device),
        "split_info": split_info,
        "rollout_metrics": rollout_metrics,
        "config": vars(args),
        "train_manifest": [
            {
                "name": traj["name"],
                "source": traj["source"],
                "source_demo": traj["source_demo"],
                "source_path": traj["source_path"],
            }
            for traj in train_trajs
        ],
        "val_manifest": [
            {
                "name": traj["name"],
                "source": traj["source"],
                "source_demo": traj["source_demo"],
                "source_path": traj["source_path"],
            }
            for traj in val_trajs
        ],
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved best model to: {best_model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved plot to: {output_dir / 'loss_curve.png'}")
    return metrics


def build_parser():
    parser = argparse.ArgumentParser(description="Behavior Cloning trainer for mixed dual-arm demonstration datasets.")
    parser.add_argument(
        "--data",
        type=str,
        nargs="+",
        required=True,
        help="One or more HDF5 dataset paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for model checkpoints and training artifacts.",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lr-decay", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=8)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument(
        "--rollout-episodes",
        type=int,
        default=20,
        help="If > 0, run environment rollouts after training and report success rate.",
    )
    parser.add_argument("--rollout-max-steps", type=int, default=600)
    parser.add_argument("--rollout-render", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
