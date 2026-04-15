import argparse
import json
import math
import random
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATA_ROOT = Path(r"D:\a_6019\demonstrations")
PRESET_DIR_KEYWORDS = {
    "human": ("human",),
    "scripted": ("script",),
    "noisy": ("noisy",),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def discover_hdf5_files(root: Path):
    return sorted(root.rglob("*.hdf5"))


def score_dataset_path(path: Path, preset: str):
    path_text = str(path).lower()
    name_text = path.name.lower()
    parent_text = path.parent.name.lower()

    score = 0
    if preset == "human":
        if "human" in path_text:
            score += 5
        if "demo1" in name_text:
            score += 2
    elif preset == "scripted":
        if "script" in path_text:
            score += 3
        if "noisy" not in path_text:
            score += 4
        if "scripted" in name_text or "demo2" in name_text:
            score += 2
    elif preset == "noisy":
        if "noisy" in path_text:
            score += 6
        if "demo3" in name_text:
            score += 2

    if preset in parent_text:
        score += 1
    return score


def resolve_preset_paths(preset: str):
    all_hdf5 = discover_hdf5_files(DEFAULT_DATA_ROOT)
    if not all_hdf5:
        raise FileNotFoundError(f"No .hdf5 files found under {DEFAULT_DATA_ROOT}")

    if preset == "all":
        return all_hdf5

    candidates = [path for path in all_hdf5 if score_dataset_path(path, preset) > 0]
    if not candidates:
        raise FileNotFoundError(f"Could not find any .hdf5 files matching preset '{preset}' under {DEFAULT_DATA_ROOT}")

    candidates.sort(key=lambda p: (score_dataset_path(p, preset), p.stat().st_mtime), reverse=True)
    best_score = score_dataset_path(candidates[0], preset)
    best_matches = [path for path in candidates if score_dataset_path(path, preset) == best_score]

    if preset in {"human", "scripted", "noisy"}:
        return [best_matches[0]]
    return best_matches


def resolve_data_paths(args):
    paths = []
    if args.preset:
        paths.extend(resolve_preset_paths(args.preset))
    if args.data:
        paths.extend(Path(p).expanduser().resolve() for p in args.data)
    if not paths:
        paths = resolve_preset_paths("all")

    unique_paths = []
    seen = set()
    for path in paths:
        resolved = Path(path).expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(resolved)

    missing = [str(path) for path in unique_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset file(s): {missing}")
    return unique_paths


def read_env_metadata(hdf5_path: Path):
    with h5py.File(hdf5_path, "r") as f:
        attrs = f["data"].attrs
        env_name = attrs.get("env", "TwoArmLift")
        env_info_raw = attrs.get("env_info")
        env_info = json.loads(env_info_raw) if env_info_raw else {}
    return env_name, env_info


def load_trajectories(hdf5_paths):
    trajectories = []
    dataset_summaries = []

    for hdf5_path in hdf5_paths:
        lengths = []
        with h5py.File(hdf5_path, "r") as f:
            demo_names = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
            for demo_name in demo_names:
                group = f["data"][demo_name]
                states = group["states"][:].astype(np.float32)
                actions = group["actions"][:].astype(np.float32)
                if len(states) != len(actions):
                    raise ValueError(
                        f"{hdf5_path.name}/{demo_name} has mismatched lengths: {len(states)} vs {len(actions)}"
                    )
                trajectories.append(
                    {
                        "name": demo_name,
                        "source": hdf5_path.stem,
                        "source_path": str(hdf5_path),
                        "states": states,
                        "actions": actions,
                    }
                )
                lengths.append(len(states))

        dataset_summaries.append(
            {
                "path": str(hdf5_path),
                "source": hdf5_path.stem,
                "num_trajectories": len(lengths),
                "total_steps": int(sum(lengths)),
                "traj_len_min": int(min(lengths)),
                "traj_len_mean": float(np.mean(lengths)),
                "traj_len_max": int(max(lengths)),
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
            f"  - {summary['source']}: "
            f"traj={summary['num_trajectories']}, "
            f"steps={summary['total_steps']}, "
            f"len[min/mean/max]={summary['traj_len_min']}/"
            f"{summary['traj_len_mean']:.1f}/{summary['traj_len_max']}"
        )


def try_import_robosuite():
    try:
        import robosuite as suite
        from robosuite.controllers import load_composite_controller_config

        return suite, load_composite_controller_config, None
    except Exception as exc:
        return None, None, exc


def make_rollout_env(env_name, env_info, horizon, render):
    suite, load_composite_controller_config, import_error = try_import_robosuite()
    if import_error is not None:
        raise ImportError(
            "Rollout evaluation requires robosuite. Install it in your PyCharm interpreter first."
        ) from import_error

    controller_config = env_info.get("controller_configs")
    if controller_config is None:
        controller_config = load_composite_controller_config(controller=None, robot="Panda")

    config = {
        "env_name": env_name or env_info.get("env_name", "TwoArmLift"),
        "robots": env_info.get("robots", ["Panda", "Panda"]),
        "controller_configs": controller_config,
        "env_configuration": env_info.get("env_configuration", "parallel"),
    }

    env = suite.make(
        **config,
        has_renderer=render,
        renderer="mjviewer",
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=20,
        horizon=horizon,
    )
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
                _ = obs  # Keeps the reset / step API explicit even though policy uses simulator state.
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

    rollout_metrics = {
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
    return rollout_metrics


def train(args):
    set_seed(args.seed)
    data_paths = resolve_data_paths(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories, dataset_summaries = load_trajectories(data_paths)
    print_dataset_summary(dataset_summaries)

    train_trajs, val_trajs = split_trajectories(trajectories, args.train_ratio, args.seed)
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
                    "preset": args.preset,
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
        checkpoint = torch.load(best_model_path, map_location=device)
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
        "preset": args.preset,
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
        "rollout_metrics": rollout_metrics,
        "config": vars(args),
        "train_manifest": [
            {"name": traj["name"], "source": traj["source"], "source_path": traj["source_path"]}
            for traj in train_trajs
        ],
        "val_manifest": [
            {"name": traj["name"], "source": traj["source"], "source_path": traj["source_path"]}
            for traj in val_trajs
        ],
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved best model to: {best_model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved plot to: {output_dir / 'loss_curve.png'}")


def build_parser():
    parser = argparse.ArgumentParser(description="Behavior Cloning trainer for dual-arm demonstration datasets.")
    parser.add_argument(
        "--data",
        type=str,
        nargs="+",
        default=None,
        help="One or more HDF5 dataset paths. Can be combined with --preset.",
    )
    parser.add_argument(
        "--preset",
        choices=["all", "human", "noisy", "scripted"],
        default="all",
        help="Use a built-in dataset preset under D:\\a_6019\\demonstrations\\demonstrations.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs_bc",
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument(
        "--rollout-episodes",
        type=int,
        default=0,
        help="If > 0, run environment rollouts after training and report success rate.",
    )
    parser.add_argument("--rollout-max-steps", type=int, default=600)
    parser.add_argument("--rollout-render", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
