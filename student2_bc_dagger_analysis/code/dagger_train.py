import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from bc_train import (
    BCMLP,
    BehaviorCloningDataset,
    count_by_source,
    evaluate,
    load_trajectories,
    plot_losses,
    print_dataset_summary,
    read_env_metadata,
    resolve_data_paths,
    set_seed,
    split_trajectories,
    stack_xy,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR_CANDIDATES = [
    SCRIPT_DIR,
    Path(r"D:\a_6019"),
]
for candidate in PROJECT_DIR_CANDIDATES:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from collect_scripted_demos import TwoArmLiftScriptedExpert, make_env

SUPPORTED_ENV_NAME = "TwoArmLift"
SUPPORTED_ROBOTS = ["Panda", "Panda"]
SUPPORTED_ENV_CONFIGURATION = "parallel"


def build_model(state_dim, action_dim, hidden_dim, num_layers, dropout, device):
    model = BCMLP(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    return model


def train_bc_once(args, data_paths, output_dir, checkpoint_name):
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
    model = build_model(
        state_dim=train_states.shape[1],
        action_dim=train_actions.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_decay, patience=args.lr_patience
    )
    criterion = nn.MSELoss()

    best_val_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_path = output_dir / checkpoint_name
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
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)

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

    plot_losses(train_losses, val_losses, output_dir / f"{best_model_path.stem}_loss_curve.png")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    summary = {
        "dataset_summaries": dataset_summaries,
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "device": str(device),
        "checkpoint_path": str(best_model_path),
        "train_sources": count_by_source(train_trajs),
        "val_sources": count_by_source(val_trajs),
    }
    return model, checkpoint, summary


def policy_action(model, checkpoint, state, device):
    state_mean = checkpoint["state_mean"].astype(np.float32)
    state_std = checkpoint["state_std"].astype(np.float32)
    action_mean = checkpoint["action_mean"].astype(np.float32)
    action_std = checkpoint["action_std"].astype(np.float32)

    state_norm = (state - state_mean) / state_std
    state_tensor = torch.tensor(state_norm, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action_norm = model(state_tensor).squeeze(0).cpu().numpy()
    action = action_norm * action_std + action_mean
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def validate_supported_env_metadata(hdf5_paths):
    env_metadatas = []
    for path in hdf5_paths:
        env_name, env_info = read_env_metadata(path)
        env_metadatas.append(
            {
                "path": str(path),
                "env_name": env_name,
                "env_info": env_info,
            }
        )

    for metadata in env_metadatas:
        env_name = metadata["env_name"]
        env_info = metadata["env_info"]
        robots = env_info.get("robots", SUPPORTED_ROBOTS)
        env_configuration = env_info.get("env_configuration", SUPPORTED_ENV_CONFIGURATION)

        if env_name != SUPPORTED_ENV_NAME:
            raise ValueError(
                f"DAgger currently supports only {SUPPORTED_ENV_NAME}, but got {env_name} from {metadata['path']}."
            )
        if list(robots) != SUPPORTED_ROBOTS:
            raise ValueError(
                f"DAgger currently supports only robots={SUPPORTED_ROBOTS}, but got {robots} from {metadata['path']}."
            )
        if env_configuration != SUPPORTED_ENV_CONFIGURATION:
            raise ValueError(
                "DAgger currently supports only env_configuration="
                f"{SUPPORTED_ENV_CONFIGURATION}, but got {env_configuration} from {metadata['path']}."
            )

    return env_metadatas


def build_project_env(args):
    env, config = make_env(
        has_renderer=args.rollout_render,
        renderer="mjviewer",
        control_freq=20,
        horizon=args.rollout_max_steps,
    )
    return env, config


def compute_beta(args, round_idx):
    if args.execution_policy == "student":
        return 0.0
    return max(args.beta_min, args.beta_start * (args.beta_decay ** round_idx))


def collect_dagger_round(args, model, checkpoint, source_hdf5_path, round_idx, output_dir):
    env_name, env_info = read_env_metadata(source_hdf5_path)
    env, runtime_config = build_project_env(args)
    expert = TwoArmLiftScriptedExpert(env)
    device = next(model.parameters()).device

    beta = compute_beta(args, round_idx)
    round_dir = output_dir / f"dagger_round_{round_idx + 1}"
    round_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = round_dir / f"dagger_round_{round_idx + 1}.hdf5"

    per_episode = []
    success_count = 0
    total_saved_steps = 0
    saved_episodes = 0
    attempted_episodes = 0

    with h5py.File(hdf5_path, "w") as f:
        data_grp = f.create_group("data")
        data_grp.attrs["env"] = env_name
        data_grp.attrs["env_info"] = json.dumps(env_info)
        data_grp.attrs["runtime_env_info"] = json.dumps(runtime_config)
        data_grp.attrs["requested_episodes"] = args.dagger_episodes
        data_grp.attrs["collection_type"] = "dagger"
        data_grp.attrs["beta"] = beta
        data_grp.attrs["keep_failed_episodes"] = bool(args.keep_failed_episodes)

        max_attempts = max(args.dagger_episodes, args.dagger_episodes * args.max_attempt_multiplier)
        while attempted_episodes < max_attempts and saved_episodes < args.dagger_episodes:
            episode_idx = attempted_episodes
            attempted_episodes += 1
            obs = env.reset()
            expert.reset()

            traj_states = []
            traj_actions = []
            reward_sum = 0.0
            success = False

            for step_idx in range(args.rollout_max_steps):
                state = env.sim.get_state().flatten().astype(np.float32)
                expert_action = expert.act(obs).astype(np.float32)
                student_action = policy_action(model, checkpoint, state, device)

                if np.random.rand() < beta:
                    exec_action = expert_action
                    executed_by = "expert"
                else:
                    exec_action = student_action
                    executed_by = "student"

                traj_states.append(state)
                traj_actions.append(expert_action)

                obs, reward, done, info = env.step(exec_action)
                reward_sum += float(reward)

                if env._check_success():
                    success = True
                    success_count += 1
                    per_episode.append(
                        {
                            "episode": episode_idx,
                            "success": True,
                            "steps": step_idx + 1,
                            "reward": reward_sum,
                            "beta": beta,
                            "last_executor": executed_by,
                        }
                    )
                    break

            if not success:
                per_episode.append(
                    {
                        "episode": episode_idx,
                        "success": False,
                        "steps": args.rollout_max_steps,
                        "reward": reward_sum,
                        "beta": beta,
                        "last_executor": executed_by,
                    }
                )

            should_save = success or args.keep_failed_episodes
            if should_save:
                saved_episodes += 1
                demo_grp = data_grp.create_group(f"demo_{saved_episodes}")
                demo_grp.create_dataset("states", data=np.asarray(traj_states, dtype=np.float32))
                demo_grp.create_dataset("actions", data=np.asarray(traj_actions, dtype=np.float32))
                demo_grp.attrs["success"] = bool(success)
                demo_grp.attrs["beta"] = beta
                total_saved_steps += len(traj_states)

            print(
                f"[DAgger Round {round_idx + 1} | Episode {episode_idx + 1:03d}] "
                f"{'SUCCESS' if success else 'FAIL'} | "
                f"steps={per_episode[-1]['steps']} | beta={beta:.3f} | "
                f"{'saved' if should_save else 'discarded'}"
            )

        data_grp.attrs["total"] = saved_episodes
        data_grp.attrs["attempted_episodes"] = attempted_episodes

    env.close()

    round_metrics = {
        "round_idx": round_idx + 1,
        "hdf5_path": str(hdf5_path),
        "requested_episodes": args.dagger_episodes,
        "saved_episodes": saved_episodes,
        "attempted_episodes": attempted_episodes,
        "success_count": success_count,
        "success_rate": float(success_count / attempted_episodes) if attempted_episodes > 0 else 0.0,
        "total_saved_steps": total_saved_steps,
        "beta": beta,
        "keep_failed_episodes": bool(args.keep_failed_episodes),
        "per_episode": per_episode,
    }
    return hdf5_path, round_metrics


def run_policy_rollout_eval(args, model, checkpoint, source_hdf5_path):
    _ = source_hdf5_path
    env, _ = build_project_env(args)
    device = next(model.parameters()).device

    results = []
    success_count = 0

    try:
        for episode_idx in range(args.eval_episodes):
            obs = env.reset()
            reward_sum = 0.0
            success = False

            for step_idx in range(args.rollout_max_steps):
                _ = obs
                state = env.sim.get_state().flatten().astype(np.float32)
                action = policy_action(model, checkpoint, state, device)
                obs, reward, done, info = env.step(action)
                reward_sum += float(reward)

                if env._check_success():
                    success = True
                    success_count += 1
                    results.append(
                        {
                            "episode": episode_idx,
                            "success": True,
                            "steps": step_idx + 1,
                            "reward": reward_sum,
                        }
                    )
                    break

            if not success:
                results.append(
                    {
                        "episode": episode_idx,
                        "success": False,
                        "steps": args.rollout_max_steps,
                        "reward": reward_sum,
                    }
                )
    finally:
        env.close()

    return {
        "episodes": args.eval_episodes,
        "success_count": success_count,
        "success_rate": float(success_count / args.eval_episodes),
        "mean_steps": float(np.mean([item["steps"] for item in results])),
        "mean_reward": float(np.mean([item["reward"] for item in results])),
        "per_episode": results,
    }


def save_json(path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_parser():
    parser = argparse.ArgumentParser(description="DAgger training loop built on top of the BC trainer.")
    parser.add_argument(
        "--data",
        type=str,
        nargs="+",
        default=None,
        help="One or more initial HDF5 dataset paths. Can be combined with --preset.",
    )
    parser.add_argument(
        "--preset",
        choices=["all", "human", "noisy", "scripted"],
        default="scripted",
        help="Initial dataset preset for the seed BC model.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs_dagger",
        help="Directory for checkpoints, DAgger HDF5 files, and metrics.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lr-decay", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=6)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--dagger-rounds", type=int, default=3)
    parser.add_argument("--dagger-episodes", type=int, default=10)
    parser.add_argument("--beta-start", type=float, default=0.7)
    parser.add_argument("--beta-decay", type=float, default=0.5)
    parser.add_argument("--beta-min", type=float, default=0.1)
    parser.add_argument(
        "--execution-policy",
        choices=["mixed", "student"],
        default="mixed",
        help="Use mixed expert-student execution or pure student execution during DAgger collection.",
    )
    parser.add_argument(
        "--keep-failed-episodes",
        action="store_true",
        help="If set, failed episodes are also written into the aggregated DAgger dataset.",
    )
    parser.add_argument(
        "--max-attempt-multiplier",
        type=int,
        default=3,
        help="Maximum collection attempts is dagger_episodes * this multiplier when failed episodes are filtered out.",
    )
    parser.add_argument("--rollout-max-steps", type=int, default=600)
    parser.add_argument("--rollout-render", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=5)
    return parser


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_data_paths = resolve_data_paths(args)
    env_metadatas = validate_supported_env_metadata(base_data_paths)
    aggregate_data_paths = list(base_data_paths)
    source_hdf5_path = base_data_paths[0]

    print("Validated environment metadata for initial datasets:")
    for metadata in env_metadatas:
        print(f"  - {metadata['path']} -> {metadata['env_name']}")

    print("=== Initial BC Training ===")
    model, checkpoint, bc_summary = train_bc_once(
        args=args,
        data_paths=aggregate_data_paths,
        output_dir=output_dir,
        checkpoint_name="bc_init_best.pt",
    )

    all_round_metrics = []
    initial_eval = run_policy_rollout_eval(args, model, checkpoint, source_hdf5_path)
    print(
        f"Initial policy rollout success_rate={initial_eval['success_rate']:.3f} "
        f"({initial_eval['success_count']}/{initial_eval['episodes']})"
    )

    for round_idx in range(args.dagger_rounds):
        print(f"\n=== DAgger Round {round_idx + 1}/{args.dagger_rounds} ===")
        dagger_hdf5_path, collection_metrics = collect_dagger_round(
            args=args,
            model=model,
            checkpoint=checkpoint,
            source_hdf5_path=source_hdf5_path,
            round_idx=round_idx,
            output_dir=output_dir,
        )
        if collection_metrics["saved_episodes"] == 0:
            round_metrics = {
                "round_idx": round_idx + 1,
                "aggregate_data_paths": [str(path) for path in aggregate_data_paths],
                "collection": collection_metrics,
                "training": None,
                "eval": None,
                "skipped": True,
                "skip_reason": "No successful DAgger episodes were saved in this round.",
            }
            all_round_metrics.append(round_metrics)
            print(
                f"Round {round_idx + 1} skipped retraining because no DAgger episodes were saved."
            )
            continue

        aggregate_data_paths.append(dagger_hdf5_path)

        round_train_dir = output_dir / f"round_{round_idx + 1}_train"
        round_train_dir.mkdir(parents=True, exist_ok=True)
        model, checkpoint, train_summary = train_bc_once(
            args=args,
            data_paths=aggregate_data_paths,
            output_dir=round_train_dir,
            checkpoint_name=f"dagger_round_{round_idx + 1}_best.pt",
        )

        eval_metrics = run_policy_rollout_eval(args, model, checkpoint, source_hdf5_path)
        round_metrics = {
            "round_idx": round_idx + 1,
            "aggregate_data_paths": [str(path) for path in aggregate_data_paths],
            "collection": collection_metrics,
            "training": train_summary,
            "eval": eval_metrics,
            "skipped": False,
        }
        all_round_metrics.append(round_metrics)

        print(
            f"Round {round_idx + 1} eval success_rate={eval_metrics['success_rate']:.3f} "
            f"({eval_metrics['success_count']}/{eval_metrics['episodes']})"
        )

    final_metrics = {
        "base_data_paths": [str(path) for path in base_data_paths],
        "validated_envs": env_metadatas,
        "initial_bc": bc_summary,
        "initial_eval": initial_eval,
        "rounds": all_round_metrics,
        "config": vars(args),
    }
    save_json(output_dir / "dagger_metrics.json", final_metrics)
    print(f"Saved DAgger metrics to: {output_dir / 'dagger_metrics.json'}")


if __name__ == "__main__":
    main()
