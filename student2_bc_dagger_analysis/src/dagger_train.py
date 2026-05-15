import argparse
import json
import math
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
    extract_policy_state,
    infer_manifest_paths,
    load_trajectories,
    make_rollout_env,
    plot_losses,
    policy_action,
    print_dataset_summary,
    read_env_metadata,
    resolve_data_paths,
    run_rollout_evaluation,
    set_seed,
    split_trajectories,
    stack_xy,
)
from dual_arm_expert import TwoArmLiftScriptedExpert


SUPPORTED_ENV_NAME = "TwoArmLift"
SUPPORTED_ROBOTS = ["Panda", "Panda"]


def build_model(state_dim, action_dim, hidden_dim, num_layers, dropout, device):
    return BCMLP(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)


def load_policy(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(
        state_dim=int(checkpoint["state_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


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
        if env_name != SUPPORTED_ENV_NAME:
            raise ValueError(
                f"DAgger currently supports only {SUPPORTED_ENV_NAME}, but got {env_name} from {metadata['path']}."
            )
        if list(robots) != SUPPORTED_ROBOTS:
            raise ValueError(
                f"DAgger currently supports only robots={SUPPORTED_ROBOTS}, but got {robots} from {metadata['path']}."
            )
    return env_metadatas


def manifest_path_for_hdf5(hdf5_path: Path):
    candidates = [
        hdf5_path.with_suffix(".manifest.json"),
        hdf5_path.parent / f"{hdf5_path.stem}.manifest.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_training_split(trajectories, base_data_paths, train_ratio: float, seed: int):
    base_path_set = {str(path) for path in base_data_paths}
    base_trajs = [traj for traj in trajectories if traj["source_path"] in base_path_set]
    extra_trajs = [traj for traj in trajectories if traj["source_path"] not in base_path_set]

    manifest_map = {}
    for path in base_data_paths:
        manifest_path = manifest_path_for_hdf5(path)
        if manifest_path is not None:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest_map[str(path)] = json.load(f)

    have_fixed_split = bool(manifest_map) and all(
        payload.get("train_demo_ids") is not None and payload.get("val_demo_ids") is not None
        for payload in manifest_map.values()
    )

    if have_fixed_split:
        base_traj_by_key = {(traj["source_path"], traj["name"]): traj for traj in base_trajs}
        train_trajs = []
        val_trajs = []
        for path in base_data_paths:
            path_str = str(path)
            payload = manifest_map.get(path_str)
            if payload is None:
                continue
            train_trajs.extend(base_traj_by_key[(path_str, demo_id)] for demo_id in payload["train_demo_ids"])
            val_trajs.extend(base_traj_by_key[(path_str, demo_id)] for demo_id in payload["val_demo_ids"])
        split_info = {
            "type": "manifest_base_plus_dagger_train_only",
            "manifest_paths": [str(manifest_path_for_hdf5(path)) for path in base_data_paths],
            "train_ratio": float(train_ratio),
            "seed": int(seed),
        }
    else:
        train_trajs, val_trajs = split_trajectories(base_trajs, train_ratio, seed)
        split_info = {
            "type": "random_base_plus_dagger_train_only",
            "train_ratio": float(train_ratio),
            "seed": int(seed),
            "manifest_paths": [],
        }

    train_trajs = list(train_trajs) + list(extra_trajs)
    split_info["base_train_trajectories"] = int(len(train_trajs) - len(extra_trajs))
    split_info["base_val_trajectories"] = int(len(val_trajs))
    split_info["extra_train_only_trajectories"] = int(len(extra_trajs))
    split_info["extra_train_only_paths"] = sorted({traj["source_path"] for traj in extra_trajs})
    split_info["train_demo_ids"] = [traj["name"] for traj in train_trajs]
    split_info["val_demo_ids"] = [traj["name"] for traj in val_trajs]
    return train_trajs, val_trajs, split_info


def train_bc_once(args, data_paths, base_data_paths, output_dir, checkpoint_name):
    trajectories, dataset_summaries = load_trajectories(data_paths)
    print_dataset_summary(dataset_summaries)

    train_trajs, val_trajs, split_info = resolve_training_split(
        trajectories=trajectories,
        base_data_paths=base_data_paths,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
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

    plot_losses(train_losses, val_losses, output_dir / "loss_curve.png")

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

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
        "config": vars(args),
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved best model to: {best_model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved plot to: {output_dir / 'loss_curve.png'}")
    return model, checkpoint, metrics


def compute_beta(args, round_idx):
    if args.execution_policy == "student":
        return 0.0
    return max(args.beta_min, args.beta_start * (args.beta_decay ** round_idx))


def collect_dagger_round(args, model, checkpoint, source_hdf5_path, round_idx, output_dir):
    env_name, env_info = read_env_metadata(source_hdf5_path)
    env = make_rollout_env(
        env_name=env_name,
        env_info=env_info,
        horizon=args.rollout_max_steps,
        render=args.rollout_render,
    )
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
        data_grp.attrs["requested_episodes"] = int(args.dagger_episodes)
        data_grp.attrs["collection_type"] = "dagger"
        data_grp.attrs["beta"] = float(beta)
        data_grp.attrs["keep_failed_episodes"] = bool(args.keep_failed_episodes)

        max_attempts = max(args.dagger_episodes, args.dagger_episodes * args.max_attempt_multiplier)
        while attempted_episodes < max_attempts and saved_episodes < args.dagger_episodes:
            episode_idx = attempted_episodes
            attempted_episodes += 1
            obs = env.reset()
            expert.reset()

            traj_states = []
            traj_actions = []
            traj_expert_actions = []
            traj_student_actions = []
            traj_executed_actions = []
            traj_executed_by = []
            reward_sum = 0.0
            success = False
            executed_by = "student"

            for step_idx in range(args.rollout_max_steps):
                state = extract_policy_state(env)
                expert_action = expert.act(obs).astype(np.float32)
                student_action = policy_action(
                    model,
                    state,
                    checkpoint["state_mean"].astype(np.float32),
                    checkpoint["state_std"].astype(np.float32),
                    checkpoint["action_mean"].astype(np.float32),
                    checkpoint["action_std"].astype(np.float32),
                    device,
                )

                if np.random.rand() < beta:
                    exec_action = expert_action
                    executed_by = "expert"
                else:
                    exec_action = student_action
                    executed_by = "student"

                traj_states.append(state)
                traj_actions.append(expert_action)
                traj_expert_actions.append(expert_action)
                traj_student_actions.append(student_action)
                traj_executed_actions.append(exec_action)
                traj_executed_by.append(executed_by)

                obs, reward, done, info = env.step(exec_action)
                _ = done, info
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
                            "beta": float(beta),
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
                        "beta": float(beta),
                        "last_executor": executed_by,
                    }
                )

            should_save = success or args.keep_failed_episodes
            if should_save:
                saved_episodes += 1
                demo_name = f"demo_{saved_episodes}"
                demo_grp = data_grp.create_group(demo_name)
                demo_grp.create_dataset("states", data=np.asarray(traj_states, dtype=np.float32))
                # `actions` is the supervised DAgger label and intentionally stores expert actions.
                demo_grp.create_dataset("actions", data=np.asarray(traj_actions, dtype=np.float32))
                demo_grp.create_dataset("expert_actions", data=np.asarray(traj_expert_actions, dtype=np.float32))
                demo_grp.create_dataset("student_actions", data=np.asarray(traj_student_actions, dtype=np.float32))
                demo_grp.create_dataset("executed_actions", data=np.asarray(traj_executed_actions, dtype=np.float32))
                demo_grp.create_dataset(
                    "executed_by",
                    data=np.asarray(traj_executed_by, dtype=object),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
                demo_grp.attrs["success"] = bool(success)
                demo_grp.attrs["beta"] = float(beta)
                demo_grp.attrs["source"] = f"dagger_round_{round_idx + 1}"
                demo_grp.attrs["source_demo"] = f"attempt_{episode_idx + 1}"
                demo_grp.attrs["action_label"] = "expert_action"
                total_saved_steps += len(traj_states)

            print(
                f"[DAgger Round {round_idx + 1} | Episode {episode_idx + 1:03d}] "
                f"{'SUCCESS' if success else 'FAIL'} | "
                f"steps={per_episode[-1]['steps']} | beta={beta:.3f} | "
                f"{'saved' if should_save else 'discarded'}"
            )

        data_grp.attrs["total"] = int(saved_episodes)
        data_grp.attrs["attempted_episodes"] = int(attempted_episodes)
        data_grp.attrs["saved_episodes"] = int(saved_episodes)
        data_grp.attrs["success_count"] = int(success_count)
        data_grp.attrs["action_label_dataset"] = "actions"
        data_grp.attrs["actions_are"] = "expert_actions"

    env.close()

    return hdf5_path, {
        "round_idx": round_idx + 1,
        "hdf5_path": str(hdf5_path),
        "requested_episodes": int(args.dagger_episodes),
        "saved_episodes": int(saved_episodes),
        "attempted_episodes": int(attempted_episodes),
        "success_count": int(success_count),
        "success_rate": float(success_count / attempted_episodes) if attempted_episodes > 0 else 0.0,
        "total_saved_steps": int(total_saved_steps),
        "beta": float(beta),
        "keep_failed_episodes": bool(args.keep_failed_episodes),
        "per_episode": per_episode,
    }


def run_policy_rollout_eval(args, model, checkpoint, source_hdf5_path):
    device = next(model.parameters()).device
    return run_rollout_evaluation(
        model=model,
        checkpoint=checkpoint,
        source_hdf5_path=source_hdf5_path,
        rollout_episodes=args.rollout_episodes,
        rollout_max_steps=args.rollout_max_steps,
        rollout_render=args.rollout_render,
        device=device,
    )


def load_initial_bc_summary(init_metrics_path: str | None):
    if not init_metrics_path:
        return None
    path = Path(init_metrics_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing init metrics: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_experiment(args):
    set_seed(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_data_paths = resolve_data_paths(args)
    env_metadatas = validate_supported_env_metadata(base_data_paths)
    aggregate_data_paths = list(base_data_paths)
    source_hdf5_path = base_data_paths[0]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")

    print("Validated environment metadata for initial datasets:")
    for metadata in env_metadatas:
        print(f"  - {metadata['path']} -> {metadata['env_name']}")

    if args.init_checkpoint:
        print("=== Loading Initial BC Checkpoint ===")
        model, checkpoint = load_policy(Path(args.init_checkpoint).expanduser().resolve(), device=device)
        initial_bc = load_initial_bc_summary(args.init_metrics)
        if initial_bc is None:
            initial_bc = {
                "checkpoint_path": str(Path(args.init_checkpoint).expanduser().resolve()),
                "loaded_from_existing_bc": True,
            }
    else:
        print("=== Initial BC Training ===")
        model, checkpoint, initial_bc = train_bc_once(
            args=args,
            data_paths=aggregate_data_paths,
            base_data_paths=base_data_paths,
            output_dir=output_dir / "initial_bc",
            checkpoint_name="bc_init_best.pt",
        )

    initial_eval = run_policy_rollout_eval(args, model, checkpoint, source_hdf5_path)
    if initial_eval["success_rate"] is None:
        print("Initial policy rollout skipped because rollout_episodes=0.")
    else:
        print(
            f"Initial policy rollout success_rate={initial_eval['success_rate']:.3f} "
            f"({initial_eval['success_count']}/{initial_eval['episodes']})"
        )

    all_round_metrics = []
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
                "skip_reason": "No DAgger episodes were saved in this round.",
            }
            all_round_metrics.append(round_metrics)
            print(f"Round {round_idx + 1} skipped retraining because no DAgger episodes were saved.")
            continue

        aggregate_data_paths.append(dagger_hdf5_path)
        round_train_dir = output_dir / f"round_{round_idx + 1}_train"
        round_train_dir.mkdir(parents=True, exist_ok=True)
        model, checkpoint, train_metrics = train_bc_once(
            args=args,
            data_paths=aggregate_data_paths,
            base_data_paths=base_data_paths,
            output_dir=round_train_dir,
            checkpoint_name=f"dagger_round_{round_idx + 1}_best.pt",
        )
        eval_metrics = run_policy_rollout_eval(args, model, checkpoint, source_hdf5_path)

        round_metrics = {
            "round_idx": round_idx + 1,
            "aggregate_data_paths": [str(path) for path in aggregate_data_paths],
            "collection": collection_metrics,
            "training": train_metrics,
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
        "initial_bc": initial_bc,
        "initial_eval": initial_eval,
        "rounds": all_round_metrics,
        "config": vars(args),
    }
    metrics_path = output_dir / "dagger_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Saved DAgger metrics to: {metrics_path}")
    return final_metrics


def build_parser():
    parser = argparse.ArgumentParser(description="Vanilla DAgger built on top of the Prompt A dual-arm BC pipeline.")
    parser.add_argument("--data", type=str, nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--init-checkpoint", type=str, default=None)
    parser.add_argument("--init-metrics", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
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
    parser.add_argument("--dagger-rounds", type=int, default=5)
    parser.add_argument("--dagger-episodes", type=int, default=30)
    parser.add_argument("--beta-start", type=float, default=1.0)
    parser.add_argument("--beta-decay", type=float, default=0.8)
    parser.add_argument("--beta-min", type=float, default=0.2)
    parser.add_argument("--execution-policy", choices=["mixed", "student"], default="mixed")
    parser.add_argument(
        "--keep-failed-episodes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save failed DAgger rollouts too; `actions` remains expert-labeled for every visited state.",
    )
    parser.add_argument("--max-attempt-multiplier", type=int, default=3)
    parser.add_argument("--rollout-episodes", type=int, default=50)
    parser.add_argument("--rollout-max-steps", type=int, default=600)
    parser.add_argument("--rollout-render", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
