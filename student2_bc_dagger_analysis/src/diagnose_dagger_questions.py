import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from bc_train import extract_policy_state, make_rollout_env, read_env_metadata, run_rollout_evaluation, set_seed
from dagger_train import load_policy, run_policy_rollout_eval
from dual_arm_expert import TwoArmLiftScriptedExpert


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_expert_rollout(source_hdf5_path: Path, episodes: int, max_steps: int, render: bool, seed: int):
    set_seed(seed)
    env_name, env_info = read_env_metadata(source_hdf5_path)
    env = make_rollout_env(env_name, env_info, max_steps, render)
    results = []
    success_count = 0
    try:
        for episode_idx in range(episodes):
            obs = env.reset()
            expert = TwoArmLiftScriptedExpert(env)
            expert.reset()
            reward_sum = 0.0
            success = False
            for step_idx in range(max_steps):
                action = expert.act(obs).astype(np.float32)
                obs, reward, done, info = env.step(action)
                _ = done, info
                reward_sum += float(reward)
                if render:
                    env.render()
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
                        "steps": max_steps,
                        "reward": reward_sum,
                    }
                )
    finally:
        env.close()
    return {
        "source_hdf5_path": str(source_hdf5_path),
        "env_name": env_name,
        "episodes": int(episodes),
        "max_steps": int(max_steps),
        "success_count": int(success_count),
        "success_rate": float(success_count / episodes) if episodes else None,
        "mean_steps": float(np.mean([item["steps"] for item in results])) if results else None,
        "mean_reward": float(np.mean([item["reward"] for item in results])) if results else None,
        "per_episode": results,
    }


def run_bc_via_dagger_eval(source_hdf5_path: Path, checkpoint_path: Path, episodes: int, max_steps: int, render: bool, seed: int):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_policy(checkpoint_path, device)

    class Args:
        rollout_episodes = episodes
        rollout_max_steps = max_steps
        rollout_render = render

    dagger_eval = run_policy_rollout_eval(Args(), model, checkpoint, source_hdf5_path)
    direct_eval = run_rollout_evaluation(
        model=model,
        checkpoint=checkpoint,
        source_hdf5_path=source_hdf5_path,
        rollout_episodes=episodes,
        rollout_max_steps=max_steps,
        rollout_render=render,
        device=device,
    )
    return {"dagger_eval": dagger_eval, "direct_bc_eval": direct_eval}


def summarize_hdf5(path: Path):
    with h5py.File(path, "r") as f:
        data = f["data"]
        demos = sorted(data.keys(), key=lambda name: int(name.split("_")[-1]) if name.split("_")[-1].isdigit() else name)
        demo_rows = []
        for demo in demos:
            grp = data[demo]
            demo_rows.append(
                {
                    "demo": demo,
                    "states_shape": list(grp["states"].shape),
                    "actions_shape": list(grp["actions"].shape),
                    "success": bool(grp.attrs.get("success", False)),
                    "source": str(grp.attrs.get("source", "")),
                    "source_demo": str(grp.attrs.get("source_demo", "")),
                    "datasets": sorted(list(grp.keys())),
                }
            )
        return {
            "path": str(path),
            "total_attr": int(data.attrs.get("total", len(demos))),
            "attempted_episodes": int(data.attrs.get("attempted_episodes", 0)),
            "saved_episodes": int(data.attrs.get("saved_episodes", len(demos))),
            "success_count": int(data.attrs.get("success_count", 0)),
            "keep_failed_episodes": bool(data.attrs.get("keep_failed_episodes", False)),
            "demos": demo_rows,
        }


def summarize_dagger_results(final_root: Path, seed: int):
    ratios = ["noise0", "noise50", "noise100"]
    summary = {}
    for ratio in ratios:
        ratio_root = final_root / "dagger" / ratio / f"seed_{seed}"
        metrics = json.loads((ratio_root / "dagger_metrics.json").read_text(encoding="utf-8"))
        rows = []
        for round_data in metrics["rounds"]:
            row = {
                "round_idx": round_data["round_idx"],
                "skipped": bool(round_data.get("skipped", False)),
                "aggregate_data_paths": round_data.get("aggregate_data_paths", []),
                "collection": round_data.get("collection", {}),
                "training_split_info": None,
                "training_data_paths": None,
                "training_train_samples": None,
                "training_val_samples": None,
            }
            if round_data.get("training"):
                train = round_data["training"]
                row["training_split_info"] = train.get("split_info")
                row["training_data_paths"] = train.get("data_paths")
                row["training_train_samples"] = train.get("train_samples")
                row["training_val_samples"] = train.get("val_samples")
            hdf5_path = ratio_root / f"dagger_round_{round_data['round_idx']}" / f"dagger_round_{round_data['round_idx']}.hdf5"
            if hdf5_path.exists():
                row["hdf5"] = summarize_hdf5(hdf5_path)
            rows.append(row)
        summary[ratio] = rows
    return summary


def build_parser():
    project_new_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-root", type=str, default=str(project_new_root / "experiments" / "final_dual_arm"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--rollout-max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-render", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    final_root = Path(args.final_root).expanduser().resolve()
    data_path = final_root / "mixed_data" / "noise0" / "dual_arm_mixed.hdf5"
    checkpoint_path = final_root / "bc_ratio" / "noise0" / f"seed_{args.seed}" / "bc_best.pt"
    output_root = final_root / "diagnostics"

    expert = run_expert_rollout(data_path, args.episodes, args.rollout_max_steps, args.rollout_render, args.seed)
    write_json(output_root / "expert_only_noise0_eval.json", expert)

    bc_clean = run_bc_via_dagger_eval(data_path, checkpoint_path, args.episodes, args.rollout_max_steps, args.rollout_render, args.seed)
    write_json(output_root / "bc_clean_via_dagger_eval.json", bc_clean)

    dagger_summary = summarize_dagger_results(final_root, args.seed)
    write_json(output_root / "dagger_aggregation_and_hdf5_summary.json", dagger_summary)

    print(
        json.dumps(
            {
                "expert_only": {
                    "success_rate": expert["success_rate"],
                    "success_count": expert["success_count"],
                    "episodes": expert["episodes"],
                    "mean_steps": expert["mean_steps"],
                },
                "bc_clean_via_dagger_eval": {
                    "success_rate": bc_clean["dagger_eval"]["success_rate"],
                    "success_count": bc_clean["dagger_eval"]["success_count"],
                    "episodes": bc_clean["dagger_eval"]["episodes"],
                    "mean_steps": bc_clean["dagger_eval"]["mean_steps"],
                },
                "output_root": str(output_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
