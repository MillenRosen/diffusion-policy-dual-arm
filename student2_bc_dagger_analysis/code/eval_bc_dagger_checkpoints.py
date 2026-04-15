import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(r"D:\a_6019")
for candidate in [SCRIPT_DIR, PROJECT_DIR]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from bc_train import BCMLP
from collect_scripted_demos import make_env


def load_policy(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = BCMLP(
        state_dim=int(checkpoint["state_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


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


def evaluate_policy(name, checkpoint_path, episodes, max_steps, device):
    model, checkpoint = load_policy(checkpoint_path, device)
    env, _ = make_env(has_renderer=False, renderer="mjviewer", control_freq=20, horizon=max_steps)

    results = []
    success_count = 0
    try:
        for episode_idx in range(episodes):
            obs = env.reset()
            total_reward = 0.0
            success = False

            for step_idx in range(max_steps):
                _ = obs
                state = env.sim.get_state().flatten().astype(np.float32)
                action = policy_action(model, checkpoint, state, device)
                obs, reward, done, info = env.step(action)
                total_reward += float(reward)

                if env._check_success():
                    success = True
                    success_count += 1
                    results.append(
                        {
                            "episode": episode_idx,
                            "success": True,
                            "steps": step_idx + 1,
                            "reward": total_reward,
                        }
                    )
                    break

            if not success:
                results.append(
                    {
                        "episode": episode_idx,
                        "success": False,
                        "steps": max_steps,
                        "reward": total_reward,
                    }
                )
            print(f"{name} episode {episode_idx + 1:02d}/{episodes}: {'SUCCESS' if success else 'FAIL'}")
    finally:
        env.close()

    return {
        "name": name,
        "checkpoint_path": str(checkpoint_path),
        "episodes": episodes,
        "max_steps": max_steps,
        "success_count": success_count,
        "success_rate": float(success_count / episodes),
        "mean_steps": float(np.mean([item["steps"] for item in results])),
        "mean_reward": float(np.mean([item["reward"] for item in results])),
        "per_episode": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument(
        "--output",
        type=str,
        default=r"D:\a_6019\outputs_eval_noisy_bc_vs_dagger_round2\stable_eval_metrics.json",
    )
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoints = {
        "BC noisy": Path(r"D:\a_6019\outputs_bc_noisy_vs_dagger\bc_best.pt"),
        "DAgger round 2": Path(r"D:\a_6019\outputs_dagger_noisy_vs_bc\round_2_train\dagger_round_2_best.pt"),
    }

    metrics = {
        "device": str(device),
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "results": [
            evaluate_policy(name, path, args.episodes, args.max_steps, device)
            for name, path in checkpoints.items()
        ],
    }

    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved stable evaluation metrics to: {output_path}")


if __name__ == "__main__":
    main()
