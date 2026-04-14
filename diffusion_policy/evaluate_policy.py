from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import robosuite as suite
import matplotlib

matplotlib.use("Agg")  # 非交互后端
import matplotlib.pyplot as plt

from diffusion import DiffusionScheduler
from eval_diffusion import DiffusionPolicy
from model import ConditionalUnet1D


def load_env_metadata(hdf5_path: str | Path):
    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        env_name = data.attrs.get("env", "TwoArmLift")
        env_info = json.loads(data.attrs["env_info"])
    return env_name, env_info


def make_env_from_hdf5(hdf5_path: str | Path, horizon=600, render=False, offscreen=False):
    """
    Create environment from HDF5 metadata.
    If offscreen=True, enable offscreen rendering for screenshots.
    """
    env_name, env_info = load_env_metadata(hdf5_path)
    controller_configs = env_info.get("controller_configs")
    env = suite.make(
        env_name=env_name,
        robots=env_info.get("robots", ["Panda", "Panda"]),
        controller_configs=controller_configs,
        env_configuration=env_info.get("env_configuration", "parallel"),
        has_renderer=render,
        has_offscreen_renderer=offscreen,
        ignore_done=True,
        use_camera_obs=offscreen,  # needed for offscreen rendering
        camera_names=["agentview"] if offscreen else [],
        camera_heights=480,
        camera_widths=640,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=20,
        horizon=horizon,
    )
    return env


def get_full_state(env):
    return env.sim.get_state().flatten().astype(np.float32)


def capture_success_screenshot(env, save_path):
    """Capture an image from the environment's agentview camera."""
    img = env.sim.render(camera_name="agentview", height=480, width=640, depth=False)
    plt.imsave(save_path, img)


def plot_results(results_dict, output_dir):
    """Generate and save result figures."""
    episodes = results_dict["episodes"]
    success_count = results_dict["success_count"]
    success_rate = results_dict["success_rate"]
    episode_steps = results_dict["episode_steps"]  # list of steps per episode

    # 1. Success rate bar chart
    plt.figure(figsize=(6, 4))
    plt.bar(["Success", "Failure"], [success_count, episodes - success_count], color=["green", "red"])
    plt.ylabel("Number of Episodes")
    plt.title(f"Success Rate: {success_rate*100:.1f}% ({success_count}/{episodes})")
    plt.tight_layout()
    plt.savefig(output_dir / "success_rate.png", dpi=150)
    plt.close()

    # 2. Steps histogram
    plt.figure(figsize=(8, 5))
    plt.hist(episode_steps, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel("Episode Length (steps)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Episode Lengths")
    plt.axvline(x=np.mean(episode_steps), color='red', linestyle='--', label=f"Mean: {np.mean(episode_steps):.1f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "steps_histogram.png", dpi=150)
    plt.close()

    # 3. Steps per episode line plot
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, episodes+1), episode_steps, marker='o', linestyle='-', markersize=4)
    plt.xlabel("Episode Index")
    plt.ylabel("Steps Taken")
    plt.title("Steps per Episode")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "steps_per_episode.png", dpi=150)
    plt.close()

    print(f"Result figures saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--source_hdf5_path",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "script_collect_demo2.hdf5"),
    )
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save_screenshots", action="store_true", help="Save screenshots of successful episodes")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save plots and screenshots")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location=device)
    train_args = checkpoint["args"]
    norm_stats_path = checkpoint.get("norm_stats_path", train_args["norm_stats_path"])
    norm_stats = np.load(norm_stats_path)

    model = ConditionalUnet1D(
        obs_dim=int(checkpoint["flat_obs_dim"]),
        act_dim=int(checkpoint["act_dim"]),
        action_chunk=int(checkpoint["action_chunk"]),
        cond_dim=int(train_args["cond_dim"]),
        time_dim=int(train_args["time_dim"]),
        down_dims=list(train_args["down_dims"]),
        kernel_size=int(train_args["kernel_size"]),
        n_groups=int(train_args["n_groups"]),
        dropout=float(train_args["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scheduler = DiffusionScheduler(num_steps=int(train_args["num_diffusion_steps"]), device=device)
    policy = DiffusionPolicy(
        model=model,
        scheduler=scheduler,
        obs_dim=int(checkpoint["obs_dim"]),
        act_dim=int(checkpoint["act_dim"]),
        action_chunk=int(checkpoint["action_chunk"]),
        obs_horizon=int(checkpoint["obs_horizon"]),
        norm_stats=norm_stats,
        device=device,
    )

    # Set output directory
    if args.output_dir is None:
        output_dir = Path(args.model_path).parent / "evaluation_results"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    episode_steps = []
    successful_screenshots = 0

    for episode_idx in range(args.num_episodes):
        # For screenshots, use offscreen rendering; otherwise use render flag
        use_offscreen = args.save_screenshots and (successful_screenshots < 5)  # limit to 5 screenshots
        env = make_env_from_hdf5(
            args.source_hdf5_path,
            horizon=args.max_steps,
            render=args.render,
            offscreen=use_offscreen
        )
        policy.reset()
        env.reset()
        obs = get_full_state(env)
        success = False
        steps_taken = args.max_steps

        for step_idx in range(args.max_steps):
            action = policy.act(obs)
            obs, _, _, _ = env.step(action)
            obs = get_full_state(env)
            if env._check_success():
                success = True
                success_count += 1
                steps_taken = step_idx + 1
                episode_steps.append(steps_taken)
                if use_offscreen and successful_screenshots < 5:
                    screenshot_path = output_dir / f"success_ep{episode_idx+1}.png"
                    capture_success_screenshot(env, screenshot_path)
                    successful_screenshots += 1
                break
        if not success:
            episode_steps.append(args.max_steps)

        print(f"Episode {episode_idx + 1}: {'success' if success else 'fail'} steps={steps_taken}")
        env.close()

    # Save results JSON
    results = {
        "episodes": args.num_episodes,
        "success_count": success_count,
        "success_rate": success_count / args.num_episodes if args.num_episodes else 0.0,
        "mean_steps": float(np.mean(episode_steps)) if episode_steps else None,
        "episode_steps": episode_steps,
        "model_path": args.model_path,
        "source_hdf5_path": args.source_hdf5_path,
    }
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate plots
    plot_results(results, output_dir)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()