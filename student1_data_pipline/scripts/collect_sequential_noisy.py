from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import numpy as np
from robosuite.wrappers import DataCollectionWrapper

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.envs.sequential_env import make_sequential_env, sequential_env_info_json
from core.experts.sequential_expert import SequentialExpert
from core.noise.registry import build_composite_noise
from core.pipeline.hdf5 import gather_demonstrations_as_hdf5


def rollout_one_episode(
    env,
    expert,
    noise,
    max_steps: int = 600,
    render: bool = False,
    verbose: bool = False,
):
    obs = env.reset()
    expert.reset()
    if noise is not None:
        noise.reset()

    for t in range(max_steps):
        clean_action = expert.act(obs)
        expert_info = expert.info() if hasattr(expert, "info") else None

        exec_action = clean_action
        if noise is not None:
            exec_action = noise.apply(
                clean_action,
                obs=obs,
                expert_info=expert_info,
                t=t,
            )

        obs, reward, done, info = env.step(exec_action)

        if render:
            env.render()

        if verbose and t % 20 == 0:
            stage_val = expert_info.get("stage") if isinstance(expert_info, dict) else getattr(expert, "stage", None)
            print(
                f"t={t}, stage={stage_val}, "
                f"pot_z={obs['pot_pos'][2]:.4f}, "
                f"rel0={np.linalg.norm(obs['gripper0_to_handle0']):.4f}, "
                f"rel1={np.linalg.norm(obs['gripper1_to_handle1']):.4f}, "
                f"success={env._check_success()}"
            )

        if env._check_success():
            return True, t + 1

    return False, max_steps


def collect_sequential_noisy(
    num_episodes: int,
    save_dir: str,
    max_steps: int = 600,
    render: bool = False,
    verbose: bool = False,
    keep_tmp: bool = False,
    hdf5_name: str = "demo_noisy.hdf5",
    mode: str = "robust",
    order_mode: str = "left_first",
    use_gaussian: bool = True,
    gaussian_sigma: float = 0.02,
    gaussian_seed: int | None = 42,
    use_temporal: bool = False,
    temporal_sigma: float = 0.015,
    temporal_smoothing: float = 0.75,
    temporal_seed: int | None = 43,
    use_stage_aware: bool = False,
    stage_base_sigma: float = 0.02,
    stage_seed: int | None = 44,
):
    os.makedirs(save_dir, exist_ok=True)

    base_env, config = make_sequential_env(
        has_renderer=render,
        renderer="mjviewer",
        control_freq=20,
        horizon=max_steps,
        order_mode=order_mode,
    )
    env_info = sequential_env_info_json(config)

    tmp_dir = os.path.join(save_dir, f"tmp_sequential_noisy_{str(time.time()).replace('.', '_')}")
    env = DataCollectionWrapper(base_env, tmp_dir)
    expert = SequentialExpert(env, mode=mode, order_mode=order_mode)
    noise = build_composite_noise(
        use_gaussian=use_gaussian,
        use_temporal=use_temporal,
        use_stage_aware=use_stage_aware,
        gaussian_kwargs={"sigma": gaussian_sigma, "seed": gaussian_seed},
        temporal_kwargs={"sigma": temporal_sigma, "smoothing": temporal_smoothing, "seed": temporal_seed},
        stage_aware_kwargs={"base_sigma": stage_base_sigma, "seed": stage_seed},
    )

    success_count = 0
    stats = []

    for ep in range(num_episodes):
        print(f"\n========== Episode {ep:03d} ==========")
        success, steps = rollout_one_episode(
            env=env,
            expert=expert,
            noise=noise,
            max_steps=max_steps,
            render=render,
            verbose=verbose,
        )
        success_count += int(success)
        stats.append((success, steps))

        print(
            f"[Episode {ep:03d}] {'SUCCESS' if success else 'FAIL'} | "
            f"steps={steps} | success_rate_so_far={success_count}/{ep + 1}"
        )

        env.reset()

    env.close()

    hdf5_path, saved_eps = gather_demonstrations_as_hdf5(
        tmp_dir,
        save_dir,
        env_info,
        hdf5_name=hdf5_name,
    )

    print("\n===== COLLECTION SUMMARY =====")
    print(f"Requested episodes : {num_episodes}")
    print(f"Successful rollouts: {success_count}")
    print(f"Saved demos in hdf5: {saved_eps}")
    print(f"HDF5 path          : {hdf5_path}")
    print(f"Order mode         : {order_mode}")
    print(f"Gaussian enabled   : {use_gaussian} (sigma={gaussian_sigma})")
    print(f"Temporal enabled   : {use_temporal} (sigma={temporal_sigma}, smoothing={temporal_smoothing})")
    print(f"Stage-aware enabled: {use_stage_aware} (base_sigma={stage_base_sigma})")
    print(f"Temporary dir      : {tmp_dir}")

    if not keep_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("Temporary raw directory removed.")

    return hdf5_path, saved_eps, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect noisy sequential-bimanual expert demonstrations in official robosuite HDF5 format.")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--save_dir", type=str, default="./demonstrations/sequential/noisy")
    parser.add_argument("--hdf5_name", type=str, default="demo_noisy.hdf5")
    parser.add_argument("--mode", type=str, choices=["strict", "robust"], default="robust")
    parser.add_argument("--order_mode", type=str, choices=["left_first", "right_first"], default="left_first")

    parser.add_argument("--use_gaussian", action="store_true", help="Enable Gaussian action noise.")
    parser.add_argument("--gaussian_sigma", type=float, default=0.02)
    parser.add_argument("--gaussian_seed", type=int, default=42)

    parser.add_argument("--use_temporal", action="store_true", help="Enable temporal correlated action noise.")
    parser.add_argument("--temporal_sigma", type=float, default=0.015)
    parser.add_argument("--temporal_smoothing", type=float, default=0.75)
    parser.add_argument("--temporal_seed", type=int, default=43)

    parser.add_argument("--use_stage_aware", action="store_true", help="Enable stage-aware action noise.")
    parser.add_argument("--stage_base_sigma", type=float, default=0.02)
    parser.add_argument("--stage_seed", type=int, default=44)

    parser.add_argument("--render", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--keep_tmp", action="store_true")
    args = parser.parse_args()

    use_gaussian = args.use_gaussian or (not args.use_temporal and not args.use_stage_aware)

    collect_sequential_noisy(
        num_episodes=args.num_episodes,
        save_dir=args.save_dir,
        max_steps=args.max_steps,
        render=args.render,
        verbose=args.verbose,
        keep_tmp=args.keep_tmp,
        hdf5_name=args.hdf5_name,
        mode=args.mode,
        order_mode=args.order_mode,
        use_gaussian=use_gaussian,
        gaussian_sigma=args.gaussian_sigma,
        gaussian_seed=args.gaussian_seed,
        use_temporal=args.use_temporal,
        temporal_sigma=args.temporal_sigma,
        temporal_smoothing=args.temporal_smoothing,
        temporal_seed=args.temporal_seed,
        use_stage_aware=args.use_stage_aware,
        stage_base_sigma=args.stage_base_sigma,
        stage_seed=args.stage_seed,
    )


if __name__ == "__main__":
    main()
