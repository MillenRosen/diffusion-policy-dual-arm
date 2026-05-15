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
from core.pipeline.hdf5 import gather_demonstrations_as_hdf5


def rollout_one_episode(env, expert, max_steps: int = 600, render: bool = False, verbose: bool = False):
    obs = env.reset()
    expert.reset()

    for t in range(max_steps):
        action = expert.act(obs)
        obs, reward, done, info = env.step(action)

        if render:
            env.render()

        if verbose and t % 20 == 0:
            print(
                f"t={t}, stage={expert.stage}, "
                f"pot_z={obs['pot_pos'][2]:.4f}, "
                f"rel0={np.linalg.norm(obs['gripper0_to_handle0']):.4f}, "
                f"rel1={np.linalg.norm(obs['gripper1_to_handle1']):.4f}, "
                f"success={env._check_success()}"
            )

        if env._check_success():
            return True, t + 1

    return False, max_steps


def collect_sequential_clean(
    num_episodes: int,
    save_dir: str,
    max_steps: int = 600,
    render: bool = False,
    verbose: bool = False,
    keep_tmp: bool = False,
    hdf5_name: str = "demo.hdf5",
    mode: str = "robust",
    order_mode: str = "left_first",
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

    tmp_dir = os.path.join(save_dir, f"tmp_sequential_clean_{str(time.time()).replace('.', '_')}")
    env = DataCollectionWrapper(base_env, tmp_dir)
    expert = SequentialExpert(env, mode=mode, order_mode=order_mode)

    success_count = 0
    stats = []

    for ep in range(num_episodes):
        print(f"\n========== Episode {ep:03d} ==========")
        success, steps = rollout_one_episode(
            env=env,
            expert=expert,
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
    print(f"Temporary dir      : {tmp_dir}")

    if not keep_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("Temporary raw directory removed.")

    return hdf5_path, saved_eps, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect clean sequential-bimanual expert demonstrations in official robosuite HDF5 format.")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--save_dir", type=str, default="./demonstrations/sequential/expert")
    parser.add_argument("--hdf5_name", type=str, default="demo.hdf5")
    parser.add_argument("--mode", type=str, choices=["strict", "robust"], default="robust")
    parser.add_argument("--order_mode", type=str, choices=["left_first", "right_first"], default="left_first")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--keep_tmp", action="store_true")
    args = parser.parse_args()

    collect_sequential_clean(
        num_episodes=args.num_episodes,
        save_dir=args.save_dir,
        max_steps=args.max_steps,
        render=args.render,
        verbose=args.verbose,
        keep_tmp=args.keep_tmp,
        hdf5_name=args.hdf5_name,
        mode=args.mode,
        order_mode=args.order_mode,
    )


if __name__ == "__main__":
    main()
