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

from core.envs.single_arm_env import make_single_arm_env, single_arm_env_info_json
from core.experts.single_arm_expert import SingleArmExpert
from core.pipeline.hdf5 import gather_demonstrations_as_hdf5


def _get_cube_pos(obs):
    if "cube_pos" in obs:
        return np.asarray(obs["cube_pos"], dtype=np.float32)
    if "cubeA_pos" in obs:
        return np.asarray(obs["cubeA_pos"], dtype=np.float32)
    if "object-state" in obs:
        arr = np.asarray(obs["object-state"], dtype=np.float32).reshape(-1)
        if arr.size >= 3:
            return arr[:3]
    return None


def rollout_one_episode(env, expert, max_steps: int = 400, render: bool = False, verbose: bool = False):
    obs = env.reset()
    expert.reset()

    for t in range(max_steps):
        action = expert.act(obs)
        obs, reward, done, info = env.step(action)

        if render:
            env.render()

        if verbose and t % 20 == 0:
            cube_pos = _get_cube_pos(obs)
            cube_z = float(cube_pos[2]) if cube_pos is not None else float("nan")
            eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
            dist = float(np.linalg.norm(eef - cube_pos)) if cube_pos is not None else float("nan")
            print(
                f"t={t}, stage={expert.stage}, "
                f"cube_z={cube_z:.4f}, "
                f"eef_to_cube={dist:.4f}, "
                f"success={env._check_success()}"
            )

        if env._check_success():
            return True, t + 1

    return False, max_steps


def collect_single_arm_clean(
    num_episodes: int,
    save_dir: str,
    max_steps: int = 400,
    render: bool = False,
    verbose: bool = False,
    keep_tmp: bool = False,
    hdf5_name: str = "demo.hdf5",
    mode: str = "robust",
):
    os.makedirs(save_dir, exist_ok=True)

    base_env, config = make_single_arm_env(
        has_renderer=render,
        renderer="mjviewer",
        control_freq=20,
        horizon=max_steps,
    )
    env_info = single_arm_env_info_json(config)

    tmp_dir = os.path.join(save_dir, f"tmp_single_arm_clean_{str(time.time()).replace('.', '_')}")
    env = DataCollectionWrapper(base_env, tmp_dir)
    expert = SingleArmExpert(env, mode=mode)

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
    print(f"Temporary dir      : {tmp_dir}")

    if not keep_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("Temporary raw directory removed.")

    return hdf5_path, saved_eps, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect clean single-arm expert demonstrations in official robosuite HDF5 format.")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--save_dir", type=str, default="./demonstrations/single_arm/expert")
    parser.add_argument("--hdf5_name", type=str, default="demo.hdf5")
    parser.add_argument("--mode", type=str, choices=["strict", "robust"], default="robust")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--keep_tmp", action="store_true")
    args = parser.parse_args()

    collect_single_arm_clean(
        num_episodes=args.num_episodes,
        save_dir=args.save_dir,
        max_steps=args.max_steps,
        render=args.render,
        verbose=args.verbose,
        keep_tmp=args.keep_tmp,
        hdf5_name=args.hdf5_name,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
