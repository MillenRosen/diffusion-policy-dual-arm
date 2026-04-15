import argparse
import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
import robosuite.utils.transform_utils as T


def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def quat_to_yaw(quat):
    mat = T.quat2mat(quat)
    yaw = np.arctan2(mat[1, 0], mat[0, 0])
    return yaw


def shortest_perpendicular_yaw_error(target_yaw, current_yaw):
    err_pos = wrap_to_pi(target_yaw + np.pi / 2.0 - current_yaw)
    err_neg = wrap_to_pi(target_yaw - np.pi / 2.0 - current_yaw)
    return err_pos if abs(err_pos) <= abs(err_neg) else err_neg


class TwoArmLiftScriptedExpert:
    def __init__(self, env):
        self.env = env
        self.stage = 0
        self.stage_step = 0
        self.last_stage_reason = "init"

        self.approach_z_offset = 0.08
        self.grasp_z_offset = -0.005

        self.approach_gain = 4.0
        self.hover_align_gain = 3.0
        self.descend_gain_xy = 2.5
        self.descend_gain_z = 3.5
        self.grasp_stabilize_gain = 1.5

        self.approach_thresh = 0.03
        self.hover_xy_thresh = 0.015
        self.hover_z_thresh = 0.012
        self.rot_align_thresh = 0.10
        self.descend_norm_thresh = 0.012
        self.descend_z_thresh = 0.008
        self.loose_descend_norm_thresh = 0.018

        self.hover_rotate_min_steps = 6
        self.hover_rotate_timeout_steps = 60
        self.descend_timeout_steps = 100
        self.grasp_hold_steps = 20

        self.rot_gain = 1.6
        self.rot_clip = 0.30
        self.rot_cmd_sign = -1.0

        # Anti-jitter only: deadzone + hysteresis + stronger smoothing + command slew limit
        self.yaw_deadzone = 0.045
        self.yaw_reactivate_thresh = 0.075
        self.filtered_rot_deadzone = 0.012

        self.lift_cmd = 0.9

        self.rotation_locked = False
        self.locked_yaw0 = None
        self.locked_yaw1 = None

        self.rot_decay = 0.90
        self.max_rot_delta_per_step = 0.05
        self.current_rot_cmd0 = np.zeros(3, dtype=np.float32)
        self.current_rot_cmd1 = np.zeros(3, dtype=np.float32)
        self.rot_control_active0 = True
        self.rot_control_active1 = True
        self.rot_control_active0 = True
        self.rot_control_active1 = True

    def reset(self):
        self.stage = 0
        self.stage_step = 0
        self.last_stage_reason = "reset"
        self.rotation_locked = False
        self.locked_yaw0 = None
        self.locked_yaw1 = None
        self.current_rot_cmd0 = np.zeros(3, dtype=np.float32)
        self.current_rot_cmd1 = np.zeros(3, dtype=np.float32)

    def _build_robot_action(self, robot, arm_delta, gripper_cmd):
        action_dict = {
            "right": arm_delta.astype(np.float32),
            "right_gripper": np.array([gripper_cmd], dtype=np.float32),
        }
        return robot.create_action_vector(action_dict)

    def _current_eef_yaw(self, obs, robot_idx):
        eef_quat = obs["robot0_eef_quat"] if robot_idx == 0 else obs["robot1_eef_quat"]
        return quat_to_yaw(eef_quat)

    def _perpendicular_yaw_error(self, obs, robot_idx):
        pot_yaw = quat_to_yaw(obs["pot_quat"])
        eef_yaw = self._current_eef_yaw(obs, robot_idx)
        return shortest_perpendicular_yaw_error(pot_yaw, eef_yaw)

    def _compute_rot_cmd(self, obs, robot_idx):
        yaw_err = self._perpendicular_yaw_error(obs, robot_idx)

        # Hysteresis: once rotation is effectively done, keep it silent unless the error
        # grows clearly larger again. This prevents zero / non-zero / zero chatter.
        if robot_idx == 0:
            active = self.rot_control_active0
        else:
            active = self.rot_control_active1

        if active:
            if abs(yaw_err) < self.yaw_deadzone:
                active = False
        else:
            if abs(yaw_err) > self.yaw_reactivate_thresh:
                active = True

        if robot_idx == 0:
            self.rot_control_active0 = active
        else:
            self.rot_control_active1 = active

        if not active:
            return np.zeros(3, dtype=np.float32)

        rot_cmd = np.zeros(3, dtype=np.float32)
        rot_cmd[2] = np.clip(
            self.rot_cmd_sign * self.rot_gain * yaw_err,
            -self.rot_clip,
            self.rot_clip,
        )
        return rot_cmd

    def _smooth_rot_cmd(self, current_cmd, raw_cmd):
        smoothed = self.rot_decay * current_cmd + (1 - self.rot_decay) * raw_cmd

        out = np.zeros(3, dtype=np.float32)
        delta_z = np.clip(
            smoothed[2] - current_cmd[2],
            -self.max_rot_delta_per_step,
            self.max_rot_delta_per_step,
        )
        out[2] = current_cmd[2] + delta_z

        if abs(out[2]) < self.filtered_rot_deadzone:
            out[2] = 0.0
        return out

    def _xy_norm(self, err):
        return float(np.linalg.norm(err[:2]))

    def _lock_current_yaws(self, obs):
        self.rotation_locked = True
        self.locked_yaw0 = self._current_eef_yaw(obs, 0)
        self.locked_yaw1 = self._current_eef_yaw(obs, 1)

    def act(self, obs):
        eef0 = obs["robot0_eef_pos"]
        eef1 = obs["robot1_eef_pos"]
        handle0 = obs["handle0_xpos"]
        handle1 = obs["handle1_xpos"]

        grip0 = -1.0
        grip1 = -1.0

        delta0_xyz = np.zeros(3, dtype=np.float32)
        delta1_xyz = np.zeros(3, dtype=np.float32)
        rot0 = np.zeros(3, dtype=np.float32)
        rot1 = np.zeros(3, dtype=np.float32)

        if self.stage == 0:
            target0 = handle0 + np.array([0.0, 0.0, self.approach_z_offset], dtype=np.float32)
            target1 = handle1 + np.array([0.0, 0.0, self.approach_z_offset], dtype=np.float32)

            err0 = target0 - eef0
            err1 = target1 - eef1

            delta0_xyz = self.approach_gain * err0
            delta1_xyz = self.approach_gain * err1

            if np.linalg.norm(err0) < self.approach_thresh and np.linalg.norm(err1) < self.approach_thresh:
                self.stage = 1
                self.stage_step = 0
                self.last_stage_reason = "hover_pose_ready"

        elif self.stage == 1:
            target0 = handle0 + np.array([0.0, 0.0, self.approach_z_offset], dtype=np.float32)
            target1 = handle1 + np.array([0.0, 0.0, self.approach_z_offset], dtype=np.float32)

            err0 = target0 - eef0
            err1 = target1 - eef1

            delta0_xyz = self.hover_align_gain * err0
            delta1_xyz = self.hover_align_gain * err1

            raw_rot0 = self._compute_rot_cmd(obs, robot_idx=0)
            raw_rot1 = self._compute_rot_cmd(obs, robot_idx=1)

            self.current_rot_cmd0 = self._smooth_rot_cmd(self.current_rot_cmd0, raw_rot0)
            self.current_rot_cmd1 = self._smooth_rot_cmd(self.current_rot_cmd1, raw_rot1)

            rot0 = self.current_rot_cmd0.copy()
            rot1 = self.current_rot_cmd1.copy()

            yaw_err0 = abs(self._perpendicular_yaw_error(obs, 0))
            yaw_err1 = abs(self._perpendicular_yaw_error(obs, 1))

            self.stage_step += 1

            hover_pos_ready = (
                self._xy_norm(err0) < self.hover_xy_thresh
                and self._xy_norm(err1) < self.hover_xy_thresh
                and abs(err0[2]) < self.hover_z_thresh
                and abs(err1[2]) < self.hover_z_thresh
            )
            rot_ready = (yaw_err0 < self.rot_align_thresh and yaw_err1 < self.rot_align_thresh)

            if hover_pos_ready and rot_ready and self.stage_step >= self.hover_rotate_min_steps:
                self._lock_current_yaws(obs)
                self.stage = 2
                self.stage_step = 0
                self.last_stage_reason = "hover_perpendicular_rotation_ready"
            elif hover_pos_ready and self.stage_step >= self.hover_rotate_timeout_steps:
                self._lock_current_yaws(obs)
                self.stage = 2
                self.stage_step = 0
                self.last_stage_reason = "hover_rotation_timeout_locked"

        elif self.stage == 2:
            target0 = handle0 + np.array([0.0, 0.0, self.grasp_z_offset], dtype=np.float32)
            target1 = handle1 + np.array([0.0, 0.0, self.grasp_z_offset], dtype=np.float32)

            err0 = target0 - eef0
            err1 = target1 - eef1

            delta0_xyz = np.array([
                self.descend_gain_xy * err0[0],
                self.descend_gain_xy * err0[1],
                self.descend_gain_z * err0[2],
            ], dtype=np.float32)
            delta1_xyz = np.array([
                self.descend_gain_xy * err1[0],
                self.descend_gain_xy * err1[1],
                self.descend_gain_z * err1[2],
            ], dtype=np.float32)

            self.current_rot_cmd0 = np.zeros(3, dtype=np.float32)
            self.current_rot_cmd1 = np.zeros(3, dtype=np.float32)
            rot0 = np.zeros(3, dtype=np.float32)
            rot1 = np.zeros(3, dtype=np.float32)

            self.stage_step += 1

            strict_ready = (
                np.linalg.norm(err0) < self.descend_norm_thresh
                and np.linalg.norm(err1) < self.descend_norm_thresh
                and abs(err0[2]) < self.descend_z_thresh
                and abs(err1[2]) < self.descend_z_thresh
            )

            loose_ready = (
                np.linalg.norm(err0) < self.loose_descend_norm_thresh
                and np.linalg.norm(err1) < self.loose_descend_norm_thresh
            )

            timeout_ready = (
                self.stage_step >= self.descend_timeout_steps
                and np.linalg.norm(err0) < 0.025
                and np.linalg.norm(err1) < 0.025
            )

            if strict_ready:
                self.stage = 3
                self.stage_step = 0
                self.last_stage_reason = "descend_pose_ready"
            elif loose_ready:
                self.stage = 3
                self.stage_step = 0
                self.last_stage_reason = "descend_loose_ready"
            elif timeout_ready:
                self.stage = 3
                self.stage_step = 0
                self.last_stage_reason = "descend_timeout_ready"

        elif self.stage == 3:
            target0 = handle0 + np.array([0.0, 0.0, self.grasp_z_offset], dtype=np.float32)
            target1 = handle1 + np.array([0.0, 0.0, self.grasp_z_offset], dtype=np.float32)

            err0 = target0 - eef0
            err1 = target1 - eef1

            delta0_xyz = self.grasp_stabilize_gain * err0
            delta1_xyz = self.grasp_stabilize_gain * err1

            rot0 = np.zeros(3, dtype=np.float32)
            rot1 = np.zeros(3, dtype=np.float32)

            grip0 = 1.0
            grip1 = 1.0

            self.stage_step += 1
            if self.stage_step >= self.grasp_hold_steps:
                self.stage = 4
                self.stage_step = 0
                self.last_stage_reason = "grasp_hold_done"

        elif self.stage == 4:
            delta0_xyz = np.array([0.0, 0.0, self.lift_cmd], dtype=np.float32)
            delta1_xyz = np.array([0.0, 0.0, self.lift_cmd], dtype=np.float32)

            rot0 = np.zeros(3, dtype=np.float32)
            rot1 = np.zeros(3, dtype=np.float32)

            grip0 = 1.0
            grip1 = 1.0

        arm0 = np.zeros(6, dtype=np.float32)
        arm1 = np.zeros(6, dtype=np.float32)

        arm0[:3] = np.clip(delta0_xyz, -1.0, 1.0)
        arm1[:3] = np.clip(delta1_xyz, -1.0, 1.0)

        arm0[3:] = np.clip(rot0, -1.0, 1.0)
        arm1[3:] = np.clip(rot1, -1.0, 1.0)

        a0 = self._build_robot_action(self.env.robots[0], arm0, grip0)
        a1 = self._build_robot_action(self.env.robots[1], arm1, grip1)
        return np.clip(np.concatenate([a0, a1]), -1.0, 1.0)


def make_env(has_renderer=False, renderer="mjviewer", control_freq=20, horizon=600):
    controller_config = load_composite_controller_config(
        controller=None,
        robot="Panda",
    )

    config = {
        "env_name": "TwoArmLift",
        "robots": ["Panda", "Panda"],
        "controller_configs": controller_config,
        "env_configuration": "parallel",
    }

    env = suite.make(
        **config,
        has_renderer=has_renderer,
        renderer=renderer,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=control_freq,
        horizon=horizon,
    )
    return env


def run_one_episode(env, expert, max_steps=600, render=False, verbose=False, pause_on_fail=False):
    obs = env.reset()
    expert.reset()

    last_stage = expert.stage
    stage_history = [expert.stage]
    stage_reason_history = [expert.last_stage_reason]

    for t in range(max_steps):
        action = expert.act(obs)
        obs, reward, done, info = env.step(action)

        if render:
            env.render()

        if expert.stage != last_stage:
            stage_history.append(expert.stage)
            stage_reason_history.append(expert.last_stage_reason)
            print(f">>> stage changed: {last_stage} -> {expert.stage} ({expert.last_stage_reason})")
            last_stage = expert.stage

        if verbose and t % 20 == 0:
            eef0 = obs["robot0_eef_pos"]
            eef1 = obs["robot1_eef_pos"]
            handle0 = obs["handle0_xpos"]
            handle1 = obs["handle1_xpos"]

            pot_yaw = quat_to_yaw(obs["pot_quat"])
            eef0_yaw = quat_to_yaw(obs["robot0_eef_quat"])
            eef1_yaw = quat_to_yaw(obs["robot1_eef_quat"])
            yaw_err0 = expert._perpendicular_yaw_error(obs, 0)
            yaw_err1 = expert._perpendicular_yaw_error(obs, 1)

            print(
                f"t={t}, stage={expert.stage}, "
                f"stage_step={expert.stage_step}, "
                f"pot_z={obs['pot_pos'][2]:.4f}, "
                f"rel0={np.linalg.norm(obs['gripper0_to_handle0']):.4f}, "
                f"rel1={np.linalg.norm(obs['gripper1_to_handle1']):.4f}, "
                f"dz0={handle0[2]-eef0[2]:.4f}, "
                f"dz1={handle1[2]-eef1[2]:.4f}, "
                f"pot_yaw={pot_yaw:.4f}, "
                f"eef0_yaw={eef0_yaw:.4f}, "
                f"eef1_yaw={eef1_yaw:.4f}, "
                f"perp_yaw_err0={yaw_err0:.4f}, "
                f"perp_yaw_err1={yaw_err1:.4f}, "
                f"locked={expert.rotation_locked}, "
                f"success={env._check_success()}"
            )

        if env._check_success():
            return {
                "success": True,
                "steps": t,
                "final_stage": expert.stage,
                "final_pot_z": float(obs["pot_pos"][2]),
                "final_rel0": float(np.linalg.norm(obs["gripper0_to_handle0"])),
                "final_rel1": float(np.linalg.norm(obs["gripper1_to_handle1"])),
                "stage_history": stage_history,
                "stage_reason_history": stage_reason_history,
            }

    result = {
        "success": False,
        "steps": max_steps,
        "final_stage": expert.stage,
        "final_pot_z": float(obs["pot_pos"][2]),
        "final_rel0": float(np.linalg.norm(obs["gripper0_to_handle0"])),
        "final_rel1": float(np.linalg.norm(obs["gripper1_to_handle1"])),
        "stage_history": stage_history,
        "stage_reason_history": stage_reason_history,
    }

    if pause_on_fail and render:
        print("\nEpisode failed. Inspect the viewer, then press Enter to continue...")
        input()

    return result


def summarize(results):
    num_eps = len(results)
    num_success = sum(r["success"] for r in results)
    success_rate = num_success / num_eps if num_eps > 0 else 0.0

    success_steps = [r["steps"] for r in results if r["success"]]
    avg_success_steps = float(np.mean(success_steps)) if success_steps else None

    fail_stage_counts = {}
    for r in results:
        if not r["success"]:
            fail_stage_counts[r["final_stage"]] = fail_stage_counts.get(r["final_stage"], 0) + 1

    return {
        "num_episodes": num_eps,
        "num_success": num_success,
        "success_rate": success_rate,
        "avg_success_steps": avg_success_steps,
        "fail_stage_counts": fail_stage_counts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--pause_on_fail", action="store_true")
    args = parser.parse_args()

    env = make_env(has_renderer=args.render)
    expert = TwoArmLiftScriptedExpert(env)

    results = []

    for ep in range(args.num_episodes):
        print(f"\n========== Episode {ep:02d} ==========")

        result = run_one_episode(
            env=env,
            expert=expert,
            max_steps=args.max_steps,
            render=args.render,
            verbose=args.verbose,
            pause_on_fail=args.pause_on_fail,
        )
        results.append(result)

        status = "SUCCESS" if result["success"] else "FAIL"
        print(
            f"[Episode {ep:02d}] {status} | "
            f"steps={result['steps']} | "
            f"final_stage={result['final_stage']} | "
            f"pot_z={result['final_pot_z']:.4f} | "
            f"rel0={result['final_rel0']:.4f} | "
            f"rel1={result['final_rel1']:.4f} | "
            f"stages={result['stage_history']} | "
            f"reasons={result['stage_reason_history']}"
        )

    env.close()

    summary = summarize(results)

    print("\n===== BENCHMARK SUMMARY =====")
    print(f"Episodes        : {summary['num_episodes']}")
    print(f"Successes       : {summary['num_success']}")
    print(f"Success Rate    : {summary['success_rate'] * 100:.2f}%")
    if summary["avg_success_steps"] is not None:
        print(f"Avg Success Step: {summary['avg_success_steps']:.2f}")
    else:
        print("Avg Success Step: N/A")
    print(f"Fail Stage Count: {summary['fail_stage_counts']}")


if __name__ == "__main__":
    main()
