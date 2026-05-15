from __future__ import annotations

from typing import Any, Dict

import numpy as np
import robosuite.utils.transform_utils as T

from .base_expert import BaseExpert


def wrap_to_pi(x: float) -> float:
    return float((x + np.pi) % (2 * np.pi) - np.pi)


def quat_to_yaw(quat: np.ndarray) -> float:
    mat = T.quat2mat(quat)
    return float(np.arctan2(mat[1, 0], mat[0, 0]))


class SingleArmExpert(BaseExpert):
    """
    Single-arm Lift expert with yaw alignment.

    Stages:
        0: approach hover
        1: hover + rotate
        2: descend
        3: grasp hold
        4: lift
    """

    def __init__(self, env: Any, mode: str = "robust"):
        super().__init__(env)
        if mode not in {"strict", "robust"}:
            raise ValueError("mode must be 'strict' or 'robust'")
        self.mode = mode

        self.approach_z_offset = 0.10
        self.grasp_z_offset = -0.01

        self.approach_gain = 4.5
        self.hover_align_gain = 3.2
        self.descend_gain_xy = 2.8
        self.descend_gain_z = 4.0
        self.grasp_stabilize_gain = 1.8
        self.lift_cmd = 0.85

        self.rot_gain = 1.4
        self.rot_clip = 0.28
        self.rot_cmd_sign = -1.0
        self.yaw_deadzone = 0.04
        self.yaw_reactivate_thresh = 0.07
        self.filtered_rot_deadzone = 0.012
        self.rot_decay = 0.90
        self.max_rot_delta_per_step = 0.05

        self._set_mode_params(mode)
        self.on_reset()

    def _set_mode_params(self, mode: str) -> None:
        if mode == "strict":
            self.approach_thresh = 0.030
            self.hover_xy_thresh = 0.015
            self.hover_z_thresh = 0.012
            self.rot_align_thresh = 0.10
            self.hover_rotate_min_steps = 5
            self.hover_rotate_timeout_steps = 60

            self.descend_xy_thresh = 0.012
            self.descend_z_thresh = 0.012
            self.descend_loose_norm_thresh = 0.020
            self.descend_timeout_steps = 80
            self.grasp_hold_steps = 16
        else:
            self.approach_thresh = 0.040
            self.hover_xy_thresh = 0.020
            self.hover_z_thresh = 0.016
            self.rot_align_thresh = 0.16
            self.hover_rotate_min_steps = 4
            self.hover_rotate_timeout_steps = 40

            self.descend_xy_thresh = 0.018
            self.descend_z_thresh = 0.016
            self.descend_loose_norm_thresh = 0.028
            self.descend_timeout_steps = 55
            self.grasp_hold_steps = 12

    def on_reset(self) -> None:
        self.current_rot_cmd = np.zeros(3, dtype=np.float32)
        self.rot_control_active = True

    def _get_cube_pos(self, obs: Dict[str, Any]) -> np.ndarray:
        if "cube_pos" in obs:
            return np.asarray(obs["cube_pos"], dtype=np.float32)
        if "cubeA_pos" in obs:
            return np.asarray(obs["cubeA_pos"], dtype=np.float32)
        if "object-state" in obs:
            arr = np.asarray(obs["object-state"], dtype=np.float32).reshape(-1)
            if arr.size >= 3:
                return arr[:3]
        raise KeyError("Cannot find cube position in observation.")

    def _get_cube_quat(self, obs: Dict[str, Any]) -> np.ndarray:
        for key in ("cube_quat", "cubeA_quat"):
            if key in obs:
                return np.asarray(obs[key], dtype=np.float32)
        if "object-state" in obs:
            arr = np.asarray(obs["object-state"], dtype=np.float32).reshape(-1)
            if arr.size >= 7:
                quat = arr[3:7]
                if np.linalg.norm(quat) > 1e-6:
                    return quat / np.linalg.norm(quat)
        # fallback to identity rotation
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def _get_eef_pos(self, obs: Dict[str, Any]) -> np.ndarray:
        return np.asarray(obs["robot0_eef_pos"], dtype=np.float32)

    def _get_eef_quat(self, obs: Dict[str, Any]) -> np.ndarray:
        return np.asarray(obs["robot0_eef_quat"], dtype=np.float32)

    def _build_robot_action(self, robot: Any, arm_delta: np.ndarray, gripper_cmd: float) -> np.ndarray:
        action_dict = {
            "right": arm_delta.astype(np.float32),
            "right_gripper": np.array([gripper_cmd], dtype=np.float32),
        }
        return robot.create_action_vector(action_dict)

    def _yaw_error(self, obs: Dict[str, Any]) -> float:
        cube_yaw = quat_to_yaw(self._get_cube_quat(obs))
        eef_yaw = quat_to_yaw(self._get_eef_quat(obs))
        return wrap_to_pi(cube_yaw - eef_yaw)

    def _compute_rot_cmd(self, obs: Dict[str, Any]) -> np.ndarray:
        yaw_err = self._yaw_error(obs)
        if self.rot_control_active:
            if abs(yaw_err) < self.yaw_deadzone:
                self.rot_control_active = False
        else:
            if abs(yaw_err) > self.yaw_reactivate_thresh:
                self.rot_control_active = True

        if not self.rot_control_active:
            return np.zeros(3, dtype=np.float32)

        rot_cmd = np.zeros(3, dtype=np.float32)
        rot_cmd[2] = np.clip(
            self.rot_cmd_sign * self.rot_gain * yaw_err,
            -self.rot_clip,
            self.rot_clip,
        )
        return rot_cmd

    def _smooth_rot_cmd(self, raw_cmd: np.ndarray) -> np.ndarray:
        smoothed = self.rot_decay * self.current_rot_cmd + (1.0 - self.rot_decay) * raw_cmd
        out = np.zeros(3, dtype=np.float32)
        delta_z = np.clip(smoothed[2] - self.current_rot_cmd[2], -self.max_rot_delta_per_step, self.max_rot_delta_per_step)
        out[2] = self.current_rot_cmd[2] + delta_z
        if abs(out[2]) < self.filtered_rot_deadzone:
            out[2] = 0.0
        self.current_rot_cmd = out.copy()
        return out

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        eef = self._get_eef_pos(obs)
        cube = self._get_cube_pos(obs)

        grip = -1.0
        delta_xyz = np.zeros(3, dtype=np.float32)
        rot = np.zeros(3, dtype=np.float32)

        if self.stage == 0:
            target = cube + np.array([0.0, 0.0, self.approach_z_offset], dtype=np.float32)
            err = target - eef
            delta_xyz = self.approach_gain * err
            if np.linalg.norm(err) < self.approach_thresh:
                self.stage = 1
                self.stage_step = 0
                self.last_stage_reason = "hover_pose_ready"

        elif self.stage == 1:
            target = cube + np.array([0.0, 0.0, self.approach_z_offset], dtype=np.float32)
            err = target - eef
            delta_xyz = self.hover_align_gain * err
            raw_rot = self._compute_rot_cmd(obs)
            rot = self._smooth_rot_cmd(raw_rot)
            self.stage_step += 1

            hover_ready = self._xy_norm(err) < self.hover_xy_thresh and abs(float(err[2])) < self.hover_z_thresh
            rot_ready = abs(self._yaw_error(obs)) < self.rot_align_thresh

            if hover_ready and rot_ready and self.stage_step >= self.hover_rotate_min_steps:
                self.current_rot_cmd = np.zeros(3, dtype=np.float32)
                self.rot_control_active = False
                self.stage = 2
                self.stage_step = 0
                self.last_stage_reason = "hover_rotation_ready"
            elif hover_ready and self.stage_step >= self.hover_rotate_timeout_steps:
                self.current_rot_cmd = np.zeros(3, dtype=np.float32)
                self.rot_control_active = False
                self.stage = 2
                self.stage_step = 0
                self.last_stage_reason = "hover_rotation_timeout_ready"

        elif self.stage == 2:
            target = cube + np.array([0.0, 0.0, self.grasp_z_offset], dtype=np.float32)
            err = target - eef
            delta_xyz = np.array([
                self.descend_gain_xy * err[0],
                self.descend_gain_xy * err[1],
                self.descend_gain_z * err[2],
            ], dtype=np.float32)
            self.stage_step += 1

            strict_ready = self._xy_norm(err) < self.descend_xy_thresh and abs(float(err[2])) < self.descend_z_thresh
            loose_ready = float(np.linalg.norm(err)) < self.descend_loose_norm_thresh
            timeout_ready = self.stage_step >= self.descend_timeout_steps and float(np.linalg.norm(err)) < 0.03

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
            target = cube + np.array([0.0, 0.0, self.grasp_z_offset], dtype=np.float32)
            err = target - eef
            delta_xyz = self.grasp_stabilize_gain * err
            grip = 1.0
            self.stage_step += 1
            if self.stage_step >= self.grasp_hold_steps:
                self.stage = 4
                self.stage_step = 0
                self.last_stage_reason = "grasp_hold_done"

        elif self.stage == 4:
            delta_xyz = np.array([0.0, 0.0, self.lift_cmd], dtype=np.float32)
            grip = 1.0

        arm = np.zeros(6, dtype=np.float32)
        arm[:3] = np.clip(delta_xyz, -1.0, 1.0)
        arm[3:] = np.clip(rot, -1.0, 1.0)

        base_env = self.env
        if hasattr(base_env, "env") and hasattr(base_env.env, "robots"):
            robot_owner = base_env.env
        else:
            robot_owner = base_env

        action = self._build_robot_action(robot_owner.robots[0], arm, grip)
        return np.clip(action, -1.0, 1.0)

    def _xy_norm(self, err: np.ndarray) -> float:
        return float(np.linalg.norm(err[:2]))

    def info(self) -> Dict[str, Any]:
        out = super().info()
        out.update({
            "mode": self.mode,
            "yaw_error": None,
        })
        return out
