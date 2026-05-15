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


def shortest_perpendicular_yaw_error(target_yaw: float, current_yaw: float) -> float:
    err_pos = wrap_to_pi(target_yaw + np.pi / 2.0 - current_yaw)
    err_neg = wrap_to_pi(target_yaw - np.pi / 2.0 - current_yaw)
    return float(err_pos if abs(err_pos) <= abs(err_neg) else err_neg)


class DualArmExpert(BaseExpert):
    """
    Self-contained dual-arm expert with explicit rotation alignment.

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

        self.approach_z_offset = 0.085
        self.grasp_z_offset = -0.015

        self.approach_gain = 4.0
        self.hover_align_gain = 3.0
        self.descend_gain_xy = 2.5
        self.descend_gain_z = 3.6
        self.grasp_stabilize_gain = 1.6
        self.lift_cmd = 0.88

        self.rot_gain = 1.6
        self.rot_clip = 0.30
        self.rot_cmd_sign = -1.0
        self.yaw_deadzone = 0.045
        self.yaw_reactivate_thresh = 0.075
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
            self.hover_rotate_min_steps = 6
            self.hover_rotate_timeout_steps = 60

            self.descend_xy_thresh = 0.012
            self.descend_z_thresh = 0.010
            self.descend_loose_norm_thresh = 0.018
            self.descend_timeout_steps = 95
            self.grasp_hold_steps = 18
        else:
            self.approach_thresh = 0.040
            self.hover_xy_thresh = 0.020
            self.hover_z_thresh = 0.016
            self.rot_align_thresh = 0.16
            self.hover_rotate_min_steps = 4
            self.hover_rotate_timeout_steps = 35

            self.descend_xy_thresh = 0.018
            self.descend_z_thresh = 0.015
            self.descend_loose_norm_thresh = 0.026
            self.descend_timeout_steps = 72
            self.grasp_hold_steps = 14

    def on_reset(self) -> None:
        self.current_rot_cmd0 = np.zeros(3, dtype=np.float32)
        self.current_rot_cmd1 = np.zeros(3, dtype=np.float32)
        self.rot_control_active0 = True
        self.rot_control_active1 = True

    def _unwrap_robot_owner(self) -> Any:
        base_env = self.env
        if hasattr(base_env, "env") and hasattr(base_env.env, "robots"):
            return base_env.env
        return base_env

    def _build_robot_action(self, robot: Any, arm_delta: np.ndarray, gripper_cmd: float) -> np.ndarray:
        action_dict = {
            "right": arm_delta.astype(np.float32),
            "right_gripper": np.array([gripper_cmd], dtype=np.float32),
        }
        return robot.create_action_vector(action_dict)

    def _eef(self, obs: Dict[str, Any], robot_idx: int) -> np.ndarray:
        return np.asarray(obs[f"robot{robot_idx}_eef_pos"], dtype=np.float32)

    def _eef_quat(self, obs: Dict[str, Any], robot_idx: int) -> np.ndarray:
        return np.asarray(obs[f"robot{robot_idx}_eef_quat"], dtype=np.float32)

    def _handle(self, obs: Dict[str, Any], robot_idx: int) -> np.ndarray:
        return np.asarray(obs[f"handle{robot_idx}_xpos"], dtype=np.float32)

    def _xy_norm(self, err: np.ndarray) -> float:
        return float(np.linalg.norm(err[:2]))

    def _clip_arm_action(self, delta_xyz: np.ndarray, rot_xyz: np.ndarray) -> np.ndarray:
        arm = np.zeros(6, dtype=np.float32)
        arm[:3] = np.clip(np.asarray(delta_xyz, dtype=np.float32), -1.0, 1.0)
        arm[3:] = np.clip(np.asarray(rot_xyz, dtype=np.float32), -1.0, 1.0)
        return arm

    def _approach_delta(self, eef: np.ndarray, handle: np.ndarray, gain: float | None = None):
        target = handle + np.array([0.0, 0.0, self.approach_z_offset], dtype=np.float32)
        err = target - eef
        g = self.approach_gain if gain is None else gain
        return (g * err).astype(np.float32), err

    def _hover_delta(self, eef: np.ndarray, handle: np.ndarray):
        return self._approach_delta(eef, handle, gain=self.hover_align_gain)

    def _descend_delta(self, eef: np.ndarray, handle: np.ndarray):
        target = handle + np.array([0.0, 0.0, self.grasp_z_offset], dtype=np.float32)
        err = target - eef
        delta = np.array([
            self.descend_gain_xy * err[0],
            self.descend_gain_xy * err[1],
            self.descend_gain_z * err[2],
        ], dtype=np.float32)
        return delta, err

    def _stabilize_delta(self, eef: np.ndarray, handle: np.ndarray) -> np.ndarray:
        target = handle + np.array([0.0, 0.0, self.grasp_z_offset], dtype=np.float32)
        err = target - eef
        return (self.grasp_stabilize_gain * err).astype(np.float32)

    def _yaw_error(self, obs: Dict[str, Any], robot_idx: int) -> float:
        pot_yaw = quat_to_yaw(np.asarray(obs["pot_quat"], dtype=np.float32))
        eef_yaw = quat_to_yaw(self._eef_quat(obs, robot_idx))
        return shortest_perpendicular_yaw_error(pot_yaw, eef_yaw)

    def _compute_rot_cmd(self, obs: Dict[str, Any], robot_idx: int) -> np.ndarray:
        yaw_err = self._yaw_error(obs, robot_idx)
        active = self.rot_control_active0 if robot_idx == 0 else self.rot_control_active1

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
        rot_cmd[2] = np.clip(self.rot_cmd_sign * self.rot_gain * yaw_err, -self.rot_clip, self.rot_clip)
        return rot_cmd

    def _smooth_rot_cmd(self, robot_idx: int, raw_cmd: np.ndarray) -> np.ndarray:
        current = self.current_rot_cmd0 if robot_idx == 0 else self.current_rot_cmd1
        smoothed = self.rot_decay * current + (1.0 - self.rot_decay) * raw_cmd
        out = np.zeros(3, dtype=np.float32)
        delta_z = np.clip(smoothed[2] - current[2], -self.max_rot_delta_per_step, self.max_rot_delta_per_step)
        out[2] = current[2] + delta_z
        if abs(out[2]) < self.filtered_rot_deadzone:
            out[2] = 0.0
        if robot_idx == 0:
            self.current_rot_cmd0 = out.copy()
        else:
            self.current_rot_cmd1 = out.copy()
        return out

    def _clear_rot_state(self, robot_idx: int) -> None:
        if robot_idx == 0:
            self.current_rot_cmd0[:] = 0.0
            self.rot_control_active0 = False
        else:
            self.current_rot_cmd1[:] = 0.0
            self.rot_control_active1 = False

    def _descend_ready_pair(self, err0: np.ndarray, err1: np.ndarray, timeout_step: int):
        strict_ready = (
            self._xy_norm(err0) < self.descend_xy_thresh
            and self._xy_norm(err1) < self.descend_xy_thresh
            and abs(float(err0[2])) < self.descend_z_thresh
            and abs(float(err1[2])) < self.descend_z_thresh
        )
        loose_ready = (
            float(np.linalg.norm(err0)) < self.descend_loose_norm_thresh
            and float(np.linalg.norm(err1)) < self.descend_loose_norm_thresh
        )
        timeout_ready = (
            timeout_step >= self.descend_timeout_steps
            and float(np.linalg.norm(err0)) < 0.030
            and float(np.linalg.norm(err1)) < 0.030
        )

        if strict_ready:
            return True, "descend_pose_ready"
        if loose_ready:
            return True, "descend_loose_ready"
        if timeout_ready:
            return True, "descend_timeout_ready"
        return False, None

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        eef0 = self._eef(obs, 0)
        eef1 = self._eef(obs, 1)
        handle0 = self._handle(obs, 0)
        handle1 = self._handle(obs, 1)

        delta0 = np.zeros(3, dtype=np.float32)
        delta1 = np.zeros(3, dtype=np.float32)
        rot0 = np.zeros(3, dtype=np.float32)
        rot1 = np.zeros(3, dtype=np.float32)
        grip0 = -1.0
        grip1 = -1.0

        if self.stage == 0:
            delta0, err0 = self._approach_delta(eef0, handle0)
            delta1, err1 = self._approach_delta(eef1, handle1)
            if np.linalg.norm(err0) < self.approach_thresh and np.linalg.norm(err1) < self.approach_thresh:
                self.stage = 1
                self.stage_step = 0
                self.last_stage_reason = "hover_pose_ready"

        elif self.stage == 1:
            delta0, err0 = self._hover_delta(eef0, handle0)
            delta1, err1 = self._hover_delta(eef1, handle1)
            raw_rot0 = self._compute_rot_cmd(obs, 0)
            raw_rot1 = self._compute_rot_cmd(obs, 1)
            rot0 = self._smooth_rot_cmd(0, raw_rot0)
            rot1 = self._smooth_rot_cmd(1, raw_rot1)
            self.stage_step += 1

            hover_ready = (
                self._xy_norm(err0) < self.hover_xy_thresh
                and self._xy_norm(err1) < self.hover_xy_thresh
                and abs(float(err0[2])) < self.hover_z_thresh
                and abs(float(err1[2])) < self.hover_z_thresh
            )
            rot_ready = abs(self._yaw_error(obs, 0)) < self.rot_align_thresh and abs(self._yaw_error(obs, 1)) < self.rot_align_thresh

            if hover_ready and rot_ready and self.stage_step >= self.hover_rotate_min_steps:
                self._clear_rot_state(0)
                self._clear_rot_state(1)
                self.stage = 2
                self.stage_step = 0
                self.last_stage_reason = "hover_rotation_ready"
            elif hover_ready and self.stage_step >= self.hover_rotate_timeout_steps:
                self._clear_rot_state(0)
                self._clear_rot_state(1)
                self.stage = 2
                self.stage_step = 0
                self.last_stage_reason = "hover_rotation_timeout_ready"

        elif self.stage == 2:
            delta0, err0 = self._descend_delta(eef0, handle0)
            delta1, err1 = self._descend_delta(eef1, handle1)
            self.stage_step += 1
            ready, reason = self._descend_ready_pair(err0, err1, self.stage_step)
            if ready:
                self.stage = 3
                self.stage_step = 0
                self.last_stage_reason = reason or "descend_ready"

        elif self.stage == 3:
            delta0 = self._stabilize_delta(eef0, handle0)
            delta1 = self._stabilize_delta(eef1, handle1)
            grip0 = 1.0
            grip1 = 1.0
            self.stage_step += 1
            if self.stage_step >= self.grasp_hold_steps:
                self.stage = 4
                self.stage_step = 0
                self.last_stage_reason = "grasp_hold_done"

        elif self.stage == 4:
            delta0 = np.array([0.0, 0.0, self.lift_cmd], dtype=np.float32)
            delta1 = np.array([0.0, 0.0, self.lift_cmd], dtype=np.float32)
            if self.mode == "robust":
                delta0[0] += 0.01
                delta1[0] -= 0.01
            grip0 = 1.0
            grip1 = 1.0

        arm0 = self._clip_arm_action(delta0, rot0)
        arm1 = self._clip_arm_action(delta1, rot1)

        robot_owner = self._unwrap_robot_owner()
        a0 = self._build_robot_action(robot_owner.robots[0], arm0, grip0)
        a1 = self._build_robot_action(robot_owner.robots[1], arm1, grip1)
        return np.clip(np.concatenate([a0, a1]), -1.0, 1.0)

    def info(self) -> Dict[str, Any]:
        out = super().info()
        out.update({"mode": self.mode})
        return out
