from __future__ import annotations

from typing import Any, Dict, Tuple

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


class SequentialExpert(BaseExpert):
    """
    Sequential bimanual expert with explicit rotation alignment.

    Stages:
        0: first arm approach hover
        1: first arm hover + rotate
        2: first arm descend
        3: first arm grasp hold
        4: inter-arm waiting gap
        5: second arm approach hover
        6: second arm hover + rotate
        7: second arm descend
        8: second arm grasp hold
        9: post-grasp waiting gap
        10: dual-arm lift
    """

    def __init__(self, env: Any, mode: str = "robust", order_mode: str = "left_first"):
        super().__init__(env)
        if mode not in {"strict", "robust"}:
            raise ValueError("mode must be 'strict' or 'robust'")
        if order_mode not in {"left_first", "right_first"}:
            raise ValueError("order_mode must be 'left_first' or 'right_first'")

        self.mode = mode
        self.order_mode = order_mode

        self.approach_z_offset = 0.085
        self.grasp_z_offset = -0.006

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
            self.hover_rotate_min_steps = 5
            self.hover_rotate_timeout_steps = 60

            self.descend_xy_thresh = 0.012
            self.descend_z_thresh = 0.009
            self.descend_loose_norm_thresh = 0.018
            self.descend_timeout_steps = 90
            self.grasp_hold_steps = 18
            self.inter_arm_delay_steps = 28
            self.post_grasp_delay_steps = 10
        else:
            self.approach_thresh = 0.040
            self.hover_xy_thresh = 0.020
            self.hover_z_thresh = 0.016
            self.rot_align_thresh = 0.16
            self.hover_rotate_min_steps = 4
            self.hover_rotate_timeout_steps = 35

            self.descend_xy_thresh = 0.018
            self.descend_z_thresh = 0.013
            self.descend_loose_norm_thresh = 0.026
            self.descend_timeout_steps = 70
            self.grasp_hold_steps = 14
            self.inter_arm_delay_steps = 22
            self.post_grasp_delay_steps = 8

    def on_reset(self) -> None:
        self.current_rot_cmd0 = np.zeros(3, dtype=np.float32)
        self.current_rot_cmd1 = np.zeros(3, dtype=np.float32)
        self.rot_control_active0 = True
        self.rot_control_active1 = True

    def _build_robot_action(self, robot: Any, arm_delta: np.ndarray, gripper_cmd: float) -> np.ndarray:
        action_dict = {
            "right": arm_delta.astype(np.float32),
            "right_gripper": np.array([gripper_cmd], dtype=np.float32),
        }
        return robot.create_action_vector(action_dict)

    def _robot_indices(self) -> Tuple[int, int]:
        return (0, 1) if self.order_mode == "left_first" else (1, 0)

    def _eef(self, obs: Dict[str, Any], robot_idx: int) -> np.ndarray:
        return np.asarray(obs[f"robot{robot_idx}_eef_pos"], dtype=np.float32)

    def _eef_quat(self, obs: Dict[str, Any], robot_idx: int) -> np.ndarray:
        return np.asarray(obs[f"robot{robot_idx}_eef_quat"], dtype=np.float32)

    def _handle(self, obs: Dict[str, Any], robot_idx: int) -> np.ndarray:
        return np.asarray(obs[f"handle{robot_idx}_xpos"], dtype=np.float32)

    def _xy_norm(self, err: np.ndarray) -> float:
        return float(np.linalg.norm(err[:2]))

    def _approach_delta(self, eef: np.ndarray, handle: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        target = handle + np.array([0.0, 0.0, self.approach_z_offset], dtype=np.float32)
        err = target - eef
        return (self.approach_gain * err).astype(np.float32), err

    def _descend_delta(self, eef: np.ndarray, handle: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

    def _descend_ready(self, err: np.ndarray, timeout_step: int) -> tuple[bool, str | None]:
        strict_ready = self._xy_norm(err) < self.descend_xy_thresh and abs(float(err[2])) < self.descend_z_thresh
        loose_ready = float(np.linalg.norm(err)) < self.descend_loose_norm_thresh
        timeout_ready = timeout_step >= self.descend_timeout_steps and float(np.linalg.norm(err)) < 0.030

        if strict_ready:
            return True, "descend_pose_ready"
        if loose_ready:
            return True, "descend_loose_ready"
        if timeout_ready:
            return True, "descend_timeout_ready"
        return False, None

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        first_idx, second_idx = self._robot_indices()

        eef_first = self._eef(obs, first_idx)
        eef_second = self._eef(obs, second_idx)
        handle_first = self._handle(obs, first_idx)
        handle_second = self._handle(obs, second_idx)

        delta_first = np.zeros(3, dtype=np.float32)
        delta_second = np.zeros(3, dtype=np.float32)
        rot_first = np.zeros(3, dtype=np.float32)
        rot_second = np.zeros(3, dtype=np.float32)
        grip_first = -1.0
        grip_second = -1.0

        if self.stage == 0:
            delta_first, err = self._approach_delta(eef_first, handle_first)
            if np.linalg.norm(err) < self.approach_thresh:
                self.stage = 1
                self.stage_step = 0
                self.last_stage_reason = "first_hover_ready"

        elif self.stage == 1:
            delta_first, err = self._approach_delta(eef_first, handle_first)
            raw_rot = self._compute_rot_cmd(obs, first_idx)
            rot_first = self._smooth_rot_cmd(first_idx, raw_rot)
            self.stage_step += 1

            hover_ready = self._xy_norm(err) < self.hover_xy_thresh and abs(float(err[2])) < self.hover_z_thresh
            rot_ready = abs(self._yaw_error(obs, first_idx)) < self.rot_align_thresh
            if hover_ready and rot_ready and self.stage_step >= self.hover_rotate_min_steps:
                self.stage = 2
                self.stage_step = 0
                self.last_stage_reason = "first_hover_rotation_ready"
                if first_idx == 0:
                    self.current_rot_cmd0[:] = 0.0
                    self.rot_control_active0 = False
                else:
                    self.current_rot_cmd1[:] = 0.0
                    self.rot_control_active1 = False
            elif hover_ready and self.stage_step >= self.hover_rotate_timeout_steps:
                self.stage = 2
                self.stage_step = 0
                self.last_stage_reason = "first_hover_rotation_timeout_ready"

        elif self.stage == 2:
            delta_first, err = self._descend_delta(eef_first, handle_first)
            self.stage_step += 1
            ready, reason = self._descend_ready(err, self.stage_step)
            if ready:
                self.stage = 3
                self.stage_step = 0
                self.last_stage_reason = f"first_{reason}"

        elif self.stage == 3:
            delta_first = self._stabilize_delta(eef_first, handle_first)
            grip_first = 1.0
            self.stage_step += 1
            if self.stage_step >= self.grasp_hold_steps:
                self.stage = 4
                self.stage_step = 0
                self.last_stage_reason = "first_grasp_hold_done"

        elif self.stage == 4:
            delta_first = self._stabilize_delta(eef_first, handle_first)
            grip_first = 1.0
            self.stage_step += 1
            if self.stage_step >= self.inter_arm_delay_steps:
                self.stage = 5
                self.stage_step = 0
                self.last_stage_reason = "inter_arm_gap_done"

        elif self.stage == 5:
            delta_first = self._stabilize_delta(eef_first, handle_first)
            grip_first = 1.0
            delta_second, err = self._approach_delta(eef_second, handle_second)
            if np.linalg.norm(err) < self.approach_thresh:
                self.stage = 6
                self.stage_step = 0
                self.last_stage_reason = "second_hover_ready"

        elif self.stage == 6:
            delta_first = self._stabilize_delta(eef_first, handle_first)
            grip_first = 1.0
            delta_second, err = self._approach_delta(eef_second, handle_second)
            raw_rot = self._compute_rot_cmd(obs, second_idx)
            rot_second = self._smooth_rot_cmd(second_idx, raw_cmd=raw_rot)
            self.stage_step += 1

            hover_ready = self._xy_norm(err) < self.hover_xy_thresh and abs(float(err[2])) < self.hover_z_thresh
            rot_ready = abs(self._yaw_error(obs, second_idx)) < self.rot_align_thresh
            if hover_ready and rot_ready and self.stage_step >= self.hover_rotate_min_steps:
                self.stage = 7
                self.stage_step = 0
                self.last_stage_reason = "second_hover_rotation_ready"
                if second_idx == 0:
                    self.current_rot_cmd0[:] = 0.0
                    self.rot_control_active0 = False
                else:
                    self.current_rot_cmd1[:] = 0.0
                    self.rot_control_active1 = False
            elif hover_ready and self.stage_step >= self.hover_rotate_timeout_steps:
                self.stage = 7
                self.stage_step = 0
                self.last_stage_reason = "second_hover_rotation_timeout_ready"

        elif self.stage == 7:
            delta_first = self._stabilize_delta(eef_first, handle_first)
            grip_first = 1.0
            delta_second, err = self._descend_delta(eef_second, handle_second)
            self.stage_step += 1
            ready, reason = self._descend_ready(err, self.stage_step)
            if ready:
                self.stage = 8
                self.stage_step = 0
                self.last_stage_reason = f"second_{reason}"

        elif self.stage == 8:
            delta_first = self._stabilize_delta(eef_first, handle_first)
            grip_first = 1.0
            delta_second = self._stabilize_delta(eef_second, handle_second)
            grip_second = 1.0
            self.stage_step += 1
            if self.stage_step >= self.grasp_hold_steps:
                self.stage = 9
                self.stage_step = 0
                self.last_stage_reason = "second_grasp_hold_done"

        elif self.stage == 9:
            delta_first = self._stabilize_delta(eef_first, handle_first)
            grip_first = 1.0
            delta_second = self._stabilize_delta(eef_second, handle_second)
            grip_second = 1.0
            self.stage_step += 1
            if self.stage_step >= self.post_grasp_delay_steps:
                self.stage = 10
                self.stage_step = 0
                self.last_stage_reason = "post_grasp_gap_done"

        elif self.stage == 10:
            delta_first = np.array([0.0, 0.0, self.lift_cmd], dtype=np.float32)
            delta_second = np.array([0.0, 0.0, self.lift_cmd], dtype=np.float32)
            grip_first = 1.0
            grip_second = 1.0

        arm_first = np.zeros(6, dtype=np.float32)
        arm_second = np.zeros(6, dtype=np.float32)
        arm_first[:3] = np.clip(delta_first, -1.0, 1.0)
        arm_second[:3] = np.clip(delta_second, -1.0, 1.0)
        arm_first[3:] = np.clip(rot_first, -1.0, 1.0)
        arm_second[3:] = np.clip(rot_second, -1.0, 1.0)

        base_env = self.env
        if hasattr(base_env, "env") and hasattr(base_env.env, "robots"):
            robot_owner = base_env.env
        else:
            robot_owner = base_env

        actions_by_robot = [None, None]
        actions_by_robot[first_idx] = self._build_robot_action(robot_owner.robots[first_idx], arm_first, grip_first)
        actions_by_robot[second_idx] = self._build_robot_action(robot_owner.robots[second_idx], arm_second, grip_second)
        return np.clip(np.concatenate(actions_by_robot), -1.0, 1.0)

    def info(self) -> Dict[str, Any]:
        out = super().info()
        out.update({
            "mode": self.mode,
            "order_mode": self.order_mode,
            "inter_arm_delay_steps": int(self.inter_arm_delay_steps),
            "post_grasp_delay_steps": int(self.post_grasp_delay_steps),
        })
        return out
