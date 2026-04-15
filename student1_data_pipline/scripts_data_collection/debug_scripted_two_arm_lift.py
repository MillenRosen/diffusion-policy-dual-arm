import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config


class TwoArmLiftScriptedExpert:
    def __init__(self, env):
        self.env = env
        self.stage = 0
        self.stage_step = 0

        # 可调参数
        self.approach_z_offset = 0.08
        self.approach_gain = 4.0
        self.descend_gain = 3.0
        self.lift_gain = 2.0

        self.approach_thresh = 0.03
        self.descend_thresh = 0.015
        self.grasp_hold_steps = 20

    def reset(self):
        self.stage = 0
        self.stage_step = 0

    def _build_robot_action(self, robot, arm_delta, gripper_cmd):
        # robot.print_action_info_dict() 已经告诉你 arm 名字就是 "right"
        action_dict = {
            "right": arm_delta.astype(np.float32),
            "right_gripper": np.array([gripper_cmd], dtype=np.float32),
        }
        return robot.create_action_vector(action_dict)

    def act(self, obs):
        eef0 = obs["robot0_eef_pos"]
        eef1 = obs["robot1_eef_pos"]

        handle0 = obs["handle0_xpos"]
        handle1 = obs["handle1_xpos"]

        rel0 = obs["gripper0_to_handle0"]
        rel1 = obs["gripper1_to_handle1"]

        pot_pos = obs["pot_pos"]

        # 默认张开夹爪
        grip0 = -1.0
        grip1 = -1.0

        # Stage 0: 到把手上方
        if self.stage == 0:
            target0 = handle0 + np.array([0.0, 0.0, self.approach_z_offset])
            target1 = handle1 + np.array([0.0, 0.0, self.approach_z_offset])

            delta0 = self.approach_gain * (target0 - eef0)
            delta1 = self.approach_gain * (target1 - eef1)

            if np.linalg.norm(target0 - eef0) < self.approach_thresh and np.linalg.norm(target1 - eef1) < self.approach_thresh:
                self.stage = 1
                self.stage_step = 0
                print(">>> switch to stage 1: descend")

        # Stage 1: 垂直下降到把手附近
        elif self.stage == 1:
            target0 = handle0 + np.array([0.0, 0.0, -0.005])
            target1 = handle1 + np.array([0.0, 0.0, -0.005])

            err0 = target0 - eef0
            err1 = target1 - eef1

            delta0 = self.descend_gain * err0
            delta1 = self.descend_gain * err1

            if self.stage_step % 10 == 0:
                print(
                    f"[descend] dz0={handle0[2] - eef0[2]:.4f}, dz1={handle1[2] - eef1[2]:.4f}, "
                    f"norm0={np.linalg.norm(err0):.4f}, norm1={np.linalg.norm(err1):.4f}"
                )

            self.stage_step += 1

            if (
                    np.linalg.norm(err0) < 0.008
                    and np.linalg.norm(err1) < 0.008
                    and abs(err0[2]) < 0.006
                    and abs(err1[2]) < 0.006
            ):
                self.stage = 2
                self.stage_step = 0
                print(">>> switch to stage 2: close gripper")

        # Stage 2: 闭合夹爪，保持位置
        elif self.stage == 2:
            delta0 = np.zeros(6, dtype=np.float32)
            delta1 = np.zeros(6, dtype=np.float32)

            grip0 = 1.0
            grip1 = 1.0

            self.stage_step += 1
            if self.stage_step >= self.grasp_hold_steps:
                self.stage = 3
                self.stage_step = 0
                print(">>> switch to stage 3: lift")

        # Stage 3: 向上抬升，同时保持夹紧
        elif self.stage == 3:
            delta0 = np.zeros(6, dtype=np.float32)
            delta1 = np.zeros(6, dtype=np.float32)

            delta0[2] = 0.9
            delta1[2] = 0.9

            grip0 = 1.0
            grip1 = 1.0

        else:
            delta0 = np.zeros(6, dtype=np.float32)
            delta1 = np.zeros(6, dtype=np.float32)

        # 关键：只控制位置，不碰旋转，所以后 3 维给 0
        arm0 = np.zeros(6, dtype=np.float32)
        arm1 = np.zeros(6, dtype=np.float32)
        arm0[:3] = np.clip(delta0[:3], -1.0, 1.0)
        arm1[:3] = np.clip(delta1[:3], -1.0, 1.0)

        a0 = self._build_robot_action(self.env.robots[0], arm0, grip0)
        a1 = self._build_robot_action(self.env.robots[1], arm1, grip1)

        full_action = np.concatenate([a0, a1])

        return np.clip(full_action, -1.0, 1.0)


def main():
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
        has_renderer=True,
        renderer="mjviewer",
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=20,
    )

    expert = TwoArmLiftScriptedExpert(env)

    obs = env.reset()
    expert.reset()

    for t in range(600):
        action = expert.act(obs)
        obs, reward, done, info = env.step(action)
        env.render()

        if t % 20 == 0:
            print(
                f"t={t}, stage={expert.stage}, "
                f"pot_z={obs['pot_pos'][2]:.4f}, "
                f"rel0={np.linalg.norm(obs['gripper0_to_handle0']):.4f}, "
                f"rel1={np.linalg.norm(obs['gripper1_to_handle1']):.4f}, "
                f"success={env._check_success()}"
            )

        if env._check_success():
            print(f"SUCCESS at step {t}")
            break

    env.close()


if __name__ == "__main__":
    main()