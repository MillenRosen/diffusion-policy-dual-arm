import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config


def main():
    # 跟官方 collect_human_demonstrations.py 保持一致的 controller 加载方式
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

    obs = env.reset()

    print("\n===== OBS KEYS =====")
    print(sorted(list(obs.keys())))

    print("\n===== OBS SHAPES / SAMPLE VALUES =====")
    for k in sorted(obs.keys()):
        v = obs[k]
        if isinstance(v, np.ndarray):
            preview = v if v.ndim == 0 else v[: min(5, len(v))]
            print(f"{k}: shape={v.shape}, preview={preview}")
        else:
            print(f"{k}: type={type(v)}, value={v}")

    print("\n===== ACTION INFO =====")
    for i, robot in enumerate(env.robots):
        print(f"\n--- Robot {i} ---")
        robot.print_action_info_dict()

    print("\n===== ACTION SPEC =====")
    low, high = env.action_spec
    print("low shape:", low.shape)
    print("high shape:", high.shape)
    print("action dim:", low.shape[0])

    print("\n===== SUCCESS CHECK =====")
    print("check_success:", env._check_success())

    input("\nPress Enter to close...")
    env.close()


if __name__ == "__main__":
    main()