import robosuite as suite
import numpy as np
from robosuite.controllers import load_composite_controller_config

# 加载控制器（新版接口）
controller_config = load_composite_controller_config(
    controller=None,
    robot="Panda",
)

# 创建环境
env = suite.make(
    env_name="TwoArmLift",
    robots=["Panda", "Panda"],
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
)

obs = env.reset()

print("\n===== OBS KEYS =====")
for k in sorted(obs.keys()):
    print(k)

print("\n===== ACTION SPEC =====")
low, high = env.action_spec
print("action dim:", low.shape[0])

input("\nPress Enter to exit...")