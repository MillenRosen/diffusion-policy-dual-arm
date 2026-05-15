from __future__ import annotations

import json
from typing import Dict, Tuple

import robosuite as suite
from robosuite.controllers import load_composite_controller_config


def make_dual_arm_env(
    *,
    has_renderer: bool = False,
    renderer: str = "mjviewer",
    control_freq: int = 20,
    horizon: int = 600,
    reward_shaping: bool = True,
    use_object_obs: bool = True,
    use_camera_obs: bool = False,
) -> Tuple[object, Dict]:
    """
    Create a raw robosuite TwoArmLift environment.
    This is intentionally close to the official collect_scripted_demos style.
    """
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
        use_camera_obs=use_camera_obs,
        use_object_obs=use_object_obs,
        reward_shaping=reward_shaping,
        control_freq=control_freq,
        horizon=horizon,
    )
    return env, config


def dual_arm_env_info_json(config: Dict) -> str:
    return json.dumps(config)
