from __future__ import annotations

import json
from typing import Dict, Tuple

import robosuite as suite
from robosuite.controllers import load_composite_controller_config


def make_sequential_env(
    *,
    has_renderer: bool = False,
    renderer: str = "mjviewer",
    control_freq: int = 20,
    horizon: int = 600,
    reward_shaping: bool = True,
    use_object_obs: bool = True,
    use_camera_obs: bool = False,
    order_mode: str = "left_first",
) -> Tuple[object, Dict]:
    """
    Create a raw robosuite environment for sequential bimanual collection.

    We still use the official robosuite TwoArmLift environment so that the
    collected data remains fully compatible with the standard HDF5 replay path.
    The sequential behavior is implemented by the expert policy rather than by
    changing the simulator task itself.
    """
    if order_mode not in {"left_first", "right_first"}:
        raise ValueError("order_mode must be 'left_first' or 'right_first'")

    controller_config = load_composite_controller_config(
        controller=None,
        robot="Panda",
    )

    config = {
        "env_name": "TwoArmLift",
        "robots": ["Panda", "Panda"],
        "controller_configs": controller_config,
        "env_configuration": "parallel",
        "collection_mode": "sequential_bimanual",
        "order_mode": order_mode,
    }

    env = suite.make(
        env_name="TwoArmLift",
        robots=["Panda", "Panda"],
        controller_configs=controller_config,
        env_configuration="parallel",
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


def sequential_env_info_json(config: Dict) -> str:
    return json.dumps(config)
