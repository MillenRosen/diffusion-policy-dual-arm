from __future__ import annotations

from typing import Any, Dict


class BaseExpert:
    """
    Minimal base class for scripted experts.

    The collection pipeline only relies on three methods:
    - reset()
    - act(obs)
    - info()
    """

    def __init__(self, env: Any):
        self.env = env
        self.stage = 0
        self.stage_step = 0
        self.last_stage_reason = "init"

    def reset(self) -> None:
        self.stage = 0
        self.stage_step = 0
        self.last_stage_reason = "reset"
        self.on_reset()

    def on_reset(self) -> None:
        """Hook for subclasses."""
        return None

    def act(self, obs: Dict[str, Any]):
        raise NotImplementedError

    def info(self) -> Dict[str, Any]:
        return {
            "stage": int(self.stage),
            "stage_step": int(self.stage_step),
            "last_stage_reason": str(self.last_stage_reason),
        }

    @staticmethod
    def _clip_arm_action(delta_xyz, rot_xyz):
        import numpy as np

        arm = np.zeros(6, dtype=np.float32)
        arm[:3] = np.clip(delta_xyz, -1.0, 1.0)
        arm[3:] = np.clip(rot_xyz, -1.0, 1.0)
        return arm

    @staticmethod
    def _xy_norm(err) -> float:
        import numpy as np

        return float(np.linalg.norm(err[:2]))
