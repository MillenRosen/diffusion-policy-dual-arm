from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .base_noise import BaseNoise


@dataclass
class StageAwareActionNoise(BaseNoise):
    """
    Stage-aware Gaussian action noise.

    The stage value is expected in expert_info['stage'].
    Default mapping follows dual-arm scripted control:
        0: approach
        1: hover / align
        2: descend
        3: grasp hold
        4: lift
    """

    base_sigma: float = 0.03
    stage_scales: dict[int, float] = field(
        default_factory=lambda: {
            0: 1.25,
            1: 1.00,
            2: 0.70,
            3: 0.35,
            4: 0.45,
        }
    )
    clip_min: float = -1.0
    clip_max: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        return None

    def sigma_for_stage(self, stage: int | None) -> float:
        if stage is None:
            return self.base_sigma
        return float(self.base_sigma * self.stage_scales.get(int(stage), 1.0))

    def apply(self, action, obs=None, expert_info=None, t=None):
        action = np.asarray(action, dtype=np.float32)
        stage = None
        if expert_info is not None:
            stage = expert_info.get('stage', None)
        sigma = self.sigma_for_stage(stage)
        noise = self.rng.normal(0.0, sigma, size=action.shape).astype(np.float32)
        return np.clip(action + noise, self.clip_min, self.clip_max)
