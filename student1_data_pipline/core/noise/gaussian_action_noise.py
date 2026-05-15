from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base_noise import BaseNoise


@dataclass
class GaussianActionNoise(BaseNoise):
    """
    Add i.i.d. Gaussian noise in action space.

    Typical safe values for robosuite action vectors:
        sigma = 0.01 ~ 0.05
    """

    sigma: float = 0.03
    clip_min: float = -1.0
    clip_max: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        return None

    def apply(self, action, obs=None, expert_info=None, t=None):
        action = np.asarray(action, dtype=np.float32)
        noise = self.rng.normal(0.0, self.sigma, size=action.shape).astype(np.float32)
        return np.clip(action + noise, self.clip_min, self.clip_max)
