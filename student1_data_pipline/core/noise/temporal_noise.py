from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base_noise import BaseNoise


@dataclass
class TemporalCorrelatedActionNoise(BaseNoise):
    """
    First-order temporally correlated action noise.

    Implements an AR(1)-like process:
        z_t = lambda * z_{t-1} + (1 - lambda) * eps_t
        a'_t = a_t + z_t
    """

    sigma: float = 0.02
    smoothing: float = 0.75
    clip_min: float = -1.0
    clip_max: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.smoothing < 1.0:
            raise ValueError('smoothing must be in [0, 1).')
        self.rng = np.random.default_rng(self.seed)
        self._state: np.ndarray | None = None

    def reset(self) -> None:
        self._state = None

    def apply(self, action, obs=None, expert_info=None, t=None):
        action = np.asarray(action, dtype=np.float32)
        if self._state is None or self._state.shape != action.shape:
            self._state = np.zeros_like(action, dtype=np.float32)

        eps = self.rng.normal(0.0, self.sigma, size=action.shape).astype(np.float32)
        self._state = self.smoothing * self._state + (1.0 - self.smoothing) * eps
        return np.clip(action + self._state, self.clip_min, self.clip_max)
