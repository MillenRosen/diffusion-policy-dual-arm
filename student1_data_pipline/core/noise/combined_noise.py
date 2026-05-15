from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base_noise import BaseNoise


@dataclass
class CombinedNoise(BaseNoise):
    """Compose multiple noise modules sequentially."""

    noises: list[BaseNoise] = field(default_factory=list)

    def reset(self) -> None:
        for noise in self.noises:
            noise.reset()

    def apply(self, action, obs=None, expert_info=None, t=None):
        out = np.asarray(action, dtype=np.float32)
        for noise in self.noises:
            out = noise.apply(out, obs=obs, expert_info=expert_info, t=t)
        return out
