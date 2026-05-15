from __future__ import annotations

from typing import Any

import numpy as np


class BaseNoise:
    """Base interface for action / observation noise modules."""

    def reset(self) -> None:
        return None

    def apply(
        self,
        action: np.ndarray,
        obs: Any = None,
        expert_info: dict | None = None,
        t: int | None = None,
    ) -> np.ndarray:
        raise NotImplementedError
