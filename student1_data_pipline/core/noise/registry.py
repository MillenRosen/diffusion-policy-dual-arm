from __future__ import annotations

from typing import Any

from .combined_noise import CombinedNoise
from .gaussian_action_noise import GaussianActionNoise
from .temporal_noise import TemporalCorrelatedActionNoise
from .stage_aware_noise import StageAwareActionNoise


NOISE_TYPES = {
    'gaussian': GaussianActionNoise,
    'temporal': TemporalCorrelatedActionNoise,
    'stage_aware': StageAwareActionNoise,
}


def build_noise(noise_type: str = 'gaussian', **kwargs: Any):
    noise_type = noise_type.lower()
    if noise_type in ('none', 'null'):
        return None
    if noise_type not in NOISE_TYPES:
        raise ValueError(f'Unknown noise_type: {noise_type}')
    return NOISE_TYPES[noise_type](**kwargs)


def build_composite_noise(
    use_gaussian: bool = True,
    use_temporal: bool = False,
    use_stage_aware: bool = False,
    gaussian_kwargs: dict[str, Any] | None = None,
    temporal_kwargs: dict[str, Any] | None = None,
    stage_aware_kwargs: dict[str, Any] | None = None,
):
    noises = []
    if use_gaussian:
        noises.append(GaussianActionNoise(**(gaussian_kwargs or {})))
    if use_temporal:
        noises.append(TemporalCorrelatedActionNoise(**(temporal_kwargs or {})))
    if use_stage_aware:
        noises.append(StageAwareActionNoise(**(stage_aware_kwargs or {})))

    if not noises:
        return None
    if len(noises) == 1:
        return noises[0]
    return CombinedNoise(noises=noises)
