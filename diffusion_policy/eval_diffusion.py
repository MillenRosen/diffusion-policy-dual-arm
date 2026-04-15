from __future__ import annotations

from collections import deque

import numpy as np
import torch


class DiffusionPolicy:
    def __init__(
        self,
        model,
        scheduler,
        obs_dim,
        act_dim,
        action_chunk,
        obs_horizon,
        norm_stats,
        device,
    ):
        self.model = model
        self.scheduler = scheduler
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_chunk = action_chunk
        self.obs_horizon = obs_horizon
        self.device = device
        self.obs_mean = norm_stats["obs_mean"].astype(np.float32)
        self.obs_std = norm_stats["obs_std"].astype(np.float32)
        self.act_mean = norm_stats["act_mean"].astype(np.float32)
        self.act_std = norm_stats["act_std"].astype(np.float32)
        self.reset()

    def reset(self):
        self.obs_queue = deque(maxlen=self.obs_horizon)
        self.action_queue = deque()

    def _normalize_obs(self, obs):
        return (obs - self.obs_mean) / self.obs_std

    def _denormalize_act(self, act):
        return act * self.act_std + self.act_mean

    def _prepare_obs_condition(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        self.obs_queue.append(obs)
        while len(self.obs_queue) < self.obs_horizon:
            self.obs_queue.appendleft(obs)
        stacked = np.stack(list(self.obs_queue), axis=0)
        norm_stacked = self._normalize_obs(stacked)
        return norm_stacked.reshape(1, -1)

    @torch.no_grad()
    def act(self, obs, return_chunk=False):
        if not return_chunk and self.action_queue:
            return self.action_queue.popleft()

        obs_cond = self._prepare_obs_condition(obs)
        obs_tensor = torch.from_numpy(obs_cond).float().to(self.device)
        pred = self.scheduler.sample(
            self.model,
            obs_tensor,
            action_shape=(self.action_chunk, self.act_dim),
        )
        pred = pred.squeeze(0).cpu().numpy()
        pred = self._denormalize_act(pred)
        pred = np.clip(pred, -1.0, 1.0).astype(np.float32)

        if return_chunk:
            return pred

        self.action_queue.extend(pred.tolist())
        return np.asarray(self.action_queue.popleft(), dtype=np.float32)

