"""Replay buffer for self-play training samples."""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, List, Tuple

import numpy as np
import torch

from .config import REPLAY_CAPACITY


Sample = Tuple[np.ndarray, np.ndarray, float]  # (state_planes, policy_target, value_target)


class ReplayBuffer:
	def __init__(self, capacity: int = REPLAY_CAPACITY) -> None:
		self.buffer: Deque[Sample] = deque(maxlen=capacity)

	def push_many(self, samples: List[Sample]) -> None:
		self.buffer.extend(samples)

	def __len__(self) -> int:
		return len(self.buffer)

	def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		batch = random.sample(self.buffer, batch_size)
		states = np.stack([b[0] for b in batch], axis=0)
		policies = np.stack([b[1] for b in batch], axis=0)
		values = np.array([b[2] for b in batch], dtype=np.float32)
		return (
			torch.from_numpy(states),
			torch.from_numpy(policies),
			torch.from_numpy(values),
		)
