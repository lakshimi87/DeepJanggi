"""Residual policy/value network used by the MCTS agent."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
	COLS,
	INPUT_PLANES,
	NUM_FILTERS,
	NUM_RES_BLOCKS,
	ROWS,
	TOTAL_ACTIONS,
)


class ResidualBlock(nn.Module):
	def __init__(self, channels: int) -> None:
		super().__init__()
		self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(channels)
		self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(channels)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		identity = x
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out = out + identity
		return F.relu(out)


class JanggiNet(nn.Module):
	"""Simple AlphaZero-style network with policy and value heads."""

	def __init__(
		self,
		input_planes: int = INPUT_PLANES,
		filters: int = NUM_FILTERS,
		blocks: int = NUM_RES_BLOCKS,
		num_actions: int = TOTAL_ACTIONS,
	) -> None:
		super().__init__()
		self.stem = nn.Sequential(
			nn.Conv2d(input_planes, filters, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(filters),
			nn.ReLU(inplace=True),
		)
		self.body = nn.Sequential(*[ResidualBlock(filters) for _ in range(blocks)])

		# Policy head: 1x1 conv + flatten + linear to action logits.
		self.policy_conv = nn.Sequential(
			nn.Conv2d(filters, 32, kernel_size=1, bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
		)
		self.policy_fc = nn.Linear(32 * ROWS * COLS, num_actions)

		# Value head: 1x1 conv + linear bottleneck + scalar.
		self.value_conv = nn.Sequential(
			nn.Conv2d(filters, 16, kernel_size=1, bias=False),
			nn.BatchNorm2d(16),
			nn.ReLU(inplace=True),
		)
		self.value_fc = nn.Sequential(
			nn.Linear(16 * ROWS * COLS, 128),
			nn.ReLU(inplace=True),
			nn.Linear(128, 1),
			nn.Tanh(),
		)

	def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		feat = self.body(self.stem(x))
		p = self.policy_conv(feat).flatten(start_dim=1)
		policy_logits = self.policy_fc(p)
		v = self.value_conv(feat).flatten(start_dim=1)
		value = self.value_fc(v).squeeze(-1)
		return policy_logits, value


def device_for_model() -> torch.device:
	if torch.cuda.is_available():
		return torch.device("cuda")
	if torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")
