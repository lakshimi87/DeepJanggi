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
	NUM_SQUARES,
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
	"""AlphaZero-style network with a *fully convolutional* policy head.

	Action space (see ``config``/``board``): index = ``from_sq * NUM_SQUARES + to_sq``
	for board moves, with the final index reserved for PASS. The policy head emits
	one plane per destination square (``NUM_SQUARES`` channels) over the board grid;
	the value at channel ``to_sq`` and spatial position ``from_sq`` is the logit of
	the move ``from_sq -> to_sq``. Flattening as (row, col, channel) reproduces
	``from_sq * NUM_SQUARES + to_sq`` exactly, matching ``board.move_to_action``.

	This replaces the old ``flatten -> Linear(32*90, 8101)`` head, whose single
	weight matrix held 23.3M parameters (95% of the whole net) with no spatial
	weight sharing — the dominant cause of both the oversized checkpoint and the
	policy head failing to learn from sparse self-play visit-count targets. The
	PASS logit, which has no spatial location, is produced by a tiny head off the
	globally pooled tower features.
	"""

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

		# Policy head: 3x3 conv mixing + 1x1 conv to NUM_SQUARES destination planes.
		# No flatten->Linear: each output logit shares weights across the board.
		self.policy_conv1 = nn.Sequential(
			nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(filters),
			nn.ReLU(inplace=True),
		)
		self.policy_conv2 = nn.Conv2d(filters, NUM_SQUARES, kernel_size=1)
		# PASS has no board location; derive its logit from pooled tower features.
		self.policy_pass = nn.Linear(filters, 1)

		# Value head: 1x1 conv (32 channels, per DeepChess, to avoid the 1-channel
		# value-collapse fixed point) + linear bottleneck + scalar.
		self.value_conv = nn.Sequential(
			nn.Conv2d(filters, 32, kernel_size=1, bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
		)
		self.value_fc = nn.Sequential(
			nn.Linear(32 * ROWS * COLS, 128),
			nn.ReLU(inplace=True),
			nn.Linear(128, 1),
			nn.Tanh(),
		)

	def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		feat = self.body(self.stem(x))

		# Policy: (B, NUM_SQUARES, ROWS, COLS) -> flatten as from_sq*NUM_SQUARES+to_sq.
		p = self.policy_conv2(self.policy_conv1(feat))  # (B, to_sq, row, col)
		p = p.permute(0, 2, 3, 1).reshape(p.size(0), -1)  # (B, ROWS*COLS*NUM_SQUARES)
		pass_logit = self.policy_pass(feat.mean(dim=(2, 3)))  # (B, 1)
		policy_logits = torch.cat([p, pass_logit], dim=1)  # (B, TOTAL_ACTIONS)

		v = self.value_conv(feat).flatten(start_dim=1)
		value = self.value_fc(v).squeeze(-1)
		return policy_logits, value


def device_for_model() -> torch.device:
	if torch.cuda.is_available():
		return torch.device("cuda")
	if torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")
