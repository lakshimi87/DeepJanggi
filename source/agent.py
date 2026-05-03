"""Higher-level wrappers around MCTS for self-play and inference."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch

from .board import Janggi, Move, action_to_move, move_to_action, PASS_MOVE
from .config import (
	CHECKPOINT_DIR,
	LATEST_CHECKPOINT,
	MCTS_SIMULATIONS_PLAY,
	TEMPERATURE_MOVES,
	TOTAL_ACTIONS,
)
from .encoder import encode_state
from .mcts import MCTS, select_action_from_visits
from .network import JanggiNet, device_for_model


class NeuralAgent:
	"""An MCTS+network agent. Loads the latest checkpoint when available."""

	def __init__(
		self,
		simulations: int = MCTS_SIMULATIONS_PLAY,
		device: Optional[torch.device] = None,
		checkpoint_path: Optional[str] = None,
	) -> None:
		self.device = device or device_for_model()
		self.network = JanggiNet().to(self.device)
		self.simulations = simulations
		self.checkpoint_path = checkpoint_path or LATEST_CHECKPOINT
		self.load(self.checkpoint_path)
		self.mcts = MCTS(self.network, self.device, simulations)

	def load(self, path: str) -> bool:
		if not path or not os.path.isfile(path):
			return False
		state = torch.load(path, map_location=self.device)
		self.network.load_state_dict(state["model"])
		self.network.eval()
		return True

	def select_move(
		self,
		state: Janggi,
		temperature: float = 0.0,
		add_noise: bool = False,
	) -> Tuple[Move, np.ndarray]:
		_, visits = self.mcts.run(state, add_noise=add_noise)
		action = select_action_from_visits(visits, temperature)
		return action_to_move(action), visits


class RandomAgent:
	"""Picks a uniformly random legal move (used as a baseline in validation)."""

	def select_move(self, state: Janggi, **_kwargs) -> Tuple[Move, np.ndarray]:
		moves = state.legal_moves()
		# Avoid passing unless it is the only option, otherwise random play stalls.
		non_pass = [m for m in moves if m != PASS_MOVE]
		choices = non_pass if non_pass else moves
		move = choices[np.random.randint(len(choices))]
		visits = np.zeros(TOTAL_ACTIONS, dtype=np.float32)
		visits[move_to_action(move)] = 1.0
		return move, visits


def temperature_for_ply(ply: int) -> float:
	return 1.0 if ply < TEMPERATURE_MOVES else 0.0


def ensure_checkpoint_dir() -> None:
	os.makedirs(CHECKPOINT_DIR, exist_ok=True)
