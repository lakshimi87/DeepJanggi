"""Higher-level wrappers around MCTS for self-play and inference."""

from __future__ import annotations

import math
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch

from .board import Janggi, Move, action_to_move, move_to_action, PASS_MOVE
from .config import (
	BLUE,
	CHECKPOINT_DIR,
	HAN_BONUS,
	KING,
	LATEST_CHECKPOINT,
	MCTS_SIMULATIONS_PLAY,
	PIECE_VALUE,
	RED,
	TEMPERATURE_MOVES,
	TOTAL_ACTIONS,
	piece_color,
	piece_type,
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


_KING_VALUE = 10_000.0


def _material_eval(state: Janggi, perspective: int) -> float:
	"""Score = (perspective side material) - (opponent material), with Han bonus baked in."""
	score = 0.0
	for row in state.grid:
		for piece in row:
			if not piece:
				continue
			value = _KING_VALUE if piece_type(piece) == KING else PIECE_VALUE[piece_type(piece)]
			if piece_color(piece) == perspective:
				score += value
			else:
				score -= value
	# Han bonus: Blue gets a small structural bonus that shows up in draw scoring too.
	if perspective == BLUE:
		score += HAN_BONUS
	else:
		score -= HAN_BONUS
	return score


class MinimaxAgent:
	"""Alpha-beta minimax over the material-score evaluation.

	A stronger non-NN baseline than RandomAgent: prefers winning captures and avoids
	losing them within its search horizon. Captures the opposing king when possible.
	"""

	def __init__(self, depth: int = 2, seed: Optional[int] = None) -> None:
		self.depth = max(1, int(depth))
		self._rng = random.Random(seed)

	def select_move(self, state: Janggi, **_kwargs) -> Tuple[Move, np.ndarray]:
		me = state.side_to_move
		best_score = -math.inf
		best_moves: List[Move] = []
		for move in self._candidate_moves(state):
			child = state.clone()
			child.apply(move)
			# A king capture ends the game without switching turns, so it must be
			# scored as an immediate win for the mover rather than negated like a
			# normal child (which would make the search avoid the winning move).
			if child.is_terminal() and child.side_to_move == me:
				score = _KING_VALUE * 10
			else:
				score = -self._negamax(child, self.depth - 1, -math.inf, math.inf, me)
			if score > best_score + 1e-9:
				best_score = score
				best_moves = [move]
			elif score > best_score - 1e-9:
				best_moves.append(move)
		move = self._rng.choice(best_moves) if best_moves else PASS_MOVE
		visits = np.zeros(TOTAL_ACTIONS, dtype=np.float32)
		visits[move_to_action(move)] = 1.0
		return move, visits

	def _negamax(self, state: Janggi, depth: int, alpha: float, beta: float, me: int) -> float:
		"""Negamax: returns the score from the perspective of state.side_to_move."""
		if state.is_terminal():
			# Convert game result into a large-magnitude score.
			if state.winner == -1:
				return 0.0
			return _KING_VALUE * 10 if state.winner == state.side_to_move else -_KING_VALUE * 10
		if depth <= 0:
			return _material_eval(state, state.side_to_move)
		best = -math.inf
		mover = state.side_to_move
		for move in self._candidate_moves(state):
			child = state.clone()
			child.apply(move)
			# King capture wins without a turn switch (see select_move).
			if child.is_terminal() and child.side_to_move == mover:
				score = _KING_VALUE * 10
			else:
				score = -self._negamax(child, depth - 1, -beta, -alpha, me)
			if score > best:
				best = score
			if best > alpha:
				alpha = best
			if alpha >= beta:
				break
		# No legal non-pass moves found is unusual (pass is always legal); guard anyway.
		if best == -math.inf:
			return _material_eval(state, state.side_to_move)
		return best

	def _candidate_moves(self, state: Janggi) -> List[Move]:
		"""Skip the pass action when other moves exist (pass throws away tempo)."""
		moves = state.legal_moves()
		non_pass = [m for m in moves if m != PASS_MOVE]
		return non_pass if non_pass else moves


def temperature_for_ply(ply: int) -> float:
	return 1.0 if ply < TEMPERATURE_MOVES else 0.0


def ensure_checkpoint_dir() -> None:
	os.makedirs(CHECKPOINT_DIR, exist_ok=True)
