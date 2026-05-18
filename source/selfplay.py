"""Generate self-play games by running MCTS for both sides."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from .agent import temperature_for_ply
from .board import Janggi, action_to_move, randomize_flank_setup
from .config import TOTAL_ACTIONS
from .encoder import encode_state
from .mcts import MCTS, select_action_from_visits
from .replay import Sample


def play_one_game(mcts: MCTS, randomize_setup: bool = True) -> Tuple[List[Sample], int]:
	"""Run a single self-play game and return per-move training samples + winner.

	winner: 0 = RED, 1 = BLUE, -1 = draw.
	"""
	state = Janggi()
	if randomize_setup:
		randomize_flank_setup(state)
	history: List[Tuple[np.ndarray, np.ndarray, int]] = []  # (planes, visit_probs, side_to_move)

	while not state.is_terminal():
		visit_probs, action = _think_and_pick(mcts, state)
		planes = encode_state(state)
		history.append((planes, visit_probs, state.side_to_move))
		state.apply(action_to_move(action))

	winner = state.winner if state.winner is not None else -1

	samples: List[Sample] = []
	for planes, probs, side in history:
		if winner == -1:
			z = 0.0
		elif winner == side:
			z = 1.0
		else:
			z = -1.0
		samples.append((planes, probs, z))
	return samples, winner


def _think_and_pick(mcts: MCTS, state: Janggi) -> Tuple[np.ndarray, int]:
	root, visits = mcts.run(state, add_noise=True)
	temp = temperature_for_ply(state.ply)
	total = visits.sum()
	if total > 0:
		probs = visits / total
	else:
		probs = np.zeros(TOTAL_ACTIONS, dtype=np.float32)
	action = select_action_from_visits(visits, temp)
	return probs, action
