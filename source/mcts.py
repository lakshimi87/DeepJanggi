"""AlphaZero-style PUCT MCTS over the Janggi state."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .board import Janggi, Move, action_to_move, move_to_action
from .config import (
	DIRICHLET_ALPHA,
	DIRICHLET_EPS,
	PUCT_C,
	TOTAL_ACTIONS,
)
from .encoder import encode_state


@dataclass
class Node:
	"""A node in the search tree, indexed by the state that *follows* its parent's move."""

	prior: float = 0.0
	visit_count: int = 0
	value_sum: float = 0.0
	to_play: int = 0
	# children maps action index -> Node.
	children: Dict[int, "Node"] = field(default_factory=dict)
	expanded: bool = False

	def q_value(self) -> float:
		return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count


class MCTS:
	"""Search driver. Use `run` to build a search tree and pick a move."""

	def __init__(self, network: torch.nn.Module, device: torch.device, simulations: int) -> None:
		self.network = network
		self.device = device
		self.simulations = simulations

	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------
	def run(self, root_state: Janggi, add_noise: bool = False) -> Tuple[Node, np.ndarray]:
		root = Node(to_play=root_state.side_to_move)
		self._expand(root, root_state)
		if add_noise:
			self._add_dirichlet_noise(root)

		for _ in range(self.simulations):
			node = root
			state = root_state.clone()
			path: List[Tuple[Node, int]] = []
			# Selection: walk down using PUCT until we hit an unexpanded node.
			while node.expanded and not state.is_terminal():
				action, child = self._select_child(node)
				path.append((node, action))
				state.apply(action_to_move(action))
				node = child
			# Expansion / evaluation.
			if state.is_terminal():
				value = state.reward_for(state.side_to_move)
				# state.side_to_move at terminal is whoever would have moved next; reward is
				# from their perspective. Backprop alternates, so we negate at the leaf.
				value = -value
			else:
				value = self._expand(node, state)
			# Backprop.
			for parent, action in reversed(path):
				child = parent.children[action]
				child.visit_count += 1
				child.value_sum += value
				value = -value

		visits = np.zeros(TOTAL_ACTIONS, dtype=np.float32)
		for action, child in root.children.items():
			visits[action] = child.visit_count
		return root, visits

	# ------------------------------------------------------------------
	# Selection
	# ------------------------------------------------------------------
	def _select_child(self, node: Node) -> Tuple[int, Node]:
		total_visits = sum(child.visit_count for child in node.children.values())
		sqrt_total = math.sqrt(max(total_visits, 1))
		best_score = -float("inf")
		best_action = -1
		best_child: Optional[Node] = None
		for action, child in node.children.items():
			u = PUCT_C * child.prior * sqrt_total / (1 + child.visit_count)
			score = -child.q_value() + u  # Negate Q because the child's perspective is the opponent.
			if score > best_score:
				best_score = score
				best_action = action
				best_child = child
		assert best_child is not None
		return best_action, best_child

	# ------------------------------------------------------------------
	# Expansion
	# ------------------------------------------------------------------
	def _expand(self, node: Node, state: Janggi) -> float:
		"""Evaluate `state`, attach children for legal moves, and return the leaf value."""
		legal_moves: List[Move] = state.legal_moves()
		legal_actions = [move_to_action(m) for m in legal_moves]

		obs = encode_state(state)
		obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
		self.network.eval()
		with torch.no_grad():
			policy_logits, value = self.network(obs_t)
		policy = policy_logits.squeeze(0).cpu().numpy()
		value_scalar = float(value.item())

		# Mask illegal actions and renormalize.
		mask = np.full(TOTAL_ACTIONS, -1e9, dtype=np.float32)
		mask[legal_actions] = 0.0
		policy = policy + mask
		policy = policy - policy.max()
		exp_p = np.exp(policy)
		exp_p[~np.isfinite(exp_p)] = 0.0
		total = exp_p.sum()
		if total <= 0:
			# Defensive fallback: uniform over legal actions.
			priors = np.zeros(TOTAL_ACTIONS, dtype=np.float32)
			if legal_actions:
				priors[legal_actions] = 1.0 / len(legal_actions)
		else:
			priors = exp_p / total

		next_to_play = state.side_to_move
		for action in legal_actions:
			node.children[action] = Node(prior=float(priors[action]), to_play=next_to_play)
		node.expanded = True
		# Network value is from the perspective of `state.side_to_move`; the leaf value
		# returned to the caller is also from that perspective (the caller flips signs).
		return value_scalar

	# ------------------------------------------------------------------
	# Exploration noise
	# ------------------------------------------------------------------
	def _add_dirichlet_noise(self, root: Node) -> None:
		actions = list(root.children.keys())
		if not actions:
			return
		noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(actions))
		for action, n in zip(actions, noise):
			child = root.children[action]
			child.prior = (1 - DIRICHLET_EPS) * child.prior + DIRICHLET_EPS * float(n)


def select_action_from_visits(
	visits: np.ndarray, temperature: float
) -> int:
	"""Sample (or argmax) an action proportional to visit counts."""
	if temperature <= 1e-3:
		return int(np.argmax(visits))
	probs = np.power(visits, 1.0 / temperature)
	total = probs.sum()
	if total <= 0:
		# Fall back to uniform over non-zero actions.
		nz = np.nonzero(visits)[0]
		return int(np.random.choice(nz)) if len(nz) else 0
	probs = probs / total
	return int(np.random.choice(len(probs), p=probs))
