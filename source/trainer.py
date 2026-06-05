"""Training loop: alternate self-play with gradient updates and checkpoint to disk."""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .agent import ensure_checkpoint_dir
from .config import (
	BATCH_SIZE,
	CHECKPOINT_DIR,
	GAMES_PER_ITERATION,
	LATEST_CHECKPOINT,
	LEARNING_RATE,
	MCTS_SIMULATIONS_TRAIN,
	REPLAY_CAPACITY,
	TRAIN_STEPS_PER_ITERATION,
	WEIGHT_DECAY,
)
from .mcts import MCTS
from .network import JanggiNet, device_for_model
from .replay import ReplayBuffer
from .selfplay import play_one_game


class Trainer:
	def __init__(
		self,
		device: Optional[torch.device] = None,
		simulations: int = MCTS_SIMULATIONS_TRAIN,
		games_per_iter: int = GAMES_PER_ITERATION,
		train_steps: int = TRAIN_STEPS_PER_ITERATION,
		batch_size: int = BATCH_SIZE,
		buffer_capacity: int = REPLAY_CAPACITY,
		learning_rate: float = LEARNING_RATE,
	) -> None:
		ensure_checkpoint_dir()
		self.device = device or device_for_model()
		self.network = JanggiNet().to(self.device)
		self.optimizer = torch.optim.Adam(
			self.network.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY
		)
		self.replay = ReplayBuffer(capacity=buffer_capacity)
		self.simulations = simulations
		self.games_per_iter = games_per_iter
		self.train_steps = train_steps
		self.batch_size = batch_size
		self.iteration = 0
		self.global_step = 0

	# ------------------------------------------------------------------
	# Checkpointing
	# ------------------------------------------------------------------
	def save(self, path: str = LATEST_CHECKPOINT) -> None:
		# Write to a temp file then atomically rename, so a concurrent reader
		# (e.g. play.py loading the agent) never sees a half-written archive.
		tmp_path = f"{path}.tmp"
		torch.save(
			{
				"model": self.network.state_dict(),
				"optimizer": self.optimizer.state_dict(),
				"iteration": self.iteration,
				"global_step": self.global_step,
			},
			tmp_path,
		)
		os.replace(tmp_path, path)

	def load(self, path: str = LATEST_CHECKPOINT) -> bool:
		if not os.path.isfile(path):
			return False
		state = torch.load(path, map_location=self.device)
		self.network.load_state_dict(state["model"])
		if "optimizer" in state:
			try:
				self.optimizer.load_state_dict(state["optimizer"])
			except ValueError:
				pass
		self.iteration = state.get("iteration", 0)
		self.global_step = state.get("global_step", 0)
		return True

	# ------------------------------------------------------------------
	# Self-play
	# ------------------------------------------------------------------
	def _generate_games(self) -> dict:
		mcts = MCTS(self.network, self.device, self.simulations)
		stats = {"red": 0, "blue": 0, "draw": 0, "samples": 0}
		for _ in range(self.games_per_iter):
			samples, winner = play_one_game(mcts)
			self.replay.push_many(samples)
			stats["samples"] += len(samples)
			if winner == 0:
				stats["red"] += 1
			elif winner == 1:
				stats["blue"] += 1
			else:
				stats["draw"] += 1
		return stats

	# ------------------------------------------------------------------
	# Optimization
	# ------------------------------------------------------------------
	def _train_step(self) -> dict:
		states, policy_targets, value_targets = self.replay.sample_batch(self.batch_size)
		states = states.to(self.device)
		policy_targets = policy_targets.to(self.device)
		value_targets = value_targets.to(self.device)

		self.network.train()
		policy_logits, values = self.network(states)
		log_probs = F.log_softmax(policy_logits, dim=1)
		policy_loss = -(policy_targets * log_probs).sum(dim=1).mean()
		value_loss = F.mse_loss(values, value_targets)
		loss = policy_loss + value_loss

		self.optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=2.0)
		self.optimizer.step()
		self.global_step += 1
		return {
			"loss": float(loss.item()),
			"policy_loss": float(policy_loss.item()),
			"value_loss": float(value_loss.item()),
		}

	# ------------------------------------------------------------------
	# Driver
	# ------------------------------------------------------------------
	def run(self, total_iterations: int) -> None:
		print(f"Training on device: {self.device}")
		for _ in range(total_iterations):
			self.iteration += 1
			t0 = time.time()
			game_stats = self._generate_games()
			t_games = time.time() - t0

			if len(self.replay) < self.batch_size:
				print(
					f"[iter {self.iteration}] gathered {game_stats['samples']} samples "
					f"(buffer={len(self.replay)}); skipping training step."
				)
				self.save()
				continue

			t1 = time.time()
			running = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
			for _ in range(self.train_steps):
				stats = self._train_step()
				for k, v in stats.items():
					running[k] += v
			for k in running:
				running[k] /= max(self.train_steps, 1)
			t_train = time.time() - t1

			print(
				f"[iter {self.iteration}] "
				f"games R/B/D={game_stats['red']}/{game_stats['blue']}/{game_stats['draw']} "
				f"samples={game_stats['samples']} buf={len(self.replay)} "
				f"loss={running['loss']:.3f} "
				f"(p={running['policy_loss']:.3f}, v={running['value_loss']:.3f}) "
				f"t_games={t_games:.1f}s t_train={t_train:.1f}s"
			)
			self.save()
		print("Training finished.")
