"""Evaluate the current agent against a baseline (random or another checkpoint)."""

from __future__ import annotations

import argparse

import numpy as np

from .agent import NeuralAgent, RandomAgent
from .board import Janggi, action_to_move, move_to_action
from .config import (
	BLUE,
	LATEST_CHECKPOINT,
	MCTS_SIMULATIONS_VALIDATE,
	PLAYER_NAMES,
	RED,
)


def play_match(agent_red, agent_blue) -> int:
	state = Janggi()
	while not state.is_terminal():
		picker = agent_red if state.side_to_move == RED else agent_blue
		move, _ = picker.select_move(state, temperature=0.0, add_noise=False)
		state.apply(move)
	return state.winner if state.winner is not None else -1


def main() -> None:
	parser = argparse.ArgumentParser(description="Validate the DeepJanggi AI.")
	parser.add_argument("--games", type=int, default=10)
	parser.add_argument("--simulations", type=int, default=MCTS_SIMULATIONS_VALIDATE)
	parser.add_argument(
		"--opponent",
		choices=("random", "checkpoint"),
		default="random",
		help="random = uniform legal-move baseline. checkpoint = a saved .pt to compare against.",
	)
	parser.add_argument("--opponent-path", type=str, default="")
	parser.add_argument("--checkpoint", type=str, default=LATEST_CHECKPOINT)
	args = parser.parse_args()

	hero = NeuralAgent(simulations=args.simulations, checkpoint_path=args.checkpoint)
	if args.opponent == "random":
		villain = RandomAgent()
	else:
		villain = NeuralAgent(simulations=args.simulations, checkpoint_path=args.opponent_path)

	wins = losses = draws = 0
	for game_idx in range(args.games):
		# Alternate sides each game so we measure overall strength.
		if game_idx % 2 == 0:
			agent_red, agent_blue = villain, hero
			hero_color = BLUE
		else:
			agent_red, agent_blue = hero, villain
			hero_color = RED
		winner = play_match(agent_red, agent_blue)
		if winner == hero_color:
			wins += 1
			result = "win"
		elif winner == -1:
			draws += 1
			result = "draw"
		else:
			losses += 1
			result = "loss"
		print(
			f"Game {game_idx + 1}/{args.games}: hero={PLAYER_NAMES[hero_color]} "
			f"-> winner={PLAYER_NAMES.get(winner, 'draw')} ({result})"
		)

	total = wins + losses + draws
	score = (wins + 0.5 * draws) / max(total, 1)
	print(
		f"\nResults: {wins}W / {losses}L / {draws}D over {total} games. "
		f"Score = {score:.3f}"
	)
	print(f"Verdict: {'AI is competitive' if score >= 0.55 else 'Needs more training'}.")


if __name__ == "__main__":
	main()
