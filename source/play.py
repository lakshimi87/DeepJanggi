"""Entry point: launch the Pygame-CE UI for human vs AI play."""

from __future__ import annotations

import argparse

from .config import (
	BLUE,
	DEFAULT_DIFFICULTY,
	DIFFICULTY_LEVELS,
	DIFFICULTY_SIMULATIONS,
	LATEST_CHECKPOINT,
	RED,
)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Play DeepJanggi against the AI.")
	p.add_argument(
		"--side",
		choices=("blue", "red"),
		default="blue",
		help="Color the human will play (blue starts at the bottom by default).",
	)
	p.add_argument(
		"--difficulty",
		choices=DIFFICULTY_LEVELS,
		default=DEFAULT_DIFFICULTY,
		help="AI strength. Controls the MCTS simulation budget.",
	)
	p.add_argument(
		"--simulations",
		type=int,
		default=None,
		help="Override the difficulty's MCTS simulation count.",
	)
	p.add_argument("--checkpoint", type=str, default=LATEST_CHECKPOINT)
	return p.parse_args()


def main() -> None:
	args = parse_args()
	# Import here so we don't pay pygame's startup cost for non-GUI commands.
	from .ui import run

	human_color = BLUE if args.side == "blue" else RED
	simulations = (
		args.simulations
		if args.simulations is not None
		else DIFFICULTY_SIMULATIONS[args.difficulty]
	)
	run(
		human_color=human_color,
		difficulty=args.difficulty,
		simulations=simulations,
		checkpoint_path=args.checkpoint,
	)


if __name__ == "__main__":
	main()
