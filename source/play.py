"""Entry point: launch the Pygame-CE UI for human vs AI play."""

from __future__ import annotations

import argparse

from .config import BLUE, LATEST_CHECKPOINT, MCTS_SIMULATIONS_PLAY, RED


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Play DeepJanggi against the AI.")
	p.add_argument(
		"--side",
		choices=("blue", "red"),
		default="blue",
		help="Color the human will play (blue starts at the bottom by default).",
	)
	p.add_argument("--simulations", type=int, default=MCTS_SIMULATIONS_PLAY)
	p.add_argument("--checkpoint", type=str, default=LATEST_CHECKPOINT)
	return p.parse_args()


def main() -> None:
	args = parse_args()
	# Import here so we don't pay pygame's startup cost for non-GUI commands.
	from .ui import run

	human_color = BLUE if args.side == "blue" else RED
	run(human_color=human_color, simulations=args.simulations, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
	main()
