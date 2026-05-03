"""Entry point: run self-play training for the Janggi AI."""

from __future__ import annotations

import argparse

from .config import (
	GAMES_PER_ITERATION,
	LATEST_CHECKPOINT,
	MCTS_SIMULATIONS_TRAIN,
	TRAIN_STEPS_PER_ITERATION,
)
from .trainer import Trainer


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Train the DeepJanggi AI via self-play.")
	p.add_argument("--iterations", type=int, default=100, help="Number of self-play iterations.")
	p.add_argument("--simulations", type=int, default=MCTS_SIMULATIONS_TRAIN)
	p.add_argument("--games-per-iter", type=int, default=GAMES_PER_ITERATION)
	p.add_argument("--train-steps", type=int, default=TRAIN_STEPS_PER_ITERATION)
	p.add_argument("--resume", action="store_true", help="Resume from checkpoints/latest.pt if present.")
	return p.parse_args()


def main() -> None:
	args = parse_args()
	trainer = Trainer(
		simulations=args.simulations,
		games_per_iter=args.games_per_iter,
		train_steps=args.train_steps,
	)
	if args.resume and trainer.load(LATEST_CHECKPOINT):
		print(f"Resumed from iteration {trainer.iteration} (step {trainer.global_step}).")
	trainer.run(args.iterations)


if __name__ == "__main__":
	main()
