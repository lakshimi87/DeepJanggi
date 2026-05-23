"""Evaluate the current agent against a baseline (random or another checkpoint)."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from . import board as board_module
from .agent import MinimaxAgent, NeuralAgent, RandomAgent
from .board import Janggi, PASS_MOVE, action_to_move, move_to_action
from .config import (
	ADVISOR,
	BLUE,
	CANNON,
	CHARIOT,
	COLS,
	ELEPHANT,
	EMPTY,
	HORSE,
	KING,
	LATEST_CHECKPOINT,
	MCTS_SIMULATIONS_VALIDATE,
	PLAYER_NAMES,
	RED,
	ROWS,
	SOLDIER,
	make_piece,
)
from .encoder import encode_state


def report_checkpoint(path: str) -> None:
	if not path or not os.path.isfile(path):
		print(f"Checkpoint: {path or '(none)'} (not found)")
		return
	state = torch.load(path, map_location="cpu")
	iteration = state.get("iteration", "?")
	global_step = state.get("global_step", "?")
	print(
		f"Checkpoint: {path} "
		f"(iteration={iteration}, global_step={global_step})"
	)


def play_match(agent_red, agent_blue) -> int:
	state = Janggi()
	while not state.is_terminal():
		picker = agent_red if state.side_to_move == RED else agent_blue
		move, _ = picker.select_move(state, temperature=0.0, add_noise=False)
		state.apply(move)
	return state.winner if state.winner is not None else -1


# ---------------------------------------------------------------------------
# Position-construction helpers for the hand-crafted diagnostics below.
# ---------------------------------------------------------------------------
def empty_grid_with_kings(side_to_move: int) -> Janggi:
	"""Return a board cleared of every piece except the two kings on their palace centers."""
	state = Janggi()
	for r in range(ROWS):
		for c in range(COLS):
			state.grid[r][c] = EMPTY
	state.grid[1][4] = make_piece(RED, KING)
	state.grid[8][4] = make_piece(BLUE, KING)
	state.side_to_move = side_to_move
	return state


def place_piece(state: Janggi, r: int, c: int, color: int, ptype: int) -> None:
	state.grid[r][c] = make_piece(color, ptype)


def _network_value(network, device, state: Janggi) -> float:
	"""Run the value head on `state` and return the scalar from side_to_move's perspective."""
	obs = encode_state(state)
	obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
	network.eval()
	with torch.no_grad():
		_, value = network(obs_t)
	return float(value.item())


def _side_to_move_can_capture_king(state: Janggi, victim_color: int) -> bool:
	"""True if the player on move has a move that lands on `victim_color`'s king square."""
	target = make_piece(victim_color, KING)
	for move in state.legal_moves():
		if move == PASS_MOVE:
			continue
		_, _, tr, tc = move
		if state.grid[tr][tc] == target:
			return True
	return False


def king_in_check(state: Janggi, victim_color: int) -> bool:
	"""True if the opponent (regardless of whose turn it is) can capture `victim_color`'s king."""
	probe = state.clone()
	probe.side_to_move = RED if victim_color == BLUE else BLUE
	return _side_to_move_can_capture_king(probe, victim_color)


# ---------------------------------------------------------------------------
# Test 1: 형세 판단 — does the value head rank material balances correctly?
# ---------------------------------------------------------------------------
def _evaluation_cases():
	"""Yield (label, state, expected_sign) tuples.

	expected_sign is from side_to_move's perspective: +1 means side_to_move should be
	favored, -1 means the opponent should be favored, 0 means roughly balanced.
	"""
	cases = []

	# Starting position: balanced (Han bonus barely tips it Blue's way).
	s = Janggi()
	s.side_to_move = BLUE
	cases.append(("initial position (BLUE to move)", s, 0))

	# Red missing one chariot, Blue intact -> Blue ahead.
	s = Janggi()
	s.grid[0][0] = EMPTY
	s.side_to_move = BLUE
	cases.append(("RED down a chariot (BLUE to move)", s, +1))

	# Blue missing one chariot, Red intact -> Red ahead.
	s = Janggi()
	s.grid[9][0] = EMPTY
	s.side_to_move = BLUE
	cases.append(("BLUE down a chariot (BLUE to move)", s, -1))

	# Red stripped to king + advisors, Blue full -> Blue clearly winning.
	s = empty_grid_with_kings(BLUE)
	place_piece(s, 0, 3, RED, ADVISOR)
	place_piece(s, 0, 5, RED, ADVISOR)
	for c, pt in (
		(0, CHARIOT), (1, ELEPHANT), (2, HORSE), (3, ADVISOR), (5, ADVISOR),
		(6, HORSE), (7, ELEPHANT), (8, CHARIOT),
	):
		place_piece(s, 9, c, BLUE, pt)
	cases.append(("RED stripped vs full BLUE (BLUE to move)", s, +1))

	# Mirror: Blue stripped, Red full -> Red clearly winning.
	s = empty_grid_with_kings(RED)
	place_piece(s, 9, 3, BLUE, ADVISOR)
	place_piece(s, 9, 5, BLUE, ADVISOR)
	for c, pt in (
		(0, CHARIOT), (1, ELEPHANT), (2, HORSE), (3, ADVISOR), (5, ADVISOR),
		(6, HORSE), (7, ELEPHANT), (8, CHARIOT),
	):
		place_piece(s, 0, c, RED, pt)
	cases.append(("BLUE stripped vs full RED (RED to move)", s, +1))

	# Modest minor-piece swap: Red has cannon, Blue has horse extra -> Red slightly ahead.
	s = empty_grid_with_kings(BLUE)
	place_piece(s, 3, 4, RED, CANNON)
	place_piece(s, 6, 4, BLUE, HORSE)
	cases.append(("RED extra cannon vs BLUE extra horse (BLUE to move)", s, -1))

	return cases


def run_evaluation_test(args) -> int:
	"""Check that the value head distinguishes between materially favorable/unfavorable positions."""
	print("=== 형세 판단 테스트 (material/value-head evaluation) ===")
	hero = NeuralAgent(simulations=1, checkpoint_path=args.checkpoint)
	tolerance = max(0.0, float(args.eval_tolerance))
	cases = _evaluation_cases()
	passed = 0
	for label, state, expected_sign in cases:
		value = _network_value(hero.network, hero.device, state)
		if expected_sign == 0:
			ok = abs(value) <= max(tolerance, 0.1)
		else:
			ok = (value > tolerance) if expected_sign > 0 else (value < -tolerance)
		passed += int(ok)
		print(f"  [{'OK' if ok else 'FAIL'}] {label}: value={value:+.3f} (expected_sign={expected_sign:+d})")
	print(f"Material evaluation: {passed}/{len(cases)} cases passed.")
	return passed == len(cases)


# ---------------------------------------------------------------------------
# Test 2: 왕 안전 — given the king is under attack, does the agent save it?
# ---------------------------------------------------------------------------
def _king_safety_cases():
	"""Build positions where side_to_move's king will be captured next ply if ignored."""
	cases = []

	# A: Red chariot on column 4 threatens Blue king at the palace center.
	s = empty_grid_with_kings(BLUE)
	place_piece(s, 4, 4, RED, CHARIOT)
	cases.append(("RED chariot on column 4 threatens BLUE king (move-king only)", s))

	# B: Red chariot on row 8 threatens Blue king with nothing in between.
	s = empty_grid_with_kings(BLUE)
	place_piece(s, 8, 0, RED, CHARIOT)
	cases.append(("RED chariot on row 8 threatens BLUE king", s))

	# C: Mirror: Blue chariot on column 4 threatens Red king at the palace center.
	s = empty_grid_with_kings(RED)
	place_piece(s, 5, 4, BLUE, CHARIOT)
	cases.append(("BLUE chariot on column 4 threatens RED king", s))

	# D: Red cannon with a soldier screen threatens Blue king down column 4.
	s = empty_grid_with_kings(BLUE)
	place_piece(s, 4, 4, RED, CANNON)
	place_piece(s, 6, 4, RED, SOLDIER)  # screen for the cannon
	cases.append(("RED cannon (screened by soldier) threatens BLUE king", s))

	# E: Blue has a chariot that can capture the threatening Red chariot.
	s = empty_grid_with_kings(BLUE)
	place_piece(s, 4, 4, RED, CHARIOT)
	place_piece(s, 4, 8, BLUE, CHARIOT)  # capture along row 4
	cases.append(("BLUE can capture the threatening chariot", s))

	# F: Blue has a chariot that can block on column 4 between attacker and king.
	s = empty_grid_with_kings(BLUE)
	place_piece(s, 4, 4, RED, CHARIOT)
	place_piece(s, 7, 8, BLUE, CHARIOT)  # block by moving to (7,4)
	cases.append(("BLUE can block the threatening chariot", s))

	return cases


def run_king_safety_test(args) -> int:
	"""Check that the agent rescues its king when capture is one ply away."""
	print("=== 왕 안전 테스트 (king-in-check reactions) ===")
	hero = NeuralAgent(simulations=args.simulations, checkpoint_path=args.checkpoint)
	cases = _king_safety_cases()
	passed = 0
	for label, state in cases:
		our_color = state.side_to_move
		assert king_in_check(state, our_color), f"Bad setup: king not threatened in {label!r}"
		move, _ = hero.select_move(state, temperature=0.0, add_noise=False)
		after = state.clone()
		after.apply(move)
		# Capturing the opposing king is also a valid "save" (game over, we won).
		if after.is_terminal() and after.winner == our_color:
			safe = True
		else:
			safe = not _side_to_move_can_capture_king(after, our_color)
		passed += int(safe)
		print(f"  [{'OK' if safe else 'FAIL'}] {label}: move={move}")
	print(f"King safety: {passed}/{len(cases)} cases passed.")
	return passed == len(cases)


def main() -> None:
	parser = argparse.ArgumentParser(description="Validate the DeepJanggi AI.")
	parser.add_argument(
		"--mode",
		choices=("match", "evaluation", "king-safety", "all"),
		default="all",
		help=(
			"all (default) = run evaluation, then king-safety, then match. "
			"match = only play games against a baseline. "
			"evaluation = only check value-head positional judgment on hand-crafted material imbalances. "
			"king-safety = only check that the agent responds when its king is one ply from capture."
		),
	)
	parser.add_argument(
		"--eval-tolerance",
		type=float,
		default=0.05,
		help="Value-head magnitude threshold for declaring a non-zero sign in evaluation mode.",
	)
	parser.add_argument("--games", type=int, default=10)
	parser.add_argument("--simulations", type=int, default=MCTS_SIMULATIONS_VALIDATE)
	parser.add_argument(
		"--opponent",
		choices=("random", "minimax", "checkpoint"),
		default="minimax",
		help=(
			"random = uniform legal-move baseline. "
			"minimax = alpha-beta search with material evaluation (stronger baseline). "
			"checkpoint = a saved .pt to compare against."
		),
	)
	parser.add_argument(
		"--minimax-depth",
		type=int,
		default=3,
		help="Search depth (in plies) for the minimax baseline.",
	)
	parser.add_argument(
		"--max-ply",
		type=int,
		default=200,
		help=(
			"Cap on game length during validation. Lower than the training cap "
			"so drawn-out matches don't dominate evaluation time."
		),
	)
	parser.add_argument("--opponent-path", type=str, default="")
	parser.add_argument("--checkpoint", type=str, default=LATEST_CHECKPOINT)
	args = parser.parse_args()

	# Override the board's ply cap for this run only; encoder.py still uses the
	# training-time MAX_PLY so the network sees the same normalization it learned.
	board_module.MAX_PLY = args.max_ply

	report_checkpoint(args.checkpoint)

	if args.mode in ("evaluation", "all"):
		run_evaluation_test(args)
		print()
	if args.mode in ("king-safety", "all"):
		run_king_safety_test(args)
		print()
	if args.mode not in ("match", "all"):
		return

	hero = NeuralAgent(simulations=args.simulations, checkpoint_path=args.checkpoint)
	if args.opponent == "random":
		villain = RandomAgent()
		print("Opponent: random baseline")
	elif args.opponent == "minimax":
		villain = MinimaxAgent(depth=args.minimax_depth)
		print(f"Opponent: minimax (depth={args.minimax_depth})")
	else:
		report_checkpoint(args.opponent_path)
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
