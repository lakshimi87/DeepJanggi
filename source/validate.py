"""Ground-truth validation for DeepJanggi.

Tests the neural engine (and a classical minimax baseline) against curated
positions with a known best move, plus value-head judgement on hand-crafted
material imbalances.  Use after training to measure what the model has learned.

    ./validate.sh                     # test latest checkpoint + 10-game match
    ./validate.sh --history           # show progress across all checkpoints
    ./validate.sh --simulations 200   # more MCTS sims (slower but fairer)
    ./validate.sh --games 0           # skip the head-to-head match
    ./validate.sh --games 40          # longer match for a tighter win-rate CI
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time

import numpy as np
import torch

from . import board as board_module
from .agent import MinimaxAgent, NeuralAgent, RandomAgent
from .board import Janggi, PASS_MOVE, action_to_move
from .config import (
	ADVISOR,
	BLUE,
	CANNON,
	CHARIOT,
	CHECKPOINT_DIR,
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
	piece_type,
)
from .encoder import encode_state


# ═══════════════════════════════════════════════════════════════════════
# Position-construction helpers
# ═══════════════════════════════════════════════════════════════════════

def empty_grid_with_kings(side_to_move: int) -> Janggi:
	"""Board cleared of every piece except the two kings on their palace centers."""
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


PIECE_LETTER = {
	KING: "K", ADVISOR: "A", CHARIOT: "R", CANNON: "C",
	HORSE: "H", ELEPHANT: "E", SOLDIER: "S",
}


def describe_move(state: Janggi, move) -> str:
	"""Compact human label for a move, e.g. 'R48x44' (chariot (4,8) captures (4,4))."""
	if move == PASS_MOVE or move is None:
		return "pass" if move == PASS_MOVE else "-"
	fr, fc, tr, tc = move
	letter = PIECE_LETTER.get(piece_type(state.grid[fr][fc]), "?")
	sep = "x" if state.grid[tr][tc] != EMPTY else "-"
	return f"{letter}{fr}{fc}{sep}{tr}{tc}"


# ═══════════════════════════════════════════════════════════════════════
# Move-capture / king-safety primitives
# ═══════════════════════════════════════════════════════════════════════

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
	"""True if the opponent could capture `victim_color`'s king (regardless of turn)."""
	probe = state.clone()
	probe.side_to_move = RED if victim_color == BLUE else BLUE
	return _side_to_move_can_capture_king(probe, victim_color)


def _move_captures_king(state: Janggi, move) -> bool:
	"""Does playing `move` capture the opposing king (an immediate win)?"""
	if move == PASS_MOVE or move is None:
		return False
	after = state.clone()
	after.apply(move)
	return after.is_terminal() and after.winner == state.side_to_move


def _move_lands_on(target):
	"""Accept predicate: the move's destination square equals `target` (tr, tc)."""
	def accept(state, move):
		return move not in (PASS_MOVE, None) and tuple(move[2:]) == tuple(target)
	return accept


def _move_saves_king(state: Janggi, move) -> bool:
	"""Accept predicate for king-safety: after `move`, our king is no longer capturable."""
	if move is None:
		return False
	our_color = state.side_to_move
	after = state.clone()
	after.apply(move)
	if after.is_terminal() and after.winner == our_color:
		return True  # capturing the attacker's king is also a valid "save"
	return not _side_to_move_can_capture_king(after, our_color)


def _network_value(network, device, state: Janggi) -> float:
	"""Value head on `state`, returned from side_to_move's perspective."""
	obs = encode_state(state)
	obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
	network.eval()
	with torch.no_grad():
		_, value = network(obs_t)
	return float(value.item())


def _topk_moves(state: Janggi, visits: np.ndarray, k: int = 3):
	"""Top-k candidate moves by visit count: list of (label, probability)."""
	total = float(visits.sum())
	out = []
	for idx in np.argsort(visits)[::-1][:k]:
		if visits[idx] <= 0:
			break
		mv = action_to_move(int(idx))
		out.append((describe_move(state, mv), visits[idx] / total if total else 0.0))
	return out


# ═══════════════════════════════════════════════════════════════════════
# Ground-truth move suites
# ═══════════════════════════════════════════════════════════════════════
# Each case is (label, state, accept_fn, expected_str), where accept_fn(state,
# move) -> bool decides whether the chosen move counts as correct.

def _king_capture_cases():
	"""Positions where the side to move can capture the opposing king in one move."""
	cases = []

	# Blue chariot slides up column 4 onto the red king.
	s = empty_grid_with_kings(BLUE)
	place_piece(s, 5, 4, BLUE, CHARIOT)
	cases.append(("BLUE chariot can take RED king up column 4", s,
	              _move_captures_king, "capture RED king"))

	# Blue chariot slides along row 1 onto the red king.
	s = empty_grid_with_kings(BLUE)
	place_piece(s, 1, 0, BLUE, CHARIOT)
	cases.append(("BLUE chariot can take RED king along row 1", s,
	              _move_captures_king, "capture RED king"))

	# Mirror: red chariot slides down column 4 onto the blue king.
	s = empty_grid_with_kings(RED)
	place_piece(s, 5, 4, RED, CHARIOT)
	cases.append(("RED chariot can take BLUE king down column 4", s,
	              _move_captures_king, "capture BLUE king"))

	# Blue cannon takes the red king, screened by a soldier.
	s = empty_grid_with_kings(BLUE)
	place_piece(s, 5, 4, BLUE, CANNON)
	place_piece(s, 3, 4, BLUE, SOLDIER)  # screen for the cannon jump
	cases.append(("BLUE cannon (screened) can take RED king", s,
	              _move_captures_king, "capture RED king"))

	return cases


def _win_material_cases():
	"""Positions with a free (undefended) enemy piece to capture, biggest first."""
	cases = []

	# Lone hanging red chariot, capturable by the blue chariot.
	s = empty_grid_with_kings(BLUE)
	place_piece(s, 4, 4, RED, CHARIOT)
	place_piece(s, 4, 8, BLUE, CHARIOT)
	cases.append(("BLUE wins a hanging RED chariot", s,
	              _move_lands_on((4, 4)), "Rx(4,4)"))

	# Two captures available: prefer the chariot (13) over the horse (5).
	s = empty_grid_with_kings(BLUE)
	place_piece(s, 4, 4, RED, CHARIOT)
	place_piece(s, 2, 8, RED, HORSE)
	place_piece(s, 4, 8, BLUE, CHARIOT)  # can take either the chariot or the horse
	cases.append(("BLUE prefers the chariot over the horse", s,
	              _move_lands_on((4, 4)), "Rx(4,4) not the horse"))

	# Mirror: lone hanging blue chariot, capturable by the red chariot.
	s = empty_grid_with_kings(RED)
	place_piece(s, 5, 4, BLUE, CHARIOT)
	place_piece(s, 5, 0, RED, CHARIOT)
	cases.append(("RED wins a hanging BLUE chariot", s,
	              _move_lands_on((5, 4)), "Rx(5,4)"))

	return cases


def _king_safety_cases():
	"""Positions where side_to_move's king will be captured next ply if ignored."""
	cases = []

	# A: Red chariot on column 4 threatens the blue king (must move the king).
	s = empty_grid_with_kings(BLUE)
	place_piece(s, 4, 4, RED, CHARIOT)
	cases.append(("RED chariot threatens BLUE king (flee only)", s,
	              _move_saves_king, "save the king"))

	# B: Red chariot on row 8 threatens the blue king, nothing in between.
	s = empty_grid_with_kings(BLUE)
	place_piece(s, 8, 0, RED, CHARIOT)
	cases.append(("RED chariot on row 8 threatens BLUE king", s,
	              _move_saves_king, "save the king"))

	# C: Mirror: blue chariot on column 4 threatens the red king.
	s = empty_grid_with_kings(RED)
	place_piece(s, 5, 4, BLUE, CHARIOT)
	cases.append(("BLUE chariot threatens RED king", s,
	              _move_saves_king, "save the king"))

	# D: Red cannon with a soldier screen threatens the blue king down column 4.
	s = empty_grid_with_kings(BLUE)
	place_piece(s, 4, 4, RED, CANNON)
	place_piece(s, 6, 4, RED, SOLDIER)  # screen for the cannon
	cases.append(("RED cannon (screened) threatens BLUE king", s,
	              _move_saves_king, "save the king"))

	# E: Blue can capture the threatening red chariot.
	s = empty_grid_with_kings(BLUE)
	place_piece(s, 4, 4, RED, CHARIOT)
	place_piece(s, 4, 8, BLUE, CHARIOT)  # capture along row 4
	cases.append(("BLUE can capture the threatening chariot", s,
	              _move_saves_king, "save the king"))

	# F: Blue can block on column 4 between the attacker and the king.
	s = empty_grid_with_kings(BLUE)
	place_piece(s, 4, 4, RED, CHARIOT)
	place_piece(s, 7, 8, BLUE, CHARIOT)  # block by moving to (7,4)
	cases.append(("BLUE can block the threatening chariot", s,
	              _move_saves_king, "save the king"))

	return cases


MOVE_SUITES = [
	("King Capture", _king_capture_cases),
	("Win Material", _win_material_cases),
	("King Safety", _king_safety_cases),
]


# ═══════════════════════════════════════════════════════════════════════
# Value-head (형세 판단) tests
# ═══════════════════════════════════════════════════════════════════════
# Each case is (label, state, expected_sign), expected_sign from side_to_move's
# perspective: +1 favors side_to_move, -1 favors the opponent, 0 is balanced.

def _evaluation_cases():
	cases = []

	# Starting position: balanced (Han bonus barely tips it Blue's way).
	s = Janggi()
	s.side_to_move = BLUE
	cases.append(("initial position (BLUE to move)", s, 0))

	# Red missing one chariot -> Blue ahead.
	s = Janggi()
	s.grid[0][0] = EMPTY
	s.side_to_move = BLUE
	cases.append(("RED down a chariot (BLUE to move)", s, +1))

	# Blue missing one chariot -> Red ahead.
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

	# Modest swap: Red has an extra cannon, Blue an extra horse -> Red slightly ahead.
	s = empty_grid_with_kings(BLUE)
	place_piece(s, 3, 4, RED, CANNON)
	place_piece(s, 6, 4, BLUE, HORSE)
	cases.append(("RED extra cannon vs BLUE extra horse (BLUE to move)", s, -1))

	return cases


def classical_eval(state: Janggi) -> float:
	"""Normalised material eval from side_to_move's perspective (tanh-squashed)."""
	stm = state.side_to_move
	opp = RED if stm == BLUE else BLUE
	raw = state.material_score(stm) - state.material_score(opp)
	return math.tanh(raw / 15.0)


def _sign_str(sign: int) -> str:
	return {1: "favors mover", -1: "favors opp", 0: "balanced"}[sign]


def _eval_ok(value: float, sign: int, tolerance: float) -> bool:
	if sign == 0:
		return abs(value) <= max(tolerance, 0.1)
	return value > tolerance if sign > 0 else value < -tolerance


# ═══════════════════════════════════════════════════════════════════════
# Suite runner
# ═══════════════════════════════════════════════════════════════════════

def _has_satisfying_move(state: Janggi, accept) -> bool:
	return any(accept(state, m) for m in state.legal_moves())


def run_suite(neural, classical, sims, tolerance):
	"""Run every test.  Returns ``(res, cats)``.

	``res['nm'][cat]`` / ``res['cm'][cat]`` are per-category lists of dicts with
	keys: ok, move, exp, desc, dt, n_legal, and (neural only) top, val.
	``res['ne']`` / ``res['ce']`` hold the value-head eval results.
	"""
	cats = [name for name, _ in MOVE_SUITES]
	res = dict(nm={}, cm={}, ne=[], ce=[])

	for cat, builder in MOVE_SUITES:
		for label, state, accept, expected in builder():
			# Turn coordinate mistakes in the suite into a loud failure.
			assert _has_satisfying_move(state, accept), f"Unsolvable case: {label!r}"
			n_legal = sum(1 for m in state.legal_moves() if m != PASS_MOVE)

			if neural is not None:
				t0 = time.time()
				move, visits = neural.select_move(state, temperature=0.0, add_noise=False)
				dt = time.time() - t0
				res["nm"].setdefault(cat, []).append(dict(
					ok=accept(state, move), move=describe_move(state, move),
					exp=expected, desc=label, dt=dt, n_legal=n_legal,
					top=_topk_moves(state, visits),
					val=_network_value(neural.network, neural.device, state),
				))

			t0 = time.time()
			cmove, _ = classical.select_move(state)
			dt = time.time() - t0
			res["cm"].setdefault(cat, []).append(dict(
				ok=accept(state, cmove), move=describe_move(state, cmove),
				exp=expected, desc=label, dt=dt, n_legal=n_legal,
			))

	for label, state, sign in _evaluation_cases():
		if neural is not None:
			v = _network_value(neural.network, neural.device, state)
			res["ne"].append(dict(ok=_eval_ok(v, sign, tolerance), val=v,
			                      exp=_sign_str(sign), desc=label))
		v = classical_eval(state)
		res["ce"].append(dict(ok=_eval_ok(v, sign, tolerance), val=v,
		                      exp=_sign_str(sign), desc=label))

	return res, cats


# ═══════════════════════════════════════════════════════════════════════
# Pretty printing
# ═══════════════════════════════════════════════════════════════════════

P, F = "PASS", "FAIL"


def _score(tests):
	return sum(t["ok"] for t in tests), len(tests)


def _fmt_top(top):
	if not top:
		return "-"
	return "  ".join(f"{m} {p*100:.0f}%" for m, p in top)


def print_detail(res, cats, has_neural):
	width = 78
	for cat in cats:
		nm = res["nm"].get(cat, []) if has_neural else []
		cm = res["cm"].get(cat, [])
		p, t = _score(nm if has_neural else cm)

		header = f"  {cat}  ({'Neural' if has_neural else 'Classical'}) [{p}/{t}]"
		if has_neural and nm:
			avg_ms = 1000 * sum(x["dt"] for x in nm) / len(nm)
			header += f"   avg {avg_ms:.0f} ms/move"
		print(f"\n{'─' * width}\n{header}\n{'─' * width}")

		for i in range(max(len(nm), len(cm))):
			n = nm[i] if i < len(nm) else None
			c = cm[i] if i < len(cm) else None
			base = n or c
			print(f"  [{i+1:2d}] {base['desc']}   ({base['n_legal']} legal, expected {base['exp']})")
			if n is not None:
				tag = P if n["ok"] else F
				print(f"        N {tag:4s} {n['move']:<8s} val={n['val']:+.2f}  "
				      f"top: {_fmt_top(n['top'])}")
			if c is not None:
				tag = P if c["ok"] else F
				print(f"        C {tag:4s} {c['move']:<8s}")

	ne = res["ne"] if has_neural else []
	ce = res["ce"]
	if ne or ce:
		p, t = _score(ne if has_neural else ce)
		print(f"\n{'─' * width}")
		print(f"  Evaluation  ({'Neural' if has_neural else 'Classical'}) [{p}/{t}]")
		print(f"{'─' * width}")
		for i in range(max(len(ne), len(ce))):
			n = ne[i] if i < len(ne) else None
			c = ce[i] if i < len(ce) else None
			base = n or c
			print(f"  [{i+1:2d}] {base['desc']}   (expected {base['exp']})")
			if n is not None:
				print(f"        N {(P if n['ok'] else F):4s} val={n['val']:+.3f}")
			if c is not None:
				print(f"        C {(P if c['ok'] else F):4s} val={c['val']:+.3f}")


def print_summary(res, cats, has_neural):
	"""Side-by-side summary table.  Returns (n_pass, n_total) for neural."""
	width = 62
	print(f"\n{'=' * width}\n  SUMMARY\n{'=' * width}")
	hdr = f"  {'Category':<20s}"
	if has_neural:
		hdr += f"{'Neural':>12s}"
	hdr += f"{'Classical':>14s}"
	print(hdr)
	print(f"  {'─' * (width - 2)}")

	n_p = n_t = c_p = c_t = 0
	rows = [(cat, res["nm"].get(cat, []), res["cm"].get(cat, [])) for cat in cats]
	rows.append(("Evaluation", res["ne"], res["ce"]))

	for label, neural_tests, classical_tests in rows:
		line = f"  {label:<20s}"
		if has_neural:
			p, t = _score(neural_tests)
			n_p += p; n_t += t
			line += f"{p:>5d}/{t:<5d} "
		p, t = _score(classical_tests)
		c_p += p; c_t += t
		line += f"{p:>6d}/{t:<5d}"
		print(line)

	print(f"  {'─' * (width - 2)}")
	line = f"  {'TOTAL':<20s}"
	if has_neural:
		pct = 100 * n_p / n_t if n_t else 0
		line += f"{n_p:>5d}/{n_t:<3d} ({pct:3.0f}%) "
	pct = 100 * c_p / c_t if c_t else 0
	line += f"{c_p:>4d}/{c_t:<3d} ({pct:3.0f}%)"
	print(line)
	return n_p, n_t


def print_comparison(res, cats):
	"""Neural vs Classical: agreement, both-pass, confidence, timing."""
	width = 78
	print(f"\n{'=' * width}\n  NEURAL vs CLASSICAL\n{'=' * width}")
	print(f"  {'Category':<16s} {'Agree':>9s} {'BothOK':>9s} "
	      f"{'Conf':>8s} {'N ms':>8s} {'C ms':>8s}")
	print(f"  {'─' * (width - 4)}")

	tot_agree = tot_both = tot_n = 0
	confs, n_times, c_times = [], [], []
	for cat in cats:
		nm = res["nm"].get(cat, [])
		cm = res["cm"].get(cat, [])
		if not nm or not cm:
			continue
		agree = sum(1 for n, c in zip(nm, cm) if n["move"] == c["move"])
		both = sum(1 for n, c in zip(nm, cm) if n["ok"] and c["ok"])
		cat_conf = [n["top"][0][1] for n in nm if n["top"]]
		avg_conf = sum(cat_conf) / len(cat_conf) if cat_conf else 0.0
		avg_n = 1000 * sum(n["dt"] for n in nm) / len(nm)
		avg_c = 1000 * sum(c["dt"] for c in cm) / len(cm)
		confs.extend(cat_conf)
		n_times.extend(n["dt"] for n in nm)
		c_times.extend(c["dt"] for c in cm)
		tot_agree += agree; tot_both += both; tot_n += len(nm)
		print(f"  {cat:<16s} {agree:>4d}/{len(nm):<4d} {both:>4d}/{len(nm):<4d} "
		      f"{avg_conf*100:>6.0f}%  {avg_n:>6.0f}  {avg_c:>6.0f}")

	if tot_n:
		print(f"  {'─' * (width - 4)}")
		avg_conf = sum(confs) / len(confs) if confs else 0.0
		avg_n = 1000 * sum(n_times) / len(n_times) if n_times else 0.0
		avg_c = 1000 * sum(c_times) / len(c_times) if c_times else 0.0
		print(f"  {'TOTAL':<16s} {tot_agree:>4d}/{tot_n:<4d} {tot_both:>4d}/{tot_n:<4d} "
		      f"{avg_conf*100:>6.0f}%  {avg_n:>6.0f}  {avg_c:>6.0f}")


# ═══════════════════════════════════════════════════════════════════════
# Training-history view
# ═══════════════════════════════════════════════════════════════════════

def _numbered_checkpoints(checkpoint_dir):
	"""Saved checkpoints other than latest.pt, sorted, for the progress view."""
	files = []
	for pat in ("model_iter_*.pt", "iter_*.pt", "iteration_*.pt", "*.pt"):
		files.extend(glob.glob(os.path.join(checkpoint_dir, pat)))
	files = sorted({f for f in files if os.path.basename(f) != "latest.pt"})
	return files


def run_history(checkpoint_dir, sims, depth, tolerance):
	files = _numbered_checkpoints(checkpoint_dir)
	if not files:
		print(f"\n  No numbered checkpoints found in {checkpoint_dir}")
		print("  (DeepJanggi keeps only latest.pt by default — nothing to chart.)")
		return

	cats = [name for name, _ in MOVE_SUITES]
	print(f"\n{'=' * 70}")
	print(f"  TRAINING PROGRESS  ({len(files)} checkpoints, {sims} sims/move)")
	print(f"{'=' * 70}")
	hdr = f"  {'Iter':>5s}"
	for cat in cats:
		hdr += f"  {cat[:8]:>8s}"
	hdr += f"  {'Eval':>6s}  {'Total':>12s}"
	print(hdr)
	print(f"  {'─' * 64}")

	classical = MinimaxAgent(depth=depth)
	for fpath in files:
		ckpt = torch.load(fpath, map_location="cpu")
		it = ckpt.get("iteration", "?")
		agent = NeuralAgent(simulations=sims, checkpoint_path=fpath)
		r, _ = run_suite(agent, classical, sims, tolerance)
		gp = gt = 0
		line = f"  {str(it):>5s}"
		for cat in cats:
			p, t = _score(r["nm"].get(cat, []))
			gp += p; gt += t
			line += f"    {p:>2d}/{t:<2d}  "
		p, t = _score(r["ne"])
		gp += p; gt += t
		pct = 100 * gp / gt if gt else 0
		print(f"{line}  {p:>2d}/{t:<2d}   {gp:>2d}/{gt:<2d} ({pct:4.0f}%)")

	# Classical baseline row.
	r, _ = run_suite(None, classical, sims, tolerance)
	gp = gt = 0
	line = f"  {'base':>5s}"
	for cat in cats:
		p, t = _score(r["cm"].get(cat, []))
		gp += p; gt += t
		line += f"    {p:>2d}/{t:<2d}  "
	p, t = _score(r["ce"])
	gp += p; gt += t
	pct = 100 * gp / gt if gt else 0
	print(f"  {'─' * 64}")
	print(f"{line}  {p:>2d}/{t:<2d}   {gp:>2d}/{gt:<2d} ({pct:4.0f}%)  (classical baseline)")


# ═══════════════════════════════════════════════════════════════════════
# Head-to-head match
# ═══════════════════════════════════════════════════════════════════════

def play_match(hero, villain, opponent_label, num_games, max_ply,
               opening_temp_plies=8):
	"""Play *num_games* between the neural hero and the baseline opponent.

	Colors alternate every game so neither side keeps the first-move edge.  A
	small opening temperature stops the games collapsing into one repeated line.
	Returns ``(wins, losses, draws)`` from the hero's perspective.
	"""
	width = 78
	print(f"\n{'=' * width}")
	print(f"  MATCH  Neural vs {opponent_label}   {num_games} games (max {max_ply} plies)")
	print(f"{'=' * width}")
	print(f"  {'#':>3s}  {'Hero':>6s}  {'Winner':>6s}  {'Outcome':>8s}  "
	      f"{'Plies':>5s}  {'Time':>7s}")
	print(f"  {'─' * (width - 4)}")

	wins = losses = draws = 0
	match_t0 = time.time()
	for g in range(num_games):
		hero_is_red = (g % 2 == 1)
		hero_color = RED if hero_is_red else BLUE
		state = Janggi()
		g_t0 = time.time()
		while not state.is_terminal():
			hero_to_move = (state.side_to_move == RED) == hero_is_red
			if hero_to_move:
				temp = 0.6 if state.ply < opening_temp_plies else 0.0
				move, _ = hero.select_move(state, temperature=temp, add_noise=False)
			else:
				move, _ = villain.select_move(state)
			if move is None:
				break
			state.apply(move)

		winner = state.winner
		if winner == hero_color:
			wins += 1; outcome = "WIN"
		elif winner == -1 or winner is None:
			draws += 1; outcome = "DRAW"
		else:
			losses += 1; outcome = "LOSS"

		dt = time.time() - g_t0
		print(f"  {g+1:>3d}  {PLAYER_NAMES[hero_color]:>6s}  "
		      f"{PLAYER_NAMES.get(winner, 'draw'):>6s}  {outcome:>8s}  "
		      f"{state.ply:>5d}  {dt:>6.1f}s")

	total = wins + losses + draws
	score = wins + 0.5 * draws
	win_rate = 100 * score / total if total else 0.0
	print(f"  {'─' * (width - 4)}")
	print(f"  RESULT  W={wins}  L={losses}  D={draws}   "
	      f"score {score:.1f}/{total}   win rate {win_rate:.1f}%   "
	      f"({time.time() - match_t0:.1f}s)")
	print(f"  Verdict: {'AI is competitive' if win_rate >= 55 else 'needs more training'}.")
	return wins, losses, draws


def report_checkpoint(path: str) -> None:
	if not path or not os.path.isfile(path):
		print(f"  Checkpoint       : not found ({path or '(none)'})")
		return
	state = torch.load(path, map_location="cpu")
	print(f"  Checkpoint       : {path} "
	      f"(iteration={state.get('iteration', '?')}, "
	      f"global_step={state.get('global_step', '?')})")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
	parser = argparse.ArgumentParser(
		description="DeepJanggi — ground-truth validation",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument("--checkpoint", default=LATEST_CHECKPOINT,
	                    help="Checkpoint under test.")
	parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR,
	                    help="Directory of numbered checkpoints (for --history).")
	parser.add_argument("--simulations", type=int, default=MCTS_SIMULATIONS_VALIDATE,
	                    help="MCTS simulations per test/match move.")
	parser.add_argument("--minimax-depth", type=int, default=3,
	                    help="Search depth (plies) for the classical minimax baseline.")
	parser.add_argument("--eval-tolerance", type=float, default=0.05,
	                    help="Value-head magnitude threshold for a non-zero sign.")
	parser.add_argument("--history", action="store_true",
	                    help="Evaluate every numbered checkpoint and chart progress.")
	parser.add_argument("--games", type=int, default=10,
	                    help="Head-to-head games to play (0 to skip the match).")
	parser.add_argument("--opponent", choices=("random", "minimax", "checkpoint"),
	                    default="minimax",
	                    help="Match opponent. The test suite always uses minimax as "
	                         "the classical baseline.")
	parser.add_argument("--opponent-path", default="",
	                    help="Checkpoint for --opponent checkpoint.")
	parser.add_argument("--max-ply", type=int, default=200,
	                    help="Per-game ply cap during the match (shorter than training).")
	args = parser.parse_args()

	# Cap match length for this run only; encoder.py still uses the training-time
	# MAX_PLY so the network sees the normalization it learned.
	board_module.MAX_PLY = args.max_ply

	print(f"{'=' * 56}")
	print(f"  DeepJanggi Ground Truth Validation")
	print(f"{'=' * 56}")
	report_checkpoint(args.checkpoint)
	print(f"  MCTS simulations : {args.simulations}")
	print(f"  Minimax depth    : {args.minimax_depth}")

	has_neural = bool(args.checkpoint) and os.path.isfile(args.checkpoint)
	neural = NeuralAgent(simulations=args.simulations,
	                     checkpoint_path=args.checkpoint) if has_neural else None
	if not has_neural:
		print("  (no checkpoint — running the classical engine only)")
	classical = MinimaxAgent(depth=args.minimax_depth)

	t0 = time.time()
	results, cats = run_suite(neural, classical, args.simulations, args.eval_tolerance)
	elapsed = time.time() - t0

	print_detail(results, cats, has_neural)
	n_pass, n_total = print_summary(results, cats, has_neural)
	if has_neural:
		print_comparison(results, cats)
	print(f"\n  Suite completed in {elapsed:.1f}s")

	# Head-to-head match.
	if has_neural and args.games > 0:
		if args.opponent == "random":
			villain, label = RandomAgent(), "Random"
		elif args.opponent == "checkpoint":
			report_checkpoint(args.opponent_path)
			villain = NeuralAgent(simulations=args.simulations,
			                      checkpoint_path=args.opponent_path)
			label = "Checkpoint"
		else:
			villain, label = classical, f"Minimax(d{args.minimax_depth})"
		play_match(neural, villain, label, args.games, args.max_ply)

	# Training history.
	if args.history:
		run_history(args.checkpoint_dir, args.simulations,
		            args.minimax_depth, args.eval_tolerance)

	# CI exit code: 0 if the neural engine passes >=60% of the suite.
	if has_neural and n_total > 0:
		sys.exit(0 if n_pass / n_total >= 0.6 else 1)


if __name__ == "__main__":
	main()
