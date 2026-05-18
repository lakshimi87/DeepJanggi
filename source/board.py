"""Janggi board state and move generation.

Implements the standard Korean chess rules: piece movement, palace constraints,
horse/elephant blocking, cannon screen-jump, soldier promotion to diagonal palace
travel, capturing the opposing king as the win condition, and the pass action.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

from .config import (
	ADVISOR,
	BLUE,
	BLUE_PALACE,
	CANNON,
	CHARIOT,
	COLS,
	ELEPHANT,
	EMPTY,
	HAN_BONUS,
	HORSE,
	KING,
	MAX_PLY,
	NUM_SQUARES,
	PALACE_DIAGONAL_NEIGHBORS,
	PALACE_DIAGONAL_NODES,
	PASS_ACTION,
	PIECE_VALUE,
	RED,
	RED_PALACE,
	ROWS,
	SOLDIER,
	make_piece,
	piece_color,
	piece_type,
)


Move = Tuple[int, int, int, int]  # (from_row, from_col, to_row, to_col); pass = (-1,-1,-1,-1).
PASS_MOVE: Move = (-1, -1, -1, -1)


def square_index(row: int, col: int) -> int:
	return row * COLS + col


def index_square(idx: int) -> Tuple[int, int]:
	return idx // COLS, idx % COLS


def move_to_action(move: Move) -> int:
	"""Encode a move into a flat action index in [0, TOTAL_ACTIONS)."""
	if move == PASS_MOVE:
		return PASS_ACTION
	fr, fc, tr, tc = move
	return square_index(fr, fc) * NUM_SQUARES + square_index(tr, tc)


def action_to_move(action: int) -> Move:
	if action == PASS_ACTION:
		return PASS_MOVE
	src, dst = divmod(action, NUM_SQUARES)
	fr, fc = index_square(src)
	tr, tc = index_square(dst)
	return (fr, fc, tr, tc)


def in_palace(color: int, row: int, col: int) -> bool:
	rmin, rmax, cmin, cmax = RED_PALACE if color == RED else BLUE_PALACE
	return rmin <= row <= rmax and cmin <= col <= cmax


def in_bounds(row: int, col: int) -> bool:
	return 0 <= row < ROWS and 0 <= col < COLS


# Initial standard setup (inner-elephant variant): symmetric layout, both sides identical.
INITIAL_SETUP: List[List[int]] = [
	[
		make_piece(RED, CHARIOT), make_piece(RED, ELEPHANT), make_piece(RED, HORSE),
		make_piece(RED, ADVISOR), EMPTY, make_piece(RED, ADVISOR),
		make_piece(RED, HORSE), make_piece(RED, ELEPHANT), make_piece(RED, CHARIOT),
	],
	[EMPTY] * 4 + [make_piece(RED, KING)] + [EMPTY] * 4,
	[EMPTY, make_piece(RED, CANNON), EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, make_piece(RED, CANNON), EMPTY],
	[
		make_piece(RED, SOLDIER), EMPTY, make_piece(RED, SOLDIER), EMPTY,
		make_piece(RED, SOLDIER), EMPTY, make_piece(RED, SOLDIER), EMPTY,
		make_piece(RED, SOLDIER),
	],
	[EMPTY] * COLS,
	[EMPTY] * COLS,
	[
		make_piece(BLUE, SOLDIER), EMPTY, make_piece(BLUE, SOLDIER), EMPTY,
		make_piece(BLUE, SOLDIER), EMPTY, make_piece(BLUE, SOLDIER), EMPTY,
		make_piece(BLUE, SOLDIER),
	],
	[EMPTY, make_piece(BLUE, CANNON), EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, make_piece(BLUE, CANNON), EMPTY],
	[EMPTY] * 4 + [make_piece(BLUE, KING)] + [EMPTY] * 4,
	[
		make_piece(BLUE, CHARIOT), make_piece(BLUE, ELEPHANT), make_piece(BLUE, HORSE),
		make_piece(BLUE, ADVISOR), EMPTY, make_piece(BLUE, ADVISOR),
		make_piece(BLUE, HORSE), make_piece(BLUE, ELEPHANT), make_piece(BLUE, CHARIOT),
	],
]


def swap_flank(grid: List[List[int]], row: int, flank: str) -> None:
	"""Swap the horse and elephant on one back-rank flank in place.

	Janggi opening choice: each side independently picks left/right flank as 마상 or 상마.
	"""
	if flank == "left":
		c1, c2 = 1, 2
	elif flank == "right":
		c1, c2 = 6, 7
	else:
		raise ValueError(f"flank must be 'left' or 'right', got {flank!r}")
	grid[row][c1], grid[row][c2] = grid[row][c2], grid[row][c1]


def randomize_flank_setup(state: "Janggi", rng: Optional[random.Random] = None) -> None:
	"""Randomly swap horse/elephant on each flank for each side (4 independent coin flips)."""
	r = rng or random
	for back_row in (0, 9):
		for flank in ("left", "right"):
			if r.random() < 0.5:
				swap_flank(state.grid, back_row, flank)


@dataclass
class Janggi:
	"""Mutable Janggi game state with copy/clone for search."""

	grid: List[List[int]] = field(default_factory=lambda: copy.deepcopy(INITIAL_SETUP))
	side_to_move: int = BLUE  # Blue (Han) traditionally moves first in our convention.
	ply: int = 0
	winner: Optional[int] = None  # None until terminal; -1 indicates draw.
	consecutive_passes: int = 0

	# ------------------------------------------------------------------
	# Cloning
	# ------------------------------------------------------------------
	def clone(self) -> "Janggi":
		new = Janggi.__new__(Janggi)
		new.grid = [row[:] for row in self.grid]
		new.side_to_move = self.side_to_move
		new.ply = self.ply
		new.winner = self.winner
		new.consecutive_passes = self.consecutive_passes
		return new

	# ------------------------------------------------------------------
	# Status
	# ------------------------------------------------------------------
	def is_terminal(self) -> bool:
		return self.winner is not None

	def material_score(self, color: int) -> float:
		score = 0.0
		for row in self.grid:
			for piece in row:
				if piece and piece_color(piece) == color:
					score += PIECE_VALUE[piece_type(piece)]
		if color == BLUE:
			score += HAN_BONUS
		return score

	def find_king(self, color: int) -> Optional[Tuple[int, int]]:
		target = make_piece(color, KING)
		for r in range(ROWS):
			for c in range(COLS):
				if self.grid[r][c] == target:
					return r, c
		return None

	# ------------------------------------------------------------------
	# Move generation
	# ------------------------------------------------------------------
	def legal_moves(self) -> List[Move]:
		"""All legal moves for the current side, including the pass action."""
		moves: List[Move] = []
		me = self.side_to_move
		for r in range(ROWS):
			for c in range(COLS):
				piece = self.grid[r][c]
				if piece == EMPTY or piece_color(piece) != me:
					continue
				ptype = piece_type(piece)
				if ptype == KING or ptype == ADVISOR:
					self._gen_palace_step(r, c, me, moves)
				elif ptype == CHARIOT:
					self._gen_chariot(r, c, me, moves)
				elif ptype == CANNON:
					self._gen_cannon(r, c, me, moves)
				elif ptype == HORSE:
					self._gen_horse(r, c, me, moves)
				elif ptype == ELEPHANT:
					self._gen_elephant(r, c, me, moves)
				elif ptype == SOLDIER:
					self._gen_soldier(r, c, me, moves)
		# Pass is always available.
		moves.append(PASS_MOVE)
		return moves

	# Palace-bound 1-step movers (king + advisor) ----------------------
	def _gen_palace_step(self, r: int, c: int, me: int, out: List[Move]) -> None:
		palace = RED_PALACE if me == RED else BLUE_PALACE
		rmin, rmax, cmin, cmax = palace
		for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
			nr, nc = r + dr, c + dc
			if rmin <= nr <= rmax and cmin <= nc <= cmax:
				if self._can_land(nr, nc, me):
					out.append((r, c, nr, nc))
		if (r, c) in PALACE_DIAGONAL_NODES[me]:
			for nr, nc in PALACE_DIAGONAL_NEIGHBORS[(r, c)]:
				# Only stay on our own palace's diagonal partners.
				if (nr, nc) in PALACE_DIAGONAL_NODES[me] and self._can_land(nr, nc, me):
					out.append((r, c, nr, nc))

	# Chariot (rook-like, plus palace diagonals) ----------------------
	def _gen_chariot(self, r: int, c: int, me: int, out: List[Move]) -> None:
		for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
			nr, nc = r + dr, c + dc
			while in_bounds(nr, nc):
				occ = self.grid[nr][nc]
				if occ == EMPTY:
					out.append((r, c, nr, nc))
				else:
					if piece_color(occ) != me:
						out.append((r, c, nr, nc))
					break
				nr += dr
				nc += dc
		# Palace diagonal slides: only along the palace's drawn diagonal lines.
		self._gen_palace_diagonal_slide(r, c, me, out, allow_screen=False)

	# Cannon (jumps a single non-cannon screen, never captures cannons) ---
	def _gen_cannon(self, r: int, c: int, me: int, out: List[Move]) -> None:
		for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
			nr, nc = r + dr, c + dc
			screen_found = False
			while in_bounds(nr, nc):
				occ = self.grid[nr][nc]
				if not screen_found:
					if occ != EMPTY:
						if piece_type(occ) == CANNON:
							break  # Cannon cannot use another cannon as screen.
						screen_found = True
				else:
					if occ == EMPTY:
						out.append((r, c, nr, nc))
					else:
						if piece_color(occ) != me and piece_type(occ) != CANNON:
							out.append((r, c, nr, nc))
						break
				nr += dr
				nc += dc
		self._gen_palace_diagonal_slide(r, c, me, out, allow_screen=True)

	def _gen_palace_diagonal_slide(
		self, r: int, c: int, me: int, out: List[Move], allow_screen: bool
	) -> None:
		"""Slide along palace diagonals, used by chariot (no screen) and cannon (screen)."""
		# A palace diagonal slide line passes through (r0, r0+1, r0+2) where r0 is the palace top.
		# Two diagonals per palace: top-left to bottom-right and top-right to bottom-left.
		for palace_color in (RED, BLUE):
			rmin = 0 if palace_color == RED else 7
			# Diagonals as ordered cell lists.
			diagonals = [
				[(rmin, 3), (rmin + 1, 4), (rmin + 2, 5)],
				[(rmin, 5), (rmin + 1, 4), (rmin + 2, 3)],
			]
			for line in diagonals:
				if (r, c) not in line:
					continue
				idx = line.index((r, c))
				for direction in (-1, 1):
					i = idx + direction
					screen_found = False
					while 0 <= i < len(line):
						nr, nc = line[i]
						occ = self.grid[nr][nc]
						if not allow_screen:
							if occ == EMPTY:
								out.append((r, c, nr, nc))
							else:
								if piece_color(occ) != me:
									out.append((r, c, nr, nc))
								break
						else:
							if not screen_found:
								if occ != EMPTY:
									if piece_type(occ) == CANNON:
										break
									screen_found = True
							else:
								if occ == EMPTY:
									out.append((r, c, nr, nc))
								else:
									if piece_color(occ) != me and piece_type(occ) != CANNON:
										out.append((r, c, nr, nc))
									break
						i += direction

	# Horse: 1 orthogonal + 1 diagonal outward, blocked by leg cell -----
	def _gen_horse(self, r: int, c: int, me: int, out: List[Move]) -> None:
		patterns = [
			((-1, 0), [(-2, -1), (-2, 1)]),
			((1, 0), [(2, -1), (2, 1)]),
			((0, -1), [(-1, -2), (1, -2)]),
			((0, 1), [(-1, 2), (1, 2)]),
		]
		for leg, targets in patterns:
			lr, lc = r + leg[0], c + leg[1]
			if not in_bounds(lr, lc) or self.grid[lr][lc] != EMPTY:
				continue
			for dr, dc in targets:
				nr, nc = r + dr, c + dc
				if in_bounds(nr, nc) and self._can_land(nr, nc, me):
					out.append((r, c, nr, nc))

	# Elephant: 1 orthogonal + 2 diagonal outward; both intermediate cells empty.
	def _gen_elephant(self, r: int, c: int, me: int, out: List[Move]) -> None:
		patterns = [
			((-1, 0), (-2, -1), (-3, -2)),
			((-1, 0), (-2, 1), (-3, 2)),
			((1, 0), (2, -1), (3, -2)),
			((1, 0), (2, 1), (3, 2)),
			((0, -1), (-1, -2), (-2, -3)),
			((0, -1), (1, -2), (2, -3)),
			((0, 1), (-1, 2), (-2, 3)),
			((0, 1), (1, 2), (2, 3)),
		]
		for s1, s2, dest in patterns:
			r1, c1 = r + s1[0], c + s1[1]
			r2, c2 = r + s2[0], c + s2[1]
			rd, cd = r + dest[0], c + dest[1]
			if not in_bounds(rd, cd):
				continue
			if not in_bounds(r1, c1) or not in_bounds(r2, c2):
				continue
			if self.grid[r1][c1] != EMPTY or self.grid[r2][c2] != EMPTY:
				continue
			if self._can_land(rd, cd, me):
				out.append((r, c, rd, cd))

	# Soldier: forward / sideways; in enemy palace, also forward-diagonal along palace lines.
	def _gen_soldier(self, r: int, c: int, me: int, out: List[Move]) -> None:
		forward = 1 if me == RED else -1  # Red moves down, Blue moves up.
		moves = [(r + forward, c), (r, c - 1), (r, c + 1)]
		for nr, nc in moves:
			if in_bounds(nr, nc) and self._can_land(nr, nc, me):
				out.append((r, c, nr, nc))
		# Diagonal forward in enemy palace.
		enemy_palace_color = BLUE if me == RED else RED
		if in_palace(enemy_palace_color, r, c):
			# Compose the palace diagonals where soldier sits and pick the forward one only.
			diagonal_targets = []
			if (r, c) in PALACE_DIAGONAL_NODES[enemy_palace_color]:
				for nr, nc in PALACE_DIAGONAL_NEIGHBORS[(r, c)]:
					if (nr, nc) in PALACE_DIAGONAL_NODES[enemy_palace_color]:
						diagonal_targets.append((nr, nc))
			for nr, nc in diagonal_targets:
				# Forward = increases row for red, decreases row for blue.
				if (me == RED and nr > r) or (me == BLUE and nr < r):
					if self._can_land(nr, nc, me):
						out.append((r, c, nr, nc))

	def _can_land(self, r: int, c: int, me: int) -> bool:
		occ = self.grid[r][c]
		return occ == EMPTY or piece_color(occ) != me

	# ------------------------------------------------------------------
	# Move application
	# ------------------------------------------------------------------
	def apply(self, move: Move) -> None:
		assert not self.is_terminal()
		if move == PASS_MOVE:
			self.consecutive_passes += 1
			self._switch_turn()
			self._maybe_finalize()
			return
		fr, fc, tr, tc = move
		captured = self.grid[tr][tc]
		self.grid[tr][tc] = self.grid[fr][fc]
		self.grid[fr][fc] = EMPTY
		self.consecutive_passes = 0
		# King capture wins immediately.
		if captured and piece_type(captured) == KING:
			self.winner = self.side_to_move
			self.ply += 1
			return
		self._switch_turn()
		self._maybe_finalize()

	def _switch_turn(self) -> None:
		self.side_to_move = RED if self.side_to_move == BLUE else BLUE
		self.ply += 1

	def _maybe_finalize(self) -> None:
		# Two consecutive passes => terminate, score by material.
		if self.consecutive_passes >= 2 or self.ply >= MAX_PLY:
			self.winner = self._score_winner()

	def _score_winner(self) -> int:
		red_score = self.material_score(RED)
		blue_score = self.material_score(BLUE)
		if red_score > blue_score:
			return RED
		if blue_score > red_score:
			return BLUE
		return -1  # Draw sentinel.

	# ------------------------------------------------------------------
	# Convenience
	# ------------------------------------------------------------------
	def piece_at(self, r: int, c: int) -> int:
		return self.grid[r][c]

	def reward_for(self, color: int) -> float:
		"""Game-end reward in [-1, 1] from the perspective of the given color."""
		assert self.is_terminal()
		if self.winner == -1:
			return 0.0
		return 1.0 if self.winner == color else -1.0

	def __repr__(self) -> str:  # pragma: no cover - debugging aid
		symbol = {
			(RED, KING): "K", (RED, ADVISOR): "A", (RED, CHARIOT): "C",
			(RED, CANNON): "N", (RED, HORSE): "H", (RED, ELEPHANT): "E", (RED, SOLDIER): "P",
			(BLUE, KING): "k", (BLUE, ADVISOR): "a", (BLUE, CHARIOT): "c",
			(BLUE, CANNON): "n", (BLUE, HORSE): "h", (BLUE, ELEPHANT): "e", (BLUE, SOLDIER): "p",
		}
		lines = []
		for r in range(ROWS):
			row = []
			for c in range(COLS):
				p = self.grid[r][c]
				if p == EMPTY:
					row.append(".")
				else:
					row.append(symbol[(piece_color(p), piece_type(p))])
			lines.append(" ".join(row))
		who = "BLUE" if self.side_to_move == BLUE else "RED"
		return f"<Janggi ply={self.ply} turn={who}>\n" + "\n".join(lines)
