"""Translate a Janggi state into a torch tensor for the network.

Layout (channel-first, shape = (INPUT_PLANES, ROWS, COLS)):
  - 14 piece planes ordered as red K/A/C/N/H/E/P, blue K/A/C/N/H/E/P
  - 1 plane filled with 1.0 if the side-to-move is BLUE, else 0.0
  - 1 plane with normalized ply (ply / MAX_PLY)

We deliberately keep board coordinates in a fixed orientation so action indices
match the underlying board indices regardless of who is moving.
"""

from __future__ import annotations

import numpy as np

from .board import Janggi
from .config import (
	BLUE,
	COLS,
	EMPTY,
	INPUT_PLANES,
	MAX_PLY,
	NUM_PIECE_TYPES,
	ROWS,
	piece_color,
	piece_type,
)


def encode_state(state: Janggi) -> np.ndarray:
	"""Return a (INPUT_PLANES, ROWS, COLS) float32 tensor."""
	planes = np.zeros((INPUT_PLANES, ROWS, COLS), dtype=np.float32)
	for r in range(ROWS):
		for c in range(COLS):
			piece = state.grid[r][c]
			if piece == EMPTY:
				continue
			pc = piece_color(piece)
			pt = piece_type(piece)
			plane_idx = (pt - 1) + (NUM_PIECE_TYPES if pc == BLUE else 0)
			planes[plane_idx, r, c] = 1.0
	if state.side_to_move == BLUE:
		planes[14, :, :] = 1.0
	planes[15, :, :] = min(state.ply / MAX_PLY, 1.0)
	return planes
