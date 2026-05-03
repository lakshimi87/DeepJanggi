"""Project-wide constants for board geometry, piece codes, action space, and training."""

import os


# Board geometry: 10 rows x 9 cols (Korean chess standard).
ROWS = 10
COLS = 9
NUM_SQUARES = ROWS * COLS

# Players.
RED = 0
BLUE = 1
PLAYER_NAMES = {RED: "red", BLUE: "blue"}

# Piece type codes (unsigned). 0 = empty.
EMPTY = 0
KING = 1
ADVISOR = 2
CHARIOT = 3
CANNON = 4
HORSE = 5
ELEPHANT = 6
SOLDIER = 7
NUM_PIECE_TYPES = 7

# Encoded piece values placed on the board: encode color into the piece id.
# Red pieces: 1..7, Blue pieces: 8..14. Use helpers below to manipulate.
def make_piece(color: int, ptype: int) -> int:
	return color * NUM_PIECE_TYPES + ptype


def piece_color(piece: int) -> int:
	return 0 if piece <= NUM_PIECE_TYPES else 1


def piece_type(piece: int) -> int:
	return ((piece - 1) % NUM_PIECE_TYPES) + 1 if piece else 0


# Palace bounds: (row_min, row_max, col_min, col_max) inclusive for each side.
RED_PALACE = (0, 2, 3, 5)
BLUE_PALACE = (7, 9, 3, 5)

# Palace diagonal node coordinates (cells where pieces may travel along diagonals).
PALACE_DIAGONAL_NODES = {
	RED: {(0, 3), (0, 5), (1, 4), (2, 3), (2, 5)},
	BLUE: {(7, 3), (7, 5), (8, 4), (9, 3), (9, 5)},
}

# Palace diagonal adjacency: from a node, which palace nodes are 1 step diagonally connected.
PALACE_DIAGONAL_NEIGHBORS = {
	(0, 3): [(1, 4)],
	(0, 5): [(1, 4)],
	(1, 4): [(0, 3), (0, 5), (2, 3), (2, 5), (7, 3), (7, 5), (9, 3), (9, 5)],  # only red half used; blue uses (8,4)
	(2, 3): [(1, 4)],
	(2, 5): [(1, 4)],
	(7, 3): [(8, 4)],
	(7, 5): [(8, 4)],
	(8, 4): [(7, 3), (7, 5), (9, 3), (9, 5)],
	(9, 3): [(8, 4)],
	(9, 5): [(8, 4)],
}

# Material values used for tie-breaking and reward shaping. Standard Janggi scoring.
PIECE_VALUE = {
	KING: 0,
	CHARIOT: 13,
	CANNON: 7,
	HORSE: 5,
	ADVISOR: 3,
	ELEPHANT: 3,
	SOLDIER: 2,
}

# Han bonus given to Blue (Han) when scoring a draw.
HAN_BONUS = 1.5

# Action space: a "from->to" plane plus a single PASS action.
PASS_ACTION = NUM_SQUARES * NUM_SQUARES
TOTAL_ACTIONS = NUM_SQUARES * NUM_SQUARES + 1

# Game termination thresholds.
MAX_PLY = 400  # Hard cap on move count to guarantee terminal evaluation.

# Network input planes: 14 piece planes + 1 side-to-move plane + 1 ply progress plane.
INPUT_PLANES = 14 + 1 + 1

# Filesystem layout.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(ROOT_DIR, "res")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
LATEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "latest.pt")

# Training hyper-parameters (kept here so all entry points share the same defaults).
NUM_RES_BLOCKS = 6
NUM_FILTERS = 96

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
REPLAY_CAPACITY = 50_000
TRAIN_STEPS_PER_ITERATION = 200
GAMES_PER_ITERATION = 8
MCTS_SIMULATIONS_TRAIN = 80
MCTS_SIMULATIONS_PLAY = 200
MCTS_SIMULATIONS_VALIDATE = 80

# MCTS parameters.
PUCT_C = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPS = 0.25
TEMPERATURE_MOVES = 30  # First moves use temperature 1; afterwards greedy.
