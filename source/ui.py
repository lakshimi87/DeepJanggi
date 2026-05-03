"""Pygame-CE GUI for playing Janggi against the AI.

The board is always rendered with the human player on the bottom side, so the
underlying logical board (row 0 = top = red) is mirrored when the human plays red.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pygame

from .agent import NeuralAgent
from .board import Janggi, Move, PASS_MOVE
from .config import (
	BLUE,
	COLS,
	EMPTY,
	LATEST_CHECKPOINT,
	MCTS_SIMULATIONS_PLAY,
	NUM_PIECE_TYPES,
	PLAYER_NAMES,
	RED,
	RES_DIR,
	ROWS,
	make_piece,
	piece_color,
	piece_type,
	ADVISOR,
	CANNON,
	CHARIOT,
	ELEPHANT,
	HORSE,
	KING,
	SOLDIER,
)


# Layout constants.
CELL_SIZE = 64
MARGIN_LEFT = 56
MARGIN_TOP = 56
STATUS_HEIGHT = 80
BOARD_WIDTH = (COLS - 1) * CELL_SIZE
BOARD_HEIGHT = (ROWS - 1) * CELL_SIZE
WINDOW_WIDTH = MARGIN_LEFT * 2 + BOARD_WIDTH
WINDOW_HEIGHT = MARGIN_TOP + BOARD_HEIGHT + STATUS_HEIGHT + 24
PIECE_SIZE = int(CELL_SIZE * 0.92)

# Color palette.
BACKGROUND = (236, 200, 132)
LINE_COLOR = (40, 28, 18)
HIGHLIGHT_SELECT = (60, 120, 220)
HIGHLIGHT_MOVE = (40, 180, 80)
HIGHLIGHT_CAPTURE = (220, 60, 60)
TEXT_COLOR = (20, 20, 20)
STATUS_BG = (220, 178, 110)


_PIECE_FILE_NAMES: Dict[Tuple[int, int], str] = {
	(RED, KING): "red_king.svg",
	(RED, ADVISOR): "red_advisor.svg",
	(RED, CHARIOT): "red_chariot.svg",
	(RED, CANNON): "red_cannon.svg",
	(RED, HORSE): "red_horse.svg",
	(RED, ELEPHANT): "red_elephant.svg",
	(RED, SOLDIER): "red_pawn.svg",
	(BLUE, KING): "blue_king.svg",
	(BLUE, ADVISOR): "blue_advisor.svg",
	(BLUE, CHARIOT): "blue_chariot.svg",
	(BLUE, CANNON): "blue_cannon.svg",
	(BLUE, HORSE): "blue_horse.svg",
	(BLUE, ELEPHANT): "blue_elephant.svg",
	(BLUE, SOLDIER): "blue_pawn.svg",
}


def _load_piece_surfaces() -> Dict[Tuple[int, int], pygame.Surface]:
	surfaces: Dict[Tuple[int, int], pygame.Surface] = {}
	for key, fname in _PIECE_FILE_NAMES.items():
		path = os.path.join(RES_DIR, fname)
		surf = _load_svg(path, PIECE_SIZE)
		surfaces[key] = surf
	return surfaces


def _load_svg(path: str, size: int) -> pygame.Surface:
	"""Load an SVG to a surface of the requested square size, falling back gracefully."""
	# pygame-ce >= 2.5 provides load_sized_svg; older versions only support load.
	loader = getattr(pygame.image, "load_sized_svg", None)
	if loader is not None:
		try:
			surf = loader(path, (size, size))
			return surf.convert_alpha()
		except Exception:
			pass
	surf = pygame.image.load(path).convert_alpha()
	return pygame.transform.smoothscale(surf, (size, size))


@dataclass
class Renderer:
	human_color: int
	piece_surfaces: Dict[Tuple[int, int], pygame.Surface]
	font_small: pygame.font.Font
	font_large: pygame.font.Font

	def board_to_screen(self, row: int, col: int) -> Tuple[int, int]:
		"""Convert a logical board (row, col) into pixel coordinates of an intersection."""
		if self.human_color == BLUE:
			disp_row, disp_col = row, col
		else:
			disp_row, disp_col = ROWS - 1 - row, COLS - 1 - col
		x = MARGIN_LEFT + disp_col * CELL_SIZE
		y = MARGIN_TOP + disp_row * CELL_SIZE
		return x, y

	def screen_to_board(self, x: int, y: int) -> Optional[Tuple[int, int]]:
		col = round((x - MARGIN_LEFT) / CELL_SIZE)
		row = round((y - MARGIN_TOP) / CELL_SIZE)
		if 0 <= row < ROWS and 0 <= col < COLS:
			# Reverse the orientation flip if needed.
			if self.human_color == RED:
				row = ROWS - 1 - row
				col = COLS - 1 - col
			# Snap-tolerance check.
			cx, cy = self.board_to_screen(row, col)
			if abs(x - cx) <= CELL_SIZE * 0.45 and abs(y - cy) <= CELL_SIZE * 0.45:
				return row, col
		return None

	# ------------------------------------------------------------------
	# Drawing
	# ------------------------------------------------------------------
	def draw(
		self,
		surface: pygame.Surface,
		state: Janggi,
		selected: Optional[Tuple[int, int]],
		legal_targets: List[Tuple[int, int]],
		status_message: str,
	) -> None:
		surface.fill(BACKGROUND)
		self._draw_board(surface)
		self._draw_palaces(surface)
		if selected is not None:
			self._highlight_cell(surface, selected, HIGHLIGHT_SELECT, width=4)
		for tr, tc in legal_targets:
			occ = state.grid[tr][tc]
			color = HIGHLIGHT_CAPTURE if occ != EMPTY else HIGHLIGHT_MOVE
			self._draw_target_dot(surface, (tr, tc), color)
		self._draw_pieces(surface, state)
		self._draw_status(surface, state, status_message)

	def _draw_board(self, surface: pygame.Surface) -> None:
		# Vertical lines (with river break? Janggi has no river: full-height vertical lines).
		for c in range(COLS):
			x, y0 = self.board_to_screen(0, c)
			_, y1 = self.board_to_screen(ROWS - 1, c)
			pygame.draw.line(surface, LINE_COLOR, (x, y0), (x, y1), 2)
		# Horizontal lines.
		for r in range(ROWS):
			x0, y = self.board_to_screen(r, 0)
			x1, _ = self.board_to_screen(r, COLS - 1)
			pygame.draw.line(surface, LINE_COLOR, (x0, y), (x1, y), 2)

	def _draw_palaces(self, surface: pygame.Surface) -> None:
		for r0 in (0, 7):
			tl = self.board_to_screen(r0, 3)
			tr = self.board_to_screen(r0, 5)
			bl = self.board_to_screen(r0 + 2, 3)
			br = self.board_to_screen(r0 + 2, 5)
			pygame.draw.line(surface, LINE_COLOR, tl, br, 2)
			pygame.draw.line(surface, LINE_COLOR, tr, bl, 2)

	def _draw_pieces(self, surface: pygame.Surface, state: Janggi) -> None:
		for r in range(ROWS):
			for c in range(COLS):
				piece = state.grid[r][c]
				if piece == EMPTY:
					continue
				key = (piece_color(piece), piece_type(piece))
				surf = self.piece_surfaces[key]
				cx, cy = self.board_to_screen(r, c)
				rect = surf.get_rect(center=(cx, cy))
				surface.blit(surf, rect)

	def _highlight_cell(
		self, surface: pygame.Surface, cell: Tuple[int, int], color: Tuple[int, int, int], width: int = 3
	) -> None:
		cx, cy = self.board_to_screen(*cell)
		radius = PIECE_SIZE // 2 + 4
		pygame.draw.circle(surface, color, (cx, cy), radius, width)

	def _draw_target_dot(
		self, surface: pygame.Surface, cell: Tuple[int, int], color: Tuple[int, int, int]
	) -> None:
		cx, cy = self.board_to_screen(*cell)
		pygame.draw.circle(surface, color, (cx, cy), 8)

	def _draw_status(self, surface: pygame.Surface, state: Janggi, message: str) -> None:
		bar = pygame.Rect(0, MARGIN_TOP + BOARD_HEIGHT + 12, WINDOW_WIDTH, STATUS_HEIGHT)
		pygame.draw.rect(surface, STATUS_BG, bar)
		pygame.draw.rect(surface, LINE_COLOR, bar, 2)

		turn_label = f"Turn: {PLAYER_NAMES[state.side_to_move].upper()}  Ply: {state.ply}"
		txt1 = self.font_small.render(turn_label, True, TEXT_COLOR)
		surface.blit(txt1, (16, bar.y + 10))

		help_text = "Click a piece, then a destination. [P] = pass turn   [N] = new game   [Esc] = quit"
		txt2 = self.font_small.render(help_text, True, TEXT_COLOR)
		surface.blit(txt2, (16, bar.y + 36))

		if message:
			txt3 = self.font_large.render(message, True, (170, 30, 30))
			rect = txt3.get_rect(midright=(WINDOW_WIDTH - 16, bar.y + 28))
			surface.blit(txt3, rect)


class JanggiApp:
	"""Top-level game controller for the GUI."""

	def __init__(self, human_color: int, simulations: int, checkpoint_path: str) -> None:
		self.human_color = human_color
		self.simulations = simulations
		self.checkpoint_path = checkpoint_path
		self.state = Janggi()
		self.selected: Optional[Tuple[int, int]] = None
		self.legal_for_selected: List[Move] = []
		self.message = ""

		self.agent_thread: Optional[threading.Thread] = None
		self.agent_move: Optional[Move] = None
		self.agent_thinking = False
		self.agent: Optional[NeuralAgent] = None

	# ------------------------------------------------------------------
	# AI worker
	# ------------------------------------------------------------------
	def _ensure_agent(self) -> None:
		if self.agent is None:
			self.agent = NeuralAgent(
				simulations=self.simulations, checkpoint_path=self.checkpoint_path
			)

	def _start_agent_move(self) -> None:
		self._ensure_agent()
		self.agent_thinking = True
		self.message = "AI is thinking..."

		def worker() -> None:
			move, _ = self.agent.select_move(self.state.clone(), temperature=0.0, add_noise=False)
			self.agent_move = move

		self.agent_thread = threading.Thread(target=worker, daemon=True)
		self.agent_thread.start()

	def _consume_agent_move(self) -> None:
		if self.agent_move is None:
			return
		move = self.agent_move
		self.agent_move = None
		self.agent_thinking = False
		self.message = ""
		self.state.apply(move)
		self._check_terminal()

	# ------------------------------------------------------------------
	# Input handling
	# ------------------------------------------------------------------
	def handle_click(self, board_pos: Tuple[int, int]) -> None:
		if self.agent_thinking or self.state.is_terminal():
			return
		if self.state.side_to_move != self.human_color:
			return
		piece = self.state.piece_at(*board_pos)
		if self.selected is None:
			if piece != EMPTY and piece_color(piece) == self.human_color:
				self.selected = board_pos
				self.legal_for_selected = self._legal_from(board_pos)
		else:
			# Clicking another own piece reselects.
			if piece != EMPTY and piece_color(piece) == self.human_color and board_pos != self.selected:
				self.selected = board_pos
				self.legal_for_selected = self._legal_from(board_pos)
				return
			# Otherwise check if the click matches a legal target.
			for move in self.legal_for_selected:
				_, _, tr, tc = move
				if (tr, tc) == board_pos:
					self.state.apply(move)
					self.selected = None
					self.legal_for_selected = []
					self._check_terminal()
					if not self.state.is_terminal() and self.state.side_to_move != self.human_color:
						self._start_agent_move()
					return
			# Click did nothing valid - clear selection.
			self.selected = None
			self.legal_for_selected = []

	def _legal_from(self, src: Tuple[int, int]) -> List[Move]:
		out: List[Move] = []
		for move in self.state.legal_moves():
			if move == PASS_MOVE:
				continue
			fr, fc, _, _ = move
			if (fr, fc) == src:
				out.append(move)
		return out

	def pass_turn(self) -> None:
		if self.agent_thinking or self.state.is_terminal():
			return
		if self.state.side_to_move != self.human_color:
			return
		self.state.apply(PASS_MOVE)
		self.selected = None
		self.legal_for_selected = []
		self._check_terminal()
		if not self.state.is_terminal() and self.state.side_to_move != self.human_color:
			self._start_agent_move()

	def new_game(self) -> None:
		self.state = Janggi()
		self.selected = None
		self.legal_for_selected = []
		self.agent_move = None
		self.agent_thinking = False
		self.message = ""
		# If human plays red, AI (blue) goes first.
		if self.state.side_to_move != self.human_color:
			self._start_agent_move()

	def _check_terminal(self) -> None:
		if not self.state.is_terminal():
			return
		w = self.state.winner
		if w == self.human_color:
			self.message = "You win!"
		elif w == -1:
			self.message = "Draw."
		else:
			self.message = "AI wins."


def run(human_color: int, simulations: int, checkpoint_path: str) -> None:
	pygame.init()
	pygame.display.set_caption("DeepJanggi")
	screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
	clock = pygame.time.Clock()

	piece_surfaces = _load_piece_surfaces()
	font_small = pygame.font.SysFont("Helvetica", 18)
	font_large = pygame.font.SysFont("Helvetica", 26, bold=True)
	renderer = Renderer(
		human_color=human_color,
		piece_surfaces=piece_surfaces,
		font_small=font_small,
		font_large=font_large,
	)

	app = JanggiApp(human_color=human_color, simulations=simulations, checkpoint_path=checkpoint_path)
	app.new_game()

	running = True
	while running:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					running = False
				elif event.key == pygame.K_n:
					app.new_game()
				elif event.key == pygame.K_p:
					app.pass_turn()
			elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
				cell = renderer.screen_to_board(*event.pos)
				if cell is not None:
					app.handle_click(cell)

		if app.agent_move is not None:
			app._consume_agent_move()

		renderer.draw(
			screen,
			app.state,
			app.selected,
			[(m[2], m[3]) for m in app.legal_for_selected],
			app.message,
		)
		pygame.display.flip()
		clock.tick(60)

	pygame.quit()
