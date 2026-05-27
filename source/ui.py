"""Pygame-CE GUI for playing Janggi against the AI.

The board is always rendered with the human player on the bottom side, so the
underlying logical board (row 0 = top = red) is mirrored when the human plays red.
"""

from __future__ import annotations

import os
import random
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pygame

from .agent import NeuralAgent
from .board import INITIAL_SETUP, Janggi, Move, PASS_MOVE, swap_flank
from .config import (
	BLUE,
	COLS,
	DEFAULT_DIFFICULTY,
	DIFFICULTY_LEVELS,
	DIFFICULTY_SIMULATIONS,
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
MARGIN_BOTTOM = 56
BOARD_WIDTH = (COLS - 1) * CELL_SIZE
BOARD_HEIGHT = (ROWS - 1) * CELL_SIZE
SIDE_PANEL_GAP = 24
SIDE_PANEL_WIDTH = 280
SIDE_PANEL_PAD = 16
WINDOW_WIDTH = MARGIN_LEFT + BOARD_WIDTH + SIDE_PANEL_GAP + SIDE_PANEL_WIDTH + SIDE_PANEL_GAP
WINDOW_HEIGHT = MARGIN_TOP + BOARD_HEIGHT + MARGIN_BOTTOM
PIECE_SIZE = int(CELL_SIZE * 0.92)

SIDE_PANEL_X = MARGIN_LEFT + BOARD_WIDTH + SIDE_PANEL_GAP
SIDE_PANEL_Y = MARGIN_TOP
SIDE_PANEL_HEIGHT = BOARD_HEIGHT

# Color palette.
BACKGROUND = (236, 200, 132)
LINE_COLOR = (40, 28, 18)
HIGHLIGHT_SELECT = (60, 120, 220)
HIGHLIGHT_MOVE = (40, 180, 80)
HIGHLIGHT_CAPTURE = (220, 60, 60)
HIGHLIGHT_SETUP = (90, 130, 200)
HIGHLIGHT_LAST_MOVE = (210, 150, 40)
TEXT_COLOR = (20, 20, 20)
SUBTLE_TEXT = (90, 60, 30)
STATUS_BG = (220, 178, 110)
BUTTON_BG = (200, 160, 90)
BUTTON_BG_HOVER = (224, 184, 116)
BUTTON_BG_DISABLED = (190, 180, 160)
BUTTON_BORDER = (40, 28, 18)
BUTTON_TEXT = (20, 20, 20)
BUTTON_TEXT_DISABLED = (110, 100, 90)
ERROR_COLOR = (170, 30, 30)
THINKING_COLOR = (50, 100, 60)


def _back_row(color: int) -> int:
	"""Return the back-rank row index for a color (red at top, blue at bottom)."""
	return 9 if color == BLUE else 0


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
class Button:
	rect: pygame.Rect
	label: str
	enabled: bool = True

	def hit(self, x: int, y: int) -> bool:
		return self.enabled and self.rect.collidepoint(x, y)


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
		last_move: Optional[Move],
		app: "JanggiApp",
		mouse_pos: Tuple[int, int],
	) -> None:
		surface.fill(BACKGROUND)
		self._draw_board(surface)
		self._draw_palaces(surface)
		if app.setup_phase:
			self._highlight_setup_pieces(surface, state)
		if last_move is not None and last_move != PASS_MOVE:
			fr, fc, tr, tc = last_move
			self._draw_last_move_square(surface, (fr, fc))
			self._draw_last_move_square(surface, (tr, tc))
		if selected is not None:
			self._highlight_cell(surface, selected, HIGHLIGHT_SELECT, width=4)
		for tr, tc in legal_targets:
			occ = state.grid[tr][tc]
			color = HIGHLIGHT_CAPTURE if occ != EMPTY else HIGHLIGHT_MOVE
			self._draw_target_dot(surface, (tr, tc), color)
		self._draw_pieces(surface, state)
		self._draw_side_panel(surface, state, app, mouse_pos)

	def _draw_board(self, surface: pygame.Surface) -> None:
		for c in range(COLS):
			x, y0 = self.board_to_screen(0, c)
			_, y1 = self.board_to_screen(ROWS - 1, c)
			pygame.draw.line(surface, LINE_COLOR, (x, y0), (x, y1), 2)
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

	def _draw_last_move_square(self, surface: pygame.Surface, cell: Tuple[int, int]) -> None:
		cx, cy = self.board_to_screen(*cell)
		half = CELL_SIZE // 2 - 2
		rect = pygame.Rect(cx - half, cy - half, half * 2, half * 2)
		pygame.draw.rect(surface, HIGHLIGHT_LAST_MOVE, rect, 3)

	def _highlight_setup_pieces(self, surface: pygame.Surface, state: Janggi) -> None:
		"""During setup, highlight the human's horse/elephant cells eligible for a flank swap."""
		r = _back_row(self.human_color)
		for c in (1, 2, 6, 7):
			piece = state.grid[r][c]
			if piece == EMPTY:
				continue
			if piece_type(piece) in (HORSE, ELEPHANT):
				self._highlight_cell(surface, (r, c), HIGHLIGHT_SETUP, width=2)

	def _draw_side_panel(
		self,
		surface: pygame.Surface,
		state: Janggi,
		app: "JanggiApp",
		mouse_pos: Tuple[int, int],
	) -> None:
		panel = pygame.Rect(SIDE_PANEL_X, SIDE_PANEL_Y, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT)
		pygame.draw.rect(surface, STATUS_BG, panel)
		pygame.draw.rect(surface, LINE_COLOR, panel, 2)

		x = panel.x + SIDE_PANEL_PAD
		y = panel.y + SIDE_PANEL_PAD
		inner_w = panel.width - SIDE_PANEL_PAD * 2

		# Title.
		title = self.font_large.render("DeepJanggi", True, TEXT_COLOR)
		surface.blit(title, (x, y))
		y += title.get_height() + 8

		# You info.
		you_label = f"You play: {PLAYER_NAMES[app.human_color].upper()}"
		surface.blit(self.font_small.render(you_label, True, TEXT_COLOR), (x, y))
		y += 24

		# Turn / ply.
		turn_label = f"Turn: {PLAYER_NAMES[state.side_to_move].upper()}"
		surface.blit(self.font_small.render(turn_label, True, TEXT_COLOR), (x, y))
		y += 22
		ply_label = f"Ply: {state.ply}"
		surface.blit(self.font_small.render(ply_label, True, TEXT_COLOR), (x, y))
		y += 28

		# Separator.
		pygame.draw.line(surface, LINE_COLOR, (x, y), (x + inner_w, y), 1)
		y += 10

		# Status / hint area.
		if app.setup_phase:
			heading = self.font_small.render("SETUP", True, ERROR_COLOR)
			surface.blit(heading, (x, y))
			y += 22
			hint_lines = [
				"Click a HORSE or",
				"ELEPHANT on YOUR back",
				"rank to swap it with",
				"its flank partner.",
				"",
				"The AI arranges its",
				"own flanks at random.",
				"",
				"Press Start (or SPACE)",
				"to begin the game.",
			]
			for line in hint_lines:
				surface.blit(self.font_small.render(line, True, TEXT_COLOR), (x, y))
				y += 20
		else:
			msg = app.message
			if msg:
				if "thinking" in msg.lower():
					color = THINKING_COLOR
				elif state.is_terminal():
					color = ERROR_COLOR
				else:
					color = TEXT_COLOR
				for line in self._wrap_text(msg, self.font_small, inner_w):
					surface.blit(self.font_small.render(line, True, color), (x, y))
					y += 22

		# Buttons (drawn near the bottom of the panel).
		for btn in app.visible_buttons():
			self._draw_button(surface, btn, mouse_pos)

		# Help text at the bottom of the panel.
		help_lines = [
			"[N] New game",
			"[P] Pass turn",
			"[D] Difficulty (setup)",
			"[Esc] Quit",
		]
		hy = panel.y + panel.height - SIDE_PANEL_PAD - len(help_lines) * 20
		for line in help_lines:
			surface.blit(self.font_small.render(line, True, SUBTLE_TEXT), (x, hy))
			hy += 20

	def _draw_button(
		self, surface: pygame.Surface, btn: Button, mouse_pos: Tuple[int, int]
	) -> None:
		if not btn.enabled:
			bg = BUTTON_BG_DISABLED
			fg = BUTTON_TEXT_DISABLED
		elif btn.rect.collidepoint(mouse_pos):
			bg = BUTTON_BG_HOVER
			fg = BUTTON_TEXT
		else:
			bg = BUTTON_BG
			fg = BUTTON_TEXT
		pygame.draw.rect(surface, bg, btn.rect, border_radius=6)
		pygame.draw.rect(surface, BUTTON_BORDER, btn.rect, 2, border_radius=6)
		txt = self.font_small.render(btn.label, True, fg)
		txt_rect = txt.get_rect(center=btn.rect.center)
		surface.blit(txt, txt_rect)

	def _wrap_text(self, text: str, font: pygame.font.Font, max_width: int) -> List[str]:
		words = text.split()
		if not words:
			return [text]
		lines: List[str] = []
		cur = ""
		for w in words:
			candidate = (cur + " " + w).strip()
			if font.size(candidate)[0] <= max_width:
				cur = candidate
			else:
				if cur:
					lines.append(cur)
				cur = w
		if cur:
			lines.append(cur)
		return lines


class JanggiApp:
	"""Top-level game controller for the GUI."""

	def __init__(
		self,
		human_color: int,
		simulations: int,
		checkpoint_path: str,
		difficulty: str = DEFAULT_DIFFICULTY,
	) -> None:
		self.human_color = human_color
		self.difficulty = difficulty
		self.simulations = simulations
		self.checkpoint_path = checkpoint_path
		self.state = Janggi()
		self.selected: Optional[Tuple[int, int]] = None
		self.legal_for_selected: List[Move] = []
		self.last_move: Optional[Move] = None
		self.message = ""
		self.setup_phase = True

		self.agent_thread: Optional[threading.Thread] = None
		self.agent_move: Optional[Move] = None
		self.agent_thinking = False
		self.agent: Optional[NeuralAgent] = None

		# Side-panel buttons.
		bx = SIDE_PANEL_X + SIDE_PANEL_PAD
		bw = SIDE_PANEL_WIDTH - SIDE_PANEL_PAD * 2
		button_block_top = SIDE_PANEL_Y + SIDE_PANEL_HEIGHT - SIDE_PANEL_PAD - 80 - 96 - 48
		self.difficulty_button = Button(
			rect=pygame.Rect(bx, button_block_top, bw, 36),
			label=self._difficulty_label(),
		)
		self.start_button = Button(
			rect=pygame.Rect(bx, button_block_top + 44, bw, 44),
			label="Start Game",
		)
		self.reset_button = Button(
			rect=pygame.Rect(bx, button_block_top + 44 + 52, bw, 36),
			label="Reset Setup",
		)

	def _difficulty_label(self) -> str:
		return f"Difficulty: {self.difficulty.upper()}"

	def cycle_difficulty(self) -> None:
		"""Advance to the next difficulty preset and update the AI's simulation budget."""
		if not self.setup_phase:
			return
		idx = DIFFICULTY_LEVELS.index(self.difficulty)
		self.difficulty = DIFFICULTY_LEVELS[(idx + 1) % len(DIFFICULTY_LEVELS)]
		self.simulations = DIFFICULTY_SIMULATIONS[self.difficulty]
		self.difficulty_button.label = self._difficulty_label()
		if self.agent is not None:
			self.agent.simulations = self.simulations
			self.agent.mcts.simulations = self.simulations

	def visible_buttons(self) -> List[Button]:
		if self.setup_phase:
			return [self.difficulty_button, self.start_button, self.reset_button]
		return []

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
		self.last_move = move
		self._check_terminal()

	# ------------------------------------------------------------------
	# Input handling
	# ------------------------------------------------------------------
	def handle_panel_click(self, pos: Tuple[int, int]) -> bool:
		"""Process clicks that fall inside the side panel. Returns True if handled."""
		if self.setup_phase:
			if self.difficulty_button.hit(*pos):
				self.cycle_difficulty()
				return True
			if self.start_button.hit(*pos):
				self.start_game()
				return True
			if self.reset_button.hit(*pos):
				self.reset_setup()
				return True
		# Swallow any other clicks inside the panel so they don't fall through.
		panel = pygame.Rect(SIDE_PANEL_X, SIDE_PANEL_Y, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT)
		return panel.collidepoint(*pos)

	def handle_click(self, board_pos: Tuple[int, int]) -> None:
		if self.setup_phase:
			self._try_swap_setup(board_pos)
			return
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
					self.last_move = move
					self.selected = None
					self.legal_for_selected = []
					self._check_terminal()
					if not self.state.is_terminal() and self.state.side_to_move != self.human_color:
						self._start_agent_move()
					return
			# Click did nothing valid - clear selection.
			self.selected = None
			self.legal_for_selected = []

	def _try_swap_setup(self, board_pos: Tuple[int, int]) -> None:
		"""Swap a horse with its flank-partner elephant on the human's back rank only."""
		r, c = board_pos
		if r != _back_row(self.human_color):
			return
		piece = self.state.grid[r][c]
		if piece == EMPTY or piece_type(piece) not in (HORSE, ELEPHANT):
			return
		if c in (1, 2):
			partner_c = 3 - c  # 1 <-> 2
		elif c in (6, 7):
			partner_c = 13 - c  # 6 <-> 7
		else:
			return
		partner = self.state.grid[r][partner_c]
		if partner == EMPTY or piece_type(partner) not in (HORSE, ELEPHANT):
			return
		self.state.grid[r][c], self.state.grid[r][partner_c] = (
			self.state.grid[r][partner_c],
			self.state.grid[r][c],
		)

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
		if self.setup_phase:
			return
		if self.agent_thinking or self.state.is_terminal():
			return
		if self.state.side_to_move != self.human_color:
			return
		self.state.apply(PASS_MOVE)
		self.last_move = PASS_MOVE
		self.selected = None
		self.legal_for_selected = []
		self._check_terminal()
		if not self.state.is_terminal() and self.state.side_to_move != self.human_color:
			self._start_agent_move()

	def reset_setup(self) -> None:
		"""Restore the human's back-row horse/elephant layout to the default 상마마상 form."""
		if not self.setup_phase:
			return
		r = _back_row(self.human_color)
		for c in range(COLS):
			self.state.grid[r][c] = INITIAL_SETUP[r][c]

	def start_game(self) -> None:
		if not self.setup_phase:
			return
		self.setup_phase = False
		self.message = ""
		if not self.state.is_terminal() and self.state.side_to_move != self.human_color:
			self._start_agent_move()

	def new_game(self) -> None:
		self.state = Janggi()
		# The AI rearranges its own flanks at random; the human uses the swap UI.
		ai_back_row = _back_row(RED if self.human_color == BLUE else BLUE)
		for flank in ("left", "right"):
			if random.random() < 0.5:
				swap_flank(self.state.grid, ai_back_row, flank)
		self.selected = None
		self.legal_for_selected = []
		self.last_move = None
		self.agent_move = None
		self.agent_thinking = False
		self.message = ""
		self.setup_phase = True

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


def run(
	human_color: int,
	simulations: int,
	checkpoint_path: str,
	difficulty: str = DEFAULT_DIFFICULTY,
) -> None:
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

	app = JanggiApp(
		human_color=human_color,
		simulations=simulations,
		checkpoint_path=checkpoint_path,
		difficulty=difficulty,
	)
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
				elif event.key == pygame.K_d:
					app.cycle_difficulty()
				elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
					if app.setup_phase:
						app.start_game()
			elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
				if not app.handle_panel_click(event.pos):
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
			app.last_move,
			app,
			pygame.mouse.get_pos(),
		)
		pygame.display.flip()
		clock.tick(60)

	pygame.quit()
