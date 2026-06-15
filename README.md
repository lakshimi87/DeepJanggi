# DeepJanggi

A self-play deep reinforcement learning agent for **Janggi** (Korean chess), built
with PyTorch and PUCT-style Monte Carlo Tree Search and a Pygame-CE GUI for human
play. The architecture follows the AlphaZero recipe: a single residual policy/value
network is trained from games it generates against itself, with MCTS guiding both
training-time exploration and inference-time move selection.

## Features

- Full Janggi rule implementation (palace, diagonals, cannon screen-jump,
  horse/elephant blocking, soldier promotion to palace diagonals, pass action,
  king-capture win condition, material-score draw resolution).
- AlphaZero-style training loop with replay buffer, Dirichlet root noise,
  temperature schedule, and checkpoint resume.
- Pygame-CE GUI that loads the SVG piece artwork in `res/`. Human can play as
  either Blue or Red and the board orientation flips so the human side is always
  on the bottom.
- Validation harness that scores the current AI against a random baseline or any
  older checkpoint.

## Requirements

- Python 3.10+ (developed on 3.12)
- macOS / Linux (the scripts are Bash)
- The first run of any script auto-creates `.venv/` and installs deps via
  `setup.sh`.

## Quick start

```bash
./setup.sh                    # one-time: create venv and install dependencies
./train.sh --iterations 100   # self-play training (writes checkpoints/latest.pt)
./play.sh --side blue         # play vs the AI as Blue (--side red for Red)
./validate.sh                 # ground-truth suite + 10-game match vs minimax
```

The first time `train.sh` / `play.sh` / `validate.sh` is invoked it bootstraps the
virtual environment automatically.

## Project layout

```
DeepJanggi/
├── res/                       # SVG piece artwork (provided)
├── checkpoints/               # saved models; latest.pt is the rolling checkpoint
├── source/                    # all Python sources (tab-indented, English comments)
│   ├── config.py              # piece codes, board geometry, hyperparameters
│   ├── board.py               # Janggi state + move generation + rules
│   ├── encoder.py             # state -> network input tensor
│   ├── network.py             # ResNet policy/value network (PyTorch)
│   ├── mcts.py                # PUCT MCTS with Dirichlet noise
│   ├── replay.py              # experience replay buffer
│   ├── selfplay.py            # one self-play game generator
│   ├── trainer.py             # outer training loop
│   ├── agent.py               # NeuralAgent / RandomAgent wrappers
│   ├── ui.py                  # Pygame-CE renderer + controller
│   ├── train.py               # entry point: ./train.sh
│   ├── play.py                # entry point: ./play.sh
│   └── validate.py            # entry point: ./validate.sh
├── setup.sh                   # create .venv and install requirements
├── train.sh / play.sh / validate.sh
└── requirements.txt
```

## How it works

### Game state (`source/board.py`)
The board is a 10×9 grid of integer piece codes (color × type). `Janggi.legal_moves`
emits all rule-legal `(from_row, from_col, to_row, to_col)` tuples plus the pass
action. King capture ends the game; two consecutive passes or hitting the
`MAX_PLY` cap finalize via material score (with a 1.5 Han bonus for Blue).

### Action space
`TOTAL_ACTIONS = 90 * 90 + 1 = 8101`. Action index `from_idx * 90 + to_idx`,
with the last index reserved for pass. Illegal actions are masked before the
softmax in MCTS.

### Network (`source/network.py`)
Input planes: 14 piece planes + 1 side-to-move plane + 1 normalized ply plane =
16 channels of shape (10, 9). The network is an AlphaZero-style ResNet (default
16 residual blocks × 192 filters, ≈11.4M parameters — matching DeepChess's
tower scale) with two heads:
- **Policy** — *fully convolutional*: a 3×3 conv + a 1×1 conv emitting 90
  destination-square planes over the board. Flattening as
  `from_idx * 90 + to_idx` lines each logit's spatial position up with its
  move's from-square (matching `board.move_to_action`), so policy weights are
  shared across the board instead of living in one giant `Linear` matrix. The
  location-less PASS logit comes from a tiny head off the globally pooled tower
  features. (The previous `flatten → Linear(32·90, 8101)` head put 23.3M
  parameters — 95% of the whole net — into a single weight matrix with no
  spatial weight sharing, which both bloated the checkpoint and kept the policy
  from learning.)
- **Value** — 1×1 conv (32 channels) → MLP → tanh-bounded scalar

### MCTS (`source/mcts.py`)
PUCT selection with `c=1.5`, leaf evaluations from the network, and Dirichlet
noise injected at the root during self-play. Visit counts at the root form the
policy training target; the game outcome (±1, 0) becomes the value target.

### Training loop (`source/trainer.py`)
Each iteration:
1. Generate `GAMES_PER_ITERATION` self-play games and push samples to the replay
   buffer.
2. Run `TRAIN_STEPS_PER_ITERATION` SGD steps minimising
   `cross_entropy(policy_targets, policy_logits) + MSE(value, z)`.
3. Save `checkpoints/latest.pt` (model + optimizer + iteration counters).

Use `--resume` on `train.sh` to pick up from the latest checkpoint.

## Command-line reference

### `train.sh`
```
--iterations N        Number of self-play -> training iterations.
--simulations N       MCTS simulations per move during self-play.
--games-per-iter N    Games generated each iteration before training.
--train-steps N       SGD steps per iteration.
--resume              Load checkpoints/latest.pt before training.
```

### `play.sh`
```
--side {blue,red}     Which color the human plays. Blue moves first.
--simulations N       MCTS simulations per AI move.
--checkpoint PATH     Override the .pt file the AI loads.
```
Keyboard:
- `P` — pass turn
- `N` — new game
- `Esc` — quit

### `validate.sh`

Ground-truth validation modelled on DeepChess's `validate_gt`: a curated move
suite (King Capture / Win Material / King Safety) and a value-head evaluation are
scored for the neural engine *and* a classical minimax baseline side by side,
followed by a head-to-head match. Exits non-zero if the neural engine passes
under 60% of the suite (useful in CI).
```
--games N             Head-to-head games to play, 0 to skip (sides alternate).
--simulations N       MCTS simulations per test/match move.
--minimax-depth N     Search depth for the classical minimax baseline.
--eval-tolerance F    Value-head magnitude threshold for a non-zero verdict.
--opponent {random,minimax,checkpoint}   Match opponent (default minimax).
--opponent-path PATH  Required if --opponent checkpoint.
--max-ply N           Per-game ply cap during the match.
--history             Chart suite progress across numbered checkpoints.
--checkpoint-dir DIR  Where --history looks for numbered checkpoints.
--checkpoint PATH     The checkpoint under test.
```

## Notes & limitations

- A freshly trained network needs many iterations before it plays meaningfully.
  The defaults (`MCTS_SIMULATIONS_TRAIN=80`, `GAMES_PER_ITERATION=8`) are tuned
  for tractable runs on a single laptop GPU/MPS device, not for state-of-the-art
  strength.
- Bikjang (face-to-face general draw rule) is not implemented; draws are decided
  by material score on ply cap or by mutual passing.
- The opening setup is the standard "inner-elephant" arrangement on both sides
  (no masang / sangma setup choice).
