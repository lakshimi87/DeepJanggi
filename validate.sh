#!/usr/bin/env bash
# Ground-truth validation for the current AI (mirrors DeepChess's validate_gt).
# Runs a curated move suite (King Capture / Win Material / King Safety) plus a
# value-head evaluation, side-by-side neural-vs-classical, then a head-to-head
# match. Exit code is 0 when the neural engine passes >=60% of the suite.
#   ./validate.sh                   # latest checkpoint + 10-game match
#   ./validate.sh --history         # chart progress across numbered checkpoints
#   ./validate.sh --simulations 200 # more MCTS sims (slower but fairer)
#   ./validate.sh --games 0         # skip the head-to-head match
set -e
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [ ! -d ".venv" ]; then
	bash setup.sh
fi

source .venv/bin/activate
export PYTHONPATH="$ROOT_DIR"
exec python -m source.validate "$@"
