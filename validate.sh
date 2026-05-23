#!/usr/bin/env bash
# Evaluate the current AI. Modes (pass via --mode):
#   all (default) -- run evaluation, king-safety, then match
#   match         -- only play games against a baseline (random/minimax/checkpoint)
#   evaluation    -- only check value-head judgment on hand-crafted material imbalances
#   king-safety   -- only check whether the agent saves its king when capture is imminent
set -e
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [ ! -d ".venv" ]; then
	bash setup.sh
fi

source .venv/bin/activate
export PYTHONPATH="$ROOT_DIR"
exec python -m source.validate "$@"
