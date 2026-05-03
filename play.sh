#!/usr/bin/env bash
# Launch the GUI to play against the AI. Pass --side blue or --side red to choose your color.
set -e
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [ ! -d ".venv" ]; then
	bash setup.sh
fi

source .venv/bin/activate
export PYTHONPATH="$ROOT_DIR"
exec python -m source.play "$@"
