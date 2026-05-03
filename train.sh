#!/usr/bin/env bash
# Run self-play training. Resumes from checkpoints/latest.pt if available.
set -e
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [ ! -d ".venv" ]; then
	bash setup.sh
fi

source .venv/bin/activate
export PYTHONPATH="$ROOT_DIR"
exec python -m source.train "$@"
