#!/usr/bin/env bash
# Evaluate the current AI by playing matches against a baseline (random or older checkpoint).
set -e
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [ ! -d ".venv" ]; then
	bash setup.sh
fi

source .venv/bin/activate
export PYTHONPATH="$ROOT_DIR"
exec python -m source.validate "$@"
