#!/usr/bin/env bash
# Launch the GUI to play against the AI. Pass --side blue or --side red to choose
# your color; otherwise one is picked at random.
set -e
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [ ! -d ".venv" ]; then
	bash setup.sh
fi

source .venv/bin/activate
export PYTHONPATH="$ROOT_DIR"

has_side=0
for arg in "$@"; do
	case "$arg" in
		--side|--side=*) has_side=1 ;;
	esac
done

if [ "$has_side" -eq 0 ]; then
	if (( RANDOM % 2 )); then
		side="blue"
	else
		side="red"
	fi
	echo "Random side: $side"
	set -- --side "$side" "$@"
fi

exec python -m source.play "$@"
