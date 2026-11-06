#!/usr/bin/env sh
set -eu
DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
LOCAL_DIR="$DIR/local-mcp"
if [ -x "$LOCAL_DIR/.venv/bin/python" ]; then
  PYTHON="$LOCAL_DIR/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi
export PYTHONPATH="$LOCAL_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$LOCAL_DIR"
exec "$PYTHON" server.py
