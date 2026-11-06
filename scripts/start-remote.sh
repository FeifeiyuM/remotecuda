#!/usr/bin/env sh
set -eu
DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
AGENT_DIR="$DIR/remote-agent"
if [ -x "$AGENT_DIR/.venv/bin/python" ]; then
  PYTHON="$AGENT_DIR/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi
export REMOTE_CUDA_AGENT_INSTALL_DIR="${REMOTE_CUDA_AGENT_INSTALL_DIR:-$AGENT_DIR}"
cd "$AGENT_DIR"
exec "$PYTHON" main.py
