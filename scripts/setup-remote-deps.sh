#!/usr/bin/env sh
set -eu

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
AGENT_DIR="$PROJECT_ROOT/remote-agent"

# Install runtime dependencies only. The Remote CUDA Agent package itself is
# intentionally not installed; it is started directly with Python.
cd "$AGENT_DIR"
if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
mkdir -p runtime
