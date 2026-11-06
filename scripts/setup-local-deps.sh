#!/usr/bin/env sh
set -eu

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
LOCAL_DIR="$PROJECT_ROOT/local-mcp"

# Install runtime dependencies only. The local MCP package itself is
# intentionally not installed; server.py is started directly with Python.
cd "$LOCAL_DIR"
if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
