#!/usr/bin/env sh
set -eu

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
LOCAL_DIR="$PROJECT_ROOT/local-mcp"
PURGE_DATA=0

usage() {
  cat <<'EOF'
Usage: uninstall-local.sh [OPTIONS]

By default this removes generated Python build environments only:
  local-mcp/.venv
  local-mcp/build
  local-mcp/dist
  local-mcp/*.egg-info
  __pycache__ directories

Local SQLite registries under local-mcp/runtime and local.yaml are preserved.

Options:
  --purge-data      Delete the local Remote CUDA data directory (default local-mcp/runtime).
                    This removes the local server, workspace, and job registries.
  -h, --help        Show this help.

Environment overrides used only with --purge-data:
  REMOTE_CUDA_HOME
EOF
}

remove_path() {
  path="$1"
  if [ -e "$path" ] || [ -L "$path" ]; then
    rm -rf -- "$path"
    printf 'removed: %s\n' "$path"
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --purge-data) PURGE_DATA=1 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

remove_path "$LOCAL_DIR/.venv"
remove_path "$LOCAL_DIR/build"
remove_path "$LOCAL_DIR/dist"

find "$LOCAL_DIR" -type d -name '__pycache__' -prune -exec rm -rf -- {} +
find "$LOCAL_DIR" -type d -name '*.egg-info' -prune -exec rm -rf -- {} +
find "$LOCAL_DIR" -type f -name '*.pyc' -delete

if [ "$PURGE_DATA" -eq 1 ]; then
  data_home="${REMOTE_CUDA_HOME:-$LOCAL_DIR/runtime}"
  case "$data_home" in
    ""|/) echo "error: invalid data directory" >&2; exit 1 ;;
  esac
  remove_path "$data_home"
fi

echo "Local Remote CUDA MCP environment uninstall complete."
if [ "$PURGE_DATA" -eq 0 ]; then
  echo "Local registry data and configuration were preserved."
fi
