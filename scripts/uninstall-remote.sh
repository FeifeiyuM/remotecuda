#!/usr/bin/env sh
set -eu

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
AGENT_DIR="$PROJECT_ROOT/remote-agent"
PURGE_DATA=0
PURGE_CONFIG=0
STOP_SYSTEM_SERVICE=0

usage() {
  cat <<'EOF'
Usage: uninstall-remote.sh [OPTIONS]

By default this removes generated Python build environments only:
  remote-agent/.venv
  remote-agent/build
  remote-agent/dist
  remote-agent/*.egg-info
  __pycache__ directories

Workspace data, SQLite job history, tokens, and configuration are preserved.

Options:
  --purge-data          Delete the configured workspace root and Agent database.
                        This can delete source copies, artifacts, logs, and job history.
  --purge-config        Delete /etc/remote-cuda-agent and remote-agent/remote.yaml.
                        This can delete the configured token file/environment file.
  --system-service      Stop and disable the system remote-cuda-agent.service.
                        Run the whole script with root privileges when using this option.
  -h, --help            Show this help.

Environment overrides used only with --purge-data:
  REMOTE_CUDA_WORKSPACE_ROOT
  REMOTE_CUDA_DATABASE
EOF
}

need_root() {
  if [ "$(id -u)" -ne 0 ]; then
    echo "error: --$1 must be run with root privileges" >&2
    exit 1
  fi
}

remove_path() {
  path="$1"
  if [ -e "$path" ] || [ -L "$path" ]; then
    rm -rf -- "$path"
    printf 'removed: %s\n' "$path"
  fi
}

config_value() {
  key="$1"
  fallback="$2"
  file="$AGENT_DIR/remote.yaml"
  if [ -f "$file" ]; then
    value="$(sed -n "s/^[[:space:]]*${key}:[[:space:]]*//p" "$file" | tail -n 1 | tr -d '\"'"'"'')"
    if [ -n "$value" ]; then
      printf '%s\n' "$value"
      return 0
    fi
  fi
  printf '%s\n' "$fallback"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --purge-data) PURGE_DATA=1 ;;
    --purge-config) PURGE_CONFIG=1 ;;
    --system-service) STOP_SYSTEM_SERVICE=1 ;;
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

if [ "$STOP_SYSTEM_SERVICE" -eq 1 ]; then
  need_root system-service
  if command -v systemctl >/dev/null 2>&1; then
    systemctl stop remote-cuda-agent.service 2>/dev/null || true
    systemctl disable remote-cuda-agent.service 2>/dev/null || true
    echo "stopped and disabled remote-cuda-agent.service (if installed)"
  else
    echo "warning: systemctl is unavailable; service was not stopped" >&2
  fi
fi

remove_path "$AGENT_DIR/.venv"
remove_path "$AGENT_DIR/build"
remove_path "$AGENT_DIR/dist"

find "$AGENT_DIR" -type d -name '__pycache__' -prune -exec rm -rf -- {} +
find "$AGENT_DIR" -type d -name '*.egg-info' -prune -exec rm -rf -- {} +
find "$AGENT_DIR" -type f -name '*.pyc' -delete

if [ "$PURGE_DATA" -eq 1 ]; then
  workspace_root="${REMOTE_CUDA_WORKSPACE_ROOT:-$(config_value workspace_root "$AGENT_DIR/runtime/workspaces")}"
  database_path="${REMOTE_CUDA_DATABASE:-$(config_value database_path "$AGENT_DIR/runtime/agent.db")}"

  case "$workspace_root" in
    ""|/) echo "error: invalid workspace root" >&2; exit 1 ;;
  esac
  case "$database_path" in
    ""|/) echo "error: invalid database path" >&2; exit 1 ;;
  esac

  remove_path "$workspace_root"
  remove_path "$database_path"
  remove_path "$database_path-shm"
  remove_path "$database_path-wal"
fi

if [ "$PURGE_CONFIG" -eq 1 ]; then
  remove_path /etc/remote-cuda-agent
  remove_path "$AGENT_DIR/remote.yaml"
fi

echo "Remote CUDA Agent environment uninstall complete."
if [ "$PURGE_DATA" -eq 0 ]; then
  echo "Workspace data and SQLite history were preserved."
fi
