#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

echo "This removes generated Python environments for both components."
echo "Registry data, workspace data, and configuration are preserved by default."
shift_count=0
for argument in "$@"; do
  case "$argument" in
    --help|-h)
      echo "Usage: uninstall-all.sh [--purge-data remote|local|all]"
      echo "The optional purge form is intentionally explicit:"
      echo "  --purge-data remote  Uninstall remote Agent and delete its data"
      echo "  --purge-data local   Uninstall local MCP and delete its data"
      echo "  --purge-data all     Delete both local and remote data"
      exit 0
      ;;
    --purge-data)
      if [ "$#" -ne 2 ]; then
        echo "error: --purge-data requires remote, local, or all" >&2
        exit 2
      fi
      shift_count=1
      ;;
    remote|local|all)
      if [ "$shift_count" -ne 1 ]; then
        echo "error: unexpected argument: $argument" >&2
        exit 2
      fi
      ;;
    *)
      echo "error: unknown argument: $argument" >&2
      exit 2
      ;;
  esac
done

case " ${*} " in
  *" remote "*)
    "$SCRIPT_DIR/uninstall-remote.sh" --purge-data
    "$SCRIPT_DIR/uninstall-local.sh"
    ;;
  *" local "*)
    "$SCRIPT_DIR/uninstall-remote.sh"
    "$SCRIPT_DIR/uninstall-local.sh" --purge-data
    ;;
  *" all "*)
    "$SCRIPT_DIR/uninstall-remote.sh" --purge-data
    "$SCRIPT_DIR/uninstall-local.sh" --purge-data
    ;;
  *)
    "$SCRIPT_DIR/uninstall-remote.sh"
    "$SCRIPT_DIR/uninstall-local.sh"
    ;;
esac
