#!/usr/bin/env bash
# PostToolUse on Edit|Write. Warns when a file that reports results changes while no run is logged.
set -uo pipefail
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
FILE=$(jq -r '.tool_input.file_path // ""')

case "$FILE" in
  *results*|*tables*|*.tex|*README.md) ;;
  *) exit 0 ;;
esac

if [ -z "$(ls -A runs 2>/dev/null)" ]; then
  echo "Note: a results file changed but runs/ is empty. Every number needs a logged run." >&2
  exit 1
fi

exit 0
