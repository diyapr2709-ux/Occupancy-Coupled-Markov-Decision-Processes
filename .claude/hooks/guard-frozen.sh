#!/usr/bin/env bash
# PreToolUse on Edit|Write. A preregistration is frozen once committed: new files may be written, tracked ones may not change.
set -uo pipefail
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
FILE=$(jq -r '.tool_input.file_path // ""')
REL=${FILE#"$PWD"/}

case "$REL" in
  preregistrations/*) ;;
  *) exit 0 ;;
esac

if git ls-files --error-unmatch "$REL" >/dev/null 2>&1; then
  echo "Blocked: $REL is committed and frozen. A changed rule goes in a new dated preregistration; the old one stays." >&2
  exit 2
fi

exit 0
