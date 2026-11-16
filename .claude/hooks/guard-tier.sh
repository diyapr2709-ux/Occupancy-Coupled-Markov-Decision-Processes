#!/usr/bin/env bash
# PreToolUse on Bash. Enforces seed tiers, the daily run budget and the one-shot test tier.
# The seed-count guard runs first, so a launch it blocks is never written to the ledger.
set -uo pipefail
INPUT=$(cat)
printf '%s' "$INPUT" | bash "$(dirname "$0")/guard-seeds.sh" || exit $?
. "$(dirname "$0")/launch.sh" <<<"$INPUT"
launches_experiment || exit 0

TIER=$(tier_of_command)
if [ -z "$TIER" ]; then
  echo "Blocked: experiment commands must pass --tier dev, val or test." >&2
  exit 2
fi

LEDGER=runs/ledger.txt
MAX=$(grep -E 'max_training_runs:' budget.yaml 2>/dev/null | grep -oE '[0-9]+' | head -n 1)
USED=$(grep -c "^$(date +%F)" "$LEDGER" 2>/dev/null)
if [ "${USED:-0}" -ge "${MAX:-0}" ]; then
  echo "Blocked: today's budget of ${MAX:-0} runs is spent. Stop and report." >&2
  exit 2
fi

SLUG=""
if [ "$TIER" = "test" ]; then
  PREREG=$(printf '%s' "$CMD" | grep -oE -- '--prereg[= ]+[^ ]+' | sed -E 's/--prereg[= ]+//' | tail -n 1)
  if [ -z "$PREREG" ] || ! git ls-files --error-unmatch "$PREREG" >/dev/null 2>&1 || ! git diff --quiet HEAD -- "$PREREG"; then
    echo "Blocked: test seeds need --prereg pointing at a committed, unmodified preregistration." >&2
    exit 2
  fi
  SLUG=$(basename "$PREREG" .yaml)
  if [ ! -f "approvals/$SLUG.ok" ]; then
    echo "Blocked: approvals/$SLUG.ok does not exist. That file is created by hand, never by an agent." >&2
    exit 2
  fi
  if grep -q "|test|$SLUG\$" "$LEDGER" 2>/dev/null; then
    echo "Blocked: test seeds for $SLUG have already been used once." >&2
    exit 2
  fi
fi

mkdir -p runs
echo "$(date +%FT%T)|$TIER|$SLUG" >> "$LEDGER"
exit 0
