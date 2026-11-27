#!/usr/bin/env bash
# Blocks a confirmatory run with too few seeds. Called by guard-tier.sh with the hook input on stdin.
set -uo pipefail
. "$(dirname "$0")/launch.sh"
launches_experiment || exit 0
[ "$(tier_of_command)" = "test" ] || exit 0

MIN=10
if printf '%s' "$CMD" | grep -qE 'experiments/sweep_phase_diagram\.py'; then
  # a planner-only sweep trains nothing; its confirmatory count is the evaluation episodes it is paired on
  SEEDS=$(printf '%s' "$CMD" | sed -nE 's/.*--episodes[= ]+([0-9]+).*/\1/p')
  if [ -z "$SEEDS" ]; then
    echo "Blocked: a test-tier sweep must pass --episodes explicitly. The preregistration states the count." >&2
    exit 2
  fi
else
  SEEDS=$(printf '%s' "$CMD" | sed -nE 's/.*--seeds[= ]+(([0-9]+[[:space:]]*)+).*/\1/p' | wc -w | tr -d ' ')
fi

if [ "$SEEDS" -eq 0 ]; then
  echo "Blocked: a test-tier run must pass --seeds explicitly. The preregistration states the count." >&2
  exit 2
fi

if [ "$SEEDS" -lt "$MIN" ]; then
  echo "Blocked: $SEEDS seeds is below the minimum of $MIN for a test-tier run. Use the seed count the preregistration states, or supersede it with a new dated file." >&2
  exit 2
fi

exit 0
