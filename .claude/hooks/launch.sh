#!/usr/bin/env bash
# Sourced by the Bash guards. Reads the hook input and decides whether the command launches an experiment.
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
CMD=$(jq -r '.tool_input.command // ""')

# a python interpreter running experiments/train*, compare* or sweep*, as a script path or with -m
launches_experiment() {
  printf '%s' "$CMD" | grep -qE '(^|[^[:alnum:]_.-])python[0-9.]*([[:space:]]+-[A-Za-z]+)*[[:space:]]+([^[:space:]]*/)?experiments[./](train|compare|sweep)'
}

tier_of_command() {
  printf '%s' "$CMD" | grep -oE -- '--tier[= ]+(dev|val|test)' | grep -oE '(dev|val|test)$' | tail -n 1
}
