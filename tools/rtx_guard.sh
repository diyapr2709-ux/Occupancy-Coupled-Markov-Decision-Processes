#!/usr/bin/env bash
# Crash guard for experiment launches on this Windows host (Git Bash).
#
#   rtx_guard.sh check  WORKERS [-- CMD]  refuse (exit 2) unless a launch of WORKERS processes is safe now
#   rtx_guard.sh launch WORKERS -- CMD    run CMD under the guard: same checks, then supervise it
#   rtx_guard.sh status                   list guarded launches: RUNNING, EXITED <code> or LOST
#
# Why: on 26 November a run was relaunched while it was still running, two 30-worker sweeps
# overlapped, and the second died when Windows ran out of commit memory (WinError 1455). The
# host also rebooted twice that day. The guard refuses a launch that could overlap a live one
# or exhaust commit memory, and it makes a run killed by a crash visible as LOST.
#
# Liveness is decided by the run's Windows process id, not by the guard's own heartbeat: the
# guard's shell can die (for example when the agent session that started it ends) while the
# run's python.exe keeps going. A run is LOST only if it started before the last boot, or if
# its process is gone without an exit code being recorded. The pid recorded is that of the first
# Windows process the command starts; with the venv's python.exe that is its launcher, which
# waits for the interpreter it starts, so it lives exactly as long as the run. If it ever did not,
# the run's launch record would still have no report and no exit file, and the open-record check
# would still refuse.
#
# A launch record without a report that is known to be dead is listed, with its evidence, in
# tools/rtx_guard.known_dead (one line: <launch id> <evidence>). Only listed ids are skipped.
#
# It never relaunches anything by itself: a LOST run is reported, and any relaunch is a new,
# explicit command that the repository's launch hook records in runs/ledger.txt like any other.
# Launch CMD must contain the literal "python ... experiments/..." invocation, so the hook sees it,
# and an explicit "--workers N" equal to WORKERS, so the limits apply to what actually runs.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
STATE="${LOCALAPPDATA:-$HOME/AppData/Local}/oc-sac-rtx-guard"
MAX_WORKERS=30            # at most 30 worker processes at once on this 32-core host
COMMIT_PER_WORKER_MB=1310 # 1.25 x 1 050 MB: system commit in use divided by 30 planner workers on 26 November, an upper bound per worker; training workers not measured
KNOWN_DEAD="$ROOT/tools/rtx_guard.known_dead"
COMMIT_MARGIN_MB=4096     # kept free for the driver, the OS and other programs
HEAVY_PYTHON_MB=200       # a python.exe above this working set is treated as a running job
mkdir -p "$STATE"

refuse() { echo "refused: $*" >&2; return 2; }

boot_epoch() { echo $(( $(date +%s) - $(cut -d. -f1 /proc/uptime) )); }

available_commit_mb() {
  systeminfo 2>/dev/null | grep -i 'Virtual Memory: Available' | grep -oE '[0-9][0-9,]*' | head -n 1 | tr -d ','
}

commit_in_use_mb() {
  systeminfo 2>/dev/null | grep -i 'Virtual Memory: In Use' | grep -oE '[0-9][0-9,]*' | head -n 1 | tr -d ','
}

winpid_alive() { [ -n "$1" ] && tasklist //FI "PID eq $1" //FO CSV //NH 2>/dev/null | grep -q "\"$1\""; }

heavy_python_count() {
  tasklist //FI "IMAGENAME eq python.exe" //FO CSV //NH 2>/dev/null \
    | awk -F'","' -v lim="$HEAVY_PYTHON_MB" '{ m=$5; gsub(/[^0-9]/, "", m); if (m + 0 > lim * 1024) n++ } END { print n + 0 }'
}

# State of one guarded run directory: RUNNING, EXITED <code>, or LOST (<reason>).
run_state() {
  local d="$1"
  if [ -f "$d/exit" ]; then echo "EXITED $(cat "$d/exit")"
  elif [ "$(cat "$d/start")" -le "$(boot_epoch)" ]; then echo "LOST (started before the last boot)"
  elif winpid_alive "$(cat "$d/winpid" 2>/dev/null)"; then echo "RUNNING"
  else echo "LOST (process gone, no exit code recorded)"; fi
}

live_guarded() {
  local n=0 d
  for d in "$STATE"/*/; do
    [ -f "$d/start" ] || continue
    [ "$(run_state "$d")" = "RUNNING" ] && n=$((n + 1))
  done
  echo "$n"
}

# A launch record with no report could still be running unless it started before the last boot
# or its guarded run has recorded an exit code. A record whose start time cannot be read counts as open.
unguarded_open() {
  local boot started raw id dir n=0 f g exited
  boot=$(boot_epoch)
  for f in "$ROOT"/runs/*/launches/*.json; do
    [ -f "$f" ] || continue
    id=$(basename "$f" .json); dir=$(dirname "$(dirname "$f")")
    [ -f "$dir/$id.json" ] && continue
    [ -f "$KNOWN_DEAD" ] && grep -q "^$id " "$KNOWN_DEAD" && continue
    exited=0
    for g in "$STATE"/*/; do
      [ "$(cat "$g/launch_id" 2>/dev/null)" = "$id" ] && [ -f "$g/exit" ] && exited=1
    done
    [ "$exited" -eq 1 ] && continue
    raw=$(jq -r '.started // empty' "$f" 2>/dev/null)
    if [ -z "$raw" ] || ! started=$(date -d "$raw" +%s 2>/dev/null); then
      echo "  start time unreadable: $id" >&2; n=$((n + 1)); continue
    fi
    if [ "$started" -gt "$boot" ]; then echo "  open since $raw: $id" >&2; n=$((n + 1)); fi
  done
  echo "$n"
}

check() {
  local workers="$1"; shift
  [ "${1:-}" = "--" ] && shift
  local ok=0 live open heavy avail need cmd_workers
  command -v jq >/dev/null || { refuse "jq is not on PATH, so launch records cannot be read"; ok=2; }
  if [ "$#" -gt 0 ]; then
    cmd_workers=$(printf '%s ' "$@" | grep -oE -- '--workers[= ]+[0-9]+' | grep -oE '[0-9]+$' | tail -n 1)
    [ "${cmd_workers:-}" = "$workers" ] || { refuse "the command must pass --workers $workers explicitly (found: ${cmd_workers:-none})"; ok=2; }
  fi
  [ "$workers" -le "$MAX_WORKERS" ] || { refuse "$workers workers exceeds the host limit of $MAX_WORKERS"; ok=2; }
  live=$(live_guarded)
  [ "$live" -eq 0 ] || { refuse "$live guarded launch(es) are still running"; ok=2; }
  open=$(unguarded_open)
  [ "$open" -eq 0 ] || { refuse "$open launch record(s) without a report may still be running"; ok=2; }
  heavy=$(heavy_python_count)
  [ "$heavy" -eq 0 ] || { refuse "$heavy python.exe process(es) above ${HEAVY_PYTHON_MB} MB are running"; ok=2; }
  avail=$(available_commit_mb); need=$(( workers * COMMIT_PER_WORKER_MB + COMMIT_MARGIN_MB ))
  if [ -z "$avail" ]; then refuse "available commit memory could not be read"; ok=2
  elif [ "$avail" -lt "$need" ]; then refuse "${avail} MB commit memory available, ${need} MB needed for $workers workers"; ok=2; fi
  [ "$ok" -eq 0 ] && echo "check passed: $workers workers, ${avail:-?} MB commit memory available, ${need} MB needed"
  return "$ok"
}

launch() {
  local workers="$1"; shift
  [ "${1:-}" = "--" ] && shift
  check "$workers" -- "$@" || exit 2
  local id dir before pid winpid rc
  id="guard_$(date +%Y%m%d_%H%M%S)_$$"; dir="$STATE/$id"; mkdir -p "$dir"
  printf '%s\n' "$*" > "$dir/command"; git -C "$ROOT" rev-parse HEAD > "$dir/head"
  date +%s > "$dir/start"
  before=$(ls "$ROOT"/runs/*/launches/*.json 2>/dev/null | sort)
  ( cd "$ROOT" && exec "$@" ) > "$dir/output.log" 2>&1 &
  pid=$!; echo "$pid" > "$dir/pid"
  winpid=$(cat "/proc/$pid/winpid" 2>/dev/null); echo "${winpid:-}" > "$dir/winpid"
  echo "guarded launch $id, pid $pid, Windows pid ${winpid:-unknown}, output in $dir/output.log"
  while kill -0 "$pid" 2>/dev/null; do
    echo "$(date +%s) $(commit_in_use_mb)" >> "$dir/commit_in_use_mb"
    if [ ! -s "$dir/launch_id" ]; then
      comm -13 <(echo "$before") <(ls "$ROOT"/runs/*/launches/*.json 2>/dev/null | sort) | head -n 1 \
        | xargs -r basename -s .json > "$dir/launch_id"
    fi
    sleep 30
  done
  wait "$pid"; rc=$?
  echo "$rc" > "$dir/exit"; date +%s > "$dir/end"
  echo "guarded launch $id exited with code $rc (launch id: $(cat "$dir/launch_id" 2>/dev/null))"
  return "$rc"
}

status() {
  local d
  for d in "$STATE"/*/; do
    [ -f "$d/start" ] || continue
    echo "$(basename "$d")  launch=$(cat "$d/launch_id" 2>/dev/null)  $(run_state "$d")  started=$(date -d @"$(cat "$d/start")" '+%F %T')"
  done
}

case "${1:-}" in
  check) shift; check "${1:?workers}" "${@:2}" ;;
  launch) shift; launch "$@" ;;
  status) status ;;
  *) echo "usage: rtx_guard.sh check WORKERS [-- CMD] | launch WORKERS -- CMD | status" >&2; exit 64 ;;
esac
