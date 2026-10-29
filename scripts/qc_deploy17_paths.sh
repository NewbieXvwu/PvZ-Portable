#!/bin/bash
# task165: PREPARED ONLY. Main controller reviews before installing/executing this update.
set -Eeuo pipefail
R=/home/newbiexvwu/PvZ-search
W=$R/.claude/worktrees/cq
PY=/home/newbiexvwu/.venvs/ml/bin/python3
LOG_DIR=/home/newbiexvwu/pvz-logs
REV=2f038dbe
O=""
STOPPED=0
# A newly installed updater may be the watchdog's newest qc_deploy*.sh. Its launch
# bridge delegates to the current launcher without ever executing a deployment body.
start_qc() {
  /bin/bash -c 'source "$1"; watchdog_extract_start "$2" | /bin/bash' task165 \
    "$W/scripts/qc_watchdog.sh" "$LOG_DIR/qc_deploy17.sh"
}
helper() { "$PY" "$O/source/scripts/deploy17_boundary.py" "$1" --worktree "$W" \
  --log "$LOG_DIR/curriculum.log" --record "$O/scheduler.json"; }
update() { nice -n 10 "$PY" "$O/source/scripts/apply_deploy17_paths.py" "$1" --root "$R" \
  --logs "$LOG_DIR" --backup "$O" --revision "$REV" "${@:2}"; }
extract_start() { /bin/bash -c 'source "$1"; watchdog_extract_start "$2"' task165 \
  "$W/scripts/qc_watchdog.sh" "$LOG_DIR/qc_deploy17.sh"; }
log() { printf '%s deploy17_paths: %s\n' "$(date -Iseconds)" "$*" >> "$LOG_DIR/qc_setup.log"; }
resume_unmodified() {
  [ -n "$O" ] || return 0
  if [ "$STOPPED" = 0 ] && [ -f "$O/scheduler.json" ]; then helper resume; fi
  # This prepared updater has no start_qc. The watchdog must keep choosing the launcher.
  touch "$LOG_DIR/qc_deploy17.sh"
}
rollback_files() {
  if tmux has-session -t =qc 2>/dev/null; then
    rm -f "$O/scheduler.json"
    helper wait
    helper stop
    tmux kill-session -t =qc 2>/dev/null || true
  fi
  update restore
  /bin/bash "$O/previous-launch.sh"
  touch "$LOG_DIR/qc_deploy17.sh"
  log "restored pre-update code/paths/links from $O; state, model and seed offsets retained"
}
on_error() {
  local status=${2:-$?}
  trap - ERR INT TERM
  if [ "$STOPPED" = 1 ]; then rollback_files; else resume_unmodified; fi
  log "update failed at line $1 (status $status)"
  exit "$status"
}
trap 'on_error $LINENO' ERR
trap 'on_error $LINENO 130' INT
trap 'on_error $LINENO 143' TERM
trap resume_unmodified EXIT

[ "$(realpath "$0")" = "$LOG_DIR/qc_deploy17_paths.sh" ] || {
  printf 'After review, install this script at %s/qc_deploy17_paths.sh.\n' "$LOG_DIR" >&2; exit 2; }
[ ! -e "$LOG_DIR/stop_curr" ] || { log 'ABORT: existing stop request'; exit 1; }
exec 9> "$LOG_DIR/qc_deploy17.lock"
flock -n 9 || { log 'ABORT: deployment already in progress'; exit 1; }
case "${1:---at-boundary}" in
  --rollback)
    O=${2:-$(cat "$LOG_DIR/deploy17_paths.latest")}
    [ -f "$O/files.json" ] && [ -f "$O/previous-launch.sh" ]
    if [ -f "$O/rollback.complete" ]; then log "already rolled back: $O"; exit 0; fi
    STOPPED=1
    rollback_files
    touch "$O/rollback.complete"
    exit 0
    ;;
  --at-boundary) ;;
  *) printf 'usage: %s [--at-boundary | --rollback [backup-directory]]\n' "$0" >&2; exit 2 ;;
esac
if [ -f "$LOG_DIR/qc_deploy17.active" ] && grep -qx "$REV" "$LOG_DIR/qc_deploy17.active" &&
   [ -f "$LOG_DIR/deploy17_paths.latest" ] &&
   [ -f "$(cat "$LOG_DIR/deploy17_paths.latest")/complete" ]; then
  log "already active: $REV"
  touch "$LOG_DIR/qc_deploy17.sh"
  exit 0
fi
O=$LOG_DIR/deploy17_paths_$(date +%Y%m%dT%H%M%S)_$$
mkdir -p "$O/source"
FILES=(scripts/deploy17_boundary.py scripts/apply_deploy17_paths.py
       scripts/stabilize_production_paths.py scripts/retire_task20.py scripts/qc_deploy17.sh
       scripts/curriculum_queue.py scripts/day_orig_tasks.py python/pvz_frame_store.py python/pvz_recipe.py
       python/pvz_training_pack.py python/pvz_corpus_policy.py
       configs/panels/unified_v1.json
       configs/panels/unified_v2.json configs/panels/day_identity.json)
for file in "${FILES[@]}"; do
  mkdir -p "$(dirname "$O/source/$file")"
  git -C "$R" show "$REV:$file" > "$O/source/$file"
done
extract_start > "$O/previous-launch.sh"
update plan > "$O/review.json"
ENGINE=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["executable_source"])' "$O/review.json")
# One binary copy happens while collection continues, before freezing the scheduler.
update stage-engine --engine "$ENGINE" > "$O/executable.json"
log "prepared $REV; awaiting a durable group boundary; preview $O/review.json"
helper wait
STOPPED=1
helper stop
tmux kill-session -t =qc 2>/dev/null || true
printf '%s\n' "$O" > "$LOG_DIR/deploy17_paths.latest"
update apply --engine "$ENGINE"
extract_start > "$O/new-launch.sh"
start_qc
touch "$O/complete"
log "activated $REV with fixed panels/executable and 256-archive retirement; backup $O"
