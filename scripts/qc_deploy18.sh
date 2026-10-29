#!/bin/bash
# obsrec: PREPARED ONLY. Deploy18 adds rich root attachments to the deploy17 launch.
# Build the pinned obsrec engine in build18 before review/install; never build at the switch.
set -Eeuo pipefail
R=/home/newbiexvwu/PvZ-search
W=$R/.claude/worktrees/cq
PACKAGE=$R/.claude/worktrees/obsrec
PY=/home/newbiexvwu/.venvs/ml/bin/python3
REV=de36241a
LOG_DIR=/home/newbiexvwu/pvz-logs
MODE14P=$LOG_DIR/qc_deploy14p.mode
MODE14I=$LOG_DIR/qc_deploy14i.mode
MODE14H=$LOG_DIR/qc_deploy14h.mode
MODE15=$LOG_DIR/qc_deploy15.mode
MODE16=$LOG_DIR/qc_deploy16.mode
ACTIVE16=$LOG_DIR/qc_deploy16.active
MODE17=$LOG_DIR/qc_deploy17.mode
ACTIVE17=$LOG_DIR/qc_deploy17.active
MODE18=$LOG_DIR/qc_deploy18.mode
ACTIVE18=$LOG_DIR/qc_deploy18.active
OPENING=$W/artifacts/task119/opening_panel.json
PANEL3=$W/artifacts/task73c/panel_v3_source.json
PANEL5=$W/configs/gate_v5_panel.json
EXTRA=$W/artifacts/task137/training_panel.json
EXTRA17=$W/artifacts/task165/training_panel.json
NEWEXE=$W/build6/pvz-portable
EXE15=$R/.claude/worktrees/obsup/build7/pvz-portable
EXE16=$R/.claude/worktrees/prodint16/build16/pvz-portable
EXE18=$PACKAGE/build18/pvz-portable
A15=$W/recipes/dw4i2_v2sleep.json
B15=$W/recipes/dw4i2_rebalanced_p05_v2sleep.json
A16=$W/recipes/dw4i2_v2timer.json
B16=$W/recipes/dw4i2_rebalanced_p05_v2timer.json
B17=$W/recipes/dw4i2_v2timer_history10.json
QC_RECIPE=""
QC_ALTERNATE=""
QC_EXECUTABLE=""
QC_PANEL=""
QC_GATE=""
QC_SELECTION=""
QC_EXTRA=""
QC_MIX=""
QC_OBS=""
start_qc() {
  QC_GATE=""
  QC_SELECTION=""
  QC_EXTRA=$EXTRA
  QC_MIX=""
  QC_OBS=""
  if [ -f "$ACTIVE18" ] && [ -f "$MODE18" ] && grep -qx obs_v3 "$MODE18"; then
    QC_RECIPE=$A16; QC_ALTERNATE=$B17; QC_EXECUTABLE=$EXE18; QC_PANEL=$PANEL5
    QC_EXTRA=$EXTRA17; QC_MIX="--demo-start-share 0.25 --random-deck-share 0.25"
    QC_GATE="--gate-version 5"; QC_OBS="--obs-v3"
    QC_SELECTION="--candidates-per-recipe 2 --training-parallelism 1 --selection-holdout-manifest $W/configs/student_selection_holdout.json"
  elif [ -f "$ACTIVE17" ] && [ -f "$MODE17" ] && grep -qx datamix "$MODE17"; then
    QC_RECIPE=$A16; QC_ALTERNATE=$B17; QC_EXECUTABLE=$EXE16; QC_PANEL=$PANEL5
    QC_EXTRA=$EXTRA17; QC_MIX="--demo-start-share 0.25 --random-deck-share 0.25"
    QC_GATE="--gate-version 5"
    QC_SELECTION="--candidates-per-recipe 2 --training-parallelism 1 --selection-holdout-manifest $W/configs/student_selection_holdout.json"
  elif [ -f "$ACTIVE16" ] && [ -f "$MODE16" ] && grep -qx v2timer "$MODE16"; then
    QC_RECIPE=$A16; QC_ALTERNATE=$B16; QC_EXECUTABLE=$EXE16; QC_PANEL=$PANEL5
    QC_GATE="--gate-version 5"
    QC_SELECTION="--candidates-per-recipe 2 --training-parallelism 1 --selection-holdout-manifest $W/configs/student_selection_holdout.json"
  elif [ -f "$MODE15" ] && grep -qx v2sleep "$MODE15"; then
    QC_RECIPE=$A15; QC_ALTERNATE=$B15; QC_EXECUTABLE=$EXE15; QC_PANEL=$PANEL5
    QC_GATE="--gate-version 5"
    QC_SELECTION="--candidates-per-recipe 2 --training-parallelism 1 --selection-holdout-manifest $W/configs/student_selection_holdout.json"
  elif [ -f "$MODE14I" ] && grep -qx gatev5 "$MODE14I"; then
    QC_RECIPE=$W/recipes/dw4i2.json; QC_ALTERNATE=$W/recipes/dw4i2_rebalanced_p05.json
    QC_EXECUTABLE=$NEWEXE; QC_PANEL=$PANEL5; QC_GATE="--gate-version 5"
    QC_SELECTION="--candidates-per-recipe 2 --training-parallelism 1 --selection-holdout-manifest $W/configs/student_selection_holdout.json"
  elif [ -f "$MODE14H" ] && grep -qx rebalance "$MODE14H"; then
    QC_RECIPE=$W/recipes/dw4i2.json; QC_ALTERNATE=$W/recipes/dw4i2_rebalanced_p05.json
    QC_EXECUTABLE=$NEWEXE; QC_PANEL=$PANEL3
    QC_SELECTION="--candidates-per-recipe 2 --training-parallelism 1 --selection-holdout-manifest $W/configs/student_selection_holdout.json"
  else
    QC_RECIPE=$W/recipes/dw4i2.json; QC_ALTERNATE=$W/recipes/dw4i2_rebalanced.json
    QC_EXECUTABLE=$NEWEXE; QC_PANEL=$PANEL3
  fi
  tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $QC_RECIPE --alternate-recipe $QC_ALTERNATE $QC_SELECTION --opening-manifest $OPENING --executable $QC_EXECUTABLE --promotion-panel $QC_PANEL $QC_GATE --extra-manifest $QC_EXTRA $QC_MIX $QC_OBS 2>&1 | tee -a $LOG_DIR/qc.out"
}
# All mutations below run only after explicit main-controller approval/execution.
FILES=(scripts/curriculum_queue.py scripts/search_play.py scripts/render_episode.py
       python/pvz_env.py python/pvz_research_data.py python/pvz_student_tokens.py
       python/pvz_token_observation.py)
O=""
SAVED=0
STOPPED=0
log() { printf '%s deploy18: %s\n' "$(date -Iseconds)" "$*" >> "$LOG_DIR/qc_setup.log"; }
helper() {
  "$PY" "$PACKAGE/scripts/deploy17_boundary.py" "$1" --worktree "$W" \
    --log "$LOG_DIR/curriculum.log" --record "$O/scheduler.json"
}
resume_paused() { if [ -f "$O/scheduler.json" ]; then helper resume; fi; }
stop_paused_qc() {
  STOPPED=1
  if [ -f "$O/scheduler.json" ]; then helper stop; fi
  tmux kill-session -t =qc 2>/dev/null || true
}
wait_boundary() {
  if tmux has-session -t =qc 2>/dev/null; then helper wait; fi
}
backup_files() {
  local file mode
  for file in "${FILES[@]}"; do
    mkdir -p "$(dirname "$O/before/$file")"
    if [ -e "$W/$file" ]; then cp -p "$W/$file" "$O/before/$file";
    else touch "$O/before/$file.absent"; fi
  done
  mkdir -p "$O/modes"
  for mode in "$MODE18" "$ACTIVE18"; do
    if [ -e "$mode" ]; then cp -p "$mode" "$O/modes/$(basename "$mode")";
    else touch "$O/modes/$(basename "$mode").absent"; fi
  done
  printf 'complete\n' > "$O/backed_up"
  SAVED=1
}
restore_files() {
  local file mode
  for file in "${FILES[@]}"; do
    if [ -e "$O/before/$file.absent" ]; then rm -f "$W/$file";
    else cp -p "$O/before/$file" "$W/$file"; fi
  done
  for mode in "$MODE18" "$ACTIVE18"; do
    if [ -e "$O/modes/$(basename "$mode").absent" ]; then rm -f "$mode";
    else cp -p "$O/modes/$(basename "$mode")" "$mode"; fi
  done
}
on_error() {
  local status=${2:-$?}
  trap - ERR INT TERM EXIT
  if [ "$STOPPED" = 1 ] && [ "$SAVED" = 1 ]; then
    log "failed at line $1 (status $status); restoring prior files/launch"
    if tmux has-session -t =qc 2>/dev/null; then helper wait || true; fi
    stop_paused_qc || true
    restore_files
    source "$O/previous-launch.sh"
    touch "$O/rollback.complete"
  else
    resume_paused
    log "aborted before switch at line $1 (status $status); current service retained"
  fi
  exit "$status"
}
trap 'on_error $LINENO' ERR
trap 'on_error $LINENO 130' INT
trap 'on_error $LINENO 143' TERM
# If interrupted while waiting/backup, never leave the current scheduler SIGSTOPed.
trap 'if [ "$STOPPED" = 0 ] && [ -n "$O" ]; then resume_paused; fi' EXIT

# --- switch: no action is performed merely by committing/copying this prepared script ---
[ "$(realpath "$0")" = "$LOG_DIR/qc_deploy18.sh" ] || {
  printf 'After review, install this script at %s/qc_deploy18.sh (watchdog deployment guard).\n' "$LOG_DIR" >&2; exit 2; }
[ ! -e "$LOG_DIR/stop_curr" ] || { log 'ABORT: existing stop request'; exit 1; }
exec 9> "$LOG_DIR/qc_deploy18.lock"
flock -n 9 || { log 'ABORT: another deploy18 switch is running'; exit 1; }
case "${1:-}" in
  --rollback)
    O=${2:-$(cat "$LOG_DIR/deploy18.latest")}
    [ -f "$O/backed_up" ] && [ -f "$O/previous-launch.sh" ]
    if [ -f "$O/rollback.complete" ]; then log "already rolled back: $O"; exit 0; fi
    SAVED=1
    rm -f "$O/scheduler.json"
    wait_boundary
    stop_paused_qc
    restore_files
    source "$O/previous-launch.sh"
    touch "$O/rollback.complete"
    log "rollback complete from $O; controller state, seed offsets and ledgers retained"
    exit 0
    ;;
  ''|--at-boundary) ;;
  *) printf 'usage: %s [--at-boundary | --rollback [backup-directory]]\n' "$0" >&2; exit 2 ;;
esac
[ ! -e "$ACTIVE18" ] || { log 'ABORT: deploy18 already active'; exit 1; }
[ -f "$ACTIVE17" ] && [ -f "$MODE17" ] && grep -qx datamix "$MODE17"
[ -x "$PY" ] && [ -x "$EXE18" ]
O=$LOG_DIR/deploy18_$(date +%Y%m%dT%H%M%S)_$$
mkdir -p "$O/install"
# Read pinned small source files before pausing qc. No frame replay, pack, or build here.
for file in "${FILES[@]}"; do
  mkdir -p "$(dirname "$O/install/$file")"
  git -C "$PACKAGE" show "$REV:$file" > "$O/install/$file"
done
/bin/bash -c 'source "$1"; watchdog_extract_start "$2"' deploy18 \
  "$W/scripts/qc_watchdog.sh" "$LOG_DIR/qc_deploy17.sh" > "$O/previous-launch.sh"
cp -p "$LOG_DIR/qc_deploy17.sh" "$O/previous-deploy.sh"
log "prepared obsrec source $REV and build18; waiting for collection boundary"
wait_boundary
# Frozen idle scheduler: no next collector can race the backup and switch.
backup_files
printf '%s\n' "$O" > "$LOG_DIR/deploy18.latest"
stop_paused_qc
for file in "${FILES[@]}"; do
  mkdir -p "$(dirname "$W/$file")"
  cp "$O/install/$file" "$W/$file.deploy18.building"
  mv "$W/$file.deploy18.building" "$W/$file"
done
printf 'obs_v3\n' > "$MODE18"
printf '%s\n' "$REV" > "$ACTIVE18.building"
mv "$ACTIVE18.building" "$ACTIVE18"
start_qc
log "activated obs-v3 root attachments; deploy17 recipes/mix/state/ledgers retained; backup $O"
