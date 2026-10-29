#!/bin/bash
# task165: PREPARED ONLY. The main controller reviews/installs this file; do not auto-run it.
# Application source is pinned; reuse deploy16's binary. No rebuild/replay/pack/seed reset.
set -Eeuo pipefail
R=/home/newbiexvwu/PvZ-search
W=$R/.claude/worktrees/cq
PACKAGE=$R
PY=/home/newbiexvwu/.venvs/ml/bin/python3
REV=18313d3c
LOG_DIR=/home/newbiexvwu/pvz-logs
MODE14P=$LOG_DIR/qc_deploy14p.mode
MODE14I=$LOG_DIR/qc_deploy14i.mode
MODE14H=$LOG_DIR/qc_deploy14h.mode
MODE15=$LOG_DIR/qc_deploy15.mode
MODE16=$LOG_DIR/qc_deploy16.mode
ACTIVE16=$LOG_DIR/qc_deploy16.active
MODE17=$LOG_DIR/qc_deploy17.mode
ACTIVE17=$LOG_DIR/qc_deploy17.active
OPENING=$W/artifacts/task119/opening_panel.json
PANEL3=$W/artifacts/task73c/panel_v3_source.json
PANEL5=$W/configs/gate_v5_panel.json
EXTRA=$W/artifacts/task137/training_panel.json
EXTRA17=$W/artifacts/task165/training_panel.json
NEWEXE=$W/build6/pvz-portable
EXE15=$R/.claude/worktrees/obsup/build7/pvz-portable
EXE16=$R/artifacts/production/deploy16/pvz-portable
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
start_qc() {
  QC_GATE=""
  QC_SELECTION=""
  QC_EXTRA=$EXTRA
  QC_MIX=""
  if [ -f "$ACTIVE17" ] && [ -f "$MODE17" ] && grep -qx datamix "$MODE17"; then
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
  tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $QC_RECIPE --alternate-recipe $QC_ALTERNATE $QC_SELECTION --opening-manifest $OPENING --executable $QC_EXECUTABLE --promotion-panel $QC_PANEL $QC_GATE --extra-manifest $QC_EXTRA $QC_MIX 2>&1 | tee -a $LOG_DIR/qc.out"
}
# All mutations below run only after explicit main-controller approval/execution.
FILES=(scripts/curriculum_queue.py scripts/make_random_deck_cells.py scripts/deck_rules.py
       scripts/make_multiplier_tasks.py scripts/make_day_panel.py
       scripts/day_orig_tasks.py scripts/render_episode.py scripts/unified_panel.py
       scripts/pack_training_rows.py scripts/train_rollout_policy.py
       python/pvz_recipe.py python/pvz_training_pack.py python/pvz_training_rows.py
       python/pvz_frame_store.py
       configs/deploy17_random_deck_pool.json configs/deploy17_deck_holdouts.json
       recipes/dw4i2_v2timer.json recipes/dw4i2_v2timer_history10.json)
GENERATED=artifacts/task165/training_panel.json
O=""
SAVED=0
STOPPED=0
log() { printf '%s deploy17: %s\n' "$(date -Iseconds)" "$*" >> "$LOG_DIR/qc_setup.log"; }
helper() {
  if [ ! -f "$O/boundary-helper.py" ]; then
    git -C "$PACKAGE" show "$REV:scripts/deploy17_boundary.py" > "$O/boundary-helper.py"
  fi
  "$PY" "$O/boundary-helper.py" "$1" --worktree "$W" \
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
  for file in "${FILES[@]}" "$GENERATED"; do
    mkdir -p "$(dirname "$O/before/$file")"
    if [ -e "$W/$file" ]; then cp -p "$W/$file" "$O/before/$file";
    else touch "$O/before/$file.absent"; fi
  done
  mkdir -p "$O/modes"
  for mode in "$MODE17" "$ACTIVE17"; do
    if [ -e "$mode" ]; then cp -p "$mode" "$O/modes/$(basename "$mode")";
    else touch "$O/modes/$(basename "$mode").absent"; fi
  done
  printf 'complete\n' > "$O/backed_up"
  SAVED=1
}
restore_files() {
  local file mode
  for file in "${FILES[@]}" "$GENERATED"; do
    if [ -e "$O/before/$file.absent" ]; then rm -f "$W/$file";
    else cp -p "$O/before/$file" "$W/$file"; fi
  done
  for mode in "$MODE17" "$ACTIVE17"; do
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
[ "$(realpath "$0")" = "$LOG_DIR/qc_deploy17.sh" ] || {
  printf 'After review, install this script at %s/qc_deploy17.sh (watchdog deployment guard).\n' "$LOG_DIR" >&2; exit 2; }
[ ! -e "$LOG_DIR/stop_curr" ] || { log 'ABORT: existing stop request'; exit 1; }
exec 9> "$LOG_DIR/qc_deploy17.lock"
flock -n 9 || { log 'ABORT: another deploy17 switch is running'; exit 1; }
case "${1:-}" in
  --rollback)
    O=${2:-$(cat "$LOG_DIR/deploy17.latest")}
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
[ ! -e "$ACTIVE17" ] || { log 'ABORT: deploy17 already active'; exit 1; }
[ -f "$ACTIVE16" ] && [ -f "$MODE16" ] && grep -qx v2timer "$MODE16"
[ -x "$PY" ] && [ -x "$EXE16" ]
O=$LOG_DIR/deploy17_$(date +%Y%m%dT%H%M%S)_$$
mkdir -p "$O/install"
# Fetch pinned small source/config files and generate the new panel before pausing qc.
for file in "${FILES[@]}"; do
  mkdir -p "$(dirname "$O/install/$file")"
  git -C "$PACKAGE" show "$REV:$file" > "$O/install/$file"
done
/bin/bash -c 'source "$1"; watchdog_extract_start "$2"' deploy17 \
  "$W/scripts/qc_watchdog.sh" "$LOG_DIR/qc_deploy16.sh" > "$O/previous-launch.sh"
cp -p "$LOG_DIR/qc_deploy16.sh" "$O/previous-deploy.sh"
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 "$PY" "$W/scripts/make_random_deck_cells.py" \
  --pool-registry "$O/install/configs/deploy17_random_deck_pool.json" \
  --heldout-panel "$O/install/configs/deploy17_deck_holdouts.json" \
  --heldout-panel "$W/configs/gate_v5_panel.json" \
  --heldout-panel "$W/configs/student_selection_v2.json" \
  --heldout-panel "$W/configs/student_selection_holdout.json" --out "$O/panel"
mkdir -p "$O/install/$(dirname "$GENERATED")"
cp "$O/panel/training_panel.json" "$O/install/$GENERATED"
log "prepared source $REV and frozen pool; waiting for collection boundary"
wait_boundary
# Frozen idle scheduler: no next collector can race the backup and switch.
backup_files
printf '%s\n' "$O" > "$LOG_DIR/deploy17.latest"
stop_paused_qc
for file in "${FILES[@]}" "$GENERATED"; do
  mkdir -p "$(dirname "$W/$file")"
  cp "$O/install/$file" "$W/$file.deploy17.building"
  mv "$W/$file.deploy17.building" "$W/$file"
done
printf 'datamix\n' > "$MODE17"
printf '%s\n' "$REV" > "$ACTIVE17.building"
mv "$ACTIVE17.building" "$ACTIVE17"
start_qc
log "activated demo<=25%, random25%, shared-pack historical-weight B; binary/state/ledgers retained; backup $O"
