#!/bin/bash
# task150 / deploy16. Prepared only: the main controller executes this at a group boundary.
# Mode16 is a prepared selector; ACTIVE16 is written only after installation, so a
# watchdog that discovers this prepared script still launches the current service.
# No corpus rebuild, replay, training, or hash checks belong in this switch.
set -Eeuo pipefail
R=/home/newbiexvwu/PvZ-search
W=$R/.claude/worktrees/cq
PACKAGE=$R/.claude/worktrees/prodint16
PY=/home/newbiexvwu/.venvs/ml/bin/python3
REV=4e479b04
FILES=(scripts/curriculum_queue.py scripts/select_student.py scripts/promotion_gate.py
       python/pvz_env.py python/pvz_frame_store.py python/pvz_recipe.py
       python/pvz_rollout_policy.py python/pvz_student_observation.py python/pvz_value_net.py
       scripts/pack_training_rows.py scripts/train_rollout_policy.py scripts/train_value_net.py
       scripts/render_episode.py scripts/make_multiplier_tasks.py
       recipes/dw4i2_v2timer.json recipes/dw4i2_rebalanced_p05_v2timer.json)
NEWEXE=$W/build6/pvz-portable
EXE15=$R/.claude/worktrees/obsup/build7/pvz-portable
EXE16=$PACKAGE/build16/pvz-portable
MODE14P=/home/newbiexvwu/pvz-logs/qc_deploy14p.mode
MODE14I=/home/newbiexvwu/pvz-logs/qc_deploy14i.mode
MODE14H=/home/newbiexvwu/pvz-logs/qc_deploy14h.mode
MODE15=/home/newbiexvwu/pvz-logs/qc_deploy15.mode
MODE16=/home/newbiexvwu/pvz-logs/qc_deploy16.mode
ACTIVE16=/home/newbiexvwu/pvz-logs/qc_deploy16.active
OPENING=$W/artifacts/task119/opening_panel.json
PANEL3=$W/artifacts/task73c/panel_v3_source.json
PANEL5=$W/configs/gate_v5_panel.json
EXTRA=$W/artifacts/task137/training_panel.json
A15=$W/recipes/dw4i2_v2sleep.json
B15=$W/recipes/dw4i2_rebalanced_p05_v2sleep.json
A16=$W/recipes/dw4i2_v2timer.json
B16=$W/recipes/dw4i2_rebalanced_p05_v2timer.json
LOG_DIR=/home/newbiexvwu/pvz-logs
L=$LOG_DIR/qc_setup.log
O=$LOG_DIR/deploy16_$(date +%Y%m%dT%H%M%S)
STOP_FLAG=$LOG_DIR/stop_curr
STOPPED=0
SAVED=0
# These bindings are top-level literals for watchdog_extract_start's dependency closure.
QC_RECIPE=""
QC_ALTERNATE=""
QC_EXECUTABLE=""
QC_PANEL=""
QC_GATE=""
QC_SELECTION=""
start_qc() {
  QC_GATE=""
  QC_SELECTION=""
  if [ -f "$ACTIVE16" ] && [ -f "$MODE16" ] && grep -qx v2timer "$MODE16"; then
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
  tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $QC_RECIPE --alternate-recipe $QC_ALTERNATE $QC_SELECTION --opening-manifest $OPENING --executable $QC_EXECUTABLE --promotion-panel $QC_PANEL $QC_GATE --extra-manifest $EXTRA 2>&1 | tee -a $LOG_DIR/qc.out"
}
log() { printf '%s deploy16: %s\n' "$(date -Iseconds)" "$*" >> "$L"; }
stop_qc() {
  if ! tmux has-session -t =qc 2>/dev/null; then return; fi
  # No stop flag/drain: the main controller chooses the group boundary.
  pids=$(pgrep -f 'worktrees/cq/scripts/(search_play|train_rollout_policy)' || true)
  tmux kill-session -t =qc || true
  sleep 4
  for pid in $pids; do kill -TERM "$pid" 2>/dev/null || true; done
  sleep 5
}
restore_files() {
  if [ "$SAVED" = 1 ]; then
    for file in "${FILES[@]}"; do
      if [ -e "$O/$file.before" ]; then
        cp -p "$O/$file.before" "$W/$file"
      else
        rm -f "$W/$file"
      fi
    done
  fi
  rm -f "$ACTIVE16"
}
rollback() {
  trap - ERR
  log "$1; restoring prior code and launch arguments"
  stop_qc || exit 1
  restore_files
  start_qc
  exit 1
}
on_error() {
  status=$?
  trap - ERR
  if [ "$STOPPED" = 1 ]; then rollback "failure $status at line $1"; fi
  log "ABORT before switch: status $status line $1"
  exit "$status"
}
trap 'on_error $LINENO' ERR

# --- switch: preparation and cache construction happen outside this script ---
[ ! -e "$STOP_FLAG" ] || { log 'ABORT: another stop request exists'; exit 1; }
if [ "${1:-}" = "--rollback" ]; then
  O=$(ls -dt "$LOG_DIR"/deploy16_* 2>/dev/null | head -1)
  [ -n "${O:-}" ] && [ -f "$O/backed_up" ] || { log 'ABORT: no complete deploy16 backup'; exit 1; }
  stop_qc; STOPPED=1; SAVED=1
  restore_files
  start_qc
  log 'manual rollback complete; controller state, seed offsets and ledgers retained'
  exit 0
fi
[ ! -e "$ACTIVE16" ] || { log 'ABORT: deploy16 is already active'; exit 1; }
[ -f "$MODE14P" ] && grep -qx deckcells "$MODE14P" || {
  log 'ABORT: deploy14p deckcells must already be active'; exit 1; }
[ -f "$MODE16" ] && grep -qx v2timer "$MODE16" || {
  log 'ABORT: qc_deploy16.mode must select v2timer'; exit 1; }
[ -x "$PY" ] && [ -x "$EXE16" ] && [ -f "$EXTRA" ]
# Obtain the pinned source files before stopping qc. This is small file IO only.
mkdir -p "$O/install"
for file in "${FILES[@]}"; do
  mkdir -p "$(dirname "$O/install/$file")"
  git -C "$PACKAGE" show "$REV:$file" > "$O/install/$file"
done
log "prepared $REV on independent build16; no production binary replacement"
stop_qc; STOPPED=1
for file in "${FILES[@]}"; do
  if [ -e "$W/$file" ]; then
    mkdir -p "$(dirname "$O/$file")"
    cp -p "$W/$file" "$O/$file.before"
  fi
done
printf 'backed up\n' > "$O/backed_up"
SAVED=1
for file in "${FILES[@]}"; do
  mkdir -p "$(dirname "$W/$file")"
  cp "$O/install/$file" "$W/$file"
done
printf '%s\n' "$REV" > "$ACTIVE16.building"
mv "$ACTIVE16.building" "$ACTIVE16"
n0=$(grep -c ' rc=' "$LOG_DIR/curriculum.log" || true)
start_qc
log 'v2timer A/B + systemic failure pause activated; incumbent and controller state retained'
for ((attempt=0; attempt<120; attempt++)); do
  sleep 30
  if ! tmux has-session -t =qc 2>/dev/null; then break; fi
  n1=$(grep -c ' rc=' "$LOG_DIR/curriculum.log" || true)
  if [ "$n1" -gt "$n0" ]; then
    if tail -n 200 "$LOG_DIR/curriculum.log" | grep ' rc=' | tail -1 | grep -q ' rc=0 '; then
      log 'first new group rc=0; deployment complete'; exit 0
    fi
    break
  fi
done
if tmux has-session -t =qc 2>/dev/null && [ "$(grep -c ' rc=' "$LOG_DIR/curriculum.log" || true)" -le "$n0" ]; then
  log 'first new group still running/paused after 60 min; left live for main controller'; exit 0
fi
rollback 'first new group failed or qc exited'
