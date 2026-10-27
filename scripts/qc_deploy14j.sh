#!/bin/bash
# task134: prepared for the controller; do not execute during delivery.
# Overlay only curriculum_queue.py from deploy14i. Keep all current mode files.
# start_qc is copied verbatim from the current ~/pvz-logs/qc_deploy15.sh.
set -Eeuo pipefail
R=/home/newbiexvwu/PvZ-search
W=$R/.claude/worktrees/cq
PACKAGE=$R/.claude/worktrees/obsup
PY=/home/newbiexvwu/.venvs/ml/bin/python3
REV=26a80f6674593103f2417ff8a2224d98b0592219
BASE_SHA=0559cdd4edaeb6bf1f4ccb5d48177fea4676eb5768bac4796d2345f46f17bfb8
NEW_SHA=195fc4bdc602201170350e45fc5eb81bee93a51919899b9b5e1c3594a23675eb
NEWEXE=$W/build6/pvz-portable
EXE15=$PACKAGE/build7/pvz-portable
MODE14I=/home/newbiexvwu/pvz-logs/qc_deploy14i.mode
MODE14H=/home/newbiexvwu/pvz-logs/qc_deploy14h.mode
MODE15=/home/newbiexvwu/pvz-logs/qc_deploy15.mode
OPENING=$W/artifacts/task119/opening_panel.json
PANEL3=$W/artifacts/task73c/panel_v3_source.json
PANEL5=$W/configs/gate_v5_panel.json
A15=$W/recipes/dw4i2_v2sleep.json
B15=$W/recipes/dw4i2_rebalanced_p05_v2sleep.json
LOG_DIR=/home/newbiexvwu/pvz-logs
L=$LOG_DIR/qc_setup.log
O=$LOG_DIR/deploy14j_$(date +%Y%m%dT%H%M%S)
STOP_FLAG=$LOG_DIR/stop_curr
CODE=$W/scripts/curriculum_queue.py
BACKUP=$O/curriculum_queue.py.before
STOPPED=0; SAVED=0; OWN_STOP=0
start_qc() {
  if [ -f "$MODE15" ] && grep -qx v2sleep "$MODE15"; then
    tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $A15 --alternate-recipe $B15 --candidates-per-recipe 2 --training-parallelism 1 --selection-holdout-manifest $W/configs/student_selection_holdout.json --opening-manifest $OPENING --executable $EXE15 --promotion-panel $PANEL5 --gate-version 5 2>&1 | tee -a ~/pvz-logs/qc.out"
  elif [ -f "$MODE14I" ] && grep -qx gatev5 "$MODE14I"; then
    tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $W/recipes/dw4i2.json --alternate-recipe $W/recipes/dw4i2_rebalanced_p05.json --candidates-per-recipe 2 --training-parallelism 1 --selection-holdout-manifest $W/configs/student_selection_holdout.json --opening-manifest $OPENING --executable $NEWEXE --promotion-panel $PANEL5 --gate-version 5 2>&1 | tee -a ~/pvz-logs/qc.out"
  elif [ -f "$MODE14H" ] && grep -qx rebalance "$MODE14H"; then
    tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $W/recipes/dw4i2.json --alternate-recipe $W/recipes/dw4i2_rebalanced_p05.json --candidates-per-recipe 2 --training-parallelism 1 --selection-holdout-manifest $W/configs/student_selection_holdout.json --opening-manifest $OPENING --executable $NEWEXE --promotion-panel $PANEL3 2>&1 | tee -a ~/pvz-logs/qc.out"
  else
    tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $W/recipes/dw4i2.json --alternate-recipe $W/recipes/dw4i2_rebalanced.json --opening-manifest $OPENING --executable $NEWEXE --promotion-panel $PANEL3 2>&1 | tee -a ~/pvz-logs/qc.out"
  fi
}
log() { printf '%s deploy14j: %s\n' "$(date -Iseconds)" "$*" >> "$L"; }
cleanup() { if [ "$OWN_STOP" = 1 ]; then rm -f "$STOP_FLAG"; fi; }
stop_qc() {
  if ! tmux has-session -t =qc 2>/dev/null; then return; fi
  touch "$STOP_FLAG"; OWN_STOP=1
  for ((i=0; i<720; i++)); do
    if ! tmux has-session -t =qc 2>/dev/null; then
      rm -f "$STOP_FLAG"; OWN_STOP=0; return
    fi
    sleep 30
  done
  log 'ABORT: qc did not drain within six hours; no forced switch'
  return 1
}
rollback() {
  trap - ERR
  log "$1; restoring the previous controller with unchanged start_qc arguments"
  stop_qc || exit 1
  if [ "$SAVED" = 1 ]; then
    printf '%s  %s\n' "$BASE_SHA" "$BACKUP" | sha256sum -c - || exit 1
    cp -p "$BACKUP" "$CODE" || exit 1
  fi
  printf '%s  %s\n' "$BASE_SHA" "$CODE" | sha256sum -c - || exit 1
  start_qc
  exit 1
}
on_error() {
  local status=$?
  trap - ERR
  if [ "$STOPPED" = 1 ]; then rollback "failure $status at line $1"; fi
  log "ABORT before switch: status $status line $1"
  exit "$status"
}
trap cleanup EXIT
trap 'on_error $LINENO' ERR

[ ! -e "$STOP_FLAG" ] || { log 'ABORT: another stop request exists'; exit 1; }
printf '%s  %s\n' "$BASE_SHA" "$CODE" | sha256sum -c -
git -C "$W" show "$REV:scripts/curriculum_queue.py" | sha256sum | grep -q "^$NEW_SHA "
[ -x "$PY" ]
if [ -f "$MODE15" ] && grep -qx v2sleep "$MODE15"; then
  [ -x "$EXE15" ]
else
  [ -x "$NEWEXE" ]
fi
mkdir -p "$O"
log "preflight passed for deploy14i controller; overlay $REV; no corpus replay or pack warmup"
stop_qc; STOPPED=1
printf '%s  %s\n' "$BASE_SHA" "$CODE" | sha256sum -c -
cp -p "$CODE" "$BACKUP"; SAVED=1
git -C "$W" restore --source "$REV" --worktree -- scripts/curriculum_queue.py
printf '%s  %s\n' "$NEW_SHA" "$CODE" | sha256sum -c -
"$PY" -c 'import ast, sys; ast.parse(open(sys.argv[1]).read())' "$CODE"
n0=$(grep -c ' rc=' "$LOG_DIR/curriculum.log" || true)
start_qc
log 'anonymous-memory budget and offset-scoped collection labels activated; modes and state retained'
for ((i=0; i<120; i++)); do
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
rollback 'first new group failed or timed out'
