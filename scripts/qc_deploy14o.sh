#!/bin/bash
# task141 / deploy14o: coalesce a training backlog onto the newest round.
#
# Only the main controller runs this script. It replaces the curriculum controller only,
# then restarts qc. A waiting training job whose round is behind a newer queued/running/
# trained one is marked superseded instead of trained and gated. stop_qc here is the
# desktop version: kill the session, do not drain a group. Background packing and
# training jobs are not touched.
set -Eeuo pipefail
R=/home/newbiexvwu/PvZ-search
W=$R/.claude/worktrees/cq
PACKAGE=$R/.claude/worktrees/obsup
PY=/home/newbiexvwu/.venvs/ml/bin/python3
REV=ce6dda325cf68be81feb2e6711b35843a5943979
FILES=(scripts/curriculum_queue.py)
NEW_SHAS=(9d4829c0545cad6c2340a019d4221549de2f38ab3e2fbee1a96a8f8b15ad3701)
# What production may be before the switch: deploy14l (task137), deploy14m (task138)
# or this deploy (a re-run after activation, including --rollback, must not abort).
BASELINE_CONTROLLER_SHAS=(a9d028fe2109f6039d6c373b0b47f8f757e8d4b020148337f962b40bcbb8886b
                          f98df24cbb5d8af69a237df6465da5c2fba77e55b8b231907544fdb7d68c76e9
                          9d4829c0545cad6c2340a019d4221549de2f38ab3e2fbee1a96a8f8b15ad3701)
NEWEXE=$W/build6/pvz-portable
EXE15=$PACKAGE/build7/pvz-portable
MODE14L=/home/newbiexvwu/pvz-logs/qc_deploy14l.mode
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
O=$LOG_DIR/deploy14o_$(date +%Y%m%dT%H%M%S)
EXTRA=$W/artifacts/task137/training_panel.json
STOPPED=0; SAVED=0
# Copied from the desktop's latest deploy script (deploy14m): the extra manifest follows
# the deploy14l mode file, so a plain restart keeps the 16 random-deck cells.
extra_arg() {
  if [ -f "$MODE14L" ] && grep -qx deckcells "$MODE14L"; then
    printf '%s' "--extra-manifest $EXTRA"
  fi
}
start_qc() {
  local extra
  extra=$(extra_arg)
  if [ -f "$MODE15" ] && grep -qx v2sleep "$MODE15"; then
    tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $A15 --alternate-recipe $B15 --candidates-per-recipe 2 --training-parallelism 1 --selection-holdout-manifest $W/configs/student_selection_holdout.json --opening-manifest $OPENING --executable $EXE15 --promotion-panel $PANEL5 --gate-version 5 $extra 2>&1 | tee -a ~/pvz-logs/qc.out"
  elif [ -f "$MODE14I" ] && grep -qx gatev5 "$MODE14I"; then
    tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $W/recipes/dw4i2.json --alternate-recipe $W/recipes/dw4i2_rebalanced_p05.json --candidates-per-recipe 2 --training-parallelism 1 --selection-holdout-manifest $W/configs/student_selection_holdout.json --opening-manifest $OPENING --executable $NEWEXE --promotion-panel $PANEL5 --gate-version 5 $extra 2>&1 | tee -a ~/pvz-logs/qc.out"
  elif [ -f "$MODE14H" ] && grep -qx rebalance "$MODE14H"; then
    tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $W/recipes/dw4i2.json --alternate-recipe $W/recipes/dw4i2_rebalanced_p05.json --candidates-per-recipe 2 --training-parallelism 1 --selection-holdout-manifest $W/configs/student_selection_holdout.json --opening-manifest $OPENING --executable $NEWEXE --promotion-panel $PANEL3 $extra 2>&1 | tee -a ~/pvz-logs/qc.out"
  else
    tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $W/recipes/dw4i2.json --alternate-recipe $W/recipes/dw4i2_rebalanced.json --opening-manifest $OPENING --executable $NEWEXE --promotion-panel $PANEL3 $extra 2>&1 | tee -a ~/pvz-logs/qc.out"
  fi
}
log() { printf '%s deploy14o: %s\n' "$(date -Iseconds)" "$*" >> "$L"; }
check_baseline() {
  local current sha accepted
  accepted=0
  current=$(sha256sum "$W/scripts/curriculum_queue.py" | cut -d' ' -f1)
  for sha in "${BASELINE_CONTROLLER_SHAS[@]}"; do
    if [ "$current" = "$sha" ]; then accepted=1; fi
  done
  [ "$accepted" = 1 ] || { log "ABORT: controller sha $current is neither deploy14l, deploy14m nor deploy14o"; return 1; }
}
check_new_files() {
  local index blob
  for ((index=0; index<${#FILES[@]}; index++)); do
    blob=$(git -C "$W" show "${REV}:${FILES[index]}" | sha256sum | cut -d' ' -f1)
    [ "$blob" = "${NEW_SHAS[index]}" ] || {
      log "ABORT: ${FILES[index]} at ${REV} has sha $blob, expected ${NEW_SHAS[index]}"; return 1; }
  done
  # The blob, not the installed file: preflight runs before the switch. grep -c reads the
  # whole stream, so pipefail cannot trip on an early exit.
  git -C "$W" show "${REV}:scripts/curriculum_queue.py" | grep -c -- 'superseded_by' >/dev/null
}
# Cheap: parse the blob and check the panel size; no training or game process.
check_controller() {
  "$PY" - "$W" <<'PY'
import ast, json, sys
ast.parse(open(sys.argv[1] + '/scripts/curriculum_queue.py').read())
PY
}
check_extra_panel() {
  if [ -f "$MODE14L" ] && grep -qx deckcells "$MODE14L"; then
    "$PY" - "$EXTRA" <<'PY'
import json, sys
panel = json.load(open(sys.argv[1]))
assert len(panel['tasks']) == 144
PY
  fi
}
# Desktop stop_qc: kill the session, no drain. Only *orphaned* search_play workers of
# this worktree are reaped (parent is init): a gate evaluation started by the dead
# controller still has its own parent and is left alone.
stop_qc() {
  local pid ppid
  if ! tmux has-session -t =qc 2>/dev/null; then return; fi
  tmux kill-session -t =qc
  for ((attempt=0; attempt<60; attempt++)); do
    if ! tmux has-session -t =qc 2>/dev/null; then break; fi
    sleep 2
  done
  for ((attempt=0; attempt<30; attempt++)); do
    if ! pgrep -f "$W/scripts/search_play.py" >/dev/null 2>&1; then return; fi
    for pid in $(pgrep -f "$W/scripts/search_play.py" || true); do
      ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
      if [ "${ppid:-0}" = "1" ]; then kill -TERM "$pid" 2>/dev/null || true; fi
    done
    sleep 2
  done
}
restore_files() {
  local index
  if [ "$SAVED" = 1 ]; then
    for ((index=0; index<${#FILES[@]}; index++)); do
      if [ -e "$O/${FILES[index]}.before" ]; then
        cp -p "$O/${FILES[index]}.before" "$W/${FILES[index]}"
      else
        rm -f "$W/${FILES[index]}"
      fi
    done
  fi
}
rollback() {
  trap - ERR
  log "$1; restoring the previous controller and restarting"
  stop_qc
  restore_files
  check_baseline || exit 1
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
trap 'on_error $LINENO' ERR

if [ "${1:-}" = "--rollback" ]; then
  # Manual rollback: newest backup, restart the previous controller.
  trap - ERR
  O=$(ls -dt "$LOG_DIR"/deploy14o_* 2>/dev/null | head -1)
  [ -n "${O:-}" ] && [ -d "$O" ] || { log 'ABORT: no deploy14o backup directory'; exit 1; }
  log "manual rollback from $O"
  stop_qc
  for file in "${FILES[@]}"; do
    if [ -e "$O/$file.before" ]; then cp -p "$O/$file.before" "$W/$file"; else rm -f "$W/$file"; fi
  done
  check_baseline || exit 1
  start_qc
  log 'rolled back to the previous controller; state, offsets and ledgers retained'
  exit 0
fi
check_baseline
check_new_files
[ -x "$PY" ]
if [ -f "$MODE15" ] && grep -qx v2sleep "$MODE15"; then
  [ -x "$EXE15" ]
else
  [ -x "$NEWEXE" ]
fi
check_extra_panel
log "preflight passed for $REV (coalesce training backlog; controller only)"
stop_qc; STOPPED=1
check_baseline
mkdir -p "$O/scripts"
for file in "${FILES[@]}"; do
  if [ -e "$W/$file" ]; then cp -p "$W/$file" "$O/$file.before"; fi
done
SAVED=1
for file in "${FILES[@]}"; do
  mkdir -p "$(dirname "$W/$file")"
  git -C "$W" show "${REV}:$file" > "$W/$file"
done
for ((index=0; index<${#FILES[@]}; index++)); do
  printf '%s  %s\n' "${NEW_SHAS[index]}" "$W/${FILES[index]}" | sha256sum -c -
done
check_controller
n0=$(grep -c ' rc=' "$LOG_DIR/curriculum.log" || true)
start_qc
log 'task141 backlog coalescing activated; controller, state, offsets and ledgers retained'
for ((attempt=0; attempt<120; attempt++)); do
  sleep 30
  if ! tmux has-session -t =qc 2>/dev/null; then break; fi
  n1=$(grep -c ' rc=' "$LOG_DIR/curriculum.log" || true)
  if [ "$n1" -gt "$n0" ]; then
    if tail -n 200 "$LOG_DIR/curriculum.log" | grep ' rc=' | tail -1 | grep -q ' rc=0 '; then
      log 'first new group rc=0; deployment complete'
      exit 0
    fi
    break
  fi
done
rollback 'first new group failed or timed out'
