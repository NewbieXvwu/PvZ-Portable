#!/bin/bash
set -Eeuo pipefail
R=/home/newbiexvwu/PvZ-search
W=$R/.claude/worktrees/cq
PACKAGE=$R/.claude/worktrees/obsup
PY=/home/newbiexvwu/.venvs/ml/bin/python3
REV=6dae2c2e0ddaa082179f30d5a961cc4877eb3a8b
CONTROLLER_SHA=72e5732db6755798a1ac2206a597e423b6e5a7507170e04df2cf5ea0d6e2a6dd
FILES=(scripts/select_student.py python/pvz_promotion.py python/pvz_selection_refresh.py)
BASE_SHAS=(de8db43420aad3a57d667e9553279fcb46f7b45290b49073bbe334cba8718e63
           7bdb872d55e85ca33a786b528496ce8564e643012d71c053dce9ae361c2ec906
           aeff69c1636e3fa977aa2dfebda8459e5e16e6664f4b6e742eeb6224196fd15d)
NEW_SHAS=(34b272b8660882e5de100c7442c86eba149e55659e4509c540e8062461f2a041
          c16ba93079afc63294059e4c9051a34872794bddd8f5b49a8e949096feafc6b7
          47c0ae09fa0fc8a58c3a8746c8866a54f5b5faf7ff9924dd4cce5c9a9683b0a5)
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
O=$LOG_DIR/deploy14k_$(date +%Y%m%dT%H%M%S)
STOP_FLAG=$LOG_DIR/stop_curr
CODE=$W/scripts/curriculum_queue.py
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
log() { printf '%s deploy14k: %s\n' "$(date -Iseconds)" "$*" >> "$L"; }
cleanup() { if [ "$OWN_STOP" = 1 ]; then rm -f "$STOP_FLAG"; fi; }
check_baseline() {
  printf '%s  %s\n' "$CONTROLLER_SHA" "$CODE" | sha256sum -c -
  local index
  for ((index=0; index<${#FILES[@]}; index++)); do
    printf '%s  %s\n' "${BASE_SHAS[index]}" "$W/${FILES[index]}" | sha256sum -c -
  done
}
stop_qc() {
  if ! tmux has-session -t =qc 2>/dev/null; then return; fi
  touch "$STOP_FLAG"; OWN_STOP=1
  for ((attempt=0; attempt<720; attempt++)); do
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
  log "$1; restoring selection files with unchanged start_qc arguments"
  stop_qc || exit 1
  if [ "$SAVED" = 1 ]; then
    local index
    for ((index=0; index<${#FILES[@]}; index++)); do
      printf '%s  %s\n' "${BASE_SHAS[index]}" "$O/${FILES[index]}.before" | sha256sum -c - || exit 1
    done
    for file in "${FILES[@]}"; do cp -p "$O/$file.before" "$W/$file" || exit 1; done
  fi
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
trap cleanup EXIT
trap 'on_error $LINENO' ERR

[ ! -e "$STOP_FLAG" ] || { log 'ABORT: another stop request exists'; exit 1; }
check_baseline
for ((index=0; index<${#FILES[@]}; index++)); do
  git -C "$W" show "$REV:${FILES[index]}" | sha256sum | grep -q "^${NEW_SHAS[index]} "
done
[ -x "$PY" ]
if [ -f "$MODE15" ] && grep -qx v2sleep "$MODE15"; then
  [ -x "$EXE15" ]
else
  [ -x "$NEWEXE" ]
fi
mkdir -p "$O/scripts" "$O/python"
log "preflight passed for cf94a8cb controller and selection sources; overlay $REV"
stop_qc; STOPPED=1
check_baseline
for file in "${FILES[@]}"; do cp -p "$W/$file" "$O/$file.before"; done
SAVED=1
git -C "$W" restore --source "$REV" --worktree -- "${FILES[@]}"
for ((index=0; index<${#FILES[@]}; index++)); do
  printf '%s  %s\n' "${NEW_SHAS[index]}" "$W/${FILES[index]}" | sha256sum -c -
  "$PY" -c 'import ast, sys; ast.parse(open(sys.argv[1]).read())' "$W/${FILES[index]}"
done
n0=$(grep -c ' rc=' "$LOG_DIR/curriculum.log" || true)
start_qc
log 'paired selection and day 16-game budget activated; controller, modes and state retained'
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
rollback 'first new group failed or timed out'
