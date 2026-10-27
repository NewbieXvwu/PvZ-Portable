#!/bin/bash
# task137 / deploy14l: add the 16 v5 random-deck curriculum cells.
#
# Only the main controller runs this script. It replaces the curriculum controller,
# the selection-panel builder and adds the random-deck generator + frozen registry,
# then restarts qc once between groups so the controller can load the extra panel.
# Background packing / training jobs are not drained.
set -Eeuo pipefail
R=/home/newbiexvwu/PvZ-search
W=$R/.claude/worktrees/cq
PACKAGE=$R/.claude/worktrees/obsup
PY=/home/newbiexvwu/.venvs/ml/bin/python3
REV=9aa61709b82c7cf50d7049bdaed74b8a0e9448c4
# Production controller has not changed since deploy14j (admitfix cf94a8cb).
CONTROLLER_SHA=72e5732db6755798a1ac2206a597e423b6e5a7507170e04df2cf5ea0d6e2a6dd
FILES=(scripts/curriculum_queue.py python/pvz_selection_refresh.py
       scripts/deck_rules.py scripts/make_random_deck_cells.py
       configs/random_deck_cells_v5.json)
NEW_SHAS=(a9d028fe2109f6039d6c373b0b47f8f757e8d4b020148337f962b40bcbb8886b
          616b5d5e6285dbb351b8ae3d9fc4d658f06bc5956bd64b2ed94e2048a1c44397
          70f8f662ba54529a16cbed3b381f57396ff4f67f3fc30d5ab250c5c81770da20
          ff0af1f053f2d6ee26490ba5222a40e3d2b187d0a8cc3b114d3e2c31799b0714
          2a3135a72fa11dc4e12193dbebf3f1208f7188a6568d3d965c4cd2cd00a6729f)
# Selection builder is either the deploy14k overlay (47c0...) or the admitfix base (aeff...).
SELECTION_SHAS=(47c0ae09fa0fc8a58c3a8746c8866a54f5b5faf7ff9924dd4cce5c9a9683b0a5
                aeff69c1636e3fa977aa2dfebda8459e5e16e6664f4b6e742eeb6224196fd15d)
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
O=$LOG_DIR/deploy14l_$(date +%Y%m%dT%H%M%S)
STOP_FLAG=$LOG_DIR/stop_curr
EXTRA=$W/artifacts/task137/training_panel.json
STOPPED=0; SAVED=0; OWN_STOP=0; EXTRA_ARG=""
start_qc() {
  if [ -f "$MODE15" ] && grep -qx v2sleep "$MODE15"; then
    tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $A15 --alternate-recipe $B15 --candidates-per-recipe 2 --training-parallelism 1 --selection-holdout-manifest $W/configs/student_selection_holdout.json --opening-manifest $OPENING --executable $EXE15 --promotion-panel $PANEL5 --gate-version 5 $EXTRA_ARG 2>&1 | tee -a ~/pvz-logs/qc.out"
  elif [ -f "$MODE14I" ] && grep -qx gatev5 "$MODE14I"; then
    tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $W/recipes/dw4i2.json --alternate-recipe $W/recipes/dw4i2_rebalanced_p05.json --candidates-per-recipe 2 --training-parallelism 1 --selection-holdout-manifest $W/configs/student_selection_holdout.json --opening-manifest $OPENING --executable $NEWEXE --promotion-panel $PANEL5 --gate-version 5 $EXTRA_ARG 2>&1 | tee -a ~/pvz-logs/qc.out"
  elif [ -f "$MODE14H" ] && grep -qx rebalance "$MODE14H"; then
    tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $W/recipes/dw4i2.json --alternate-recipe $W/recipes/dw4i2_rebalanced_p05.json --candidates-per-recipe 2 --training-parallelism 1 --selection-holdout-manifest $W/configs/student_selection_holdout.json --opening-manifest $OPENING --executable $NEWEXE --promotion-panel $PANEL3 $EXTRA_ARG 2>&1 | tee -a ~/pvz-logs/qc.out"
  else
    tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $W/recipes/dw4i2.json --alternate-recipe $W/recipes/dw4i2_rebalanced.json --opening-manifest $OPENING --executable $NEWEXE --promotion-panel $PANEL3 $EXTRA_ARG 2>&1 | tee -a ~/pvz-logs/qc.out"
  fi
}
log() { printf '%s deploy14l: %s\n' "$(date -Iseconds)" "$*" >> "$L"; }
cleanup() { if [ "$OWN_STOP" = 1 ]; then rm -f "$STOP_FLAG"; fi; }
check_baseline() {
  printf '%s  %s\n' "$CONTROLLER_SHA" "$W/scripts/curriculum_queue.py" | sha256sum -c -
  local current sha accepted=0
  current=$(sha256sum "$W/python/pvz_selection_refresh.py" | cut -d' ' -f1)
  for sha in "${SELECTION_SHAS[@]}"; do
    if [ "$current" = "$sha" ]; then accepted=1; fi
  done
  [ "$accepted" = 1 ] || { log "ABORT: selection builder sha $current is neither deploy14k nor admitfix"; return 1; }
}
check_new_files() {
  local index
  for ((index=0; index<${#FILES[@]}; index++)); do
    git -C "$W" show "$REV:${FILES[index]}" | sha256sum | grep -q "^${NEW_SHAS[index]} "
  done
}
check_extra_panel() {
  "$PY" "$W/scripts/make_random_deck_cells.py" --check
  "$PY" "$W/scripts/make_random_deck_cells.py" --out "$W/artifacts/task137"
  "$PY" - "$EXTRA" <<'PY'
import json, sys
panel = json.load(open(sys.argv[1]))
tasks = panel['tasks']
cells = {task['task_id'].split('_m')[0] for task in tasks}
assert len(tasks) == 144 and len(cells) == 16, (len(tasks), len(cells))
assert min(task['zombie_count_multiplier'] for task in tasks) == 0.5
seeds = [seed for task in tasks for seed in task['seeds']]
assert len(seeds) == len(set(seeds)) == 294912
PY
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
  rm -f "$MODE14L"; EXTRA_ARG=""
}
rollback() {
  trap - ERR
  log "$1; restoring the previous controller and restarting without --extra-manifest"
  stop_qc || exit 1
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
trap cleanup EXIT
trap 'on_error $LINENO' ERR

[ ! -e "$STOP_FLAG" ] || { log 'ABORT: another stop request exists'; exit 1; }
check_baseline
check_new_files
[ -x "$PY" ]
if [ -f "$MODE15" ] && grep -qx v2sleep "$MODE15"; then
  [ -x "$EXE15" ]
else
  [ -x "$NEWEXE" ]
fi
check_extra_panel
log "preflight passed for $REV controller, selection builder and 144-task extra panel"
stop_qc; STOPPED=1
check_baseline
mkdir -p "$O/scripts" "$O/python" "$O/configs"
for file in "${FILES[@]}"; do
  if [ -e "$W/$file" ]; then cp -p "$W/$file" "$O/$file.before"; fi
done
SAVED=1
for file in "${FILES[@]}"; do
  mkdir -p "$(dirname "$W/$file")"
  git -C "$W" show "$REV:$file" > "$W/$file"
done
for ((index=0; index<${#FILES[@]}; index++)); do
  printf '%s  %s\n' "${NEW_SHAS[index]}" "$W/${FILES[index]}" | sha256sum -c -
done
"$PY" -c 'import ast, sys; [ast.parse(open(path).read()) for path in sys.argv[1:]]' \
  "$W/scripts/curriculum_queue.py" "$W/python/pvz_selection_refresh.py" \
  "$W/scripts/deck_rules.py" "$W/scripts/make_random_deck_cells.py"
check_extra_panel
printf 'deckcells\n' > "$MODE14L"
EXTRA_ARG="--extra-manifest $EXTRA"
n0=$(grep -c ' rc=' "$LOG_DIR/curriculum.log" || true)
start_qc
log 'task137 random-deck cells activated; controller, state and seed offsets retained'
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
