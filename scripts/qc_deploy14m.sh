#!/bin/bash
# task138 / deploy14m: let the collection pool widen inside a group.
#
# Only the main controller runs this script. It replaces search_play.py and the
# curriculum controller, adds growable_pool.py, then restarts qc. stop_qc here is
# the desktop version: kill the session, do not drain a group. Background packing
# and training jobs are not touched.
set -Eeuo pipefail
R=/home/newbiexvwu/PvZ-search
W=$R/.claude/worktrees/cq
PACKAGE=$R/.claude/worktrees/obsup
PY=/home/newbiexvwu/.venvs/ml/bin/python3
REV=e66ee8d6dda4dacde3d2a9d079e73bf5645c3e76
FILES=(scripts/search_play.py scripts/curriculum_queue.py scripts/growable_pool.py)
NEW_SHAS=(4f4ba54165bb6aeeb6fc5e6eeff22bd588e2b9b643fb64798e4130d13382c9f8
          f98df24cbb5d8af69a237df6465da5c2fba77e55b8b231907544fdb7d68c76e9
          6b9da78bcfd0eb07e8d2c22bbc554f798490d15d443d02e5b363d970af6b99f9)
# What production may be before the switch: deploy14l (task137) or this deploy.
BASELINE_SEARCH_SHAS=(6f9103447dd4e5d066c6b6b26ff516b7165c3d8a89f4c67ea3cf5f63540acf01
                      4f4ba54165bb6aeeb6fc5e6eeff22bd588e2b9b643fb64798e4130d13382c9f8)
BASELINE_CONTROLLER_SHAS=(a9d028fe2109f6039d6c373b0b47f8f757e8d4b020148337f962b40bcbb8886b
                          f98df24cbb5d8af69a237df6465da5c2fba77e55b8b231907544fdb7d68c76e9)
NEWEXE=$W/build6/pvz-portable
EXE15=$PACKAGE/build7/pvz-portable
MODE14M=/home/newbiexvwu/pvz-logs/qc_deploy14m.mode
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
O=$LOG_DIR/deploy14m_$(date +%Y%m%dT%H%M%S)
EXTRA=$W/artifacts/task137/training_panel.json
STOPPED=0; SAVED=0
# The extra manifest follows the deploy14l mode file, so a plain restart keeps it.
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
log() { printf '%s deploy14m: %s\n' "$(date -Iseconds)" "$*" >> "$L"; }
check_baseline() {
  local current sha accepted
  accepted=0
  current=$(sha256sum "$W/scripts/search_play.py" | cut -d' ' -f1)
  for sha in "${BASELINE_SEARCH_SHAS[@]}"; do
    if [ "$current" = "$sha" ]; then accepted=1; fi
  done
  [ "$accepted" = 1 ] || { log "ABORT: search_play sha $current is neither deploy14l nor deploy14m"; return 1; }
  accepted=0
  current=$(sha256sum "$W/scripts/curriculum_queue.py" | cut -d' ' -f1)
  for sha in "${BASELINE_CONTROLLER_SHAS[@]}"; do
    if [ "$current" = "$sha" ]; then accepted=1; fi
  done
  [ "$accepted" = 1 ] || { log "ABORT: controller sha $current is neither deploy14l nor deploy14m"; return 1; }
}
check_new_files() {
  local index
  for ((index=0; index<${#FILES[@]}; index++)); do
    git -C "$W" show "${REV}:${FILES[index]}" | sha256sum | grep -q "^${NEW_SHAS[index]} "
  done
}
# Cheap: the registry check is pure Python, and the pool test uses trivial jobs.
check_pool() {
  "$PY" "$W/scripts/search_play.py" --help | grep -q -- '--initial-workers'
  "$PY" "$W/scripts/curriculum_queue.py" --help | grep -q -- '--extra-manifest'
  ( cd "$W" && "$PY" python/test_growable_pool.py ) >/dev/null
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
# Desktop stop_qc: kill the session, no drain. Orphaned search_play children of this
# worktree are reaped so the next group does not share the machine with them.
stop_qc() {
  if ! tmux has-session -t =qc 2>/dev/null; then return; fi
  tmux kill-session -t =qc
  for ((attempt=0; attempt<60; attempt++)); do
    if ! tmux has-session -t =qc 2>/dev/null; then break; fi
    sleep 2
  done
  for ((attempt=0; attempt<30; attempt++)); do
    if ! pgrep -f "$W/scripts/search_play.py" >/dev/null 2>&1; then break; fi
    sleep 2
  done
  pkill -f "$W/scripts/search_play.py" 2>/dev/null || true
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
  rm -f "$MODE14M"
}
rollback() {
  trap - ERR
  log "$1; restoring the previous search_play/controller and restarting"
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
  # Manual rollback: newest backup, drop the mode file, restart without the new pool.
  trap - ERR
  O=$(ls -dt "$LOG_DIR"/deploy14m_* 2>/dev/null | head -1)
  [ -n "${O:-}" ] && [ -d "$O" ] || { log 'ABORT: no deploy14m backup directory'; exit 1; }
  log "manual rollback from $O"
  stop_qc
  for file in "${FILES[@]}"; do
    if [ -e "$O/$file.before" ]; then cp -p "$O/$file.before" "$W/$file"; else rm -f "$W/$file"; fi
  done
  rm -f "$MODE14M"
  check_baseline || exit 1
  start_qc
  log 'rolled back to the previous search_play/controller; state, offsets and ledgers retained'
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
check_pool
log "preflight passed for $REV (pool growth; search_play + controller + growable_pool)"
stop_qc; STOPPED=1
mkdir -p "$O/scripts"
for file in "${FILES[@]}"; do
  if [ -e "$W/$file" ]; then cp -p "$W/$file" "$O/$file.before"; fi
done
SAVED=1
for file in "${FILES[@]}"; do
  git -C "$W" show "${REV}:$file" > "$W/$file"
done
for ((index=0; index<${#FILES[@]}; index++)); do
  printf '%s  %s\n' "${NEW_SHAS[index]}" "$W/${FILES[index]}" | sha256sum -c -
done
"$PY" -c 'import ast, sys; [ast.parse(open(path).read()) for path in sys.argv[1:]]' \
  "$W/scripts/search_play.py" "$W/scripts/curriculum_queue.py" "$W/scripts/growable_pool.py"
check_pool
printf 'groupwidth\n' > "$MODE14M"
n0=$(grep -c ' rc=' "$LOG_DIR/curriculum.log" || true)
start_qc
log 'task138 pool growth activated; state, offsets and ledgers retained'
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
