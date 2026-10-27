#!/bin/bash
# task131: prepared only. Do not execute as part of delivery.
# Prior mode: 14i=gatev5 -> i; otherwise 14h=rebalance -> h; otherwise g.
# Inactive/rollback start_qc is copied verbatim from the accepted deploy14i.
# build7 lives in the obsup package; build6, old recipes and 14i waiter stay intact.
set -Eeuo pipefail
R=/home/newbiexvwu/PvZ-search
W=$R/.claude/worktrees/cq
PACKAGE=$R/.claude/worktrees/obsup
PY=/home/newbiexvwu/.venvs/ml/bin/python3
REV=f928321899bc56868ea4d00efcf6e4f64714a18f
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
O=$LOG_DIR/deploy15_$(date +%Y%m%dT%H%M%S)
STOP_FLAG=$LOG_DIR/stop_curr
STOPPED=0; SAVED=0; OWN_STOP=0
CODE_FILES=(MEMORY/docs/adventure-scoreboard-r11-32.json MEMORY/docs/adventure-scoreboard.json MEMORY/docs/task115-teacher-arm-weak-levels.md configs/adventure_scoreboard.json configs/gate_v5_panel.json configs/student_selection_holdout.json python/pvz_env.py python/pvz_frame_store.py python/pvz_promotion.py python/pvz_recipe.py python/pvz_rollout_policy.py python/pvz_selection_refresh.py python/pvz_student_observation.py python/pvz_training_rows.py python/pvz_value_net.py recipes/dw4i2_rebalanced.json recipes/dw4i2_rebalanced_p05.json recipes/dw4i2_rebalanced_p05_v2sleep.json recipes/dw4i2_v2sleep.json scripts/curriculum_queue.py scripts/make_gate_v5_panel.py scripts/make_rebalance_panel.py scripts/pack_training_rows.py scripts/promotion_gate.py scripts/render_episode.py scripts/search_play.py scripts/select_student.py scripts/train_rollout_policy.py scripts/train_value_net.py src/EnvironmentRollout.cpp src/EnvironmentRollout.h src/EnvironmentValueNet.cpp src/EnvironmentValueNet.h src/Lawn/Zombie.cpp src/Lawn/Zombie.h src/LawnApp.cpp src/LawnApp.h src/main.cpp)
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
service() {
  if [ -f "$MODE14I" ] && grep -qx gatev5 "$MODE14I"; then printf i;
  elif [ -f "$MODE14H" ] && grep -qx rebalance "$MODE14H"; then printf h;
  else printf g; fi
}
log() { printf '%s deploy15: %s\n' "$(date -Iseconds)" "$*" >> "$L"; }
code_state() {
  "$PY" "$PACKAGE/scripts/qc_deploy15_state.py" "$1" --root "$W" --output "$O" \
    --package "$PACKAGE" --baselines "$PACKAGE/configs/deploy15_baselines.json" \
    --service "$PREVIOUS" --revision "$REV" --state "$LOG_DIR/curriculum_state.json"
}
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
  log "$1; restoring deploy14$PREVIOUS code and arguments"
  stop_qc || exit 1
  if [ "$SAVED" = 1 ]; then code_state restore || exit 1; fi
  code_state restored || exit 1
  rm -f "$MODE15"
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

# --- preflight: only small code/recipe hashes and two engine files ---
[ ! -e "$STOP_FLAG" ] || { log 'ABORT: another stop request exists'; exit 1; }
if [ -f "$MODE15" ] && grep -qx v2sleep "$MODE15"; then
  log 'ABORT: deploy15 is already active'; exit 1
fi
PREVIOUS=$(service)
code_state preflight
code_state boundary
mkdir -p "$O"
log "preflight passed for deploy14$PREVIOUS; no corpus replay or pack warmup"
stop_qc; STOPPED=1
code_state boundary
code_state save; SAVED=1
git -C "$W" restore --source "$REV" --worktree -- "${CODE_FILES[@]}"
code_state installed
# No pack preparation here. The ordinary background pipeline builds missing v2sleep
# label segments once, with two builders; collection can continue during that work.
timeout 300 bash -c 'ulimit -v 8000000; exec env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$@"' \
  task131 "$PY" "$W/scripts/curriculum_queue.py" --dry-run --rounds 100 --workers 16 \
  --round-games 768 --games-per-group 48 --manifest "$W/artifacts/stage4/multiplier_panel.json" \
  --recipe "$A15" --alternate-recipe "$B15" --candidates-per-recipe 2 --training-parallelism 1 \
  --selection-holdout-manifest "$W/configs/student_selection_holdout.json" \
  --opening-manifest "$OPENING" --executable "$EXE15" --promotion-panel "$PANEL5" --gate-version 5 \
  --log "$O/dryrun.log" > "$O/dryrun.out" 2>&1 || rollback 'dry-run failed'
grep -q '"adventure_tasks": 38' "$O/dryrun.out" || rollback 'selection preview missing'
code_state boundary
printf 'v2sleep\n' > "$MODE15.building"; mv "$MODE15.building" "$MODE15"
n0=$(grep -c ' rc=' "$LOG_DIR/curriculum.log" || true)
start_qc
log 'build7 + v2sleep activated; old incumbent stays live until a trained candidate passes gate v5'
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
