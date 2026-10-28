#!/bin/bash
# task142 / deploy14p: ship task137+task138+task140+task141 as one deployment.
#
# Run only on the desktop, only from the main controller. Replaces the controller,
# the selection builder, search_play and adds growable_pool / deck_rules /
# make_random_deck_cells / the v5 registry, then restarts qc once between groups.
# Background packing / training jobs are not drained.
set -Eeuo pipefail
R=/home/newbiexvwu/PvZ-search
W=$R/.claude/worktrees/cq
PACKAGE=$R/.claude/worktrees/obsup
PY=/home/newbiexvwu/.venvs/ml/bin/python3
REV=4beab441b81364a6c4681252900382e8e1d19b6a
FILES=(scripts/curriculum_queue.py python/pvz_selection_refresh.py
       scripts/search_play.py scripts/growable_pool.py scripts/deck_rules.py
       scripts/make_random_deck_cells.py configs/random_deck_cells_v5.json)
NEW_SHAS=(5ef3377f2483ec29465c061ae4c30acc66a99bc53902fd21879949c1fd018bb2
          616b5d5e6285dbb351b8ae3d9fc4d658f06bc5956bd64b2ed94e2048a1c44397
          4f4ba54165bb6aeeb6fc5e6eeff22bd588e2b9b643fb64798e4130d13382c9f8
          933487655638ecdc2045fa3d251d2fb7b17d0b1bda50e1a9bc4fd3de0f3078d3
          70f8f662ba54529a16cbed3b381f57396ff4f67f3fc30d5ab250c5c81770da20
          ff0af1f053f2d6ee26490ba5222a40e3d2b187d0a8cc3b114d3e2c31799b0714
          2a3135a72fa11dc4e12193dbebf3f1208f7188a6568d3d965c4cd2cd00a6729f)
# Production baseline from /Users/newbiexvwu/pvz-logs/live_prod_shas.txt (controller
# admitfix cf94a8cb, selection trio task136 6dae2c2e). The optional second entry is
# this deploy's own sha, so a re-run after activation (and --rollback) passes.
PROD_FILES=(scripts/curriculum_queue.py scripts/select_student.py
            python/pvz_selection_refresh.py python/pvz_promotion.py
            scripts/search_play.py python/pvz_training_rows.py)
PROD_SHAS=(72e5732db6755798a1ac2206a597e423b6e5a7507170e04df2cf5ea0d6e2a6dd
           34b272b8660882e5de100c7442c86eba149e55659e4509c540e8062461f2a041
           47c0ae09fa0fc8a58c3a8746c8866a54f5b5faf7ff9924dd4cce5c9a9683b0a5
           c16ba93079afc63294059e4c9051a34872794bddd8f5b49a8e949096feafc6b7
           6f9103447dd4e5d066c6b6b26ff516b7165c3d8a89f4c67ea3cf5f63540acf01
           1878b1c2000630a44bd81c4e6093a97e24ae9c9b9dc22b8bd7d27ed06921bcee)
DEPLOY_SHAS=("" "" 616b5d5e6285dbb351b8ae3d9fc4d658f06bc5956bd64b2ed94e2048a1c44397
             "" 4f4ba54165bb6aeeb6fc5e6eeff22bd588e2b9b643fb64798e4130d13382c9f8 "")
NEWEXE=$W/build6/pvz-portable
EXE15=$PACKAGE/build7/pvz-portable
MODE14P=/home/newbiexvwu/pvz-logs/qc_deploy14p.mode
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
O=$LOG_DIR/deploy14p_$(date +%Y%m%dT%H%M%S)
STOP_FLAG=$LOG_DIR/stop_curr
EXTRA=$W/artifacts/task137/training_panel.json
STOPPED=0; SAVED=0
log() { printf '%s deploy14p: %s\n' "$(date -Iseconds)" "$*" >> "$L"; }
start_qc() {
  local extra=""
  if [ -f "$MODE14P" ] && grep -qx deckcells "$MODE14P"; then extra="--extra-manifest $EXTRA"; fi
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
check_baseline() {
  local index current accepted
  for ((index=0; index<${#PROD_FILES[@]}; index++)); do
    current=$(sha256sum "$W/${PROD_FILES[index]}" | cut -d' ' -f1)
    accepted=0
    if [ "$current" = "${PROD_SHAS[index]}" ]; then accepted=1; fi
    if [ -n "${DEPLOY_SHAS[index]}" ] && [ "$current" = "${DEPLOY_SHAS[index]}" ]; then accepted=1; fi
    [ "$accepted" = 1 ] || { log "ABORT: ${PROD_FILES[index]} sha $current is neither production nor deploy14p"; return 1; }
  done
}
check_revision() {
  git -C "$W" cat-file -e "$REV^{commit}" 2>/dev/null || {
    log "ABORT: commit $REV is not in $W; fetch origin prodint first"; return 1; }
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
  # Main controller runs this right after a gate decision is logged. This controller
  # drains queued training on the stop flag (hours of idle collection), so kill the
  # session and its running processes directly instead of asking it to stop.
  if ! tmux has-session -t =qc 2>/dev/null; then return; fi
  [ -S /tmp/tmux-1000/default ] || { log 'ABORT: tmux socket missing'; return 1; }
  local w; w=$(pgrep -f 'worktrees/cq/scripts/(search_play|train_rollout_policy)' || true)
  tmux kill-session -t =qc; sleep 4
  for p in $w; do kill -TERM "$p" 2>/dev/null; done; sleep 5
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
  rm -f "$MODE14P"
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
trap 'on_error $LINENO' ERR

[ ! -e "$STOP_FLAG" ] || { log 'ABORT: another stop request exists'; exit 1; }
if [ "${1:-}" = "--rollback" ]; then
  # Manual rollback: newest backup, mode removed, start_qc without --extra-manifest.
  O=$(ls -dt "$LOG_DIR"/deploy14p_* 2>/dev/null | head -1)
  [ -n "${O:-}" ] && [ -d "$O" ] || { log 'ABORT: no deploy14p backup directory'; exit 1; }
  log "manual rollback from $O"
  stop_qc; STOPPED=1
  SAVED=1
  restore_files
  check_baseline || exit 1
  start_qc
  log 'rolled back to the previous controller; state, offsets and ledgers retained'
  exit 0
fi
check_baseline
check_revision
check_new_files
[ -x "$PY" ]
if [ -f "$MODE15" ] && grep -qx v2sleep "$MODE15"; then
  [ -x "$EXE15" ]
else
  [ -x "$NEWEXE" ]
fi
log "preflight passed for $REV controller, selection builder, search_play and extra panel"
stop_qc; STOPPED=1
check_baseline
mkdir -p "$O"
for file in "${FILES[@]}"; do
  if [ -e "$W/$file" ]; then
    mkdir -p "$(dirname "$O/$file")"
    cp -p "$W/$file" "$O/$file.before"
  fi
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
  "$W/scripts/search_play.py" "$W/scripts/growable_pool.py" \
  "$W/scripts/deck_rules.py" "$W/scripts/make_random_deck_cells.py"
check_extra_panel
printf 'deckcells\n' > "$MODE14P"
n0=$(grep -c ' rc=' "$LOG_DIR/curriculum.log" || true)
start_qc
log 'task137/138/140/141 activated; controller, state and seed offsets retained'
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
  log 'first new group still running after 60 min; left live, controller checks rc manually'; exit 0
fi
rollback 'first new group failed or qc exited'
