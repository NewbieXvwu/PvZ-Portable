#!/bin/bash
# task143 / deploy15: switch the curriculum to the v2sleep observation and the
# build7 engine on top of deploy14p. Prepared only; the main controller runs it.
#
# Baseline: the six production files as deploy14p leaves them (controller
# 5ef3377f, selection 616b5d5e, search_play 4f4ba541). deploy14p must already be
# active (qc_deploy14p.mode=deckcells) — deploy15 keeps its --extra-manifest.
# Replaced: the six observation modules, the row packer, the trainer, and the
# two v2sleep recipes. The controller is unchanged (5ef3377f in both branches):
# production trains every round from scratch, so no warm-start hook is needed.
# build7 already exists in the obsup worktree (task131); this script only pins
# its hash, it does not rebuild the engine.
set -Eeuo pipefail
R=/home/newbiexvwu/PvZ-search
W=$R/.claude/worktrees/cq
PY=/home/newbiexvwu/.venvs/ml/bin/python3
REV=fc817121cb0cb834233e51ec1848acb2d62c5c43
FILES=(python/pvz_env.py python/pvz_frame_store.py python/pvz_recipe.py
       python/pvz_rollout_policy.py python/pvz_student_observation.py
       python/pvz_value_net.py scripts/pack_training_rows.py
       scripts/train_rollout_policy.py
       recipes/dw4i2_v2sleep.json recipes/dw4i2_rebalanced_p05_v2sleep.json)
NEW_SHAS=(a677a54723fce04b1b2a78fea02fbbf736c50acc76b217fe4e3e20cdb286def6
          ef383dbf7c56054be81e7af129d0b993b88a71949922574c6769bc01ddaf25a5
          ed99c2a16efdd5410fac554a618d8238c188f7f92d2487d534c0e00166ddc23c
          55c519b3ead717288aeba2a84bd053d6163fc2abcc7f3af6bb105dbf49bc33e3
          55f3984afee7df1c786a0dd8feaf57b8b83e77eb6a09e8263f2c0db69bf0b620
          f5420b0468f15ce7f7f402587f5b1c5e705cbb54b7ba82a727cecc1878a3a47c
          2faa5fc53355e9f317e9672bcde3b6a2f3eb1a3c8241c2e3028e23d58967b9b5
          d8d6295b3b9c039544c0f6fa6be5eeeb47f966b2da20d00a9eff0af1f692a002
          c0512b2bc290520052cfc20929930cac1bfdf8382602b747655dc6370ed6f1a6
          df4da572c068c4296bd452914264b8e29de57e78797efd46cd3be5b0a2da4200)
# Post-deploy14p production baseline. deploy15 does not touch these files; the
# check only proves the previous deployment is in place before switching.
PROD_FILES=(scripts/curriculum_queue.py scripts/select_student.py
            python/pvz_selection_refresh.py python/pvz_promotion.py
            scripts/search_play.py python/pvz_training_rows.py)
PROD_SHAS=(5ef3377f2483ec29465c061ae4c30acc66a99bc53902fd21879949c1fd018bb2
           34b272b8660882e5de100c7442c86eba149e55659e4509c540e8062461f2a041
           616b5d5e6285dbb351b8ae3d9fc4d658f06bc5956bd64b2ed94e2048a1c44397
           c16ba93079afc63294059e4c9051a34872794bddd8f5b49a8e949096feafc6b7
           4f4ba54165bb6aeeb6fc5e6eeff22bd588e2b9b643fb64798e4130d13382c9f8
           1878b1c2000630a44bd81c4e6093a97e24ae9c9b9dc22b8bd7d27ed06921bcee)
NEWEXE=$W/build6/pvz-portable
EXE15=$R/.claude/worktrees/obsup/build7/pvz-portable
EXE15_SHA=c35633bc71b10890d9fa2a7025ac45f88d4ef759d945f8b84c7572a3c141dec8
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
O=$LOG_DIR/deploy15_$(date +%Y%m%dT%H%M%S)
STOP_FLAG=$LOG_DIR/stop_curr
EXTRA=$W/artifacts/task137/training_panel.json
STOPPED=0; SAVED=0
log() { printf '%s deploy15: %s\n' "$(date -Iseconds)" "$*" >> "$L"; }
start_qc() {
  if [ -f "$MODE15" ] && grep -qx v2sleep "$MODE15"; then
    tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $A15 --alternate-recipe $B15 --candidates-per-recipe 2 --training-parallelism 1 --selection-holdout-manifest $W/configs/student_selection_holdout.json --opening-manifest $OPENING --executable $EXE15 --promotion-panel $PANEL5 --gate-version 5 --extra-manifest $EXTRA 2>&1 | tee -a ~/pvz-logs/qc.out"
  elif [ -f "$MODE14I" ] && grep -qx gatev5 "$MODE14I"; then
    tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $W/recipes/dw4i2.json --alternate-recipe $W/recipes/dw4i2_rebalanced_p05.json --candidates-per-recipe 2 --training-parallelism 1 --selection-holdout-manifest $W/configs/student_selection_holdout.json --opening-manifest $OPENING --executable $NEWEXE --promotion-panel $PANEL5 --gate-version 5 --extra-manifest $EXTRA 2>&1 | tee -a ~/pvz-logs/qc.out"
  elif [ -f "$MODE14H" ] && grep -qx rebalance "$MODE14H"; then
    tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $W/recipes/dw4i2.json --alternate-recipe $W/recipes/dw4i2_rebalanced_p05.json --candidates-per-recipe 2 --training-parallelism 1 --selection-holdout-manifest $W/configs/student_selection_holdout.json --opening-manifest $OPENING --executable $NEWEXE --promotion-panel $PANEL3 --extra-manifest $EXTRA 2>&1 | tee -a ~/pvz-logs/qc.out"
  else
    tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $W/recipes/dw4i2.json --alternate-recipe $W/recipes/dw4i2_rebalanced.json --opening-manifest $OPENING --executable $NEWEXE --promotion-panel $PANEL3 --extra-manifest $EXTRA 2>&1 | tee -a ~/pvz-logs/qc.out"
  fi
}
check_baseline() {
  local index current
  for ((index=0; index<${#PROD_FILES[@]}; index++)); do
    current=$(sha256sum "$W/${PROD_FILES[index]}" | cut -d' ' -f1)
    [ "$current" = "${PROD_SHAS[index]}" ] || {
      log "ABORT: ${PROD_FILES[index]} sha $current is not the post-deploy14p baseline"; return 1; }
  done
}
check_revision() {
  git -C "$W" cat-file -e "$REV^{commit}" 2>/dev/null || {
    log "ABORT: commit $REV is not in $W; fetch origin obsup2 first"; return 1; }
}
check_new_files() {
  local index
  for ((index=0; index<${#FILES[@]}; index++)); do
    git -C "$W" show "$REV:${FILES[index]}" | sha256sum | grep -q "^${NEW_SHAS[index]} "
  done
}
check_recipes() {
  "$PY" - "$A15" "$B15" <<'PY'
import json, sys
for path in sys.argv[1:]:
    recipe = json.load(open(path))
    assert recipe['obs'] == 'v2sleep', (path, recipe.get('obs'))
    assert recipe['init'] is None, (path, recipe.get('init'))
    assert int(recipe['precompute_workers']) == 2, (path, recipe.get('precompute_workers'))
PY
}
check_extra_panel() {
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
  # The controller drains queued training on the stop flag (hours of idle
  # collection), so kill the session and its running processes directly.
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
  rm -f "$MODE15"
}
rollback() {
  trap - ERR
  log "$1; restoring the deploy14p code and restarting without v2sleep"
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

# --- preflight: only code hashes, recipe fields, two engines and the panel ---
[ ! -e "$STOP_FLAG" ] || { log 'ABORT: another stop request exists'; exit 1; }
if [ "${1:-}" = "--rollback" ]; then
  O=$(ls -dt "$LOG_DIR"/deploy15_* 2>/dev/null | head -1)
  [ -n "${O:-}" ] && [ -d "$O" ] || { log 'ABORT: no deploy15 backup directory'; exit 1; }
  [ -e "$O/backed_up" ] || { log "ABORT: $O has no complete backup marker"; exit 1; }
  log "manual rollback from $O"
  stop_qc; STOPPED=1
  SAVED=1
  restore_files
  check_baseline || exit 1
  start_qc
  log 'rolled back to deploy14p; state, offsets and ledgers retained'
  exit 0
fi
check_baseline
[ -f "$MODE14P" ] && grep -qx deckcells "$MODE14P" || {
  log 'ABORT: deploy14p (deckcells) is not active; run deploy14p first'; exit 1; }
[ -f "$EXTRA" ] || { log 'ABORT: task137 training panel missing'; exit 1; }
check_revision
check_new_files
[ -x "$PY" ] && [ -x "$NEWEXE" ]
[ -x "$EXE15" ] || { log "ABORT: build7 missing at $EXE15"; exit 1; }
[ "$(sha256sum "$EXE15" | cut -d' ' -f1)" = "$EXE15_SHA" ] || {
  log 'ABORT: build7 hash differs from the task131 binary'; exit 1; }
check_recipes
check_extra_panel
"$PY" -c 'import ast, sys; [ast.parse(open(path).read()) for path in sys.argv[1:]]' \
  "$W/python/pvz_env.py" "$W/python/pvz_frame_store.py" "$W/python/pvz_recipe.py" \
  "$W/python/pvz_rollout_policy.py" "$W/python/pvz_student_observation.py" \
  "$W/python/pvz_value_net.py" "$W/scripts/pack_training_rows.py" \
  "$W/scripts/train_rollout_policy.py"
log "preflight passed for $REV; build7 $EXE15_SHA; no dry-run or corpus replay"
stop_qc; STOPPED=1
check_baseline
mkdir -p "$O"
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
  git -C "$W" show "$REV:$file" > "$W/$file"
done
for ((index=0; index<${#FILES[@]}; index++)); do
  printf '%s  %s\n' "${NEW_SHAS[index]}" "$W/${FILES[index]}" | sha256sum -c -
done
check_recipes
printf 'v2sleep\n' > "$MODE15.building"; mv "$MODE15.building" "$MODE15"
n0=$(grep -c ' rc=' "$LOG_DIR/curriculum.log" || true)
start_qc
log 'build7 + v2sleep activated; the first round rebuilds all label segments in the background'
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
