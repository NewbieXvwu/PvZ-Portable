#!/bin/bash
# task200: PREPARED ONLY. Main controller reviews and installs this package.
# Reuse the deploy20 engine, deploy22 data/import setup and deploy23 roof leaf route;
# only the live-cell planner changes. No build/replay/pack here.
set -Eeuo pipefail
R=$HOME/PvZ-search
W=$R/.claude/worktrees/cq
L=$R/.claude/worktrees/llmimport
PACKAGE=$R
PY=$HOME/.venvs/ml/bin/python3
REV=${PVZ_DEPLOY24_REV:-$(git -C "$PACKAGE" rev-parse origin/deploy24)}
BASE=27d2fec9
LOG_DIR=$HOME/pvz-logs
COLD_RATES=$LOG_DIR/teacher_rates.json
MODE23=$LOG_DIR/qc_deploy23.mode
ACTIVE23=$LOG_DIR/qc_deploy23.active
MODE24=$LOG_DIR/qc_deploy24.mode
ACTIVE24=$LOG_DIR/qc_deploy24.active
OPENING=$W/artifacts/task119/opening_panel.json
PANEL5=$W/configs/gate_v5_panel.json
EXTRA17=$W/artifacts/task165/training_panel.json
EXE20=$R/artifacts/production/deploy20/pvz-portable
VALUE5=$R/artifacts/value_net/v5/model.pt
VROOF=$R/artifacts/value_net/v5roof/model.pt
A16=$W/recipes/dw4i2_v2timer.json
B24=$W/recipes/dw4i2_v2timer_seed2cold.json
start_qc() {
  tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --min-eval-units 4 --python $PY --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $A16 --alternate-recipe $B24 --candidates-per-recipe 1 --training-parallelism 1 --opening-manifest $OPENING --executable $EXE20 --promotion-panel $PANEL5 --gate-version 7 --extra-manifest $EXTRA17 --demo-start-share 0.25 --random-deck-share 0.25 --gpt-import-dir $L/artifacts/task105/import --gpt-archive-root $L --gpt-cold-rates $COLD_RATES --obs-v3 --native-games --resource-dir $HOME/.cache/pvz-research-resources --frames $R/artifacts/student/frames_v2 --out $R/artifacts/hard --rollout-policy $R/artifacts/student/dw4i_s1/model.pt --value $VALUE5 --value-roof $VROOF --static-value $R/artifacts/hard/value_branch1/model.pt 2>&1 | tee -a $LOG_DIR/qc.out"
}
# All mutations below run only after explicit main-controller approval/execution.
IMPORT_FILES=(scripts/import_llm_run.py scripts/task105_import_batch.py scripts/pvz_live_import.py)
FILES=(scripts/pvz_live_curriculum.py scripts/pvz_live_import.py scripts/deploy22_import.py
       scripts/curriculum_queue.py scripts/search_play.py scripts/render_episode.py
       scripts/human_demo.py scripts/make_multiplier_tasks.py scripts/promotion_gate.py
       scripts/deploy17_boundary.py scripts/select_student.py scripts/train_rollout_policy.py scripts/day_orig_tasks.py
       scripts/pack_training_rows.py scripts/backfill_frame_context.py
       scripts/import_llm_run.py scripts/make_roof_panel.py
       scripts/unified_panel.py scripts/make_day_panel.py
       python/pvz_promotion_v7.py recipes/dw4i2_v2timer_warm.json
       python/pvz_common.py python/pvz_env.py python/pvz_promotion.py python/pvz_recipe.py
       python/pvz_selection_refresh.py python/pvz_student_selection.py python/pvz_training_rows.py
       python/pvz_research_data.py python/pvz_student_tokens.py python/pvz_token_blocks.py
       python/pvz_token_observation.py python/pvz_frame_store.py
       python/pvz_training_pack.py recipes/dw4i2_v2timer.json recipes/dw4i2_v2timer_history10.json
       recipes/dw4i2_v2timer_seed2cold.json recipes/dw4i2_v2timer_l7keep25.json recipes/panel_v2.json
       recipes/dw4i.json recipes/dw4i2.json recipes/dw4i2_rebalanced.json
       recipes/dw4i2_rebalanced_p05.json recipes/dw4i2_rebalanced_p05_v2sleep.json
       recipes/dw4i2_rebalanced_p05_v2timer.json recipes/dw4i2_v2sleep.json)
REMOVED=()
O=""
SAVED=0
STOPPED=0
log() { printf '%s deploy24: %s\n' "$(date -Iseconds)" "$*" >> "$LOG_DIR/qc_setup.log"; }
helper() {
  "$PY" "$O/install/scripts/deploy17_boundary.py" "$1" --worktree "$W" \
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
previous_launch() {
  /bin/bash -c 'source "$1"; watchdog_extract_start "$2"' deploy23 \
    "$W/scripts/qc_watchdog.sh" "$LOG_DIR/qc_deploy23.sh" > "$O/previous-launch.sh"
  cp -p "$LOG_DIR/qc_deploy23.sh" "$O/previous-deploy.sh"
}
backup_files() {
  local file mode
  for file in "${FILES[@]}" "${REMOVED[@]}"; do
    mkdir -p "$(dirname "$O/before/$file")"
    if [ -e "$W/$file" ]; then cp -p "$W/$file" "$O/before/$file";
    else touch "$O/before/$file.absent"; fi
  done
  mkdir -p "$O/modes"
  for mode in "$MODE24" "$ACTIVE24"; do
    if [ -e "$mode" ]; then cp -p "$mode" "$O/modes/$(basename "$mode")";
    else touch "$O/modes/$(basename "$mode").absent"; fi
  done
  for file in "${IMPORT_FILES[@]}"; do
    mkdir -p "$(dirname "$O/before_import/$file")"
    if [ -e "$L/$file" ]; then cp -p "$L/$file" "$O/before_import/$file";
    else touch "$O/before_import/$file.absent"; fi
  done
  printf 'complete\n' > "$O/backed_up"
  SAVED=1
}
restore_files() {
  local file mode
  for file in "${FILES[@]}" "${REMOVED[@]}"; do
    if [ -e "$O/before/$file.absent" ]; then rm -f "$W/$file";
    else cp -p "$O/before/$file" "$W/$file"; fi
  done
  for mode in "$MODE24" "$ACTIVE24"; do
    if [ -e "$O/modes/$(basename "$mode").absent" ]; then rm -f "$mode";
    else cp -p "$O/modes/$(basename "$mode")" "$mode"; fi
  done
  for file in "${IMPORT_FILES[@]}"; do
    if [ -e "$O/before_import/$file.absent" ]; then rm -f "$L/$file";
    else cp -p "$O/before_import/$file" "$L/$file"; fi
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
[ "$(realpath "$0")" = "$LOG_DIR/qc_deploy24.sh" ] || {
  printf 'After review, install this script at %s/qc_deploy24.sh (watchdog deployment guard).\n' "$LOG_DIR" >&2; exit 2; }
[ ! -e "$LOG_DIR/stop_curr" ] || { log 'ABORT: existing stop request'; exit 1; }
exec 9> "$LOG_DIR/qc_deploy24.lock"
flock -n 9 || { log 'ABORT: another deploy24 switch is running'; exit 1; }
mapfile -t REMOVED < <(git -C "$PACKAGE" diff --name-only --diff-filter=D "$BASE" "$REV" -- python scripts recipes configs experiments)
case "${1:-}" in
  --rollback)
    O=${2:-$(cat "$LOG_DIR/deploy24.latest")}
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
[ ! -e "$ACTIVE24" ] || { log 'ABORT: deploy24 already active'; exit 1; }
[ -f "$ACTIVE23" ] && [ -f "$MODE23" ] && grep -qx native "$MODE23"
[ -x "$PY" ] && [ -x "$EXE20" ] && [ -f "$VALUE5" ] && [ -f "$VROOF" ]
O=$LOG_DIR/deploy24_$(date +%Y%m%dT%H%M%S)_$$
mkdir -p "$O/install"
# Read pinned small source files before pausing qc. No frame replay, pack, or build here.
for file in "${FILES[@]}"; do
  mkdir -p "$(dirname "$O/install/$file")"
  git -C "$PACKAGE" show "$REV:$file" > "$O/install/$file"
done
previous_launch
log "prepared source $REV, reusing unchanged deploy20 engine; waiting for collection boundary"
wait_boundary
# Frozen idle scheduler: no next collector can race the backup and switch.
backup_files
printf '%s\n' "$O" > "$LOG_DIR/deploy24.latest"
stop_paused_qc
for file in "${FILES[@]}"; do
  mkdir -p "$(dirname "$W/$file")"
  cp "$O/install/$file" "$W/$file.deploy24.building"
  mv "$W/$file.deploy24.building" "$W/$file"
done
for file in "${REMOVED[@]}"; do rm -f "$W/$file"; done
"$PY" "$W/scripts/deploy22_import.py" --root "$L" --source "$W/scripts/pvz_live_import.py"
printf 'native\n' > "$MODE24"
printf '%s\n' "$REV" > "$ACTIVE24.building"
mv "$ACTIVE24.building" "$ACTIVE24"
start_qc
log "activated task200 per-cell live renewal and bounded task seeds; deploy23 roof leaf value, import pools and gate7/deploy20 engine retained; backup $O"
