#!/bin/bash
# task128: prepared for the controller; do not execute during delivery.
# Gate v5 scores the adventure levels the game actually plays (original 1.0x, scoreboard
# decks) instead of synthetic curriculum cells; the two registered heldout cells keep
# running for monitoring only (no veto, not in the overall rate). The selection body is
# the same adventure set on independent seeds, and the task126 unseen decks become
# score-only. Python only: no engine change, build6 stays.
#
# WHICH DEPLOYMENT IS IN SERVICE (this decides both the un-activated/rollback arguments
# and the code baseline the script verifies before touching anything):
#   * ~/pvz-logs/qc_deploy14h.mode contains "rebalance" -> deploy14h was executed:
#     A+B(p=0.5), K=4 serial, --selection-holdout-manifest, gate v4, deploy14h baseline.
#   * otherwise -> deploy14g is in service (its mode file says "rebalance" since
#     2026-10-27 02:13): A+B(p=1), K=2, no selection holdout, gate v4, deploy14g baseline.
# Activation writes qc_deploy14i.mode=gatev5. Rollback deletes that file, so start_qc and
# the baseline check fall back to the branch above; state/offsets/ledgers are retained.
set -Eeuo pipefail
R=/home/newbiexvwu/PvZ-search; W=$R/.claude/worktrees/cq; PY=/home/newbiexvwu/.venvs/ml/bin/python3
REV=adbedd63ff5bd21336c3706b11ef920c65845f57
ENGINE_SHA=de952714651c6544f4dba16f8ae832fe1ad1cb75bfc3e1d5495964d3d9a83845
NEWEXE=$W/build6/pvz-portable
OLDEXE=$W/build6/pvz-portable
MODE14I=/home/newbiexvwu/pvz-logs/qc_deploy14i.mode
MODE14H=/home/newbiexvwu/pvz-logs/qc_deploy14h.mode
OPENING=$W/artifacts/task119/opening_panel.json
PANEL3=$W/artifacts/task73c/panel_v3_source.json
PANEL5=$W/configs/gate_v5_panel.json
L=/home/newbiexvwu/pvz-logs/qc_setup.log
O=/home/newbiexvwu/pvz-logs/deploy14i_$(date +%Y%m%dT%H%M%S)
STOP_FLAG=/home/newbiexvwu/pvz-logs/stop_curr
STOPPED=0; SAVED=0; OWN_STOP=0; STARTED=0
CODE_FILES=(scripts/curriculum_queue.py scripts/train_rollout_policy.py python/pvz_recipe.py python/pvz_training_rows.py scripts/make_rebalance_panel.py recipes/dw4i2_rebalanced.json recipes/dw4i2_rebalanced_p05.json scripts/search_play.py python/test_save_candidates.py python/test_incremental_pack.py python/test_multiseed_production.py python/pvz_selection_refresh.py configs/student_selection_holdout.json python/test_selection_holdout.py python/pvz_promotion.py scripts/promotion_gate.py scripts/select_student.py scripts/make_gate_v5_panel.py configs/gate_v5_panel.json configs/adventure_scoreboard.json python/test_promotion.py python/test_gate_v5_panel.py python/test_promotion_evaluation.py python/test_curriculum_pipeline.py python/test_backfill_round_log.py)
# make_gate_v5_panel --check reads these frozen inputs after the overlay. They
# are absent from both prior deployment packages; restore removes them again.
CODE_FILES+=(MEMORY/docs/adventure-scoreboard-r11-32.json MEMORY/docs/task115-teacher-arm-weak-levels.md)
log() { echo "$(date -Iseconds) deploy14i: $*" >> "$L"; }
service() { if [ -f "$MODE14H" ] && grep -qx rebalance "$MODE14H"; then printf 'h'; else printf 'g'; fi; }
start_qc() {
  if [ -f "$MODE14I" ] && grep -qx gatev5 "$MODE14I"; then
    tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $W/recipes/dw4i2.json --alternate-recipe $W/recipes/dw4i2_rebalanced_p05.json --candidates-per-recipe 2 --training-parallelism 1 --selection-holdout-manifest $W/configs/student_selection_holdout.json --opening-manifest $OPENING --executable $NEWEXE --promotion-panel $PANEL5 --gate-version 5 2>&1 | tee -a ~/pvz-logs/qc.out"
  elif [ -f "$MODE14H" ] && grep -qx rebalance "$MODE14H"; then
    tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $W/recipes/dw4i2.json --alternate-recipe $W/recipes/dw4i2_rebalanced_p05.json --candidates-per-recipe 2 --training-parallelism 1 --selection-holdout-manifest $W/configs/student_selection_holdout.json --opening-manifest $OPENING --executable $NEWEXE --promotion-panel $PANEL3 2>&1 | tee -a ~/pvz-logs/qc.out"
  else
    tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $W/recipes/dw4i2.json --alternate-recipe $W/recipes/dw4i2_rebalanced.json --opening-manifest $OPENING --executable $NEWEXE --promotion-panel $PANEL3 2>&1 | tee -a ~/pvz-logs/qc.out"
  fi
}
choose_mode() { printf '%s\n' "$1" > "$MODE14I.building"; mv "$MODE14I.building" "$MODE14I"; }

start_watchdog() {
  if ! tmux has-session -t =qc-watchdog 2>/dev/null; then
    tmux new-session -d -s qc-watchdog -x 200 -y 50 "bash $W/scripts/qc_watchdog.sh >> /home/newbiexvwu/pvz-logs/qc_watchdog.out 2>&1"
  fi
}
cleanup() { if [ "$OWN_STOP" = 1 ]; then rm -f "$STOP_FLAG"; fi; }
stop_qc() {
  if ! tmux has-session -t =qc 2>/dev/null; then return; fi
  touch "$STOP_FLAG"; OWN_STOP=1
  # Group boundary + background drain. Never kill a running game/trainer.
  for ((i=0; i<720; i++)); do
    if ! tmux has-session -t =qc 2>/dev/null; then
      rm -f "$STOP_FLAG"; OWN_STOP=0; return
    fi
    sleep 30
  done
  log "ABORT: qc did not drain within six hours; code was not forcibly switched"
  exit 1
}

code_state() {
  "$PY" - "$1" "$W" "$O" "$REV" "$(service)" <<'PYCODE'
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

action, root, output, revision, service = sys.argv[1:6]
root, output = Path(root), Path(output)
# Baseline of the deployment that is in service right now. 'g' = deploy14g never
# upgraded to 14h; 'h' = deploy14h already executed (its mode file says rebalance).
BASELINE_G = {"scripts/curriculum_queue.py":"f1bbebd6a401324c89bdc22de5c974d17a0191d84f52e836af83df1b3cc52bae", "scripts/train_rollout_policy.py":"4de501a0548e2f3198fbc5b002fb834c4faa676d02c2515226fbcfcb4d8c412d", "python/pvz_recipe.py":"e067c9e3f45bb61f27276ddc85e045b3ba61424ccc4dd4d98a28e16420e188cd", "python/pvz_training_rows.py":"e83fe2f2df5464886499344aa693e34469a2f7ea08b4e392589cc8679de215ea", "scripts/make_rebalance_panel.py":"435a7ff0c9f9cde8cb12aaed6f4446beba43bca435c86055dff231243d26a038", "recipes/dw4i2_rebalanced.json":"fa9bf3767a55f0ff63f57a25b653422a704a1815d6ca80e275c95a1eb5e20028", "recipes/dw4i2_rebalanced_p05.json":None, "scripts/search_play.py":"52d00f236a6b2a66e3e83305602582896aa9103ea8098a7c21ab6a7a40f9e00f", "python/test_save_candidates.py":None, "python/test_incremental_pack.py":"6232dbb46e3319e77992405c3dca64552f2c1da6c195658283a2053aa51067aa", "python/test_multiseed_production.py":None, "python/pvz_selection_refresh.py":"0a74b8d150e1f2d041a7413ed86f5bbf05fdb2e7143a5c63ba86cf440f40c3df", "configs/student_selection_holdout.json":None, "python/test_selection_holdout.py":None, "python/pvz_promotion.py":"3ad0b4f08a2d413688ab132912ae662e25420ab5d3b9e4052fe4ed995e51e9ec", "scripts/promotion_gate.py":"7937f4c87880b255a4a7c0625254f0adf34493719a901150ffd40a8f9154042f", "scripts/select_student.py":"e636c6d3de99fb47c6f7d707a99f96d8febc76dda0f5db9e08018a92009d69be", "scripts/make_gate_v5_panel.py":None, "configs/gate_v5_panel.json":None, "configs/adventure_scoreboard.json":None, "python/test_promotion.py":"b645da43aeaf4124676daa410ce51043430403307ef687eea44269c68eec93e6", "python/test_gate_v5_panel.py":None, "python/test_promotion_evaluation.py":"887b05788d63a51a6344df08a8c550be6cfe05c07957a260e8dad648e43d20e7", "python/test_curriculum_pipeline.py":"fd359d75b437514877d2aa1f6734b24da55facafa9c3b6507b048d74333075e4", "python/test_backfill_round_log.py":None}
BASELINE_H = {"scripts/curriculum_queue.py":"fe7442f1b65a753f2f5955933bff107ebb2a01103e82525c4c4585f2651ca3bf", "scripts/train_rollout_policy.py":"eb1dd6a3fc39403e47041b0543d068a4fb9357443f42909eb755ccbb980e0650", "python/pvz_recipe.py":"cbd8ea3211a830fe01bd40aa6ddd1c5e8579290d823e5326ae1933899e881ff1", "python/pvz_training_rows.py":"1878b1c2000630a44bd81c4e6093a97e24ae9c9b9dc22b8bd7d27ed06921bcee", "scripts/make_rebalance_panel.py":"435a7ff0c9f9cde8cb12aaed6f4446beba43bca435c86055dff231243d26a038", "recipes/dw4i2_rebalanced.json":"fa9bf3767a55f0ff63f57a25b653422a704a1815d6ca80e275c95a1eb5e20028", "recipes/dw4i2_rebalanced_p05.json":"db051d589965a9bb3db7feb02e0e500b925bc00c048f51ff8d584cd9ac6425ba", "scripts/search_play.py":"6f9103447dd4e5d066c6b6b26ff516b7165c3d8a89f4c67ea3cf5f63540acf01", "python/test_save_candidates.py":"86aebab36bb6e680cabd1ce478b2c43ab5487110a2cd1dffb061e5a19f8cc66d", "python/test_incremental_pack.py":"6d8eaeb8bbf7530ebafe56c70d4d1baffc2e767980437d3fe734f8019974cae3", "python/test_multiseed_production.py":"f158f25fa27303cabc7b93ebae4c7a3d62a6d0eece79e720de02753906b7da88", "python/pvz_selection_refresh.py":"0cbc2f40b691f4bf2c525322f4be064a6a83945a74cce5682465cc78ebf6e7e6", "configs/student_selection_holdout.json":"a9ea7b20c7ff5aaf4ed8abf9ae845a32afcdfcbc2ad77f43d998682db2ffdf14", "python/test_selection_holdout.py":"4a88d07dcb1db760e2e4d9b109275765d9239326478709eaa8fa28091abd55c2", "python/pvz_promotion.py":"3ad0b4f08a2d413688ab132912ae662e25420ab5d3b9e4052fe4ed995e51e9ec", "scripts/promotion_gate.py":"7937f4c87880b255a4a7c0625254f0adf34493719a901150ffd40a8f9154042f", "scripts/select_student.py":"e636c6d3de99fb47c6f7d707a99f96d8febc76dda0f5db9e08018a92009d69be", "scripts/make_gate_v5_panel.py":None, "configs/gate_v5_panel.json":None, "configs/adventure_scoreboard.json":None, "python/test_promotion.py":"b645da43aeaf4124676daa410ce51043430403307ef687eea44269c68eec93e6", "python/test_gate_v5_panel.py":None, "python/test_promotion_evaluation.py":"887b05788d63a51a6344df08a8c550be6cfe05c07957a260e8dad648e43d20e7", "python/test_curriculum_pipeline.py":"fd359d75b437514877d2aa1f6734b24da55facafa9c3b6507b048d74333075e4", "python/test_backfill_round_log.py":None}
baseline = {'g': BASELINE_G, 'h': BASELINE_H}[service]
baseline.update({
    'MEMORY/docs/adventure-scoreboard-r11-32.json': None,
    'MEMORY/docs/task115-teacher-arm-weak-levels.md': None,
})
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
immutable = {
    str(root / 'build6/pvz-portable'): 'de952714651c6544f4dba16f8ae832fe1ad1cb75bfc3e1d5495964d3d9a83845',
    str(root / 'artifacts/stage4/multiplier_panel.json'): '68b37b98d3e29e83ac583fa5897372e7ed248213ec633d79edcf33e01eae26c3',
    str(root / 'artifacts/task73c/panel_v3_source.json'): 'bb0a9129b88023fe8c7722f267f1d184883ddea538860354421f6664d8e2f636',
}
logs = Path('/home/newbiexvwu/pvz-logs')
critical = list(immutable) + [str(logs / name) for name in (
    'teacher_rates.json', 'curriculum_state.json', 'gate_seed_ledger.json', 'selection_seed_ledger.json', 'start_cell_seed_ledger.json')]
if action in ('preflight', 'restored'):
    for name, expected in baseline.items():
        if sha(root / name) != expected:
            raise SystemExit(f'code baseline changed: {name}; in-service deployment is deploy14{service} '
                             f'-- refresh this deployment package')
    for name, expected in immutable.items():
        if sha(Path(name)) != expected:
            raise SystemExit(f'deploy14i engine/panel changed: {name}')
elif action == 'save':
    for name, expected in baseline.items():
        if sha(root / name) != expected:
            raise SystemExit(f'code changed while qc was draining: {name}')
        if expected is not None:
            target = output / 'code.before' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, target)
    (output / 'critical.before.json').write_text(json.dumps({name: sha(Path(name)) for name in critical}, indent=2))
elif action == 'restore':
    for name, expected in baseline.items():
        target, saved = root / name, output / 'code.before' / name
        if expected is None:
            target.unlink(missing_ok=True)
        else:
            if sha(saved) != expected:
                raise SystemExit(f'code backup corrupt: {name}')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(saved, target)
elif action == 'installed':
    for name in baseline:
        content = subprocess.check_output(['git', '-C', str(root), 'show', f'{revision}:{name}'])
        if sha(root / name) != hashlib.sha256(content).hexdigest():
            raise SystemExit(f'candidate overlay mismatch: {name}')
elif action == 'boundary':
    state = json.loads((logs / 'curriculum_state.json').read_text())
    if any(job.get('status') != 'done' and not job.get('isolated')
           for job in state.get('training_jobs', [])):
        raise SystemExit('gate switch refused: queued/running gate jobs remain')
    (output / 'gate_switch.json').write_text(json.dumps(dict(
        completed_round=state['round'], gate_version=5,
        engine_sha256='de952714651c6544f4dba16f8ae832fe1ad1cb75bfc3e1d5495964d3d9a83845'), indent=2))
elif action == 'unchanged':
    before = json.loads((output / 'critical.before.json').read_text())
    after = {name: sha(Path(name)) for name in critical}
    (output / 'critical.after_dryrun.json').write_text(json.dumps(after, indent=2))
    if before != after:
        raise SystemExit('dry-run changed engine, collection/gate panel, rates, state or seed ledgers')
else:
    raise SystemExit(f'unknown code-state action: {action}')
PYCODE
}

rollback() {
  trap - ERR
  log "$1 -> restoring the in-service deployment (deploy14$(service)); state/offsets/ledgers retained"
  if tmux has-session -t =qc 2>/dev/null; then stop_qc; fi
  if [ "$SAVED" = 1 ]; then code_state restore || { log 'ABORT: code restore failed'; exit 1; }; fi
  code_state restored || { log 'ABORT: restored baseline verification failed'; exit 1; }
  # Dropping the activation marker sends start_qc back to deploy14h (if it was applied)
  # or deploy14g, which is exactly what was in service before this script ran.
  rm -f "$MODE14I"
  start_qc
  log 'restored the previous deployment code and arguments; build6 unchanged'
  exit 1
}

on_error() {
  local status=$?
  trap - ERR
  if [ "$STOPPED" = 1 ]; then rollback "command failed (status $status, line $1)"; fi
  log "ABORT before switch (status $status, line $1)"
  exit "$status"
}
trap cleanup EXIT
trap 'on_error $LINENO' ERR

# --- preflight: no qc stop until the exact in-service baseline is verified ---
mkdir -p "$O"
[ ! -e "$STOP_FLAG" ] || { log 'ABORT: another stop request exists'; exit 1; }
git -C "$W" cat-file -e "$REV^{commit}"
code_state preflight
code_state boundary
log "preflight OK; code $REV; in-service deploy14$(service); build6, source panels and gate v4 retained"
stop_qc; STOPPED=1

# --- backup and overlay the task128 Python code, panels and manifests ---
code_state save; SAVED=1
git -C "$W" restore --source "$REV" --worktree -- "${CODE_FILES[@]}"
code_state installed
"$PY" "$W/scripts/make_rebalance_panel.py" --out "$OPENING"
# The committed v5 panel must regenerate byte-identically from its pinned sources.
"$PY" "$W/scripts/make_gate_v5_panel.py" --check

# --- dry-run with the gate-v5 arguments; no seed allocation or training ---
timeout 300 env PVZ_NO_POPUP=1 "$PY" "$W/scripts/curriculum_queue.py" --dry-run --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest "$W/artifacts/stage4/multiplier_panel.json" --recipe "$W/recipes/dw4i2.json" --alternate-recipe "$W/recipes/dw4i2_rebalanced_p05.json" --candidates-per-recipe 2 --training-parallelism 1 --selection-holdout-manifest "$W/configs/student_selection_holdout.json" --opening-manifest "$OPENING" --log "$O/dryrun.log" --executable "$NEWEXE" --promotion-panel "$PANEL5" --gate-version 5 > "$O/dryrun.out" 2>&1 || rollback 'dry-run failed'
grep -q 'selection preview (no allocation, no engine, no state write)' "$O/dryrun.out" || rollback 'selection preview missing'
grep -q '"adventure_tasks": 36' "$O/dryrun.out" || rollback 'adventure selection body missing'
code_state unchanged
log 'dry-run passed; 36 adventure levels x 9 games per candidate + 3 random families + 2 score-only unseen decks; production state unchanged'

# Partial-round groups continue unchanged. Only issued queued/running jobs must
# finish before switching arguments; there is no new drain round or backfill.
code_state boundary
log 'no queued gate jobs; switching to gate v5'

# --- restart and require a NEW completed group, not a stale rc=0 line ---
choose_mode gatev5
STARTED=1
n0=$(grep -c ' rc=' /home/newbiexvwu/pvz-logs/curriculum.log || true)
start_qc
log "qc restarted with code $REV and gate v5; production decision remains the controller's"
for ((i=0; i<120; i++)); do
  sleep 30
  if ! tmux has-session -t =qc 2>/dev/null; then break; fi
  n1=$(grep -c ' rc=' /home/newbiexvwu/pvz-logs/curriculum.log || true)
  if [ "$n1" -gt "$n0" ]; then
    if tail -n 200 /home/newbiexvwu/pvz-logs/curriculum.log | grep ' rc=' | tail -n 1 | grep -q ' rc=0 '; then
      log 'first new group rc=0'
      start_watchdog
      log 'deployment complete; watchdog in tmux qc-watchdog (120s checks, 2 restarts/30min)'
      exit 0
    fi
    break
  fi
done
rollback 'first new group failed, timed out or qc exited'
