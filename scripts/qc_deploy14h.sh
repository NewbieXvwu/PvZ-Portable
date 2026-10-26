#!/bin/bash
# task121/122: prepared for the controller; do not execute during delivery.
# Tempered level balancing (p=0.5, 20x ceiling) is recipe B; fog night cells (L31-40) get
# search_play --save-candidates 1 by the controller default (task122). Python search only: no engine change.
set -Eeuo pipefail
R=/home/newbiexvwu/PvZ-search; W=$R/.claude/worktrees/cq; PY=/home/newbiexvwu/.venvs/ml/bin/python3
REV=9afe653685c60997abe1f165dd0758d50dc846d7
ENGINE_SHA=de952714651c6544f4dba16f8ae832fe1ad1cb75bfc3e1d5495964d3d9a83845
NEWEXE=$W/build6/pvz-portable
OLDEXE=$W/build6/pvz-portable
MODE_CHOICE=/home/newbiexvwu/pvz-logs/qc_deploy14h.mode
OPENING=$W/artifacts/task119/opening_panel.json
PANEL3=$W/artifacts/task73c/panel_v3_source.json
L=/home/newbiexvwu/pvz-logs/qc_setup.log
O=/home/newbiexvwu/pvz-logs/deploy14h_$(date +%Y%m%dT%H%M%S)
STOP_FLAG=/home/newbiexvwu/pvz-logs/stop_curr
STOPPED=0; SAVED=0; OWN_STOP=0; STARTED=0
CODE_FILES=(scripts/curriculum_queue.py scripts/train_rollout_policy.py python/pvz_recipe.py python/pvz_training_rows.py scripts/make_rebalance_panel.py recipes/dw4i2_rebalanced.json recipes/dw4i2_rebalanced_p05.json scripts/search_play.py python/test_save_candidates.py)
log() { echo "$(date -Iseconds) deploy14h: $*" >> "$L"; }
start_qc() {
  if [ -f "$MODE_CHOICE" ] && grep -qx rebalance "$MODE_CHOICE"; then
    tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $W/recipes/dw4i2.json --alternate-recipe $W/recipes/dw4i2_rebalanced_p05.json --opening-manifest $OPENING --executable $NEWEXE --promotion-panel $PANEL3 2>&1 | tee -a ~/pvz-logs/qc.out"
  else
    tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $W/recipes/dw4i2.json --alternate-recipe $W/recipes/dw4i2_rebalanced.json --opening-manifest $OPENING --executable $OLDEXE --promotion-panel $PANEL3 2>&1 | tee -a ~/pvz-logs/qc.out"
  fi
}
choose_mode() { printf '%s\n' "$1" > "$MODE_CHOICE.building"; mv "$MODE_CHOICE.building" "$MODE_CHOICE"; }

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
  "$PY" - "$1" "$W" "$O" "$REV" <<'PYCODE'
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

action, root, output, revision = sys.argv[1:]
root, output = Path(root), Path(output)
baseline = json.loads(r'''{"scripts/curriculum_queue.py":"f1bbebd6a401324c89bdc22de5c974d17a0191d84f52e836af83df1b3cc52bae","scripts/train_rollout_policy.py":"4de501a0548e2f3198fbc5b002fb834c4faa676d02c2515226fbcfcb4d8c412d","python/pvz_recipe.py":"e067c9e3f45bb61f27276ddc85e045b3ba61424ccc4dd4d98a28e16420e188cd","python/pvz_training_rows.py":"e83fe2f2df5464886499344aa693e34469a2f7ea08b4e392589cc8679de215ea","scripts/make_rebalance_panel.py":"435a7ff0c9f9cde8cb12aaed6f4446beba43bca435c86055dff231243d26a038","recipes/dw4i2_rebalanced.json":"fa9bf3767a55f0ff63f57a25b653422a704a1815d6ca80e275c95a1eb5e20028","recipes/dw4i2_rebalanced_p05.json":null,"scripts/search_play.py":"52d00f236a6b2a66e3e83305602582896aa9103ea8098a7c21ab6a7a40f9e00f","python/test_save_candidates.py":null}''')
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
            raise SystemExit(f'code baseline changed: {name}; controller must refresh this deployment package')
    for name, expected in immutable.items():
        if sha(Path(name)) != expected:
            raise SystemExit(f'deploy14 engine/panel changed: {name}')
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
        raise SystemExit('engine switch refused: queued/running gate jobs remain')
    (output / 'gate_switch.json').write_text(json.dumps(dict(
        completed_round=state['round'], gate_version=4, engine_sha256='de952714651c6544f4dba16f8ae832fe1ad1cb75bfc3e1d5495964d3d9a83845'), indent=2))
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
  log "$1 -> deploy14g rollback; state/offsets/ledgers retained"
  if tmux has-session -t =qc 2>/dev/null; then stop_qc; fi
  if [ "$SAVED" = 1 ]; then code_state restore || { log 'ABORT: code restore failed'; exit 1; }; fi
  code_state restored || { log 'ABORT: restored baseline verification failed'; exit 1; }
  choose_mode baseline
  start_qc
  log 'restored deploy14g code and arguments; build6 unchanged'
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

# --- preflight: no qc stop until the exact baseline is verified ---
mkdir -p "$O"
[ ! -e "$STOP_FLAG" ] || { log 'ABORT: another stop request exists'; exit 1; }
git -C "$W" cat-file -e "$REV^{commit}"
code_state preflight
code_state boundary
log "preflight OK; code $REV; build6, source panels and gate v4 retained"
stop_qc; STOPPED=1

# --- backup and overlay only the task121 code ---
code_state save; SAVED=1
git -C "$W" restore --source "$REV" --worktree -- "${CODE_FILES[@]}"
code_state installed
"$PY" "$W/scripts/make_rebalance_panel.py" --out "$OPENING"

# --- dry-run with deploy14 arguments; no seed allocation or training ---
timeout 300 env PVZ_NO_POPUP=1 "$PY" "$W/scripts/curriculum_queue.py" --dry-run --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest "$W/artifacts/stage4/multiplier_panel.json" --recipe "$W/recipes/dw4i2.json" --alternate-recipe "$W/recipes/dw4i2_rebalanced_p05.json" --opening-manifest "$OPENING" --log "$O/dryrun.log" --executable "$NEWEXE" --promotion-panel "$PANEL3" > "$O/dryrun.out" 2>&1 || rollback 'dry-run failed'
grep -q 'selection preview (no allocation, no engine, no state write)' "$O/dryrun.out" || rollback 'selection preview missing'
code_state unchanged
log 'dry-run passed; engine/panels/rates/state/ledgers unchanged; recipe A/B and opening cells staged'

# Partial-round groups continue unchanged. Only issued queued/running jobs must
# finish before switching arguments; there is no new drain round or backfill.
code_state boundary
log 'no queued gate jobs; keep gate v4 and build6'

# --- restart and require a NEW completed group, not a stale rc=0 line ---
choose_mode rebalance
STARTED=1
n0=$(grep -c ' rc=' /home/newbiexvwu/pvz-logs/curriculum.log || true)
start_qc
log "qc restarted with code $REV; production decision remains the controller's"
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
