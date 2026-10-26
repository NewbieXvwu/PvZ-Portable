#!/bin/bash
# task92c PREPARED ONLY. Do not run until the controller decides after task84b.
# Code-only overlay on deploy14; keep its engine, collection panel and gate panel.
# No recipes, teacher rates, state, offsets or seed ledgers are restored on rollback.
set -Eeuo pipefail
R=/home/newbiexvwu/PvZ-search; W=$R/.claude/worktrees/cq; PY=/home/newbiexvwu/.venvs/ml/bin/python3
REV=c6e2f2bd60504c2af44dfd23e0ca05a74b0c376a
ENGINE_SHA=76ff36dc026f67641aefccd5f30f6cdae65eee14ee197dc24497413123a7874a
NEWEXE=$W/build5/pvz-portable
PANEL3=$W/artifacts/task73c/panel_v3_source.json
L=/home/newbiexvwu/pvz-logs/qc_setup.log
O=/home/newbiexvwu/pvz-logs/deploy14c_$(date +%Y%m%dT%H%M%S)
STOP_FLAG=/home/newbiexvwu/pvz-logs/stop_curr
STOPPED=0; SAVED=0; OWN_STOP=0
CODE_FILES=(scripts/curriculum_queue.py scripts/select_student.py scripts/pvz_tasks.py scripts/search_play.py
            python/pvz_student_selection.py python/pvz_selection_refresh.py configs/student_selection_v2.json)
log() { echo "$(date -Iseconds) deploy14c: $*" >> "$L"; }
start_qc() { tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $W/recipes/dw4i2.json --executable $1 --promotion-panel $PANEL3 2>&1 | tee -a ~/pvz-logs/qc.out"; }
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
baseline = json.loads(r'''{"scripts/curriculum_queue.py":"58a93196087c4b296acfa8ba0f528880116f604c45ca1fb12bfb9f366c650736","scripts/select_student.py":"6162eb46076b794f8ef3d2dac38ee6c4ba12d55aeb484c821db3c0e436ec99b3","scripts/pvz_tasks.py":"61279b49f572a4b1232d4b39a6b7349a36385e2b8e3e7081d61826c1096519bf","scripts/search_play.py":"1e7f07c687acc89ebb380af7e8dac90a947c0a46bc3f692e6780d8d09b4116c5","python/pvz_student_selection.py":null,"python/pvz_selection_refresh.py":null,"configs/student_selection_v2.json":null}''')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
immutable = {
    str(root / 'build5/pvz-portable'): '76ff36dc026f67641aefccd5f30f6cdae65eee14ee197dc24497413123a7874a',
    str(root / 'artifacts/stage4/multiplier_panel.json'): '68b37b98d3e29e83ac583fa5897372e7ed248213ec633d79edcf33e01eae26c3',
    str(root / 'artifacts/task73c/panel_v3_source.json'): 'bb0a9129b88023fe8c7722f267f1d184883ddea538860354421f6664d8e2f636',
}
logs = Path('/home/newbiexvwu/pvz-logs')
critical = list(immutable) + [str(logs / name) for name in (
    'teacher_rates.json', 'curriculum_state.json', 'gate_seed_ledger.json', 'selection_seed_ledger.json')]
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
  log "$1 -> code rollback; current state/offsets/ledgers are retained"
  if tmux has-session -t =qc 2>/dev/null; then stop_qc; fi
  if [ "$SAVED" = 1 ]; then code_state restore || { log 'ABORT: code restore failed'; exit 1; }; fi
  code_state restored || { log 'ABORT: restored baseline verification failed'; exit 1; }
  start_qc "$NEWEXE"
  log 'rolled back code and restarted the same deploy14 engine/panels'
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
log "preflight OK; candidate $REV; engine and source panels stay on deploy14"
stop_qc; STOPPED=1

# --- code backup + overlay only; do not copy/rebuild engines or panels ---
code_state save; SAVED=1
git -C "$W" restore --source "$REV" --worktree -- "${CODE_FILES[@]}"
code_state installed

# --- dry-run with deploy14 arguments; no seed allocation or training ---
timeout 300 env PVZ_NO_POPUP=1 "$PY" "$W/scripts/curriculum_queue.py" --dry-run --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest "$W/artifacts/stage4/multiplier_panel.json" --recipe "$W/recipes/dw4i2.json" --executable "$NEWEXE" --promotion-panel "$PANEL3" > "$O/dryrun.out" 2>&1 || rollback 'dry-run failed'
grep -q 'selection preview (no allocation, no engine, no state write)' "$O/dryrun.out" || rollback 'selection preview missing'
code_state unchanged
log 'dry-run passed; engine/panels/rates/state/ledgers unchanged'

# --- restart and require a NEW completed group, not a stale rc=0 line ---
n0=$(grep -c ' rc=' /home/newbiexvwu/pvz-logs/curriculum.log || true)
start_qc "$NEWEXE"
log "qc restarted with code $REV; production decision remains the controller's"
for ((i=0; i<120; i++)); do
  sleep 30
  if ! tmux has-session -t =qc 2>/dev/null; then break; fi
  n1=$(grep -c ' rc=' /home/newbiexvwu/pvz-logs/curriculum.log || true)
  if [ "$n1" -gt "$n0" ]; then
    if tail -n 200 /home/newbiexvwu/pvz-logs/curriculum.log | grep ' rc=' | tail -n 1 | grep -q ' rc=0 '; then
      log 'first new group rc=0; deployment complete'; exit 0
    fi
    break
  fi
done
rollback 'first new group failed, timed out or qc exited'
