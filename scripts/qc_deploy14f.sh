#!/bin/bash
# task114: deploy14e + task99 renewable start seeds/plane packing + task95 build4.
# Controller-only package: prepare and copy this file; do not execute for task114.
# Group-boundary stop, no extra v3/full-round drain. Refuse queued/running gate jobs.
# Reuse task95 gate evidence; never replay games as deployment preflight.
# Rollback returns to build5. After live collection starts, retain renewal-aware
# code so seed tables/offsets are not truncated by the previous controller.
set -Eeuo pipefail
R=/home/newbiexvwu/PvZ-search; W=$R/.claude/worktrees/cq; PY=/home/newbiexvwu/.venvs/ml/bin/python3
REV=88cb4310fd7c0ad7652789320ad7649238630201
ENGINE_SHA=de952714651c6544f4dba16f8ae832fe1ad1cb75bfc3e1d5495964d3d9a83845
OLD_SHA=76ff36dc026f67641aefccd5f30f6cdae65eee14ee197dc24497413123a7874a
SRC=${PVZ_DEPLOY14F_SOURCE:-$R}
ENGINE=$SRC/artifacts/task95/engine/pvz-portable-build4
NEWEXE=$W/build6/pvz-portable
OLDEXE=$W/build5/pvz-portable
ENGINE_CHOICE=/home/newbiexvwu/pvz-logs/qc_deploy14f.engine
PANEL3=$W/artifacts/task73c/panel_v3_source.json
L=/home/newbiexvwu/pvz-logs/qc_setup.log
O=/home/newbiexvwu/pvz-logs/deploy14f_$(date +%Y%m%dT%H%M%S)
STOP_FLAG=/home/newbiexvwu/pvz-logs/stop_curr
STOPPED=0; SAVED=0; OWN_STOP=0; STARTED=0
CODE_FILES=(scripts/curriculum_queue.py python/pvz_promotion.py scripts/train_rollout_policy.py scripts/pack_training_rows.py scripts/gate_seed_inventory.py scripts/check_start_seed_state.py MEMORY/docs/gate-seed-inventory.json src/EnvironmentRollout.cpp)
log() { echo "$(date -Iseconds) deploy14f: $*" >> "$L"; }
start_qc() {
  # A copied draft can become the live watchdog newest script before deployment.
  # Until activation use build5; after rollback the persisted choice also stays build5.
  if [ -f "$ENGINE_CHOICE" ]; then set -- "$(cat "$ENGINE_CHOICE")"; else set -- "$OLDEXE"; fi
  tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $W/recipes/dw4i2.json --executable $1 --promotion-panel $PANEL3 2>&1 | tee -a ~/pvz-logs/qc.out"
}
choose_engine() { printf '%s\n' "$1" > "$ENGINE_CHOICE.building"; mv "$ENGINE_CHOICE.building" "$ENGINE_CHOICE"; }

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
baseline = json.loads(r'''{"scripts/curriculum_queue.py":"d9fbde9ed08ffba81b42ed72d25d21831ef1235940c4e1cbfdf13d6464f74590","python/pvz_promotion.py":"88095712ab7ab3d6563c5765a6122eed3e835ed67320acd76f448d5772afab1d","scripts/train_rollout_policy.py":"6bffc3e5cbf10c57d5dea7e84a3e465f6c78c31ff5934ac8ad9c9ace91d2ca59","scripts/pack_training_rows.py":"de5e9d01a47f15acf45a2be9977b7924aa1cad7fbeb63676ccf9a4e5acc44274","scripts/gate_seed_inventory.py":null,"scripts/check_start_seed_state.py":null,"MEMORY/docs/gate-seed-inventory.json":null,"src/EnvironmentRollout.cpp":"75913f5c8ab8cf96adbedeede2b69ec81e13ce289f738f1140b40c211ca9b85f"}''')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
immutable = {
    str(root / 'build5/pvz-portable'): '76ff36dc026f67641aefccd5f30f6cdae65eee14ee197dc24497413123a7874a',
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
  log "$1 -> build5 rollback; current state/offsets/ledgers are retained"
  if tmux has-session -t =qc 2>/dev/null; then stop_qc; fi
  if [ "$STARTED" = 0 ]; then
    if [ "$SAVED" = 1 ]; then code_state restore || { log 'ABORT: code restore failed'; exit 1; }; fi
    code_state restored || { log 'ABORT: restored baseline verification failed'; exit 1; }
  else
    # New live groups may have renewed seeds already. Old code would truncate
    # their tables on restart; keep the new compatible code and current state.
    log 'retaining renewal-aware code and current seed tables for engine rollback'
    code_state installed || { log 'ABORT: live code changed; controller intervention required'; exit 1; }
  fi
  choose_engine "$OLDEXE"
  start_qc "$OLDEXE"
  log 'rolled back engine to build5; state, offsets and ledgers retained'
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
[ "$(sha256sum "$ENGINE" | cut -d' ' -f1)" = "$ENGINE_SHA" ] || { log 'ABORT: task95 build4 SHA mismatch'; exit 1; }
# Only read the small, already completed task95 evidence. No game/hash-suite replay.
"$PY" - "$SRC/MEMORY/docs/task95-engine-gate.json" "$ENGINE_SHA" <<'EVIDENCE'
import json
import sys
document = json.load(open(sys.argv[1]))
assert document['selected_candidate'] == 'build4'
assert document['binaries']['build4']['sha256'] == sys.argv[2]
assert document['logic_gate_build4']['pass'] and document['boss_l50_build4']['pass']
assert document['six_family_hash_suite_build4']['pass'] and document['throughput_build4']['pass']
EVIDENCE
code_state boundary
log "preflight OK; code $REV and task95 build4; source panels and gate v4 retained"
stop_qc; STOPPED=1

# --- backup code; overlay code and copy the immutable task95 build4 binary ---
code_state save; SAVED=1
git -C "$W" restore --source "$REV" --worktree -- "${CODE_FILES[@]}"
code_state installed
mkdir -p "$(dirname "$NEWEXE")"
cp "$ENGINE" "$NEWEXE"
[ "$(sha256sum "$NEWEXE" | cut -d' ' -f1)" = "$ENGINE_SHA" ] || rollback 'build4 copy SHA mismatch'

# --- dry-run with deploy14 arguments; no seed allocation or training ---
timeout 300 env PVZ_NO_POPUP=1 "$PY" "$W/scripts/curriculum_queue.py" --dry-run --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest "$W/artifacts/stage4/multiplier_panel.json" --recipe "$W/recipes/dw4i2.json" --executable "$NEWEXE" --promotion-panel "$PANEL3" > "$O/dryrun.out" 2>&1 || rollback 'dry-run failed'
grep -q 'selection preview (no allocation, no engine, no state write)' "$O/dryrun.out" || rollback 'selection preview missing'
code_state unchanged
log 'dry-run passed; build5/panels/rates/state/ledgers unchanged; build4 staged'

# Partial-round groups continue unchanged. Only issued queued/running jobs must
# finish before an engine switch; there is no new drain round or backfill.
code_state boundary
log 'no queued gate jobs; keep gate v4 and switch to task95 build4'

# --- restart and require a NEW completed group, not a stale rc=0 line ---
choose_engine "$NEWEXE"
STARTED=1
n0=$(grep -c ' rc=' /home/newbiexvwu/pvz-logs/curriculum.log || true)
start_qc "$NEWEXE"
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
