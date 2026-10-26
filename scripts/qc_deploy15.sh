#!/bin/bash
# task95: engine-only deployment. Written for the main controller; never auto-executed.
# Keep deploy14fix code, panel, recipe and rates; deploy the archived PGO binary.
set -u
R=/home/newbiexvwu/PvZ-search
W=$R/.claude/worktrees/cq
SRC=$R/.claude/worktrees/deploy15
PY=/home/newbiexvwu/.venvs/ml/bin/python3
REV=0b8dcc26 # Current qc_deploy14.sh: 7722c0d1 plus the seeds-exhausted fix.
ENGINE_SHA=07703a5781e70d2c3b1b3ad7b0c473ddfaf2a313ac2ce4f5724035ba44e184be
OLD_SHA=76ff36dc026f67641aefccd5f30f6cdae65eee14ee197dc24497413123a7874a
ENGINE=$SRC/artifacts/task95/engine/pvz-portable
NEWEXE=$W/build6/pvz-portable
OLDEXE=$W/build5/pvz-portable
O=/home/newbiexvwu/pvz-logs/deploy15
L=/home/newbiexvwu/pvz-logs/qc_setup.log
PANEL3=$W/artifacts/task73c/panel_v3_source.json
mkdir -p "$O"
log() { echo "$(date -Iseconds) $*" >> "$L"; }
abort() { log "deploy15: ABORT $*"; exit 1; }
start_qc() { tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $W/recipes/dw4i2.json --executable $1 --promotion-panel $PANEL3 2>&1 | tee -a /home/newbiexvwu/pvz-logs/qc.out"; }
stop_qc() {
  tmux has-session -t =qc 2>/dev/null || { log "deploy15: qc already down"; return; }
  touch /home/newbiexvwu/pvz-logs/stop_curr
  for i in $(seq 1 240); do
    tmux has-session -t =qc 2>/dev/null || { rm -f /home/newbiexvwu/pvz-logs/stop_curr; return; }
    sleep 30
  done
  rm -f /home/newbiexvwu/pvz-logs/stop_curr
  abort "qc did not exit within 120 min"
}
rollback() {
  log "deploy15: $1 -> engine-only ROLLBACK to deploy14"
  tmux kill-session -t =qc 2>/dev/null || true
  sleep 5
  start_qc "$OLDEXE"
  exit 1
}

# Read-only preflight. Refuse code drift rather than checking out an older scripts tree.
[ "$(sha256sum "$ENGINE" | cut -d' ' -f1)" = "$ENGINE_SHA" ] || abort "archived engine SHA mismatch"
[ "$(sha256sum "$OLDEXE" | cut -d' ' -f1)" = "$OLD_SHA" ] || abort "deploy14 rollback engine SHA mismatch"
git -C "$W" diff --quiet "$REV" -- scripts python recipes configs || abort "code differs from deploy14fix $REV"
"$PY" - "$SRC" "$ENGINE" "$ENGINE_SHA" "$OLD_SHA" <<'PY' || abort "task95 gate evidence missing, incomplete or failed"
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys
root = Path(sys.argv[1])
engine = Path(sys.argv[2]).resolve()
new_sha, old_sha = sys.argv[3:5]
gate = json.loads((root / 'artifacts/task95/gate/gate_evidence.json').read_text())
speed = json.loads((root / 'artifacts/task95/throughput_abba.json').read_text())
boss = json.loads((root / 'artifacts/task95/boss_l50.json').read_text())
probe = json.loads((root / 'MEMORY/docs/task95-engine-gate.json').read_text())['state_hash_probe']['new']
assert gate['pass'] is True and gate['seeds'] == 8 and len(gate['cells']) == 6
assert {cell['cell'] for cell in gate['cells']} == {'day', 'night', 'pool', 'conveyor', 'roof', 'fog'}
assert all(cell['games'] == 16 for cell in gate['cells'])
assert all(not cell['mismatches'] for cell in gate['cells'])
assert Path(gate['new_executable']).resolve() == engine
assert speed['pass'] is True and len(speed['games']) == 16
assert Path(speed['new']).resolve() == engine
assert Counter(row['cell'] for row in speed['games']) == {'day': 8, 'pool': 8}
assert len({(row['cell'], row['seed']) for row in speed['games']}) == 16
assert all(math.isfinite(row[tag + '_seconds']) and row[tag + '_seconds'] > 0
           for row in speed['games'] for tag in ('old', 'new'))
assert all(row['old_decisions'] == row['new_decisions'] for row in speed['games'])
median = statistics.median(row['old_seconds'] / row['new_seconds'] for row in speed['games'])
assert median >= 0.98 and abs(median - speed['median_throughput_ratio']) < 1e-12
assert speed['median_throughput_ratio'] >= 0.98 and not speed['decision_mismatch']
assert boss['pass'] is True and len(boss['games']) == 4 and all(boss['checks'].values())
assert boss['task_id'] == 'boss_l50_orig_eval' and boss['seeds'] == list(range(85000000, 85000004))
assert boss['commands']['new']['executable_sha256'] == new_sha
assert boss['commands']['old']['executable_sha256'] == old_sha
assert Path(probe['executable']).resolve() == engine
assert probe['multi_process']['distinct'] == probe['same_process']['distinct'] == 1
PY
log "deploy15: preflight passed; waiting for qc group boundary"
stop_qc
cp "$W/artifacts/stage4/multiplier_panel.json" "$O/multiplier_panel.before.json" || rollback "panel backup failed"
cp /home/newbiexvwu/pvz-logs/teacher_rates.json "$O/teacher_rates.before.json" || rollback "rates backup failed"
cp /home/newbiexvwu/pvz-logs/curriculum_state.json "$O/curriculum_state.before.json" || rollback "state backup failed"
mkdir -p "$W/build6" || rollback "cannot create engine directory"
cp "$ENGINE" "$NEWEXE" || rollback "engine copy failed"
[ "$(sha256sum "$NEWEXE" | cut -d' ' -f1)" = "$ENGINE_SHA" ] || rollback "engine copy corrupted"
timeout 300 env PVZ_NO_POPUP=1 "$PY" "$W/scripts/curriculum_queue.py" --dry-run --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest "$W/artifacts/stage4/multiplier_panel.json" --recipe "$W/recipes/dw4i2.json" --executable "$NEWEXE" --promotion-panel "$PANEL3" > "$O/dryrun.out" 2>&1 || rollback "dry-run failed"
n0=$(grep -c ' rc=' /home/newbiexvwu/pvz-logs/curriculum.log)
start_qc "$NEWEXE"
log "deploy15: restarted with archived 07703a57; deploy14fix code/panel/rates retained"
for i in $(seq 1 120); do
  sleep 30
  [ "$(grep -c ' rc=' /home/newbiexvwu/pvz-logs/curriculum.log)" -gt "$n0" ] && break
  tmux has-session -t =qc 2>/dev/null || break
done
if [ "$(grep -c ' rc=' /home/newbiexvwu/pvz-logs/curriculum.log)" -gt "$n0" ] && tmux has-session -t =qc 2>/dev/null && grep ' rc=' /home/newbiexvwu/pvz-logs/curriculum.log | tail -1 | grep -q ' rc=0 '; then
  log "deploy15: first group rc=0, OK"
  exit 0
fi
rollback "first group failed or qc exited"
