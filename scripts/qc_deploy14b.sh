#!/bin/bash
# task97: deploy14 + exactly eleven measured LLM demo start cells.
# Prepared only; the controller owner executes this script at a group boundary.
# Keep build5, scripts/python/recipes (including the plane_of hotfix), both panels,
# and the gate ledger. Roll back only this deployment's config/rates/state files.
set -Eeuo pipefail
L=/home/newbiexvwu/pvz-logs/qc_setup.log
R=/home/newbiexvwu/PvZ-search
W=$R/.claude/worktrees/cq
SRC=$R/.claude/worktrees/redeploy
PY=/home/newbiexvwu/.venvs/ml/bin/python3
NEWEXE=$W/build5/pvz-portable
PANEL3=$W/artifacts/task73c/panel_v3_source.json
CONFIG=$W/configs/human_demo_tasks.json
PANEL=$W/artifacts/stage4/multiplier_panel.json
RATES=/home/newbiexvwu/pvz-logs/teacher_rates.json
STATE=/home/newbiexvwu/pvz-logs/curriculum_state.json
STOP=/home/newbiexvwu/pvz-logs/stop_curr
PAYLOAD=$SRC/configs/deploy14b
AUDIT=$SRC/MEMORY/docs/task97-deploy14b-audit.json
O=
log() { /usr/bin/printf '%s %s\n' "$(/usr/bin/date -Iseconds)" "$*" >> "$L"; }
# Verbatim start_qc from qc_deploy14.sh; same engine, recipe, and gate v3 panel.
start_qc() { tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $W/recipes/dw4i2.json --executable $1 --promotion-panel $PANEL3 2>&1 | tee -a ~/pvz-logs/qc.out"; }
preflight() {
  "$PY" - "$W" "$PAYLOAD" "$AUDIT" "$L" <<'PY'
import hashlib, json, sys
from pathlib import Path
w, payload, audit, log = map(Path, sys.argv[1:])
d = json.loads(audit.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
for p, key in [(w/'build5/pvz-portable', 'engine_sha256'),
               (w/'configs/human_demo_tasks.json', 'baseline_config_sha256'),
               (w/'artifacts/stage4/multiplier_panel.json', 'panel_sha256'),
               (payload/'human_demo_tasks.json', 'new_config_sha256'),
               (payload/'teacher_rates.delta.json', 'rates_delta_sha256')]:
    if sha(p) != d[key]: raise SystemExit(f'preflight SHA mismatch: {p}')
statuses = [line for line in log.read_text().splitlines()
            if 'deploy14:' in line and ('ROLLBACK' in line or 'first group after restart' in line)]
if not statuses or 'first group after restart rc=0, OK' not in statuses[-1]:
    raise SystemExit('deploy14 has no successful latest deployment')
for record in d['archives']:
    p = w / record['archive']
    for name, key in [('meta.json', 'meta_sha256'), ('ticks.jsonl', 'ticks_sha256')]:
        if sha(p/name) != record[key]: raise SystemExit(f'archive changed: {p/name}')
rates = json.loads((payload/'teacher_rates.delta.json').read_text())
if set(rates) != set(d['cells']) or len(rates) != 11:
    raise SystemExit('expected exactly 11 measured rates')
for name, row in rates.items():
    if (row['wins'], row['games']) != (d['cells'][name]['wins'], 8):
        raise SystemExit(f'rate differs from evidence: {name}')
PY
}
stop_qc() {
  if ! /usr/bin/tmux has-session -t =qc 2>/dev/null; then
    log 'deploy14b: qc already down, no stop needed'
    return
  fi
  [ ! -e "$STOP" ] || { log 'deploy14b: ABORT existing stop flag'; exit 1; }
  /usr/bin/touch "$STOP"
  # A dead controller cannot emit a new "stopping" line. Watch the session itself.
  for ((i=0; i<240; i++)); do
    if ! /usr/bin/tmux has-session -t =qc 2>/dev/null; then
      /usr/bin/rm -f "$STOP"
      return
    fi
    /usr/bin/sleep 30
  done
  /usr/bin/rm -f "$STOP"
  log 'deploy14b: ABORT qc did not exit within 120 min'
  exit 1
}
rollback() {
  trap - ERR
  log "deploy14b: $1 -> ROLLBACK to deploy14"
  /usr/bin/tmux kill-session -t =qc 2>/dev/null || true
  /usr/bin/sleep 5
  # All backups are taken together after qc has drained its background work.
  /usr/bin/cp "$O/human_demo_tasks.before.json" "$CONFIG" || exit 1
  /usr/bin/cp "$O/teacher_rates.before.json" "$RATES" || exit 1
  /usr/bin/cp "$O/curriculum_state.before.json" "$STATE" || exit 1
  # Panel, engine, gate ledger, and code were never replaced.
  start_qc "$NEWEXE" || { log 'deploy14b: rollback restart failed'; exit 1; }
  log 'deploy14b: restored deploy14 config/rates/state + same build5 and current hotfixes'
  exit 1
}
preflight || { log 'deploy14b: ABORT preflight'; exit 1; }
[ ! -e "$STOP" ] || { log 'deploy14b: ABORT existing stop flag'; exit 1; }
log 'deploy14b: preflight OK, waiting for qc group boundary'
stop_qc
# Recheck after the wait; do not overwrite another deployment's newer config.
preflight || { log 'deploy14b: ABORT preflight changed while waiting'; exit 1; }
O=$(/usr/bin/mktemp -d /home/newbiexvwu/pvz-logs/deploy14b.XXXXXX)
/usr/bin/cp "$CONFIG" "$O/human_demo_tasks.before.json"
/usr/bin/cp "$RATES" "$O/teacher_rates.before.json"
/usr/bin/cp "$STATE" "$O/curriculum_state.before.json"
/usr/bin/cp "$PANEL" "$O/multiplier_panel.before.json"
/usr/bin/cp "$PAYLOAD/human_demo_tasks.json" "$O/human_demo_tasks.after.json"
/usr/bin/cp "$PAYLOAD/teacher_rates.delta.json" "$O/teacher_rates.delta.json"
/usr/bin/cp "$AUDIT" "$O/audit.json"
trap 'rollback "command failed at line $LINENO"' ERR
/usr/bin/cp "$O/human_demo_tasks.after.json" "$CONFIG"
"$PY" - "$RATES" "$O/teacher_rates.delta.json" <<'PY'
import json, sys
from pathlib import Path
p, delta = map(Path, sys.argv[1:])
d = json.loads(p.read_text())
d.setdefault('bases', {}).update(json.loads(delta.read_text()))
temporary = p.with_suffix('.deploy14b.building')
temporary.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')
temporary.replace(p)
PY
log "deploy14b: installed 11 LLM cells and measured rates; backups $O; engine/panels/code unchanged"
/usr/bin/cp "$STATE" "$O/curriculum_state.dryrun.json"
# Explicit isolated state/log/stop/ledger paths: planning never spends live gate seeds.
/usr/bin/timeout 300 /usr/bin/env PVZ_NO_POPUP=1 "$PY" "$W/scripts/curriculum_queue.py" \
  --dry-run --rounds 100 --workers 2 --round-games 768 --games-per-group 48 \
  --manifest "$PANEL" --recipe "$W/recipes/dw4i2.json" --executable "$NEWEXE" \
  --promotion-panel "$PANEL3" --teacher-rates "$RATES" \
  --state "$O/curriculum_state.dryrun.json" --log "$O/dryrun.log" \
  --stop-flag "$O/stop" --seed-ledger "$O/gate_seed_ledger.dryrun.json" > "$O/dryrun.out" 2>&1
"$PY" - "$O" <<'PY'
import json, sys
from pathlib import Path
o = Path(sys.argv[1])
log = (o/'dryrun.log').read_text()
for task_id in json.loads((o/'teacher_rates.delta.json').read_text()):
    if f'start cell added: {task_id} ' not in log:
        raise SystemExit(f'dry-run did not register {task_id}')
if (o/'gate_seed_ledger.dryrun.json').exists():
    raise SystemExit('dry-run unexpectedly allocated gate seeds')
PY
log 'deploy14b: dry-run 11/11 LLM cells registered'
n0=$(/usr/bin/grep -c ' rc=' /home/newbiexvwu/pvz-logs/curriculum.log || true)
start_qc "$NEWEXE"
log 'deploy14b: restarted with deploy14 start_qc and unchanged build5'
fresh=0
for ((i=0; i<120; i++)); do
  /usr/bin/sleep 30
  n1=$(/usr/bin/grep -c ' rc=' /home/newbiexvwu/pvz-logs/curriculum.log || true)
  if [ "$n1" -gt "$n0" ]; then fresh=1; break; fi
  /usr/bin/tmux has-session -t =qc 2>/dev/null || break
done
if [ "$fresh" = 1 ] && /usr/bin/tmux has-session -t =qc 2>/dev/null && \
   /usr/bin/grep ' rc=' /home/newbiexvwu/pvz-logs/curriculum.log | /usr/bin/tail -1 | /usr/bin/grep -q ' rc=0 '; then
  log 'deploy14b: first group after restart rc=0, OK'
  exit 0
fi
rollback 'first new group failed, timed out, or qc exited'
