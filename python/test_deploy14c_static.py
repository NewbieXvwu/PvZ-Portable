"""Parse the prepared deployment package. Never execute/source the deployment script."""
import ast
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/qc_deploy14c.sh'
EXPECTED_START = 'start_qc() { tmux new-session -d -s qc -x 200 -y 50 "env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $W/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest $W/artifacts/stage4/multiplier_panel.json --recipe $W/recipes/dw4i2.json --executable $1 --promotion-panel $PANEL3 2>&1 | tee -a ~/pvz-logs/qc.out"; }'


def test_deployment_shell_and_embedded_python_parse_without_running():
    subprocess.run(['bash', '-n', str(SCRIPT)], check=True)
    code = SCRIPT.read_text().split("<<'PYCODE'\n", 1)[1].split('\nPYCODE', 1)[0]
    ast.parse(code)


def test_deploy14_start_command_engine_and_source_panels_are_pinned():
    text = SCRIPT.read_text()
    assert re.search(r'^start_qc\(\).*$', text, re.M).group() == EXPECTED_START
    assert 'NEWEXE=$W/build5/pvz-portable' in text
    assert 'PANEL3=$W/artifacts/task73c/panel_v3_source.json' in text
    for digest in ('76ff36dc026f67641aefccd5f30f6cdae65eee14ee197dc24497413123a7874a',
                   '68b37b98d3e29e83ac583fa5897372e7ed248213ec633d79edcf33e01eae26c3',
                   'bb0a9129b88023fe8c7722f267f1d184883ddea538860354421f6664d8e2f636'):
        assert digest in text
    assert re.search(r'^REV=[0-9a-f]{40}$', text, re.M)


def test_code_overlay_and_rollback_never_replace_data_or_seed_ledgers():
    text = SCRIPT.read_text()
    files = re.search(r'CODE_FILES=\((.*?)\)', text, re.S).group(1).split()
    baseline = json.loads(re.search(r"baseline = json.loads\(r'''(.*?)'''\)", text, re.S).group(1))
    assert set(files) == set(baseline)
    assert len(files) == 7
    assert all(name.startswith(('python/', 'scripts/')) or name == 'configs/student_selection_v2.json' for name in files)
    assert 'restore --source "$REV" --worktree -- "${CODE_FILES[@]}"' in text
    assert 'tmux kill-session' not in text
    assert 'git -C "$W" checkout' not in text
    restore = text.split("elif action == 'restore':", 1)[1].split("elif action == 'installed':", 1)[0]
    assert 'for name, expected in baseline.items()' in restore
    assert 'critical' not in restore


def test_success_requires_dry_run_and_unchanged_state_before_restart():
    text = SCRIPT.read_text()
    dry = text.index('timeout 300 env PVZ_NO_POPUP=1')
    unchanged = text.index('\ncode_state unchanged', dry)
    restart = text.index('\nstart_qc "$NEWEXE"', unchanged)
    assert dry < unchanged < restart
    assert "|| rollback 'dry-run failed'" in text
    assert "|| rollback 'selection preview missing'" in text
    assert '[ "$n1" -gt "$n0" ]' in text  # cannot accept a pre-deployment rc=0
