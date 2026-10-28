"""task131: prepared deploy15 preserves prior launches and restores its overlay."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import qc_deploy15_state as state


@pytest.mark.parametrize('service', ['g', 'h', 'i'])
def test_deploy15_preserves_prior_launch_and_rollback(tmp_path, service):
    logs = tmp_path / 'pvz-logs'
    logs.mkdir()
    if service in ('h', 'i'):
        (logs / 'qc_deploy14h.mode').write_text('rebalance\n')
    if service == 'i':
        (logs / 'qc_deploy14i.mode').write_text('gatev5\n')
    for version in ('14i', '15'):
        text = (ROOT / f'scripts/qc_deploy{version}.sh').read_text().split('# --- preflight:', 1)[0]
        functions = tmp_path / (version + '.sh')
        functions.write_text(text.replace('/home/newbiexvwu', str(tmp_path)))
    runner = '''
source "$1"
tmux() { if [ "$1" = has-session ]; then return 1; fi; printf '%s\\n' "$*" >> "$LOG_CALLS"; }
start_qc
'''
    expected = tmp_path / 'expected'
    result = subprocess.run(['bash', '-c', runner, 'test', str(tmp_path / '14i.sh')],
                            env={**os.environ, 'LOG_CALLS': str(expected)},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    observed = tmp_path / 'observed'
    result = subprocess.run(['bash', '-c', runner + '''
printf 'v2sleep\\n' > "$MODE15"
start_qc
PREVIOUS=$(service); SAVED=0
code_state() { :; }
rollback injected
''', 'test', str(tmp_path / '15.sh')],
        env={**os.environ, 'LOG_CALLS': str(observed)},
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 1, result.stderr
    calls = observed.read_text().splitlines()
    assert calls[0] == calls[2] == expected.read_text().strip()
    assert 'build7/pvz-portable' in calls[1]
    assert 'dw4i2_v2sleep.json' in calls[1] and 'dw4i2_rebalanced_p05_v2sleep.json' in calls[1]
    assert '--gate-version 5' in calls[1] and '--training-parallelism 1' in calls[1]
    assert not (logs / 'qc_deploy15.mode').exists()


@pytest.mark.parametrize('service', ['g', 'i'])
def test_deploy15_baseline_checks_and_restores_code(tmp_path, service):
    root, package, output = (tmp_path / name for name in ('cq', 'package', 'backup'))
    for folder in (root / 'build6', package / 'build7', output):
        folder.mkdir(parents=True)
    (root / 'build6/pvz-portable').write_bytes(b'old engine')
    (package / 'build7/pvz-portable').write_bytes(b'new engine')
    (root / 'code.py').write_text('old ' + service)
    original = state.sha(root / 'code.py')
    baselines = tmp_path / 'baselines.json'
    baselines.write_text(json.dumps(dict(old_engine_sha256=state.sha(root / 'build6/pvz-portable'),
        new_engine_sha256=state.sha(package / 'build7/pvz-portable'),
        baselines={service: {'code.py': original, 'new.json': None}})))
    args = SimpleNamespace(root=root, package=package, output=output, baselines=baselines,
                           service=service, revision='unused', action='preflight')
    state.run(args)
    args.action = 'save'
    state.run(args)
    (root / 'code.py').write_text('candidate')
    (root / 'new.json').write_text('{}')
    args.action = 'preflight'
    with pytest.raises(ValueError, match='baseline changed'):
        state.run(args)
    args.action = 'restore'
    state.run(args)
    args.action = 'restored'
    state.run(args)
    assert state.sha(root / 'code.py') == original and not (root / 'new.json').exists()


def test_deploy15_watchdog_extracts_only_the_launch_function():
    result = subprocess.run(['bash', '-c', 'source "$1"; watchdog_extract_start "$2"', 'test',
                             str(ROOT / 'scripts/qc_watchdog.sh'), str(ROOT / 'scripts/qc_deploy15.sh')],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    subprocess.run(['bash', '-n'], input=result.stdout, text=True, check=True)
    assert 'qc_deploy15_state.py' not in result.stdout
