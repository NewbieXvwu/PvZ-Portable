"""task143: deploy15 starts the v2sleep curriculum on build7 on top of deploy14p."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _functions(tmp_path):
    text = (ROOT / 'scripts/qc_deploy15.sh').read_text().split('# --- preflight:', 1)[0]
    path = tmp_path / 'functions.sh'
    path.write_text(text.replace('/home/newbiexvwu', str(tmp_path)))
    return path


def _start(tmp_path, modes, pre):
    _functions(tmp_path)
    calls = tmp_path / 'calls'
    runner = f'''
{pre}
source "$1"
tmux() {{ if [ "$1" = has-session ]; then return 1; fi; printf '%s\\n' "$*" >> "$LOG_CALLS"; }}
MODE14P=$2; MODE15=$3; MODE14I=$4; MODE14H=$5; EXTRA=$6
{modes}
start_qc
'''
    result = subprocess.run(['bash', '-c', runner, 'test', str(tmp_path / 'functions.sh'),
                             str(tmp_path / '14p.mode'), str(tmp_path / '15.mode'),
                             str(tmp_path / '14i.mode'), str(tmp_path / '14h.mode'),
                             str(tmp_path / 'panel.json')],
                            env={'LOG_CALLS': str(calls), 'PATH': '/usr/bin:/bin'},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    return calls.read_text().splitlines()[0]


def test_deploy15_activation_uses_build7_v2sleep_and_deckcells(tmp_path):
    (tmp_path / '14p.mode').write_text('deckcells\n')
    (tmp_path / '15.mode').write_text('v2sleep\n')
    call = _start(tmp_path, '', '')
    assert 'build7/pvz-portable' in call
    assert 'dw4i2_v2sleep.json' in call and 'dw4i2_rebalanced_p05_v2sleep.json' in call
    assert '--gate-version 5' in call and '--training-parallelism 1' in call
    assert '--extra-manifest' in call


def test_deploy15_rollback_branch_keeps_build6_and_deckcells(tmp_path):
    (tmp_path / '14p.mode').write_text('deckcells\n')
    (tmp_path / '14i.mode').write_text('gatev5\n')
    call = _start(tmp_path, '', '')
    assert 'build6/pvz-portable' in call and 'build7/' not in call
    assert '--extra-manifest' in call and '--gate-version 5' in call


def test_deploy15_requires_deploy14p_and_always_passes_the_panel(tmp_path):
    (tmp_path / '14i.mode').write_text('gatev5\n')
    call = _start(tmp_path, '', '')
    assert call.count('--extra-manifest') == 1

    text = (ROOT / 'scripts/qc_deploy15.sh').read_text()
    assert 'deploy14p (deckcells) is not active' in text
    assert 'EXTRA=$W/artifacts/task137/training_panel.json' in text


def test_deploy15_watchdog_extracts_only_the_launch_function():
    result = subprocess.run(['bash', '-c', 'source "$1"; watchdog_extract_start "$2"', 'test',
                             str(ROOT / 'scripts/qc_watchdog.sh'), str(ROOT / 'scripts/qc_deploy15.sh')],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    subprocess.run(['bash', '-n'], input=result.stdout, text=True, check=True)
    assert 'start_qc' in result.stdout and 'stop_qc' not in result.stdout
