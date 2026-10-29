"""task150: exercise the prepared launcher in temporary directories with a fake tmux."""
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('active,legacy,expected', [
    (True, '14i', 'build16'),
    (False, '14i', 'build6'),
    (False, '15', 'build7'),
])
def test_deploy16_watchdog_launch_selects_active_package_and_preserves_deckcells(tmp_path, active, legacy, expected):
    result = subprocess.run(['bash', '-c', 'source "$1"; watchdog_extract_start "$2"', 'test',
        str(ROOT / 'scripts/qc_watchdog.sh'), str(ROOT / 'scripts/qc_deploy16.sh')],
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert 'stop_qc' not in result.stdout
    script = tmp_path / 'launch.sh'
    script.write_text(result.stdout.replace('/home/newbiexvwu', str(tmp_path)))
    logs = tmp_path / 'pvz-logs'
    logs.mkdir()
    (logs / 'qc_deploy16.mode').write_text('v2timer\n')
    if active:
        (logs / 'qc_deploy16.active').write_text('installed\n')
    (logs / f'qc_deploy{legacy}.mode').write_text('v2sleep\n' if legacy == '15' else 'gatev5\n')
    result = subprocess.run(['bash', '-c',
        'tmux() { printf "%s\\n" "$*"; }; source "$1"', 'test', str(script)],
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    call = result.stdout
    assert f'/{expected}/pvz-portable' in call
    assert call.count('--extra-manifest') == 1
    assert 'task137/training_panel.json' in call
    assert '--gate-version 5' in call and '--training-parallelism 1' in call
    if active:
        assert 'dw4i2_v2timer.json' in call and 'dw4i2_rebalanced_p05_v2timer.json' in call
    else:
        assert 'v2timer.json' not in call


def test_deploy16_stop_tolerates_a_process_exiting_before_kill(tmp_path):
    script = tmp_path / 'functions.sh'
    script.write_text((ROOT / 'scripts/qc_deploy16.sh').read_text()
        .split('# --- switch:', 1)[0].replace('/home/newbiexvwu', str(tmp_path)))
    result = subprocess.run(['bash', '-c', '''
source "$1"
tmux() { if [ "$1" = has-session ]; then return 0; fi; return 1; }
pgrep() { printf '12345\n'; }
kill() { return 1; }
sleep() { return 0; }
stop_qc
''', 'test', str(script)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
