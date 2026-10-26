"""Exercise deployment launch/rollback with shell fakes; never execute the package main."""
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/qc_deploy14f.sh'


@pytest.fixture
def package(tmp_path):
    # Only load variable/function definitions. Every path is private and tmux is a fake.
    source = SCRIPT.read_text().split('# --- preflight:', 1)[0]
    source = source.replace('/home/newbiexvwu', str(tmp_path))
    p = tmp_path / 'functions.sh'
    p.write_text(source)
    (tmp_path / 'pvz-logs').mkdir()
    def run(commands):
        return subprocess.run(['bash', '-c', '''
source "$1"
tmux() { if [ "$1" = has-session ]; then return 1; fi; printf '%s\\n' "$*" >> "$ENGINE_CHOICE.calls"; }
code_state() { printf '%s\\n' "$1" >> "$ENGINE_CHOICE.actions"; }
''' + commands, 'test', str(p)], text=True, capture_output=True, timeout=10)
    return tmp_path / 'pvz-logs/qc_deploy14f.engine', run


def test_draft_watchdog_falls_back_to_build5_then_tracks_activation_and_rollback(package):
    choice, run = package
    result = run('''
start_qc "$NEWEXE"
choose_engine "$NEWEXE"
start_qc "$NEWEXE"
choose_engine "$OLDEXE"
start_qc "$NEWEXE"
''')
    assert result.returncode == 0, result.stderr
    calls = choice.with_suffix('.engine.calls').read_text().splitlines()
    assert len(calls) == 3
    assert '--executable ' in calls[0] and '/build5/pvz-portable' in calls[0]
    assert '/build6/pvz-portable' in calls[1]
    assert '/build5/pvz-portable' in calls[2]
    assert choice.read_text().strip().endswith('/build5/pvz-portable')


@pytest.mark.parametrize('started,actions', [(0, ['restore', 'restored']), (1, ['installed'])])
def test_rollback_keeps_renewal_code_after_live_collection_and_selects_build5(package, started, actions):
    choice, run = package
    result = run(f'SAVED=1; STARTED={started}; rollback injected')
    assert result.returncode == 1, result.stderr
    assert choice.with_suffix('.engine.actions').read_text().splitlines() == actions
    assert choice.read_text().strip().endswith('/build5/pvz-portable')
    assert '/build5/pvz-portable' in choice.with_suffix('.engine.calls').read_text()


@pytest.mark.parametrize('name,choice', [('qc_deploy14f.sh', 'ENGINE_CHOICE'),
                                        ('qc_deploy14g.sh', 'MODE_CHOICE')])
def test_watchdog_can_extract_the_new_launcher_without_running_the_deployment(name, choice):
    result = subprocess.run(['bash', '-c', 'source "$1"; watchdog_extract_start "$2"',
                             'test', str(ROOT / 'scripts/qc_watchdog.sh'), str(ROOT / 'scripts' / name)],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert f'{choice}=' in result.stdout and 'OLDEXE=' in result.stdout
    assert 'code_state' not in result.stdout and 'choose_engine' not in result.stdout
    subprocess.run(['bash', '-n'], input=result.stdout, text=True, check=True)


def test_rebalance_launcher_draft_activation_and_live_rollback(tmp_path):
    source = (ROOT / 'scripts/qc_deploy14g.sh').read_text().split('# --- preflight:', 1)[0]
    source = source.replace('/home/newbiexvwu', str(tmp_path))
    script = tmp_path / 'functions.sh'
    script.write_text(source)
    logs = tmp_path / 'pvz-logs'
    logs.mkdir()
    result = subprocess.run(['bash', '-c', '''
source "$1"
tmux() { if [ "$1" = has-session ]; then return 1; fi; printf '%s\n' "$*" >> "$MODE_CHOICE.calls"; }
code_state() { printf '%s\n' "$1" >> "$MODE_CHOICE.actions"; }
start_qc
choose_mode rebalance
start_qc
SAVED=1; STARTED=1; rollback injected
''', 'test', str(script)], text=True, capture_output=True, timeout=10)
    assert result.returncode == 1, result.stderr
    calls = (logs / 'qc_deploy14g.mode.calls').read_text().splitlines()
    assert len(calls) == 3
    assert '--alternate-recipe' not in calls[0] and '--opening-manifest' not in calls[0]
    assert '--alternate-recipe' in calls[1] and '--opening-manifest' in calls[1]
    assert '--alternate-recipe' not in calls[2] and '--opening-manifest' not in calls[2]
    assert all('/build6/pvz-portable' in call for call in calls)
    assert (logs / 'qc_deploy14g.mode').read_text().strip() == 'baseline'
    assert (logs / 'qc_deploy14g.mode.actions').read_text().splitlines() == ['restore', 'restored']
