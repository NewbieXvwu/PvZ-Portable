"""Execute the actual shell watchdog against private tmux/pgrep/date fakes."""
import fcntl
import os
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/qc_watchdog.sh'


@pytest.fixture
def watchdog(tmp_path):
    tools = tmp_path / 'bin'
    tools.mkdir()
    (tools / 'tmux').write_text('''#!/bin/bash
printf '%s\\n' "$*" >> "$WATCHDOG_FIXTURE/tmux.calls"
case "$1" in
 has-session) test -f "$WATCHDOG_FIXTURE/alive";;
 new-session)
  if [[ -e /proc/$$/fd/9 ]]; then echo LOCK_LEAK >&2; exit 99; fi
  test ! -f "$WATCHDOG_FIXTURE/start_fails" || exit 1
  touch "$WATCHDOG_FIXTURE/alive";;
 *) exit 99;;
esac
''')
    (tools / 'pgrep').write_text('''#!/bin/bash
test -f "$WATCHDOG_FIXTURE/deploying"
''')
    (tools / 'date').write_text('''#!/bin/bash
if [[ "$1" == '+%s' ]]; then cat "$WATCHDOG_FIXTURE/now"; else /usr/bin/date "$@"; fi
''')
    for tool in tools.iterdir():
        tool.chmod(0o755)
    now = int(time.time())
    (tmp_path / 'now').write_text(str(now))
    env = dict(os.environ, WATCHDOG_FIXTURE=str(tmp_path), PATH=f'{tools}:{os.environ["PATH"]}')
    def run(function='watchdog_tick', *params):
        return subprocess.run(['bash', '-c', 'source "$1"; LOG_DIR=$2; shift 2; "$@"',
                               'test', str(SCRIPT), str(tmp_path), function, *map(str, params)],
                              env=env, text=True, capture_output=True, timeout=10)
    def deployment(name='qc_deploy_new.sh', tag='new', modified=now):
        p = tmp_path / name
        p.write_text(f'''#!/bin/bash
R=/safe/repo; W=$R/.claude/worktrees/cq; PY=/python
NEWEXE=$W/build5/pvz-portable
PANEL3=$W/panel.json
O=/unneeded/$(date +%s)
start_qc() {{
 tmux new-session -d -s qc "env PVZ_NO_POPUP=1 $PY $W/scripts/curriculum_queue.py --tag {tag} --executable $1 --promotion-panel $PANEL3"
}}
touch {tmp_path}/DEPLOYMENT_MAIN_RAN
start_qc "$NEWEXE"
''')
        os.utime(p, (modified, modified))
        return p
    deployment()
    return tmp_path, now, run, deployment


def new_sessions(root):
    path = root / 'tmux.calls'
    return [line for line in path.read_text().splitlines() if line.startswith('new-session')] if path.exists() else []


def test_dead_qc_uses_latest_function_only_and_archives_last_50(watchdog):
    root, now, run, deploy = watchdog
    deploy('qc_deploy_older.sh', 'old', now - 500)
    (root / 'qc.out').write_text(''.join(f'line {i}\n' for i in range(80)))
    result = run()
    assert result.returncode == 0, result.stderr
    assert len(new_sessions(root)) == 1 and '--tag new' in new_sessions(root)[0]
    assert '/safe/repo/.claude/worktrees/cq/build5/pvz-portable' in new_sessions(root)[0]
    assert not (root / 'DEPLOYMENT_MAIN_RAN').exists()
    assert next(root.glob('qc_crash_*.out')).read_text().splitlines() == [f'line {i}' for i in range(30, 80)]
    assert 'start_qc' in (root / 'qc_setup.log').read_text()
    assert 'LOCK_LEAK' not in (root / 'qc_setup.log').read_text()
    assert not list(root.glob('qc_watchdog_start_*.sh'))


@pytest.mark.parametrize('blocker', ['stop_curr', 'deploying'])
def test_stop_or_deployment_suppresses_restart(watchdog, blocker):
    root, _, run, _ = watchdog
    (root / blocker).touch()
    assert run().returncode == 0
    assert not new_sessions(root)
    assert not (root / 'qc_watchdog.restarts').exists()


def test_two_restarts_then_dead_session_latches_halt(watchdog):
    root, now, run, _ = watchdog
    for _ in range(2):
        assert run().returncode == 0
        (root / 'alive').unlink()
    assert run().returncode == 0
    assert len(new_sessions(root)) == 2
    assert (root / 'qc_watchdog.halted').exists()
    assert 'CRITICAL HALTED' in (root / 'qc_setup.log').read_text()
    (root / 'now').write_text(str(now + 3600))
    run()
    assert len(new_sessions(root)) == 2  # Manual intervention, not an automatic crash loop.


def test_two_launch_failures_also_latch_halt(watchdog):
    root, _, run, _ = watchdog
    (root / 'start_fails').touch()
    run()
    run()
    run()
    assert len(new_sessions(root)) == 2
    assert (root / 'qc_watchdog.halted').exists()


def test_expired_restart_attempt_does_not_spend_current_window(watchdog):
    root, now, run, _ = watchdog
    (root / 'qc_watchdog.restarts').write_text(f'{now - 1801}\n{now - 1801}\n')
    run()
    assert len(new_sessions(root)) == 1


def test_corrupt_budget_refuses_restart(watchdog):
    root, _, run, _ = watchdog
    (root / 'qc_watchdog.restarts').write_text('corrupt\n')
    run()
    assert not new_sessions(root)
    assert 'restart refused' in (root / 'qc_setup.log').read_text()


def test_stale_live_log_warns_without_restart_and_touch_is_not_progress(watchdog):
    root, now, run, _ = watchdog
    (root / 'alive').touch()
    log = root / 'curriculum.log'
    log.write_text('first line\n')
    os.utime(log, (now - 1900, now - 1900))
    run()
    assert 'no new line' in (root / 'qc_setup.log').read_text()
    (root / 'qc_setup.log').write_text('')
    log.write_text('first line\nsecond line\n')
    run()
    assert not (root / 'qc_setup.log').read_text()
    (root / 'now').write_text(str(now + 1900))
    os.utime(log, (now + 1900, now + 1900))
    run()
    assert 'no new line' in (root / 'qc_setup.log').read_text()
    assert not new_sessions(root)


def test_bad_latest_deployment_does_not_fall_back_to_older(watchdog):
    root, now, run, deploy = watchdog
    deploy('qc_deploy_unknown.sh', modified=now + 5).write_text('touch /DO_NOT_RUN\n')
    run()
    assert not new_sessions(root)
    assert 'cannot extract start_qc' in (root / 'qc_setup.log').read_text()


@pytest.mark.parametrize('calling_convention', ['alternate_variable', 'no_argument'])
def test_actual_start_invocation_is_preserved(watchdog, calling_convention):
    root, _, run, _ = watchdog
    p = root / 'qc_deploy_new.sh'
    text = p.read_text()
    if calling_convention == 'alternate_variable':
        text = text.replace('NEWEXE', 'EXE')
    else:
        text = text.replace('--executable $1', '--executable $NEWEXE')
        text = text.replace('start_qc "$NEWEXE"', 'start_qc')
    p.write_text(text)
    result = run()
    assert result.returncode == 0, result.stderr
    assert len(new_sessions(root)) == 1
    assert '/safe/repo/.claude/worktrees/cq/build5/pvz-portable' in new_sessions(root)[0]


def test_second_watchdog_exits_under_existing_lock(watchdog):
    root, _, run, _ = watchdog
    with (root / 'qc_watchdog.lock').open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert run('watchdog_main').returncode == 0
    assert not new_sessions(root)


def test_extract_current_real_deploy_without_executing_it(watchdog):
    root, _, run, _ = watchdog
    paths = list(Path('/home/newbiexvwu/pvz-logs').glob('qc_deploy*.sh'))
    if not paths:
        pytest.skip('desktop deployment scripts are local, not in git')
    latest = max(paths, key=lambda p: p.stat().st_mtime_ns)
    result = run('watchdog_extract_start', latest)
    assert result.returncode == 0, result.stderr
    assert 'tmux new-session' in result.stdout and 'start_qc "$NEWEXE"' in result.stdout
    assert 'stop_qc' not in result.stdout and 'checkout' not in result.stdout
    subprocess.run(['bash', '-n'], input=result.stdout, text=True, check=True)
    assert not new_sessions(root)
