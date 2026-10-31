"""Execute the actual shell watchdog against private tmux/pgrep/date fakes."""
import fcntl
import os
import shutil
import subprocess
import time
import sys
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
  if [[ -e /dev/fd/9 ]]; then echo LOCK_LEAK >&2; exit 99; fi
  test ! -f "$WATCHDOG_FIXTURE/start_fails" || exit 1
  touch "$WATCHDOG_FIXTURE/alive";;
 *) exit 99;;
esac
''')
    (tools / 'pgrep').write_text('''#!/bin/bash
test -f "$WATCHDOG_FIXTURE/deploying"
''')
    (tools / 'date').write_text('''#!/bin/bash
if [[ "$1" == '+%s' ]]; then cat "$WATCHDOG_FIXTURE/now"; else "$WATCHDOG_REAL_DATE" "$@"; fi
''')
    for name in ('stat', 'mktemp'):
        replacement = shutil.which('g' + name)
        if replacement:
            (tools / name).symlink_to(replacement)
    for tool in tools.iterdir():
        tool.chmod(0o755)
    now = int(time.time())
    (tmp_path / 'now').write_text(str(now))
    env = dict(os.environ, WATCHDOG_FIXTURE=str(tmp_path), PATH=f'{tools}:{os.environ["PATH"]}',
               WATCHDOG_REAL_DATE=shutil.which('gdate') or shutil.which('date'))
    def run(function='watchdog_tick', *params):
        return subprocess.run(['bash', '-c',
            'source "$1"; LOG_DIR=$2; PYTHON=$3; QC_LAUNCH=$4; REPO=$2/repo; WORKTREE=$REPO/cq; shift 4; "$@"',
            'test', str(SCRIPT), str(tmp_path), sys.executable,
            str(SCRIPT.with_name('qc_launch.py')), function, *map(str, params)],
            env=env, text=True, capture_output=True, timeout=10)
    return tmp_path, now, run



def new_sessions(root):
    path = root / 'tmux.calls'
    return [line for line in path.read_text().splitlines() if line.startswith('new-session')] if path.exists() else []


def test_dead_qc_uses_shared_launcher_and_archives_last_50(watchdog):
    root, now, run = watchdog
    (root / 'qc.out').write_text(''.join(f'line {i}\n' for i in range(80)))
    result = run()
    assert result.returncode == 0, result.stderr
    assert len(new_sessions(root)) == 1
    assert '--obs-v3' in new_sessions(root)[0] and '--native-games' in new_sessions(root)[0]
    assert str(root / 'repo/cq/scripts/curriculum_queue.py') in new_sessions(root)[0]
    assert next(root.glob('qc_crash_*.out')).read_text().splitlines() == [f'line {i}' for i in range(30, 80)]
    assert 'qc_launch.py' in (root / 'qc_setup.log').read_text()
    assert 'LOCK_LEAK' not in (root / 'qc_setup.log').read_text()
    assert not list(root.glob('qc_watchdog_start_*.sh'))


@pytest.mark.parametrize('blocker', ['stop_curr', 'deploying'])
def test_stop_or_deployment_suppresses_restart(watchdog, blocker):
    root, _, run = watchdog
    (root / blocker).touch()
    assert run().returncode == 0
    assert not new_sessions(root)
    assert not (root / 'qc_watchdog.restarts').exists()


def test_two_restarts_then_dead_session_latches_halt(watchdog):
    root, now, run = watchdog
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
    root, _, run = watchdog
    (root / 'start_fails').touch()
    run()
    run()
    run()
    assert len(new_sessions(root)) == 2
    assert (root / 'qc_watchdog.halted').exists()


def test_expired_restart_attempt_does_not_spend_current_window(watchdog):
    root, now, run = watchdog
    (root / 'qc_watchdog.restarts').write_text(f'{now - 1801}\n{now - 1801}\n')
    run()
    assert len(new_sessions(root)) == 1


def test_corrupt_budget_refuses_restart(watchdog):
    root, _, run = watchdog
    (root / 'qc_watchdog.restarts').write_text('corrupt\n')
    run()
    assert not new_sessions(root)
    assert 'restart refused' in (root / 'qc_setup.log').read_text()


def test_stale_live_log_warns_without_restart_and_touch_is_not_progress(watchdog):
    root, now, run = watchdog
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






@pytest.mark.skipif(shutil.which('flock') is None, reason='Linux flock command is unavailable')
def test_second_watchdog_exits_under_existing_lock(watchdog):
    root, _, run = watchdog
    with (root / 'qc_watchdog.lock').open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert run('watchdog_main').returncode == 0
    assert not new_sessions(root)


