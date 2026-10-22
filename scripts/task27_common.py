"""Isolated task27 execution, evidence and small explicit git checkpoints."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path('/home/newbiexvwu/PvZ-search/.claude/worktrees/dw5')
MAIN = Path('/home/newbiexvwu/PvZ-search')
OUT = ROOT/'artifacts/task27'
PYTHON = '/home/newbiexvwu/.venvs/ml/bin/python3'
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'python')]
import dagger_experiment_runner as guard
from run_identity_validation import descendants, trainers

guard.OUT = OUT
guard.descendants = descendants
guard.ENV = {**os.environ, 'PYTHONPATH':str(ROOT/'python')+':'+str(ROOT/'scripts'),
             'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'MALLOC_ARENA_MAX':'2',
             'CUDA_VISIBLE_DEVICES':'', 'PVZ_EXECUTABLE':str(ROOT/'build/pvz-portable'),
             'PVZ_RESOURCE_DIR':'/home/newbiexvwu/.cache/pvz-research-resources'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2)+'\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args, **options):
    return subprocess.check_output(['git','-C',str(ROOT),*args], **options)


def record(message, *documents, merge=False):
    assert git('branch','--show-current').strip() == b'dw5'
    staged = git('diff','--cached','--name-only').decode().splitlines()
    assert merge or not staged, staged
    today = datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
    book = ROOT/'MEMORY/docs/task27-dw5i.md'
    before, after = book.read_text().split('\n## 问题', 1)
    book.write_text(before+'\n- '+today+'，'+message+'\n\n## 问题'+after)
    daily = ROOT/f'MEMORY/{today}.md'
    with daily.open('a') as stream:
        stream.write('- task27：'+message+'\n')
    paths = [book, daily, *documents]
    for path in paths:
        path = Path(path)
        assert path.is_absolute() and path.is_relative_to(ROOT)
        assert path.suffix not in ('.pt','.npz','.log')
        assert path.stat().st_size < (100_000 if path.suffix == '.json' else 1_000_000)
    git('add', *map(str, paths))
    names = git('diff','--cached','--name-only').decode().splitlines()
    assert all(Path(name).suffix not in ('.pt','.npz','.log') for name in names)
    stats = git('diff','--cached','--numstat').decode()
    assert all(line.split('\t')[0] != '-' for line in stats.splitlines())
    for name in names:
        if name.endswith('.json'):
            assert len(git('show', ':'+name)) < 100_000, name
    print(stats + git('diff','--cached','--stat').decode(), flush=True)
    tracked = git('ls-files','-z','artifacts','logs')
    result = subprocess.run(['git','-C',str(ROOT),'check-ignore','-z','--stdin','--no-index'],
                            input=tracked, stdout=subprocess.PIPE)
    assert result.returncode in (0,1) and not result.stdout
    subprocess.run(['git','-C',str(ROOT),'commit','-m','task27: '+message.split('：')[0][:90]],check=True)
    subprocess.run(['git','-C',str(ROOT),'push','origin','dw5'],check=True)


def run_stage(command, tag, **options):
    OUT.mkdir(parents=True, exist_ok=True)
    result = OUT/(tag+'.result.json')
    if result.exists():
        assert read(result)['returncode'] == 0, 'inspect and preserve failed attempt: '+tag
        assert read(OUT/(tag+'.command.json'))['command'] == command
        return read(result)
    pidfile = OUT/(tag+'.process.json')
    assert not pidfile.exists() or not guard.live(read(pidfile)['pid']), 'stage already live: '+tag
    result = guard.run(command, tag, cwd=ROOT, **options)
    assert result['returncode'] == 0, result
    return result
