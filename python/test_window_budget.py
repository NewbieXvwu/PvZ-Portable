import json
from pathlib import Path
from argparse import Namespace
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))


def test_uniform_budget_reduces_epochs_then_same_rows():
    from decision_window_budget import choose_budget, merged, subtract
    assert merged([(0,10),(5,20),(30,40)],0,35)==[[0,20],[30,35]]
    assert subtract([[0,20]],[[4,8],[12,16]])==[[0,4],[8,12],[16,20]]
    budget = choose_budget({str(i):10000 for i in range(8)},20_000_000,.25)
    assert budget['epochs']==3 and budget['train_row_fraction']==1
    assert budget['planned_seed1_days'] <= 3
    slow = choose_budget({str(i):100 for i in range(8)},20_000_000,.25)
    assert slow['epochs']==1 and 0 < slow['train_row_fraction'] < 1
    assert slow['planned_seed1_days'] <= 3
    assert slow['planned_all_gpu_hours']==2*slow['planned_seed1_gpu_hours']


def test_seed2_barrier_waits_for_all_seed1_even_when_dependencies_pending(tmp_path):
    from gpu_window_runner import Runner
    queue = tmp_path/'queue'; queue.mkdir()
    for name, after, deps, priority in [('s1',[],[str(tmp_path/'ready')],10),('s2',['s1'],[],110)]:
        (queue/(name+'.json')).write_text(json.dumps(dict(id=name,out=str(tmp_path/name),recipe=str(tmp_path/'recipe'),
            host_gib=1,gpu_gib=1,command=['python'],after_jobs=after,depends_on=deps,priority=priority)))
    runner = Runner(Namespace(queue=queue,production_log=tmp_path/'prod',memory_max_gib=5))
    assert list(runner.candidates())==[]
    (tmp_path/'ready').touch()
    assert [j['id'] for j,s in runner.candidates()]==['s1']
    (tmp_path/'s1').mkdir()
    (tmp_path/'s1/window.json').write_text(json.dumps(dict(status='complete',seconds=1,attempts=[])))
    assert [j['id'] for j,s in runner.candidates()]==['s2']


def test_disk_bounded_token_cache_streams_uncached_validation(tmp_path):
    import numpy as np
    import sqlite3
    from pvz_frame_store import write_frames
    from pvz_student_tokens import frame_tokens
    from pvz_research_data import SparseTokens, keep_training_rows
    from prepare_decision_data import tokens
    source='seed_1/example/frames.npz'
    frame=dict(sun=100,wave=0,tick=0,scene={},lanes=list(range(5)),plants=[],zombies=[],mowers=[],packets=[])
    frames=[dict(frame,tick=i*60) for i in range(4)]
    path=tmp_path/'split/raw'/source; path.parent.mkdir(parents=True)
    write_frames(path,frames)
    db=sqlite3.connect(tmp_path/'split/sources.sqlite')
    db.execute('CREATE TABLE sources (source TEXT,rows BLOB,n INTEGER)')
    db.execute('INSERT INTO sources VALUES (?,?,?)',(source,np.array([1,3],np.int32).tobytes(),2)); db.commit(); db.close()
    np.save(tmp_path/'split/unseen.npy',np.array([False,True]))
    budget=tmp_path/'budget.json'; budget.write_text(json.dumps(dict(train_row_fraction=1,sidecar_max_bytes=1<<30,preview_rows=1)))
    tokens(Namespace(sources=tmp_path/'split',out=tmp_path/'tokens',budget=budget),lambda:False)
    store=SparseTokens(tmp_path/'tokens')
    assert store.meta['stored_rows']==2 and store.meta['heldout_rows']==1
    path.unlink()  # Both train and common preview remain readable without original frames on Mac.
    for i,row in enumerate((1,3)):
        for key,value in frame_tokens(frames[row-1]).items():
            np.testing.assert_array_equal(store.sample(i)[key],value)
    indices=np.arange(10000)
    assert np.array_equal(keep_training_rows(indices,.1),np.r_[keep_training_rows(indices[:5000],.1),keep_training_rows(indices[5000:],.1)])


def test_measured_report_publishes_only_docs_in_disposable_checkout(tmp_path, monkeypatch):
    import subprocess
    import publish_window_budget as publisher
    remote=tmp_path/'remote'; workspace=tmp_path/'workspace'; workspace.mkdir()
    def git(*args,cwd=workspace):
        return subprocess.run(['git','-C',str(cwd),*args],capture_output=True,text=True,check=True).stdout
    subprocess.run(['git','init','--bare',str(remote)],capture_output=True,check=True)
    git('init','-b','gpuwindow')
    git('config','user.name','Window fixture'); git('config','user.email','window@example.invalid')
    task=workspace/'MEMORY/docs/task162-gpu-window-runner.md'; task.parent.mkdir(parents=True)
    task.write_text('# task162\n')
    git('add','MEMORY'); git('commit','-m','initial fixture'); git('remote','add','origin',str(remote))
    git('push','-u','origin','gpuwindow')
    state=tmp_path/'state'; probe=state/'decision-data/throughput'; probe.mkdir(parents=True)
    (probe/'report.json').write_text(json.dumps(dict(revision='fixture',budget=dict(epochs=1,train_row_fraction=.1,
        planned_seed1_days=2,sidecar_max_bytes=1<<30))))
    (probe/'report.md').write_text('\nMeasured fixture: 123 rows/s.\n')
    out=state/'decision-data/publication'
    monkeypatch.setattr(sys,'argv',['publish','--state',str(state),'--workspace',str(workspace),'--out',str(out)])
    publisher.main()
    receipt=json.loads((out/'ready.json').read_text())
    assert receipt['workspace_fast_forwarded']
    assert 'Measured fixture' in task.read_text()
    assert not (state/'publication/fixture').exists()
