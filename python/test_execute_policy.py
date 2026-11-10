"""task264: execution identity and the opt-in focus boundary, without full games."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
import curriculum_queue as queue
import pvz_rollout_policy as policy
import search_play


@pytest.mark.parametrize('execute,expected', [(None, ['gpt.pt']), ('gpt.pt', ['gpt.pt']),
                                              ('student.pt', ['gpt.pt', 'student.pt'])])
def test_execute_policy_loads_each_identity_once(tmp_path, monkeypatch, execute, expected):
    loaded, exported = [], {}
    def load(path):
        loaded.append(path)
        return SimpleNamespace(obs='v2timer', identity=path)
    def export(net, path):
        exported[Path(path).name] = net.identity
    monkeypatch.setattr(policy, 'load_student', load)
    monkeypatch.setattr(policy, 'export_student', export)
    monkeypatch.setattr(search_play, '_W', {})
    search_play._load_nets('value.bin', 'gpt.pt', execute_path=execute)
    assert loaded == expected
    assert search_play._W['fast'].identity == (execute or 'gpt.pt')
    assert exported['student.bin'] == 'gpt.pt'
    assert search_play._W['split_policy'] == (execute == 'student.pt')
    if execute == 'student.pt':
        assert exported['execute.bin'] == 'student.pt'
    search_play._W['rollout_tmp'].cleanup()


@pytest.mark.parametrize('base,level,enabled,expected', [
    ('fog_good_l32_train', 32, True, 'student.pt'),
    ('fog_good_l34_train', 34, True, 'student.pt'),
    ('fog_good_l36_train', 36, True, 'student.pt'),
    ('roof_user_l47_user', 47, True, 'student.pt'),
    ('roof_user_l49_user', 49, True, 'student.pt'),
    ('fog_good_l34_train', 34, False, None),
    ('fog_other', 34, True, None), ('pool_user_l22_user', 22, True, None),
    ('boss', 50, True, None),
])
def test_execute_policy_focus_boundary(base, level, enabled, expected):
    import json
    focus = json.loads((ROOT / 'configs/curriculum_focus.json').read_text())
    focus['execute_incumbent'] = enabled
    args = SimpleNamespace(focus_config=focus, rollout_policy='student.pt')
    item = dict(base=base, task_definition=dict(level=level))
    assert queue.focus_execute_policy(args, item) == expected


def test_execute_policy_transport_keeps_gpt_rollout(tmp_path):
    args = SimpleNamespace(python=sys.executable, rollout_policy='student.pt',
        rollout_policy_gpt='gpt.pt', static_value='static.pt', value='value.pt',
        executable='engine', resource_dir=tmp_path, frames=tmp_path/'frames',
        out=tmp_path/'out', workers=4)
    default = queue.command_for(args, tmp_path/'panel.json', 'task', 0, 16, 'test')
    split = queue.command_for(args, tmp_path/'panel.json', 'task', 0, 16, 'test',
                              execute_policy='student.pt')
    index = split.index('--execute-policy')
    assert split[index:index+2] == ['--execute-policy', 'student.pt']
    assert split[:index] + split[index+2:] == default
    assert split[split.index('--rollout-policy-gpt')+1] == 'gpt.pt'
    assert search_play._terrain_policy('student.pt', 'gpt.pt', dict(level=34)) == 'gpt.pt'


@pytest.mark.parametrize('execute,expected', [('None', 'gpt.pt'), ('student.pt', 'student.pt')])
def test_execute_policy_group_records_routed_identity(tmp_path, monkeypatch, execute, expected):
    import json
    args = SimpleNamespace(state=tmp_path/'state.json', log=tmp_path/'queue.log',
        out=tmp_path/'out', workers=4, rollout_policy='student.pt',
        beta_threshold=.1, starved_beta=.5, probe_interval=3)
    state = dict(offsets={}, cells={'focus': dict(windows={})}, student_round=163)
    item = dict(base='focus', task_id='task', offset=0, games=1, label='test',
                multiplier=.5, manifest='panel.json', review_of='focus')
    def collect(*a, **kw):
        folder = args.out/'test_1'
        folder.mkdir(parents=True)
        (folder/'games.jsonl').write_text('{"won": false}\n')
        (folder/'config.json').write_text(json.dumps(dict(execute_policy=execute, rollout_policy='gpt.pt')))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(queue, 'mem_available_bytes', lambda: None)
    monkeypatch.setattr(queue, 'command_for', lambda *a, **kw: ['synthetic'])
    monkeypatch.setattr(queue.subprocess, 'run', collect)
    queue._run_group(args, state, item, 1, 0)
    assert item['execute_policy'] == expected
    assert item['rollout_policy'] == 'gpt.pt'
    assert item['student_round'] == 163
