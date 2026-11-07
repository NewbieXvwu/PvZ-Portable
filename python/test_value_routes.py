"""task252: coupled leaf settings and their production/Mac transport, without games."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]

import curriculum_queue as queue
import mac_collect
import qc_launch
import search_play

ROUTES = {'BACKGROUND_5_ROOF': {'value': 'finetune-t3.pt', 'margin_relative': .02}}


@pytest.mark.parametrize('level,expected', [
    (9, ('production.pt', 0)), (23, ('production.pt', 0)),
    (33, ('production.pt', 0)), (35, ('production.pt', 0)),
    (47, ('finetune-t3.pt', .02)), (50, ('roof.pt', 0)),
])
def test_value_routes_couple_checkpoint_and_margin(level, expected):
    routes = search_play.parse_value_routes(json.dumps(ROUTES))
    assert search_play._leaf_route('production.pt', 'roof.pt', {'level': level}, routes) == expected


def test_value_routes_require_the_models_margin_and_keep_legacy_callers():
    with pytest.raises(argparse.ArgumentTypeError):
        search_play.parse_value_routes('{"BACKGROUND_5_ROOF": {"value": "finetune-t3.pt"}}')
    assert search_play._leaf_value('production.pt', 'roof.pt', {'level': 47}) == 'roof.pt'
    assert search_play._leaf_value('production.pt', 'roof.pt', {'level': 33}) == 'production.pt'


def test_value_routes_queue_transport(tmp_path):
    args = SimpleNamespace(python=sys.executable, rollout_policy='student.pt',
        rollout_policy_gpt='gptbc.pt', static_value='static.pt', value='production.pt',
        value_roof='roof.pt', value_routes=ROUTES, executable='engine', resource_dir=tmp_path,
        frames=tmp_path / 'frames', out=tmp_path / 'out', workers=3)
    command = queue.command_for(args, tmp_path / 'panel.json', 'roof', 0, 32, 'test')
    assert json.loads(command[command.index('--value-routes') + 1]) == ROUTES
    assert command[command.index('--static-value') + 1] == 'static.pt'
    assert command[command.index('--value-roof') + 1] == 'roof.pt'


def test_value_routes_launch_default_and_rollback(tmp_path):
    args = SimpleNamespace(repo=tmp_path, worktree=None, python=None, logs=tmp_path / 'logs',
        review_share=None, rounds=1, workers=3, recipe=None, alternate_recipe=None,
        b_init=None, b_initial_epochs=None, b_epochs=None, a_restart=True,
        b_from_incumbent=True, executable=None, value_routes=None)
    command = qc_launch.command(args)
    routes = json.loads(command[command.index('--value-routes') + 1])
    assert routes == {'BACKGROUND_5_ROOF': dict(
        value=str(tmp_path / 'artifacts/value_net/task252_finetune_t3/model.pt'), margin_relative=.02)}
    args.value_routes = {}
    command = qc_launch.command(args)
    assert json.loads(command[command.index('--value-routes') + 1]) == {}


def test_value_routes_mac_download_retains_margin_and_remote_source(tmp_path, monkeypatch):
    args = SimpleNamespace(cache=tmp_path, repo='test-repo')
    state = dict(student='/remote/student.pt', value='/remote/value.pt',
        static='/remote/static.pt', rollout_policy_gpt='/remote/gptbc.pt',
        value_roof='/remote/roof.pt', value_routes={
            'BACKGROUND_5_ROOF': dict(value='/remote/finetune-t3.pt', margin_relative=.02)})
    exported = {}

    def export(args, verb, payload):
        assert verb == 'export'
        exported.update(payload['files'])
        for name, source in payload['files'].items():
            target = args.cache / 'download' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.encode())

    monkeypatch.setattr(mac_collect, 'remote', export)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        HfApi=lambda: SimpleNamespace(create_commit=lambda **kwargs: None),
        CommitOperationDelete=lambda **kwargs: kwargs,
        hf_hub_download=lambda **kwargs: str(tmp_path / 'download' / Path(kwargs['filename']).name)))
    directory, student, routes = mac_collect.inputs(args, state, {}, 'test')
    local_value, margin = search_play._leaf_route('value.pt', 'roof.pt', {'level': 47}, routes)
    assert Path(local_value).parent == directory
    assert Path(local_value).read_bytes() == b'/remote/finetune-t3.pt'
    assert student.read_bytes() == b'/remote/student.pt'
    assert margin == .02
    assert '/remote/finetune-t3.pt' in exported.values()
    assert state['value_routes']['BACKGROUND_5_ROOF']['value'] == '/remote/finetune-t3.pt'
