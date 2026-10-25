"""Public observation history must survive demonstration takeover and export."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]
import task78_student_context as context
from test_task78_pipeline import label_fixture


@pytest.fixture
def demo(monkeypatch):
    prefix = [dict(tick=tick, wave=3, plants=[], zombies=[], action=dict(type='wait',ticks=60))
              for tick in (100, 200, 300, 400)]
    continuation = [dict(tick=300, wave=3, plants=[], zombies=[]),
                    dict(tick=360, wave=3, plants=[], zombies=[]),
                    dict(tick=420, wave=4, plants=[], zombies=[])]
    monkeypatch.setattr(context, 'frame_path', lambda path: path/'frames')
    monkeypatch.setattr(context, 'read_frames', lambda path:
                        prefix if str(path)=='/demo/frames' else continuation)
    monkeypatch.setattr(context, 'sha', lambda _: 'same')
    return dict(demo_archive='/demo', demo_decision=3, frames_path='/continuation',
                frames_sha256='same', roots=[dict(frame=1),dict(frame=2)]), prefix


def test_takeover_restores_previous_frame_action_and_full_wave_age(demo):
    row, prefix = demo
    state = context.initial_state(row, 'v2')
    assert state['frame'] == prefix[1]
    assert state['action'] == prefix[2]['action']
    assert state['history'].wave_tick == 100
    # The fourth human frame must not leak into the takeover prefix.
    assert state['history'].wave == 3
    assert context.initial_state(dict(demo_archive=None), 'v2') == {}


def test_missing_prefix_changes_current_wave_but_converges_after_next_wave(demo):
    row, _ = demo
    histories = context.expected_histories(row, 'v2')
    assert histories[1]['wave_tick'] == 100
    assert histories[2]['wave_tick'] == 420
    cold = context.expected_histories({**row, 'demo_archive':None}, 'v2')
    assert cold[1]['wave_tick'] == 300
    assert cold[2] == histories[2]
    with pytest.raises(ValueError, match='student history differs'):
        context.validate_root_history(row, 1, dict(history=dict(history=cold[1])))
    context.validate_root_history(row, 2, dict(history=dict(history=cold[2])))


@pytest.mark.parametrize('length', [0,5,1.5])
def test_invalid_takeover_length_is_rejected(demo, length):
    row, _ = demo
    with pytest.raises(ValueError, match='prefix length'):
        context.initial_state({**row, 'demo_decision':length}, 'v2')


def test_export_rejects_context_error_before_reading_terminal_labels(label_fixture, monkeypatch):
    # Reuse the complete physical-replay fixture: matching board hashes must not
    # let an observation-history error through the exporter.
    import task78_export as export
    row, replay, directory = label_fixture
    row['demo_archive'] = '/demo'
    complete = directory.parent/'complete.json'
    record = json.loads(complete.read_text())
    record['identity']['source'] = row
    complete.write_text(json.dumps(record))
    root = directory/'root.json'
    record = json.loads(root.read_text())
    record['history'] = dict(history=dict(obs='v2',wave=1,wave_tick=100,plants=[]))
    root.write_text(json.dumps(record))
    monkeypatch.setattr(export, 'expected_histories', lambda *_:
                        {3:dict(obs='v2',wave=1,wave_tick=0,plants=[])})
    with pytest.raises(ValueError, match='student history differs'):
        export.audit_labels(0,2500,row,Path('/sources.json'),replay)


def test_audit_reports_all_missing_jobs_instead_of_vacuous_success(demo, tmp_path):
    row, _ = demo
    row['group'] = 'fog_demo'
    manifest = tmp_path/'sources.json'
    manifest.write_text(json.dumps(dict(archives=[row])))
    report = context.audit(manifest, tmp_path/'labels')
    assert report['checked_roots'] == 6
    assert report['failed_roots'] == 6
    assert report['passed'] is False
