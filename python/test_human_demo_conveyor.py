"""End-to-end scripted recording and replay for original conveyor levels."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from human_demo import Recorder, verify
from pvz_env import TaskSpec
from pvz_event_env import EventWaitEnv
from pvz_tasks import profile_for_deck

RESOURCE = Path('/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN')
EXECUTABLE = Path('/tmp/pvz-boss50-build/pvz-portable')
def record_and_verify(level: int, seed: int, task_id: str, archive: Path) -> dict:
    task = TaskSpec(level=level, seed=seed, playthrough=2, profile=profile_for_deck(()),
                    zombie_count_multiplier=1.0, conveyor_start='original')
    with EventWaitEnv(RESOURCE, executable=EXECUTABLE, headless=False) as env:
        recorder = Recorder(env, archive, task_id, [], task, 'task85c conveyor e2e')
        if level == 30:
            recorder.step({'type': 'wait', 'ticks': 300})
        env.demo_click_packet(0)
        event = env.human_input()
        assert event['action']['type'] == 'wait', event
        recorder.step(event['action'])
        env.demo_click_cell(0, 0)
        event = env.human_input()
        assert event['action']['type'] == 'plant', event
        recorder.step(event['action'])
        assert recorder.observation['plants'], recorder.observation
        recorder.close()
    report = verify(archive, RESOURCE, EXECUTABLE)
    assert report['verified'], report
    meta = json.loads((archive / 'meta.json').read_text())
    assert meta['task']['deck'] == []
    assert meta['environment_task']['conveyor_start'] == 'original'
    assert meta['outcome']['final_tick'] >= 300 if level == 30 else meta['outcome']['final_tick'] >= 0
    return report


def test_original_conveyor_human_record_replay():
    with tempfile.TemporaryDirectory(prefix='pvz-task85c-') as tmp:
        root = Path(tmp)
        reports = [
            record_and_verify(50, 20261050, 'demo_boss_50_orig_m100', root / 'l50'),
            record_and_verify(30, 20261030, 'demo_pool_30_orig_m100', root / 'l30'),
        ]
        assert all(report['verified'] for report in reports)
