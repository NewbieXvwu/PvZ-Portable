import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from human_demo import Recorder, export, replay, verify, record_user_pick
from pvz_env import TaskSpec
from pvz_event_env import EventWaitEnv
from pvz_frame_store import read_frames
from pvz_tasks import profile_for_deck

RESOURCE = Path(os.environ.get('PVZ_RESOURCE_DIR', str(Path.home() / '.cache/pvz-research-resources')))
BINARY = Path(os.environ.get('PVZ_EXECUTABLE', ROOT / 'build/pvz-portable'))


def test_record_and_demo_start_replay_without_archive_hashes(tmp_path):
    class DemoEnv:
        def begin_demo(self):
            pass

        def reset(self, *, deck, task):
            self.observation = dict(tick=0, wave=1, wave_count=1, result=0,
                                    terminal=False, sun=50, loadout_context={'zombie_roster': []})
            return self.observation, {}

        def deck_options(self):
            return {'available_plants': list(range(6))}

        def deck_warnings(self, deck):
            return {'warnings': []}

        def step(self, action):
            self.observation = {**self.observation, 'tick': self.observation['tick'] + action['ticks']}
            return self.observation, 0, False, False, dict(ok=True, events={})

    env = DemoEnv()
    deck = list(range(6))
    task = TaskSpec(level=1, seed=271, profile=profile_for_deck(deck))
    archive = tmp_path / 'demo'
    recorder = Recorder(env, archive, 'demo_test', deck, task)
    recorder.step(dict(type='wait', ticks=3))
    recorder.close()
    obs, meta, frames = replay(env, archive, decision=2, fog_observation='human_v2')
    assert meta['schema_version'] == 2
    assert {path.name for path in archive.iterdir()} == {'meta.json', 'ticks.jsonl', 'frames.npz'}
    assert obs['tick'] == frames[-1]['tick'] == 3
    assert meta['operations'] == 3
    assert replay(env, archive)[0]['tick'] == 3
    changed = tmp_path / 'changed'
    obs, _, _ = replay(env, archive, overrides={1: dict(type='wait', ticks=1)}, save=changed)
    assert obs['tick'] == 1
    assert replay(env, changed)[0]['tick'] == 1


def test_user_pick_records_slot_order_and_replays_fixed_deck(tmp_path):
    selected = [7, 0, 1, 2, 3, 4, 5, 6, 17, 27]

    class DemoEnv:
        def begin_demo(self):
            pass

        def reset(self, *, deck, task):
            self.task = task
            self.deck = deck
            return dict(tick=0, wave=0, wave_count=20, result=0, terminal=False, sun=50,
                        loadout_context={'zombie_roster': []}), dict(deck=selected)

    env = DemoEnv()
    recorder = record_user_pick(env, tmp_path, 31, 209)
    assert env.task.loadout_mode == 'user_pick'
    assert env.task.fog_observation == 'human_v2'
    assert env.task.profile.seed_slot_count == 10
    assert not env.task.profile.imitater_owned
    assert not env.deck
    recorder.close()
    meta = json.loads((recorder.archive / 'meta.json').read_text())
    assert meta['task']['id'].startswith('user_l31_')
    assert meta['task']['deck'] == selected
    assert meta['label_source'] == 'human'
    assert meta['environment_task']['loadout_mode'] == 'fixed'
    assert (recorder.archive / 'frames.npz').is_file()
    replay(env, recorder.archive)
    assert env.task.loadout_mode == 'fixed'
    assert env.deck == selected


@unittest.skipUnless(BINARY.is_file() and (RESOURCE / 'main.pak').is_file(), 'requires simulator and resources')
class HumanDemoTest(unittest.TestCase):
    def test_record_replay_export_and_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'raw'
            deck = [0, 1, 2, 3, 4, 5]
            task = TaskSpec(level=1, seed=271, profile=profile_for_deck(deck), wave_cap=1)
            with EventWaitEnv(RESOURCE, executable=BINARY) as env:
                recorder = Recorder(env, archive, 'demo_test', deck, task, '向日葵示范')
                recorder.step(dict(type='plant', packet=1, row=2, col=0))
                recorder.step(dict(type='wait', ticks=65))
                recorder.step(dict(type='shovel', row=2, col=0))
                recorder.step(dict(type='collect', x=790, y=590))
                recorder.step(dict(type='wait', ticks=10))
                recorder.close()
            report = verify(archive, RESOURCE, BINARY)
            self.assertTrue(report['verified'])
            self.assertEqual(report['tick'], 75)
            with EventWaitEnv(RESOURCE, executable=BINARY) as env:
                changed, changed_meta, _ = replay(env, archive, overrides={2: dict(type='wait', ticks=5)},
                                                 save=root / 'changed')
                self.assertEqual(changed['tick'], 20)
                self.assertFalse(changed_meta['outcome']['override_rejected'])
            self.assertTrue(verify(root / 'changed', RESOURCE, BINARY)['verified'])
            self.assertEqual(json.loads((root / 'changed/meta.json').read_text())['label_source'], 'counterfactual')
            from episode_query import _whatif
            meta = json.loads((archive / 'meta.json').read_text())
            raw_frames = read_frames(archive / 'frames.npz')
            text = _whatif(archive, meta, raw_frames, 2, ['wait:5'], str(RESOURCE), False, None, 1, None)
            self.assertNotIn('重放失败', text)
            self.assertIn('随后继续执行原局录下的操作', text)
            with EventWaitEnv(RESOURCE, executable=BINARY) as env:
                obs, meta, prefix = replay(env, archive, decision=3)
                self.assertEqual(obs['tick'], prefix[-1]['tick'])
                state_hash = env.state_hash()
                snapshot = env.snapshot()
                env.step(dict(type='wait', ticks=20))
                env.restore(snapshot)
                self.assertEqual(env.state_hash(), state_hash)
            target = export(archive, root / 'frames_v2')
            frames = read_frames(target / 'frames.npz')
            self.assertTrue(all(frame['label_source'] == 'human' for frame in frames))
            self.assertFalse(any('note' in frame for frame in frames))
            self.assertEqual(json.loads((target / 'meta.json').read_text())['note'], '向日葵示范')
            from pack_training_rows import packed_load
            from train_rollout_policy import load
            import torch
            data = root / 'frames_v2'
            base = packed_load(data, {'demo_test'}, label_weight={'demo_test': 2.0})
            weighted = load(data, {'demo_test'}, label_weight={'demo_test': 2.0}, source_weight={'human': .2})
            torch.testing.assert_close(weighted[5], base[5] * .2)
            weighted[5].zero_()
            fresh = load(data, {'demo_test'}, label_weight={'demo_test': 2.0}, source_weight={'human': .3})
            torch.testing.assert_close(fresh[5], base[5] * .3)
            lines = (archive / 'ticks.jsonl').read_text().splitlines()
            bad = json.loads(lines[4])
            bad['end_tick'] += 1
            lines[4] = json.dumps(bad)
            (archive / 'ticks.jsonl').write_text('\n'.join(lines) + '\n')
            with self.assertRaisesRegex(RuntimeError, 'operation 5'):
                verify(archive, RESOURCE, BINARY)

    def test_teacher_starts_at_demo_and_scores_demo_candidate(self):
        import torch
        import search_play
        from pvz_rollout_policy import RolloutNet, export_student
        from pvz_value_net import ValueNet
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'raw'
            deck = [0, 1, 2, 3, 4, 5]
            task = TaskSpec(level=1, seed=271, profile=profile_for_deck(deck), wave_cap=1)
            with EventWaitEnv(RESOURCE, executable=BINARY) as env:
                recorder = Recorder(env, archive, 'demo_test', deck, task)
                recorder.step(dict(type='wait', ticks=10))
                recorder.step(dict(type='plant', packet=1, row=2, col=0))
                recorder.close()
                net = RolloutNet(width=13, obs='v2lite').eval()
                with torch.no_grad():
                    net.plant.bias.fill_(-100)
                checkpoint = root / 'student.pt'
                torch.save(dict(state_dict=net.state_dict(), obs='v2lite'), checkpoint)
                # The teacher scores leaves inside the engine (task37), so the worker needs the
                # exported value network exactly as production's _load_nets leaves it.
                from export_value_net import export_value_net
                value_bin = root / 'value.bin'
                export_value_net(ValueNet(width=8).eval(), value_bin)
                native = root / 'student.bin'
                export_student(net, native)
                env.load_rollout_net(native)
                manifest = root / 'tasks.json'
                manifest.write_text(json.dumps(dict(tasks=[dict(task_id='demo_test', level=1,
                                        seeds=[271], deck=deck, wave_cap=1)])))
                args = SimpleNamespace(manifest=manifest, task_id='demo_test', resource_dir=RESOURCE,
                                       demo_archive=archive, demo_decision=2, frames=root / 'teacher',
                                       frames_label='demo', dagger_beta=1.0, rollout_policy=checkpoint,
                                       min_gap=0, per_card=1, halving=False, samples=1, horizon=60)
                with patch.dict(search_play._W, dict(env=env, fast=net,
                                                     value_files=(str(value_bin), str(value_bin))), clear=True), \
                     patch.object(search_play, 'MAX_ACTIONS', 1), patch.object(search_play, '_static_scores', return_value=[]):
                    result = search_play._play(dict(args=args, seed=271, arm='fast_search'))
                self.assertEqual(result['searched'], 1)
                self.assertEqual(result['demo_candidate_added'], 1)
            teacher = root / 'teacher/seed_271/demo_fast_search'
            frames = read_frames(teacher / 'frames.npz')
            self.assertEqual(frames[0]['tick'], 10)
            self.assertIn([1, 2, 0], [target[:3] for target in frames[1]['targets']])
            self.assertEqual(json.loads((teacher / 'meta.json').read_text())['label_source'], 'teacher')

    def test_terminal_recording_replays_final_wave_and_result(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'raw'
            deck = [0, 1, 2, 3, 4, 5]
            task = TaskSpec(level=1, seed=271, profile=profile_for_deck(deck), wave_cap=1)
            with EventWaitEnv(RESOURCE, executable=BINARY) as env:
                recorder = Recorder(env, archive, 'demo_terminal', deck, task)
                recorder.step(dict(type='wait', ticks=20000))
                self.assertTrue(recorder.observation['terminal'])
                result, wave = recorder.observation['result'], recorder.observation['wave']
                recorder.close()
            report = verify(archive, RESOURCE, BINARY)
            self.assertEqual((report['result'], report['wave']), (result, wave))


if __name__ == '__main__':
    unittest.main()
