import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from human_demo import Recorder, export, replay, verify
from pvz_env import TaskSpec
from pvz_event_env import EventWaitEnv
from pvz_frame_store import read_frames
from pvz_tasks import profile_for_deck

RESOURCE = Path(os.environ.get('PVZ_RESOURCE_DIR', '/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN'))
BINARY = Path(os.environ.get('PVZ_EXECUTABLE', ROOT / 'build/pvz-portable'))


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
            bad['state_hash'] = 'wrong'
            lines[4] = json.dumps(bad)
            (archive / 'ticks.jsonl').write_text('\n'.join(lines) + '\n')
            with self.assertRaisesRegex(RuntimeError, 'operation 5'):
                verify(archive, RESOURCE, BINARY)


if __name__ == '__main__':
    unittest.main()
