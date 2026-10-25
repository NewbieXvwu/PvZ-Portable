"""Fresh-process hash regression, including constructor fields unused by adventure."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'python'))
from pvz_env import PlayerProfileContext, TaskSpec
from pvz_event_env import EventWaitEnv
from state_hash_determinism import segments

RESOURCE = Path(os.environ.get('PVZ_RESOURCE_DIR', '/nonexistent'))
BINARY = Path(os.environ.get('PVZ_EXECUTABLE', ROOT / 'build/pvz-portable'))


@unittest.skipUnless(BINARY.is_file() and (RESOURCE / 'main.pak').is_file(),
                     'requires simulator and resources')
class StateHashDeterminismTest(unittest.TestCase):
    def test_unused_cloud_counters_and_fresh_process_hashes(self):
        hashes = []
        with tempfile.TemporaryDirectory() as directory:
            for repeat in range(8):
                with EventWaitEnv(RESOURCE, executable=BINARY) as env:
                    env.begin_demo()
                    env.reset(deck=[0, 1, 2, 3, 4, 5], task=TaskSpec(
                        level=7, seed=30000, profile=PlayerProfileContext(),
                        preplanted=tuple((0, row, 0) for row in range(5))))
                    path = Path(directory) / f'{repeat}.bin'
                    self.assertTrue(env._command(f'STATE_BYTES "{path}"')['ok'])
                    fields = segments(path)
                    for index in range(6):
                        self.assertEqual(fields[f'Challenge.mCloudsCounter[{index}]']['hex'], '00000000')
                    hashes.append([env.state_hash()])
                    snapshot = env.snapshot()
                    projectile_samples = 0
                    for ticks in (1, 59, 940, *([60] * 50)):
                        env.step({'type': 'wait', 'ticks': ticks})
                        hashes[-1].append(env.state_hash())
                        self.assertTrue(env._command(f'STATE_BYTES "{path}"')['ok'])
                        fields = segments(path)
                        for name, value in fields.items():
                            if name.startswith('Projectile['):
                                self.assertEqual(value['hex'], 'ffffffff')
                                projectile_samples += 1
                            if name.startswith('Coin[') and name.endswith('.mIsBeingCollected') and value['hex'] == '00':
                                prefix = name.rsplit('.', 1)[0]
                                self.assertEqual(fields[prefix + '.mCollectX']['hex'], '00000000')
                                self.assertEqual(fields[prefix + '.mCollectY']['hex'], '00000000')
                    self.assertGreater(projectile_samples, 0)
                    env.restore(snapshot)
                    self.assertEqual(env.state_hash(), hashes[-1][0])
        self.assertTrue(all(value == hashes[0] for value in hashes))


if __name__ == '__main__':
    unittest.main()
