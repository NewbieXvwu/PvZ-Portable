"""task81 llmplay 接口的纯逻辑测试：不需要引擎二进制。"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'scripts' / 'llmplay'))
sys.path.insert(0, str(ROOT / 'python'))

from llmplay.game import Game, ROOT as PACKAGE_ROOT, load_tasks, task_spec  # noqa: E402
from llmplay import build_package  # noqa: E402
from pvz_tasks import profile_for_deck  # noqa: E402
from pvz_env import TaskSpec  # noqa: E402


class TaskSpecTest(unittest.TestCase):
    def test_roof_49_spec_matches_human_demo_derivation(self):
        task = load_tasks()['demo_roof_49_doom_m200']
        spec, deck, seed = task_spec(task)
        self.assertEqual(deck, task['deck'])
        self.assertEqual(seed, task['seeds'][0])
        self.assertEqual(spec.level, 49)
        self.assertEqual(spec.zombie_count_multiplier, task['zombie_count_multiplier'])
        # human_demo.main 用的是同一套推导：profile_for_deck(deck) + TaskSpec(**task_extra)
        self.assertEqual(spec.profile, profile_for_deck(tuple(task['deck'])))

    def test_seed_override(self):
        task = load_tasks()['demo_roof_48_doom_m200']
        _, _, seed = task_spec(task, 12345)
        self.assertEqual(seed, 12345)

    def test_unknown_task_lists_alternatives(self):
        with self.assertRaises(ValueError):
            Game('no_such_task')


class LogRoundTripTest(unittest.TestCase):
    def test_save_and_load_keeps_operations(self):
        ops = [dict(action=dict(type='plant', packet=0, col=0, row=0), tick=0, end_tick=0,
                    ok=True, state_hash='aa'),
               dict(action=dict(type='wait', ticks=60), tick=0, end_tick=60,
                    ok=True, state_hash='bb')]
        frames = [dict(tick=0, wave=1, sun=50, plants=[], zombies=[], mowers=[], packets=[],
                       scene={}, row_terrain=[], lanes=[], totals={}, graves=[],
                       action=None, action_rejected=False, label_source='llm_gpt_pro', operation=0)]
        game = Game('demo_roof_49_doom_m200', ops=ops, frames=frames, tick=60)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.json'
            game.save(path)
            loaded = Game.load(path)
        self.assertEqual(loaded.ops, ops)
        self.assertEqual(loaded.tick, 60)
        self.assertEqual(loaded.seed, game.seed)
        self.assertEqual(len(loaded.frames), 1)

    def test_load_rejects_foreign_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.json'
            path.write_text(json.dumps({'schema': 'something_else'}), encoding='utf-8')
            with self.assertRaises(ValueError):
                Game.load(path)


class PackageTest(unittest.TestCase):
    def test_task_manifest_is_roof_only_and_covers_level_41(self):
        manifest = build_package.task_manifest()
        ids = [task['task_id'] for task in manifest['tasks']]
        self.assertTrue(all(task_id.startswith('demo_roof_') or task_id == 'smoke_roof_41'
                            for task_id in ids))
        smoke = [task for task in manifest['tasks'] if task['task_id'] == 'smoke_roof_41']
        self.assertEqual(len(smoke), 1)
        self.assertEqual(smoke[0]['level'], 41)
        # 卡组不手写：必须等于 deck_for_level(41)
        import pvz_tasks
        self.assertEqual(smoke[0]['deck'], list(pvz_tasks.deck_for_level(41)))

    def test_extra_seeds_are_disjoint_from_default(self):
        manifest = build_package.task_manifest()
        seeds = build_package.extra_seeds(manifest)
        for task in manifest['tasks']:
            self.assertEqual(seeds[task['task_id']][0], task['seeds'][0])
            self.assertEqual(len(set(seeds[task['task_id']])), 3)

    def test_package_root_is_repo_root(self):
        self.assertEqual(PACKAGE_ROOT, ROOT)


if __name__ == '__main__':
    unittest.main()
