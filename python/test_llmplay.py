"""task81 llmplay 接口的纯逻辑测试：不需要引擎二进制。"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'scripts' / 'llmplay'))
sys.path.insert(0, str(ROOT / 'python'))

# 测试不许在仓库里留会话文件：预算状态写到临时目录。
_SESSION_DIR = tempfile.mkdtemp(prefix='llmplay-test-')
os.environ['PVZ_LLMPLAY_SESSION'] = str(Path(_SESSION_DIR) / 'budget.json')

from llmplay.game import Game, ROOT as PACKAGE_ROOT, load_tasks, task_spec  # noqa: E402
from llmplay import budget, build_package  # noqa: E402
from pvz_tasks import profile_for_deck  # noqa: E402


def _bump_budget(path, count):
    """子进程里给同一份预算加 tick（并发测试用，必须模块级才能被 fork/pickle）。"""
    from llmplay import budget as budget_module
    session = budget_module.Session(Path(path))
    session.claim_legend()
    for _ in range(count):
        session.note_ticks(46, 1, 'manual')
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

    def test_task_manifest_registers_roof_46_and_47(self):
        """§G.1：L46/L47 的族已由主控登记，包里的任务清单必须带上。"""
        ids = {task['task_id'] for task in build_package.task_manifest()['tasks']}
        self.assertIn('demo_roof_46_doom_m200', ids)
        self.assertIn('demo_roof_47_doom_m200', ids)

    def test_extra_seeds_are_disjoint_from_default(self):
        manifest = build_package.task_manifest()
        seeds = build_package.extra_seeds(manifest)
        for task in manifest['tasks']:
            self.assertEqual(seeds[task['task_id']][0], task['seeds'][0])
            self.assertEqual(len(set(seeds[task['task_id']])), 3)

    def test_package_root_is_repo_root(self):
        self.assertEqual(PACKAGE_ROOT, ROOT)


class BudgetTest(unittest.TestCase):
    """§A：止损由接口计数，模型绕不过。"""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='llmplay-budget-')
        self.session = budget.reset_session(Path(self.dir) / 'budget.json')

    def test_search_closes_after_budget_and_names_the_next_step(self):
        self.session.calibrate(1000, 1.0)          # 1000 tick/s
        level = 49
        self.session.note_ticks(level, self.session.search_limit + 1, 'search')
        self.assertTrue(self.session.level(level)['search_closed'])
        with self.assertRaises(RuntimeError) as caught:
            self.session.require_search(level, 'branch()')
        self.assertIn('按波手动推进', str(caught.exception))

    def test_three_flat_checkpoints_mark_search_bad(self):
        self.session.calibrate(1000, 1.0)
        level = 48
        limit = self.session.search_limit
        for fraction in (0.4, 0.8, 1.0):
            self.session.note_ticks(level, int(fraction * limit) - self.session.level(level)['search_ticks'], 'search')
        state = self.session.level(level)
        self.assertEqual(len(state['checkpoints']), 3)
        self.assertTrue(state['search_bad'])

    def test_two_flat_checkpoints_switch_to_manual_before_budget_end(self):
        self.session.calibrate(1000, 1.0)
        for fraction in (0.4, 0.8):
            self.session.note_ticks(49, int(fraction * self.session.search_limit)
                                    - self.session.level(49)['search_ticks'], 'search')
        self.assertTrue(self.session.level(49)['search_bad'])
        self.assertEqual(self.session.mode(49), 'manual')
        self.assertLess(self.session.level(49)['search_ticks'], self.session.search_limit)

    def test_search_budget_exhaustion_switches_to_manual(self):
        self.session.calibrate(1000, 1.0)
        self.session.record_wave(49, 12)
        self.session.note_ticks(49, self.session.search_limit, 'search')
        self.assertEqual(self.session.mode(49), 'manual')

    def test_wave_progress_resets_rewrite_counter(self):
        self.session.calibrate(1000, 1.0)
        level = 49
        self.session.note_rewind(level, 100, 900)
        self.session.note_rewind(level, 100, 900)
        self.assertEqual(self.session.level(level)['segment_rewrites'], 2)
        self.session.record_wave(level, 5)
        self.assertEqual(self.session.level(level)['segment_rewrites'], 0)

    def test_ninth_rewrite_in_one_segment_is_refused(self):
        self.session.calibrate(1000, 1.0)
        level = 49
        for _ in range(budget.SEGMENT_REWRITES):
            self.session.note_rewind(level, 100, 900)
        with self.assertRaises(RuntimeError) as caught:
            self.session.note_rewind(level, 100, 900)
        self.assertIn('退回更早的一段', str(caught.exception))

    def test_earlier_rewind_is_allowed_after_segment_limit(self):
        for _ in range(budget.SEGMENT_REWRITES):
            self.session.note_rewind(49, 100, 900)
        self.session.note_rewind(49, 99, 900)
        self.assertEqual(self.session.level(49)['segment_start'], 99)
        self.assertEqual(self.session.level(49)['segment_rewrites'], 1)

    def test_best_log_survives_new_session_and_worse_run(self):
        original = dict(tick=900, task_id='best', ops=[{'action': 'original'}])
        path = self.session.save_best_log(49, original, wave=10)
        other = budget.Session(self.session.path)
        other.save_best_log(49, dict(tick=1000, task_id='worse'), wave=9)
        self.assertEqual(json.loads(path.read_text()), original)
        later = dict(tick=950, task_id='later')
        other.save_best_log(49, later, wave=10)
        self.assertEqual(json.loads(path.read_text()), later)
        self.assertEqual(budget.Session(self.session.path).level(49)['best_log'], str(path))

    def test_branch_counter_is_per_process_not_persisted(self):
        """分支引擎随进程消失；计数写进磁盘会让下一个脚本一开局就顶着上限。"""
        path = Path(self.dir) / 'branch.json'
        first = budget.Session(path)
        first.note_branch(46, +3)
        self.assertEqual(first.branches, 3)
        second = budget.Session(path)
        self.assertEqual(second.branches, 0)

    def test_branch_limit_and_close(self):
        session = budget.Session(Path(self.dir) / 'branch-limit.json')
        for _ in range(budget.BRANCH_LIMIT_SEARCH):
            session.note_branch(46, +1)
        with self.assertRaises(RuntimeError) as caught:
            session.require_branch(46)
        self.assertIn('close()', str(caught.exception))
        session.note_branch(46, -1)
        session.require_branch(46)          # 关掉一个之后可以再开

    def test_concurrent_processes_do_not_lose_updates(self):
        """3 个子代理是 3 个进程：预算的读改写必须串行，不能互相覆盖。"""
        import multiprocessing
        try:
            context = multiprocessing.get_context('fork')
        except ValueError:
            self.skipTest('当前平台没有 fork')
        path = Path(self.dir) / 'budget-concurrent.json'
        processes = [context.Process(target=_bump_budget, args=(str(path), 50))
                     for _ in range(4)]
        for process in processes:
            process.start()
        for process in processes:
            process.join(30)
        self.assertTrue(all(process.exitcode == 0 for process in processes))
        self.assertEqual(budget.Session(path).level(46)['total_ticks'], 200)

    def test_legend_claim_does_not_overwrite_other_process_progress(self):
        stale = budget.Session(self.session.path)
        self.session.note_ticks(49, 123, 'manual')
        self.assertTrue(stale.claim_legend())
        self.assertFalse(self.session.claim_legend())
        self.assertEqual(budget.Session(self.session.path).level(49)['total_ticks'], 123)

    def test_session_path_does_not_depend_on_cwd(self):
        """§A：换个工作目录不能把预算/墙钟清零。"""
        previous = os.environ.pop('PVZ_LLMPLAY_SESSION', None)
        cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
                os.chdir(first)
                one = budget.default_path()
                os.chdir(second)
                two = budget.default_path()
        finally:
            os.chdir(cwd)
            if previous is not None:
                os.environ['PVZ_LLMPLAY_SESSION'] = previous
        self.assertEqual(one, two)
        self.assertEqual(one, Path.home() / '.llmplay_session' / 'budget.json')

    def test_status_line_has_every_field(self):
        self.session.calibrate(1000, 1.0)
        self.session.note_ticks(49, 500, 'search')
        self.session.record_wave(49, 12)
        line = self.session.status_line(49)
        for piece in ('已用', '/ 3:00', '本关 L49', '模式=搜索', '搜索预算', '最远 12 波', '本段重写'):
            self.assertIn(piece, line)

    def test_new_game_gate_uses_two_hours_and_forty_five_minutes(self):
        from unittest.mock import patch
        started = self.session.data['started_at']
        for elapsed in (90 * 60, 165 * 60 - 1):
            with self.subTest(elapsed=elapsed), patch.object(budget.time, 'time', return_value=started + elapsed):
                self.session.require_new_game_allowed()
                self.assertFalse(self.session.stop_new_games)
        with patch.object(budget.time, 'time', return_value=started + 165 * 60):
            with self.assertRaisesRegex(RuntimeError, '墙钟已到 2:45'):
                self.session.require_new_game_allowed()
            self.assertIn('已到 2:45：停止开新局', self.session.status_line(49))

    def test_status_clock_displays_hours_and_minutes(self):
        from unittest.mock import patch
        started = self.session.data['started_at']
        with patch.object(budget.time, 'time', return_value=started + 72 * 60 + 34):
            self.assertIn('[已用 1:12 / 3:00', self.session.status_line(49))


class GridTest(unittest.TestCase):
    """§G.2：格内标僵尸类型，放不下就截断并标 +n。"""

    def test_single_type(self):
        from llmplay import scene
        self.assertEqual(scene._zombie_tags([{'type': 23, 'x': 800.0}]), 'Gar1')

    def test_peashooter_type_zero_is_not_an_empty_packet(self):
        from types import SimpleNamespace
        from llmplay import scene
        game = SimpleNamespace(obs=dict(sun=100, packets=[dict(index=0, type=0, cost=100,
                                                              cooldown=0, active=True)]))
        text = scene.cards(game)
        self.assertIn('豌豆射手', text)
        self.assertNotIn('空槽', text)

    def test_two_types_fit(self):
        from llmplay import scene
        zombies = [{'type': 23, 'x': 800.0}, {'type': 24, 'x': 820.0}]
        self.assertEqual(scene._zombie_tags(zombies), 'Gar1 Imp1')

    def test_truncation_counts_hidden_zombies(self):
        from llmplay import scene
        zombies = [{'type': 23, 'x': 800.0}, {'type': 24, 'x': 820.0}, {'type': 0, 'x': 950.0}]
        self.assertEqual(scene._zombie_tags(zombies), 'Gar1 +2')

    def test_cell_width_is_fixed(self):
        from llmplay import scene
        cell = scene._cell({'type': 39, 'health': 85, 'max_health': 100},
                           [{'type': 23, 'x': 800.0}, {'type': 24, 'x': 820.0}])
        self.assertEqual(len(cell), scene.CELL_WIDTH)
        self.assertTrue(cell.startswith('Mel85'))


class AutoObserveTest(unittest.TestCase):
    """§D：退出时只自动打印最后推进的那一局，别被 branch 淹没。"""

    def test_only_last_active_game_prints_full_observe(self):
        import contextlib
        import io
        from llmplay import game as game_module

        class Fake:
            def __init__(self, name):
                self.task_id = name
                self.tick = 10
                self.env = object()
                self._dirty = True

            def observe(self):
                self._dirty = False
                return 'FULL ' + self.task_id

        first, second = Fake('first'), Fake('second')
        game_module._LIVE_GAMES.extend([first, second])
        game_module._LAST_ACTIVE = second
        try:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                game_module._auto_observe()
        finally:
            for fake in (first, second):
                game_module._LIVE_GAMES.remove(fake)
            game_module._LAST_ACTIVE = None
        output = buffer.getvalue()
        self.assertIn('FULL second', output)
        self.assertNotIn('FULL first', output)
        self.assertIn('1 个其它局', output)


if __name__ == '__main__':
    unittest.main()
