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
os.environ['LLMPLAY_OUT'] = str(Path(_SESSION_DIR) / 'out')

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

    def test_missing_exec_bit_is_added_before_start(self):
        from llmplay.game import _ensure_executable
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pvz-portable'
            path.write_bytes(b'not really an engine')
            path.chmod(0o644)
            _ensure_executable(path)
            self.assertTrue(os.access(path, os.X_OK))
            _ensure_executable(path)                      # 幂等
            self.assertEqual(path.read_bytes(), b'not really an engine')

    def test_ensure_executable_leaves_missing_file_alone(self):
        from llmplay.game import _ensure_executable
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nope'
            _ensure_executable(path)
            self.assertFalse(path.exists())

    def test_load_rejects_foreign_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.json'
            path.write_text(json.dumps({'schema': 'something_else'}), encoding='utf-8')
            with self.assertRaises(ValueError):
                Game.load(path)


class PackageTest(unittest.TestCase):
    def test_task_manifest_covers_all_five_layers(self):
        """task81c：包里必须是 A–E 五层，L41 由登记族提供（不再临时造 smoke 任务）。"""
        manifest = build_package.task_manifest()
        ids = [task['task_id'] for task in manifest['tasks']]
        plan = json.loads((ROOT / 'configs' / 'llmplay_seeds.json').read_text(encoding='utf-8'))
        expected = [task_id for layer in ('A', 'B', 'C', 'D', 'E')
                    for task_id in plan['layers'][layer]]
        self.assertEqual(ids, expected)
        by_id = {task['task_id']: task for task in manifest['tasks']}
        self.assertEqual(by_id['demo_roof_41_doom_m200']['level'], 41)
        self.assertEqual(by_id['demo_roof_49_doom_m400']['zombie_count_multiplier'], 4.0)
        self.assertEqual(by_id['fog79_l39_snow_blover_v2_m200']['zombie_count_multiplier'], 2.0)
        self.assertEqual(by_id['demo_pool_30_orig_m100']['conveyor_start'], 'original')
        self.assertEqual(by_id['demo_pool_30_orig_m100']['deck'], [])
        self.assertEqual(by_id['demo_storm_40_orig_m100']['conveyor_start'], 'original')
        self.assertEqual(by_id['demo_boss_50_orig_m100']['level'], 50)
        self.assertEqual(by_id['demo_boss_50_orig_m100']['deck'], [])

    def test_task_manifest_registers_roof_46_and_47(self):
        """§G.1：L46/L47 的族已由主控登记，包里的任务清单必须带上。"""
        ids = {task['task_id'] for task in build_package.task_manifest()['tasks']}
        self.assertIn('demo_roof_46_doom_m200', ids)
        self.assertIn('demo_roof_47_doom_m200', ids)

    def test_extra_seeds_are_forty_per_task_and_disjoint(self):
        """task81c：每任务 40 个种子，块内不重复，且第一个就是清单默认种子。"""
        manifest = build_package.task_manifest()
        seeds = build_package.extra_seeds(manifest)
        flat = []
        for task in manifest['tasks']:
            values = seeds[task['task_id']]
            self.assertEqual(len(values), 40)
            self.assertEqual(values[0], task['seeds'][0])
            self.assertEqual(len(set(values)), 40)
            flat.extend(values)
        self.assertEqual(len(set(flat)), len(flat))
        self.assertTrue(all(81_000_000 <= seed < 81_100_000 for seed in flat))

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
        for piece in ('已用', '本关 L49', '模式=搜索', '搜索预算', '最远 12 波', '本段重写'):
            self.assertIn(piece, line)

    def test_no_time_limit_even_after_years(self):
        from unittest.mock import patch
        started = self.session.data['started_at']
        for elapsed in (0, 165 * 60, 10 * 365 * 86400):
            with self.subTest(elapsed=elapsed), patch.object(budget.time, 'time', return_value=started + elapsed):
                self.session.require_new_game_allowed()
                self.assertFalse(self.session.stop_new_games)
                game = Game('demo_roof_49_doom_m200')
                game.close()
                line = self.session.status_line(49)
                self.assertNotIn('/ 3:00', line)
                self.assertNotIn('停止', line)
                self.assertNotIn('打包', line)

    def test_status_clock_displays_hours_and_minutes(self):
        from unittest.mock import patch
        started = self.session.data['started_at']
        with patch.object(budget.time, 'time', return_value=started + 72 * 60 + 34):
            self.assertIn('[已用 1:12 |', self.session.status_line(49))


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




class RobustnessTest(unittest.TestCase):
    """task81d：换未来也能赢 —— 种子块、闭环循环、meta 记录。"""

    def setUp(self):
        budget.reset_session(Path(tempfile.mkdtemp(prefix='llmplay-robustness-')) / 'budget.json')

    def test_branch_preserves_manual_patch_and_copies_assessment(self):
        from unittest.mock import patch
        parent = Game('demo_roof_49_doom_m200', tick=60, ops=[dict(end_tick=60)])
        parent.manual_patch = True
        parent.robustness_report = {'checkpoints': {'10': {'wins': 5, 'k': 8}}}
        with patch.object(parent, '_start'):
            child = parent.branch()
        try:
            self.assertTrue(child.manual_patch)
            self.assertEqual(child.robustness_report, parent.robustness_report)
            child.robustness_report['checkpoints']['10']['wins'] = 0
            self.assertEqual(parent.robustness_report['checkpoints']['10']['wins'], 5)
        finally:
            child.close()
            parent.close()

    def test_rewind_invalidates_assessment_of_the_old_trajectory(self):
        from unittest.mock import patch
        game = Game('demo_roof_49_doom_m200', tick=60,
                    robustness={'passed': True, 'checkpoints': {'10': {'wins': 8, 'k': 8}}})
        with patch.object(game, '_start'), patch.object(game, '_replay_to'):
            game.rewind(30)
        self.assertTrue(game.manual_patch)
        self.assertIsNone(game.robustness_report)
        game.close()

    def test_internal_trials_cannot_rewind_branch_or_export(self):
        from unittest.mock import patch
        trial = Game('demo_roof_49_doom_m200', tick=60, ops=[dict(end_tick=60)], _internal=True)
        with tempfile.TemporaryDirectory() as directory, patch.object(trial, '_start') as start:
            for name, args in [('rewind', (0,)), ('branch', ()), ('save', (Path(directory)/'run.json',)),
                               ('export_archive', (Path(directory)/'archive',)), ('robustness', (lambda g: None,))]:
                with self.subTest(method=name), self.assertRaisesRegex(RuntimeError, '试打局'):
                    getattr(trial, name)(*args)
            self.assertEqual(list(Path(directory).iterdir()), [])
            start.assert_not_called()
        trial.close()

    def test_internal_trial_charges_search_ticks_in_manual_mode(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        trial = Game('demo_roof_49_doom_m200', tick=60, ops=[dict(end_tick=60)], _internal=True)
        trial.obs = dict(wave=10, tick=60)
        trial.env = SimpleNamespace(_last_response={'state_hash': 'diagnostic'}, close=lambda: None)
        trial.session.set_mode(trial.budget_key, 'manual')
        def step(action):
            trial.tick += action['ticks']
            trial.obs['tick'] = trial.tick
            return True
        with patch.object(trial, '_start'), patch.object(trial, '_step_raw', side_effect=step), \
                patch.object(trial, '_note_events'), patch.object(trial, '_maybe_frame'):
            trial.wait(30)
        self.assertEqual(trial.session.level(trial.budget_key)['search_ticks'], 30)
        self.assertEqual(trial.session.level(trial.budget_key)['manual_ticks'], 0)
        trial.close()

    def test_pass_requires_five_wins_out_of_eight_actual_futures(self):
        from unittest.mock import patch
        class Env:
            result = 0
            def snapshot_fast(self): return 'snapshot'
            def restore_fast(self, snapshot): self.result = 0
            def observe(self): return dict(tick=0, wave=10, result=self.result)
            def reseed(self, seed): self.result = int(seed < wins)
            def release_snapshot(self, snapshot): pass
        class Trial:
            tick = 0
            terminal = True
            ops = []
            def __init__(self): self.env = Env()
            def _start(self): pass
            def _seek_wave_start(self, wave, tick): pass
            def _reset_bookkeeping(self, obs): self.obs = obs
            def close(self): pass
        def policy(game): game.wait(30)
        for k, wins, passed in [(None, 5, True), (8, 4, False), (7, 7, False), (5, 5, False)]:
            with self.subTest(k=k, wins=wins):
                parent = Game('demo_roof_49_doom_m200', tick=60)
                parent.wave_history = [dict(wave=10, tick=0)]
                with patch.object(parent, '_start'), patch.object(parent, '_persist_best'), \
                        patch('llmplay.game.Game', return_value=Trial()), \
                        patch('llmplay.game.robustness_seeds', return_value={10: list(range(8))}):
                    report = parent.robustness(policy, k=k, checkpoints=[10])
                self.assertEqual(report['passed'], passed)
                self.assertEqual(report['k'], k if k is not None else 8)
                parent.close()

    def test_wave_start_is_first_tick_inside_a_coarse_wait(self):
        class Env:
            tick = 10
            _last_response = {'state_hash': 'diagnostic'}
            def observe(self):
                return dict(tick=self.tick, wave=9 if self.tick < 17 else 10,
                            terminal=False, result=0, plants=[], zombies=[], defenses=[])
            def snapshot_fast(self): return self.tick
            def restore_fast(self, tick): self.tick = tick
            def release_snapshot(self, snapshot): pass
            def step(self, action):
                self.tick += action['ticks']
                return self.observe(), 0, False, False, dict(ok=True, events={})
            def close(self): pass
        prefix = dict(action=dict(type='wait', ticks=10), tick=0, end_tick=10,
                      ok=True, state_hash='diagnostic')
        trial = Game('demo_roof_49_doom_m200', ops=[prefix], tick=10, _internal=True)
        trial.env = Env()
        trial._reset_bookkeeping(trial.env.observe())
        trial._seek_wave_start(10, 35)
        self.assertEqual(trial.tick, 17)
        self.assertEqual(trial.obs['wave'], 10)
        self.assertEqual(trial.ops[-1]['action'], dict(type='wait', ticks=7))
        self.assertGreater(trial.session.level(trial.budget_key)['search_ticks'], 7)
        trial.close()

    def test_policy_source_unavailable_is_not_replaced_by_a_name(self):
        from unittest.mock import patch
        from llmplay.game import _policy_source
        with patch('llmplay.game.inspect.getsource', side_effect=OSError('no source')):
            with self.assertRaisesRegex(ValueError, r'\.py 文件'):
                _policy_source(lambda game: None)

    def test_fixed_action_list_is_rejected_as_a_policy(self):
        from llmplay.game import _policy_source
        with self.assertRaisesRegex(TypeError, '闭环打法函数'):
            _policy_source([dict(type='wait', ticks=60)])

    def test_seed_block_is_fixed_and_disjoint_from_task_seeds(self):
        from llmplay.game import robustness_seeds
        plan = robustness_seeds()
        self.assertEqual(sorted(plan), [0, 10, 20, 30])
        self.assertTrue(all(len(seeds) == 8 for seeds in plan.values()))
        flat = [seed for seeds in plan.values() for seed in seeds]
        self.assertTrue(all(83_000_000 <= seed < 83_000_400 for seed in flat))
        self.assertEqual(len(set(flat)), 32)
        task_seeds = {seed for seeds in json.loads(
            (ROOT / 'configs' / 'llmplay_seeds.json').read_text(encoding='utf-8'))['tasks'].values()
            for seed in seeds}
        self.assertFalse(set(flat) & task_seeds)

    def test_truncate_ops_splits_wait_at_target(self):
        from llmplay.game import _truncate_ops
        ops = [dict(action=dict(type='wait', ticks=60), tick=0, end_tick=60, ok=True, state_hash='a'),
               dict(action=dict(type='wait', ticks=60), tick=60, end_tick=120, ok=True, state_hash='b')]
        kept = _truncate_ops(ops, 90)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0]['end_tick'], 60)
        self.assertEqual(kept[1]['action'], dict(type='wait', ticks=30))
        self.assertEqual(kept[1]['end_tick'], 90)
        self.assertEqual(_truncate_ops(ops, 0), [])

    def test_drive_policy_stops_at_limit_and_requires_progress(self):
        from llmplay.game import _drive_policy

        class Fake:
            def __init__(self):
                self.terminal = False
                self.tick = 0
                self.ops = []

        def bump(game):
            game.tick += 10
            game.ops.append(1)

        game = Fake()
        _drive_policy(game, bump, limit_tick=35)
        self.assertEqual(game.tick, 40)          # 允许最后一次越过上限

        idle = Fake()
        with self.assertRaises(RuntimeError) as caught:
            _drive_policy(idle, lambda g: None, limit_tick=100)
        self.assertIn('没有推进', str(caught.exception))

    def test_internal_trial_does_not_record_wave_progress(self):
        """稳健性试打的波数不能算进「最远存活波」，否则会骗过止损判据。"""
        session = budget.reset_session(Path(tempfile.mkdtemp(prefix='llmplay-trial-')) / 'budget.json')
        ops = [dict(action=dict(type='wait', ticks=60), tick=0, end_tick=60, ok=True, state_hash='aa')]
        trial = Game('demo_roof_49_doom_m200', ops=ops, tick=60, _internal=True)
        trial.session = session
        before = dict(session.level(49))
        trial.obs = dict(wave=30, sun=50, plants=[])
        trial.tick = 60
        trial._close_wave(30)
        after = dict(session.level(49))
        self.assertEqual(after['last_progress_wave'], before['last_progress_wave'])
        self.assertEqual(after['max_wave'], before['max_wave'])
        trial.close()

    def test_wave_tick_comes_from_wave_history(self):
        ops = [dict(action=dict(type='wait', ticks=60), tick=0, end_tick=60, ok=True, state_hash='aa')]
        game = Game('demo_roof_49_doom_m200', ops=ops, tick=60)
        self.assertEqual(game._wave_tick(0), 0)
        self.assertIsNone(game._wave_tick(10))
        game.wave_history = [dict(wave=0, tick=0), dict(wave=10, tick=5000)]
        self.assertEqual(game._wave_tick(10), 5000)

    def test_policy_source_hash_is_stable(self):
        from llmplay.game import _policy_source
        def policy(game):
            game.wait(10)
        first = _policy_source(policy)
        self.assertEqual(first, _policy_source(policy))
        self.assertIn('game.wait(10)', first)

    def test_log_and_archive_meta_carry_robustness(self):
        report = dict(policy_sha256='ab' * 32, k=8, passed=True,
                      checkpoints={'10': dict(wins=6, k=8, reached=True)})
        ops = [dict(action=dict(type='wait', ticks=60), tick=0, end_tick=60, ok=True, state_hash='aa')]
        game = Game('demo_roof_49_doom_m200', ops=ops, tick=60, robustness=report)
        game.manual_patch = True
        document = game.log()
        self.assertEqual(document['robustness'], report)
        self.assertTrue(document['manual_patch'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.json'
            game.save(path)
            loaded = Game.load(path)
        self.assertEqual(loaded.robustness_report, report)
        self.assertTrue(loaded.manual_patch)


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


def _deliver_fixture(archive, out):
    from llmplay.delivery import record_archive
    os.environ['LLMPLAY_OUT'] = out
    record_archive(Path(archive))


def _install_resource_fixture(root):
    from llmplay.resources import install
    install(Path(root))


def _run_id_fixture(out, result):
    from llmplay.delivery import ensure_run_id
    os.environ['LLMPLAY_OUT'] = out
    Path(result).write_text(ensure_run_id())


class PortableRuntimeTest(unittest.TestCase):
    def setUp(self):
        from unittest.mock import patch
        self.tmp=tempfile.TemporaryDirectory(prefix='llmplay-portable-')
        self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name)
        self.patch=patch.dict(os.environ,LLMPLAY_OUT=str(self.base/'out'))
        self.patch.start()
        self.addCleanup(self.patch.stop)
        budget.reset_session(self.base/'budget.json')

    def test_output_priority_override_sandbox_then_sibling(self):
        from unittest.mock import patch
        from llmplay import runtime
        package=self.base/'package'
        sandbox=self.base/'sandbox'
        with patch.object(runtime,'DEFAULT_SANDBOX_OUT',sandbox):
            self.assertEqual(runtime.output_path(package),self.base/'out')
            os.environ.pop('LLMPLAY_OUT')
            self.assertEqual(runtime.output_path(package),self.base/'out')
            sandbox.mkdir()
            self.assertEqual(runtime.output_path(package),sandbox)
            with patch('llmplay.runtime.os.access',return_value=False):
                self.assertEqual(runtime.output_path(package),self.base/'out')

    def test_paths_prints_actual_output_directory(self):
        import io,contextlib
        from llmplay import paths,out_dir
        os.environ['LLMPLAY_OUT']=str(self.base/'路径 with spaces')
        buffer=io.StringIO()
        with contextlib.redirect_stdout(buffer):result=paths()
        self.assertEqual(result['out_dir'],str(out_dir()))
        self.assertEqual(json.loads(buffer.getvalue()),result)

    def test_run_id_is_persisted_and_order_is_stable(self):
        from llmplay import progress,next_task,Game
        from llmplay.delivery import seed_order
        first=progress()
        run_id=first['run_id']
        self.assertEqual(len(run_id),32)
        self.assertEqual(json.loads((self.base/'out/progress.json').read_text())['run_id'],run_id)
        self.assertEqual(progress()['run_id'],run_id)
        task='demo_roof_46_doom_m200'
        ordered=first['tasks'][task]['available_seeds']
        self.assertEqual(set(ordered),set(range(81000000,81000040)))
        self.assertEqual(ordered,seed_order(task,run_id,list(range(81000000,81000040))))
        self.assertEqual(next_task()['seed'],ordered[0])
        with Game(task) as game:
            self.assertEqual(game.run_id,run_id)
            self.assertEqual(game.seed,ordered[0])
            saved=game.save(self.base/'run.json')
        with Game.load(saved) as restored:self.assertEqual(restored.run_id,run_id)

    def test_independent_runs_get_different_seed_orders(self):
        from llmplay import progress
        first=progress()
        os.environ['LLMPLAY_OUT']=str(self.base/'another-out')
        second=progress()
        self.assertNotEqual(first['run_id'],second['run_id'])
        task='demo_roof_46_doom_m200'
        self.assertNotEqual(first['tasks'][task]['available_seeds'],second['tasks'][task]['available_seeds'])

    def test_concurrent_initialization_shares_one_run_id(self):
        import multiprocessing
        context=multiprocessing.get_context('fork')
        targets=[self.base/('id-%s'%i) for i in range(4)]
        processes=[context.Process(target=_run_id_fixture,args=(str(self.base/'out'),str(p))) for p in targets]
        for process in processes:process.start()
        for process in processes:process.join(30)
        self.assertTrue(all(p.exitcode==0 for p in processes))
        self.assertEqual(len({p.read_text() for p in targets}),1)

    def test_resume_empty_run_preserves_id_and_seed_order(self):
        from llmplay import progress,resume
        first=progress()
        source=self.base/'out/out.zip'
        os.environ['LLMPLAY_OUT']=str(self.base/'resumed')
        self.assertNotEqual(progress()['run_id'],first['run_id'])
        result=resume(source)
        self.assertEqual(result['run_id'],first['run_id'])
        task='demo_roof_46_doom_m200'
        self.assertEqual(result['tasks'][task]['available_seeds'],first['tasks'][task]['available_seeds'])

    def test_offline_import_scanner_detects_nested_and_aliased_modules(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('offline_selfcheck',ROOT/'scripts/llmplay/selfcheck.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        network_imports=module.network_imports
        source=self.base/'bad.py'
        source.write_text('def nested():\n    import socket as s\n    from urllib.request import urlopen\n    import requests\n')
        self.assertEqual(len(network_imports([source])),3)
        source.write_text('import json\nfrom pathlib import Path\n')
        self.assertEqual(network_imports([source]),[])

    def test_selfcheck_reports_unsupported_environment_before_import(self):
        import importlib.util,contextlib,io
        from unittest.mock import patch
        spec=importlib.util.spec_from_file_location('portable_selfcheck',ROOT/'scripts/llmplay/selfcheck.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for old_python in (True,False):
            buffer=io.StringIO()
            with patch.object(module.sys,'version_info',(3,9) if old_python else (3,10)), \
                    patch.object(module.platform,'system',return_value='Linux' if old_python else 'Darwin'), \
                    contextlib.redirect_stdout(buffer):
                self.assertEqual(module.main(),1)
            self.assertIn('Python >= 3.10' if old_python else 'x86_64 Linux',buffer.getvalue())

    def test_selfcheck_scans_before_importing_or_installing_interface(self):
        import importlib.util,contextlib,io
        from unittest.mock import patch
        spec=importlib.util.spec_from_file_location('scan_first_selfcheck',ROOT/'scripts/llmplay/selfcheck.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        (self.base/'llmplay').mkdir()
        source=self.base/'llmplay/bad.py'
        source.write_text('import socket\n')
        buffer=io.StringIO()
        with patch.object(module,'ROOT',self.base), \
                patch.object(module.tempfile,'mkdtemp',side_effect=AssertionError('不能在扫描前初始化')), \
                contextlib.redirect_stdout(buffer):
            self.assertEqual(module.main(),1)
        self.assertIn('接口含联网模块 import',buffer.getvalue())

    def test_selfcheck_does_not_scan_unrelated_user_scripts(self):
        import importlib.util,contextlib,io
        from unittest.mock import patch
        spec=importlib.util.spec_from_file_location('sdk_scope_selfcheck',ROOT/'scripts/llmplay/selfcheck.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        (self.base/'llmplay').mkdir()
        (self.base/'llmplay/sdk.py').write_text('import json\n')
        (self.base/'user_script.py').write_text('import socket\n')
        with patch.object(module,'ROOT',self.base), \
                patch.object(module.tempfile,'mkdtemp',side_effect=RuntimeError('扫描已通过')), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'扫描已通过'):module.main()


class ResourceTest(unittest.TestCase):
    def setUp(self):
        import zipfile
        import hashlib
        self.tmp = tempfile.TemporaryDirectory(prefix='llmplay-resource-')
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base/'llmplay_pkg'
        (self.root/'configs').mkdir(parents=True)
        self.exe = self.root/'engine'
        self.exe.write_bytes(b'test-engine')
        self.engine_sha = hashlib.sha256(self.exe.read_bytes()).hexdigest()
        files = {'main.pak': b'pak', 'properties/partner.xml': b'<properties/>',
                 'cache64/compiled/reanim/test.reanim.compiled': b'compiled-definition'}
        manifest = dict(schema='llmplay_resources_v1', engine_sha256=self.engine_sha,
                        files={name: hashlib.sha256(blob).hexdigest() for name,blob in files.items()})
        self.zip = self.base/'llmplay_res.zip'
        with zipfile.ZipFile(self.zip,'w') as bundle:
            for name,blob in files.items(): bundle.writestr('res/'+name,blob)
            bundle.writestr('res/llmplay_resources.json',json.dumps(manifest))
        self.set_marker()

    def set_marker(self):
        from llmplay.resources import sha256
        (self.root/'configs/llmplay_package.json').write_text(json.dumps(dict(
            engine_sha256=self.engine_sha, resource_zip_sha256=sha256(self.zip))))

    def test_automatic_install_and_persistent_cache_directory(self):
        from llmplay.resources import install, app_data
        resource = install(self.root)
        self.assertEqual((resource/'main.pak').read_bytes(),b'pak')
        session = budget.Session(self.base/'session/budget.json')
        first = app_data(self.exe,resource,session)
        self.assertEqual((first/'cache64/compiled/reanim/test.reanim.compiled').read_bytes(),b'compiled-definition')
        second = app_data(self.exe,resource,budget.Session(session.path))
        self.assertEqual(first,second)
        self.zip.unlink()          # 装好后不需要反复解包，也不再要求 zip 存在
        self.assertEqual(install(self.root),resource)

    def test_missing_resources_have_actionable_error(self):
        from unittest.mock import patch
        from llmplay.resources import install
        self.zip.unlink()
        with patch.dict(os.environ,PVZ_LLMPLAY_RES_ZIP=str(self.zip)):
            with self.assertRaisesRegex(FileNotFoundError,'同时上传'): install(self.root)

    def test_zip_pair_mismatch_does_not_install(self):
        from llmplay.resources import install
        with self.zip.open('ab') as stream:stream.write(b'changed')
        with self.assertRaisesRegex(ValueError,'SHA256'): install(self.root)
        self.assertFalse((self.root/'res').exists())

    def test_member_corruption_does_not_publish_partial_resources(self):
        import zipfile
        from llmplay.resources import install
        with zipfile.ZipFile(self.zip) as source:
            files={name:source.read(name) for name in source.namelist()}
        files['res/main.pak']=b'bad'
        with zipfile.ZipFile(self.zip,'w') as target:
            for name,blob in files.items():target.writestr(name,blob)
        self.set_marker()
        with self.assertRaisesRegex(ValueError,'损坏'): install(self.root)
        self.assertFalse((self.root/'res').exists())

    def test_unsafe_resource_zip_is_rejected(self):
        import zipfile
        from llmplay.resources import install
        with zipfile.ZipFile(self.zip,'a') as bundle:bundle.writestr('../escape','bad')
        self.set_marker()
        with self.assertRaisesRegex(ValueError,'不安全'): install(self.root)
        self.assertFalse((self.base/'escape').exists())

    def test_concurrent_first_import_installs_once(self):
        import multiprocessing
        context=multiprocessing.get_context('fork')
        processes=[context.Process(target=_install_resource_fixture,args=(str(self.root),)) for _ in range(3)]
        for process in processes:process.start()
        for process in processes:process.join(30)
        self.assertTrue(all(p.exitcode==0 for p in processes))
        self.assertEqual((self.root/'res/main.pak').read_bytes(),b'pak')

    def test_other_engine_does_not_reuse_incompatible_cache(self):
        from llmplay.resources import install,app_data
        resource=install(self.root)
        self.exe.write_bytes(b'other-engine')
        path=app_data(self.exe,resource,budget.Session(self.base/'budget.json'))
        self.assertFalse((path/'cache64').exists())

    def test_game_and_internal_trial_use_the_same_app_data(self):
        from unittest.mock import patch
        from llmplay.resources import install
        resource=install(self.root)
        budget.reset_session(self.base/'game-budget.json')
        games=[Game('demo_roof_46_doom_m200',executable=self.exe,resource_dir=resource,_internal=internal)
               for internal in (False,True)]
        try:
            with patch('llmplay.game.PvZEnv') as env:
                for game in games:
                    with patch.object(game,'_reset_engine'):game._start()
                saves=[call.kwargs['save_dir'] for call in env.call_args_list]
                self.assertEqual(saves[0],saves[1])
                self.assertEqual(len(saves),2)
        finally:
            for game in games:game.close()


class DeliveryTest(unittest.TestCase):
    def setUp(self):
        from unittest.mock import patch
        self.tmp = tempfile.TemporaryDirectory(prefix='llmplay-delivery-')
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.out = self.base / 'out'
        self.env = patch.dict(os.environ, LLMPLAY_OUT=str(self.out))
        self.env.start()
        self.addCleanup(self.env.stop)
        self.session = budget.reset_session(self.base / 'budget.json')
        self.task = 'demo_roof_46_doom_m200'

    def archive(self, seed, *, won=True, robust=True, task=None):
        path = self.base / ('input-%s-%s' % (seed, task or self.task))
        path.mkdir(exist_ok=True)
        meta = dict(schema='episode_archive_v1', task=dict(id=task or self.task, seed=seed),
                    manual_patch=True, note='不同布阵参数 A=2', operations=1, frame_count=0,
                    outcome=dict(result=1 if won else 2, won=won, final_tick=60, final_wave=10),
                    robustness=dict(policy_sha256='ab'*32, passed=robust,
                                    checkpoints={'10': dict(wins=5 if robust else 4, k=8, reached=True)}))
        (path / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
        (path / 'ticks.jsonl').write_text(json.dumps(dict(action=dict(type='wait', ticks=60),
                                                        tick=0, end_tick=60, ok=True))+'\n')
        (path / 'frames.jsonl').write_text('')
        return path

    def test_atomic_snapshot_contains_all_deliverables_and_budget(self):
        import zipfile
        from llmplay.delivery import record_archive
        self.session.note_ticks(self.task, 123, 'search')
        record_archive(self.archive(81000000))
        document = json.loads((self.out / 'progress.json').read_text())
        record = document['tasks'][self.task]['seeds']['81000000']
        self.assertTrue(record['manual_patch'])
        self.assertEqual(set(document['layers']), set('ABCDE'))
        self.assertIn('不同布阵参数', (self.out / 'report.md').read_text())
        with zipfile.ZipFile(self.out / 'out.zip') as bundle:
            self.assertEqual(json.loads(bundle.read('out/progress.json')), document)
            self.assertEqual(json.loads(bundle.read('out/budget.json'))['levels'][self.task]['search_ticks'], 123)
            self.assertIn('out/' + record['archive'] + '/ticks.jsonl', bundle.namelist())

    def test_compression_failure_preserves_previous_generation(self):
        from unittest.mock import patch
        from llmplay.delivery import record_archive
        record_archive(self.archive(81000000))
        before = {name: (self.out / name).read_bytes() for name in ('progress.json', 'report.md', 'out.zip', 'budget.json')}
        with patch('llmplay.delivery.zipfile.ZipFile.write', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                record_archive(self.archive(81000001))
        self.assertEqual(before, {name: (self.out / name).read_bytes() for name in before})

    def test_concurrent_delivery_keeps_both_seeds(self):
        import multiprocessing
        from llmplay.delivery import progress
        context = multiprocessing.get_context('fork')
        inputs = [self.archive(81000000 + n) for n in range(3)]
        processes = [context.Process(target=_deliver_fixture, args=(str(path), str(self.out))) for path in inputs]
        for process in processes: process.start()
        for process in processes: process.join(30)
        self.assertTrue(all(p.exitcode == 0 for p in processes))
        self.assertEqual(len(progress()['tasks'][self.task]['seeds']), 3)

    def test_publish_failure_preserves_snapshot_and_can_retry(self):
        from unittest.mock import patch
        from llmplay.delivery import record_archive
        record_archive(self.archive(81000000))
        before = {name: (self.out / name).read_bytes() for name in ('progress.json', 'report.md', 'out.zip')}
        replace = os.replace
        def fail_commit(source, target):
            if Path(target).name == 'current': raise OSError('rename failed')
            return replace(source, target)
        with patch('llmplay.delivery.os.replace', side_effect=fail_commit):
            with self.assertRaisesRegex(OSError, 'rename failed'):
                record_archive(self.archive(81000001))
        self.assertEqual(before, {name: (self.out / name).read_bytes() for name in before})
        record_archive(self.archive(81000001))
        self.assertEqual(len(json.loads((self.out/'progress.json').read_text())['archives']), 2)

    def test_default_resume_uses_progress_without_hash_or_replay(self):
        from unittest.mock import patch
        from llmplay import resume, next_task, list_tasks
        from llmplay.delivery import record_archive
        self.session.note_ticks(self.task, 345, 'search')
        self.session.note_rewind(self.task, 10, 20)
        for n in range(5): record_archive(self.archive(81000000 + n, robust=n < 3))
        source = self.out / 'out.zip'
        previous = json.loads((self.out/'progress.json').read_text())
        os.environ['LLMPLAY_OUT'] = str(self.base / 'new')
        target_session = budget.reset_session(self.base / 'new-budget.json')
        with patch('llmplay.delivery._archive_digest', side_effect=AssertionError('no hashing')) as digest, \
                patch('llmplay.delivery.verify_archive', side_effect=AssertionError('no replay')) as verify:
            document = resume(source)
        self.assertEqual(digest.call_count, 0)
        self.assertEqual(verify.call_count, 0)
        self.assertEqual(document['archives'], previous['archives'])
        self.assertEqual(document['tasks'], previous['tasks'])
        self.assertTrue(document['tasks'][self.task]['complete'])
        self.assertEqual(next_task()['task_id'], 'demo_roof_47_doom_m200')
        self.assertNotIn(self.task, list_tasks())
        self.assertEqual(target_session.level(self.task)['search_ticks'], 345)
        self.assertEqual(target_session.level(self.task)['segment_rewrites'], 1)
        with patch('llmplay.delivery.verify_archive', return_value=True): resume(source)
        self.assertEqual(target_session.level(self.task)['search_ticks'], 345)

    def test_tampered_archive_is_restored_by_default_and_checked_only_in_full(self):
        from unittest.mock import patch
        from llmplay.delivery import record_archive, resume
        import zipfile
        for n in range(3): record_archive(self.archive(81000000 + n))
        source = self.base/'source'
        with zipfile.ZipFile(self.out/'out.zip') as bundle: bundle.extractall(source)
        prior = json.loads((source/'out/progress.json').read_text())
        damaged = source/'out'/prior['archives'][1]['archive']
        (damaged/'ticks.jsonl').write_text('tampered\n')
        os.environ['LLMPLAY_OUT'] = str(self.base/'resumed')
        with patch('llmplay.delivery._archive_digest', side_effect=AssertionError('no hashing')), \
                patch('llmplay.delivery.verify_archive', side_effect=AssertionError('no replay')) as verify:
            document = resume(source)
        self.assertEqual(verify.call_count, 0)
        self.assertEqual(sum(e['valid'] for e in document['archives']), 3)
        self.assertEqual((Path(os.environ['LLMPLAY_OUT'])/prior['archives'][1]['archive']/'ticks.jsonl').read_text(), 'tampered\n')
        def replay(archive, **kwargs):
            if archive == damaged: raise ValueError('tick mismatch')
            return True
        os.environ['LLMPLAY_OUT'] = str(self.base/'full')
        with patch('llmplay.delivery.verify_archive', side_effect=replay) as verify:
            document = resume(source, verify='full')
        self.assertEqual(verify.call_count, 3)
        self.assertEqual(sum(e['valid'] for e in document['archives']), 2)
        invalid = next(e for e in document['archives'] if not e['valid'])
        self.assertEqual(invalid['seed'], 81000001)
        self.assertIn('tick mismatch', invalid['invalid_reason'])

    def test_default_resume_preserves_invalid_record_and_full_can_revalidate(self):
        from unittest.mock import patch
        from llmplay.delivery import record_archive, resume
        import zipfile
        record_archive(self.archive(81000000))
        source = self.base/'source'
        with zipfile.ZipFile(self.out/'out.zip') as bundle: bundle.extractall(source)
        p = source/'out/progress.json'
        prior = json.loads(p.read_text())
        prior['archives'][0]['valid'] = False
        prior['archives'][0]['won'] = False
        prior['archives'][0]['robust'] = False
        p.write_text(json.dumps(prior))
        os.environ['LLMPLAY_OUT'] = str(self.base/'resumed')
        with patch('llmplay.delivery._archive_digest', side_effect=AssertionError('no hashing')), \
                patch('llmplay.delivery.verify_archive', side_effect=AssertionError('no replay')) as verify:
            document = resume(source)
        self.assertEqual(verify.call_count, 0)
        self.assertEqual(document['archives'], prior['archives'])
        self.assertEqual(document['tasks'][self.task]['wins'], 0)
        os.environ['LLMPLAY_OUT'] = str(self.base/'full')
        with patch('llmplay.delivery.verify_archive', return_value=True) as verify:
            document = resume(source, verify='full')
        self.assertEqual(verify.call_count, 1)
        self.assertTrue(document['archives'][0]['valid'])

    def test_parallel_full_resume_matches_serial_and_caps_engines(self):
        from unittest.mock import patch
        from contextlib import redirect_stdout
        import io
        import threading
        import time
        from llmplay.delivery import record_archive, resume
        for n in range(12): record_archive(self.archive(81000000 + n))
        source = self.out/'out.zip'
        active, peak = 0, 0
        lock = threading.Lock()
        def verify(archive, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(.01)
                if json.loads((archive/'meta.json').read_text())['task']['seed'] % 3 == 0:
                    raise ValueError('test invalid')
                return True
            finally:
                with lock: active -= 1
        results = []
        for workers in (1, 8):
            peak = 0
            os.environ['LLMPLAY_OUT'] = str(self.base/f'resumed-{workers}')
            output = io.StringIO()
            with patch('llmplay.delivery.os.cpu_count', return_value=workers), \
                    patch('llmplay.delivery.verify_archive', side_effect=verify) as replay, redirect_stdout(output):
                results.append(resume(source, verify='full'))
            self.assertEqual(replay.call_count, 12)
            self.assertEqual(peak, min(workers, 2))
            self.assertIn('resume verify: 10/12', output.getvalue())
        for key in ('archives', 'tasks', 'layers', 'run_id', 'complete'):
            self.assertEqual(results[0][key], results[1][key])

    def test_resume_rejects_unknown_verify_mode(self):
        from llmplay import resume
        with self.assertRaisesRegex(ValueError, 'verify'):
            resume(self.base, verify='other')

    def test_default_resume_needs_progress_and_does_not_fall_back_to_replay(self):
        from unittest.mock import patch
        from llmplay import resume
        archive = self.archive(81000000)
        with patch('llmplay.delivery.verify_archive') as replay:
            with self.assertRaisesRegex(ValueError, 'progress.json'):
                resume(archive)
        replay.assert_not_called()

    def test_default_resume_preserves_non_digest_path_and_ignores_unlisted_archive(self):
        from unittest.mock import patch
        from llmplay.delivery import record_archive, resume
        import shutil
        import zipfile
        record_archive(self.archive(81000000))
        source = self.base/'source'
        with zipfile.ZipFile(self.out/'out.zip') as bundle: bundle.extractall(source)
        p = source/'out/progress.json'
        prior = json.loads(p.read_text())
        archive = source/'out'/prior['archives'][0]['archive']
        relative = 'archives/saved-archive'
        archive.rename(source/'out'/relative)
        prior['archives'][0]['archive'] = relative
        p.write_text(json.dumps(prior))
        (source/'out'/relative/'meta.json').write_text('opaque metadata')
        shutil.copytree(self.archive(81000001), source/'out/archives/unlisted')
        os.environ['LLMPLAY_OUT'] = str(self.base/'resumed')
        with patch('llmplay.delivery._archive_digest', side_effect=AssertionError('no hashing')), \
                patch('llmplay.delivery.verify_archive', side_effect=AssertionError('no replay')), \
                patch('llmplay.delivery._entry', side_effect=AssertionError('no metadata validation')):
            document = resume(source)
        self.assertEqual(document['archives'], prior['archives'])
        self.assertEqual((Path(os.environ['LLMPLAY_OUT'])/relative/'meta.json').read_text(), 'opaque metadata')
        self.assertFalse((Path(os.environ['LLMPLAY_OUT'])/'archives/unlisted').exists())

    def test_resume_invalid_archive_does_not_count_and_seed_is_retried(self):
        from unittest.mock import patch
        from llmplay import resume, next_task
        from llmplay.delivery import record_archive
        archive = self.archive(81000000)
        record_archive(archive)
        os.environ['LLMPLAY_OUT'] = str(self.base / 'new')
        with patch('llmplay.delivery.verify_archive', side_effect=ValueError('tick mismatch')):
            document = resume(archive, verify='full')
        self.assertEqual(document['tasks'][self.task]['wins'], 0)
        self.assertFalse(document['archives'][0]['valid'])
        self.assertIn('tick mismatch', document['archives'][0]['invalid_reason'])
        self.assertIn(81000000, document['tasks'][self.task]['available_seeds'])
        self.assertEqual(next_task()['seed'], document['tasks'][self.task]['available_seeds'][0])

    def test_resume_marks_missing_archive_invalid(self):
        import shutil
        from llmplay import resume
        from llmplay.delivery import record_archive
        record_archive(self.archive(81000000))
        import zipfile
        source = self.base / 'missing'
        with zipfile.ZipFile(self.out/'out.zip') as bundle: bundle.extractall(source)
        shutil.rmtree(source/'out'/'archives')
        os.environ['LLMPLAY_OUT'] = str(self.base/'new')
        document = resume(source)
        self.assertEqual(document['tasks'][self.task]['wins'], 0)
        self.assertFalse(document['archives'][0]['valid'])
        self.assertIn('缺失', document['archives'][0]['invalid_reason'])

    def test_resume_restores_policy_and_saved_run_files(self):
        from unittest.mock import patch
        from llmplay import resume
        from llmplay.delivery import record_archive
        self.out.mkdir()
        (self.out/'policy.py').write_text('def policy(g): g.wait(30)\n')
        (self.out/'run.json').write_text('{"tick": 120}\n')
        record_archive(self.archive(81000000))
        source = self.out/'out.zip'
        target = self.base/'new'
        os.environ['LLMPLAY_OUT'] = str(target)
        with patch('llmplay.delivery.verify_archive', return_value=True): resume(source)
        self.assertEqual((target/'policy.py').read_bytes(), (self.out/'policy.py').read_bytes())
        self.assertEqual((target/'run.json').read_bytes(), (self.out/'run.json').read_bytes())

    def test_resume_incomplete_archive_keeps_task_identity(self):
        from llmplay import resume
        archive = self.archive(81000000)
        (archive/'ticks.jsonl').unlink()
        document = resume(archive, verify='full')
        entry = document['archives'][0]
        self.assertEqual(entry['task_id'], self.task)
        self.assertEqual(entry['seed'], 81000000)
        self.assertFalse(entry['valid'])

    def test_same_seed_multiple_exports_count_once_and_best_is_used(self):
        from llmplay.delivery import record_archive, progress
        record_archive(self.archive(81000000, robust=False))
        record_archive(self.archive(81000000, robust=True))
        record_archive(self.archive(81000000, won=False))
        state = progress()['tasks'][self.task]
        self.assertEqual((state['wins'], state['robust']), (1, 1))

    def test_robustness_counts_actual_checkpoint_not_passed_flag(self):
        from llmplay.delivery import record_archive, progress
        path = self.archive(81000000)
        meta = json.loads((path / 'meta.json').read_text())
        meta['robustness']['checkpoints']['10']['k'] = 7
        (path / 'meta.json').write_text(json.dumps(meta))
        record_archive(path)
        self.assertEqual(progress()['tasks'][self.task]['robust'], 0)

    def test_all_layers_must_meet_the_completion_standard(self):
        from unittest.mock import patch
        from llmplay.delivery import _derive, next_task
        plan = json.loads((ROOT/'configs/llmplay_seeds.json').read_text())
        entries = []
        for task, seeds in plan['tasks'].items():
            for seed in seeds[:5]:
                entries.append(dict(task_id=task, seed=seed, valid=True, won=True, robust=True,
                                    robustness={}, outcome={'final_tick': 1}))
        document = _derive(entries)
        self.assertTrue(document['complete'])
        with patch('llmplay.delivery.progress', return_value=document): self.assertIsNone(next_task())
        entries.pop()
        document = _derive(entries)
        self.assertFalse(document['complete'])
        with patch('llmplay.delivery.progress', return_value=document): self.assertEqual(next_task()['layer'], 'E')

    def test_budget_is_per_task_and_manual_never_exhausts(self):
        first = Game(self.task)
        second = Game('demo_roof_46_doom_m300')
        self.session.note_ticks(first.budget_key, 10**12, 'manual')
        self.assertEqual(self.session.level(second.budget_key)['total_ticks'], 0)
        self.assertFalse(self.session.level(first.budget_key)['exhausted'])
        first.close()
        second.close()

    def test_budget_import_does_not_reopen_search_or_reset_rewrites(self):
        self.session.note_ticks(self.task, self.session.search_limit, 'search')
        for _ in range(8): self.session.note_rewind(self.task, 10, 20)
        previous = self.session.snapshot()
        restored = budget.reset_session(self.base/'restored-budget.json')
        restored.import_budget(previous)
        with self.assertRaisesRegex(RuntimeError, '搜索已停止'): restored.require_search(self.task)
        with self.assertRaisesRegex(RuntimeError, '退回更早'): restored.note_rewind(self.task, 10, 20)
        restored.note_ticks(self.task, 100, 'manual')
        restored.import_budget(previous)
        self.assertEqual(restored.level(self.task)['manual_ticks'], 100)

    def test_list_tasks_skips_full_seeds_and_returns_all_unfinished_seeds(self):
        from llmplay import list_tasks
        from llmplay.delivery import record_archive
        self.assertEqual(len(list_tasks()[self.task]['seeds']), 40)
        record_archive(self.archive(81000000))
        self.assertNotIn(81000000, list_tasks()[self.task]['seeds'])
        self.assertEqual(len(list_tasks()[self.task]['seeds']), 39)

    def test_resume_rejects_path_traversal(self):
        import zipfile
        from llmplay import resume
        path = self.base/'unsafe.zip'
        with zipfile.ZipFile(path, 'w') as bundle: bundle.writestr('../escape', 'bad')
        with self.assertRaisesRegex(ValueError, '不安全'): resume(path)


if __name__ == '__main__':
    unittest.main()
