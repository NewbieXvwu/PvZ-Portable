"""task29 第 2 步：课程控制器的元规则单测（不跑引擎、不读真实清单）。"""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]

from curriculum_queue import (DEMOTE_BELOW, FLOOR_ROUNDS, FLOOR_SHARE, GIB, MEMORY_RESERVE_BYTES,
                              CURRICULUM_LADDERS, MIN_SAMPLES, PACK_BUILDER_BYTES, PROMOTE_ABOVE,
                              WINDOW, _descendant_rss,
                              adjust, allocate, build_plan, cell_level, command_for, label_for,
                              load_demo_start_cells, load_teacher_rates, make_state, pack_command,
                              pack_labels, parse_task_id, plan_round, rate, reconcile_cells,
                              reconcile_start_cells, startup_worker_estimate, student_rate,
                              training_base_ids, worker_count)

VALUES = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
NAMES = {'m125': 1.25, 'm150': 1.5, 'm175': 1.75, 'm200': 2.0, 'm250': 2.5, 'm300': 3.0}
N10_LEVELS = (11, 12, 13, 14, 16, 17, 18, 19)


def cell(wins, games, mult_index=0, last_round=0, level=7):
    return dict(family='f', level=level, mult_index=mult_index, last_round=last_round,
                windows={str(VALUES[mult_index]): {'recent': [1] * wins + [0] * (games - wins),
                                                   'wins': wins, 'games': games}})


class CurriculumRulesTest(unittest.TestCase):
    def test_zones_admission_only_measured_roof_and_one_fog_family(self):
        ids = training_base_ids()
        self.assertEqual({name for name in ids if name.startswith('roof_')},
                         {'roof_l41_train_r7'})
        # task70 B6: fog_user_v2 (human_v2) replaces fog_user; task79b admits the measured L36-L38
        # human_v2 bases; L39 remains demonstration-only; fog_f8/f8p stay out.
        self.assertEqual({name for name in ids if name.startswith('fog')},
                         {'fog_user_v2_l31_snow', 'fog79b_l36_best', 'fog79b_l37_best',
                          'fog79b_l38_best'})
        self.assertIn('pool_l25_orig_train', ids)
        self.assertEqual(CURRICULUM_LADDERS['native_l25'], [0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])

    def test_parse_task_id(self):
        self.assertEqual(parse_task_id('day_l7_train_d7_m150', NAMES), ('day_l7_train_d7', 1.5))
        self.assertEqual(parse_task_id('night_l20_train_n10_orig_m300', NAMES),
                         ('night_l20_train_n10_orig', 3.0))
        self.assertEqual(parse_task_id('pool_l21_train_p9k', NAMES), ('pool_l21_train_p9k', 1.0))

    def test_promote_and_demote(self):
        cells = {'high': cell(9, 10), 'low': cell(3, 10), 'small': cell(4, 4),
                 'top': cell(9, 10, mult_index=len(VALUES) - 1), 'bottom': cell(2, 10)}
        changes = adjust(cells, VALUES)
        self.assertEqual(cells['high']['mult_index'], 1)
        self.assertEqual(cells['low']['mult_index'], 0)          # 1.0 不再降
        self.assertEqual(cells['small']['mult_index'], 0)        # 样本不足
        self.assertEqual(cells['top']['mult_index'], len(VALUES) - 1)   # 已到顶
        self.assertEqual(len(changes), 1)
        # 升到 1.25 后再按 1.25 的窗口降回来
        cells['high']['windows']['1.25'] = {'recent': [0] * 10, 'wins': 0, 'games': 10}
        adjust(cells, VALUES)
        self.assertEqual(cells['high']['mult_index'], 0)

    def test_sliding_window_rate(self):
        bucket = {'recent': [0] * 200 + [1] * 56, 'wins': 56, 'games': 256}
        cells = {'c': dict(family='f', level=7, mult_index=0, last_round=0, windows={'1.0': bucket})}
        self.assertAlmostEqual(rate(cells['c'], 1.0), 56 / 256)
        self.assertLessEqual(len(bucket['recent']), WINDOW)

    def test_floor_serves_starved_cells_first(self):
        # last_round=0 的格（从未采过）最先轮到；保底组局数减半；预算封顶。
        cells = {f'c{i}': cell(6, 10, mult_index=1, last_round=1) for i in range(4)}
        cells.update({f's{i}': cell(0, 10, mult_index=0, last_round=0) for i in range(4)})
        picks = allocate(cells, VALUES, budget=60, per_group=10, round_index=5)
        floor_games = 10 // FLOOR_SHARE
        self.assertEqual([p[0] for p in picks[:2]], ['s0', 's1'])   # 饿死格按字母序先轮到
        self.assertEqual([p[2] for p in picks[:2]], [floor_games] * 2)   # 保底组是小局数
        self.assertTrue(all(p[2] == 10 for p in picks[2:]))    # 其余组按整组局数
        self.assertEqual([p[0] for p in picks[2:4]], ['s2', 's3'])  # 没轮到保底的饿死格在可教排序里最前
        self.assertLessEqual(sum(p[2] for p in picks), 60)
        for base, multiplier, games in picks:
            self.assertIn(multiplier, VALUES)

    def test_floor_bounds_starvation_over_rounds(self):
        # 预算紧张下连续模拟多轮（last_round 由"组完成"写入，如 run_round 的落盘）：
        # 任何格两次被采的间隔不超过 FLOOR_ROUNDS + 1 轮。
        cells = {f'c{i:02d}': cell(5, 10, last_round=0) for i in range(20)}
        last_seen = {name: 0 for name in cells}
        for round_index in range(1, 13):
            picks = allocate(cells, VALUES, budget=50, per_group=10, round_index=round_index)
            for base, _multiplier, _games in picks:
                gap = round_index - last_seen[base]
                self.assertLessEqual(gap, FLOOR_ROUNDS + 1, base)
                cells[base]['last_round'] = round_index
                last_seen[base] = round_index
        self.assertEqual(max(last_seen.values()), 12)          # 每个格都真的被轮到过

    def test_allocate_does_not_mutate_last_round(self):
        # task59：last_round 只由 run_round 在组完成时写；plan/allocate 不得改状态。
        cells = {'a': cell(5, 10, last_round=0), 'b': cell(5, 10, last_round=1)}
        before = {name: value['last_round'] for name, value in cells.items()}
        allocate(cells, VALUES, budget=30, per_group=10, round_index=7)
        self.assertEqual({name: value['last_round'] for name, value in cells.items()}, before)

    def test_crash_between_groups_replans_unrun_groups(self):
        # 组间崩溃重启：重新 plan 时本轮已完成的组不重采，没跑的组照常排出。
        bases = ('day_l7_train_d7', 'night_l7_train_n7', 'pool_l21_train_p9k')
        tasks = {}
        for base in bases:
            for mult, suffix in [(1.0, ''), (1.25, '_m125')]:
                task_id = base + suffix
                tasks[task_id] = dict(task_id=task_id, level=7, deck=[0, 1, 2, 3, 4, 5, 7],
                                      zombie_count_multiplier=mult, seeds=list(range(40)))
        state = dict(round=0, offsets={}, cells={base: cell(5, 10) for base in bases})
        manifests = {task_id: Path('/tmp/panel.json') for task_id in tasks}
        plan1, _ = plan_round(state, tasks, NAMES, VALUES, manifests, budget=40, per_group=10)
        self.assertEqual([item['base'] for item in plan1], list(bases))
        self.assertEqual([item['games'] for item in plan1], [10, 10, 10])
        for item in plan1[:2]:                                 # 前两组完成，然后控制器崩溃
            state['cells'][item['base']]['last_round'] = 1
            state['offsets'][item['task_id']] = item['offset'] + item['games']
        plan2, _ = plan_round(state, tasks, NAMES, VALUES, manifests, budget=40, per_group=10)
        completed = {item['base'] for item in plan1[:2]}
        self.assertFalse(completed & {item['base'] for item in plan2})
        self.assertEqual([item['base'] for item in plan2], ['pool_l21_train_p9k'])
        self.assertEqual(plan2[0]['offset'], 0)                # 偏移接在原计划之后

    def test_starved_cell_gets_floor_even_with_tight_budget(self):
        # 预算只够一个保底组时，最久没采的格拿到保底局数（减半），不为零。
        cells = {'old': cell(5, 10, last_round=0), 'new': cell(5, 10, last_round=1)}
        picks = allocate(cells, VALUES, budget=6, per_group=10, round_index=9)
        self.assertEqual([(p[0], p[2]) for p in picks], [('old', 10 // FLOOR_SHARE)])

    def test_teacher_wins_student_loses_ranks_first(self):
        # L20 情形：学生 0/10、老师 0.9 -> 排在中难度格前面（task57 的核心诉求）。
        cells = {
            'l20': cell(0, 10, level=20),   # 学生全败，老师 0.9 -> 差距 0.9
            'mid': cell(6, 10, level=17),   # 0.6，老师未知按可教 -> 差距 0.4
            'l24': cell(0, 10, level=24),   # 学生全败，老师 0.1 -> 差距 0.1，沉底
        }
        teacher = {'level:20': 0.9, 'level:24': 0.1}
        picks = allocate(cells, VALUES, budget=20, per_group=10, round_index=1, teacher=teacher)
        bases = [p[0] for p in picks]
        self.assertEqual(bases, ['l20', 'mid'])            # 预算只够两组，l24 被挤出

    def test_unknown_teacher_and_student_gap_tiebreaks_by_age(self):
        # 老师都未知时按 1 - 学生胜率排；同分时最久没采的先。
        cells = {
            'fresh_hard': cell(0, 10, last_round=1),        # 差距 1.0（上一轮采过）
            'old_hard': cell(0, 10, last_round=0),          # 差距 1.0，更久没采
            'easy': cell(9, 10),                            # 差距 0.1
        }
        picks = allocate(cells, VALUES, budget=30, per_group=10, round_index=2)
        self.assertEqual([p[0] for p in picks], ['old_hard', 'fresh_hard', 'easy'])

    def test_build_plan_cold_start_uses_pending_volume(self):
        # round 2 的真实规模：202 个标签、约 5 万局 ≈ 每标签 0.3 GiB 原始档案，MemAvailable 40 GiB。
        # 冷启动并发应明显大于 1（旧算法按已建段估得 2 GiB → 并发 1），且峰值不超可用预算。
        pending = [300 * 1024 * 1024] * 202
        workers, per_builder = build_plan(pending, recipe_workers=32, available=40 * GIB)
        self.assertGreater(workers, 1)
        self.assertEqual(per_builder, PACK_BUILDER_BYTES + pending[0])
        self.assertLessEqual(workers * per_builder, 40 * GIB - MEMORY_RESERVE_BYTES)
        self.assertEqual(workers, min(32, (40 * GIB - MEMORY_RESERVE_BYTES) // per_builder))

    def test_build_plan_bounds(self):
        # 配方值是上限；待建段为空并发 1（pack 无活可干）；可用内存极小时并发仍 >=1。
        self.assertEqual(build_plan([GIB] * 5, 2, None), (2, 3 * GIB))
        self.assertEqual(build_plan([], 8, 100 * GIB), (1, PACK_BUILDER_BYTES))
        self.assertEqual(build_plan([GIB], 8, 7 * GIB), (1, 3 * GIB))
        self.assertEqual(build_plan([GIB] * 9, 32, 100 * GIB), (9, 3 * GIB))

    def test_pack_memory_limit_matches_build_plan_per_builder(self):
        # pack() 的单建造者假设与 build_plan 一致：2 GiB 底数 + 该批最大的标签原始档案。
        import pack_training_rows as packer
        with tempfile.TemporaryDirectory() as tmp:
            big = Path(tmp) / 'big' / 'frames.npz'
            big.parent.mkdir()
            with big.open('wb') as handle:
                handle.truncate(3 * GIB)                     # 稀疏文件，只占 st_size
            jobs = [('d', 'c', None, None, [big])]
            self.assertEqual(packer.pack_memory_limit(40 * GIB, jobs),
                             (40 * GIB - MEMORY_RESERVE_BYTES) // (PACK_BUILDER_BYTES + 3 * GIB))
            self.assertEqual(packer.pack_memory_limit(40 * GIB, [('d', 'c', None, None, [])]),
                             (40 * GIB - MEMORY_RESERVE_BYTES) // PACK_BUILDER_BYTES)
            self.assertEqual(packer.pack_memory_limit(GIB, jobs), 1)

    def test_pending_builds_uses_merged_catalog_and_segment_contract(self):
        # 与 _prepare_corpus 同一合并视图（采集优先）、与 pack() 同一套指纹函数：
        # 视图链接不齐直接待建；链接齐全时按 pack 的指纹查段文件，段在即已建。
        from pvz_recipe import load_recipe, training_row_options, write_recipe
        from pvz_training_pack import fingerprint_paths, pack_options, segment_options, segment_path
        import curriculum_queue as queue
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, coll = root / 'base', root / 'coll'
            (base / 's1' / 'pool_a').mkdir(parents=True)
            (base / 's1' / 'pool_a' / 'frames.jsonl').write_bytes(b'x' * 100)
            (coll / 's1' / 'pool_a').mkdir(parents=True)
            (coll / 's1' / 'pool_a' / 'frames.jsonl').write_bytes(b'y' * 200)   # 采集同位覆盖
            (coll / 's2' / 'pool_b').mkdir(parents=True)
            (coll / 's2' / 'pool_b' / 'frames.jsonl').write_bytes(b'z' * 50)
            recipe = load_recipe(ROOT / 'recipes/dw4i.json')
            recipe['data'] = str(base)
            recipe_path = root / 'recipe.json'
            write_recipe(recipe_path, recipe)
            pipeline = object.__new__(queue.TrainingPipeline)
            pipeline.args = SimpleNamespace(recipe=recipe_path, frames=coll, log=root / 'log')
            pipeline.directory = root / 'pipeline'
            self.assertEqual(dict(pipeline.pending_builds(['pool_a', 'pool_b'], recipe)),
                             {'pool_a': 200, 'pool_b': 50})          # 视图还没有链接
            view = root / 'pipeline' / 'frames'
            (view / 's2' / 'pool_b').mkdir(parents=True)
            (view / 's2' / 'pool_b' / 'frames.jsonl').symlink_to(coll / 's2' / 'pool_b' / 'frames.jsonl')
            options = pack_options(**training_row_options(recipe, ['pool_a', 'pool_b']))
            fingerprint = fingerprint_paths(view, [view / 's2' / 'pool_b' / 'frames.jsonl'])
            target = segment_path(view, segment_options(options, 'pool_b', fingerprint))
            target.parent.mkdir(parents=True)
            target.write_bytes(b'')
            self.assertEqual(dict(pipeline.pending_builds(['pool_a', 'pool_b'], recipe)), {'pool_a': 200})

    def test_load_teacher_rates_missing_file_and_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_teacher_rates(Path(tmp) / 'absent.json'), {})
            path = Path(tmp) / 'rates.json'
            path.write_text(json.dumps({'levels': {'20': {'wins': 29, 'games': 32}, '24': 0.1},
                                        'bases': {'pool_l21_train_p9k': 0.5}}))
            rates = load_teacher_rates(path)
            self.assertAlmostEqual(rates['level:20'], 29 / 32)
            self.assertAlmostEqual(rates['level:24'], 0.1)
            self.assertAlmostEqual(rates['pool_l21_train_p9k'], 0.5)

    def test_plan_round_offsets_and_exhaustion(self):
        tasks = {}
        for mult, suffix in [(1.0, ''), (1.25, '_m125')]:
            task_id = 'day_l7_train_d7' + suffix
            tasks[task_id] = dict(task_id=task_id, level=7, deck=[0, 1, 2, 3, 4, 5, 7],
                                  zombie_count_multiplier=mult, seeds=list(range(20)))
        state = dict(round=0, offsets={}, cells={'day_l7_train_d7': cell(5, 10)})
        manifests = {task_id: Path('/tmp/panel.json') for task_id in tasks}
        plan, _ = plan_round(state, tasks, NAMES, VALUES, manifests,
                             budget=10, per_group=10)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]['task_id'], 'day_l7_train_d7')
        self.assertEqual(plan[0]['offset'], 0)
        state['offsets']['day_l7_train_d7'] = 15
        with self.assertRaises(SystemExit):
            plan_round(state, tasks, NAMES, VALUES, manifests,
                       budget=10, per_group=10)

    def test_stop_flag_stops_before_any_group(self):
        import curriculum_queue as queue
        with tempfile.TemporaryDirectory() as tmp:
            flag = Path(tmp) / 'stop'
            flag.write_text('')
            original = queue.STOP_FLAG
            queue.STOP_FLAG = flag
            try:
                args = SimpleNamespace(log=Path(tmp) / 'log', state=Path(tmp) / 'state.json', out=Path(tmp),
                                       workers=2)
                state = dict(round=0, cells={}, offsets={}, log=[])
                plan = [dict(base='x', multiplier=1.0, task_id='t', offset=0, games=2,
                             manifest='/m', label='l')]
                self.assertFalse(queue.run_round(args, state, plan, []))
                self.assertEqual(state['round'], 0)
                self.assertEqual(state['offsets'], {})
            finally:
                queue.STOP_FLAG = original

    def test_student_rate_uses_current_window_then_latest(self):
        self.assertAlmostEqual(student_rate(cell(2, 10), VALUES[0]), 0.2)
        # 刚换档还没数据：退回最近一个非空窗口（执行现实）。
        self.assertAlmostEqual(student_rate(cell(1, 10, mult_index=1), 2.0), 0.1)
        # 全无数据：None，第一个组保持 β=0 的探针。
        self.assertIsNone(student_rate(cell(0, 0), VALUES[0]))

    def test_command_for_passes_dagger_beta(self):
        args = SimpleNamespace(python=Path('/nonexistent/python'), rollout_policy=Path('/m.pt'),
                               static_value=Path('/sv.pt'), value=Path('/v.pt'), workers=3,
                               executable=Path('/exe'), resource_dir=Path('/res'),
                               frames=Path('/frames'), out=Path('/out'))
        base = command_for(args, Path('/panel.json'), 'day_l7_train_d7_m125', 0, 2, 'c1d7m125')
        self.assertEqual(base[base.index('--dagger-beta') + 1], '0')
        mixed = command_for(args, Path('/panel.json'), 'day_l7_train_d7_m125', 0, 2, 'c1d7m125', beta=0.5)
        self.assertEqual(mixed[mixed.index('--dagger-beta') + 1], '0.5')

    def test_run_round_mixes_beta_for_starved_cells(self):
        import curriculum_queue as queue
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            commands = []

            def fake_run(command, **_kwargs):
                commands.append(command)
                return SimpleNamespace(returncode=0)

            original_flag, original_run = queue.STOP_FLAG, queue.subprocess.run
            queue.STOP_FLAG = root / 'absent-stop-flag'   # 不存在：组间不停
            queue.subprocess.run = fake_run
            try:
                for wins, games, expected in [(0, 10, '0.5'), (5, 10, '0'), (0, 0, '0')]:
                    label = f'c1d7m{wins}g{games}'
                    folder = root / 'out' / f'{label}_2'
                    folder.mkdir(parents=True)
                    rows = [{'won': True}, {'won': False}]
                    (folder / 'games.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
                    args = SimpleNamespace(state=root / 'state.json', log=root / 'log', out=root / 'out',
                                           python=Path('/nonexistent/python'), rollout_policy=Path('/m.pt'),
                                           static_value=Path('/sv.pt'), value=Path('/v.pt'), workers=2,
                                           executable=Path('/exe'), resource_dir=Path('/res'),
                                           frames=Path('/frames'),
                                           beta_threshold=0.1, starved_beta=0.5, probe_interval=3,
                                           seed_ledger=root / 'gate_ledger.json')
                    state = dict(round=0, offsets={}, log=[],
                                 cells={'day_l7_train_d7': cell(wins, games)})
                    plan = [dict(base='day_l7_train_d7', multiplier=1.0, task_id='day_l7_train_d7',
                                 offset=0, games=2, manifest='/m', label=label)]
                    self.assertTrue(queue.run_round(args, state, plan, []))
                    self.assertEqual(commands[-1][commands[-1].index('--dagger-beta') + 1], expected)
                    self.assertIn(f'beta={expected}', (root / 'log').read_text())
                    self.assertEqual(state['log'][0]['groups'][0]['beta'], float(expected))
            finally:
                queue.STOP_FLAG, queue.subprocess.run = original_flag, original_run

    def test_mixed_groups_bypass_student_window_with_probe_cadence(self):
        # 修正 2b：β>0 组的胜负不进学生胜率窗口（一部分步是搜索在走），另存 mixed；
        # 每 probe_interval 组保留一个 β=0 探针组，窗口只吃探针与普通 β0 组。
        import curriculum_queue as queue
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            commands = []

            def fake_run(command, **_kwargs):
                commands.append(command)
                return SimpleNamespace(returncode=0)

            original_flag, original_run = queue.STOP_FLAG, queue.subprocess.run
            queue.STOP_FLAG = root / 'absent-stop-flag'
            queue.subprocess.run = fake_run
            try:
                args = SimpleNamespace(state=root / 'state.json', log=root / 'log', out=root / 'out',
                                       python=Path('/nonexistent/python'), rollout_policy=Path('/m.pt'),
                                       static_value=Path('/sv.pt'), value=Path('/v.pt'), workers=2,
                                       executable=Path('/exe'), resource_dir=Path('/res'),
                                       frames=Path('/frames'), beta_threshold=0.1, starved_beta=0.5,
                                       probe_interval=3)
                state = dict(round=0, offsets={}, log=[],
                             cells={'day_l7_train_d7': cell(0, 10)})   # 学生全败
                for round_index in (1, 2, 3):
                    label = f'c{round_index}d7m100'
                    folder = root / 'out' / f'{label}_2'
                    folder.mkdir(parents=True)
                    rows = [{'won': True}, {'won': False}]   # 混入窗口会造出 50% 假胜率
                    (folder / 'games.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
                    plan = [dict(base='day_l7_train_d7', multiplier=1.0, task_id='day_l7_train_d7',
                                 offset=0, games=2, manifest='/m', label=label)]
                    self.assertTrue(queue.run_round(args, state, plan, []))
                betas = [entry['groups'][0]['beta'] for entry in state['log']]
                self.assertEqual(betas, [0.5, 0.5, 0.0])     # 第 3 组是 β=0 探针
                self.assertEqual(commands[-1][commands[-1].index('--dagger-beta') + 1], '0')
                starved_cell = state['cells']['day_l7_train_d7']
                bucket = starved_cell['windows']['1.0']
                self.assertEqual(bucket['games'], 12)        # 旧窗口 + 探针组
                self.assertEqual(bucket['recent'], [0] * 10 + [1, 0])
                mixed = bucket['mixed']
                self.assertEqual((mixed['wins'], mixed['games']), (2, 4))
                self.assertEqual(starved_cell['mixed_since_probe'], 0)
                self.assertAlmostEqual(student_rate(starved_cell, 1.0), 1 / 12)   # 探针口径，不是 50%
                self.assertEqual(adjust({'d': starved_cell}, VALUES), [])   # 不会因 mixed 误升档
                # allocate 的可教差距不受 mixed 影响：全败格仍排在 0.5 胜率格前
                # （若 mixed 污染窗口，两者差距同为 0.5，按更久未采会排反）。
                rival = {'d': starved_cell, 'mid': cell(5, 10, last_round=1)}
                picks = allocate(rival, VALUES, budget=20, per_group=10, round_index=4)
                self.assertEqual([p[0] for p in picks], ['d', 'mid'])
            finally:
                queue.STOP_FLAG, queue.subprocess.run = original_flag, original_run

    def test_estimate_peak_prefers_measured_train_peak(self):
        # task61：训练的不可回收峰值是匿名内存——最近实测优先，否则按实测系数回退。
        import curriculum_queue as queue
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = object.__new__(queue.TrainingPipeline)
            pipeline.directory = Path(tmp)
            pipeline.build_peak = 0
            pipeline.last_train_peak = 5 * GIB
            self.assertEqual(pipeline.estimate_peak(), 5 * GIB)
            pipeline.last_train_peak = 0
            self.assertEqual(pipeline.estimate_peak(), queue.TRAIN_ANON_EXTRA)   # 无段 → 纯常数

    def test_admit_fails_fast_when_request_exceeds_mem_total(self):
        # task61：required > MemTotal 时报错让训练任务失败退出，不静默死等。
        import curriculum_queue as queue
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = object.__new__(queue.TrainingPipeline)
            pipeline.directory = Path(tmp)
            pipeline.build_peak = 0
            pipeline.last_train_peak = 0
            original = queue.mem_total_bytes
            queue.mem_total_bytes = lambda: GIB
            try:
                with self.assertRaisesRegex(RuntimeError, 'cannot fit'):
                    pipeline._admit()
            finally:
                queue.mem_total_bytes = original

    def test_command_measures_child_anon_peak(self):
        import threading

        import curriculum_queue as queue
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = object.__new__(queue.TrainingPipeline)
            pipeline.lock = threading.Lock()
            pipeline.lock_fd = None
            pipeline.process = None
            peak = pipeline._command([sys.executable, '-c', 'import time; time.sleep(2.5)'],
                                     Path(tmp) / 'cmd.log')
            self.assertIsInstance(peak, int)
            self.assertGreaterEqual(peak, 0)

    def test_pack_labels_union_the_log_and_the_base_corpus(self):
        state = {'log': [{'groups': [{'label': 'c1d7m150'}]},
                         {'groups': [{'label': 'c2d7m125'}, {'label': 'c1d7m150'}]}]}
        self.assertEqual(pack_labels(state), ['c1d7m150_fast_search', 'c2d7m125_fast_search'])
        self.assertEqual(pack_labels(state, 'base_b,base_a'),
                         ['base_a', 'base_b', 'c1d7m150_fast_search', 'c2d7m125_fast_search'])

    def test_recipe_pack_command_matches_actual_trainer_load_request(self):
        import argparse
        import json
        from unittest.mock import patch
        import train_rollout_policy as trainer
        import pack_training_rows as packer
        from pvz_recipe import load_recipe, training_row_options, write_recipe
        from pvz_training_pack import pack_options

        class Loaded(Exception):
            pass

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            variants = [load_recipe(ROOT / f'recipes/{name}.json')
                        for name in ('dw4i', 'dw5i', 'identity_dw3_s1')]
            variants.append(dict(variants[0], context=True, aux_weight=1.0, aux_horizon=3,
                                 cell_soft_weight=0.0))
            for i, recipe in enumerate(variants):
                with self.subTest(variant=i):
                    recipe.update(out=str(root / f'out{i}'), data=str(root / 'frames'))
                    path = root / f'recipe{i}.json'
                    write_recipe(path, recipe)
                    args = SimpleNamespace(python=Path('/nonexistent/python'),
                                           frames=root / 'frames', recipe=path)
                    command = pack_command(args, recipe['labels'])
                    self.assertEqual(command[0], sys.executable)
                    with patch.object(sys, 'argv', command[1:]), \
                         patch.object(packer, 'archive_fingerprint', return_value='same-sources'), \
                         patch.object(packer, 'pack', return_value=root / 'pack.npz') as build, \
                         patch.object(packer, 'prune', return_value=[]):
                        packer.main()
                    packed_options = build.call_args.args[1]
                    with patch.object(sys, 'argv', ['trainer', '--recipe', str(path)]), \
                         patch.object(trainer, 'Metrics'), \
                         patch.object(trainer, 'load', side_effect=Loaded) as load:
                        with self.assertRaises(Loaded):
                            trainer.main()
                    requested = dict(load.call_args.kwargs)
                    requested.pop('row_observer')
                    self.assertEqual(requested, training_row_options(load_recipe(path)))
                    expected = pack_options(**requested, sources='same-sources')
                    self.assertEqual(packed_options, expected)
                    self.assertEqual(load.call_args.args[0], args.frames)
                    self.assertIn('--prune', command)

    def test_removed_pack_flags_are_not_exposed(self):
        import subprocess
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/curriculum_queue.py'), '--help'],
                                capture_output=True, text=True, check=True)
        self.assertIn('--recipe', result.stdout)
        self.assertNotIn('--pack-', result.stdout)

    def test_labels(self):
        self.assertEqual(label_for(3, 'day_l7_train_d7', 1.5), 'c3d7d7m150')
        self.assertEqual(label_for(1, 'night_l12_train_n10', 1.0), 'c1n12n10m100')
        # task71: 示范起点格的标签记起点决策号，不记倍率
        self.assertEqual(label_for(2, 'demo_fog_33_cactus_snow_m200', 839, start=839),
                         'c2demofog33cactussnowm200s839')


class ReconcileCellsTest(unittest.TestCase):
    """task39：已有状态按当前训练集合增删格子（不上真实状态）。"""

    def panel_tasks(self, families=('n10p', 'n10q')):
        tasks = {}
        for family in families:
            for level in N10_LEVELS:
                tasks[f'night_l{level}_train_{family}'] = dict(
                    task_id=f'night_l{level}_train_{family}', level=level, zombie_count_multiplier=1.0)
                tasks[f'night_l{level}_train_{family}_m125'] = dict(
                    task_id=f'night_l{level}_train_{family}_m125', level=level,
                    zombie_count_multiplier=1.25)
        return tasks

    def n10p_state(self):
        cells = {}
        for index, level in enumerate(N10_LEVELS):
            base = f'night_l{level}_train_n10p'
            multiplier = VALUES[index % len(VALUES)]
            cells[base] = dict(family=f'night_l{level}', level=level, mult_index=index % len(VALUES),
                               last_round=index,
                               windows={str(multiplier): {'recent': [1] * index + [0] * (index % 3),
                                                          'wins': index, 'games': index + index % 3}})
        return dict(schema_version=1, panel='panel.json', round=2, cells=cells,
                    offsets={'night_l11_train_n10p_m125': 96, 'day_l7_train_d7_m125': 48}, log=[])

    def test_adds_n10q_and_retires_n10p(self):
        tasks = self.panel_tasks()
        state = self.n10p_state()
        original_cells = copy.deepcopy(state['cells'])
        original_offsets = copy.deepcopy(state['offsets'])
        bucket = {'1.0': {'recent': [1, 0], 'wins': 1, 'games': 2}}
        lines = reconcile_cells(state, tasks, NAMES, {'night_l11_train_n10q': bucket})
        self.assertEqual(len(lines), 16)                                   # 8 retired + 8 added
        self.assertFalse(any(base.endswith('_n10p') for base in state['cells']))
        self.assertEqual(sum(base.endswith('_n10q') for base in state['cells']), 8)
        for level in N10_LEVELS:
            base = f'night_l{level}_train_n10p'
            self.assertEqual(state['retired_cells'][base], original_cells[base])   # 原样保存
        self.assertTrue(any(line == 'cell retired: night_l11_train_n10p (superseded by night_n10q)'
                            for line in lines))
        added = state['cells']['night_l12_train_n10q']
        self.assertEqual(added, dict(family='night_l12', level=12, mult_index=0, last_round=0, windows={}))
        self.assertEqual(state['cells']['night_l11_train_n10q']['windows'], bucket)
        self.assertEqual(state['offsets'], original_offsets)

    def test_idempotent(self):
        tasks = self.panel_tasks()
        state = self.n10p_state()
        reconcile_cells(state, tasks, NAMES, {})
        snapshot = copy.deepcopy(state)
        self.assertEqual(reconcile_cells(state, tasks, NAMES, {}), [])
        self.assertEqual(state, snapshot)

    def test_restore_when_superseded_is_cleared(self):
        import curriculum_queue as queue
        tasks = self.panel_tasks()
        state = self.n10p_state()
        reconcile_cells(state, tasks, NAMES, {})
        retired_before = copy.deepcopy(state['retired_cells'])
        original = queue.SUPERSEDED
        queue.SUPERSEDED = {}
        try:
            lines = reconcile_cells(state, tasks, NAMES, {})
        finally:
            queue.SUPERSEDED = original
        self.assertEqual(len(lines), 8)
        self.assertTrue(all(line.startswith('cell restored: ') for line in lines))
        self.assertEqual(state['retired_cells'], {})
        for base, cell in retired_before.items():
            self.assertEqual(state['cells'][base], cell)
        self.assertEqual(sum(base.endswith('_n10q') for base in state['cells']), 8)   # n10q 不动

    def test_make_state_equals_empty_state_plus_reconcile(self):
        tasks = self.panel_tasks(('n10q',))
        tasks['day_l7_train_d7'] = dict(task_id='day_l7_train_d7', level=7, zombie_count_multiplier=1.0)
        windows = {'night_l11_train_n10q': {'1.0': {'recent': [1], 'wins': 1, 'games': 1}}}
        offsets = {'night_l11_train_n10q_m125': 48}
        made = make_state(tasks, NAMES, windows, dict(offsets), 'panel.json')
        manual = dict(schema_version=1, panel='panel.json', round=0, cells={},
                      offsets=dict(offsets), log=[])
        reconcile_cells(manual, tasks, NAMES, windows)
        self.assertEqual(made, manual)
        self.assertIn('day_l7_train_d7', made['cells'])
        self.assertEqual(len(made['cells']), 9)

    def test_training_ids_drop_n10p_keep_n10q(self):
        ids = training_base_ids()
        self.assertFalse(any(base.endswith('_n10p') for base in ids))
        self.assertEqual(sum(base.endswith('_n10q') for base in ids), 8)


class WorkerCountTest(unittest.TestCase):
    def test_scales_with_available_memory(self):
        per_worker = 900 * 1024 ** 2          # 0.9 GiB
        self.assertEqual(worker_count(20 * GIB, per_worker, 16), 15)    # (20-6)/0.9 = 15.5
        self.assertEqual(worker_count(10 * GIB, per_worker, 16), 4)     # (10-6)/0.9 = 4.4
        self.assertEqual(worker_count(28 * GIB, 512 * 1024 ** 2, 8), 8)  # 上限
        self.assertEqual(worker_count(4 * GIB, per_worker, 16), 1)      # 低于保留也不为 0
        self.assertEqual(MEMORY_RESERVE_BYTES, 6 * GIB)

    def test_unknown_memory_or_estimate_keeps_the_upper_limit(self):
        self.assertEqual(worker_count(None, 900 * 1024 ** 2, 16), 16)   # 没有 /proc
        self.assertEqual(worker_count(20 * GIB, 0, 16), 16)             # 没有实测值

    def test_descendant_rss_excludes_root_and_unrelated(self):
        table = {1: (0, 100), 10: (1, 1000), 11: (10, 500), 12: (1, 200), 20: (0, 9000)}
        self.assertEqual(_descendant_rss(1, table), 1700)

    def test_command_for_overrides_probe_workers_frames_and_out(self):
        args = SimpleNamespace(python=Path('/nonexistent/python'), workers=16,
                               rollout_policy=Path('/m/student.pt'), static_value=Path('/m/static.pt'),
                               value=Path('/m/value.pt'), executable=Path('/m/pvz-portable'),
                               resource_dir=Path('/m/resources'), frames=Path('/m/frames'),
                               out=Path('/m/out'))
        command = command_for(args, Path('/m/panel.json'), 'day_l7', 5, 1, 'oomprobe',
                              workers=1, frames=Path('/tmp/probe-frames'), out=Path('/tmp/probe'))
        self.assertEqual(command[command.index('--workers') + 1], '1')
        self.assertEqual(command[command.index('--frames') + 1], '/tmp/probe-frames')
        self.assertEqual(command[command.index('--out') + 1], '/tmp/probe/oomprobe_1')
        self.assertEqual(command[command.index('--seed-offset') + 1], '5')

    def test_startup_estimate_without_probe(self):
        import curriculum_queue as queue
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(log=Path(tmp) / 'log')
            original = queue.mem_available_bytes
            try:
                queue.mem_available_bytes = lambda: None       # 没有 /proc
                self.assertEqual(startup_worker_estimate(args, []), 0)
                queue.mem_available_bytes = lambda: 20 * GIB  # 有内存但没有计划
                self.assertEqual(startup_worker_estimate(args, []), queue.DEFAULT_WORKER_BYTES)
            finally:
                queue.mem_available_bytes = original

class TrainingRestartTest(unittest.TestCase):
    """task61 追加：杀掉后台训练进程 + 控制器重启后，任务干净重跑，不丢不重。"""

    def _job(self, tmp, status, round_index=1):
        folder = Path(tmp) / f'round_{round_index:06d}'
        folder.mkdir(parents=True, exist_ok=True)
        recipe = folder / 'recipe.json'
        recipe.write_text('{}')
        model = folder / 'model.pt'
        model.write_bytes(b'model-bytes')
        return dict(round=round_index, recipe=str(recipe), model=str(model), status=status)

    def test_init_requeues_non_done_jobs_and_keeps_done(self):
        from curriculum_queue import TrainingPipeline
        with tempfile.TemporaryDirectory() as tmp:
            state = {'training_jobs': [self._job(tmp, 'running'), self._job(tmp, 'failed', 2),
                                       self._job(tmp, 'done', 3)]}
            started = []
            def fake_run(self, job):
                started.append(job['round'])
                return {}
            args = SimpleNamespace(state=Path(tmp) / 'state.json',
                                   seed_ledger=Path(tmp) / 'gate_ledger.json')
            with mock.patch.object(TrainingPipeline, '_run', fake_run), \
                    mock.patch.object(TrainingPipeline, '_plan_build', lambda self, recipe: None):
                pipeline = TrainingPipeline(args, state)
            pipeline.executor.shutdown(wait=True)
            self.assertEqual(sorted(started), [1, 2])          # running/failed 都重跑
            statuses = {job['round']: job['status'] for job in state['training_jobs']}
            self.assertEqual(statuses, {1: 'queued', 2: 'queued', 3: 'done'})   # done 不动
            self.assertEqual(len(pipeline.futures), 2)

    def test_run_reuses_trained_json_without_retraining(self):
        # 训练完成（trained.json 已写）、评估前被杀：重启后凭 sha 校验复用报告，直接进评估，
        # _admit/_command（重训入口）被调到即失败。
        from curriculum_queue import TrainingPipeline
        from promotion_gate import sha as file_sha
        with tempfile.TemporaryDirectory() as tmp:
            job = self._job(tmp, 'queued')
            trained = {'model_sha256': file_sha(job['model']), 'recipe_sha256': file_sha(job['recipe']),
                       'round': job['round']}
            (Path(job['recipe']).parent / 'trained.json').write_text(json.dumps(trained))
            pipeline = TrainingPipeline.__new__(TrainingPipeline)
            pipeline.failed = False
            pipeline.args = SimpleNamespace(log=Path(tmp) / 'log')
            pipeline.lock_fd = None
            def forbidden(*_a, **_k):
                raise AssertionError('retrain path was entered')
            pipeline._admit = forbidden
            pipeline._command = forbidden
            pipeline._prepare_corpus = forbidden
            pipeline._evaluate = lambda _job: {'promotion': {'approved': False, 'reasons': ['x']}}
            report = pipeline._run(copy.deepcopy(job))
            self.assertEqual(report['round'], job['round'])
            self.assertEqual(report['promotion']['approved'], False)
            self.assertTrue((Path(job['recipe']).parent / 'complete.json').is_file())

    def test_run_rejects_changed_trained_candidate(self):
        # trained.json 里的 sha 和模型/recipe 对不上（半截文件或被改动）：拒绝复用，报错留给人看。
        from curriculum_queue import TrainingPipeline
        with tempfile.TemporaryDirectory() as tmp:
            job = self._job(tmp, 'queued')
            trained = {'model_sha256': '0' * 64, 'recipe_sha256': '0' * 64, 'round': job['round']}
            (Path(job['recipe']).parent / 'trained.json').write_text(json.dumps(trained))
            pipeline = TrainingPipeline.__new__(TrainingPipeline)
            pipeline.failed = False
            pipeline.args = SimpleNamespace(log=Path(tmp) / 'log')
            pipeline.lock_fd = None
            with self.assertRaises(ValueError):
                pipeline._run(copy.deepcopy(job))
            self.assertTrue(pipeline.failed)

    def test_build_plan_prefers_measured_per_builder(self):
        # task61 追加：单建造者基线实测优先（round 2 实测 0.37 GiB vs 假设 2 GiB），无实测回退。
        measured = int(0.37 * GIB)
        pending = [500 * 2 ** 20, 300 * 2 ** 20]
        _workers, per = build_plan(pending, 6, None, measured_per_builder=measured)
        self.assertEqual(per, measured + max(pending))
        _workers, per = build_plan(pending, 6, None)
        self.assertEqual(per, PACK_BUILDER_BYTES + max(pending))

    def _bare_pipeline(self, tmp):
        from curriculum_queue import TrainingPipeline
        pipeline = TrainingPipeline.__new__(TrainingPipeline)
        pipeline.failed = False
        pipeline.args = SimpleNamespace(log=Path(tmp) / 'log', python=Path(sys.executable))
        pipeline.lock_fd = None
        pipeline.build_workers = 6
        pipeline.build_peak = 12 * GIB          # 15:22 生产现场：pack plan 估到 12.1 GiB
        pipeline.last_train_peak = int(3.53 * GIB)
        pipeline.directory = Path(tmp)          # estimate_peak 在这里 glob 段目录
        return pipeline

    def test_pack_end_clears_build_peak_from_reservation(self):
        # 打包一结束（训练/评估期间），estimate_peak 只反映训练估算，不再扣 12 GiB。
        import curriculum_queue as queue
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._bare_pipeline(tmp)
            job = self._job(tmp, 'queued')
            peaks = {}
            def fake_command(command, log):
                peaks['pack' if 'pack' in str(command) else 'train'] = pipeline.build_peak
                return int(0.40 * GIB) if 'pack' in str(command) else int(3.53 * GIB)
            pipeline._admit = lambda: None
            pipeline._prepare_corpus = lambda labels: None
            pipeline._command = fake_command
            pipeline._evaluate = lambda _job: {'promotion': {'approved': False, 'reasons': ['x']}}
            with mock.patch.object(queue, 'load_recipe',
                                   lambda path, kind=None: {'labels': ['x'], 'data': '/tmp/frames'}), \
                    mock.patch.object(queue, 'pack_command', lambda args, labels: ['pack-cmd']):
                report = pipeline._run(copy.deepcopy(job))
            self.assertEqual(peaks['pack'], 12 * GIB)          # 打包进行时仍按 pack plan 扣
            self.assertEqual(peaks['train'], 0)                # 打包一结束就清零
            self.assertEqual(pipeline.build_peak, 0)
            self.assertEqual(pipeline.last_train_peak, int(3.53 * GIB))
            self.assertEqual(report['train_peak_anon_bytes'], int(3.53 * GIB))
            self.assertEqual(pipeline.estimate_peak(), int(3.53 * GIB))   # reservation 只按训练估算

    def test_pack_failure_still_clears_build_peak(self):
        import curriculum_queue as queue
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._bare_pipeline(tmp)
            job = self._job(tmp, 'queued')
            pipeline._admit = lambda: None
            def broken(_labels):
                raise RuntimeError('no archives for labels')
            pipeline._prepare_corpus = broken
            with mock.patch.object(queue, 'load_recipe',
                                   lambda path, kind=None: {'labels': ['x'], 'data': '/tmp/frames'}), \
                    mock.patch.object(queue, 'pack_command', lambda args, labels: ['pack-cmd']):
                with self.assertRaises(RuntimeError):
                    pipeline._run(copy.deepcopy(job))
            self.assertEqual(pipeline.build_peak, 0)           # 失败同样清零

    def test_init_restores_measured_pack_peak(self):
        import curriculum_queue as queue
        with tempfile.TemporaryDirectory() as tmp:
            job = self._job(tmp, 'done')
            job['timing'] = {'pack_peak_anon_bytes': int(0.40 * GIB),
                             'train_peak_anon_bytes': int(3.53 * GIB)}
            state = {'training_jobs': [job]}
            args = SimpleNamespace(state=Path(tmp) / 'state.json')
            def fake_run(self, _job):
                return {}
            with mock.patch.object(queue.TrainingPipeline, '_run', fake_run), \
                    mock.patch.object(queue.TrainingPipeline, '_plan_build', lambda self, recipe: None), \
                    mock.patch.object(queue, 'load_recipe',
                                      lambda path, kind=None: {'precompute_workers': 2}):
                pipeline = queue.TrainingPipeline(args, state)
            self.assertEqual(pipeline.last_pack_peak, int(0.40 * GIB) // 2)   # 单建造者实测
            self.assertEqual(pipeline.last_train_peak, int(3.53 * GIB))


class DemoStartCellsTest(unittest.TestCase):
    """task71：示范起点格——档位是起点决策号，未来随机流走 RESEED。"""

    BASE = 'demo_fog_33_cactus_snow_m200'
    LADDER = [800, 600, 400]

    def start_cell(self, mult_index=0, windows=None):
        return dict(family='demo_start', level=33, mult_index=mult_index, last_round=0,
                    windows=windows or {}, ladder=list(self.LADDER), task_id=self.BASE,
                    archive='/archives/demo', seeds=[900 + index for index in range(6)])

    def test_ladder_promotes_earlier_and_demotes_later(self):
        cells = {'start': self.start_cell(windows={'800': {'recent': [1] * 10, 'wins': 10, 'games': 10}})}
        changes = adjust(cells, VALUES)
        self.assertEqual(cells['start']['mult_index'], 1)      # 升档 = 起点往前挪（800 -> 600）
        self.assertEqual(len(changes), 1)
        cells['start']['windows']['600'] = {'recent': [0] * 10, 'wins': 0, 'games': 10}
        adjust(cells, VALUES)
        self.assertEqual(cells['start']['mult_index'], 0)      # 降档 = 起点往后挪
        # 最晚一档（下标 0）不再降；最早一档（下标 2）不再升
        bottom = {'start': self.start_cell(mult_index=2, windows={'400': {'recent': [1] * 10, 'wins': 10, 'games': 10}})}
        adjust(bottom, VALUES)
        self.assertEqual(bottom['start']['mult_index'], 2)
        top = {'start': self.start_cell(windows={'800': {'recent': [0] * 10, 'wins': 0, 'games': 10}})}
        adjust(top, VALUES)
        self.assertEqual(top['start']['mult_index'], 0)

    def test_cell_level_reads_the_ladder(self):
        self.assertEqual(cell_level(self.start_cell(mult_index=2), VALUES), 400)
        self.assertEqual(cell_level(cell(5, 10, mult_index=3), VALUES), 1.75)

    def test_plan_round_slices_the_future_seed_block(self):
        state = dict(round=0, offsets={}, cells={self.BASE: self.start_cell()})
        manifests = {self.BASE: Path('/tmp/human_demo_tasks.json')}
        plan, _ = plan_round(state, {}, NAMES, VALUES, manifests, budget=2, per_group=2)
        self.assertEqual(len(plan), 1)
        item = plan[0]
        self.assertEqual((item['task_id'], item['demo_decision'], item['offset']), (self.BASE, 800, 0))
        self.assertEqual(item['future_seeds'], [900, 901])
        self.assertEqual(item['label'], 'c1demofog33cactussnowm200s800')
        self.assertEqual(item['multiplier'], 800)             # 档位 = 起点决策号
        state['offsets'][self.BASE] = 5
        plan, _ = plan_round(state, {}, NAMES, VALUES, manifests, budget=1, per_group=1)
        self.assertEqual(plan[0]['future_seeds'], [905])       # 只剩一个未来种子
        state['offsets'][self.BASE] = 6
        with self.assertRaises(SystemExit):
            plan_round(state, {}, NAMES, VALUES, manifests, budget=1, per_group=1)

    def test_start_command_is_deterministic_and_carries_the_start(self):
        args = SimpleNamespace(python=Path('/nonexistent/python'), rollout_policy=Path('/m.pt'),
                               static_value=Path('/sv.pt'), value=Path('/v.pt'), workers=4,
                               executable=Path('/exe'), resource_dir=Path('/res'),
                               frames=Path('/frames'), out=Path('/out'))
        first = command_for(args, Path('/tmp/manifest.json'), self.BASE, 0, 2, 'c1x',
                            demo_archive='/archives/demo', demo_decision=800, future_seeds=[900, 901])
        second = command_for(args, Path('/tmp/manifest.json'), self.BASE, 0, 2, 'c1x',
                             demo_archive='/archives/demo', demo_decision=800, future_seeds=[900, 901])
        self.assertEqual(first, second)
        self.assertEqual(first[first.index('--demo-archive') + 1], '/archives/demo')
        self.assertEqual(first[first.index('--demo-decision') + 1], '800')
        self.assertEqual(first[first.index('--future-seeds') + 1], '900,901')
        # 偏移必须是 0：search_play 在 --demo-archive 下要求示范种子落在选中的清单段里
        self.assertEqual(first[first.index('--seed-offset') + 1], '0')
        self.assertEqual(first[first.index('--seeds') + 1], '2')
        plain = command_for(args, Path('/tmp/panel.json'), 'day_l7_train_d7', 4, 2, 'c1d7m100')
        self.assertNotIn('--demo-archive', plain)
        self.assertEqual(plain[plain.index('--seed-offset') + 1], '4')

    def test_load_demo_start_cells_validates_and_resolves(self):
        document = {'tasks': [
            {'task_id': self.BASE, 'level': 33, 'demo_start': {
                'archive': 'artifacts/human_demo/raw/seed_20261033/' + self.BASE,
                'decisions': self.LADDER, 'index': 1, 'future_seeds': [900, 901, 902]}},
            {'task_id': 'plain', 'level': 31},
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            # 存档不进 git：在临时目录里放一份已验证的 meta，不依赖本机有没有用户示范
            archive = Path(tmp) / 'archive'
            archive.mkdir()
            (archive / 'meta.json').write_text(json.dumps({'verified': True}))
            document['tasks'][0]['demo_start']['archive'] = str(archive)
            path = Path(tmp) / 'tasks.json'
            path.write_text(json.dumps(document))
            cells, manifests = load_demo_start_cells(path)
            self.assertEqual(set(cells), {self.BASE})
            self.assertEqual(cells[self.BASE]['ladder'], self.LADDER)
            self.assertEqual(cells[self.BASE]['mult_index'], 1)
            self.assertTrue(Path(cells[self.BASE]['archive']).is_absolute())
            self.assertEqual(manifests, {self.BASE: path})
            document['tasks'][0]['demo_start']['decisions'] = [400, 600, 800]
            path.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                load_demo_start_cells(path)
            # 存档不在就没有起点可恢复：启动时就报，而不是组里 replay 才炸
            document['tasks'][0]['demo_start']['decisions'] = self.LADDER
            document['tasks'][0]['demo_start']['archive'] = str(Path(tmp) / 'none')
            path.write_text(json.dumps(document))
            with self.assertRaises(ValueError) as raised:
                load_demo_start_cells(path)
            self.assertIn('archive missing', str(raised.exception))

    def test_reconcile_leaves_panel_cells_byte_identical(self):
        tasks = {f'night_l{level}_train_n10q': dict(task_id=f'night_l{level}_train_n10q', level=level,
                                                    zombie_count_multiplier=1.0)
                 for level in N10_LEVELS}
        specs = {self.BASE: self.start_cell()}
        plain = make_state(tasks, NAMES, {}, {}, 'panel.json')
        with_start = make_state(tasks, NAMES, {}, {}, 'panel.json', specs)
        self.assertEqual(json.dumps(plain['cells'], sort_keys=True),
                         json.dumps({k: v for k, v in with_start['cells'].items() if k != self.BASE},
                                    sort_keys=True))
        self.assertIn(self.BASE, with_start['cells'])
        # 再对齐一次是幂等的；清单删掉这个格就退休
        self.assertEqual(reconcile_start_cells(with_start, specs), [])
        self.assertEqual(reconcile_start_cells(with_start, {}), [f'start cell retired: {self.BASE}'])
        self.assertNotIn(self.BASE, with_start['cells'])

    def test_reconcile_updates_a_changed_ladder_and_clamps_the_rung(self):
        state = dict(cells={self.BASE: self.start_cell(mult_index=2)}, retired_cells={})
        spec = dict(self.start_cell(), ladder=[800, 600])
        lines = reconcile_start_cells(state, {self.BASE: spec})
        self.assertEqual(state['cells'][self.BASE]['mult_index'], 1)
        self.assertEqual(len(lines), 1)
        self.assertIn('ladder', lines[0])

    def test_registered_manifest_start_cells(self):
        manifest = ROOT / 'configs/human_demo_tasks.json'
        archives = [task['demo_start']['archive'] for task in json.loads(manifest.read_text())['tasks']
                    if 'demo_start' in task]
        if not all((ROOT / archive).is_dir() for archive in archives):
            self.skipTest('user demo archives are not in git; present only where they were recorded/fetched')
        cells, _ = load_demo_start_cells(manifest)
        # task70 B6: fog start cells use the _hv2 family (human_v2 observation, same archives /
        # ladders / future seed blocks); roof start cells have no fog and keep the base task id.
        self.assertEqual(set(cells), {
                                      'demo_roof_48_doom_m400_s81017005_a1b6aa9d1b411',
                                      'llmplay3_a_l43_m100_s139002036_ac2b18c7ca5ba',
                                      'demo_pool_30_orig_m100_s81028026_a833d82b102f8',
                                      'fog79_l37_snow_split_v2_m200_s81023017_aa21819990cfc',
                                      'llmplay3_a_l16_m100_s139013014_ad40eec907f86',
                                      'llmplay3_a_l8_m100_s139011025_afd353941ad39',
                                      'llmplay3_a_l43_m100_s139002032_ad1e988ab72bd',
                                      'llmplay3_a_l49_m100_s139008008_af3d5ebc85da0',
                                      'fog79_l38_snow_magnet_v2_m200_s81024016_a2160cfe3610c',
                                      'llmplay3_a_l49_m100_s139008018_afddc32872008',
                                      'llmplay3_e_l42_m150_s139023001_a79fce0529796',
                                      'llmplay3_a_l38_m100_s139020036_ab30fee4a68a9',
                                      'fog79_l36_snow_split_v2_m200_s81021023_a792d33d80a32',
                                      'fog79_l37_snow_split_v2_m200_s81023028_a655a9b359aa0',
                                      'llmplay3_e_l43_m200_s139026004_a81e9666c27a6',
                                      'llmplay3_a_l8_m100_s139011022_a9cca2a01dbe2',
                                      'llmplay3_a_l22_m100_s139015036_acc78a777829d',
                                      'fog79_l39_snow_cactus_v2_m200_s81026027_ab1a4e140e65c',
                                      'llmplay3_a_l43_m100_s139002004_a277aba6fac8e',
                                      'fog79_l36_snow_magnet_v2_m200_s81020004_aae53c17b1cc9',
                                      'llmplay3_a_l45_m100_s139004023_a518e5bc8a678',
                                      'llmplay3_a_l6_m100_s139010032_afc436bc3441a',
                                      'llmplay3_a_l16_m100_s139013008_a0f4c35d6a43b',
                                      'llmplay3_a_l32_m100_s139018032_a248a64d726f9',
                                      'fog79_l39_snow_blover_v2_m200_s81027036_ac39151a7ff85',
                                      'fog79_l36_snow_magnet_v2_m200_s81020039_a7e90ee200da8',
                                      'fog79_l37_snow_split_v2_m200_s81023014_a024cd5a95a15',
                                      'llmplay3_a_l38_m100_s139020017_a3a47f54df0b1',
                                      'demo_boss_50_orig_m100_s81030006_a8d5d48ed3506',
                                      'demo_pool_30_orig_m100_s81028037_a44947a5a9751',
                                      'llmplay3_a_l41_m100_s139000005_aa8ab40f919ed',
                                      'demo_roof_47_doom_m400_s81015010_aae32346e38ab',
                                      'llmplay3_a_l46_m100_s139005022_a885f56ce7e89',
                                      'fog79_l38_snow_tall_v2_m200_s81025011_a53424129042b',
                                      'llmplay3_c_l9_d02_s139071027_a735f3d5f5672',
                                      'llmplay3_a_l12_m100_s139012007_ad67abb33a85b',
                                      'demo_roof_47_doom_m400_s81015033_ad0b0b39aced3',
                                      'fog79_l36_snow_split_v2_m200_s81021019_a5d91e919fb71',
                                      'llmplay3_a_l6_m100_s139010000_a950ce75376b7',
                                      'llmplay3_a_l36_m100_s139019005_a94b4cd7d9be6',
                                      'fog79_l39_snow_cactus_v2_m200_s81026004_a3d09c2268952',
                                      'demo_roof_47_doom_m400_s81015033_a88214948cc76',
                                      'fog79_l37_snow_magnet_v2_m200_s81022001_a99b58b48ec68',
                                      'llmplay3_a_l46_m100_s139005006_a008a192955c2',
                                      'llmplay3_a_l38_m100_s139020019_aa6a62ea2f04e',
                                      'demo_roof_48_doom_m400_s81017005_a1c45d17d3dab',
                                      'fog79_l36_snow_magnet_v2_m200_s81020024_a4cbdd0e0dae1',
                                      'llmplay3_a_l18_m100_s139014000_ac7f314ed485a',
                                      'demo_pool_30_orig_m100_s81028023_abb23b99f8320',
                                      'llmplay3_a_l6_m100_s139010006_a5dabd9e268d3',
                                      'llmplay3_a_l38_m100_s139020020_a43845aa94f63',
                                      'fog79_l38_snow_tall_v2_m200_s81025015_a659bd0f7c938',
                                      'fog79_l39_snow_blover_v2_m200_s81027000_a8af5eac706d0',
                                      'llmplay3_a_l43_m100_s139002011_a9cd04e94a93c',
                                      'llmplay3_a_l44_m100_s139003022_a74ddb9a9c6c4',
                                      'fog79_l37_snow_split_v2_m200_s81023036_aaa08735e969a',
                                      'llmplay3_a_l18_m100_s139014039_a2aeb97353c1d',
                                      'llmplay3_a_l36_m100_s139019022_ad6c5badea9e1',
                                      'demo_boss_50_orig_m100_s81030022_a3daa2da13e0e',
                                      'fog79_l37_snow_split_v2_m200_s81023028_a7348922ba9a7',
                                      'llmplay3_a_l8_m100_s139011017_a0d40c9a13585',
                                      'llmplay3_a_l44_m100_s139003017_a58f865ccd903',
                                      'fog79_l38_snow_magnet_v2_m200_s81024007_ae2f2c0b739a0',
                                      'llmplay3_c_l7_d02_s139066022_a3fa0b8ae493a',
                                      'llmplay3_a_l6_m100_s139010005_a9de96c8870ff',
                                      'fog79_l39_snow_cactus_v2_m200_s81026007_adab6656b879c',
                                      'llmplay3_a_l44_m100_s139003006_ae86a69d28e4f',
                                      'llmplay3_a_l28_m100_s139017009_aa780cc5d7c8e',
                                      'fog79_l36_snow_magnet_v2_m200_s81020033_aaf239adc9da2',
                                      'llmplay3_a_l44_m100_s139003035_a37e49e9af77c',
                                      'llmplay3_e_l43_m200_s139026020_a7e1441b8794b',
                                      'demo_roof_47_doom_m400_s81015033_ad0285b74c501',
                                      'fog79_l37_snow_magnet_v2_m200_s81022002_a492433842832',
                                      'llmplay3_a_l44_m100_s139003028_a66aa4b80c40a',
                                      'llmplay3_a_l2_m100_s139009019_adff4cc660fe5',
                                      'llmplay3_a_l45_m100_s139004020_a8ffd2b3ee7e2',
                                      'demo_boss_50_orig_m100_s81030004_a1b04b1d5e750',
                                      'demo_roof_47_doom_m300_s81014023_a07b2ce3d00a9',
                                      'demo_roof_43_melon_m200_s81008016_aa92990be1063',
                                      'demo_roof_43_gloom_m200_s81009031_a7a94c06f41c0',
                                      'demo_roof_47_doom_m200_s81001025_a4ff2a6ce95ac',
                                      'demo_roof_48_doom_m200_s81002021_a9a95ba12ad42',
                                      'demo_roof_43_melon_m200_s81008033_a9ec6bb93ae6e',
                                      'demo_roof_48_doom_m400_s81017035_a942ebac22713',
                                      'demo_roof_48_doom_m400_s81017009_a68d7671f414e',
                                      'demo_roof_42_melon_m200_s81005029_a6732a3efca5e',
                                      'demo_roof_46_doom_m200_s81000009_a96df3ebb957e',
                                      'demo_roof_47_doom_m400_s81015033_ab28636337f75',
                                      'demo_roof_42_melon_m200_s81005024_a13103f4ef58b',
                                      'demo_roof_43_gloom_m200_s81009020_ae34321a42060',
                                      'demo_roof_47_doom_m400_s81015008_a4952be9ea06e',
                                      'demo_roof_49_doom_m200_s81003027_a635c541c7855',
                                      'demo_roof_49_doom_m300_s81018038_a1167943e3d30',
                                      'demo_roof_48_doom_m400_s81017009_aba744a1ce915',
                                      'demo_roof_48_doom_m400_s81017033_ada9e694cc296',
                                      'demo_roof_47_doom_m200_s81001012_a9571d549276e',
                                      'demo_roof_48_doom_m200_s81002026_a9d1a6841b4c1',
                                      'demo_roof_44_gloom_m200_s81011016_ae9b79c5a5188',
                                      'demo_roof_43_melon_m200_s81008015_a2e646f477cf4',
                                      'demo_roof_46_doom_m300_s81012024_aab854d1b4eed',
                                      'demo_roof_47_doom_m400_s81015034_a42d58c5b2563',
                                      'demo_roof_43_gloom_m200_s81009021_a7a72ae7f8d27',
                                      'demo_roof_43_melon_m200_s81008028_aa6bcc305003c',
                                      'demo_roof_49_doom_m300_s81018025_ac3fd89912552',
                                      'demo_roof_48_doom_m300_s81016001_ad800e4825a23',
                                      'demo_roof_46_doom_m200_s81000014_a9b3d3d7c79f2',
                                      'demo_roof_48_doom_m300_s81016004_af8da995a72d4',
                                      'demo_roof_43_gloom_m200_s81009020_ae0352a1641d7',
                                      'demo_roof_43_melon_m200_s81008015_a2e5670bc32a1',
                                      'demo_roof_47_doom_m300_s81014011_ad4199f407694',
                                      'demo_roof_49_doom_m300_s81018038_af172d0d63fe6',
                                      'demo_roof_46_doom_m400_s81013023_af75087d39844',
                                      'demo_roof_47_doom_m200_s81001025_a66f03f060618',
                                      'demo_roof_43_gloom_m200_s81009012_a75e6293e695b',
                                      'demo_roof_48_doom_m300_s81016036_a504fc3f54a4e',
                                      'demo_roof_49_doom_m300_s81018025_a517b204444e2',
                                      'demo_roof_44_gloom_m200_s81011004_a293fa56162d8',
                                      'demo_roof_47_doom_m300_s81014030_a11fa85c2ac02',
                                      'demo_roof_49_doom_m300_s81018023_ad7aea4260899',
                                      'demo_roof_48_doom_m300_s81016006_a1ae178df0157',
                                      'demo_roof_46_doom_m400_s81013000_acb325eb04e28',
                                      'demo_roof_44_melon_m200_s81010025_a8572b42cd9be',
                                      'demo_roof_42_gloom_m200_s81006038_a958fb9ef3f39',
                                      'demo_roof_41_doom_m200_s81004032_a5f21f91e1659',
                                      'demo_roof_43_gloom_m200_s81009031_a234247aae65c',
                                      'demo_roof_46_doom_m400_s81013034_a97439bf8296e',
                                      'demo_roof_42_gloommelon_m200_s81007019_a2e18bc36e4b0',
                                      'demo_roof_44_melon_m200_s81010013_a235ac44900a3',
                                      'demo_roof_43_melon_m200_s81008021_a89351b31e702',
                                      'demo_roof_46_doom_m200_s81000015_a0ac9851fc140',
                                      'demo_roof_44_melon_m200_s81010039_ab9832ead286e',
                                      'demo_roof_46_doom_m300_s81012001_a493b626dee79',
                                      'demo_roof_49_doom_m300_s81018039_a73b3cc9a08ed',
                                      'demo_roof_41_doom_m200_s81004018_a738b2b0ecec9',
                                      'demo_roof_46_doom_m400_s81013003_af38e12b9e063',
                                      'demo_roof_49_doom_m300_s81018038_ab7fce1bf5dd6',
                                      'demo_roof_46_doom_m200_s81000014_a1815934f3344',
                                      'demo_roof_47_doom_m400_s81015034_aed5dcec186ad',
                                      'demo_roof_49_doom_m300_s81018038_ac5ecd81d42f0',
                                      'demo_roof_49_doom_m300_s81018038_af4f95dd3637a',
                                      'demo_roof_42_gloom_m200_s81006023_a2a2fef025c37',
                                      'demo_roof_48_doom_m400_s81017009_a7528f954a548',
                                      'demo_roof_48_doom_m400_s81017035_a05dc38c82a78',
                                      'demo_roof_49_doom_m300_s81018028_a6c6fa15f504b',
                                      'demo_roof_44_gloom_m200_s81011000_a8ba5a531897a',
                                      'demo_roof_46_doom_m200_s81000037_af847e312eaf4',
                                      'demo_roof_46_doom_m300_s81012021_a6a2b46ccfd18',
                                      'demo_roof_46_doom_m400_s81013023_a19feb29aacd2',
                                      'demo_roof_42_gloom_m200_s81006006_a0f5e0e8c7392',
                                      'demo_roof_44_melon_m200_s81010029_aef585f51192a',
                                      'demo_roof_48_doom_m400_s81017009_a4032837a73c0',
                                      'demo_roof_46_doom_m300_s81012038_afdddbc59d0a4',
                                      'demo_roof_46_doom_m200_s81000037_a0579edb42586',
                                      'demo_roof_49_doom_m300_s81018023_a419777bba80c',
                                      'demo_roof_48_doom_m400_s81017009_af324c4a9c41c',
                                      'demo_roof_47_doom_m300_s81014011_aa883bd245596',
                                      'demo_roof_47_doom_m400_s81015033_afc760d64285f',
                                      'demo_roof_49_doom_m300_s81018025_a9f8360396743',
                                      'demo_roof_43_melon_m200_s81008028_a9ed9509123f5',
                                      'demo_roof_44_melon_m200_s81010025_aa2210719fa74',
                                      'demo_roof_46_doom_m400_s81013034_a3c1d957e32d3',
                                      'demo_roof_41_doom_m200_s81004010_aeba6a3f97904',
                                      'demo_roof_47_doom_m300_s81014002_a52c20beccbe2',
                                      'demo_roof_48_doom_m200_s81002031_a7739aa087363',
                                      'demo_roof_48_doom_m400_s81017033_a9d1bfdc6b4f6',
                                      'demo_roof_42_melon_m200_s81005015_a6d764c239e52',
                                      'demo_roof_48_doom_m400_s81017035_a0a9b8e401742',
                                      'demo_roof_48_doom_m400_s81017037_a5f9721e6e57a',
                                      'demo_roof_48_doom_m400_s81017037_a7ffb5a2fa892',
                                      'demo_roof_49_doom_m200_s81003029_a238ff87b9891',
                                      'demo_roof_46_doom_m400_s81013034_ac772b9c13a94',
                                      'demo_roof_46_doom_m400_s81013000_a717ddb86eacb',
                                      'demo_roof_49_doom_m300_s81018028_aae136d6f9868',
                                      'demo_roof_48_doom_m300_s81016020_ad937a699653f',
                                      'demo_roof_46_doom_m400_s81013023_ad1f6a0595c91',
                                      'demo_roof_49_doom_m200_s81003027_adb4448e57500',
                                      'demo_roof_48_doom_m400_s81017009_aae0cdc7c8c70',
                                      'demo_roof_47_doom_m300_s81014013_ab4379313acdb',
                                      'demo_roof_49_doom_m300_s81018025_af4de48569a56',
                                      'demo_roof_48_doom_m300_s81016021_a49c1adbccc2a',
                                      'demo_roof_47_doom_m400_s81015033_ac85bfe174b39',
                                      'demo_roof_46_doom_m400_s81013034_a38200516869e',
                                      'demo_roof_42_gloom_m200_s81006008_a1373e7f674f0',
                                      'demo_roof_44_gloom_m200_s81011024_ab012b5861608',
                                      'demo_roof_48_doom_m400_s81017009_acdc59889ab49',
                                      'demo_roof_47_doom_m200_s81001013_ac358dd8c5669',
                                      'demo_roof_44_melon_m200_s81010039_a99ce92484016',
                                      'demo_roof_42_gloommelon_m200_s81007033_addc40552585a',
                                      'demo_roof_48_doom_m300_s81016001_a43a27d580e94',
                                      'demo_roof_46_doom_m400_s81013023_a2b8719fbc6e4',
                                      'demo_roof_44_melon_m200_s81010028_a101cdf2d112f',
                                      'demo_roof_49_doom_m300_s81018038_a39598b6f6507',
                                      'demo_roof_48_doom_m300_s81016004_a7a49219cbe85',
                                      'demo_roof_42_gloommelon_m200_s81007003_a8235f2f44b9f',
                                      'demo_roof_48_doom_m200_s81002015_a593220c8a627',
                                      'demo_roof_42_gloommelon_m200_s81007031_aca1b6fd647da',
                                      'demo_roof_48_doom_m400_s81017009_a13c1b37a7fee',
                                      'demo_roof_49_doom_m300_s81018025_af9c5a540123c',
                                      'demo_roof_49_doom_m300_s81018038_a1cfad16dd958',
                                      'demo_roof_46_doom_m400_s81013003_a80456ec97d1a',
                                      'demo_roof_41_doom_m200_s81004010_a7d4279c01933',
                                      'demo_roof_48_doom_m300_s81016021_a65ce7aa379d1',
                                      'demo_roof_41_doom_m200_s81004032_a1a09b12ae5e4',
                                      'demo_roof_46_doom_m400_s81013023_a3bddf9ef1f06',
                                      'demo_roof_42_gloom_m200_s81006028_ad1b26af5d500',
                                      'demo_roof_46_doom_m400_s81013003_a77c6e301961f',
                                      'demo_roof_44_gloom_m200_s81011006_ad0fdb0947e7a',
                                      'demo_roof_41_doom_m200_s81004012_a0db8f34a63eb',
                                      'demo_roof_49_doom_m300_s81018038_a01dbebf68025',
                                      'demo_roof_49_doom_m200_s81003029_a0ee6a4b3a240',
                                      'demo_roof_47_doom_m200_s81001027_a0cd7c1da5ed6',
                                      'demo_roof_42_melon_m200_s81005030_a14c6e890465e',
                                      'demo_roof_46_doom_m400_s81013034_a0b5d49cc4337',
                                      'demo_roof_49_doom_m300_s81018031_a1a5ec4ca1648',
                                      'demo_roof_47_doom_m400_s81015033_aededdfc594d7',
                                      'demo_roof_46_doom_m300_s81012036_a215b9ee07aca',
                                      'demo_roof_46_doom_m300_s81012024_a28122d1f4761',
                                      'demo_roof_49_doom_m300_s81018025_a1fd3c9c31ee6',
                                      'demo_roof_49_doom_m300_s81018031_a42fdc60708c3',
                                      'demo_roof_48_doom_m400_s81017009_a6f0fd3657c8f',
                                      'demo_roof_48_doom_m400_s81017033_a3b04cc473a54',
                                      'demo_roof_49_doom_m300_s81018025_a7807ae2af9c0',
                                      'demo_roof_47_doom_m400_s81015033_a25752597f880',
                                      'demo_roof_46_doom_m400_s81013034_a95b3c83b3117',
                                      'demo_roof_49_doom_m300_s81018031_a5643fb3b3a94',
                                      'demo_roof_48_doom_m400_s81017009_ade8f716cc5a6',
                                      'demo_roof_47_doom_m400_s81015008_a947797c5e5ac',
                                      'demo_roof_49_doom_m300_s81018039_abbb7695c3e9b',
                                      'demo_roof_49_doom_m300_s81018038_ac7d6503ce49f',
                                      'demo_roof_49_doom_m300_s81018038_ab9a9dc926eba',
                                      'demo_roof_49_doom_m300_s81018031_af5eca6499648',
                                      'demo_roof_49_doom_m300_s81018025_a3f24aded1465',
                                      'demo_roof_47_doom_m400_s81015008_ab8fbc633121c',
                                      'demo_roof_46_doom_m400_s81013034_a20a1e2664571',
                                      'demo_roof_49_doom_m200_s81003013_a4e6fcb5e4996',
                                      'demo_roof_47_doom_m200_s81001034_ac69cb1230821',
                                      'demo_roof_49_doom_m200_s81003030_ab55ca5144a6f',
                                      'demo_roof_46_doom_m400_s81013034_ae4297f3de852',
                                      'demo_roof_42_gloommelon_m200_s81007003_ad6f5263b2983',
                                      'demo_roof_48_doom_m200_s81002021_a9fbd8d4f95d4',
                                      'demo_roof_48_doom_m400_s81017009_a0f54c99db605',
                                      'demo_roof_46_doom_m400_s81013034_adf0905fed58d',
                                      'demo_roof_48_doom_m400_s81017009_a0ef754ff984d',
                                      'demo_roof_42_gloommelon_m200_s81007033_a10702a1d26b4',
                                      'demo_roof_47_doom_m200_s81001012_afc74a6a3377d',
                                      'demo_roof_49_doom_m300_s81018038_a89bc0898ebc3',
                                      'demo_roof_42_melon_m200_s81005023_a44f4df7bdd32',
                                      'demo_roof_42_gloom_m200_s81006008_ac0ca2bebd54d',
                                      'demo_roof_48_doom_m400_s81017009_a1fc326a73ff2',
                                      'demo_roof_48_doom_m400_s81017009_a4955d974f877',
                                      'demo_roof_48_doom_m400_s81017009_ae11fa7624867',
                                      'demo_roof_42_melon_m200_s81005030_a676e35addf78',
                                      'demo_roof_47_doom_m400_s81015034_ae40cd3fa2143',
                                      'demo_roof_42_gloom_m200_s81006038_a59f59d59f0a8',
                                      'demo_roof_49_doom_m200_s81003027_a8232e1373885',
                                      'demo_roof_41_doom_m200_s81004012_aa0fbf5c5f8ee',
                                      'demo_roof_47_doom_m200_s81001034_aca04f199330b',
                                      'demo_roof_44_gloom_m200_s81011000_a7cf7bbfa6c43',
                                      'demo_roof_49_doom_m200_s81003013_a2806be443b99',
                                      'demo_roof_49_doom_m300_s81018025_afdf76dbc6886',
                                      'demo_roof_46_doom_m200_s81000003_afce0a31d40ca',
                                      'demo_roof_49_doom_m300_s81018028_a84641027514a',
                                      'demo_roof_46_doom_m400_s81013023_af6c77bb2b969',
                                      'demo_roof_48_doom_m300_s81016006_a05540174ee38',
                                      'demo_roof_48_doom_m200_s81002026_a0a99a572aeb0',
                                      'demo_roof_48_doom_m400_s81017009_add4060026737',
                                      'demo_roof_43_gloom_m200_s81009017_afa82dcd1d782',
                                      'demo_roof_49_doom_m200_s81003022_a406030af1bed',
                                      'demo_roof_46_doom_m300_s81012038_a6660e2455a1e',
                                      'demo_roof_44_gloom_m200_s81011016_aedd62096c4e7',
                                      'demo_roof_47_doom_m300_s81014023_a15c3959b8a35',
                                      'demo_roof_48_doom_m200_s81002026_a399a23ed6087',
                                      'demo_roof_49_doom_m300_s81018025_a4af037f0c455',
                                      'demo_roof_42_gloom_m200_s81006023_a45f738170579',
                                      'demo_roof_48_doom_m300_s81016036_af1a4518f8fcd',
                                      'demo_roof_47_doom_m400_s81015034_aecccba486c27',
                                      'demo_roof_48_doom_m400_s81017009_a3a2f0b81ad99',
                                      'demo_roof_49_doom_m200_s81003030_af9f310f210c5',
                                      'demo_roof_43_melon_m200_s81008021_a9ba966f1a71e',
                                      'demo_roof_46_doom_m400_s81013023_aa2187d4c5053',
                                      'demo_roof_48_doom_m300_s81016020_ab6dec4110c1c',
                                      'demo_roof_47_doom_m200_s81001025_a190561a5db4d',
                                      'demo_roof_41_doom_m200_s81004015_ad8034929d1a8',
                                      'demo_roof_46_doom_m300_s81012036_a2ecd687f1d9e',
                                      'demo_roof_49_doom_m300_s81018038_a5859ba200e02',
                                      'demo_roof_44_gloom_m200_s81011006_adbf7fee25b97',
                                      'demo_roof_49_doom_m300_s81018025_a5eb948eac49b',
                                      'demo_roof_49_doom_m200_s81003027_a932c8fe7c3f3',
                                      'demo_roof_42_melon_m200_s81005015_aab470400b7ce',
                                      'demo_roof_42_gloommelon_m200_s81007031_af0b1d270b7f6',
                                      'demo_roof_42_melon_m200_s81005024_ad8ce3ea73b04',
                                      'demo_roof_46_doom_m300_s81012021_aaa9305526b15',
                                      'demo_roof_44_melon_m200_s81010013_aaf1b9d24db61',
                                      'demo_roof_46_doom_m200_s81000009_ad527d717cbb1',
                                      'demo_roof_49_doom_m300_s81018038_a24b32da2df4f',
                                      'demo_roof_46_doom_m200_s81000015_a79860d56b606',
                                      'demo_roof_46_doom_m400_s81013034_a29490b12f1bc',
                                      'demo_roof_43_gloom_m200_s81009017_aa31926eb00cb',
                                      'demo_roof_42_gloommelon_m200_s81007027_ae96b4ef6e882',
                                      'demo_roof_48_doom_m200_s81002017_a1a1248a2dd03',
                                      'demo_roof_47_doom_m300_s81014013_a2957adb511d3',
                                      'demo_roof_48_doom_m200_s81002031_a7adc1574be88',
                                      'demo_roof_49_doom_m200_llm',
                                      'demo_fog_31_snow_m200_hv2', 'demo_fog_32_snow_m200_hv2',
                                      'demo_fog_33_cactus_snow_m200_hv2', 'demo_fog_33_cattail_snow_m200_hv2',
                                      'demo_fog_34_cactus_snow_m200_hv2', 'demo_fog_39_cactus_snow_m200_hv2',
                                      'demo_roof_42_melon_m200', 'demo_roof_43_melon_m200',
                                      'demo_roof_44_melon_m200', 'demo_roof_42_gloommelon_m200',
                                      'demo_roof_49_doom_m200', 'demo_roof_48_doom_m200',
                                      'demo_roof_46_doom_m200',
                                      # task87: LLM 示范起点格（_llm = 本机种子另示范，_sNNN = 外种子族）
                                      'demo_roof_46_doom_m200_llm', 'demo_roof_48_doom_m200_llm',
                                      'demo_roof_46_doom_m200_s20261047', 'demo_roof_46_doom_m200_s20261048',
                                      'demo_roof_47_doom_m200', 'demo_roof_47_doom_m200_s20261048',
                                      'demo_roof_47_doom_m200_s20261049',
                                      'demo_roof_48_doom_m200_s20261049', 'demo_roof_48_doom_m200_s20261050',
                                      'demo_roof_49_doom_m200_s20261050', 'demo_roof_49_doom_m200_s20261051'})
        for task_id, spec in cells.items():
            self.assertGreaterEqual(len(spec['ladder']), 1, task_id)
            self.assertEqual(spec['ladder'], sorted(spec['ladder'], reverse=True), task_id)
            self.assertTrue(Path(spec['archive']).is_dir(), spec['archive'])
            self.assertGreaterEqual(len(spec['seeds']), 32, task_id)
            # 起点之前的示范动作走 human 导出，不走起点格（task87 起 LLM 示范格 label_source=llm_gpt_pro）
            meta = json.loads((Path(spec['archive']) / 'meta.json').read_text())
            self.assertTrue(meta['verified'], task_id)
            self.assertIn(meta['label_source'], ('human', 'llm_gpt_pro', 'llm_agent'), task_id)


if __name__ == '__main__':
    unittest.main()
