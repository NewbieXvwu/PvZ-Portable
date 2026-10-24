"""task29 第 2 步：课程控制器的元规则单测（不跑引擎、不读真实清单）。"""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]

from curriculum_queue import (DEMOTE_BELOW, FLOOR_ROUNDS, FLOOR_SHARE, GIB, MEMORY_RESERVE_BYTES,
                              MIN_SAMPLES, PACK_BUILDER_BYTES, PROMOTE_ABOVE, WINDOW, _descendant_rss,
                              adjust, allocate, build_plan, command_for, label_for, load_teacher_rates,
                              make_state, pack_command, pack_labels, parse_task_id, plan_round, rate,
                              reconcile_cells, startup_worker_estimate, training_base_ids, worker_count)

VALUES = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
NAMES = {'m125': 1.25, 'm150': 1.5, 'm175': 1.75, 'm200': 2.0, 'm250': 2.5, 'm300': 3.0}
N10_LEVELS = (11, 12, 13, 14, 16, 17, 18, 19)


def cell(wins, games, mult_index=0, last_round=0, level=7):
    return dict(family='f', level=level, mult_index=mult_index, last_round=last_round,
                windows={str(VALUES[mult_index]): {'recent': [1] * wins + [0] * (games - wins),
                                                   'wins': wins, 'games': games}})


class CurriculumRulesTest(unittest.TestCase):
    def test_zones_admission_only_measured_roof_and_no_fog(self):
        ids = training_base_ids()
        self.assertEqual({name for name in ids if name.startswith('roof_')},
                         {'roof_l41_train_r7'})
        self.assertFalse(any(name.startswith('fog_') for name in ids))

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
        # 预算紧张下连续模拟多轮：任何格两次被采的间隔不超过 FLOOR_ROUNDS + 1 轮。
        cells = {f'c{i:02d}': cell(5, 10, last_round=0) for i in range(20)}
        last_seen = {name: 0 for name in cells}
        for round_index in range(1, 13):
            allocate(cells, VALUES, budget=50, per_group=10, round_index=round_index)
            for name, picked in cells.items():
                if picked['last_round'] == round_index:
                    gap = round_index - last_seen[name]
                    self.assertLessEqual(gap, FLOOR_ROUNDS + 1, name)
                    last_seen[name] = round_index
        self.assertEqual(max(last_seen.values()), 12)          # 每个格都真的被轮到过

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
            'fresh_hard': cell(0, 10, last_round=1),        # 差距 1.0
            'old_hard': cell(0, 10, last_round=0),          # 差距 1.0，更久没采
            'easy': cell(9, 10),                            # 差距 0.1
        }
        picks = allocate(cells, VALUES, budget=30, per_group=10, round_index=1)
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
        # 与 _prepare_corpus 同一合并视图（采集优先）、与 pack() 同一段契约：段文件在即已建。
        import hashlib
        from pvz_recipe import load_recipe, training_row_options, write_recipe
        from pvz_training_pack import pack_options, segment_options, segment_path
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
                             {'pool_a': 200, 'pool_b': 50})
            fingerprint = hashlib.sha1(b's2/pool_b:50').hexdigest()[:16]
            options = pack_options(**training_row_options(recipe, ['pool_a', 'pool_b']))
            target = segment_path(root / 'pipeline' / 'frames', segment_options(options, 'pool_b', fingerprint))
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


if __name__ == '__main__':
    unittest.main()
