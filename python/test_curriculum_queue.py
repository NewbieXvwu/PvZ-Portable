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

import curriculum_queue
from curriculum_queue import (DEMOTE_BELOW, FLOOR_ROUNDS, FLOOR_SHARE, GIB, MEMORY_RESERVE_BYTES,
                              CURRICULUM_LADDERS, MIN_SAMPLES, PACK_BUILDER_BYTES, PROMOTE_ABOVE,
                              WINDOW, _descendant_rss,
                              adjust, allocate, build_plan, cell_level, command_for, growth_args, label_for,
                              load_demo_start_cells, load_teacher_rates, make_state, pack_command,
                              pack_labels, parse_levels, parse_task_id, plan_round, rate,
                              reconcile_cells, save_candidates_for,
                              reconcile_start_cells, renew_start_seeds, startup_worker_estimate,
                              student_rate, training_base_ids, worker_count)
from pvz_promotion import START_SEED_BLOCK, START_SEED_POOL

VALUES = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
NAMES = {'m125': 1.25, 'm150': 1.5, 'm175': 1.75, 'm200': 2.0, 'm250': 2.5, 'm300': 3.0}
N10_LEVELS = (11, 12, 13, 14, 16, 17, 18, 19)


def cell(wins, games, mult_index=0, last_round=0, level=7):
    return dict(family='f', level=level, mult_index=mult_index, last_round=last_round,
                windows={str(VALUES[mult_index]): {'recent': [1] * wins + [0] * (games - wins),
                                                   'wins': wins, 'games': games}})


class ReviewShareTest(unittest.TestCase):
    def fixture(self, multiplier=3.):
        base = 'night_l20_train_n10_orig'
        suffix = '' if multiplier == 1. else '_m050' if multiplier == .5 else '_m300'
        task_id = base + suffix
        tasks = {task_id: dict(task_id=task_id, level=20, deck=[0, 1, 2, 3, 4, 7],
                              zombie_count_multiplier=multiplier, seeds=list(range(1000, 1200)))}
        state = dict(round=0, log=[], offsets={task_id: 7}, cells={base: dict(
            family='f', level=20, last_round=0, mult_index=[.5, 1., 3.].index(multiplier),
            windows={str(multiplier): dict(recent=[0, 1] * 4, wins=4, games=8)})})
        return state, tasks, {'m050': .5, 'm300': 3.}, [.5, 1., 3.], {task_id: Path('/panel')}

    def test_review_share_extra_quota_uses_source_seeds_and_same_deck(self):
        for multiplier, share, extra in ((3., .25, 12), (.5, .25, 12), (3., .5, 24),
                                         (3., 0., 0), (1., .25, 0)):
            with self.subTest(multiplier=multiplier, share=share):
                args = self.fixture(multiplier)
                original = copy.deepcopy(args)
                plan, _ = plan_round(*args, budget=48, per_group=48, review_share=share)
                self.assertEqual(sum(item['games'] for item in plan), 48 + extra)
                self.assertEqual(plan[0]['games'], 48)
                if extra:
                    main, review = plan
                    self.assertEqual(review['task_id'], main['task_id'])
                    self.assertEqual(review['offset_key'], main['task_id'])
                    self.assertEqual(review['offset'], 55)
                    self.assertEqual(review['task_definition']['seeds'], list(range(1055, 1055 + extra)))
                    self.assertEqual(review['task_definition']['deck'], args[1][main['task_id']]['deck'])
                    self.assertEqual(review['task_definition']['zombie_count_multiplier'], 1.)
                self.assertEqual(args, original)

    def test_review_share_resume_keeps_pending_reviews_outside_course_budget(self):
        args = self.fixture()
        state, tasks, names, values, manifests = args
        normal, review = plan_round(*args, budget=48, per_group=48)[0]
        state['completed_groups'] = [dict(normal, status='done')]
        state['offsets'][normal['task_id']] = 55
        state['cells'][normal['base']]['last_round'] = 1
        base = 'night_l14_train_n10'
        state['cells'][base] = dict(family='f', level=14, last_round=0, mult_index=1, windows={})
        tasks[base] = dict(task_id=base, level=14, deck=[0, 1, 2, 3, 4, 7],
                           zombie_count_multiplier=1., seeds=list(range(2000, 2100)))
        manifests[base] = Path('/panel')
        pending, _ = plan_round(*args, budget=96, per_group=48)
        self.assertEqual(sum(p['games'] for p in pending if not p.get('review_of')), 48)
        self.assertEqual([p for p in pending if p.get('review_of')], [review])
        state['completed_groups'].append(dict(review, status='done'))
        state['offsets'][normal['task_id']] = 67
        pending, _ = plan_round(*args, budget=96, per_group=48)
        self.assertEqual([p['games'] for p in pending], [48])

    def test_review_share_demo_uses_actual_multiplier_and_original_future_seed_book(self):
        state, tasks, *_ = self.fixture()
        task_id, task = next(iter(tasks.items()))
        base, target = next(iter(state['cells'].items()))
        target.update(archive='/recorded', seeds=list(range(2000, 2200)), archive_cursor=4)
        group = dict(base=base, task_id=task_id, multiplier=90, demo_archive='/recorded',
            demo_decision=90, future_seeds=[1], manifest='/panel', label='demo',
            games=48, offset=7, offset_key=base, task_definition=dict(task, zombie_count_multiplier=1.))
        before = copy.deepcopy(target)
        self.assertEqual(curriculum_queue.review_groups(state, [group], tasks, .25, None, 256, []), [])
        group['task_definition']['zombie_count_multiplier'] = 3.
        review, = curriculum_queue.review_groups(state, [group], tasks, .25, None, 256, [])
        self.assertEqual(review['task_definition']['seeds'], list(range(2055, 2067)))
        self.assertNotIn('demo_archive', review)
        self.assertNotIn('future_seeds', review)
        self.assertEqual(review['offset_key'], base)
        self.assertEqual(target, before)

    def test_review_share_results_feed_training_without_changing_curriculum(self):
        state, tasks, names, values, manifests = self.fixture()
        review = plan_round(state, tasks, names, values, manifests, 48, 48)[0][1]
        before = copy.deepcopy(state['cells'])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(state=root / 'state.json', log=root / 'log', out=root / 'out',
                stop_flag=root / 'stop', python=Path('/python'), rollout_policy=Path('/student'),
                static_value=Path('/static'), value=Path('/value'), workers=2, executable=Path('/engine'),
                resource_dir=Path('/resources'), frames=root / 'frames', beta_threshold=.1,
                starved_beta=.5, probe_interval=3)
            def collect(command, **kwargs):
                definition = json.loads(command[command.index('--task-json') + 1])
                self.assertEqual(definition['zombie_count_multiplier'], 1.)
                self.assertEqual(command[command.index('--seed-offset') + 1], '0')
                folder = Path(command[command.index('--out') + 1])
                folder.mkdir(parents=True)
                (folder / 'games.jsonl').write_text('{"won":true}\n' * review['games'])
                return SimpleNamespace(returncode=0)
            with mock.patch.object(curriculum_queue.subprocess, 'run', side_effect=collect), \
                 mock.patch.object(curriculum_queue, 'mem_available_bytes', return_value=None):
                self.assertTrue(curriculum_queue.run_round(args, state, [review], []))
        self.assertEqual(state['cells'], before)
        self.assertEqual(state['offsets'][review['offset_key']], 67)
        self.assertIn(review['label'] + '_fast_search', pack_labels(state))


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
                handle.truncate(3 * 2 ** 20)                     # only st_size is read
            jobs = [('d', 'c', None, None, [big])]
            self.assertEqual(packer.pack_memory_limit(40 * GIB, jobs),
                             (40 * GIB - MEMORY_RESERVE_BYTES) // (PACK_BUILDER_BYTES + 3 * 2 ** 20))
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
        plan, changes = plan_round(state, tasks, NAMES, VALUES, manifests,
                                   budget=10, per_group=10)
        self.assertEqual(plan, [])
        self.assertTrue(any('seeds exhausted' in line and 'group skipped' in line for line in changes))
        self.assertIn('day_l7_train_d7', state['cells'])

    def test_save_candidates_switch_is_zone_level_with_per_task_override(self):
        self.assertEqual(parse_levels('31-40'), frozenset(range(31, 41)))
        self.assertEqual(parse_levels('31,32,37'), frozenset({31, 32, 37}))
        self.assertEqual(parse_levels(''), frozenset())
        with self.assertRaises(ValueError):
            parse_levels('40-31')
        levels = parse_levels('31-40')
        self.assertEqual(save_candidates_for(dict(level=32), None, levels), 1)
        self.assertEqual(save_candidates_for(dict(level=7), None, levels), 0)
        self.assertEqual(save_candidates_for(dict(level=7), dict(save_candidates=1), levels), 1)
        self.assertEqual(save_candidates_for(dict(level=32), dict(save_candidates=0), levels), 0)

    def test_command_for_passes_save_candidates_only_when_on(self):
        args = SimpleNamespace(python=Path('/nonexistent/python'), rollout_policy=Path('/m.pt'),
                               static_value=Path('/sv.pt'), value=Path('/v.pt'), workers=3,
                               executable=Path('/exe'), resource_dir=Path('/res'),
                               frames=Path('/frames'), out=Path('/out'))
        off = command_for(args, Path('/panel.json'), 'fog_l32_train', 0, 2, 'c1fogm100')
        self.assertNotIn('--save-candidates', off)
        on = command_for(args, Path('/panel.json'), 'fog_l32_train', 0, 2, 'c1fogm100',
                         save_candidates=1)
        self.assertEqual(on[on.index('--save-candidates') + 1], '1')

    def test_plan_round_marks_fog_cells_and_keeps_other_zones_off(self):
        tasks = {}
        for base, level in (('day_l7_train_d7', 7), ('fog_user_v2_l31_snow', 32)):
            for mult, suffix in [(1.0, ''), (1.25, '_m125')]:
                task_id = base + suffix
                tasks[task_id] = dict(task_id=task_id, level=level, deck=[0, 1, 2, 3, 4, 5, 7],
                                      zombie_count_multiplier=mult, seeds=list(range(40)))
        state = dict(round=0, offsets={}, cells={'day_l7_train_d7': cell(5, 10, level=7),
                                                 'fog_user_v2_l31_snow': cell(5, 10, level=32)})
        manifests = {task_id: Path('/tmp/panel.json') for task_id in tasks}
        plan, _ = plan_round(state, tasks, NAMES, VALUES, manifests, budget=20, per_group=10,
                             save_candidates_levels=parse_levels('31-40'))
        marks = {item['base']: item['save_candidates'] for item in plan}
        self.assertEqual(marks, {'day_l7_train_d7': 0, 'fog_user_v2_l31_snow': 1})
        # 清单逐格覆盖优先于区段默认：关掉雾格、打开日格。
        tasks['fog_user_v2_l31_snow']['save_candidates'] = 0
        tasks['day_l7_train_d7']['save_candidates'] = 1
        state['round'] = 1
        plan, _ = plan_round(state, tasks, NAMES, VALUES, manifests, budget=20, per_group=10,
                             save_candidates_levels=parse_levels('31-40'))
        self.assertEqual({item['base']: item['save_candidates'] for item in plan},
                         {'day_l7_train_d7': 1, 'fog_user_v2_l31_snow': 0})

    def test_native_ladder_cell_is_not_a_start_cell(self):
        # 10-26: native L25 carries a multiplier ladder too; reconcile_start_cells retired it and
        # plan_round planned it as a demo start. Start cells are recognised by their archive.
        state = dict(round=0, offsets={}, cells={'pool_l25_orig_train': dict(
            cell(5, 10), ladder=list(CURRICULUM_LADDERS['native_l25']), mult_index=0)})
        self.assertEqual(reconcile_start_cells(state, {}), [])
        self.assertIn('pool_l25_orig_train', state['cells'])
        self.assertNotIn('pool_l25_orig_train', state.get('retired_cells', {}))

    def test_exhausted_start_cell_is_retired_not_fatal(self):
        state = dict(round=0, offsets={'demo_x_m200_start': 250},
                     cells={'demo_x_m200': dict(cell(5, 10), ladder=[400, 300], mult_index=0,
                                                task_id='demo_x_m200_start', seeds=list(range(256)),
                                                archive='/a', family='f', level=32)})
        plan, changes = plan_round(state, {}, NAMES, VALUES, {'demo_x_m200_start': Path('/tmp/p.json')},
                                   budget=10, per_group=10)
        self.assertEqual(plan, [])
        self.assertNotIn('demo_x_m200', state['cells'])
        self.assertIn('demo_x_m200', state['retired_cells'])
        self.assertTrue(any('start cell retired' in line for line in changes))

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

    def test_round_log_keeps_completed_groups_across_restart(self):
        import curriculum_queue as queue
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(state=root / 'state.json', log=root / 'log', out=root,
                                   stop_flag=root / 'stop', workers=2,
                                   rollout_policy=Path('/m.pt'), beta_threshold=0.1,
                                   starved_beta=0.5, probe_interval=3)
            plan = [dict(base=f'base{index}', multiplier=1.0, task_id=f'task{index}',
                         offset=0, games=2, manifest='/manifest', label=f'group{index}')
                    for index in range(4)]
            plan[0].update(demo_archive='/demo', demo_decision=100, future_seeds=[10, 11])
            state = dict(round=0, offsets={}, log=[],
                         cells={item['base']: cell(0, 0) for item in plan})
            calls = []

            def collect(_command, **_kwargs):
                item = plan[len(calls)]
                folder = root / f'{item["label"]}_2'
                folder.mkdir()
                (folder / 'games.jsonl').write_text('{"won":true}\n{"won":false}\n')
                calls.append(item['label'])
                if len(calls) == 2:
                    args.stop_flag.touch()
                return SimpleNamespace(returncode=0)

            with mock.patch.object(queue, 'command_for', return_value=[]), \
                    mock.patch.object(queue.subprocess, 'run', side_effect=collect), \
                    mock.patch.object(queue, 'mem_available_bytes', return_value=None):
                self.assertFalse(queue.run_round(args, state, plan, []))
                state = json.loads(args.state.read_text())
                self.assertEqual(state['round'], 0)
                self.assertEqual(len(state['completed_groups']), 2)
                saved_cells = copy.deepcopy(state['cells'])
                saved_offsets = dict(state['offsets'])
                state['completed_groups'].append(copy.deepcopy(state['completed_groups'][0]))
                args.stop_flag.unlink()
                args.pipeline = mock.Mock()
                args.pipeline.reservation.return_value = 0
                args.pipeline.foreground_limit.return_value = 2
                self.assertTrue(queue.run_round(args, state, plan[2:], []))
            stored = json.loads(args.state.read_text())
            groups = stored['log'][0]['groups']
            self.assertEqual(stored['round'], 1)
            self.assertEqual([group['label'] for group in groups], calls)
            self.assertEqual(len(groups), 4)
            self.assertEqual(groups[0]['future_seeds'], [10, 11])
            self.assertNotIn('completed_groups', stored)
            for base, saved in saved_cells.items():
                if base in ('base0', 'base1'):
                    self.assertEqual(stored['cells'][base], saved)
            for task, offset in saved_offsets.items():
                self.assertEqual(stored['offsets'][task], offset)
            self.assertEqual(pack_labels(stored), sorted(label + '_fast_search' for label in calls))
            args.pipeline.enqueue.assert_called_once_with()
            self.assertIn('done: 4 groups, 8 games', args.log.read_text())

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
                        for name in ('dw4i', 'dw4i2_v2timer', 'dw4i2_v2timer_history10')]
            variants.append(dict(variants[0], context=True, aux_weight=1.0, aux_horizon=3,
                                 cell_soft_weight=0.0))
            variants.append(dict(variants[0], label_keep={'.*': .25}))
            for i, recipe in enumerate(variants):
                with self.subTest(variant=i):
                    recipe.update(out=str(root / f'out{i}'), data=str(root / 'frames'))
                    path = root / f'recipe{i}.json'
                    write_recipe(path, recipe)
                    args = SimpleNamespace(python=Path(sys.executable),
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
        self.assertEqual(worker_count(20 * GIB, per_worker, 16), 16)
        self.assertEqual(worker_count(10 * GIB, per_worker, 16), 7)
        self.assertEqual(worker_count(28 * GIB, 512 * 1024 ** 2, 8), 8)  # 上限
        self.assertEqual(worker_count(2 * GIB, per_worker, 16), 1)
        self.assertEqual(MEMORY_RESERVE_BYTES, 3 * GIB)

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

    def test_group_command_admits_the_start_width_and_the_cap(self):
        """task138: 一组开工时把'开工宽度'和'上限'分开传，池子才有扩容空间。"""
        args = SimpleNamespace(python=Path('/nonexistent/python'), workers=16,
                               rollout_policy=Path('/m/student.pt'), static_value=Path('/m/static.pt'),
                               value=Path('/m/value.pt'), executable=Path('/m/pvz-portable'),
                               resource_dir=Path('/m/resources'), frames=Path('/m/frames'),
                               out=Path('/m/out'))
        command = command_for(args, Path('/m/panel.json'), 'day_l7', 5, 48, 'c1d7m100',
                              workers=2, upper_workers=16, per_worker_bytes=2 * GIB,
                              headroom_bytes=4 * GIB)
        self.assertEqual(command[command.index('--workers') + 1], '16')
        self.assertEqual(command[command.index('--initial-workers') + 1], '2')
        self.assertEqual(command[command.index('--worker-memory-bytes') + 1], str(2 * GIB))
        self.assertEqual(command[command.index('--memory-headroom-bytes') + 1], str(4 * GIB))

    def test_growth_args_keep_the_old_fixed_width(self):
        args = SimpleNamespace(workers=16)
        # 没有声明上限（探针、一次性诊断）= 固定宽度，行为与 task138 之前一致。
        self.assertEqual(growth_args(args, 3, None, 0, 0), ['--workers', '3'])
        # 上限已给但没实测 RSS：池子不扩容，参数仍写明上限供日志审计。
        self.assertEqual(growth_args(args, 3, 16, 0, 0),
                         ['--workers', '16', '--initial-workers', '3',
                          '--worker-memory-bytes', '0', '--memory-headroom-bytes', '0'])
        # 宽度已到上限且没有实测值：不必多传参数。
        self.assertEqual(growth_args(args, 16, 16, 0, 0), ['--workers', '16'])

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

def queue_health():
    """A background job stub whose recovery gate is always open."""
    return type('Health', (), {'wait': lambda self: False, 'pause': lambda self, reason: None,
                               'resume': lambda self: None})()


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
            args = SimpleNamespace(state=Path(tmp) / 'state.json', workers=2,
                                   seed_ledger=Path(tmp) / 'gate_ledger.json')
            with mock.patch.object(TrainingPipeline, '_run', fake_run), \
                    mock.patch.object(TrainingPipeline, '_plan_build', lambda self, recipe: None):
                pipeline = TrainingPipeline(args, state)
            pipeline.executor.shutdown(wait=True)
            self.assertEqual(sorted(started), [1, 2])          # running/failed 都重跑
            statuses = {job['round']: job['status'] for job in state['training_jobs']}
            self.assertEqual(statuses, {1: 'queued', 2: 'queued', 3: 'done'})   # done 不动
            self.assertEqual(len(pipeline.futures), 2)

    def test_supersede_targets_the_newest_waiting_round(self):
        # task141：只有"会带来更新数据"的作业才使旧作业作废；isolated / 已 superseded 不算。
        from curriculum_queue import TrainingPipeline
        pipeline = TrainingPipeline.__new__(TrainingPipeline)
        jobs = [dict(round=index, status='queued') for index in (28, 29, 30)]
        pipeline.state = {'training_jobs': jobs}
        self.assertEqual([pipeline._superseded_by(job) for job in jobs], [30, 30, None])
        pipeline.state['training_jobs'].append(dict(round=31, status='failed', isolated=True))
        pipeline.state['training_jobs'].append(dict(round=32, status='superseded'))
        self.assertEqual(pipeline._superseded_by(jobs[0]), 30)
        pipeline.state['training_jobs'].append(dict(round=31, status='failed', attempts=1))
        self.assertEqual(pipeline._superseded_by(jobs[0]), 31)   # 会重试的新作业仍然算

    def test_backlog_trains_only_the_newest_round(self):
        # task141：队列 [28, 29, 30] 时执行器只训 30；28/29 标 superseded，记录被谁取代，
        # 不训练不过门禁；重启后两者保持 superseded，不会被重新排队。
        from curriculum_queue import GATE_VERSION, TrainingPipeline
        with tempfile.TemporaryDirectory() as tmp:
            jobs = [self._job(tmp, 'queued', index) for index in (28, 29, 30)]
            for job in jobs:
                job['seed_block'] = [1000, 1008]
            state = {'training_jobs': jobs}
            args = SimpleNamespace(state=Path(tmp) / 'state.json', log=Path(tmp) / 'log',
                                   python=Path(sys.executable), workers=2,
                                   seed_ledger=Path(tmp) / 'ledger.json',
                                   gate_version=GATE_VERSION)
            commands = []
            def fake_command(self, command, log):
                commands.append('pack' if command == ['pack'] else 'train')
                return 0
            def fake_evaluate(self, job):
                return {'promotion': dict(version=GATE_VERSION,
                                          policy=curriculum_queue.promotion_policy(GATE_VERSION),
                                          approved=False, contract={},
                                          reasons=['evaluation_error: stub'])}
            with mock.patch.object(TrainingPipeline, '_plan_build', lambda self, recipe: None), \
                    mock.patch.object(TrainingPipeline, '_admit', lambda self: None), \
                    mock.patch.object(TrainingPipeline, '_prepare_corpus', lambda self, labels: None), \
                    mock.patch.object(TrainingPipeline, '_command', fake_command), \
                    mock.patch.object(TrainingPipeline, '_evaluate', fake_evaluate), \
                    mock.patch.object(TrainingPipeline, 'evaluation_contract', lambda self, job: {}), \
                    mock.patch.object(curriculum_queue, 'load_recipe',
                                      lambda path, kind=None: {'labels': ['x'], 'data': '/tmp/frames'}), \
                    mock.patch.object(curriculum_queue, 'pack_command', lambda args, labels: ['pack']):
                pipeline = TrainingPipeline(args, state)
                pipeline.executor.shutdown(wait=True)
                pipeline.poll()
                statuses = {job['round']: job['status'] for job in state['training_jobs']}
                self.assertEqual(statuses, {28: 'superseded', 29: 'superseded', 30: 'done'})
                self.assertEqual({job['round']: job.get('superseded_by')
                                  for job in state['training_jobs'][:2]}, {28: 30, 29: 30})
                self.assertEqual(commands, ['pack', 'train'])       # 只有第 30 轮真的训练
                log = (Path(tmp) / 'log').read_text()
                self.assertEqual(log.count('superseded by round 30'), 2)
                self.assertFalse((Path(tmp) / 'ledger.json').exists())   # 种子账本未被写入
                saved = json.loads((Path(tmp) / 'state.json').read_text())
                self.assertEqual([job['status'] for job in saved['training_jobs']],
                                 ['superseded', 'superseded', 'done'])
                restarted = TrainingPipeline(args, saved)           # 重启：不复活旧作业
                restarted.executor.shutdown(wait=True)
                self.assertEqual(restarted.futures, [])
                self.assertEqual([job['status'] for job in saved['training_jobs']],
                                 ['superseded', 'superseded', 'done'])

    def test_run_reuses_trained_json_without_retraining(self):
        # 训练完成（trained.json 已写）、评估前被杀：重启后复用报告，直接进评估，
        # _admit/_command（重训入口）被调到即失败。
        from curriculum_queue import TrainingPipeline
        with tempfile.TemporaryDirectory() as tmp:
            job = self._job(tmp, 'queued')
            trained = {'round': job['round']}
            (Path(job['recipe']).parent / 'trained.json').write_text(json.dumps(trained))
            pipeline = TrainingPipeline.__new__(TrainingPipeline)
            pipeline.failed = False
            pipeline.args = SimpleNamespace(log=Path(tmp) / 'log')
            pipeline.lock_fd = None
            pipeline.game_health = SimpleNamespace(wait=lambda: True, pause=lambda message: None)
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
        pipeline.args = SimpleNamespace(log=Path(tmp) / 'log', python=Path(sys.executable),
                                        workers=2, game_health=None)
        pipeline.game_health = queue_health()
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
            args = SimpleNamespace(state=Path(tmp) / 'state.json', workers=2)
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

    def setUp(self):
        self.archive_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.archive_dir.cleanup)
        self.archive = Path(self.archive_dir.name)
        (self.archive / 'meta.json').write_text(json.dumps({'verified': True, 'task': {'seed': 42}}))

    BASE = 'demo_fog_33_cactus_snow_m200'
    LADDER = [800, 600, 400]

    def start_cell(self, mult_index=0, windows=None):
        return dict(family='demo_start', level=33, mult_index=mult_index, last_round=0,
                    windows=windows or {}, ladder=list(self.LADDER), task_id=self.BASE,
                    archive=str(self.archive), seeds=[900 + index for index in range(6)])

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
        plan, _ = plan_round(state, {self.BASE: {'seeds': [42]}}, NAMES, VALUES, manifests, budget=2, per_group=2)
        self.assertEqual(len(plan), 1)
        item = plan[0]
        self.assertEqual((item['task_id'], item['demo_decision'], item['offset']), (self.BASE, 800, 0))
        self.assertEqual(item['future_seeds'], [900, 901])
        self.assertEqual(item['label'], 'c1demofog33cactussnowm200s800_o0')
        self.assertEqual(item['multiplier'], 800)             # 档位 = 起点决策号
        state['offsets'][self.BASE] = 5
        plan, _ = plan_round(state, {self.BASE: {'seeds': [42]}}, NAMES, VALUES, manifests, budget=1, per_group=1)
        self.assertEqual(plan[0]['future_seeds'], [905])       # 只剩一个未来种子
        self.assertEqual(plan[0]['label'], 'c1demofog33cactussnowm200s800_o5')
        state['offsets'][self.BASE] = 6
        plan, changes = plan_round(state, {self.BASE: {'seeds': [42]}}, NAMES, VALUES, manifests, budget=1, per_group=1)
        self.assertEqual(plan, [])
        self.assertNotIn(self.BASE, state['cells'])
        self.assertIn(self.BASE, state['retired_cells'])
        self.assertTrue(any('start cell retired' in change for change in changes))

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
            # Missing archives are loaded as specs; planning retires the affected cell.
            document['tasks'][0]['demo_start']['decisions'] = self.LADDER
            document['tasks'][0]['demo_start']['archive'] = str(Path(tmp) / 'none')
            path.write_text(json.dumps(document))
            cells, _ = load_demo_start_cells(path)
            self.assertIn(self.BASE, cells)

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
        self.assertEqual(set(cells), {'demo_fog_31_snow_m200_hv2', 'demo_fog_32_snow_m200_hv2',
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
                                      'demo_roof_49_doom_m200_s20261050', 'demo_roof_49_doom_m200_s20261051',
                                      # task107: 原生传送带用户示范起点格（L40 沿用 task75 的 storm 口径）
                                      'demo_pool_30_orig_m100', 'demo_storm_40_orig_m100',
                                      'demo_boss_50_orig_m100',
                                      # task112: L25 小僵尸泳池 / L45 蹦极闪电战的原生传送带起点格
                                      'demo_pool_25_orig_m100', 'demo_roof_45_orig_m100'})
        for task_id, spec in cells.items():
            self.assertGreaterEqual(len(spec['ladder']), 1, task_id)
            self.assertEqual(spec['ladder'], sorted(spec['ladder'], reverse=True), task_id)
            self.assertTrue(Path(spec['archive']).is_dir(), spec['archive'])
            self.assertGreaterEqual(len(spec['seeds']), 32, task_id)
            # 起点之前的示范动作走 human 导出，不走起点格（task87 起 LLM 示范格 label_source=llm_gpt_pro）
            meta = json.loads((Path(spec['archive']) / 'meta.json').read_text())
            self.assertTrue(meta['verified'], task_id)
            self.assertIn(meta['label_source'], ('human', 'llm_gpt_pro'), task_id)


class StartSeedRenewalTest(unittest.TestCase):
    """task99：起点格的未来种子按需续领（多段种子表），不再用完即退休。"""

    BASE = 'demo_fog_32_snow_m200'

    def setUp(self):
        self.archive_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.archive_dir.cleanup)
        self.archive = Path(self.archive_dir.name)
        (self.archive / 'meta.json').write_text(json.dumps({'verified': True, 'task': {'seed': 42}}))
        self.tasks = {self.BASE: {'seeds': [42]}}


    def cell(self, size=256):
        seeds = list(range(size))
        return dict(family='demo_start', level=32, mult_index=0, last_round=0, windows={},
                    ladder=[1231, 1119], task_id=self.BASE, archive=str(self.archive),
                    seeds=seeds, seed_blocks=[[0, size]])

    def spec(self, size=256):
        return dict(self.cell(size), windows={})

    def test_plan_round_renews_before_retiring(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / 'start_ledger.json'
            state = dict(round=0, offsets={self.BASE: 240}, cells={self.BASE: self.cell()})
            plan, changes = plan_round(state, self.tasks, NAMES, VALUES, {self.BASE: Path('/tmp/p.json')},
                                       budget=48, per_group=48, start_seed_ledger=ledger)
            block = [START_SEED_POOL[0], START_SEED_POOL[0] + START_SEED_BLOCK]
            self.assertEqual(len(plan), 1)
            self.assertEqual(plan[0]['offset'], 240)          # 偏移语义不变：仍是表头绝对下标
            self.assertEqual(plan[0]['future_seeds'],
                             list(range(240, 256)) + list(range(block[0], block[0] + 32)))
            self.assertEqual(state['cells'][self.BASE]['seed_blocks'], [[0, 256], block])
            self.assertNotIn(self.BASE, state.get('retired_cells', {}))
            self.assertTrue(any('seeds renewed +256' in line for line in changes))
            self.assertEqual(json.loads(ledger.read_text())['allocated'], [block])

    def test_restart_at_manifest_end_can_renew(self):
        state = dict(round=0, offsets={self.BASE: 256}, cells={self.BASE: self.cell()})
        reconcile_start_cells(state, {self.BASE: self.spec()})
        with tempfile.TemporaryDirectory() as tmp:
            plan, _ = plan_round(state, self.tasks, NAMES, VALUES,
                                 {self.BASE: Path('/tmp/p.json')}, 48, 48,
                                 start_seed_ledger=Path(tmp) / 'ledger.json')
        self.assertEqual(plan[0]['offset'], 256)
        self.assertEqual(plan[0]['future_seeds'], list(range(START_SEED_POOL[0], START_SEED_POOL[0] + 48)))

    def test_renewed_blocks_are_disjoint_and_ordered(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / 'start_ledger.json'
            cell = self.cell()
            first = renew_start_seeds(cell, ledger)
            second = renew_start_seeds(cell, ledger)
            self.assertEqual(first[1], second[0])             # 相邻但不相交，从不复用
            self.assertEqual(cell['seeds'][256:], list(range(*first)) + list(range(*second)))
            self.assertEqual(cell['seed_blocks'], [[0, 256], first, second])
            self.assertEqual(json.loads(ledger.read_text())['allocated'], [first, second])

    def test_large_group_renews_until_enough_seeds_are_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / 'start_ledger.json'
            state = dict(round=0, offsets={self.BASE: 240}, cells={self.BASE: self.cell()})
            plan, _ = plan_round(state, self.tasks, NAMES, VALUES, {self.BASE: Path('/tmp/p.json')},
                                 budget=300, per_group=300, start_seed_ledger=ledger)
            self.assertEqual(len(plan), 1)
            self.assertEqual(len(plan[0]['future_seeds']), 300)
            self.assertEqual(len(json.loads(ledger.read_text())['allocated']), 2)
            self.assertNotIn(self.BASE, state.get('retired_cells', {}))

    def test_explicit_seed_order_survives_renewal_and_restart(self):
        from check_start_seed_state import check_state
        with tempfile.TemporaryDirectory() as tmp:
            cell = dict(self.cell(), seeds=[8, 9, 2, 3, 6], seed_blocks=[])
            spec = dict(cell)
            state = dict(round=0, offsets={self.BASE: 3}, cells={self.BASE: cell})
            ledger = Path(tmp) / 'start_ledger.json'
            plan, _ = plan_round(state, self.tasks, NAMES, VALUES, {self.BASE: Path('/tmp/p.json')},
                                 budget=10, per_group=10, start_seed_ledger=ledger)
            state = json.loads(json.dumps(state))
            reconcile_start_cells(state, {self.BASE: spec})
            self.assertEqual(state['cells'][self.BASE]['seeds'][:5], spec['seeds'])
            self.assertEqual(plan[0]['future_seeds'][:2], [3, 6])
            blocks = [tuple(block) for block in json.loads(ledger.read_text())['allocated']]
            self.assertEqual(check_state(state, blocks), ([], []))

    def test_two_cells_never_get_the_same_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / 'start_ledger.json'
            first, second = self.cell(), dict(self.cell(), task_id='other')
            left = renew_start_seeds(first, ledger)
            right = renew_start_seeds(second, ledger)
            self.assertLessEqual(left[1], right[0])            # 相邻块，跨格也不相交
            self.assertFalse(set(first['seeds'][256:]) & set(second['seeds'][256:]))
            self.assertEqual(json.loads(ledger.read_text())['allocated'], [left, right])

    def test_manifest_block_inside_the_renewal_pool_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / 'archive'
            archive.mkdir()
            (archive / 'meta.json').write_text(json.dumps({'verified': True, 'label_source': 'human'}))
            manifest = Path(tmp) / 'manifest.json'
            manifest.write_text(json.dumps({'tasks': [dict(
                task_id='demo_x_m200', level=32, seeds=[1], zombie_count_multiplier=2.0,
                deck=[0, 1, 2, 3, 4, 5],
                demo_start=dict(archive=str(archive), decisions=[800, 600], index=0,
                                future_seed_start=START_SEED_POOL[0], future_seed_count=256))]}))
            with self.assertRaises(ValueError) as caught:
                load_demo_start_cells(manifest)
            self.assertIn('renewal pool', str(caught.exception))
            document = json.loads(manifest.read_text())
            document['tasks'][0]['demo_start']['future_seeds'] = [9, START_SEED_POOL[0], 3]
            manifest.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, 'renewal pool'):
                load_demo_start_cells(manifest)

    def test_registered_start_blocks_are_unique_after_deploy14d_hotfix(self):
        tasks = json.loads((ROOT / 'configs/human_demo_tasks.json').read_text())['tasks']
        owners = {}
        for task in tasks:
            start = task.get('demo_start')
            if not start:
                continue
            seeds = start.get('future_seeds') or range(start['future_seed_start'],
                start['future_seed_start'] + start.get('future_seed_count', 256))
            for seed in seeds:
                self.assertNotIn(seed, owners, f'{task["task_id"]} overlaps {owners.get(seed)}')
                owners[seed] = task['task_id']
        self.assertEqual(len(owners), 29 * 256)
        # task99's seven inherited starts were invalid and were removed by cbd96daf.
        stale = {'demo_roof_41_doom_m200', 'demo_roof_46_doom_m300', 'demo_roof_46_doom_m400',
                 'demo_roof_48_doom_m300', 'demo_roof_48_doom_m400',
                 'demo_roof_49_doom_m300', 'demo_roof_49_doom_m400'}
        self.assertTrue(all('demo_start' not in task for task in tasks if task['task_id'] in stale))

    def test_pool_is_outside_every_registered_segment(self):
        inventory = json.loads((ROOT / 'MEMORY/docs/gate-seed-inventory.json').read_text())
        overlaps = [[max(START_SEED_POOL[0], low), min(START_SEED_POOL[1], high)]
                    for low, high in inventory['occupied']
                    if (low, high) != START_SEED_POOL and low < START_SEED_POOL[1]
                    and START_SEED_POOL[0] < high]
        self.assertEqual(overlaps, [])

    def test_restart_keeps_the_renewed_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / 'start_ledger.json'
            state = dict(round=0, offsets={}, cells={self.BASE: self.cell()})
            renew_start_seeds(state['cells'][self.BASE], ledger)
            path = Path(tmp) / 'state.json'
            path.write_text(json.dumps(state))
            reloaded = json.loads(path.read_text())
            lines = reconcile_start_cells(reloaded, {self.BASE: self.spec()})
            self.assertEqual(lines, [])                       # 清单块仍是第一段：不覆盖、不报改动
            self.assertEqual(len(reloaded['cells'][self.BASE]['seeds']), 512)
            self.assertEqual(len(reloaded['cells'][self.BASE]['seed_blocks']), 2)
            reloaded['offsets'][self.BASE] = 300              # 偏移继续在续领段里走
            plan, _ = plan_round(reloaded, self.tasks, NAMES, VALUES, {self.BASE: Path('/tmp/p.json')},
                                 budget=10, per_group=10)
            self.assertEqual(plan[0]['future_seeds'],
                             list(range(START_SEED_POOL[0] + 44, START_SEED_POOL[0] + 54)))

    def test_twelve_rounds_of_renewals_survive_restarts(self):
        # 长跑：每轮 48 局、每轮都重启一次；种子表要按需长到第三段，且全程不退休、不重采。
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / 'start_ledger.json'
            state = dict(round=0, offsets={}, cells={self.BASE: self.cell()})
            used = []
            for round_index in range(12):
                state['round'] = round_index
                plan, _ = plan_round(state, self.tasks, NAMES, VALUES, {self.BASE: Path('/tmp/p.json')},
                                     budget=48, per_group=48, start_seed_ledger=ledger)
                self.assertEqual(len(plan), 1, round_index)
                item = plan[0]
                self.assertEqual(len(item['future_seeds']), item['games'])
                used.extend(item['future_seeds'])
                state['offsets'][self.BASE] = item['offset'] + item['games']
                state['cells'][self.BASE]['last_round'] = round_index + 1   # run_round 的落盘
                state = json.loads(json.dumps(state))            # 落盘
                reconcile_start_cells(state, {self.BASE: self.spec()})   # 重启
            cell = state['cells'][self.BASE]
            self.assertNotIn(self.BASE, state.get('retired_cells', {}))
            self.assertEqual(len(cell['seed_blocks']), 3)        # 256 + 256 + 256 ≥ 576
            self.assertEqual(cell['seeds'], [seed for low, high in cell['seed_blocks']
                                             for seed in range(low, high)])
            self.assertEqual(json.loads(ledger.read_text())['allocated'], cell['seed_blocks'][1:])
            self.assertEqual(state['offsets'][self.BASE], len(used))
            self.assertEqual(len(set(used)), len(used))           # 一个种子都没重复
            self.assertGreater(len(used), 512)                    # 确实用到了第三段
            self.assertLessEqual(len(used), len(cell['seeds']))

    def test_spec_block_change_resets_the_table(self):
        state = dict(round=0, offsets={}, cells={self.BASE: self.cell()})
        with tempfile.TemporaryDirectory() as tmp:
            renew_start_seeds(state['cells'][self.BASE], Path(tmp) / 'start_ledger.json')
        spec = self.spec()
        spec['seeds'] = list(range(500, 756))
        spec['seed_blocks'] = [[500, 756]]
        lines = reconcile_start_cells(state, {self.BASE: spec})
        self.assertEqual(state['cells'][self.BASE]['seeds'], list(range(500, 756)))
        self.assertEqual(state['cells'][self.BASE]['seed_blocks'], [[500, 756]])
        self.assertTrue(any('seeds' in line for line in lines))

    def test_retired_start_cell_reactivates_and_renews(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / 'start_ledger.json'
            retired = dict(self.cell(), retired_reason='seeds exhausted')
            state = dict(round=0, offsets={self.BASE: 216}, cells={},
                         retired_cells={self.BASE: retired})
            lines = reconcile_start_cells(state, {self.BASE: self.spec()}, restore=(self.BASE,))
            self.assertIn(self.BASE, state['cells'])
            self.assertNotIn(self.BASE, state['retired_cells'])
            self.assertTrue(any('start cell restored' in line for line in lines))
            plan, changes = plan_round(state, self.tasks, NAMES, VALUES, {self.BASE: Path('/tmp/p.json')},
                                       budget=48, per_group=48, start_seed_ledger=ledger)
            self.assertEqual(len(plan), 1)
            self.assertEqual(len(plan[0]['future_seeds']), 48)
            self.assertEqual(plan[0]['future_seeds'][40:],
                             list(range(START_SEED_POOL[0], START_SEED_POOL[0] + 8)))
            self.assertTrue(any('seeds renewed' in line for line in changes))

    def test_retired_multiblock_cell_is_only_restored_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            cell = self.cell()
            renew_start_seeds(cell, Path(tmp) / 'start_ledger.json')
            cell['mult_index'] = 1
            cell['windows'] = {'1119': {'wins': 2, 'games': 3, 'recent': [1, 0, 1]}}
            state = dict(round=0, offsets={self.BASE: 300}, cells={},
                         retired_cells={self.BASE: cell})
            state = json.loads(json.dumps(state))
            original = json.loads(json.dumps(state))
            self.assertEqual(reconcile_start_cells(state, {self.BASE: self.spec()}), [])
            self.assertEqual(state, original)
            reconcile_start_cells(state, {self.BASE: self.spec()}, restore=(self.BASE,))
            self.assertEqual(state['cells'][self.BASE], original['retired_cells'][self.BASE])
            self.assertEqual(state['offsets'][self.BASE], 300)
            self.assertNotIn(self.BASE, state['retired_cells'])

    def test_restore_warns_when_future_seeds_overlap_a_live_cell(self):
        state = dict(round=0, offsets={self.BASE: 216},
                     cells={'other': dict(self.cell(), task_id='other')},
                     retired_cells={self.BASE: self.cell()})
        specs = {self.BASE: self.spec(), 'other': dict(self.spec(), task_id='other')}
        lines = reconcile_start_cells(state, specs, restore=(self.BASE,))
        self.assertTrue(any('WARNING future seeds overlap other' in line for line in lines))

    def test_legacy_restore_survives_the_next_restart(self):
        state = dict(round=0, offsets={}, cells={}, retired_cells={self.BASE: self.cell()})
        reconcile_start_cells(state, {}, restore=(self.BASE,))
        self.assertTrue(state['cells'][self.BASE]['legacy_restored'])
        reconcile_start_cells(state, {})                      # 清单里没有它，也不能再退休
        self.assertIn(self.BASE, state['cells'])
        self.assertNotIn(self.BASE, state['retired_cells'])

    def test_corrupt_ledger_retires_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / 'start_ledger.json'
            ledger.write_text(json.dumps(dict(pool=list(START_SEED_POOL),
                                              pools=[list(START_SEED_POOL)],
                                              allocated=[[1, 2]])))   # 块不在池里 → 账本损坏
            state = dict(round=0, offsets={self.BASE: 250}, cells={self.BASE: self.cell()})
            plan, changes = plan_round(state, self.tasks, NAMES, VALUES, {self.BASE: Path('/tmp/p.json')},
                                       budget=10, per_group=10, start_seed_ledger=ledger)
            self.assertEqual(plan, [])
            self.assertIn(self.BASE, state['retired_cells'])
            self.assertTrue(any('seed renewal failed' in line and 'corrupt' in line
                                for line in changes))

    def test_exhausted_pool_retires_the_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / 'start_ledger.json'
            ledger.write_text(json.dumps(dict(pool=list(START_SEED_POOL), pools=[list(START_SEED_POOL)],
                                              allocated=[list(START_SEED_POOL)])))
            state = dict(round=0, offsets={self.BASE: 250}, cells={self.BASE: self.cell()})
            plan, changes = plan_round(state, self.tasks, NAMES, VALUES, {self.BASE: Path('/tmp/p.json')},
                                       budget=10, per_group=10, start_seed_ledger=ledger)
            self.assertEqual(plan, [])
            self.assertIn(self.BASE, state['retired_cells'])
            self.assertEqual(state['retired_cells'][self.BASE]['retired_reason'], 'seeds exhausted')
            self.assertTrue(any('seed renewal failed' in line for line in changes))
            self.assertTrue(any('start cell retired' in line for line in changes))


if __name__ == '__main__':
    unittest.main()


class WorkerProbeFloorTest(unittest.TestCase):
    def test_implausibly_small_probe_falls_back_to_default(self):
        import curriculum_queue as queue
        args = SimpleNamespace(log=Path(tempfile.mkdtemp()) / 'log')
        measure, memory = queue.measure_worker_bytes, queue.mem_available_bytes
        queue.mem_available_bytes = lambda: 16 * queue.GIB
        try:
            queue.measure_worker_bytes = lambda _args, _item: 20 * 1024 * 1024
            self.assertEqual(queue.startup_worker_estimate(args, [dict()]), queue.DEFAULT_WORKER_BYTES)
            queue.measure_worker_bytes = lambda _args, _item: queue.GIB // 2
            self.assertEqual(queue.startup_worker_estimate(args, [dict()]), queue.GIB // 2)
        finally:
            queue.measure_worker_bytes, queue.mem_available_bytes = measure, memory


class RandomDeckCellsTest(unittest.TestCase):
    """task137：随机卡组格（0.5 起步的阶梯、每轮份额上限、挑选/训练两批卡组）。"""

    def setUp(self):
        import make_random_deck_cells as generator
        registry = json.loads((ROOT / 'configs/random_deck_cells_v5.json').read_text())
        panels = generator.build_panels(registry)
        self.tasks = {task['task_id']: task for task in panels['training_panel']['tasks']}
        self.names = {'m%03d' % round(m * 100): m for m in generator.LADDER if m != 1.0}
        self.one_cell = {task_id: task for task_id, task in self.tasks.items()
                         if task_id.startswith('rd137_l7_train_d01')}

    def test_load_panel_merges_extra_manifest(self):
        from curriculum_queue import load_panel
        with tempfile.TemporaryDirectory() as directory:
            main = Path(directory) / 'main.json'
            main.write_text(json.dumps(dict(tasks=[dict(task_id='day_l7_train_d7', level=7,
                                                        zombie_count_multiplier=1.0, seeds=[1])])))
            extra = Path(directory) / 'extra.json'
            extra.write_text(json.dumps(dict(tasks=list(self.one_cell.values()))))
            with mock.patch.object(curriculum_queue, 'source_paths', lambda overrides: {}):
                tasks, values, names, manifests = load_panel(main, [extra])
        self.assertEqual(values, [1.0])                      # <1 档不进全局倍率表
        self.assertEqual(names['m050'], 0.5)
        self.assertIn('rd137_l7_train_d01_m050', tasks)
        self.assertEqual(manifests['rd137_l7_train_d01'], extra)
        self.assertEqual(manifests['day_l7_train_d7'], main)

    def test_load_panel_rejects_duplicate_task(self):
        from curriculum_queue import load_panel
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'p.json'
            document = dict(tasks=list(self.one_cell.values()))
            path.write_text(json.dumps(document))
            with mock.patch.object(curriculum_queue, 'source_paths', lambda overrides: {}):
                with self.assertRaises(ValueError):
                    load_panel(path, [path])

    def test_cells_start_at_half_and_follow_the_manifest_ladder(self):
        state = dict(cells={}, retired_cells={})
        reconcile_cells(state, self.tasks, self.names, {})
        cell_state = state['cells']['rd137_l7_train_d01']
        self.assertEqual(cell_state['ladder'], [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])
        self.assertEqual(cell_state['mult_index'], 0)
        self.assertTrue(cell_state['random_deck_cell'])
        self.assertEqual(cell_level(cell_state, VALUES), 0.5)

    def test_random_cells_are_retired_when_the_extra_panel_is_dropped(self):
        state = dict(cells={}, retired_cells={})
        reconcile_cells(state, self.tasks, self.names, {})
        self.assertIn('rd137_l7_train_d01', state['cells'])
        lines = reconcile_cells(state, {'day_l7_train_d7': self.tasks['rd137_l7_train_d01']},
                                self.names, {})
        self.assertNotIn('rd137_l7_train_d01', state['cells'])
        self.assertIn('rd137_l7_train_d01', state['retired_cells'])
        self.assertTrue(any('cell retired' in line for line in lines))

    def test_half_multiplier_demotes_and_floor_holds(self):
        ladder = [0.5, 0.75, 1.0, 1.25]
        cells = {
            'top': dict(family='f', level=7, mult_index=2, last_round=0, random_deck_cell=True,
                        ladder=list(ladder),
                        windows={'1.0': {'recent': [0] * 10, 'wins': 0, 'games': 10}}),
            'bottom': dict(family='f', level=7, mult_index=0, last_round=0, random_deck_cell=True,
                           ladder=list(ladder),
                           windows={'0.5': {'recent': [0] * 10, 'wins': 0, 'games': 10}}),
        }
        adjust(cells, VALUES)
        self.assertEqual(cells['top']['mult_index'], 1)     # 1.0 -> 0.75
        self.assertEqual(cells['bottom']['mult_index'], 0)  # 0.5 不再降

    def test_allocation_keeps_random_cells_under_the_share_cap(self):
        cells = {f'day_l{i}_train_d7': cell(0, 10, last_round=0) for i in range(12)}
        cells.update({f'rd137_l7_train_d0{i}': dict(family='f', level=7, mult_index=0, last_round=0,
                                                    random_deck_cell=True, windows={})
                      for i in range(1, 5)})
        picks = allocate(cells, VALUES, budget=100, per_group=10, round_index=3,
                         random_deck_cap=15)
        random_games = sum(games for base, _m, games in picks
                           if cells[base].get('random_deck_cell'))
        self.assertLessEqual(random_games, 15)
        self.assertGreater(random_games, 0)
        # 预算是上限：15 局保底 + 8 个整组 80 局，剩 5 局不够一组。
        self.assertLessEqual(sum(games for _b, _m, games in picks), 100)

    def test_zero_cap_excludes_random_cells(self):
        cells = {f'day_l{i}_train_d7': cell(0, 10, last_round=0) for i in range(4)}
        cells['rd137_l7_train_d01'] = dict(family='f', level=7, mult_index=0, last_round=0,
                                           random_deck_cell=True, windows={})
        picks = allocate(cells, VALUES, budget=20, per_group=10, round_index=1,
                         random_deck_cap=0)
        self.assertNotIn('rd137_l7_train_d01', {base for base, _m, _g in picks})

    def test_plan_round_uses_fifteen_percent_of_the_budget(self):
        state = make_state(self.tasks, self.names, {}, {}, 'panel')
        plan, _changes = plan_round(state, self.tasks, self.names, VALUES,
                                    {task_id: Path('/tmp/panel.json') for task_id in self.tasks},
                                    budget=200, per_group=10)
        random_games = sum(item['games'] for item in plan if item['base'].startswith('rd137_'))
        self.assertLessEqual(random_games, int(200 * 0.15))
        self.assertEqual({item['multiplier'] for item in plan if item['base'].startswith('rd137_')},
                         {0.5})

    def test_data_levers_quota_and_external_cells_use_existing_planner(self):
        import make_random_deck_cells as generator
        registry = json.loads(generator.REGISTRY.read_text())
        external = dict(task_id='rd159_l45_train_cad0fa53dae0', level=45, terrain='roof',
                        deck=[1, 2, 3, 4, 6, 20, 29, 30, 32, 33], source='task148')
        panels = generator.build_panels(registry, [external])
        tasks = {task['task_id']: task for task in panels['training_panel']['tasks']}
        def planned(**kwargs):
            state = make_state(tasks, self.names, {}, {}, 'panel')
            state['round'] = 4
            for base, cell_state in state['cells'].items():
                cell_state['last_round'] = 0 if base == external['task_id'] else 1
            return plan_round(state, tasks, self.names, VALUES,
                              {task_id: Path('/tmp/panel.json') for task_id in tasks},
                              budget=768, per_group=48, **kwargs)[0]
        baseline = planned()
        self.assertEqual(baseline, planned(random_deck_share=.15))
        for plan, count in ((baseline, 4), (planned(random_deck_share=.25), 8)):
            self.assertEqual(len(plan), count)
            self.assertEqual({item['games'] for item in plan}, {24})
            self.assertEqual({item['multiplier'] for item in plan}, {.5})
            self.assertIn(external['task_id'], {item['base'] for item in plan})
        self.assertEqual(planned(random_deck_share=0), [])

