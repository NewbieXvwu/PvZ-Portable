"""task29 第 2 步：课程控制器的元规则单测（不跑引擎、不读真实清单）。"""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]

from curriculum_queue import (DEMOTE_BELOW, FRONTIER_HIGH, FRONTIER_LOW, GIB, MEMORY_RESERVE_BYTES,
                              MIN_SAMPLES, PROMOTE_ABOVE, WINDOW, _descendant_rss, adjust, allocate,
                              command_for, label_for, parse_task_id, plan_round, rate,
                              startup_worker_estimate, worker_count)

VALUES = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
NAMES = {'m125': 1.25, 'm150': 1.5, 'm175': 1.75, 'm200': 2.0, 'm250': 2.5, 'm300': 3.0}


def cell(wins, games, mult_index=0, last_round=0):
    return dict(family='f', level=7, mult_index=mult_index, last_round=last_round,
                windows={str(VALUES[mult_index]): {'recent': [1] * wins + [0] * (games - wins),
                                                   'wins': wins, 'games': games}})


class CurriculumRulesTest(unittest.TestCase):
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

    def test_allocate_guard_share_and_budget(self):
        # 全部格已经升到 1.25，这样第二阶段的“当前倍率”不会是 1.0，能单独看防遗忘那 12%。
        cells = {f'c{i}': cell(6, 10, mult_index=1, last_round=i) for i in range(10)}
        picks = allocate(cells, VALUES, budget=100, per_group=10, guard_ratio=0.12, round_index=1)
        self.assertLessEqual(sum(p[2] for p in picks), 100)
        at_one = [p for p in picks if p[1] == 1.0]
        self.assertEqual(len(at_one), 1)                       # 12% -> 至少 1 组，但只有 1 组
        self.assertEqual(at_one[0][0], 'c0')                   # 最久没采的优先
        self.assertEqual(len([p for p in picks if p[1] == 1.25]), 9)
        self.assertGreater(len(picks), 1)
        for base, multiplier, games in picks:
            self.assertEqual(games, 10)
            self.assertIn(multiplier, VALUES)

    def test_frontier_preferred_over_resolved(self):
        cells = {
            'edge': cell(6, 10),            # 0.6 -> frontier
            'easy': cell(10, 10),           # 1.0 -> 会先升档，当前倍率无数据
            'hard': cell(1, 10),            # 0.1 -> 1.0 的难关照样采
        }
        adjust(cells, VALUES)
        picks = allocate(cells, VALUES, budget=40, per_group=10, guard_ratio=0.0, round_index=1)
        bases = [p[0] for p in picks]
        self.assertIn('edge', bases)
        self.assertIn('hard', bases)

    def test_plan_round_offsets_and_exhaustion(self):
        tasks = {}
        for mult, suffix in [(1.0, ''), (1.25, '_m125')]:
            task_id = 'day_l7_train_d7' + suffix
            tasks[task_id] = dict(task_id=task_id, level=7, deck=[0, 1, 2, 3, 4, 5, 7],
                                  zombie_count_multiplier=mult, seeds=list(range(20)))
        state = dict(round=0, offsets={}, cells={'day_l7_train_d7': cell(5, 10)})
        manifests = {task_id: Path('/tmp/panel.json') for task_id in tasks}
        plan, _ = plan_round(state, tasks, NAMES, VALUES, manifests,
                             budget=10, per_group=10, guard_ratio=1.0)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]['task_id'], 'day_l7_train_d7')
        self.assertEqual(plan[0]['offset'], 0)
        state['offsets']['day_l7_train_d7'] = 15
        with self.assertRaises(SystemExit):
            plan_round(state, tasks, NAMES, VALUES, manifests,
                       budget=10, per_group=10, guard_ratio=1.0)

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

    def test_labels(self):
        self.assertEqual(label_for(3, 'day_l7_train_d7', 1.5), 'c3d7d7m150')
        self.assertEqual(label_for(1, 'night_l12_train_n10', 1.0), 'c1n12n10m100')


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
