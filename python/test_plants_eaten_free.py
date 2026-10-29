"""task65: plants_eaten_free 差分在快照恢复与分支推演下正确。

需要本地 build/pvz-portable 与游戏资源（PVZ_RESOURCE_DIR），缺了就跳过——
与 test_cpp_rollout 同一约定。零费植物用 pvz_constants.plant_defs() 的游戏费用
列现解析（费用为 0 的与 >0 的各取一种），测试自己不写死植物编号。
"""

from __future__ import annotations

import os
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pvz_constants import plant_defs
from pvz_event_env import EventWaitEnv
from pvz_env import TaskSpec

RESOURCE = Path(os.environ.get("PVZ_RESOURCE_DIR", str(Path.home() / ".cache/pvz-research-resources")))
BINARY = Path(os.environ.get("PVZ_EXECUTABLE", ROOT / "build/pvz-portable"))


def _pair() -> tuple[int, int]:
    """(零费植物, 付费植物) 的 seed id——按游戏自己的费用表选。"""
    free = next(d for d in plant_defs() if d["cost"] == 0)
    paid = next(d for d in plant_defs() if d["seed"] == "PEASHOOTER" and d["cost"] > 0)
    return free["type"], paid["type"]


FREE, PAID = _pair()
EVENT_KEYS = ("zombies_killed", "plants_eaten", "plants_eaten_free",
              "mower_triggered", "waves_started")


@unittest.skipUnless(BINARY.is_file() and (RESOURCE / "main.pak").is_file(),
                     "requires simulator and game resources")
class PlantsEatenFreeTest(unittest.TestCase):
    def setUp(self):
        self.env = EventWaitEnv(RESOURCE, executable=BINARY)
        self.addCleanup(self.env.close)
        # 同一行放三株：付费的在前（col 8），零费垫材跟在后面（col 7/6），
        # 僵尸一路吃过去，两列计数都会动。
        self.env.reset(task=TaskSpec(level=7, seed=30000, zombie_count_multiplier=3.0,
                                     preplanted=((PAID, 2, 8), (FREE, 2, 7), (FREE, 2, 6))))

    def _run_until_eaten(self, max_steps: int = 60) -> Counter:
        totals: Counter = Counter()
        for _ in range(max_steps):
            _obs, _reward, terminal, truncated, info = self.env.step({"type": "wait", "ticks": 400})
            events = info["events"]
            self.assertLessEqual(events.get("plants_eaten_free", 0), events.get("plants_eaten", 0))
            totals.update(events)
            if terminal or truncated:
                break
            if totals["plants_eaten"] >= 2 and totals["plants_eaten_free"] >= 1:
                break
        return totals

    def test_free_column_tracks_paid_and_free_eats(self):
        totals = self._run_until_eaten()
        self.assertGreaterEqual(totals["plants_eaten"], 2, f"植物没被吃到位：{dict(totals)}")
        self.assertGreaterEqual(totals["plants_eaten_free"], 1, "零费植物没有被吃到")
        self.assertLess(totals["plants_eaten_free"], totals["plants_eaten"],
                        "付费植物也应被吃，两列不该相等")

    def test_branch_diffs_match_straight_line_from_same_parent(self):
        totals = self._run_until_eaten()
        self.assertGreaterEqual(totals["plants_eaten"], 1, f"植物没被吃到位：{dict(totals)}")
        parent = self.env.snapshot_fast()
        self.addCleanup(self.env.release_snapshot, parent)
        action = {"type": "wait", "ticks": 400}
        (branch,) = self.env.branch_snapshot(parent, [action], retain_snapshots=False)
        branch_events = branch["events"]
        for key in EVENT_KEYS:
            self.assertIn(key, branch_events)
        self.env.restore_fast(parent)
        _obs, _reward, _terminal, _truncated, info = self.env.step(dict(action))
        for key in EVENT_KEYS:
            self.assertEqual(branch_events[key], info["events"][key],
                             f"分支推演与直放不一致：{key}")


if __name__ == "__main__":
    unittest.main()
