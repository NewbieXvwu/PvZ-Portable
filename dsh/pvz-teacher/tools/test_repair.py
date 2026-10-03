#!/usr/bin/env python3
"""`logging_proxy.py` 里 `_repair` 的回归测试。

为什么单独测：这个修复的存在理由是一个**已经确认过的失败**
（严格网关拒收 DSH 发的 `thinking: {type:"enabled"}`），
而端到端复现需要那个网关还活着。逻辑本身可以离线测，就离线测，
免得下次端点不可用时连"修复到底对不对"都验不了。

用法
----
    python3 test_repair.py          # 退出码 0 = 全过
    python3 -m unittest test_repair # 也可以走 unittest
"""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("logging_proxy", _HERE / "logging_proxy.py")
assert _spec and _spec.loader
lp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lp)


def repair(fix_budget: int, body: dict) -> tuple[dict, str | None]:
    lp.FIX_THINKING_BUDGET = fix_budget
    out, note = lp._repair(json.dumps(body).encode("utf-8"))
    return json.loads(out), note


class TestThinkingBudgetRepair(unittest.TestCase):
    def test_disabled_when_fix_is_zero(self):
        """默认关闭时一个字节都不改 —— 代理首先是观察工具，其次才是修复器。"""
        out, note = repair(0, {"thinking": {"type": "enabled"}, "max_tokens": 65536})
        self.assertEqual(out["thinking"], {"type": "enabled"})
        self.assertIsNone(note)

    def test_adds_missing_budget(self):
        out, note = repair(32768, {"thinking": {"type": "enabled"}, "max_tokens": 65536})
        self.assertEqual(out["thinking"]["budget_tokens"], 32768)
        self.assertEqual(out["thinking"]["type"], "enabled")
        self.assertEqual(note, "thinking.budget_tokens=32768")

    def test_clamped_by_max_tokens(self):
        """预算必须给输出留位置：ceiling = max_tokens - 1024。"""
        out, _ = repair(32768, {"thinking": {"type": "enabled"}, "max_tokens": 8192})
        self.assertEqual(out["thinking"]["budget_tokens"], 7168)

    def test_never_below_1024(self):
        """网关的下界是 1024，钳制不能把它压到下面去。"""
        out, _ = repair(32768, {"thinking": {"type": "enabled"}, "max_tokens": 2000})
        self.assertEqual(out["thinking"]["budget_tokens"], 1024)

    def test_existing_budget_is_preserved(self):
        out, note = repair(32768, {"thinking": {"type": "enabled", "budget_tokens": 4096}})
        self.assertEqual(out["thinking"]["budget_tokens"], 4096)
        self.assertIsNone(note)

    def test_disabled_thinking_untouched(self):
        out, note = repair(32768, {"thinking": {"type": "disabled"}, "max_tokens": 65536})
        self.assertEqual(out["thinking"], {"type": "disabled"})
        self.assertIsNone(note)

    def test_no_thinking_field_untouched(self):
        out, note = repair(32768, {"max_tokens": 65536})
        self.assertNotIn("thinking", out)
        self.assertIsNone(note)

    def test_non_json_body_passes_through(self):
        lp.FIX_THINKING_BUDGET = 32768
        raw = b"not json at all"
        out, note = lp._repair(raw)
        self.assertEqual(out, raw)
        self.assertIsNone(note)

    def test_known_edge_max_tokens_at_boundary(self):
        """已知边界：max_tokens == 1024 时两条约束无法同时满足。

        Anthropic 要求 budget_tokens < max_tokens，而网关要求 budget >= 1024，
        所以 max_tokens <= 1024 的请求**无论怎么改都过不去**。
        这里钉住"会退化成 1024、然后被网关拒"这个行为，
        免得以后有人以为这是修好了。
        """
        out, _ = repair(32768, {"thinking": {"type": "enabled"}, "max_tokens": 1024})
        self.assertEqual(out["thinking"]["budget_tokens"], 1024)


if __name__ == "__main__":
    unittest.main(verbosity=2)
