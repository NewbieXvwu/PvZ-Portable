"""最小示例：按 next_task() 打一局，打印局面，导出存档。

    python3 examples/play.py [task_id] [seed]

不给参数就用 next_task() 返回的那一项（当前圈当前层必须做的那一项）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llmplay import Game, next_task, out_dir, load_tasks  # noqa: E402


def main() -> int:
    todo = next_task()
    print("当前该做：%s" % json.dumps(todo, ensure_ascii=False))
    item = todo["item"]
    task_id = sys.argv[1] if len(sys.argv) > 1 else item["task_id"]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else item.get("seed")
    task = load_tasks()[task_id]
    deck = item.get("deck") or task.get("reference_deck") or []
    if len(deck) < 6:                       # C 层的指定卡组由任务自己带
        deck = task.get("deck") or []
    with Game(task_id, seed=seed, deck=deck, reason="先试参考卡组", strategy="参考卡组") as game:
        out = out_dir() / ("%s_seed%s" % (task_id, game.seed))
        print(game.observe())
        print(json.dumps(game.legal(), ensure_ascii=False)[:2000])
        # 这里换成你自己的策略：读局面 -> 选动作 -> 推进 -> 再读。
        game.wait(600)
        print(json.dumps(game.summary(), ensure_ascii=False))
        game.save(out.with_suffix(".json"))
        game.deck_review = "示例局：只跑 600 tick，不作评价"   # A/D 层导出前必填
        game.export_archive(out)
        print("存档：%s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
