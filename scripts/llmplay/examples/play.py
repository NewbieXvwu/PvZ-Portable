"""最小示例：打一关，打印局面，导出存档。

    python3 examples/play.py llmplay2_f_l42_m100 [seed]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llmplay import Game, out_dir, load_tasks  # noqa: E402


def main() -> int:
    task_id = sys.argv[1] if len(sys.argv) > 1 else "llmplay2_f_l42_m100"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
    reference = load_tasks()[task_id].get("reference_deck")
    with Game(task_id, seed=seed, deck=reference, reason="先试记分板参考卡组", strategy="参考卡组") as game:
        out = out_dir() / ("%s_seed%s" % (task_id, game.seed))
        print(game.observe())
        print(json.dumps(game.legal(), ensure_ascii=False)[:2000])
        # 这里换成你自己的策略：读局面 -> 选动作 -> 推进 -> 再读。
        game.wait(600)
        print(json.dumps(game.summary(), ensure_ascii=False))
        game.save(out.with_suffix(".json"))
        game.export_archive(out)
        print("存档：%s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
