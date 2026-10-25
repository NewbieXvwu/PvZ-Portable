"""最小示例：打一关，打印局面，导出存档。

    python3 examples/play.py demo_roof_49_doom_m200 [seed]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llmplay import Game  # noqa: E402


def main() -> int:
    task_id = sys.argv[1] if len(sys.argv) > 1 else "demo_roof_49_doom_m200"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
    out = Path("/mnt/data/out") / ("%s_seed%s" % (task_id, seed if seed is not None else "default"))
    with Game(task_id, seed=seed) as game:
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
