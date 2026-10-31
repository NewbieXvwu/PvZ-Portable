"""最小示例：按 next_task() 打一局，打印局面，导出存档。

    python3 examples/play.py --shard 0 --shards 3 --deck 0,1,2,3,4,6 --reason '经济…主力…防守…应急…地形…'

使用 next_task() 返回的关卡与种子，卡组与思路由调用者指定。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llmplay import Game, next_task, out_dir, roster  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--shards", type=int)
    parser.add_argument("--deck", help="按当前地形和配额自行设计的植物编号列表，逗号分隔")
    parser.add_argument("--reason", default="", help="经济/主力/防守/应急/地形专用卡的核心思路")
    args = parser.parse_args()
    todo = next_task(shard=args.shard, shards=args.shards)
    print("当前该做：%s" % json.dumps(todo, ensure_ascii=False))
    item = todo["item"]
    task_id, seed = item["task_id"], item["seed"]
    if item["layer"] == "decks" and item["action"] != "resume":
        print("选卡前出怪表：%s" % json.dumps(roster(task_id, seed=seed), ensure_ascii=False))
    if item["layer"] == "decks" and item["action"] != "resume" and (not args.deck or not args.reason):
        parser.error("扩卡组主线须提供--deck与--reason；传送带支线由引擎发牌")
    deck = [] if item["layer"] == "native" else [int(card) for card in (args.deck or "").split(",") if card]
    game = (Game.load_archive(out_dir() / item["archive"]) if item["action"] == "resume" else
            Game(task_id, seed=seed, deck=deck, reason=args.reason))
    with game:
        out = out_dir() / ("%s_seed%s" % (task_id, game.seed))
        print(game.observe())
        print(json.dumps(game.legal(), ensure_ascii=False)[:2000])
        # 这里换成你自己的策略：读局面 -> 选动作 -> 推进 -> 再读。
        game.wait(600)
        print(json.dumps(game.summary(), ensure_ascii=False))
        game.save(out.with_suffix(".json"))
        game.deck_review = "示例局：只跑 600 tick"
        game.export_archive(out)
        print("存档：%s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
