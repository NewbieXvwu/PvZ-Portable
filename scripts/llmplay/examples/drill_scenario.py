"""task237 最小出题示例：真实存档某决策 → 注入跳跳 → 设阳光 → 交给玩家解。

这个文件只演示「摆题」这一步，不内置战术。把 ``player(game)`` 换成你的闭环打法
（只读 ``game.obs``，用 ``game.plant / game.wait / ...`` 推进），然后在 horizon 处用
task234 的局部判据验收：注入的威胁从场上消失、没有丢小推车、房子没破。

    python3 scripts/llmplay/examples/drill_scenario.py \
        --archive /path/to/archive --decision 812 \
        --row 2 --x 300 --sun 500 --horizon 900

存档可以是 human_demo 的 raw/导出存档，也可以是 llmplay 的 archives 成员；``--decision``
沿用 human_demo 的约定：1 = 存档初始状态，N = 第 N 个操作执行之前。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "python"), str(ROOT / "scripts" / "llmplay")]

from llmplay.game import Game  # noqa: E402

POGO = 18


def _zombies(observation):
    """精简/完整观测都吃：返回 {id: (type, row, x, on_board)}。"""
    out = {}
    for zombie in observation.get("zombies") or []:
        if isinstance(zombie, dict):
            out[int(zombie["id"])] = (int(zombie["type"]), int(zombie["row"]),
                                      float(zombie["x"]), bool(zombie.get("on_board")))
        else:
            out[int(zombie[9])] = (int(zombie[1]), int(zombie[0]),
                                   float(zombie[2]), bool(zombie[5]))
    return out


def _ready_mowers(observation):
    return sum(1 for defense in observation.get("defenses") or []
               if (defense["state"] if isinstance(defense, dict) else defense[1]) == 1)


def player(game: Game) -> None:
    """换成你的打法：每一步只看 game.obs，用 game.plant/game.wait 推进。"""
    # 示例只推进一步，不代表这是一套能解题的打法；你的闭环打法可以在这里循环，
    # 外层随后会把它推进到 horizon。
    game.wait(60)


def judge(game: Game, threat_ids: set[int], mowers_before: int) -> dict:
    """task234 的局部判据：根威胁消失/离场 + 小推车与房子不变。"""
    alive = {zid for zid, (_type, _row, _x, on_board) in _zombies(game.obs).items() if on_board}
    left = sorted(threat_ids & alive)
    mowers_after = _ready_mowers(game.obs)
    return dict(success=not left and mowers_after >= mowers_before and int(game.obs.get("result") or 0) != 2,
                threats_left=left, mowers_before=mowers_before, mowers_after=mowers_after,
                tick=game.tick, result=int(game.obs.get("result") or 0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--decision", type=int, default=1,
                        help="1 = 存档初始状态，N = 第 N 个操作执行之前")
    parser.add_argument("--zombie-type", type=int, default=POGO)
    parser.add_argument("--row", type=int, default=2)
    parser.add_argument("--x", type=float, default=300.0)
    parser.add_argument("--sun", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=900, help="从注入点起最多推进多少 tick")
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--resource-dir", type=Path)
    args = parser.parse_args()

    kwargs = {}
    if args.executable:
        kwargs["executable"] = args.executable
    if args.resource_dir:
        kwargs["resource_dir"] = args.resource_dir
    game = Game.from_archive(args.archive, decision=args.decision, **kwargs)
    try:
        game.observe()                         # 触发重放，拿到完整局面
        mowers_before = _ready_mowers(game.obs)
        if not game.spawn(args.zombie_type, args.row, args.x):
            raise SystemExit("SPAWN 被引擎拒绝：检查地形（例如屋顶不能放海豚）和行列范围")
        threat_ids = {zid for zid, (kind, _row, _x, on_board) in _zombies(game.obs).items()
                      if kind == args.zombie_type and on_board}
        if not game.set_sun(args.sun):
            raise SystemExit("SET_SUN 被引擎拒绝：阳光必须在 0..9990")
        root_tick = game.tick
        player(game)
        while not game.terminal and game.tick - root_tick < args.horizon:
            game.wait(60)
        report = judge(game, threat_ids, mowers_before)
        report.update(root_tick=root_tick, horizon=args.horizon, sun=args.sun,
                      injected=dict(type=args.zombie_type, row=args.row, x=args.x))
        print(json.dumps(report, ensure_ascii=False))
        return 0 if report["success"] else 1
    finally:
        game.close()


if __name__ == "__main__":
    raise SystemExit(main())
