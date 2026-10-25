"""会话预算与止损：全部由接口计数、写磁盘，模型无法绕过（task81 §A）。

主控 2026-10-25 17:30 的修订：LLM 没有时间观念，所以"用分钟约束它"等于没约束。
这里把墙钟和搜索预算都变成**接口自己的状态**：

- 会话第一次启动时把开始时间写进 `budget.json`；
- 搜索预算按"模拟的引擎 tick 总数"计（不是墙钟）—— 开场先跑一次标定，
  用实测吞吐把"约 25 分钟的计算量"换算成 tick 数；
- 每用掉 40% 搜索预算记一个检查点（最远存活波）。连续两个检查点没有提升 ≥1 波
  就判定搜索不佳；
- 判定不佳或预算用尽后，搜索类入口（`branch()`）对这一关直接抛异常，
  异常消息就是下一步怎么做；
- 手动段的重写次数也在这里数：同一段第 9 次 rewind 拒绝执行；
- 单关总预算（搜索 + 手动）用尽时，提示保存最远进度并换下一关；
- 墙钟到 2:45 后新开局入口开始拒绝执行。

时间单位说明：引擎 100 tick ≈ 1 秒游戏时间，但**预算按 tick 计**是为了跟机器速度
解耦 —— 同一份预算在快慢机器上是同样的"计算量"。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

WALL_LIMIT_SECONDS = 3 * 60 * 60
STOP_NEW_GAME_SECONDS = 2 * 60 * 45      # 2:45 之后不许开新局
SEARCH_MINUTES = 25                       # 每关先给搜索的时间预算（换算成 tick）
LEVEL_MINUTES = 70                        # 单关总预算（搜索 + 手动）
CHECKPOINT_FRACTION = 0.40                # 每用掉 40% 搜索预算记一次最远波
SEGMENT_REWRITES = 8                      # 同一段允许 rewind 的次数
BRANCH_LIMIT_SEARCH = 8                   # 搜索模式下同时存在的分支持数上限


def _clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return "%d:%02d" % (seconds // 60, seconds % 60)


class Session:
    """一份会话。状态全部落在 `budget.json`，进程重启后接着算。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = None
        self.load()

    # ------------------------------------------------------------ 持久化

    def load(self) -> None:
        if self.path.is_file():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = dict(schema="llmplay_budget_v1", started_at=time.time(),
                         calibrated_rate=None, search_limit_ticks=None,
                         level_limit_ticks=None, levels={}, out_dir=None)
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")

    # ------------------------------------------------------------ 时间

    @property
    def elapsed(self) -> float:
        return time.time() - float(self.data["started_at"])

    @property
    def stop_new_games(self) -> bool:
        return self.elapsed >= STOP_NEW_GAME_SECONDS

    def require_new_game_allowed(self) -> None:
        if self.stop_new_games:
            raise RuntimeError(
                "墙钟已到 2:45：立即停止开新局。现在只做三件事 —— "
                "把现有存档打包成 out.zip、写 report.md、结束。")

    # ------------------------------------------------------------ 标定

    def calibrate(self, ticks: int, seconds: float) -> None:
        """用一次实测把'约 25 分钟的计算量'换算成 tick 预算。"""
        if self.data.get("calibrated_rate") and self.data.get("search_limit_ticks"):
            return
        rate = max(1.0, float(ticks) / max(seconds, 1e-3))
        self.data["calibrated_rate"] = rate
        self.data["search_limit_ticks"] = int(rate * SEARCH_MINUTES * 60)
        self.data["level_limit_ticks"] = int(rate * LEVEL_MINUTES * 60)
        self.save()

    @property
    def search_limit(self) -> int:
        return int(self.data.get("search_limit_ticks") or 4_000_000)

    @property
    def level_limit(self) -> int:
        return int(self.data.get("level_limit_ticks") or 12_000_000)

    # ------------------------------------------------------------ 每关状态

    def level(self, level) -> dict:
        key = str(int(level))
        levels = self.data.setdefault("levels", {})
        if key not in levels:
            levels[key] = dict(level=int(level), mode="search", search_ticks=0,
                               manual_ticks=0, total_ticks=0, max_wave=0,
                               checkpoints=[], search_bad=False, search_closed=False,
                               segment_rewrites=0, segment_start=0,
                               last_progress_wave=0, best_archive=None,
                               branches=0, exhausted=False)
        return levels[key]

    def note_ticks(self, level, ticks: int, mode: str) -> None:
        if ticks <= 0:
            return
        state = self.level(level)
        state["total_ticks"] += int(ticks)
        state["search_ticks" if mode == "search" else "manual_ticks"] += int(ticks)
        if mode == "search":
            self._maybe_checkpoint(state)
        if state["total_ticks"] >= self.level_limit:
            state["exhausted"] = True
        self.save()

    def _maybe_checkpoint(self, state: dict) -> None:
        """用掉 40% / 80% / 100% 搜索预算时各记一次最远波（§A）。"""
        limit = self.search_limit
        used = state["search_ticks"]
        for fraction in (CHECKPOINT_FRACTION, 2 * CHECKPOINT_FRACTION, 1.0):
            if used < fraction * limit:
                continue
            if any(abs(float(c.get("fraction", -1)) - fraction) < 1e-9 for c in state["checkpoints"]):
                continue
            state["checkpoints"].append(dict(fraction=fraction, at_ticks=used,
                                             max_wave=state["max_wave"],
                                             elapsed=round(self.elapsed, 1)))
        history = state["checkpoints"]
        if len(history) >= 3:
            before, prev, last = (h["max_wave"] for h in history[-3:])
            if last < prev + 1 and prev < before + 1:
                state["search_bad"] = True
                state["search_closed"] = True
        if used >= limit:
            state["search_closed"] = True

    def record_wave(self, level, wave: int) -> None:
        state = self.level(level)
        wave = int(wave or 0)
        if wave > state["max_wave"]:
            state["max_wave"] = wave
        if wave > state["last_progress_wave"]:
            state["last_progress_wave"] = wave
            state["segment_rewrites"] = 0
        self.save()

    def record_best_archive(self, level, path) -> None:
        state = self.level(level)
        state["best_archive"] = str(path)
        self.save()

    # ------------------------------------------------------------ 门禁

    def mode(self, level) -> str:
        return self.level(level)["mode"]

    def set_mode(self, level, mode: str) -> None:
        self.level(level)["mode"] = mode
        self.save()

    def search_reason(self, level) -> str | None:
        """搜索为什么被关掉；没关就是 None。"""
        state = self.level(level)
        if state["search_closed"] or state["search_bad"]:
            if state["search_bad"]:
                return "连续两个检查点最远波没有提升"
            return "搜索预算已用尽"
        return None

    def require_search(self, level, what: str = "搜索") -> None:
        reason = self.search_reason(level)
        if reason:
            raise RuntimeError(
                "本关搜索已停止（%s）：%s 不再可用。改用按波手动推进："
                "observe() 读当前局面 → 写一段覆盖接下来 1–3 波的脚本执行 → "
                "看结果 → 不满意就 rewind() 重写这一段。" % (reason, what))

    def note_branch(self, level, delta: int = 1) -> None:
        state = self.level(level)
        state["branches"] = max(0, state["branches"] + delta)
        self.save()

    def require_branch(self, level) -> None:
        self.require_search(level, "branch()")
        state = self.level(level)
        if state["branches"] >= BRANCH_LIMIT_SEARCH:
            raise RuntimeError(
                "同时存在的分支持数已达上限 %d：先把不用的分支关掉（close()），"
                "或者直接在主局上 rewind 试错。" % BRANCH_LIMIT_SEARCH)

    def note_rewind(self, level, target_tick: int, current_tick: int) -> None:
        state = self.level(level)
        if target_tick < state["segment_start"]:
            # 退回更早一段：重写计数重来。
            state["segment_start"] = int(target_tick)
            state["segment_rewrites"] = 0
        else:
            state["segment_rewrites"] += 1
            if state["segment_rewrites"] > SEGMENT_REWRITES:
                raise RuntimeError(
                    "本段已经 rewind %d 次仍没过去：不要再改这一段了，"
                    "退回更早的一段重写（rewind 到一个更早的 tick）。"
                    % SEGMENT_REWRITES)
        self.save()

    def require_not_exhausted(self, level) -> None:
        state = self.level(level)
        if state["exhausted"]:
            raise RuntimeError(
                "本关总预算（搜索 + 手动）已用尽：先 export_archive() 保存最远进度的存档，"
                "换下一关。有余力再回来。")

    # ------------------------------------------------------------ 状态行

    def status_line(self, level=None) -> str:
        left = max(0, WALL_LIMIT_SECONDS - self.elapsed)
        head = "[已用 %s / 3:00" % _clock(self.elapsed)
        if level is None:
            return head + " | 未开局" + self._tail(None) + "]"
        state = self.level(level)
        limit = max(1, self.search_limit)
        percent = int(100 * min(1.0, state["search_ticks"] / limit))
        line = (head + " | 本关 L%s 模式=%s | 搜索预算 %d%% | 最远 %s 波 | 本段重写 %d/%d"
                % (state["level"], "手动" if state["mode"] == "manual" else "搜索",
                   percent, state["max_wave"], state["segment_rewrites"], SEGMENT_REWRITES))
        if self.stop_new_games:
            line += " | **已到 2:45：停止开新局，打包 out.zip 并写 report.md**"
        elif left < 900:
            line += " | 剩 %s" % _clock(left)
        return line + "]"

    def _tail(self, _level) -> str:
        return ""


_SESSION: Session | None = None


def default_path() -> Path:
    """会话文件固定在用户主目录，**不随当前目录变**（§A：模型换个 cwd 不能重置预算）。"""
    override = os.environ.get("PVZ_LLMPLAY_SESSION")
    if override:
        return Path(override)
    try:
        home = Path.home()
    except RuntimeError:            # 没有 HOME 的极端环境：退回当前目录
        home = Path.cwd()
    return home / ".llmplay_session" / "budget.json"


def session() -> Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = Session(default_path())
    return _SESSION


def reset_session(path=None) -> Session:
    """测试用：换一个会话文件。"""
    global _SESSION
    _SESSION = Session(Path(path) if path else default_path())
    return _SESSION
