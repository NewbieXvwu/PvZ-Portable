"""持久化搜索止损：固定模拟 tick 预算、进度检查点与同段八次重写。

task81e：墙钟仅显示已用时间，没有会话或单关总上限。搜索止损后仍可手动推进。
状态按任务族保存，跨脚本、跨续跑延续；旧会话的搜索 tick 额度照常保留。
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:          # Windows 开发机：没有 flock，退化为单进程
    fcntl = None

SEARCH_LIMIT_TICKS = 4_000_000             # 固定模拟量；不再由机器墙钟吞吐决定
CHECKPOINT_FRACTION = 0.40                # 每用掉 40% 搜索预算记一次最远波
SEGMENT_REWRITES = 8                      # 同一段允许 rewind 的次数
BRANCH_LIMIT_SEARCH = 8                   # 搜索模式下同时存在的分支持数上限


def _clock(seconds: float) -> str:
    minutes = max(0, int(seconds)) // 60
    return "%d:%02d" % (minutes // 60, minutes % 60)


class Session:
    """一份会话。状态全部落在 `budget.json`，进程重启后接着算。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = None
        # 分支持数是**进程内**状态：进程退出时分支引擎也没了，不能写进 budget.json，
        # 否则每个脚本都会把上一条脚本的分支算进来，8 次之后永远开不了新分支。
        self.branches = 0
        with self._transaction():
            pass

    # ------------------------------------------------------------ 持久化

    def load(self) -> None:
        if self.path.is_file():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = dict(schema="llmplay_budget_v1", started_at=time.time(),
                         search_limit_ticks=SEARCH_LIMIT_TICKS, levels={}, out_dir=None)
        self.save()

    def save(self) -> None:
        # 原子替换：读的一方不会读到写了一半的 JSON。
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        os.replace(tmp, self.path)

    def refresh(self) -> None:
        """重新读磁盘：3 个子代理是 3 个进程，各自的内存副本会过期。"""
        try:
            if self.path.is_file():
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass

    def claim_legend(self) -> bool:
        with self._transaction():
            first = not self.data.get("legend_shown")
            self.data["legend_shown"] = True
            return first

    @contextmanager
    def _transaction(self):
        """跨进程读改写：flock 保护，避免 3 个子代理互相覆盖预算。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_name(self.path.name + ".lock")
        with open(lock, "a+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                self.refresh()
                if self.data is None:
                    self.load()
                yield
                self.save()
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    # ------------------------------------------------------------ 时间

    @property
    def elapsed(self) -> float:
        return time.time() - float(self.data["started_at"])

    @property
    def stop_new_games(self) -> bool:
        return False

    def require_new_game_allowed(self) -> None:
        # 兼容旧调用方；墙钟只作参考，永远不封锁新局。
        return

    # ------------------------------------------------------------ 标定

    def calibrate(self, ticks: int, seconds: float) -> None:
        """兼容旧脚本：搜索预算现为固定模拟量，参数不影响预算。"""
        with self._transaction():
            if self.data.get("search_limit_ticks"):
                return
            self.data["search_limit_ticks"] = SEARCH_LIMIT_TICKS

    @property
    def search_limit(self) -> int:
        return int(self.data.get("search_limit_ticks") or SEARCH_LIMIT_TICKS)

    # ------------------------------------------------------------ 每关状态

    def level(self, level) -> dict:
        key = str(level)
        levels = self.data.setdefault("levels", {})
        if key not in levels:
            levels[key] = dict(level=level, mode="search", search_ticks=0,
                               manual_ticks=0, total_ticks=0, max_wave=0,
                               checkpoints=[], search_bad=False, search_closed=False,
                               segment_rewrites=0, segment_start=0,
                               last_progress_wave=0, best_archive=None,
                               exhausted=False)
        return levels[key]

    def register_task(self, task_id: str, level: int) -> None:
        with self._transaction():
            self.level(task_id)["level"] = int(level)

    def snapshot(self) -> dict:
        with self._transaction():
            return json.loads(json.dumps(self.data))

    def import_budget(self, document: dict) -> None:
        """合并续跑计数，同一份交付重复导入不会重复计数或缩减既有预算。"""
        with self._transaction():
            self.data["started_at"] = min(self.data["started_at"],
                                          float(document.get("started_at", self.data["started_at"])))
            if document.get("search_limit_ticks"):
                self.data["search_limit_ticks"] = document["search_limit_ticks"]
            for key, incoming in document.get("levels", {}).items():
                state = self.level(key)
                for field in ("search_ticks", "manual_ticks", "total_ticks", "max_wave",
                              "last_progress_wave", "segment_rewrites"):
                    state[field] = max(state[field], incoming.get(field, 0))
                for field in ("search_bad", "search_closed"):
                    state[field] = bool(state[field] or incoming.get(field))
                state["level"] = incoming.get("level", state["level"])
                state["segment_start"] = incoming.get("segment_start", state["segment_start"])
                state["checkpoints"] = max((state["checkpoints"], incoming.get("checkpoints", [])), key=len)
                if state["search_closed"] or incoming.get("mode") == "manual":
                    state["mode"] = "manual"

    def note_ticks(self, level, ticks: int, mode: str) -> None:
        if ticks <= 0:
            return
        with self._transaction():
            state = self.level(level)
            state["total_ticks"] += int(ticks)
            state["search_ticks" if mode == "search" else "manual_ticks"] += int(ticks)
            if mode == "search":
                self._maybe_checkpoint(state)

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
        if len(history) >= 2:
            before = history[-3]["max_wave"] if len(history) >= 3 else 0
            prev, last = (h["max_wave"] for h in history[-2:])
            if last < prev + 1 and prev < before + 1:
                state["search_bad"] = True
                state["search_closed"] = True
        if used >= limit:
            state["search_closed"] = True
        if state["search_closed"]:
            state["mode"] = "manual"

    def record_wave(self, level, wave: int) -> None:
        with self._transaction():
            state = self.level(level)
            wave = int(wave or 0)
            if wave > state["max_wave"]:
                state["max_wave"] = wave
            if wave > state["last_progress_wave"]:
                state["last_progress_wave"] = wave
                state["segment_rewrites"] = 0

    def record_best_archive(self, level, path) -> None:
        with self._transaction():
            state = self.level(level)
            state["best_archive"] = str(path)

    def save_best_log(self, level, log: dict, *, wave: int, won: bool = False) -> Path:
        """按最远波/同波 tick 保留操作日志，供别的脚本或分支取回。"""
        with self._transaction():
            state = self.level(level)
            score = [int(won), int(wave), int(log["tick"])]
            previous = state.get("best_log")
            if previous and Path(previous).is_file() and score <= state.get("best_score", []):
                return Path(previous)
            path = self.path.parent / "best" / ("%s.json" % level)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(log, ensure_ascii=False, separators=(",", ":")) + "\n",
                           encoding="utf-8")
            os.replace(tmp, path)
            state.update(best_log=str(path), best_score=score)
            return path

    # ------------------------------------------------------------ 门禁

    def mode(self, level) -> str:
        self.refresh()
        return self.level(level)["mode"]

    def set_mode(self, level, mode: str) -> None:
        with self._transaction():
            self.level(level)["mode"] = mode

    def search_reason(self, level) -> str | None:
        """搜索为什么被关掉；没关就是 None。"""
        self.refresh()
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
        self.branches = max(0, self.branches + int(delta))

    def require_branch(self, level) -> None:
        self.require_search(level, "branch()")
        if self.branches >= BRANCH_LIMIT_SEARCH:
            raise RuntimeError(
                "同时存在的分支持数已达上限 %d：先把不用的分支关掉（close()），"
                "或者直接在主局上 rewind 试错。" % BRANCH_LIMIT_SEARCH)

    def note_rewind(self, level, target_tick: int, current_tick: int) -> None:
        with self._transaction():
            state = self.level(level)
            if not state["segment_rewrites"] or target_tick < state["segment_start"]:
                # 退回更早一段：重写计数重来。
                state["segment_start"] = int(target_tick)
                state["segment_rewrites"] = 1
            else:
                state["segment_rewrites"] += 1
                if state["segment_rewrites"] > SEGMENT_REWRITES:
                    raise RuntimeError(
                        "本段已经 rewind %d 次仍没过去：不要再改这一段了，"
                        "退回更早的一段重写（rewind 到一个更早的 tick）。"
                        % SEGMENT_REWRITES)

    # ------------------------------------------------------------ 状态行

    def status_line(self, level=None) -> str:
        self.refresh()
        head = "[已用 %s" % _clock(self.elapsed)
        if level is None:
            return head + " | 未开局" + self._tail(None) + "]"
        state = self.level(level)
        limit = max(1, self.search_limit)
        percent = int(100 * min(1.0, state["search_ticks"] / limit))
        line = (head + " | 本关 L%s 模式=%s | 搜索预算 %d%% | 最远 %s 波 | 本段重写 %d/%d"
                % (state["level"], "手动" if state["mode"] == "manual" else "搜索",
                   percent, state["max_wave"], state["segment_rewrites"], SEGMENT_REWRITES))
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
