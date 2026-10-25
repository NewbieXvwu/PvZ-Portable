"""LLM 打全程的沙箱接口：一局 = (任务, 操作序列)。

设计立场（task81）
------------------
1. **磁盘上的操作日志是唯一真相。** 工具调用之间进程可能被回收，所以每局
   就是 (任务, 操作序列)；读档到 tick N = reset 后重放 N 之前的所有操作。
   进程内额外留一个快照做加速，但**不依赖**它。
2. **一次调用多步。** 模型写脚本一次跑几十到几百步，最后只打印摘要 ——
   不是为了让人读，是为了让模型在一屏里做决定。
3. **不在这里放任何打法规则。** 合法动作由引擎自己给的 legal_actions
   (Board::CanPlantAt) 决定；这个文件只搬运。
4. 动作时间轴完全自由：任意 tick 都能执行合法动作，不限于任何决策点。

时间轴单位是 tick（引擎的 mMainCounter，100 tick 约 1 秒）。
"""
from __future__ import annotations

import atexit
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path


# 包内布局与仓库一致：<root>/python/pvz_env.py、<root>/scripts/render_episode.py。
# 这样这两个文件可以**原样**打包，不用改它们的 ROOT 推导。
def _find_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "python" / "pvz_env.py").is_file() and (path / "scripts" / "render_episode.py").is_file():
            return path
    raise RuntimeError("找不到引擎包根目录（需要 python/pvz_env.py 与 scripts/render_episode.py）")


ROOT = _find_root(Path(__file__).resolve().parent)
for _p in (ROOT / "python", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pvz_env import PvZEnv, TaskSpec  # noqa: E402
from pvz_tasks import profile_for_deck  # noqa: E402
from render_episode import _frame  # noqa: E402

from . import budget  # noqa: E402

# 还活着、且自上次 observe 之后又推进过的局：进程退出时自动打印一次完整 observe（§D）。
# 模型不用记得自己调用 —— 脚本怎么结束都会看到局面。
# 只打印**最后推进过**的那一局：搜索脚本可能开出几十个 branch，全打印会把输出淹掉。
_LIVE_GAMES: list = []
_LAST_ACTIVE = None


def _mark_active(game) -> None:
    global _LAST_ACTIVE
    _LAST_ACTIVE = game


def _auto_observe() -> None:
    live = [game for game in list(_LIVE_GAMES)
            if getattr(game, "_dirty", False) and game.env is not None]
    if not live:
        return
    primary = _LAST_ACTIVE if _LAST_ACTIVE in live else live[-1]
    try:
        print(primary.observe())
    except Exception as error:      # 自动打印失败不能吞掉脚本的真实结果
        print("[自动 observe 失败：%s]" % error)
    others = [game for game in live if game is not primary]
    if others:
        print("（本次脚本还推进了 %d 个其它局：%s；要看哪个就 print(那个局.observe())）"
              % (len(others), "、".join("%s@tick%s" % (game.task_id, game.tick)
                                       for game in others)))


atexit.register(_auto_observe)

SCHEMA = "llmplay_log_v1"
ARCHIVE_SCHEMA = "episode_archive_v1"
LABEL_SOURCE = "llm_gpt_pro"
FRAME_INTERVAL = 60          # 与 human_demo 的帧节奏一致（每 60 tick 一帧）
DEFAULT_MANIFEST = ROOT / "configs" / "human_demo_tasks.json"
ROBUSTNESS_CONFIG = ROOT / "configs" / "llmplay_robustness.json"
ROBUSTNESS_SCHEMA = "llmplay_robustness_v1"


def _default_resource_dir() -> Path:
    for candidate in (ROOT / "res", Path.home() / ".cache" / "pvz-research-resources"):
        if (candidate / "main.pak").is_file():
            return candidate
    return ROOT / "res"


def _default_executable() -> Path:
    return ROOT / "build" / ("pvz-portable.exe" if os.name == "nt" else "pvz-portable")


def _ensure_executable(path: Path) -> None:
    """解压工具不一定保留可执行位（Python 的 zipfile 就不保留），跑引擎前补上。

    沙箱里很常见：包用 zipfile 解开，selfcheck 之外的自写脚本直接 Popen 会报
    PermissionError。这里补一次，任何入口都不会因为这个卡住。
    """
    try:
        if path.is_file() and not os.access(path, os.X_OK):
            path.chmod(path.stat().st_mode | 0o111)
    except OSError:
        pass


def load_tasks(manifest: str | Path | None = None) -> dict:
    """任务清单 -> {task_id: 定义}。清单就是仓库的 configs/human_demo_tasks.json。"""
    path = Path(manifest) if manifest else DEFAULT_MANIFEST
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {task["task_id"]: task for task in doc.get("tasks") or []}


def task_spec(task: dict, seed=None):
    """任务定义 -> (TaskSpec, 卡组, 种子)。与 human_demo 用同一套推导。"""
    seeds = task.get("seeds") or []
    actual_seed = int(seed if seed is not None else (seeds[0] if seeds else 0))
    deck = [int(card) for card in task["deck"]]
    extra = {"wave_cap": task.get("wave_cap"),
             "zombie_count_multiplier": task.get("zombie_count_multiplier")}
    preplanted = tuple(tuple(int(v) for v in p) for p in (task.get("preplanted") or ()))
    if preplanted:
        extra["preplanted"] = preplanted
    if task.get("conveyor_start"):
        extra["conveyor_start"] = task["conveyor_start"]
    if task.get("fog_observation"):
        extra["fog_observation"] = task["fog_observation"]
    spec = TaskSpec(level=int(task["level"]), seed=actual_seed, playthrough=2,
                    profile=profile_for_deck(tuple(deck), bool(task.get("first_aid_owned", False))),
                    **{key: value for key, value in extra.items() if value is not None})
    return spec, deck, actual_seed




def robustness_seeds(path: str | Path | None = None) -> dict:
    """稳健性检查点 -> 新未来种子（包里 `configs/llmplay_robustness.json`）。

    这块种子与任务种子分开（task81d）：任务是「已知未来」的局，稳健性是「换一个同样
    合理的未来还能不能赢」。种子值写死在配置里，模型改不了，也不能拿任务种子代替。
    """
    document = json.loads(Path(path or ROBUSTNESS_CONFIG).read_text(encoding="utf-8"))
    if document.get("schema") != ROBUSTNESS_SCHEMA:
        raise ValueError("不是 %s 配置：%s" % (ROBUSTNESS_SCHEMA, path or ROBUSTNESS_CONFIG))
    return {int(entry["checkpoint"]): [int(seed) for seed in entry["seeds"]]
            for entry in document.get("seeds") or []}


def _truncate_ops(ops, tick: int) -> list:
    """把操作日志截到 tick（跨越目标 tick 的 wait 切成两段），语义与 _replay_to 一致。"""
    kept = []
    for op in ops:
        if op["end_tick"] <= tick:
            kept.append(dict(op))
            continue
        if op["action"].get("type") == "wait" and op["tick"] < tick:
            kept.append(dict(op, action={"type": "wait", "ticks": tick - op["tick"]}, end_tick=tick))
        break
    return kept


def _policy_source(policy) -> str:
    """打法函数的源码（拿不到源码时退回限定名）。稳健性 meta 里只记哈希。"""
    try:
        return inspect.getsource(policy)
    except (OSError, TypeError):
        return "%s.%s" % (getattr(policy, "__module__", "?"), getattr(policy, "__qualname__", repr(policy)))


def _drive_policy(game, policy, *, limit_tick: int, max_calls: int = 100000) -> None:
    """让闭环打法函数一步接一步地推进，直到终局或到 tick 上限。

    「闭环」是硬要求：每一步只能看当前局面决定动作。函数必须通过 game.plant /
    game.shovel / game.wait 之类的接口推进局面；一次调用什么都不做就直接报错，
    否则外层循环会空转。
    """
    calls = 0
    while not game.terminal and game.tick < limit_tick:
        before = (game.tick, len(game.ops))
        policy(game)
        calls += 1
        if game.terminal:
            return
        if (game.tick, len(game.ops)) == before:
            raise RuntimeError(
                "闭环打法函数第 %d 次调用后局面没有推进（tick 与操作数都没变）。"
                "每次调用至少要 plant / shovel / wait 一次。" % calls)
        if calls >= max_calls:
            raise RuntimeError("闭环打法函数调用次数超过 %d，疑似死循环。" % max_calls)


class Game:
    """一局。构造时只读任务，第一次用引擎时才真正 reset + 重放。"""

    def __init__(self, task_id: str, *, seed=None, manifest=None, executable=None,
                 resource_dir=None, ops=None, frames=None, tick=None, note: str = "",
                 robustness=None, manual_patch: bool = False, _internal: bool = False):
        tasks = load_tasks(manifest)
        if task_id not in tasks:
            raise ValueError("没有任务 %r。可用：%s" % (task_id, sorted(tasks)))
        self.task_id = task_id
        self.task_definition = tasks[task_id]
        self.spec, self.deck, self.seed = task_spec(self.task_definition, seed)
        self.manifest = Path(manifest) if manifest else DEFAULT_MANIFEST
        self.executable = Path(executable) if executable else _default_executable()
        self.resource_dir = Path(resource_dir) if resource_dir else _default_resource_dir()
        self.note = note
        self.ops = list(ops or [])
        self.frames = list(frames or [])
        self.tick = int(tick if tick is not None else (self.ops[-1]["end_tick"] if self.ops else 0))
        self.env = None
        self.obs = None
        self.initial_obs = None
        self.initial_hash = None
        self.terminal = False
        self.totals = {}
        self.max_wave = 0
        self.saw_mowers = False
        self._cache = None
        # §B.6 / §B.7：事件流与逐波历史。observe() 打印后清空事件，timeline() 用历史。
        self.events = []
        self.wave_history = []
        self._plants_snapshot = {}
        self._zombies_snapshot = {}
        self._mower_snapshot = {}
        self._wave_stats = dict(kills=0, lost=0, mowers=0)
        self._observed_at = None
        self._dirty = False
        self._is_branch = False
        # task81d：稳健性结果与「是否用 rewind 手工补救过」。导出存档时写进 meta。
        # 属性不能叫 robustness —— 那是方法名，会被这个赋值遮住。
        self.robustness_report = robustness
        self.manual_patch = bool(manual_patch)
        # RESEED 过的局不能再按操作日志重放（未来已经换过），所以内部试打局带这个标记。
        self._reseeded = None
        # 内部试打局（robustness 的 k 个新未来）不写「最远进度」、不进自动 observe。
        self._internal = bool(_internal)
        self.session = budget.session()
        if not self._internal and not self.ops and not tick:
            # 真正的新开局才受 2:45 门禁限制；读档/重放不受影响（§A）。
            self.session.require_new_game_allowed()
        if not self._internal:
            _LIVE_GAMES.append(self)

    # ------------------------------------------------------------ 引擎生命周期

    def _calibrate_once(self) -> None:
        """会话第一次启动时实测吞吐，把"约 25 分钟计算量"换算成 tick 预算（§A）。

        标定在一个独立的小局上跑（300 tick），不占本关的预算。
        """
        if self.session.data.get("search_limit_ticks"):
            return
        ticks = 300
        probe = None
        try:
            probe = PvZEnv(resource_dir=self.resource_dir, executable=self.executable,
                           headless=True)
            probe.reset(deck=self.deck, task=self.spec)
            # 只计推进的墙钟，不含引擎启动/资源加载 —— 那部分不随 tick 数增长。
            started = time.time()
            probe.step({"type": "wait", "ticks": ticks})
            self.session.calibrate(ticks, max(time.time() - started, 1e-3))
        except Exception:
            self.session.calibrate(ticks, 0.3)      # 兜底：按 1000 tick/s 估
        finally:
            if probe is not None:
                probe.close()

    def _start(self) -> None:
        if self.env is not None:
            return
        _ensure_executable(self.executable)
        self._calibrate_once()
        target = self.tick          # _reset_engine 会把 self.tick 归零，先记住
        self.env = PvZEnv(resource_dir=self.resource_dir, executable=self.executable, headless=True)
        self._reset_engine()
        if self.ops or target:
            self._replay_to(target)

    def _reset_engine(self) -> None:
        assert self.env is not None
        # 进 demo 模式只为一件事：让**每条**响应（含 reset 的那条）都带 state_hash。
        # 官方 human_demo.py verify 比的就是响应里的哈希（不是 STATE_HASH 命令的），
        # 两条路径算出来的值实测不同，所以导出存档必须在同一条路径上取哈希。
        # DEMO_BEGIN 只改鼠标/窗口状态，不动模拟（人类示范重放与无 demo 重放终局一致）。
        self.env.begin_demo()
        self.initial_obs, _ = self.env.reset(deck=self.deck, task=self.spec)
        # 开局哈希必须紧跟在 reset 之后取：之后任何一条命令都会覆盖 _last_response。
        self.initial_hash = self.env._last_response.get("state_hash")
        self.env.set_lean_observation(True)
        # 精简观测才带 legal_actions（Board::CanPlantAt 的结果），所以从第一帧起就用它。
        self._reset_bookkeeping(self.env.observe())
        if not self.frames:
            self._frame_now(None)

    def _reset_bookkeeping(self, obs: dict) -> None:
        self.obs = obs
        self.tick = int(obs["tick"])
        self.terminal = bool(obs.get("terminal"))
        self.max_wave = int(obs.get("wave") or 0)
        self.totals = {}
        if obs.get("defenses"):
            self.saw_mowers = True
        self._plants_snapshot, self._zombies_snapshot = self._snapshot()
        self._mower_snapshot = self._mower_states()
        self.events = []
        self.wave_history = []
        self._wave_stats = dict(kills=0, lost=0, mowers=0)
        self._close_wave(self.max_wave)

    # ------------------------------------------------------------ 事件差分

    def _snapshot(self):
        """(row, col) -> 植物类型 / 僵尸 id -> (类型, 行, x, 在场上, 啃食中)。

        精简观测是定长数组，完整观测是字典，两种都要吃（observe 用完整观测渲染）。
        """
        plants = {}
        for plant in (self.obs or {}).get("plants") or []:
            if isinstance(plant, dict):
                if plant.get("squished"):
                    continue
                plants[(int(plant["row"]), int(plant["col"]))] = int(plant["type"])
            else:
                if plant[5]:
                    continue
                plants[(int(plant[0]), int(plant[1]))] = int(plant[2])
        zombies = {}
        for zombie in (self.obs or {}).get("zombies") or []:
            if isinstance(zombie, dict):
                zombies[int(zombie["id"])] = (int(zombie["type"]), int(zombie["row"]),
                                              float(zombie["x"]), bool(zombie.get("on_board")),
                                              bool(zombie.get("is_eating")))
            else:
                zombies[int(zombie[9])] = (int(zombie[1]), int(zombie[0]), float(zombie[2]),
                                           bool(zombie[5]), bool(zombie[6]))
        return plants, zombies

    def _mower_states(self) -> dict:
        states = {}
        for defense in (self.obs or {}).get("defenses") or []:
            if isinstance(defense, dict):
                states[int(defense["row"])] = int(defense["state"])
            else:
                states[int(defense[0])] = int(defense[1])
        return states

    def _note_events(self, action, ok, before_plants, before_zombies, before_mowers,
                     before_wave: int = 0) -> None:
        """把这一步发生的事记成事件（§B.6）。只记事实，不解释该怎么做。"""
        after_plants, after_zombies = self._snapshot()
        tick = self.tick
        for cell, seed in after_plants.items():
            if cell in before_plants:
                continue
            reason = "种下"
            if action.get("type") == "plant" and (int(action["row"]), int(action["col"])) == cell:
                reason = "种下"
            self.events.append(dict(kind="plant", tick=tick, row=cell[0], col=cell[1],
                                    plant=seed, how=reason))
        for cell, seed in before_plants.items():
            if cell in after_plants:
                continue
            self.events.append(dict(kind="lost", tick=tick, row=cell[0], col=cell[1],
                                    plant=seed,
                                    reason=self._loss_reason(cell, before_zombies)))
            self._wave_stats["lost"] += 1
        for zombie_id, info in before_zombies.items():
            if zombie_id in after_zombies:
                continue
            kind, row = info[0], info[1]
            if kind == 20:      # 蹦极僵尸抓完就走，不算击杀
                continue
            self.events.append(dict(kind="kill", tick=tick, zombie=kind, row=row))
            self._wave_stats["kills"] += 1
        for zombie_id, info in after_zombies.items():
            if zombie_id in before_zombies or not info[3]:
                continue
            if info[0] == 20:   # 蹦极出现：把它要偷的格记下来
                self.events.append(dict(kind="bungee", tick=tick, zombie=info[0], row=info[1]))
        wave = int((self.obs or {}).get("wave") or 0)
        started = wave > before_wave
        if started:
            self.events.append(dict(kind="wave", tick=tick, wave=wave))
        after_mowers = self._mower_states()
        for row, state in after_mowers.items():
            if before_mowers.get(row) == 1 and state != 1:
                self.events.append(dict(kind="mower", tick=tick, row=row))
                self._wave_stats["mowers"] += 1
        if started:
            self._close_wave(wave)
        self._plants_snapshot, self._zombies_snapshot = after_plants, after_zombies
        self._mower_snapshot = after_mowers

    def _loss_reason(self, cell, before_zombies) -> str:
        row, col = cell
        for info in before_zombies.values():
            kind, zrow, zx, on_board, eating = info
            if zrow != row or not on_board:
                continue
            zcol = int((zx - 40) // 80)
            if eating and zcol == col:
                return "被吃"
            if kind == 20 and abs(zcol - col) <= 1:
                return "被偷（蹦极）"
            if kind in (22, 23, 32) and abs(zcol - col) <= 1:
                return "被砸（巨人/投石车）"
        return "自毁或其他"

    def _close_wave(self, wave: int) -> None:
        """跨过一波时记一行历史（§B.7）。事件流不清空 —— 那是给 observe() 的。"""
        if self.wave_history and self.wave_history[-1]["wave"] == wave:
            return
        stats = self._wave_stats
        self.wave_history.append(dict(wave=wave, tick=self.tick,
                                      sun=(self.obs or {}).get("sun"),
                                      plants=len((self.obs or {}).get("plants") or []),
                                      kills=stats["kills"], lost=stats["lost"],
                                      mowers=stats["mowers"]))
        self._wave_stats = dict(kills=0, lost=0, mowers=0)
        self.max_wave = max(self.max_wave, wave)
        if not self._internal:
            # 内部试打局走的是 RESEED 换过的未来，不能算进「最远存活波」——
            # 否则稳健性试打会把止损判据骗过去（§A 的检查点看的就是这个数）。
            self.session.record_wave(self.spec.level, wave)

    def _replay_to(self, target: int) -> None:
        """reset 后重放到 tick target，并把 ops / frames 截断到那一点。

        操作日志是唯一真相：重放结束后内存里的 ops 就是"当前状态"的完整描述，
        所以这里同时改写 ops（跨越目标 tick 的 wait 会被切成两段）。
        """
        assert self.env is not None
        if self._reseeded is not None:
            raise RuntimeError(
                "这局用 RESEED %d 换过未来，操作日志已经不能重放（未来变了）。"
                "稳健性试打局不能 rewind / save / export；请新建一局。" % self._reseeded)
        cached = self._cache
        start_ops = 0
        if cached and cached["tick"] <= target and self.ops[:cached["n"]] == cached["ops"]:
            self.env.restore_fast(cached["snapshot"])
            self.obs = cached["obs"]
            self.tick = cached["tick"]
            self.terminal = cached["terminal"]
            self.max_wave = cached["max_wave"]
            self.totals = dict(cached["totals"])
            self.saw_mowers = cached["saw_mowers"]
            self.wave_history = [dict(row) for row in cached["wave_history"]]
            self._wave_stats = dict(cached["wave_stats"])
            self.events = []
            self._plants_snapshot, self._zombies_snapshot = self._snapshot()
            self._mower_snapshot = self._mower_states()
            start_ops = cached["n"]
        else:
            self._reset_engine()
            self._cache = None

        start_tick = self.tick
        def replay_step(action):
            before_wave = int(self.obs.get("wave") or 0)
            plants, zombies = self._plants_snapshot, self._zombies_snapshot
            mowers = self._mower_snapshot
            ok = self._step_raw(action)
            self._note_events(action, ok, plants, zombies, mowers, before_wave)

        kept = []
        for op in self.ops[start_ops:]:
            action = op["action"]
            if op["end_tick"] <= target:
                replay_step(action)
                kept.append(op)
                continue
            if action.get("type") == "wait" and op["tick"] < target:
                partial = target - op["tick"]
                replay_step({"type": "wait", "ticks": partial})
                kept.append(dict(op, action={"type": "wait", "ticks": partial}, end_tick=target))
            break
        self.ops = self.ops[:start_ops] + kept
        self.frames = [f for f in self.frames if f.get("operation", 0) <= len(self.ops)]
        self.tick = int(self.obs["tick"])
        # 重放本身也是模拟：它同样吃预算（§A 按引擎 tick 计，不按墙钟）。
        self.session.note_ticks(self.spec.level, self.tick - start_tick,
                                self.session.mode(self.spec.level))
        self._plants_snapshot, self._zombies_snapshot = self._snapshot()
        self._mower_snapshot = self._mower_states()

    def _step_raw(self, action: dict) -> bool:
        assert self.env is not None
        obs, _, terminal, _, info = self.env.step(action)
        self.obs = obs
        self.tick = int(obs["tick"])
        self.terminal = bool(terminal)
        self.max_wave = max(self.max_wave, int(obs.get("wave") or 0))
        if obs.get("defenses"):
            self.saw_mowers = True
        for name, value in info["events"].items():
            self.totals[name] = self.totals.get(name, 0) + value
        return bool(info["ok"])

    def _apply(self, action: dict) -> bool:
        self._start()
        assert self.env is not None
        level = self.spec.level
        if self.session.level(level)["exhausted"]:
            # §A：单关总预算用尽时，接口自己把最远进度存下来，再告诉模型换关。
            saved = self._auto_export_best()
            raise RuntimeError(
                "本关总预算（搜索 + 手动）已用尽。最远进度的存档已经自动导出到 %s，"
                "现在换下一关，有余力再回来。" % saved)
        before = self.tick
        before_wave = int((self.obs or {}).get("wave") or 0)
        before_plants, before_zombies = self._plants_snapshot, self._zombies_snapshot
        before_mowers = self._mower_snapshot
        ok = self._step_raw(action)
        self._note_events(action, ok, before_plants, before_zombies, before_mowers, before_wave)
        self.session.note_ticks(level, self.tick - before, self.session.mode(level))
        self._dirty = True
        if not self._internal:
            _mark_active(self)
        # 精简观测的响应不带 state_hash（那是给搜索用的轻量路径），存档要逐操作哈希，
        # 所以缺了就单独问一次。一次命令的代价，比关掉精简观测便宜得多。
        state_hash = self.env._last_response.get("state_hash") or self.env.state_hash()
        self.ops.append(dict(action=action, tick=before, end_tick=self.tick, ok=ok,
                             state_hash=state_hash))
        self._maybe_frame(action, ok)
        if self.terminal or int(self.obs.get("wave") or 0) > before_wave:
            self._persist_best()
        return ok

    def _persist_best(self):
        # 内部试打局走的是换过的未来，写进「最远进度」会把真局的纪录冲掉。
        if self._internal or self.env is None or self.obs is None:
            return None
        return self.session.save_best_log(self.spec.level, self.log(), wave=self.max_wave,
                                          won=int(self.obs.get("result") or 0) == 1)

    def _auto_export_best(self) -> str:
        """从磁盘上的最远操作日志导出，不能用当前较差的分支覆盖纪录。"""
        best = None
        try:
            path = self._persist_best()
            best = Game.load(path, manifest=self.manifest, executable=self.executable,
                             resource_dir=self.resource_dir)
            out = Path.cwd() / "llmplay_out"
            target = out / ("%s_tick%d" % (best.task_id, best.tick))
            index = 1
            while target.exists():
                target = out / ("%s_tick%d_%d" % (best.task_id, best.tick, index))
                index += 1
            best.export_archive(target)
            self.session.record_best_archive(self.spec.level, target)
            return str(target)
        except Exception as error:      # 存档失败不能掩盖"预算用尽"这件事
            return "(自动导出失败：%s)" % error
        finally:
            if best is not None:
                best.close()

    def _maybe_frame(self, action: dict, ok: bool) -> None:
        if action.get("type") != "wait":
            self._frame_now(action, rejected=not ok)
        elif self.tick - self._last_frame_tick() >= FRAME_INTERVAL or self.terminal:
            self._frame_now({"type": "wait", "ticks": self.tick - self._last_frame_tick()})

    def _last_frame_tick(self) -> int:
        return int(self.frames[-1]["tick"]) if self.frames else 0

    def _frame_now(self, action, rejected: bool = False) -> None:
        assert self.obs is not None
        frame = _frame(self.obs)
        frame.update(action=action, action_rejected=rejected, label_source=LABEL_SOURCE,
                     operation=len(self.ops))
        self.frames.append(frame)

    # ------------------------------------------------------------ 动作

    def plant(self, card: int, row: int, col: int) -> bool:
        """种下第 card 个卡槽的植物。card 是 observe()/legal() 里的卡槽序号。"""
        self._start()
        packets = (self.obs or {}).get("packets") or []
        if not 0 <= int(card) < len(packets):
            raise ValueError("卡槽 %s 不存在；本局卡槽是 0..%d（见 observe() 的卡槽行）"
                             % (card, len(packets) - 1))
        return self._apply({"type": "plant", "packet": int(card),
                            "col": int(col), "row": int(row)})

    def shovel(self, row: int, col: int) -> bool:
        """铲掉 (row, col) 上的植物。"""
        return self._apply({"type": "shovel", "col": int(col), "row": int(row)})

    def wait(self, ticks: int) -> None:
        """推进 ticks 个 tick。内部按 60 tick 分块（每块落一帧，便于事后回看）。"""
        self._start()
        remaining = int(ticks)
        if remaining < 0:
            raise ValueError("ticks 必须是非负整数")
        while remaining > 0 and not self.terminal:
            chunk = min(FRAME_INTERVAL, remaining)
            self._apply({"type": "wait", "ticks": chunk})
            remaining -= chunk

    def wait_until(self, target=None, *, wave=None, max_ticks: int = 30000, step: int = 15) -> None:
        """等到条件成立。target 可以是 tick（int）、"wave:N"、或 f(game) -> bool。

        wave= 关键字等价于 "wave:N"。条件不成立时最多再推进 max_ticks。
        """
        self._start()
        assert self.obs is not None
        if wave is not None:
            goal = int(wave)
            pred = lambda: int(self.obs["wave"]) >= goal  # noqa: E731
        elif callable(target):
            pred = lambda: bool(target(self))  # noqa: E731
        elif isinstance(target, str):
            if not target.startswith("wave"):
                raise ValueError("字符串条件只支持 'wave:N'")
            goal = int(target.split(":", 1)[1])
            pred = lambda: int(self.obs["wave"]) >= goal  # noqa: E731
        elif isinstance(target, int):
            goal_tick = int(target)
            pred = lambda: self.tick >= goal_tick  # noqa: E731
        else:
            raise TypeError("wait_until 需要 int tick、'wave:N' 或回调")
        spent = 0
        while not pred() and not self.terminal and spent < max_ticks:
            n = min(step, max_ticks - spent)
            self.wait(n)
            spent += n

    def rewind(self, tick: int) -> None:
        """回到 tick（从开局重放）。之后的操作会覆盖原来的后续。"""
        self._start()
        target = int(tick)
        if target < 0:
            raise ValueError("tick 必须是非负整数")
        if target > self.tick:
            raise ValueError("不能向前 rewind：当前 tick %d，目标 %d" % (self.tick, target))
        # 同一段第 9 次重写直接拒绝（§A）：计数在 budget.json 里，模型绕不过。
        self.session.note_rewind(self.spec.level, target, self.tick)
        # task81d：用 rewind 改写过的局在 meta 里标 manual_patch。
        self.manual_patch = True
        self._persist_best()
        self._replay_to(target)

    def branch(self) -> "Game":
        """复制一份独立的局（同一任务、同一操作序列）。各自一个引擎进程。

        搜索模式才允许；搜索被判定不佳/预算用尽后，这个入口直接抛异常。
        """
        self._start()
        self.session.require_branch(self.spec.level)
        child = Game(self.task_id, seed=self.seed, manifest=self.manifest,
                    executable=self.executable, resource_dir=self.resource_dir,
                    ops=[dict(op) for op in self.ops],
                    frames=[dict(f) for f in self.frames],
                    tick=self.tick, note=self.note)
        self.session.note_branch(self.spec.level, +1)
        child._is_branch = True
        return child


    # ------------------------------------------------------------ 稳健性（task81d）

    def _wave_tick(self, wave: int) -> int | None:
        """第 wave 波开始时的 tick（wave 0 = 开局）。没打到这一波就返回 None。"""
        if wave <= 0:
            return 0
        for row in self.wave_history:
            if int(row["wave"]) == int(wave):
                return int(row["tick"])
        return None

    def robustness(self, policy, *, k: int | None = None, checkpoints=None,
                   max_ticks: int = 90000, seed_config=None) -> dict:
        """换一个同样合理的未来，这套**闭环打法**还能不能赢（task81d）。

        policy 是一个函数：`policy(game)`，每一步只看当前局面决定动作，通过
        `game.plant / game.shovel / game.wait ...` 推进。外层会反复调用它，
        直到终局或超过 max_ticks。

        对每个检查点（默认 0 / 10 / 20 / 30 波，取已经打到的）：
        读档到那一波开始的 tick，`RESEED` 成 k 个新未来（种子来自包里独立的
        稳健性块，不是任务种子），用同一个 policy 打到终局，数胜局。

        这一步消耗搜索预算（重放 + 试打都是模拟 tick）。返回的 dict 同时写进
        `game.robustness`，`export_archive()` 会把它写进 meta。
        """
        self._start()
        plan = robustness_seeds(seed_config)
        waves = sorted(plan) if checkpoints is None else [int(w) for w in checkpoints]
        if not waves:
            raise ValueError("checkpoints 不能为空")
        source = _policy_source(policy)
        policy_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
        result = dict(policy_sha256=policy_sha, k=k, manual_patch=bool(self.manual_patch),
                      checkpoints={}, passed=False)
        for wave in waves:
            tick = self._wave_tick(wave)
            seeds = plan.get(wave) or []
            if k is not None:
                seeds = seeds[:int(k)]
            if tick is None or not seeds:
                result["checkpoints"][str(wave)] = dict(wins=None, k=len(seeds), reached=False,
                                                        note="还没打到这一波（或没有配置种子）")
                continue
            kept = _truncate_ops(self.ops, tick)
            trial = Game(self.task_id, seed=self.seed, manifest=self.manifest,
                         executable=self.executable, resource_dir=self.resource_dir,
                         ops=kept, frames=[], tick=tick, _internal=True)
            wins = 0
            try:
                trial._start()
                assert trial.env is not None
                snapshot = trial.env.snapshot_fast()
                try:
                    for seed in seeds:
                        trial.env.restore_fast(snapshot)
                        trial._reseeded = int(seed)
                        trial.ops = [dict(op) for op in kept]
                        trial.frames = []
                        trial._reset_bookkeeping(trial.env.observe())
                        trial.env.reseed(int(seed))
                        trial.obs = trial.env.observe()
                        trial._reset_bookkeeping(trial.obs)
                        start = trial.tick
                        _drive_policy(trial, policy, limit_tick=start + int(max_ticks))
                        if int((trial.obs or {}).get("result") or 0) == 1:
                            wins += 1
                finally:
                    trial.env.release_snapshot(snapshot)
            finally:
                trial.close()
            result["checkpoints"][str(wave)] = dict(wins=wins, k=len(seeds), reached=True)
        wave10 = result["checkpoints"].get("10") or {}
        if wave10.get("reached"):
            # 任务书 §task81d：第 10 波检查点 8 个新未来至少赢 5 个才算「稳」。
            result["passed"] = int(wave10["wins"]) >= 5
        self.robustness_report = result
        return result

    def snapshot(self) -> None:
        """在当前状态留一个进程内快照（rewind 回到这里时免重放）。可选。"""
        self._start()
        assert self.env is not None
        if self._cache:
            try:
                self.env.release_snapshot(self._cache["snapshot"])
            except Exception:
                pass
        self._cache = dict(tick=self.tick, snapshot=self.env.snapshot_fast(),
                           ops=[dict(op) for op in self.ops], n=len(self.ops),
                           obs=dict(self.obs or {}), terminal=self.terminal,
                           max_wave=self.max_wave, totals=dict(self.totals),
                           saw_mowers=self.saw_mowers,
                           wave_history=[dict(row) for row in self.wave_history],
                           wave_stats=dict(self._wave_stats))

    # ------------------------------------------------------------ 读局面

    def _refresh_full_observation(self) -> None:
        """渲染要用完整观测（植物状态、护具/盾、弹药都在里面）；推进用精简观测。"""
        assert self.env is not None
        self.env.set_lean_observation(False)
        try:
            self.obs = self.env.observe()
        finally:
            self.env.set_lean_observation(True)
        self.tick = int(self.obs["tick"])
        self.terminal = bool(self.obs.get("terminal"))
        self.max_wave = max(self.max_wave, int(self.obs.get("wave") or 0))
        if not self._internal:
            self.session.record_wave(self.spec.level, int(self.obs.get("wave") or 0))

    def observe(self, level: str = "full") -> str:
        """完整文字局面（§B）。第一行永远是状态行，末尾是事件，最后清空事件。"""
        self._start()
        from . import scene
        self._refresh_full_observation()
        text = scene.render(self, level=level)
        self.events = []
        self._observed_at = self.tick
        self._dirty = False
        return text

    def legend(self) -> str:
        """网格与明细里那些短代码是什么意思（第一次 observe 会自动附一次）。"""
        from . import scene
        return self.session.status_line(self.spec.level) + "\n" + scene.legend()

    def timeline(self) -> str:
        """每波一行的历史（§B.7）：波号、tick、阳光、植物数、击杀、失去、小推车。"""
        self._start()
        from . import scene
        return self.session.status_line(self.spec.level) + "\n" + scene.timeline(self)

    def legal(self) -> dict:
        """当前合法动作（引擎判定）：plants 是 (卡槽, 植物, 行, 列)，shovels 是可铲的格。"""
        self._start()
        from . import scene
        self._refresh_full_observation()
        out = scene.legal(self)
        out["status"] = self.session.status_line(self.spec.level)
        return out

    def summary(self) -> dict:
        """当前波、最远存活波、剩余小推车、阳光、各行威胁。"""
        self._start()
        from . import scene
        self._refresh_full_observation()
        out = scene.summary(self)
        out["status"] = self.session.status_line(self.spec.level)
        return out

    # ------------------------------------------------------------ 存档

    def log(self) -> dict:
        return dict(schema=SCHEMA, task_id=self.task_id, seed=self.seed, deck=self.deck,
                    note=self.note, tick=self.tick, ops=self.ops,
                    frames=self.frames, totals=self.totals,
                    robustness=self.robustness_report, manual_patch=bool(self.manual_patch))

    def save(self, path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(".%s.%s.tmp" % (target.name, os.getpid()))
        temporary.write_text(json.dumps(self.log(), ensure_ascii=False, separators=(",", ":")) + "\n",
                             encoding="utf-8")
        os.replace(temporary, target)
        return target

    @classmethod
    def load(cls, path, **kwargs) -> "Game":
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        if doc.get("schema") != SCHEMA:
            raise ValueError("不是 %s 存档：%s" % (SCHEMA, path))
        return cls(doc["task_id"], seed=doc["seed"], ops=doc.get("ops") or [],
                   frames=doc.get("frames") or [], tick=doc.get("tick"),
                   note=doc.get("note", ""), robustness=doc.get("robustness"),
                   manual_patch=bool(doc.get("manual_patch", False)), **kwargs)

    def export_archive(self, directory, *, note=None) -> Path:
        """导出 episode_archive_v1 存档（ticks.jsonl / frames.jsonl / meta.json）。

        导出的存档可以直接被仓库的 scripts/human_demo.py verify 重放校验。
        """
        self._start()
        assert self.env is not None and self.obs is not None and self.initial_obs is not None
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=False)
        with (target / "ticks.jsonl").open("w", encoding="utf-8") as tape:
            for op in self.ops:
                tape.write(json.dumps(dict(action=op["action"], tick=op["tick"],
                                           end_tick=op["end_tick"], ok=op["ok"],
                                           state_hash=op["state_hash"]),
                                      separators=(",", ":")) + "\n")
        with (target / "frames.jsonl").open("w", encoding="utf-8") as stream:
            for frame in self.frames:
                stream.write(json.dumps(frame, separators=(",", ":")) + "\n")
        episode = self.env.episode or {}
        result = int(self.obs.get("result") or 0)
        roster = (self.initial_obs.get("loadout_context") or {}).get("zombie_roster")
        meta = dict(
            schema=ARCHIVE_SCHEMA, label_source=LABEL_SOURCE,
            note=self.note if note is None else note,
            task=dict(id=self.task_id, level=self.spec.level, seed=self.spec.seed,
                      deck=list(self.deck), policy="llm", playthrough=self.spec.playthrough,
                      zombie_count_multiplier=self.spec.zombie_count_multiplier,
                      wave_cap=self.spec.wave_cap,
                      preplanted=[list(p) for p in self.spec.preplanted]),
            environment_task=self._environment_task(),
            engine_sha=episode.get("source_revision"), engine_dirty=episode.get("source_dirty"),
            # executable_sha256 写成一句自解释的标记，不是 sha：
            # state_hash 跨进程不可复现（同一二进制两次 reset 会得到不同的开局哈希，
            # 实测见 task81 问题节）。human_demo.py verify 的规则是
            # "记录的 sha 与当前二进制不同 → 只比 tick/ok/终局，并在 meta 里写明"，
            # 所以这里如实声明"哈希不可复现"，让校验走那条明示的宽松路径，
            # 而不是假装哈希一致。真正的二进制指纹在 recording_executable_sha256。
            executable_sha256="not-recorded:state_hash_not_reproducible_across_processes",
            recording_executable_sha256=hashlib.sha256(Path(self.executable).read_bytes()).hexdigest(),
            resource_sha256=episode.get("resource_sha256"),
            properties_partner_sha256=episode.get("properties_partner_sha256"),
            context=dict(level=self.spec.level, multiplier=self.spec.zombie_count_multiplier,
                         roster=roster),
            hash_schema="portable_snapshot_v1_without_draw_statistics_nonreproducible_across_processes",
            initial_state_hash=self.initial_hash,
            # task81d：稳健性 = 每个检查点 k 个新未来里的胜局数 + 打法函数源码哈希。
            # manual_patch=True 表示这局用 rewind 手工补救过；稳健性按补救前的闭环打法算。
            robustness=self.robustness_report, manual_patch=bool(self.manual_patch),
            sun_collection="environment_auto", verified=False,
            frame_count=len(self.frames), operations=len(self.ops), totals=self.totals,
            outcome=dict(result=result, won=result == 1,
                         reason="通关" if result == 1 else ("僵尸进屋" if self.obs.get("terminal") else "中途结束"),
                         final_wave=self.obs.get("wave"), wave_count=self.obs.get("wave_count"),
                         final_tick=self.obs.get("tick"), actions=max(len(self.frames) - 1, 0),
                         plants_eaten=self.totals.get("plants_eaten", 0),
                         plants_eaten_free=self.totals.get("plants_eaten_free", 0)),
        )
        (target / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
        return target

    def _environment_task(self) -> dict:
        spec = self.spec
        profile = spec.profile
        out = dict(level=spec.level, seed=spec.seed, playthrough=spec.playthrough,
                   profile=dict(seed_slot_count=profile.seed_slot_count,
                                imitater_owned=profile.imitater_owned,
                                first_aid_owned=profile.first_aid_owned,
                                pool_cleaner_owned=profile.pool_cleaner_owned,
                                roof_cleaner_owned=profile.roof_cleaner_owned,
                                rake_charges=profile.rake_charges,
                                owned_upgrade_plants=list(profile.owned_upgrade_plants)),
                   forced_seeds=list(spec.forced_seeds), loadout_mode=spec.loadout_mode,
                   zombie_count_multiplier=spec.zombie_count_multiplier,
                   wave_cap=spec.wave_cap,
                   preplanted=[list(p) for p in spec.preplanted])
        if spec.conveyor_start:
            out["conveyor_start"] = spec.conveyor_start
        if spec.fog_observation:
            out["fog_observation"] = spec.fog_observation
        return out

    def close(self) -> None:
        if self.env is not None:
            try:
                self._persist_best()
            finally:
                self.env.close()
                self.env = None
        if self._is_branch:
            self.session.note_branch(self.spec.level, -1)
            self._is_branch = False
        if self in _LIVE_GAMES:
            _LIVE_GAMES.remove(self)

    def __enter__(self) -> "Game":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
