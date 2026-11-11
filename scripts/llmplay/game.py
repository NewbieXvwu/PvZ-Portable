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
import inspect
import copy
import json
import os
import sys
import uuid
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
from pvz_episode_archive import SCHEMA as ARCHIVE_SCHEMA, read_meta, write_meta
LABEL_SOURCE = "llm_agent"
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

    沙箱里很常见：包用 zipfile 解开，直接 Popen 会报
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


def task_spec(task: dict, seed=None, deck=None):
    """Use the same task/deck derivation as collection and demonstration recording."""
    from render_episode import training_task_kwargs
    definition = dict(task)
    if deck is not None:
        definition['deck'] = list(deck)
        definition.pop('deck_random', None)
        definition.pop('deck_perturb', None)
    kw = training_task_kwargs(definition, seed)
    spec = TaskSpec(level=kw['level'], seed=kw['seed'], playthrough=2,
                    profile=profile_for_deck(tuple(kw['deck']), kw['first_aid_owned']),
                    **{key: value for key, value in kw['task_extra'].items() if value is not None})
    return spec, kw['deck'], kw['seed']




def validate_deck(env, deck):
    """Deck warnings are facts from the reset native board."""
    warnings = env.deck_warnings(deck)['warnings']
    if warnings:
        raise ValueError('卡组不符合引擎要求：%s' % warnings)


def required_checkpoint(task):
    """无波次关卡使用明确的 tick 检查点，不以不存在的 wave 10 判稳健。"""
    return task.get("robustness_checkpoint", "tick:6000" if int(task["level"]) == 50 else "10")


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
    """只对实际源码取哈希；函数名不能代替实现的指纹。"""
    if not callable(policy):
        raise TypeError("policy 必须是闭环打法函数，不能是固定动作序列")
    try:
        return inspect.getsource(policy)
    except (OSError, TypeError) as error:
        raise ValueError("无法读取打法函数源码：请将 policy 写入 .py 文件，再导入调用 robustness()") from error


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


class InvalidDrillError(ValueError):
    """题目起点无目标，已记入进度并跳过，调用 next_task 继续。"""


class Game:
    """一局。构造时只读任务，第一次用引擎时才真正 reset + 重放。"""

    def __init__(self, task_id: str, *, seed=None, manifest=None, executable=None,
                 resource_dir=None, ops=None, frames=None, tick=None, note: str = "",
                 deck=None, reason="", strategy="", deck_review="", abandon_reason=None, attempt_id=None,
                 robustness=None, manual_patch: bool = False, run_id=None, _internal: bool = False,
                 _gate: bool = True, cycle=None, workflow_layer=None, _restore_claim=False,
                 deck_attempt_seeds=None, _definition=None):
        tasks = {task_id: _definition} if _definition is not None else load_tasks(manifest)
        if task_id not in tasks:
            raise ValueError("没有任务 %r。可用：%s" % (task_id, sorted(tasks)))
        self.task_id = task_id
        self.task_definition = tasks[task_id]
        from .delivery import ensure_run_id, progress
        self.run_id = run_id or ensure_run_id()
        if seed is None:
            pending = progress()["tasks"].get(task_id, {}).get("available_seeds")
            if pending:
                seed = pending[0]
        if self.task_definition.get("deck_selection") == "player" and not reason.strip():
            raise ValueError("自选卡组请填写 reason 选卡理由")
        self.spec, self.deck, self.seed = task_spec(self.task_definition, seed, deck)
        self.deck_reason, self.strategy, self.deck_review = reason, strategy, deck_review
        self.deck_attempt_seeds = list(deck_attempt_seeds or [])
        self._declared_strategy = strategy
        self.cycle = int(cycle or 1)
        self.workflow_layer = workflow_layer or self.task_definition.get("layer", "A")
        self.abandon_reason = abandon_reason
        self.attempt_id = attempt_id or uuid.uuid4().hex
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
        self.terminal = abandon_reason is not None
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
        # task220：中盘接回的来源（resume/<archive>）；普通开局是 None。
        self.resume_id = None
        self.resume_source = {}
        # task238：drill / scenario 的题面与机械判据；普通局为 None。
        self.drill_meta = None
        self._drill_missing_at_start = None
        self.scenario_meta = None
        self.future_seed = None
        self._root_operations = 0
        self._root_tick = 0
        self._root_mowers = None
        self.last_spawn_id = None
        # RESEED 过的局不能再按操作日志重放（未来已经换过），所以内部试打局带这个标记。
        self._reseeded = None
        # 内部试打局（robustness 的 k 个新未来）不写「最远进度」、不进自动 observe。
        self._internal = bool(_internal)
        # task139：层序 / 卡组差异 / 打法重复的硬把关。稳健性试打与读档恢复不受限。
        self._claimed = False
        if _gate and not self._internal:
            from .delivery import check
            context = check(self.task_id, self.deck, self.strategy, self.seed)
            self.cycle = context.get("cycle", self.cycle)
            self.workflow_layer = context.get("workflow_layer", self.workflow_layer)
            # 队列占位：别的子代理不会领到同一项；进程崩了 30 分钟后自动释放。
            self._hold_claim()
        elif _restore_claim and self.workflow_layer in ("decks", "native") and not self._internal:
            # 同片中盘恢复继续持有并最终释放原来的关卡+种子占位。
            self._hold_claim()
        self.session = budget.session()
        self.budget_key = self.task_id
        self.session.register_task(self.budget_key, self.spec.level)
        if not self._internal:
            _LIVE_GAMES.append(self)

    # ------------------------------------------------------------ 引擎生命周期

    def _start(self) -> None:
        if self.env is not None:
            return
        _ensure_executable(self.executable)
        target = self.tick          # _reset_engine 会把 self.tick 归零，先记住
        from .resources import app_data
        self.env = PvZEnv(resource_dir=self.resource_dir, executable=self.executable, headless=True,
                          save_dir=app_data(self.executable, self.resource_dir, self.session))
        try:
            self._reset_engine()
            self._capture_drill_root(0)
            if self.ops or target:
                self._replay_to(target)
            if getattr(self, "_resume_prefix_pending", False):
                self.frames = []
                self._frame_now(None)
                self._resume_prefix_pending = False
            # task238：drill/scenario 的起点就是重放终点；判据从这一刻算起。
            self._root_operations = len(self.ops)
            self._root_tick = self.tick
            self._root_mowers = self._mower_count()
            if self.drill_meta and self._drill_missing_at_start is not None:
                targets = self.drill_meta.get("targets") or []
                if len(self._drill_missing_at_start) == len(targets) and not self._internal:
                    from .delivery import invalidate_drill
                    invalidate_drill(self.task_id, self.seed, self.drill_meta,
                                     self._drill_missing_at_start)
                    self._release_claim()
                    raise InvalidDrillError("drill 起点无目标，已跳过该题全部种子。调用 next_task() 领下一题")
            if self.abandon_reason is not None:
                self.terminal = True
        except BaseException:
            self.env.close()
            self.env = None
            raise

    def _reset_engine(self) -> None:
        assert self.env is not None
        self.env.begin_demo()
        self.initial_obs, _ = self.env.reset(deck=self.deck, task=self.spec)
        if self.task_definition.get("deck_selection") == "player":
            validate_deck(self.env, self.deck)
            options = self.env.deck_options()
            field = self.task_definition.get("unlock_field", "first_adventure_available")
            available = set(options[field]) | set(self.spec.profile.owned_upgrade_plants)
            locked = sorted(set(self.deck) - available)
            if locked:
                raise ValueError("该关尚未解锁的植物：%s" % locked)
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
            if self.spec.fog_observation == "human_v2" and (self.obs or {}).get("fog"):
                # 公共观测过滤了雾中僵尸；消失不代表已经被消灭。
                self.events.append(dict(kind="unseen", tick=tick, zombie=kind, row=row))
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
            self.session.record_wave(self.budget_key, wave)

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
        self._capture_drill_root(start_ops)
        def replay_step(action):
            before_wave = int(self.obs.get("wave") or 0)
            plants, zombies = self._plants_snapshot, self._zombies_snapshot
            mowers = self._mower_snapshot
            ok = self._step_raw(action)
            self._note_events(action, ok, plants, zombies, mowers, before_wave)

        kept = []
        for index, op in enumerate(self.ops[start_ops:], start_ops + 1):
            action = op["action"]
            if op["end_tick"] <= target:
                replay_step(action)
                kept.append(op)
                self._capture_drill_root(index)
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
        self.session.note_ticks(self.budget_key, self.tick - start_tick,
                                "search" if self._internal else self.session.mode(self.budget_key))
        self._plants_snapshot, self._zombies_snapshot = self._snapshot()
        self._mower_snapshot = self._mower_states()

    def _capture_drill_root(self, operation):
        if (self.drill_meta is not None
                and operation == int(self.drill_meta.get("prefix_operations") or 0)):
            from drill_start import missing_targets
            self._drill_missing_at_start = missing_targets(
                self.env, self.obs, self.drill_meta.get("targets") or [])

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
        if self.abandon_reason is not None:
            raise RuntimeError("本局已放弃，请用 Game(task_id, deck=..., reason=...) 重开")
        self._start()
        assert self.env is not None
        level = self.budget_key
        before = self.tick
        before_wave = int((self.obs or {}).get("wave") or 0)
        before_plants, before_zombies = self._plants_snapshot, self._zombies_snapshot
        before_mowers = self._mower_snapshot
        ok = self._step_raw(action)
        if self._claimed:
            self._hold_claim()
        self._note_events(action, ok, before_plants, before_zombies, before_mowers, before_wave)
        self.session.note_ticks(level, self.tick - before,
                                "search" if self._internal else self.session.mode(level))
        self._dirty = True
        if not self._internal:
            _mark_active(self)
        self.ops.append(dict(action=action, tick=before, end_tick=self.tick, ok=ok))
        self._maybe_frame(action, ok)
        if self.terminal or int(self.obs.get("wave") or 0) > before_wave:
            self._persist_best()
        return ok

    def _persist_best(self):
        # 内部试打局走的是换过的未来，写进「最远进度」会把真局的纪录冲掉。
        if self._internal or self.env is None or self.obs is None:
            return None
        return self.session.save_best_log(self.budget_key, self.log(), wave=self.max_wave,
                                          won=int(self.obs.get("result") or 0) == 1)

    def _maybe_frame(self, action: dict, ok: bool) -> None:
        if action.get("type") != "wait":
            self._frame_now(action, rejected=not ok)
        elif self.tick - self._last_frame_tick() >= FRAME_INTERVAL or self.terminal:
            self._frame_now({"type": "wait", "ticks": self.tick - self._last_frame_tick()})

    def _last_frame_tick(self) -> int:
        return int(self.frames[-1]["tick"]) if self.frames else 0

    def _mower_count(self) -> int:
        """Available lawn mowers in the current observation (state 1 = ready)."""
        total = 0
        for defense in (self.obs or {}).get("defenses") or []:
            state = defense.get("state", 0) if isinstance(defense, dict) else defense[1]
            total += int(state) == 1
        return total

    def _frame_now(self, action, rejected: bool = False) -> None:
        assert self.obs is not None
        frame = _frame(self.obs)
        frame.update(action=action, action_rejected=rejected, label_source=LABEL_SOURCE,
                     operation=len(self.ops))
        self.frames.append(frame)

    # ------------------------------------------------------------ 动作

    def plant(self, card, row: int, col: int) -> bool:
        """种下卡片。card 可以是卡槽序号（int），也可以是卡片名（str）。

        卡槽序号就是 observe()/legal() 里的 [0] [1] …。名字写法见 cards.py：
        "doom_shroom" / "doomshroom" / "毁灭菇"。名字在**调用这一刻**才解析成卡槽，
        所以传送带关用掉一张卡、后面的卡整体前移之后，名字依然指得对。
        """
        if self.abandon_reason is not None:
            raise RuntimeError("本局已放弃，请重开")
        self._start()
        from . import cards
        if isinstance(card, str):
            index = cards.slot_for(self.obs, card)
        else:
            index = int(card)
        packets = (self.obs or {}).get("packets") or []
        if not 0 <= index < len(packets):
            raise ValueError("卡槽 %s 不存在；本局卡槽是 0..%d（见 observe() 的卡槽行）"
                             % (index, len(packets) - 1))
        return self._apply({"type": "plant", "packet": index,
                            "col": int(col), "row": int(row)})

    def shovel(self, row: int, col: int) -> bool:
        """铲掉 (row, col) 上的植物。"""
        return self._apply({"type": "shovel", "col": int(col), "row": int(row)})

    # ------------------------------------------------------------ 摆题（task237）

    def spawn(self, zombie_type: int, row: int, x: float, target_col: int | None = None) -> bool:
        """在 (row, x) 放一只僵尸；蹦极的 x 表示目标格，target_col 可覆盖。

        成功后新僵尸的引擎 id 在 ``self.last_spawn_id``，方便 declare_scenario 直接引用。
        """
        if self.abandon_reason is not None:
            raise RuntimeError("本局已放弃，请重开")
        self._start()
        from drill_start import on_board
        before = set(on_board(self.env, self.obs or {}))
        action = {"type": "spawn", "zombie_type": int(zombie_type), "row": int(row),
                  "x": float(x)}
        if target_col is not None:
            action["target_col"] = int(target_col)
        ok = self._apply(action)
        after = set(on_board(self.env, self.obs or {}))
        added = sorted(after - before)
        self.last_spawn_id = added[-1] if added else None
        return ok

    def set_sun(self, sun: int) -> bool:
        """把当前阳光设为 sun（0..9990）。只在真有棋盘时改变局面。"""
        if self.abandon_reason is not None:
            raise RuntimeError("本局已放弃，请重开")
        self._start()
        return self._apply({"type": "set_sun", "sun": int(sun)})

    def set_cooldown(self, slot: int, ticks: int) -> bool:
        """让卡槽 slot 在 ticks 次更新后可用（0 = 立刻可用）。"""
        if self.abandon_reason is not None:
            raise RuntimeError("本局已放弃，请重开")
        self._start()
        return self._apply({"type": "set_cooldown", "slot": int(slot), "ticks": int(ticks)})

    # ------------------------------------------------------------ 必刷题 / 出题（task238）

    def reseed(self, seed: int) -> bool:
        """从当前局面换一条同样合理的未来（RESEED），之后照常 plant / wait。

        它只改尚未开始的波次与在场随机流，不改当前棋盘；操作日志里会留下一条
        ``reseed`` 记录，导入时按同一条路径重放。
        """
        if self.abandon_reason is not None:
            raise RuntimeError("本局已放弃，请重开")
        self._start()
        if self._claimed and self.workflow_layer in ("drill", "scenario") and self.future_seed != int(seed):
            self._release_claim()
        self.future_seed = int(seed)
        ok = self._apply({"type": "reseed", "seed": int(seed)})
        if self.workflow_layer in ("drill", "scenario"):
            self._hold_claim()
        return ok

    def zombies(self) -> list[dict]:
        """当前在场的僵尸（id / row / type / x / hp / body_health / phase），用于声明出题目标。

        ``hp`` 是本体血量（完整观测里的 ``body_health``）；护具血量另列。
        """
        self._start()
        from drill_start import on_board
        return list(on_board(self.env, self.obs or {}).values())

    def _judge(self, block: dict) -> dict:
        assert self.env is not None and self.obs is not None
        from drill_start import judge
        if self._root_mowers is None:
            self._root_mowers = self._mower_count()
        # 11-05: _start() re-reads root mowers/tick every time the sandbox recycles the engine
        # process and Game.load replays to the *current* tick, so a mower spent before the
        # restart vanished from mower_used (72 balloon drills judged success here, failure in
        # the importer).  The item block persists the true root; the importer judges with it.
        root_mowers = block.get("mowers_start")
        root_tick = block.get("root_tick")
        return judge(self.env, self.obs, list(block.get("targets") or []),
                     int(self._root_mowers if root_mowers is None else root_mowers), self._mower_count(),
                     horizon=int(block.get("horizon") or 900),
                     root_tick=int(self._root_tick if root_tick is None else root_tick))

    def drill_result(self) -> dict:
        """按 drill_start 的机械判据算这道必刷题；不写盘，随时可查。"""
        self._start()
        if not self.drill_meta:
            raise ValueError("这不是 drill 项；请用 next_task() 返回的 layer=drill 项")
        from drill_start import horizon_reached
        result = self._judge(self.drill_meta)
        missing = self._drill_missing_at_start
        invalid = missing is None or bool(missing) or not self.drill_meta.get("targets")
        result.update(invalid_start=invalid, missing_targets=missing)
        if invalid or self.abandon_reason is not None:
            result["success"] = False
        result.update(reached=horizon_reached(self.obs, result["root_tick"],
                                              int(self.drill_meta.get("horizon") or 900)),
                      knowledge=self.drill_meta.get("knowledge"))
        return result

    def declare_scenario(self, knowledge: str, horizon: int = 900, targets=None) -> dict:
        """声明这道自出题考什么、视界多长、要清掉哪些 id；顺手跑一次“什么都不做”对照。

        对照只推进这一局的一次短视界（不写操作日志、不占训练行），然后恢复原局面。
        正常情况下不做任何事应当失败；失败不是异常，会写进导出 meta 供下游过滤。
        """
        self._start()
        assert self.env is not None and self.obs is not None
        if not str(knowledge or "").strip():
            raise ValueError("declare_scenario 需要知识点标签")
        if type(horizon) is not int or not 300 <= horizon <= 6000:
            raise ValueError("scenario horizon 必须是 300..6000 的整数")
        current = {int(z["id"]): dict(z) for z in self.zombies()}
        chosen = []
        for target in targets or []:
            target_id = int(target["id"] if isinstance(target, dict) else target)
            if target_id not in current:
                raise ValueError("目标僵尸 %d 不在场上；先 spawn 再用 g.zombies() 取 id" % target_id)
            chosen.append(current[target_id])
        if not chosen:
            raise ValueError("scenario 至少要声明一个在场目标 id（g.zombies()）")
        self.scenario_meta = dict(
            dict(self.scenario_meta or {}),
            knowledge=str(knowledge).strip(), horizon=int(horizon),
            root_tick=int(self._root_tick), prefix_operations=int(self._root_operations),
            targets=chosen, mowers_start=int(self._root_mowers or 0),
            injection_operations=[index + 1 for index, op in enumerate(self.ops)
                                  if index >= int(self._root_operations)
                                  and op.get("action", {}).get("type") in
                                  ("spawn", "set_sun", "set_cooldown", "set_wave", "reseed")])
        snapshot = self.env.snapshot_fast()
        try:
            obs, _reward, terminal, _truncated, _info = self.env.step(
                {"type": "wait", "ticks": int(horizon)})
            self.obs, self.tick, self.terminal = obs, int(obs["tick"]), bool(terminal)
            baseline = self._judge(self.scenario_meta)
        finally:
            self.env.restore_fast(snapshot)
            self._reset_bookkeeping(self.env.observe())
        self.scenario_meta["baseline"] = dict(baseline, success=bool(baseline["success"]))
        return self.scenario_meta

    def set_wave(self, wave: int, countdown: int | None = None) -> bool:
        """把当前波次设为 wave（只能往前），可选设置下一波倒计时。"""
        if self.abandon_reason is not None:
            raise RuntimeError("本局已放弃，请重开")
        self._start()
        action = {"type": "set_wave", "wave": int(wave)}
        if countdown is not None:
            action["countdown"] = int(countdown)
        return self._apply(action)

    def wait(self, ticks: int) -> None:
        """推进 ticks 个 tick。内部按 60 tick 分块（每块落一帧，便于事后回看）。"""
        if self.abandon_reason is not None:
            raise RuntimeError("本局已放弃，请重开")
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
        if self.abandon_reason is not None:
            raise RuntimeError("本局已放弃，请重开")
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
        self._require_primary("rewind()")
        self._start()
        target = int(tick)
        if target < 0:
            raise ValueError("tick 必须是非负整数")
        if target > self.tick:
            raise ValueError("不能向前 rewind：当前 tick %d，目标 %d" % (self.tick, target))
        # 同一段第 9 次重写直接拒绝（§A）：计数在 budget.json 里，模型绕不过。
        self.session.note_rewind(self.budget_key, target, self.tick)
        # task81d：用 rewind 改写过的局在 meta 里标 manual_patch。
        self.manual_patch = True
        self.robustness_report = None
        self._persist_best()
        self._replay_to(target)

    def rewind_resume(self, *, reason: str, to_tick=None):
        """记录本次救局失败，返回已认领的更早子局；三次用尽或已到开局则返回 None。"""
        self._require_primary("rewind_resume()")
        if not self.resume_id or self.workflow_layer != "resume":
            raise ValueError("只有 resume 题能回退切点")
        if not reason.strip():
            raise ValueError("请说明从哪一波开始走偏、缺什么")
        cutoff = int(self.resume_source["cutoff_tick"])
        count = int(self.resume_source.get("rewind_count", 0))
        if to_tick is not None and (type(to_tick) is not int or not 0 <= to_tick < cutoff):
            raise ValueError("to_tick 必须是早于当前切点的非负整数")
        self._start()
        if self.obs.get("result") == 1:
            raise ValueError("已通关的局不能记为救局失败")
        root_id = self.resume_source.get("original_resume_id") or self.resume_id
        source = dict(self.resume_source, original_resume_id=root_id, diagnosis=reason)
        child = None
        if count >= 3 or cutoff == 0:
            source["structural"] = True
        else:
            from .decks import resume_catalog
            original = next((item for item in resume_catalog() if item["id"] == root_id), {})
            target = cutoff // 2 if to_tick is None else to_tick
            chain = list(source.get("rewind_chain") or []) + [dict(
                parent_resume_id=self.resume_id, from_tick=cutoff, cutoff_tick=target,
                rewind_count=count + 1, diagnosis=reason)]
            child = dict(original, **{k: v for k, v in source.items() if k != "next"})
            child.update(id=root_id + "__rewind%d" % (count + 1),
                         archive=original.get("archive") or Path(self._source_archive).name,
                         seed=self.seed, deck=list(self.deck), level=self.spec.level,
                         multiplier=self.spec.zombie_count_multiplier,
                         cutoff_tick=target, rewind_count=count + 1, rewind_chain=chain)
            source["next"] = child
        self.resume_source = source
        if not self.terminal:
            self.abandon_reason, self.terminal = reason, True
        if child:
            # 发布之前用 task265 的同一占位机制保留子项，其他进程不能抢到它。
            from .workflow import hold, release
            hold(child["id"], self.seed)
        try:
            self._deliver()
        except BaseException:
            if child:
                release(child["id"], self.seed)
            raise
        finally:
            self.close()
        if child:
            return type(self).load_archive(self._source_archive, resume_item=child,
                                           executable=self.executable, resource_dir=self.resource_dir,
                                           strategy=self.strategy)
        return None

    def branch(self) -> "Game":
        """复制一份独立的局（同一任务、同一操作序列）。各自一个引擎进程。

        搜索模式才允许；搜索被判定不佳/预算用尽后，这个入口直接抛异常。
        """
        self._require_primary("branch()")
        self._start()
        self.session.require_branch(self.budget_key)
        child = Game(self.task_id, seed=self.seed, manifest=self.manifest,
                    deck=self.deck, reason=self.deck_reason, strategy=self.strategy, deck_review=self.deck_review, attempt_id=self.attempt_id,
                    deck_attempt_seeds=self.deck_attempt_seeds,
                    executable=self.executable, resource_dir=self.resource_dir,
                    ops=[dict(op) for op in self.ops],
                    frames=[dict(f) for f in self.frames],
                    tick=self.tick, note=self.note, manual_patch=self.manual_patch,
                    run_id=self.run_id, _gate=False, cycle=self.cycle, workflow_layer=self.workflow_layer,
                    robustness=copy.deepcopy(self.robustness_report))
        self.session.note_branch(self.budget_key, +1)
        child._is_branch = True
        return child


    # ------------------------------------------------------------ 稳健性（task81d）

    def _require_primary(self, operation: str) -> None:
        if self.abandon_reason is not None and operation not in ("save()", "export_archive()"):
            raise RuntimeError("已放弃的局不能调用 %s" % operation)
        if self._internal:
            raise RuntimeError("稳健性试打局只允许观察与正常动作，不能调用 %s；换过未来的操作日志不能独立重放。"
                               % operation)

    def _wave_tick(self, wave: int) -> int | None:
        """首次观测到该波的 tick（定位上界；wave 0 = 开局）；未到达返回 None。"""
        if wave <= 0:
            return 0
        for row in self.wave_history:
            if int(row["wave"]) == int(wave):
                return int(row["tick"])
        return None

    def _seek_wave_start(self, wave: int, upper_tick: int) -> None:
        """在包含波次切换的 wait 内定位首 tick；仅供内部评估重放使用。"""
        assert self._internal and self.env is not None
        if int(self.obs.get("wave") or 0) >= wave:
            return
        start = self.tick
        limit = int(upper_tick) - start
        if limit < 1:
            raise RuntimeError("检查点重放未达到第 %s 波" % wave)
        prefix = [dict(op) for op in self.ops]
        snapshot = self.env.snapshot_fast()

        def advance(ticks):
            self.env.restore_fast(snapshot)
            self.ops = [dict(op) for op in prefix]
            self.frames = []
            self._reset_bookkeeping(self.env.observe())
            self.wait(ticks)

        try:
            advance(limit)
            if int(self.obs.get("wave") or 0) < wave:
                raise RuntimeError("检查点重放未复现第 %s 波，请先核对任务与操作日志" % wave)
            low, high = 1, limit
            while low < high:
                middle = (low + high) // 2
                advance(middle)
                if int(self.obs.get("wave") or 0) >= wave:
                    high = middle
                else:
                    low = middle + 1
            advance(low)
        finally:
            if self.env is not None:
                self.env.release_snapshot(snapshot)

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
        self._require_primary("robustness()")
        plan = robustness_seeds(seed_config)
        required = str(required_checkpoint(self.task_definition))
        waves = ([required] if required.startswith("tick:") or self.task_definition.get("deck_selection") == "player" else sorted(plan)) if checkpoints is None else list(checkpoints)
        if not waves:
            raise ValueError("checkpoints 不能为空")
        source = _policy_source(policy)
        k = len(plan.get(10) or next(iter(plan.values()), [])) if k is None else int(k)
        if not 1 <= k <= 8:
            raise ValueError("k 必须在 1..8 之间；正式稳健标准固定为 8 个新未来")
        self._start()
        self._persist_best()
        result = dict(k=k, manual_patch=bool(self.manual_patch),
                      checkpoints={}, required_checkpoint=required, deck=list(self.deck), passed=False)
        for checkpoint in waves:
            key = str(checkpoint)
            is_tick = key.startswith("tick:")
            wave = 0 if is_tick else int(key)
            target_tick = int(key.split(":", 1)[1]) if is_tick else None
            tick = (target_tick if target_tick <= self.tick else None) if is_tick else self._wave_tick(wave)
            seeds = plan.get(10 if is_tick else wave) or []
            if k is not None:
                seeds = seeds[:int(k)]
            if tick is None or not seeds:
                result["checkpoints"][key] = dict(wins=None, k=len(seeds), reached=False,
                                                        note="还没打到这一波（或没有配置种子）")
                continue
            kept = _truncate_ops(self.ops, tick)
            replay_tick = tick
            if wave > 0:
                # 先停在切换所在的 wait 之前；同 tick 的后续种植也不应混入检查点。
                for index, op in enumerate(self.ops):
                    if op["tick"] < tick <= op["end_tick"]:
                        kept = [dict(entry) for entry in self.ops[:index]]
                        replay_tick = int(op["tick"])
                        break
            trial = Game(self.task_id, seed=self.seed, manifest=self.manifest,
                         executable=self.executable, resource_dir=self.resource_dir,
                         ops=kept, frames=[], tick=replay_tick, run_id=self.run_id, _internal=True,
                         deck=self.deck, reason=self.deck_reason, strategy=self.strategy)
            wins = 0
            try:
                trial._start()
                assert trial.env is not None
                if wave > 0:
                    trial._seek_wave_start(wave, tick)
                    kept = [dict(op) for op in trial.ops]
                    tick = trial.tick
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
                    if trial.env is not None:
                        trial.env.release_snapshot(snapshot)
            finally:
                trial.close()
            result["checkpoints"][key] = dict(wins=wins, k=len(seeds), reached=True, tick=tick)
        wave10 = result["checkpoints"].get(required) or {}
        if wave10.get("reached"):
            # 任务书 §task81d：第 10 波检查点 8 个新未来至少赢 5 个才算「稳」。
            result["passed"] = int(wave10["k"]) == 8 and int(wave10["wins"]) >= 5
        self.robustness_report = result
        if self.env is not None:
            self._deliver()
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
        self.terminal = self.abandon_reason is not None or bool(self.obs.get("terminal"))
        self.max_wave = max(self.max_wave, int(self.obs.get("wave") or 0))
        if not self._internal:
            self.session.record_wave(self.budget_key, int(self.obs.get("wave") or 0))

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
        return self.session.status_line(self.budget_key) + "\n" + scene.legend()

    def timeline(self) -> str:
        """每波一行的历史（§B.7）：波号、tick、阳光、植物数、击杀、失去、小推车。"""
        self._start()
        from . import scene
        return self.session.status_line(self.budget_key) + "\n" + scene.timeline(self)

    def legal(self) -> dict:
        """当前合法动作（引擎判定）：plants 是 (卡槽, 植物, 行, 列)，shovels 是可铲的格。"""
        self._start()
        from . import scene
        self._refresh_full_observation()
        out = scene.legal(self)
        out["status"] = self.session.status_line(self.budget_key)
        return out

    def summary(self) -> dict:
        """当前波、最远存活波、剩余小推车、阳光、各行威胁。"""
        self._start()
        from . import scene
        self._refresh_full_observation()
        out = scene.summary(self)
        out["status"] = self.session.status_line(self.budget_key)
        return out

    # ------------------------------------------------------------ 存档

    def log(self) -> dict:
        return dict(schema=SCHEMA, task_id=self.task_id, seed=self.seed, deck=self.deck,
                    deck_attempt_seeds=self.deck_attempt_seeds,
                    run_id=self.run_id, cycle=self.cycle, workflow_layer=self.workflow_layer,
                    note=self.note, reason=self.deck_reason, strategy=self.strategy, deck_review=self.deck_review,
                    abandon_reason=self.abandon_reason, attempt_id=self.attempt_id, tick=self.tick, ops=self.ops,
                    frames=self.frames, totals=self.totals,
                    robustness=self.robustness_report, manual_patch=bool(self.manual_patch),
                    # drill / scenario / resume 局的任务定义在源存档里，不在任务清单里；
                    # 记下源存档，load() 走 from_archive 同一条路径恢复（11-05 交付验收发现）。
                    archive=getattr(self, "_source_archive", None),
                    archive_decision=getattr(self, "_source_decision", None),
                    resume_item=(dict(self.resume_source, id=self.resume_id)
                                 if self.resume_source.get("rewind_count") else None),
                    scenario_meta=self.scenario_meta, future_seed=self.future_seed)

    def save(self, path) -> Path:
        self._require_primary("save()")
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
        if doc.get("archive"):
            game = cls.from_archive(doc["archive"], decision=doc.get("archive_decision"),
                                    resume_item=doc.get("resume_item"),
                                    strategy=doc.get("strategy", ""), **kwargs)
            game.ops, game.frames, game.tick = doc.get("ops") or [], doc.get("frames") or [], doc.get("tick")
            game._resume_prefix_pending = False
            game.attempt_id = doc.get("attempt_id") or game.attempt_id
            game.scenario_meta = doc.get("scenario_meta") or game.scenario_meta
            game.future_seed = doc.get("future_seed") or game.future_seed
            game.note, game.abandon_reason = doc.get("note", game.note), doc.get("abandon_reason")
            game.terminal = game.abandon_reason is not None
            game._restore_archive_claim()
            return game
        return cls(doc["task_id"], seed=doc["seed"], ops=doc.get("ops") or [],
                   frames=doc.get("frames") or [], tick=doc.get("tick"),
                   note=doc.get("note", ""), robustness=doc.get("robustness"),
                   deck=doc.get("deck"), reason=doc.get("reason", ""), strategy=doc.get("strategy", ""),
                   deck_review=doc.get("deck_review", ""), abandon_reason=doc.get("abandon_reason"), attempt_id=doc.get("attempt_id"),
                   run_id=doc.get("run_id"), cycle=doc.get("cycle"), workflow_layer=doc.get("workflow_layer"),
                   manual_patch=bool(doc.get("manual_patch", False)),
                   deck_attempt_seeds=doc.get("deck_attempt_seeds"),
                   _gate=False, _restore_claim=True, **kwargs)

    @classmethod
    def load_archive(cls, directory, **kwargs) -> "Game":
        """恢复已导出的局，使用存档中的任务定义（旧卡组不受现行任务变动影响）。"""
        return cls.from_archive(directory, **kwargs)

    @classmethod
    def from_archive(cls, directory, decision=None, *, resume_item=None, **kwargs) -> "Game":
        """从存档恢复；decision=N 恢复成该决策执行前的局面（1 = 初始状态）。

        只截断已有的操作日志，再由 Game._start 用现有重放路径走到那里，不另写一套
        游戏逻辑（AGENTS.md §6.6）。之后的 plant / spawn / set_sun 会接在这段前缀后。
        """
        archive = Path(directory)
        meta = read_meta(archive)
        ops = [json.loads(line) for line in (archive / "ticks.jsonl").read_text(encoding="utf-8").splitlines()]
        from pvz_frame_store import frame_path, read_frames
        frames = read_frames(frame_path(archive))
        tick = meta["outcome"]["final_tick"]
        if decision is not None:
            if type(decision) is not int or not 1 <= decision <= len(frames):
                raise ValueError("decision must be an integer between 1 and %d" % len(frames))
            if "operation" not in frames[decision - 1]:
                raise ValueError("archive has no operation column; decision restore is unavailable")
            stop = int(frames[decision - 1]["operation"])
            if not 0 <= stop <= len(ops):
                raise ValueError("archive operation index %d is outside its %d operations" % (stop, len(ops)))
            ops = ops[:stop]
            frames = frames[:decision]
            tick = int(ops[-1]["end_tick"]) if ops else int(frames[0].get("tick", 0))
        if resume_item is not None:
            tick = resume_item["cutoff_tick"]
            if type(tick) is not int or not 0 <= tick <= int(meta["outcome"]["final_tick"]):
                raise ValueError("回退切点必须在原存档前缀内")
            ops, frames = _truncate_ops(ops, tick), []
        task = meta["task"]
        kwargs.setdefault("_gate", False)
        kwargs.setdefault("_restore_claim", True)
        game = cls(task["id"], seed=task["seed"], ops=ops, frames=frames,
                   tick=tick, note=meta.get("note", ""),
                   robustness=meta.get("robustness"), manual_patch=meta.get("manual_patch", False),
                   deck=task["deck"], reason=meta.get("deck_reason", ""),
                   strategy=kwargs.pop("strategy", meta.get("strategy", "")),
                   deck_review=meta.get("deck_review", ""), abandon_reason=meta.get("abandon_reason"), attempt_id=meta.get("attempt_id"),
                   cycle=meta.get("cycle"), workflow_layer=meta.get("workflow_layer"),
                   deck_attempt_seeds=meta.get("deck_attempt_seeds"),
                   _definition={**task, **meta['environment_task'], 'task_id': task['id'],
                                'deck': task['deck'], 'seeds': [task['seed']],
                                'layer': meta.get('workflow_layer', 'A')}, **kwargs)
        game.run_id = meta.get("run_id") or game.run_id
        game._source_archive, game._source_decision = str(archive.resolve()), decision
        # task220：中盘接回项用 resume_id 认领（同一生产任务族下有多条不同起点的存档）。
        game.resume_id = meta.get("resume_id") or None
        game.resume_source = dict(meta.get("resume") or {})
        if resume_item is not None:
            game.resume_id = resume_item["id"]
            game.resume_source = dict(resume_item)
            game.resume_source["prefix_operations"] = len(ops)
            game.abandon_reason, game.terminal = None, False
            game.attempt_id = uuid.uuid4().hex
            game.robustness_report, game.manual_patch = None, False
            game._resume_prefix_pending = True
        game.workflow_layer = meta.get("workflow_layer") or game.workflow_layer
        # task238：题面随存档走，导入/重放不再另读配置。
        game.drill_meta = dict(meta.get("drill") or task.get("drill_start") or {}) or None
        game.scenario_meta = dict(meta.get("scenario") or {}) or None
        game.future_seed = ((game.drill_meta or {}).get("future_seed")
                            or (game.scenario_meta or {}).get("future_seed"))
        game._root_operations = len(ops)
        game._root_tick = int(tick)
        from pvz_env import PlayerProfileContext
        config = dict(meta["environment_task"])
        config["profile"] = PlayerProfileContext(**config["profile"])
        config["preplanted"] = tuple(tuple(p) for p in config.get("preplanted", ()))
        game.spec = TaskSpec(**config)
        game.deck = list(task["deck"])
        if kwargs["_restore_claim"]:
            game._restore_archive_claim()
        return game

    def export_archive(self, directory, *, note=None) -> Path:
        """导出统一版本存档（ticks.jsonl / frames.npz / meta.json）。

        导出的存档是引擎真正接受的动作序列，可直接 load_archive 继续中盘打。
        返回 archives 中的正式路径；成功后删除 out 内的本次临时运行目录。
        """
        self._require_primary("export_archive()")
        if self.workflow_layer == "decks":
            self._start()
        # task139：缺思路 / 缺评价的局不能进下一层，所以在写盘前就拦下来（不用重打）。
        from itertools import islice
        from .delivery import archive_requirements
        opening = [] if self.spec.conveyor_start == "original" else list(islice((self.deck[op["action"]["packet"]] for op in self.ops
                               if op.get("ok") and op["action"].get("type") == "plant"), 6))
        archive_requirements(self.task_id, self.strategy, self.deck_review,
                             abandoned=self.abandon_reason is not None, layer=self.workflow_layer,
                             opening=opening, cycle=self.cycle, deck=self.deck,
                             reason=self.deck_reason, note=self.note if note is None else note,
                             result=(self.obs or {}).get("result"))
        if self.workflow_layer == "D" and self.strategy != self._declared_strategy:
            raise ValueError("D 层打法在开局时固定，不能在导出时改名")
        self._start()
        assert self.env is not None and self.obs is not None and self.initial_obs is not None
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=False)
        with (target / "ticks.jsonl").open("w", encoding="utf-8") as tape:
            for op in self.ops:
                tape.write(json.dumps(dict(action=op["action"], tick=op["tick"],
                                           end_tick=op["end_tick"], ok=op["ok"]),
                                      separators=(",", ":")) + "\n")
        from pvz_frame_store import write_frames
        write_frames(target / "frames.npz", self.frames, cache=False)
        episode = self.env.episode or {}
        from .runtime import environment_fingerprint
        result = int(self.obs.get("result") or 0)
        roster = (self.initial_obs.get("loadout_context") or {}).get("zombie_roster")
        attempts = {}
        if self.workflow_layer == "decks":
            from .decks import attempt_seeds
            from .delivery import _read_progress, output_dir
            self.deck_attempt_seeds = attempt_seeds(_read_progress(output_dir())["archives"], self.task_id, self.deck,
                                                    self.seed, self.deck_attempt_seeds)
            attempts = dict(deck_seed_attempts=len(self.deck_attempt_seeds),
                            deck_attempt_seeds=self.deck_attempt_seeds)
        if self.workflow_layer == "drill":
            judged = self.drill_result()
            if not judged["reached"] and self.abandon_reason is None:
                raise ValueError("还没到 drill 视界（%d ticks）：先 wait 到视界再导出"
                                 % int((self.drill_meta or {}).get("horizon") or 900))
            self.drill_meta = dict(self.drill_meta or {}, **judged)
            self.drill_meta.setdefault("id", self.task_id)
            if self.future_seed is not None:
                self.drill_meta["future_seed"] = int(self.future_seed)
        elif self.workflow_layer == "scenario":
            if not self.scenario_meta:
                raise ValueError("scenario 导出前先调用 declare_scenario(knowledge, horizon, targets)")
            from drill_start import horizon_reached
            judged = self._judge(self.scenario_meta)
            if not horizon_reached(self.obs, judged["root_tick"],
                                   int(self.scenario_meta.get("horizon") or 900)):
                raise ValueError("还没到 scenario 视界（%d ticks）：先 wait 到视界再导出"
                                 % int(self.scenario_meta.get("horizon") or 900))
            baseline = self.scenario_meta.get("baseline") or {}
            if baseline.get("success"):
                raise ValueError("这道出题“什么都不做”也成功，威胁不够真；换更强的注入或更长的视界")
            self.scenario_meta = dict(self.scenario_meta, **judged)
            self.scenario_meta.setdefault("id", self.task_id)
            if self.future_seed is not None:
                self.scenario_meta["future_seed"] = int(self.future_seed)
        meta = dict(
            **attempts,
            schema=ARCHIVE_SCHEMA, label_source=LABEL_SOURCE, cycle=self.cycle,
            workflow_layer=self.workflow_layer, opening=opening,
            **({"resume_id": self.resume_id, "resume": dict(self.resume_source)}
               if self.resume_id else {}),
            **({"drill": dict(self.drill_meta or {})} if self.workflow_layer == "drill" else {}),
            **({"scenario": dict(self.scenario_meta or {})} if self.workflow_layer == "scenario" else {}),
            run_id=self.run_id, environment_fingerprint=dict(environment_fingerprint()),
            note=self.note if note is None else note,
            deck_reason=self.deck_reason, strategy=self.strategy, deck_review=self.deck_review,
            abandoned=self.abandon_reason is not None, abandon_reason=self.abandon_reason, attempt_id=self.attempt_id,
            task=dict(id=self.task_id, level=self.spec.level, seed=self.spec.seed,
                      deck=list(self.deck), policy="llm", playthrough=self.spec.playthrough,
                      zombie_count_multiplier=self.spec.zombie_count_multiplier,
                      wave_cap=self.spec.wave_cap,
                      preplanted=[list(p) for p in self.spec.preplanted]),
            environment_task=self._environment_task(),
            engine_sha=episode.get("source_revision"), engine_dirty=episode.get("source_dirty"),
            context=dict(level=self.spec.level, multiplier=self.spec.zombie_count_multiplier,
                         roster=roster),
            # task81d：稳健性 = 每个检查点 k 个新未来里的胜局数 + 打法函数源码哈希。
            # manual_patch=True 表示这局用 rewind 手工补救过；稳健性按补救前的闭环打法算。
            robustness=self.robustness_report, manual_patch=bool(self.manual_patch),
            sun_collection="environment_auto", verified=False,
            frame_count=len(self.frames), operations=len(self.ops), totals=self.totals,
            outcome=dict(result=result, won=result == 1,
                         reason=self.abandon_reason if self.abandon_reason is not None else ("通关" if result == 1 else ("僵尸进屋" if self.obs.get("terminal") else "中途结束")),
                         abandoned=self.abandon_reason is not None,
                         final_wave=self.obs.get("wave"), wave_count=self.obs.get("wave_count"),
                         final_tick=self.obs.get("tick"), actions=max(len(self.frames) - 1, 0),
                         plants_eaten=self.totals.get("plants_eaten", 0),
                         plants_eaten_free=self.totals.get("plants_eaten_free", 0)),
        )
        write_meta(target, meta)
        from .delivery import record_archive, output_dir
        document = record_archive(target, session=self.session)
        stored = output_dir() / document["archives"][-1]["archive"]
        # 仅清理本次在 out 内新建的导出运行目录；正式存档继续可加载。
        if target.resolve() != stored.resolve() and output_dir() in target.resolve().parents:
            import shutil
            shutil.rmtree(target)
            target = stored
        self._release_claim()
        return target

    def _restore_archive_claim(self) -> None:
        block = self.drill_meta or self.scenario_meta or {}
        if self.workflow_layer == "resume" or (
                self.workflow_layer in ("drill", "scenario") and any(
                    op["action"].get("type") == "reseed"
                    for op in self.ops[int(block.get("prefix_operations") or 0):])):
            self._hold_claim()

    def _hold_claim(self) -> None:
        if self._internal:
            return
        from .workflow import hold
        # 短题在 reseed 后才知道派发种子，题面里的默认未来不能拿来认领。
        seed = self.future_seed if self.workflow_layer in ("drill", "scenario") else self.seed
        self._claimed = (self.resume_id or self.task_id, seed)
        hold(*self._claimed)

    def _release_claim(self) -> None:
        if not self._claimed:
            return
        from .workflow import release
        release(*self._claimed)
        self._claimed = False

    def _deliver(self) -> None:
        """稳健性完成即导出当前局面并发布交付；内部试打不交付。"""
        from .delivery import output_dir
        import tempfile
        import shutil
        root = output_dir()
        root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".capture-", dir=root))
        try:
            self.export_archive(temporary / "archive")
        finally:
            shutil.rmtree(temporary)

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

    def abandon(self, *, reason):
        """保存真实操作序列和放弃理由，自动交付后释放本局引擎。"""
        self._require_primary("abandon()")
        if not reason.strip():
            raise ValueError("请填写放弃理由")
        self._start()
        if self.terminal:
            raise RuntimeError("本局已结束")
        self.abandon_reason = reason
        self.terminal = True
        try:
            self._deliver()
        finally:
            self.close()

    def close(self) -> None:
        self._release_claim()
        if self.env is not None:
            try:
                self._persist_best()
            finally:
                self.env.close()
                self.env = None
        if self._is_branch:
            self.session.note_branch(self.budget_key, -1)
            self._is_branch = False
        if self in _LIVE_GAMES:
            _LIVE_GAMES.remove(self)

    def __enter__(self) -> "Game":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
