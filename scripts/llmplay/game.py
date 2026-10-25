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

import hashlib
import json
import os
import sys
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

SCHEMA = "llmplay_log_v1"
ARCHIVE_SCHEMA = "episode_archive_v1"
LABEL_SOURCE = "llm_gpt_pro"
FRAME_INTERVAL = 60          # 与 human_demo 的帧节奏一致（每 60 tick 一帧）
DEFAULT_MANIFEST = ROOT / "configs" / "human_demo_tasks.json"


def _default_resource_dir() -> Path:
    for candidate in (ROOT / "res", Path.home() / ".cache" / "pvz-research-resources"):
        if (candidate / "main.pak").is_file():
            return candidate
    return ROOT / "res"


def _default_executable() -> Path:
    return ROOT / "build" / ("pvz-portable.exe" if os.name == "nt" else "pvz-portable")


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


class Game:
    """一局。构造时只读任务，第一次用引擎时才真正 reset + 重放。"""

    def __init__(self, task_id: str, *, seed=None, manifest=None, executable=None,
                 resource_dir=None, ops=None, frames=None, tick=None, note: str = ""):
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

    # ------------------------------------------------------------ 引擎生命周期

    def _start(self) -> None:
        if self.env is not None:
            return
        target = self.tick          # _reset_engine 会把 self.tick 归零，先记住
        self.env = PvZEnv(resource_dir=self.resource_dir, executable=self.executable, headless=True)
        self._reset_engine()
        if self.ops or target:
            self._replay_to(target)

    def _reset_engine(self) -> None:
        assert self.env is not None
        self.initial_obs, _ = self.env.reset(deck=self.deck, task=self.spec)
        # 开局哈希必须紧跟在 reset 之后取：之后任何一条命令都会覆盖 _last_response。
        self.initial_hash = (self.env._last_response.get("state_hash")
                             or self.env.state_hash())
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

    def _replay_to(self, target: int) -> None:
        """reset 后重放到 tick target，并把 ops / frames 截断到那一点。

        操作日志是唯一真相：重放结束后内存里的 ops 就是"当前状态"的完整描述，
        所以这里同时改写 ops（跨越目标 tick 的 wait 会被切成两段）。
        """
        assert self.env is not None
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
            start_ops = cached["n"]
        else:
            self._reset_engine()
            self._cache = None

        kept = []
        for op in self.ops[start_ops:]:
            action = op["action"]
            if op["end_tick"] <= target:
                self._step_raw(action)
                kept.append(op)
                continue
            if action.get("type") == "wait" and op["tick"] < target:
                partial = target - op["tick"]
                self._step_raw({"type": "wait", "ticks": partial})
                kept.append(dict(op, action={"type": "wait", "ticks": partial}, end_tick=target))
            break
        self.ops = self.ops[:start_ops] + kept
        self.frames = [f for f in self.frames if f.get("operation", 0) <= len(self.ops)]
        self.tick = int(self.obs["tick"])

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
        before = self.tick
        ok = self._step_raw(action)
        # 精简观测的响应不带 state_hash（那是给搜索用的轻量路径），存档要逐操作哈希，
        # 所以缺了就单独问一次。一次命令的代价，比关掉精简观测便宜得多。
        state_hash = self.env._last_response.get("state_hash") or self.env.state_hash()
        self.ops.append(dict(action=action, tick=before, end_tick=self.tick, ok=ok,
                             state_hash=state_hash))
        self._maybe_frame(action, ok)
        return ok

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
        self._replay_to(target)

    def branch(self) -> "Game":
        """复制一份独立的局（同一任务、同一操作序列）。各自一个引擎进程。"""
        self._start()
        return Game(self.task_id, seed=self.seed, manifest=self.manifest,
                    executable=self.executable, resource_dir=self.resource_dir,
                    ops=[dict(op) for op in self.ops],
                    frames=[dict(f) for f in self.frames],
                    tick=self.tick, note=self.note)

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
                           saw_mowers=self.saw_mowers)

    # ------------------------------------------------------------ 读局面

    def observe(self, level: str = "brief") -> str:
        """文字局面。brief 一屏；full 加上逐路明细与僵尸密度。"""
        self._start()
        from . import scene
        return scene.render(self, level=level)

    def legal(self) -> dict:
        """当前合法动作（引擎判定）：plants 是 (卡槽, 植物, 行, 列)，shovels 是可铲的格。"""
        self._start()
        from . import scene
        return scene.legal(self)

    def summary(self) -> dict:
        """当前波、最远存活波、剩余小推车、阳光、各行威胁。"""
        self._start()
        from . import scene
        return scene.summary(self)

    # ------------------------------------------------------------ 存档

    def log(self) -> dict:
        return dict(schema=SCHEMA, task_id=self.task_id, seed=self.seed, deck=self.deck,
                    note=self.note, tick=self.tick, ops=self.ops,
                    frames=self.frames, totals=self.totals)

    def save(self, path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.log(), ensure_ascii=False, separators=(",", ":")) + "\n",
                          encoding="utf-8")
        return target

    @classmethod
    def load(cls, path, **kwargs) -> "Game":
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        if doc.get("schema") != SCHEMA:
            raise ValueError("不是 %s 存档：%s" % (SCHEMA, path))
        return cls(doc["task_id"], seed=doc["seed"], ops=doc.get("ops") or [],
                   frames=doc.get("frames") or [], tick=doc.get("tick"),
                   note=doc.get("note", ""), **kwargs)

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
            executable_sha256=hashlib.sha256(Path(self.executable).read_bytes()).hexdigest(),
            resource_sha256=episode.get("resource_sha256"),
            properties_partner_sha256=episode.get("properties_partner_sha256"),
            context=dict(level=self.spec.level, multiplier=self.spec.zombie_count_multiplier,
                         roster=roster),
            hash_schema="portable_snapshot_v1_without_draw_statistics",
            initial_state_hash=self.initial_hash,
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
            self.env.close()
            self.env = None

    def __enter__(self) -> "Game":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
