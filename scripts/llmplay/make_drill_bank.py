"""Build the task238 drill/scenario asset bank outside git (build-time only).

The package only carries KB manifests; the prefix archives are rebuilt here from the
production archives with the existing numerical archive writer (AGENTS §6.6).  One
drill/scenario prefix is the operation tape up to the root decision plus the root
frame only -- the same layout as the task220 resume items.

  # fog drills (run on the production desktop, roots are read-only)
  python3 scripts/llmplay/make_drill_bank.py drills \
      --bank artifacts/task235/bank_all.json \
      --results artifacts/task235/results_all.jsonl \
      --root framesv2=/home/newbiexvwu/PvZ-search/artifacts/student/frames_v2 \
      --out ~/pvz-logs/task238/assets --catalog ~/pvz-logs/task238/drill235.json

  # roof/night/pool drills already referenced by task234 (run on the Mac)
  python3 scripts/llmplay/make_drill_bank.py drills --source task234 \
      --bank configs/task234_drills.json --root main=... --root gen=... \
      --out ~/pvz-logs/task238/assets --catalog ~/pvz-logs/task238/drill234.json

  # new drills from production resume prefixes + scenario base boards
  python3 scripts/llmplay/make_drill_bank.py resume-drills \
      --resume-dir ~/pvz-mac-collect/pkg238-ref/llmplay_pkg/resume \
      --catalog configs/llmplay_resume.json --out ... --catalog-out ...
  python3 scripts/llmplay/make_drill_bank.py scenarios ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]
from pvz_episode_archive import read_meta, write_meta  # noqa: E402
from pvz_frame_store import frame_path, read_frames, write_frames  # noqa: E402

COUNTER_TYPES = {
    3: "polevaulter", 7: "football", 10: "duckytube", 11: "snorkel", 12: "zamboni",
    13: "bobsled", 14: "dolphin", 15: "jack", 16: "balloon", 17: "digger",
    18: "pogo", 19: "yeti", 21: "ladder", 23: "gargantuar", 32: "redeye_gargantuar",
}
TERRAIN = ((1, 10, "day"), (11, 20, "night"), (21, 30, "pool"), (31, 40, "fog"),
           (41, 49, "roof"), (50, 50, "boss"))
KNOWLEDGE = {
    "balloon": "气球：三叶草/仙人掌/香蒲，别让它越过前排",
    "digger": "矿工：后排也要有火力/磁力菇，不能只守前排",
    "dolphin": "海豚：水路前段火力与睡莲覆盖",
    "football": "橄榄球：高坚果/地刺/寒冰拖住，别让它贴脸",
    "jack": "小丑：磁力菇吸盒/窝瓜秒杀，别让它贴脸",
    "pogo": "跳跳：高坚果/地刺/磁力菇，别让它连续跳过",
    "ladder": "梯子：磁力菇/窝瓜，别让梯子搭上高坚果",
    "zamboni": "冰车：地刺/窝瓜，冰道会加速",
    "bungee": "蹦极：关键植物被偷前清掉或留冗余",
    "catapult": "投石车：后排输出，注意它砸后排",
    "gargantuar": "巨人：地刺王/窝瓜/寒冰，拖住再集火",
    "snorkel": "潜水：水路火力覆盖",
    "duckytube": "鸭子泳圈：水路火力覆盖",
    "polevaulter": "撑杆：高坚果/地刺，别让它跳过前排",
}
SCENARIO_KNOWLEDGE = {
    "day": ["气球", "小丑", "矿工", "橄榄球", "跳跳", "撑杆", "冰车"],
    "night": ["气球", "小丑", "矿工", "橄榄球", "跳跳", "梯子", "巨人"],
    "pool": ["气球", "小丑", "矿工", "海豚", "潜水", "橄榄球", "冰车"],
    "fog": ["气球", "小丑", "矿工", "海豚", "橄榄球", "跳跳", "梯子"],
    "roof": ["气球", "小丑", "矿工", "投石车", "蹦极", "巨人", "梯子"],
}


def terrain_of(level: int, scene=None) -> str:
    for low, high, name in TERRAIN:
        if low <= level <= high:
            return name
    return "other"


def sha(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:8]


def load_ops(archive: Path):
    return [json.loads(line) for line in (archive / "ticks.jsonl").read_text().splitlines() if line.strip()]


def root_mowers(frame) -> int:
    return sum(1 for row, state in frame.get("mowers") or [] if int(state) == 1)


def frame_targets(frame, ids, limit=4):
    wanted = {int(value) for value in ids}
    return [dict(id=int(z[9]), row=int(z[0]), type=int(z[1]), x=round(float(z[2]), 1))
            for z in frame.get("zombies") or []
            if z[5] and int(z[9]) in wanted][:limit]


def ops_from_frames(frames, stop: int) -> list:
    """Collection archives have no ticks.jsonl; the frame actions are the prefix."""
    out = []
    for index in range(1, min(stop, len(frames) - 1) + 1):
        action = frames[index].get("action") or frames[index].get("input_action")
        start = int(frames[index - 1]["tick"])
        end = int(frames[index]["tick"])
        if action is None:
            action = dict(type="wait", ticks=max(0, end - start))
        else:
            # Collection frames carry recorder-only keys (`until`, `label`, ...); the
            # replay interface only accepts the action fields the engine implements.
            action = {key: value for key, value in action.items()
                      if key in ("type", "packet", "row", "col", "ticks", "x", "y",
                                 "zombie_type", "sun", "slot", "target_col", "seed",
                                 "wave", "countdown")}
        out.append(dict(action=action, tick=start, end_tick=end,
                        ok=not bool(frames[index].get("action_rejected"))))
    return out


def prefix_ops(ops):
    """采集 WAIT 会遇事件提前结束，前缀重放只等存档中实际经过的 tick。"""
    return [dict(op, action=dict(op["action"], ticks=int(op["end_tick"]) - int(op["tick"])))
            if op["action"].get("type") == "wait" else op for op in ops]


def write_prefix(archive: Path, target: Path, *, task_id: str, layer: str, block: dict,
                 knowledge: str, horizon: int, future_seeds, source: dict, decision: int | None = None):
    meta = read_meta(archive)
    frames = read_frames(frame_path(archive))
    if decision is None:
        decision = len(frames)
    root = frames[min(max(decision - 1, 0), len(frames) - 1)]
    if (archive / "ticks.jsonl").is_file():
        ops = load_ops(archive)
        # Resume prefixes have one root frame without an operation column: the whole
        # tape is the prefix.  Full recordings expose the frame's operation number.
        stop = int(root["operation"]) if "operation" in root else len(ops)
        ops = ops[:stop]
    else:
        ops = ops_from_frames(frames, decision - 1)
    ops = prefix_ops(ops)
    from dataclasses import asdict
    from drill_start import archive_task
    deck, spec = archive_task(meta, frames)
    task, env = dict(meta.get("task") or {}), asdict(spec)
    level, seed, multiplier = spec.level, spec.seed, spec.zombie_count_multiplier
    task.update(id=task_id, level=level, seed=seed, deck=deck,
                zombie_count_multiplier=multiplier, policy="llm", playthrough=int(env.get("playthrough") or 2))
    task["drill_start"] = dict(block)
    env.update(level=level, seed=seed, playthrough=int(env.get("playthrough") or 2),
               zombie_count_multiplier=multiplier)
    outcome = dict(result=0, won=False, reason=f"{layer}_root", abandoned=False,
                   final_tick=int(root["tick"]), final_wave=int(root.get("wave") or 0),
                   wave_count=int(root.get("wave_count") or 0), actions=len(ops))
    document = dict(meta)
    document.pop("resume_id", None)
    document.pop("resume", None)
    document.update(schema="episode_archive_v2", schema_version=2, label_source=f"{layer}_prefix",
                    workflow_layer=layer, task=task, deck=deck, environment_task=env,
                    context=dict(meta.get("context") or {}, level=level, multiplier=multiplier),
                    outcome=outcome, frame_count=1, operations=len(ops), verified=False,
                    note=(f"{layer} prefix: {knowledge}"), strategy="", deck_reason=knowledge,
                    drill=dict(block, id=task_id, knowledge=knowledge, terrain=block.get("terrain"),
                               horizon=horizon, root_tick=int(root["tick"]),
                               prefix_operations=len(ops), mowers_start=root_mowers(root),
                               targets=list(block.get("targets") or []),
                               future_seeds=[int(value) for value in future_seeds or []],
                               future_seed=int((future_seeds or [0])[0]),
                               source=dict(source)))
    target.mkdir(parents=True, exist_ok=True)
    with (target / "ticks.jsonl").open("w", encoding="utf-8") as tape:
        for op in ops:
            tape.write(json.dumps(dict(action=op["action"], tick=op["tick"],
                                       end_tick=op["end_tick"], ok=op["ok"]),
                                  separators=(",", ":")) + "\n")
    write_frames(target / "frames.npz", [root], cache=False)
    write_meta(target, document)
    return dict(task_id=task_id, archive=str(target.name), level=level, terrain=block.get("terrain"),
                deck=deck, multiplier=multiplier, source_seed=seed, knowledge=knowledge,
                horizon=horizon, targets=list(block.get("targets") or []),
                future_seeds=[int(value) for value in future_seeds or []],
                priority=int(block.get("priority", 3)), type=block.get("type"),
                source=dict(source))


def drill_rates(path: Path) -> dict:
    grouped = defaultdict(lambda: [0, 0])
    if path and path.is_file():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            grouped[row["drill_id"]][0] += 1
            grouped[row["drill_id"]][1] += int(bool(row.get("success")))
    return {key: dict(games=value[0], wins=value[1],
                      rate=value[1] / value[0] if value[0] else None)
            for key, value in grouped.items()}


def priority_of(rate: dict) -> int:
    if not rate or not rate.get("games"):
        return 3
    if rate["wins"] == 0:
        return 0
    if rate["wins"] == rate["games"]:
        return 2
    return 1


def check_drill_prefix(archive: Path, *, executable=None, resource_dir=None) -> dict:
    """出题后只加载一次，与沙箱共用 Game 路径，修正可唯一匹配的旧实体 ID。"""
    from llmplay.game import Game
    from render_episode import _frame
    meta = read_meta(archive)
    block = meta["drill"]
    ops = load_ops(archive)
    normalized = prefix_ops(ops)
    if normalized != ops:
        with (archive / "ticks.jsonl").open("w", encoding="utf-8") as tape:
            for op in normalized:
                tape.write(json.dumps(op, separators=(",", ":")) + "\n")
    game = Game.load_archive(archive, executable=executable, resource_dir=resource_dir,
                             _internal=True)
    try:
        game._start()
        alive = {z["id"]: z for z in game.zombies()}
        missing = [t["id"] for t in block["targets"] if t["id"] not in alive]
        result = dict(missing=missing, repaired=0, wait_fixed=sum(a != b for a, b in zip(ops, normalized)),
                      status="ok", tick=game.tick,
                      recorded_tick=block["root_tick"])
        if game.tick != int(block["root_tick"]) or game.terminal:
            return dict(result, status="rejected", reason="起点 tick 不符或已终局")
        targets, used = [], set()
        for target in block["targets"]:
            # 源帧 x 保留一位小数。只修复同类型、同行、同位置的唯一实体，不能猜题。
            candidates = [z for z in alive.values() if z["id"] not in used
                          and z["type"] == target["type"] and z["row"] == target["row"]
                          and abs(z["x"] - target["x"]) <= 1.0]
            exact = [z for z in candidates if z["id"] == target["id"]]
            if exact:
                candidates = exact
            if len(candidates) != 1:
                return dict(result, status="rejected", reason="目标无法按类型、行和位置唯一匹配")
            matched = candidates[0]
            used.add(matched["id"])
            result["repaired"] += int(matched["id"] != target["id"])
            targets.append(dict(target, id=matched["id"], x=round(matched["x"], 1)))
        if not targets:
            return dict(result, status="rejected", reason="起点目标为空")
        block.update(targets=targets, root_tick=game.tick, mowers_start=game._mower_count())
        meta["task"]["drill_start"]["targets"] = targets
        # 保存真正加载到的根帧，题面与回放起点使用同一个棋盘。
        root = _frame(game.env.privileged_state())
        root["operation"] = len(game.ops)
        write_frames(archive / "frames.npz", [root], cache=False)
        write_meta(archive, meta)
        return dict(result, targets=targets)
    finally:
        game.close()


def _check_item(item, args, results):
    archive = Path(args.out) / "drill" / Path(item["archive"]).name
    try:
        result = check_drill_prefix(archive, executable=args.executable,
                                    resource_dir=args.resource_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        result = dict(status="rejected", reason=str(exc), missing=[], repaired=0)
    results.append(dict(task_id=item["task_id"], **result))
    if result["status"] != "ok":
        return None
    return dict(item, targets=result["targets"])


def check_summary(results):
    return dict(checked=len(results), missing=sum(bool(r["missing"]) for r in results),
                wait_fixed=sum(bool(r.get("wait_fixed")) for r in results),
                repaired=sum(bool(r["repaired"]) and r["status"] == "ok" for r in results),
                rejected=sum(r["status"] != "ok" for r in results),
                rejected_fraction=sum(r["status"] != "ok" for r in results) / max(1, len(results)))


def check_drills(args):
    """核对现成题库，不需要重新挖题，不改变题号及未来种子。"""
    catalog = json.loads(args.catalog.read_text())
    results, kept = [], []
    for item in catalog["items"]:
        checked = _check_item(item, args, results)
        if checked is not None:
            kept.append(checked)
        if len(results) % 50 == 0:
            print(json.dumps(check_summary(results)), flush=True)
    catalog.update(items=kept, item_count=len(kept))
    args.checked_catalog.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + "\n")
    summary = check_summary(results)
    args.report.write_text(json.dumps(dict(summary=summary, items=results), ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(summary), flush=True)
    return 0


def build_drills(args) -> int:
    bank = json.loads(args.bank.read_text())
    drills = bank.get("drills") or bank.get("tasks") or []
    rates = drill_rates(args.results)
    roots = {}
    for entry in args.root:
        name, _, path = entry.partition("=")
        roots[name] = Path(path).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    catalog, seen, checks = [], set(), []
    for drill in drills:
        if drill.get("id") in seen:
            continue
        source_root = drill.get("source_root") or "main"
        archive = Path(drill.get("archive") or "")
        if not archive.is_absolute():
            root = roots.get(source_root)
            if root is None:
                continue
            archive = root / archive
        if not frame_path(archive).is_file():
            continue
        level = int(drill.get("level") or (read_meta(archive).get("task") or {}).get("level") or 0)
        terrain = drill.get("terrain") or terrain_of(level)
        trigger = drill.get("trigger") or {}
        name = trigger.get("name") or "normal"
        targets = list(drill.get("targets") or [])
        if not targets:
            continue
        task_id = "drill238_%s_%s" % (terrain, sha(str(drill.get("id"))))
        future_seeds = [args.seed_base + len(checks) * 3 + index for index in range(3)]
        block = dict(targets=targets, terrain=terrain, type=name,
                     criterion=drill.get("criterion") or dict(
                         threats_neutralized=True, mower_unchanged=True, house_unchanged=True),
                     priority=priority_of(rates.get(drill.get("id"))),
                     teacher=rates.get(drill.get("id")) or {},
                     source=dict(id=drill.get("id"), root=source_root,
                                 archive=drill.get("archive")),
                     decision=int(drill.get("decision") or 1))
        written = write_prefix(archive, out / "drill" / task_id, task_id=task_id, layer="drill",
                               block=block, knowledge=KNOWLEDGE.get(name, name),
                               horizon=int(drill.get("horizon") or 900),
                               future_seeds=future_seeds, source=block["source"],
                               decision=int(block["decision"]))
        written["task_id"] = task_id
        written["priority"] = block["priority"]
        written["suits"] = name
        seen.add(drill.get("id"))
        written = _check_item(written, args, checks)
        if written is not None:
            catalog.append(written)
    document = dict(schema="llmplay_drill_catalog_v1", task="task238",
                    item_count=len(catalog), items=catalog)
    args.catalog.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(dict(items=len(catalog), check=check_summary(checks), by_terrain=Counter(x["terrain"] for x in catalog),
                          by_type=Counter(x["type"] for x in catalog),
                          by_priority=Counter(x["priority"] for x in catalog)),
                     ensure_ascii=False, indent=1))
    return 0


def _resume_items(catalog_path: Path, resume_dir: Path):
    document = json.loads(catalog_path.read_text())
    for item in document.get("items") or []:
        archive = resume_dir / item["archive"]
        if (archive / "meta.json").is_file():
            yield item, archive


def build_resume_drills(args) -> int:
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    per_type = Counter()
    catalog, checks = [], []
    for item, archive in _resume_items(args.resume_catalog, args.resume_dir):
        meta = read_meta(archive)
        frames = read_frames(frame_path(archive))
        if not frames:
            continue
        root = frames[-1]
        level = int((meta.get("task") or {}).get("level") or 0)
        terrain = terrain_of(level)
        if terrain not in args.terrain:
            continue
        threats = [dict(id=int(z[9]), row=int(z[0]), type=int(z[1]), x=round(float(z[2]), 1))
                   for z in root.get("zombies") or []
                   if z[5] and int(z[1]) in COUNTER_TYPES and float(z[2]) <= args.near_x]
        if not threats:
            continue
        groups = defaultdict(list)
        for threat in threats:
            groups[COUNTER_TYPES[threat["type"]]].append(threat)
        for name, targets in groups.items():
            if per_type[(terrain, name)] >= args.max_per_type:
                continue
            per_type[(terrain, name)] += 1
            task_id = "drill238_%s_%s_%02d" % (terrain, name, per_type[(terrain, name)])
            future_seeds = [args.seed_base + len(checks) * 3 + index for index in range(3)]
            block = dict(targets=targets[:3], terrain=terrain, type=name,
                         criterion=dict(threats_neutralized=True, mower_unchanged=True,
                                        house_unchanged=True), priority=1,
                         source=dict(id=item["id"], root="resume", archive=item["archive"]),
                         decision=1)
            written = write_prefix(archive, out / "drill" / task_id, task_id=task_id, layer="drill",
                                   block=block, knowledge=KNOWLEDGE.get(name, name),
                                   horizon=args.horizon, future_seeds=future_seeds,
                                   source=block["source"], decision=1)
            written["suits"] = name
            written = _check_item(written, args, checks)
            if written is not None:
                catalog.append(written)
            if len(catalog) >= args.limit:
                break
        if len(catalog) >= args.limit:
            break
    document = dict(schema="llmplay_drill_catalog_v1", task="task238",
                    item_count=len(catalog), items=catalog)
    args.catalog.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(dict(items=len(catalog), check=check_summary(checks), by_terrain=Counter(x["terrain"] for x in catalog),
                          by_type=Counter(x["type"] for x in catalog)), ensure_ascii=False, indent=1))
    return 0


def build_scenarios(args) -> int:
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    per_terrain = Counter()
    catalog = []
    for item, archive in _resume_items(args.resume_catalog, args.resume_dir):
        meta = read_meta(archive)
        frames = read_frames(frame_path(archive))
        if not frames:
            continue
        root = frames[-1]
        level = int((meta.get("task") or {}).get("level") or 0)
        terrain = terrain_of(level)
        if terrain not in args.terrain or per_terrain[terrain] >= args.per_terrain:
            continue
        # A scenario base must be a real mid-game board, not an empty opening frame.
        if int(root.get("tick") or 0) < args.min_tick or len(root.get("plants") or []) < args.min_plants:
            continue
        threats = [z for z in root.get("zombies") or []
                   if z[5] and int(z[1]) in COUNTER_TYPES and float(z[2]) <= 600.0]
        if threats:
            continue
        task_id = "scenario238_%s_%03d" % (terrain, per_terrain[terrain] + 1)
        per_terrain[terrain] += 1
        future_seeds = [args.seed_base + per_terrain[terrain] * 2 + index for index in range(2)]
        block = dict(id=task_id, base_id=item["id"], terrain=terrain, horizon=args.horizon,
                     targets=[], criterion=dict(threats_neutralized=True, mower_unchanged=True,
                                                house_unchanged=True), prefix_operations=len(root),
                     source=dict(id=item["id"], root="resume", archive=item["archive"]))
        # build prefix with a neutral drill block then replace it with scenario metadata
        written = write_prefix(archive, out / "scenario" / task_id, task_id=task_id, layer="scenario",
                               block=dict(terrain=terrain, type="scenario", targets=[],
                                          criterion=block["criterion"], priority=1),
                               knowledge=SCENARIO_KNOWLEDGE.get(terrain, ["对策型僵尸"])[0],
                               horizon=args.horizon, future_seeds=future_seeds,
                               source=block["source"], decision=1)
        target_meta = out / "scenario" / task_id / "meta.json"
        document = json.loads(target_meta.read_text())
        document.pop("drill", None)
        document.get("task", {}).pop("drill_start", None)
        document["scenario"] = block
        document["workflow_layer"] = "scenario"
        write_meta(target_meta.parent, document)
        catalog.append(dict(task_id=task_id, archive=written["archive"], level=level,
                            terrain=terrain, deck=written["deck"], multiplier=written["multiplier"],
                            source_seed=written["source_seed"], base_id=item["id"],
                            horizon=args.horizon, future_seeds=future_seeds,
                            knowledge_candidates=SCENARIO_KNOWLEDGE.get(terrain, ["对策型僵尸"])))
    document = dict(schema="llmplay_scenario_catalog_v1", task="task238",
                    item_count=len(catalog), items=catalog)
    args.catalog.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(dict(items=len(catalog), by_terrain=Counter(x["terrain"] for x in catalog)),
                     ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    drill = sub.add_parser("drills")
    drill.add_argument("--bank", type=Path, required=True)
    drill.add_argument("--results", type=Path)
    drill.add_argument("--root", action="append", default=[], metavar="NAME=PATH")
    drill.add_argument("--out", type=Path, required=True)
    drill.add_argument("--catalog", type=Path, required=True)
    drill.add_argument("--seed-base", type=int, default=238000000)
    drill.set_defaults(run=build_drills)
    resume = sub.add_parser("resume-drills")
    resume.add_argument("--resume-dir", type=Path, required=True)
    resume.add_argument("--resume-catalog", type=Path, required=True)
    resume.add_argument("--out", type=Path, required=True)
    resume.add_argument("--catalog", type=Path, required=True)
    resume.add_argument("--terrain", nargs="+", default=["day", "night", "pool", "fog", "roof"])
    resume.add_argument("--near-x", type=float, default=460.0)
    resume.add_argument("--max-per-type", type=int, default=10)
    resume.add_argument("--limit", type=int, default=400)
    resume.add_argument("--horizon", type=int, default=1200)
    resume.add_argument("--seed-base", type=int, default=238500000)
    resume.set_defaults(run=build_resume_drills)
    scenario = sub.add_parser("scenarios")
    scenario.add_argument("--resume-dir", type=Path, required=True)
    scenario.add_argument("--resume-catalog", type=Path, required=True)
    scenario.add_argument("--out", type=Path, required=True)
    scenario.add_argument("--catalog", type=Path, required=True)
    scenario.add_argument("--terrain", nargs="+", default=["day", "night", "pool", "fog", "roof"])
    scenario.add_argument("--per-terrain", type=int, default=60)
    scenario.add_argument("--min-tick", type=int, default=5000)
    scenario.add_argument("--min-plants", type=int, default=8)
    scenario.add_argument("--horizon", type=int, default=1200)
    scenario.add_argument("--seed-base", type=int, default=239000000)
    scenario.set_defaults(run=build_scenarios)
    check = sub.add_parser("check-drills")
    check.add_argument("--catalog", type=Path, required=True)
    check.add_argument("--out", type=Path, required=True, help="含 drill/ 的前缀根目录")
    check.add_argument("--checked-catalog", type=Path, required=True)
    check.add_argument("--report", type=Path, required=True)
    check.set_defaults(run=check_drills)
    for command in (drill, resume, check):
        command.add_argument("--executable", type=Path)
        command.add_argument("--resource-dir", type=Path)
    args = parser.parse_args()
    state = args.out.expanduser() / ".drill-check"
    os.environ.setdefault("LLMPLAY_OUT", str(state / "out"))
    os.environ.setdefault("PVZ_LLMPLAY_SESSION", str(state / "budget.json"))
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
