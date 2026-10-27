"""task81e：自动交付、逐档验证续跑与五层进度。只使用标准库和包内引擎。

存档按内容寻址且不修改。progress/report/budget/out.zip 指向同一代目录，最后只原子
替换 current 链接；压缩或写盘失败时上一代四份文件仍完整可用。flock 串行化多进程更新。
"""
from __future__ import annotations

from itertools import combinations
import hashlib
import json
import os
import shutil
import tempfile
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from threading import BoundedSemaphore

from . import budget
from .runtime import output_path
from .game import (ROOT, ARCHIVE_SCHEMA, _default_executable, _default_resource_dir,
                   _ensure_executable, load_tasks, required_checkpoint)

FILES = ("meta.json", "ticks.jsonl", "frames.jsonl")
LAYERS = ("A", "B", "C", "D", "E")
REQUIRED_WINS = 5
REQUIRED_ROBUST = 3
_VERIFY_SLOTS = BoundedSemaphore(2)


def output_dir() -> Path:
    return output_path(ROOT)


def seed_order(task_id, run_id, seeds):
    """按运行ID散列排序，跨 Python/平台稳定，不依赖共享协调服务。"""
    return sorted(seeds, key=lambda seed: hashlib.sha256(
        (str(run_id) + ":" + task_id + ":" + str(seed)).encode("utf-8")).digest())


def ensure_run_id():
    root = output_dir()
    with _lock(root):
        document = _read_progress(root)
        if not document.get("run_id"):
            document = _publish(root, document["archives"], budget.session(), run_id=uuid.uuid4().hex)
        return document["run_id"]


@contextmanager
def _lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".delivery.lock").open("a+") as handle:
        if budget.fcntl is not None:
            budget.fcntl.flock(handle.fileno(), budget.fcntl.LOCK_EX)
        try:
            yield
        finally:
            if budget.fcntl is not None:
                budget.fcntl.flock(handle.fileno(), budget.fcntl.LOCK_UN)


def _read_progress(root):
    path = root / "progress.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"archives": []}


def _robust(meta):
    report = meta.get("robustness") or {}
    task = meta.get("task") or {}
    required = required_checkpoint(dict(level=task.get("level", 0)))
    cp = (report.get("checkpoints") or {}).get(str(required)) or {}
    return bool(report.get("policy_sha256") and cp.get("reached")
                and cp.get("k") == 8 and 5 <= int(cp.get("wins") or 0) <= 8)


def _entry(archive, relative, *, valid=True, error=None):
    meta = json.loads((archive / "meta.json").read_text(encoding="utf-8"))
    task, outcome = meta["task"], meta["outcome"]
    won = valid and not meta.get("abandoned") and outcome.get("result") == 1 and bool(outcome.get("won"))
    return dict(task_id=task["id"], seed=int(task["seed"]), result=outcome["result"],
                outcome=outcome, robustness=meta.get("robustness"),
                level=task["level"], deck=task["deck"], deck_reason=meta.get("deck_reason", ""),
                strategy=meta.get("strategy", ""), deck_review=meta.get("deck_review", ""),
                abandoned=bool(meta.get("abandoned")), abandon_reason=meta.get("abandon_reason"),
                archive=relative, manual_patch=bool(meta.get("manual_patch")),
                note=meta.get("note", ""), valid=valid, invalid_reason=error,
                won=won, robust=bool(won and _robust(meta)), run_id=meta.get("run_id"),
                environment_fingerprint=meta.get("environment_fingerprint"))


def _deck_progress(entries):
    """每套按植物集合去重、每个种子只记一次胜；全部尝试仍留在 archives。"""
    decks = {}
    for entry in entries:
        key = ",".join(map(str, sorted(entry.get("deck") or [])))
        group = decks.setdefault(key, dict(deck=sorted(entry.get("deck") or []),
                                          winning_seeds=set(), robust_seeds=set(), attempts=0,
                                          losses=0, strategy="", reason="", review=""))
        group["attempts"] += 1
        group["losses"] += not entry["won"]
        for target, source in (("strategy", "strategy"), ("reason", "deck_reason"), ("review", "deck_review")):
            if entry.get(source):
                group[target] = entry[source]
        if entry["won"]:
            group["winning_seeds"].add(entry["seed"])
        if entry["robust"]:
            group["robust_seeds"].add(entry["seed"])
    eligible = [key for key, group in decks.items() if len(group["winning_seeds"]) >= 2 and group["strategy"].strip()]
    selected = None
    for triple in combinations(eligible, 3):
        if len({decks[key]["strategy"].strip() for key in triple}) != 3:
            continue
        if all(max(len(set(decks[a]["deck"]) - set(decks[b]["deck"])),
                   len(set(decks[b]["deck"]) - set(decks[a]["deck"]))) >= 3
               for a, b in combinations(triple, 2)):
            selected = list(triple)
            break
    for group in decks.values():
        group["wins"] = len(group["winning_seeds"])
        group["winning_seeds"] = sorted(group["winning_seeds"])
        group["robust_seeds"] = sorted(group["robust_seeds"])
    return decks, selected


def _derive(entries, run_id=None):
    plan = json.loads((ROOT / "configs" / "llmplay_seeds.json").read_text(encoding="utf-8"))
    existing = load_tasks()
    tasks, layers = {}, {}
    for layer in plan["layers"]:
        ids = [task for task in plan["layers"][layer] if task in existing]
        for task_id in ids:
            seeds = {}
            for entry in entries:
                if entry["task_id"] != task_id or not entry["valid"]:
                    continue
                key = str(entry["seed"])
                previous = seeds.get(key)
                rank = lambda e: (e["robust"], e["won"], bool(e["robustness"]),
                                  int(e["outcome"].get("final_tick") or 0))
                if previous is None or rank(entry) >= rank(previous):
                    seeds[key] = entry
            wins = sum(e["won"] for e in seeds.values())
            robust = sum(e["robust"] for e in seeds.values())
            selectable = existing[task_id].get("deck_selection") == "player"
            attempts = [entry for entry in entries if entry["task_id"] == task_id and entry["valid"]]
            decks, diverse = _deck_progress(attempts) if selectable else ({}, None)
            complete = (wins >= 6 and robust >= REQUIRED_ROBUST and diverse is not None) if selectable else (wins >= REQUIRED_WINS and robust >= REQUIRED_ROBUST)
            tasks[task_id] = dict(layer=layer, seeds=seeds, decks=decks, diverse_decks=diverse,
                                 deck_selection=selectable,
                                 available_seeds=[seed for seed in seed_order(task_id, run_id, plan["tasks"][task_id])
                                                  if not seeds.get(str(seed), {}).get("robust")],
                                 wins=wins, robust=robust,
                                 complete=complete)
        done = sum(tasks[task]["complete"] for task in ids)
        layers[layer] = dict(completed=done, total=len(ids), complete=done == len(ids))
    return dict(schema="llmplay_progress_v1", run_id=run_id, requirements=dict(wins=REQUIRED_WINS,
                robust=REQUIRED_ROBUST, player_decks=3, wins_per_deck=2, player_wins=6,
                min_card_changes=3, core_strategy="model-declared; reviewer assesses semantic difference"), archives=entries, tasks=tasks, layers=layers,
                complete=all(layer["complete"] for layer in layers.values()))


def progress() -> dict:
    """读取已发布的完整快照；首次使用时返回五层零进度。"""
    ensure_run_id()
    document = _read_progress(output_dir())
    return _derive(document["archives"], document["run_id"])


def next_task() -> dict | None:
    """按层返回下一任务/种子及已有存档；只有全部任务达标时返回 None。"""
    document = progress()
    for task_id, task in document["tasks"].items():
        if task["complete"]:
            continue
        seeds = task["seeds"]
        if task.get("deck_selection") and task.get("diverse_decks") is None:
            seed = next((seed for seed in task["available_seeds"] if str(seed) not in seeds), task["available_seeds"][0])
            return dict(task_id=task_id, seed=seed, layer=task["layer"], archive=None,
                        action="play_diverse_decks", decks=task["decks"],
                        note="每套至少2个种子胜；三套两两换至少3张且核心思路不同，填写strategy与deck_review")
        if task["wins"] >= (6 if task.get("deck_selection") else REQUIRED_WINS):
            for entry in seeds.values():
                if entry["won"] and not entry["robust"]:
                    return dict(task_id=task_id, seed=entry["seed"], layer=task["layer"],
                                archive=str(output_dir() / entry["archive"]),
                                action="improve_robustness")
        for seed in task["available_seeds"]:
            if str(seed) not in seeds:
                return dict(task_id=task_id, seed=seed, layer=task["layer"],
                            archive=None, action="play")
        # 40 个种子都试过仍未达标时重试未完成种子，不偷偷创造新任务或扩大种子块。
        for seed in task["available_seeds"]:
            entry = seeds[str(seed)]
            if not entry["robust"]:
                return dict(task_id=task_id, seed=seed, layer=task["layer"],
                            archive=str(output_dir() / entry["archive"]), action="retry")
    return None


def _report(document):
    lines = ["# LLM 对局进度", "", "运行ID：%s" % document["run_id"], "",
             "接口自动生成；详细结果以 progress.json 为准。", "",
             "全部任务达标。" if document["complete"] else "尚未全部完成，继续 next_task()。", "",
             "每任务至少 %d 个种子通关，其中 %d 个第10波达到5/8稳健标准。" %
             (REQUIRED_WINS, REQUIRED_ROBUST),
             "自选卡任务：三套卡组各赢≥2种子、合计≥6不同种子胜、≥3种子稳健；两两换≥3张，核心思路由模型声明并供主控复核。",
             "", "| 层 | 达标任务 |", "|---|---|"]
    lines += ["| %s | %d/%d |" % (layer, info["completed"], info["total"])
              for layer, info in document["layers"].items()]
    lines += ["", "| 任务 | 试过的卡组 | 核心思路 | 胜种子数/未通关尝试 | 选卡理由 | 优缺点评价 |",
              "|---|---|---|---|---|---|"]
    for task_id, task in document["tasks"].items():
        for group in task.get("decks", {}).values():
            cells = [task_id, str(group["deck"]), group["strategy"],
                     "%s/%s" % (group["wins"], group["losses"]), group["reason"], group["review"] or "待模型补充"]
            lines.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in cells) + " |")
    lines += ["", "| 任务 | 种子 | 结果 | 稳健检查点胜局 | 手工补救 | 存档 | 打法/参数说明 |",
              "|---|---|---|---|---|---|---|"]
    for entry in document["archives"]:
        checkpoint = required_checkpoint(dict(level=entry.get("level", 0)))
        cp = ((entry.get("robustness") or {}).get("checkpoints") or {}).get(str(checkpoint)) or {}
        status = "作废：" + str(entry.get("invalid_reason")) if not entry["valid"] else (
            "通关稳健" if entry["robust"] else "通关待稳健" if entry["won"] else "放弃：" + str(entry.get("abandon_reason")) if entry.get("abandoned") else "未通关")
        cells = [entry["task_id"], str(entry["seed"]), status,
                 "%s/%s" % (cp.get("wins", "—"), cp.get("k", "—")),
                 str(entry["manual_patch"]), entry["archive"], str(entry.get("deck", [])) + " " + entry["note"]]
        lines.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in cells) + " |")
    return "\n".join(lines) + "\n"


def _archive_digest(archive):
    digest = hashlib.sha256()
    for name in FILES:
        digest.update(name.encode())
        if not (archive / name).is_file():
            digest.update(b"<missing>")
            continue
        with (archive / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _store(archive, root, *, digest=None, consume=False):
    relative = Path("archives") / (digest or _archive_digest(archive))
    return _restore_archive(archive, root, relative, consume=consume)


def _restore_archive(archive, root, relative, *, consume=False):
    relative = Path(relative)
    target = root / relative
    if not target.is_dir():
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".archive-", dir=target.parent))
        try:
            for name in FILES:
                if (archive / name).is_file():
                    if consume:
                        shutil.move(str(archive / name), str(tmp / name))
                    else:
                        shutil.copyfile(archive / name, tmp / name)
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
    return relative.as_posix()


def _publish(root, entries, session, *, run_id=None, compresslevel=None):
    run_id = run_id or _read_progress(root).get("run_id") or uuid.uuid4().hex
    document = _derive(entries, run_id)
    generations = root / ".delivery"
    generations.mkdir(exist_ok=True)
    generation = generations / uuid.uuid4().hex
    generation.mkdir()
    pending_links = []
    try:
        (generation / "progress.json").write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (generation / "report.md").write_text(_report(document), encoding="utf-8")
        (generation / "budget.json").write_text(json.dumps(session.snapshot(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary = generation / "out.zip.tmp"
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=compresslevel) as bundle:
            for name in ("progress.json", "report.md", "budget.json"):
                bundle.write(generation / name, "out/" + name)
            for relative in sorted({e["archive"] for e in entries}):
                archive = root / relative
                for name in FILES:
                    if (archive / name).is_file():
                        bundle.write(archive / name, "out/" + relative + "/" + name)
            # 模型放在 out 中的策略源码和说明也随交付保存。
            for path in sorted(root.rglob("*")):
                relative = path.relative_to(root)
                if (any(part.startswith(".") for part in relative.parts)
                        or relative.parts[0] == "archives"
                        or relative.as_posix() in ("progress.json", "report.md", "out.zip", "budget.json")
                        or path.is_symlink() or not path.is_file()
                        or ((path.parent / "meta.json").is_file() and (path.parent / "ticks.jsonl").is_file())):
                    continue
                bundle.write(path, "out/" + relative.as_posix())
        os.replace(temporary, generation / "out.zip")
        # 常驻文件入口；此后仅 current 的一次原子 rename 切换整代。
        for name in ("progress.json", "report.md", "budget.json", "out.zip"):
            alias = root / name
            expected = ".delivery/current/" + name
            if not alias.is_symlink() or os.readlink(alias) != expected:
                tmp = root / ("." + name + "." + uuid.uuid4().hex + ".link")
                pending_links.append(tmp)
                tmp.symlink_to(expected)
                os.replace(tmp, alias)
        link = generations / ("current." + uuid.uuid4().hex + ".tmp")
        pending_links.append(link)
        link.symlink_to(generation.name)
        previous = (generations / "current").resolve() if (generations / "current").exists() else None
        os.replace(link, generations / "current")
    except BaseException:
        shutil.rmtree(generation, ignore_errors=True)
        raise
    finally:
        for link in pending_links:
            link.unlink(missing_ok=True)
    # 只保留本代和上一代；不可变存档仍由 archives 持有。
    for old in generations.iterdir():
        if old.is_dir() and not old.is_symlink() and old not in (generation, previous):
            shutil.rmtree(old, ignore_errors=True)
    return document


def record_archive(archive, *, session=None):
    root, archive = output_dir(), Path(archive)
    with _lock(root):
        relative = _store(archive, root)
        entries = [e for e in _read_progress(root)["archives"] if e["archive"] != relative]
        entries.append(_entry(archive, relative))
        return _publish(root, entries, session or budget.session())


def verify_archive(archive, *, executable=None, resource_dir=None):
    """使用包内引擎逐操作核对 tick/接受结果/终局，哈希仅保留作诊断。"""
    from pvz_env import PvZEnv, TaskSpec, PlayerProfileContext
    archive = Path(archive)
    meta = json.loads((archive / "meta.json").read_text(encoding="utf-8"))
    if meta.get("schema") != ARCHIVE_SCHEMA:
        raise ValueError("存档 schema 不正确")
    config = dict(meta["environment_task"])
    task = meta["task"]
    if any(config[key] != task[key] for key in ("level", "seed")):
        raise ValueError("task 与 environment_task 不一致")
    if config.get("zombie_count_multiplier") != task.get("zombie_count_multiplier"):
        raise ValueError("task 与 environment_task 倍率不一致")
    definition = load_tasks().get(task["id"])
    if definition is not None and int(definition["level"]) != int(task["level"]):
        raise ValueError("任务族与关卡不一致")
    if bool(meta["outcome"]["won"]) != (meta["outcome"]["result"] == 1):
        raise ValueError("won 与 result 不一致")
    config["profile"] = PlayerProfileContext(**config["profile"])
    config["preplanted"] = tuple(tuple(p) for p in config.get("preplanted", ()))
    spec = TaskSpec(**config)
    exe = Path(executable) if executable else _default_executable()
    _ensure_executable(exe)
    from .resources import app_data
    resources = resource_dir or _default_resource_dir()
    with PvZEnv(resource_dir=resources, executable=exe, headless=True,
                save_dir=app_data(exe, resources, budget.session())) as env:
        env.begin_demo()
        obs, _ = env.reset(deck=task["deck"], task=spec)
        env.set_lean_observation(True)
        count = 0
        with (archive / "ticks.jsonl").open(encoding="utf-8") as stream:
            for count, line in enumerate(stream, 1):
                entry = json.loads(line)
                if obs["tick"] != entry["tick"]:
                    raise ValueError("操作 %d 起始 tick 不同" % count)
                obs, _, _, _, info = env.step(entry["action"])
                if obs["tick"] != entry["end_tick"] or bool(info["ok"]) != entry["ok"]:
                    raise ValueError("操作 %d tick/ok 不同" % count)
        if count != meta["operations"]:
            raise ValueError("操作数量不符")
        outcome = meta["outcome"]
        if (obs["tick"], obs["wave"], obs["result"]) != (outcome["final_tick"], outcome["final_wave"], outcome["result"]):
            raise ValueError("终局不同")
    with (archive / "frames.jsonl").open(encoding="utf-8") as stream:
        frames = [json.loads(line) for line in stream]
    if len(frames) != meta["frame_count"]:
        raise ValueError("帧数量不符")
    return True


def _extract(source, target):
    with zipfile.ZipFile(source) as bundle:
        for entry in bundle.infolist():
            path = PurePosixPath(entry.filename)
            if (path.is_absolute() or ".." in path.parts or "\\" in entry.filename
                    or (entry.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError("zip 中含不安全路径或软链：%s" % entry.filename)
        bundle.extractall(target)


def resume(source, verify="fast", *, executable=None, resource_dir=None) -> dict:
    """默认按旧progress直接恢复，不算摘要、不重放；full才并行重放全部存档。"""
    if verify not in ("fast", "full"):
        raise ValueError('verify 必须是 "fast" 或 "full"')
    root = output_dir()
    with tempfile.TemporaryDirectory(prefix="llmplay-resume-") as directory:
        source = Path(source).expanduser().resolve()
        extracted = source.is_file()
        if extracted:
            _extract(source, directory)
            source = Path(directory)
        if not source.is_dir():
            raise ValueError("续跑来源必须是 out.zip 或目录")
        missing, prior_run_ids, prior_entries = [], set(), {}
        progress_found = False
        for path in source.rglob("progress.json"):
            if any(part.startswith(".") for part in path.relative_to(source).parts):
                continue
            prior = json.loads(path.read_text(encoding="utf-8"))
            progress_found = True
            if prior.get("run_id"):
                prior_run_ids.add(prior["run_id"])
            for entry in prior.get("archives", []):
                relative = PurePosixPath(entry["archive"])
                if relative.is_absolute() or ".." in relative.parts or "\\" in entry["archive"]:
                    raise ValueError("progress.json 中含不安全存档路径")
                archive = path.parent / entry["archive"]
                prior_entries.setdefault(archive, []).append(entry)
                if not archive.is_dir():
                    invalid = dict(entry, valid=False, won=False, robust=False,
                                   invalid_reason="进度引用的存档缺失，无法恢复")
                    missing.append(invalid)
        if len(prior_run_ids) > 1:
            raise ValueError("续跑来源含多个 run_id，请指定一份 out.zip 或对应的 out 目录")
        if verify == "fast" and not progress_found:
            raise ValueError('续跑来源缺少progress.json；独立存档请用verify="full"导入')
        restored = ({archive: dict(records[-1]) for archive, records in prior_entries.items()
                     if archive.is_dir()} if verify == "fast" else {})
        candidates = (list(restored) if verify == "fast" else
                      sorted(p.parent for p in source.rglob("meta.json")
                             if not any(part.startswith(".") for part in p.relative_to(source).parts)))
        pending = candidates if verify == "full" else []

        def replay(archive):
            try:
                with _VERIFY_SLOTS:
                    verify_archive(archive, executable=executable, resource_dir=resource_dir)
                return True, None
            except Exception as exc:
                return False, str(exc)

        verified = {}
        if pending:
            workers = os.cpu_count() or 1
            print("resume: %d 份重放复验（workers=%d，最多2个游戏进程）" %
                  (len(pending), workers), flush=True)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(replay, archive): archive for archive in pending}
                for count, future in enumerate(as_completed(futures), 1):
                    verified[futures[future]] = future.result()
                    if count % 10 == 0:
                        print("resume verify: %d/%d" % (count, len(pending)), flush=True)
        imported = [(archive, *(verified.get(archive, (True, None)))) for archive in candidates]
        if not prior_run_ids and verify == "full":
            for archive, valid, error in imported:
                if valid:
                    run_id = json.loads((archive / "meta.json").read_text()).get("run_id")
                    if run_id:
                        prior_run_ids.add(run_id)
        restored_run_id = next(iter(prior_run_ids)) if len(prior_run_ids) == 1 else None
        with _lock(root):
            entries = _read_progress(root)["archives"]
            for entry in missing:
                entries = [e for e in entries if e["archive"] != entry["archive"]]
                entries.append(entry)
            for archive, valid, error in imported:
                relative = None
                try:
                    if archive in restored:
                        relative = restored[archive]["archive"]
                        relative = _restore_archive(archive, root, relative, consume=extracted)
                        entry = dict(restored[archive], archive=relative)
                    else:
                        relative = _store(archive, root)
                        entry = _entry(archive, relative, valid=valid, error=error)
                except (OSError, KeyError, ValueError, TypeError) as exc:
                    # 不可解析的存档同样保留作废记录，不能中止其他局的恢复。
                    relative = relative or "invalid/" + hashlib.sha256(str(archive.relative_to(source)).encode()).hexdigest()
                    entry = dict(task_id="unknown", seed=0, result=None, outcome={}, robustness=None,
                                 archive=relative, manual_patch=False, note="", valid=False,
                                 invalid_reason=str(exc), won=False, robust=False)
                entries = [e for e in entries if e["archive"] != relative]
                entries.append(entry)
            session = budget.session()
            for path in source.rglob("budget.json"):
                if not any(part.startswith(".") for part in path.relative_to(source).parts):
                    session.import_budget(json.loads(path.read_text(encoding="utf-8")))
            # 恢复策略/脚本，保留相对 out 的路径；用户文件不覆盖已存在文件。
            base = source / "out" if (source / "out").is_dir() else source
            for path in base.rglob("*"):
                relative = path.relative_to(base)
                if (path.is_symlink() or not path.is_file()
                        or relative.parts[0] == "archives"
                        or relative.as_posix() in ("progress.json", "report.md", "out.zip", "budget.json")
                        or ((path.parent / "meta.json").is_file() and (path.parent / "ticks.jsonl").is_file())
                        or any(part.startswith(".") for part in relative.parts)):
                    continue
                destination = root / relative
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, destination)
            return _publish(root, entries, session, run_id=restored_run_id, compresslevel=1)
