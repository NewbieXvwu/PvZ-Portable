"""不可变存档与小型进度快照；显式交付只压缩一次，分卷放在 out 外。"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from pvz_frame_store import frame_path

from . import budget
from . import workflow
from .runtime import output_path
from .game import (ROOT, ARCHIVE_SCHEMA, _default_executable, _default_resource_dir,
                   _ensure_executable, load_tasks, required_checkpoint)

FILES = ("meta.json", "ticks.jsonl", "frames.npz", "frames.jsonl")
LAYERS = workflow.LAYERS


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
def _lock(root, filename=".delivery.lock"):
    root.mkdir(parents=True, exist_ok=True)
    with (root / filename).open("a+") as handle:
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
    return bool(cp.get("reached") and cp.get("k") == 8
                and 5 <= int(cp.get("wins") or 0) <= 8)


def _entry(archive, relative, *, valid=True, error=None):
    meta = json.loads((archive / "meta.json").read_text(encoding="utf-8"))
    task, outcome = meta["task"], meta["outcome"]
    won = valid and not meta.get("abandoned") and outcome.get("result") == 1 and bool(outcome.get("won"))
    return dict(task_id=task["id"], seed=int(task["seed"]), result=outcome["result"],
                outcome=outcome, robustness=meta.get("robustness"),
                level=task["level"], deck=task["deck"], deck_reason=meta.get("deck_reason", ""),
                strategy=meta.get("strategy", ""), deck_review=meta.get("deck_review", ""),
                abandoned=bool(meta.get("abandoned")), abandon_reason=meta.get("abandon_reason"),
                attempt_id=meta.get("attempt_id"),
                archive=relative, manual_patch=bool(meta.get("manual_patch")),
                note=meta.get("note", ""), valid=valid, invalid_reason=error,
                won=won, robust=bool(won and _robust(meta)), run_id=meta.get("run_id"),
                cycle=int(meta.get("cycle") or 1),
                workflow_layer=meta.get("workflow_layer"), opening=meta.get("opening") or [],
                environment_fingerprint=meta.get("environment_fingerprint"),
                fog_observation=(meta.get("environment_task") or {}).get("fog_observation"),
                conveyor_start=(meta.get("environment_task") or {}).get("conveyor_start"),
                deck_seed_attempts=meta.get("deck_seed_attempts"),
                deck_attempt_seeds=meta.get("deck_attempt_seeds"),
                zombie_count_multiplier=task.get("zombie_count_multiplier"))


def _derive(entries, run_id=None):
    """判定交给 workflow：循环 / 分层 / 下传 / 每项还差什么都在那里算。"""
    return workflow.derive(entries, run_id)


def progress() -> dict:
    """读取已发布的完整快照；首次使用时返回第 1 圈 A 层零进度。"""
    ensure_run_id()
    document = _read_progress(output_dir())
    return _derive(document["archives"], document["run_id"])


def next_task(track=None, *, shard=None, shards=None) -> dict:
    """领下一项。track="C" 表示 C 轨专职子代理（C 优先，做完接主线）。"""
    if workflow.decks_mode() and (shard is not None or shards is not None):
        from .decks import configure_shard
        configure_shard(shard, shards)
    return workflow.next_task(progress(), track=track)


def check(task_id, deck, strategy="", seed=None) -> dict:
    """Game() 建局前把关：层不对 / 卡组差 <3 张 / 打法重复 一律拒绝。"""
    return workflow.check(progress(), task_id, deck, strategy, seed)


def archive_requirements(task_id, strategy="", review="", abandoned=False, **kwargs) -> None:
    """导出存档前的必填项（A/D 层的思路与评价）。"""
    workflow.archive_requirements(task_id, strategy, review, abandoned, **kwargs)


def diagnose(task_id, *, waves=None, missing="", note="", deck=None) -> Path:
    """C/E 层失败诊断；E 层用 deck 绑定卡组，写完立刻接下一组。"""
    if not str(missing).strip():
        raise ValueError("失败诊断要写清楚缺什么（missing=...），例如「第 8 波后排水路缺群伤」")
    root = output_dir()
    with _lock(root):
        document = _read_progress(root)
        derived = _derive(document["archives"], document.get("run_id"))
        main, challenge = workflow._ordered(derived)
        candidates = [item for item in main + challenge
                      if item["task_id"] == task_id and item["action"] == "diagnose"
                      and (deck is None or workflow.deck_key(item["deck"]) == workflow.deck_key(deck))]
        if len(candidates) != 1:
            raise ValueError("须先完成 C 轨指定终局或 E 层同一卡组同一倍数连续 5 个新种子的败局；E 层请传 deck")
        item = candidates[0]
        if not waves or any(type(wave) is not int or wave < 0 for wave in waves):
            raise ValueError("waves 必须填写真实失败波次")
        folder = root / "diagnosis" / str(derived["cycle"])
        folder.mkdir(parents=True, exist_ok=True)
        payload = dict(schema="llmplay_diagnosis_v1", task_id=task_id, cycle=derived["cycle"],
                       waves=list(waves), missing=str(missing), note=str(note))
        suffix = ""
        if item["layer"] == "E":
            payload.update(deck=sorted(item["deck"]), multiplier=item["multiplier"])
            suffix = "__" + workflow.deck_key(item["deck"])
        path = folder / (task_id + suffix + ".json")
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _publish(root, document["archives"], budget.session())
        workflow.release(task_id, None)
    return path


def _report(document):
    return workflow.report(document)


def _archive_digest(archive):
    digest = hashlib.sha256()
    for name in ("meta.json", "ticks.jsonl", frame_path(archive).name):
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
    document["budget"] = session.snapshot()
    generations = root / ".delivery"
    generations.mkdir(exist_ok=True)
    generation = generations / uuid.uuid4().hex
    generation.mkdir()
    pending_links = []
    try:
        (generation / "progress.json").write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (generation / "report.md").write_text(_report(document), encoding="utf-8")
        (generation / "budget.json").write_text(json.dumps(document["budget"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        registry = {key: info["registered_decks"] for key, info in document.get("levels", {}).items()}
        (generation / "decks.json").write_text(json.dumps(registry, ensure_ascii=False) + "\n", encoding="utf-8")
        # 常驻文件入口；此后仅 current 的一次原子 rename 切换整代。
        for name in ("progress.json", "report.md", "budget.json", "decks.json"):
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
        os.replace(link, generations / "current")
    except BaseException:
        shutil.rmtree(generation, ignore_errors=True)
        raise
    finally:
        for link in pending_links:
            link.unlink(missing_ok=True)
    # current 只持有小型结论；旧代和旧版压缩包立即清掉。
    for old in generations.iterdir():
        if old == generation or old.name == "current":
            continue
        if old.is_dir() and not old.is_symlink():
            shutil.rmtree(old, ignore_errors=True)
        else:
            old.unlink(missing_ok=True)
    (root / "out.zip").unlink(missing_ok=True)
    for old in root.glob(".out.zip.*"):
        if not old.is_dir():
            old.unlink(missing_ok=True)
    return document


BUNDLE_PART_BYTES = 99_000_000


def bundle_dir() -> Path:
    root = output_dir()
    return root.with_name(root.name + "_exports")


class _Volumes(io.RawIOBase):
    """Write a standard ZIP as contiguous <=99 MB pieces, without a whole ZIP copy."""
    def __init__(self, directory):
        self.directory = directory
        self.position = 0
        self.part_size = 0
        self.stream = None
        self.count = 0

    def writable(self):
        return True

    def tell(self):
        return self.position

    def write(self, data):
        view = memoryview(data)
        size = len(view)
        while view:
            if self.stream is None or self.part_size == BUNDLE_PART_BYTES:
                if self.stream is not None:
                    self.stream.close()
                self.count += 1
                self.stream = (self.directory / ("out.zip.%03d" % self.count)).open("wb")
                self.part_size = 0
            length = min(len(view), BUNDLE_PART_BYTES - self.part_size)
            self.stream.write(view[:length])
            self.part_size += length
            self.position += length
            view = view[length:]
        return size

    def close(self):
        if self.stream is not None:
            self.stream.close()
        super().close()


def export_bundle() -> list[Path]:
    """只交付存档、进度、卡组登记和 .py 策略；固定放在 out 的同级 *_exports。"""
    root, destination = output_dir(), bundle_dir()
    ensure_run_id()
    # 打包单独串行；长时间压缩不持有对局发布锁。
    with _lock(root.parent, "." + root.name + "_exports.lock"):
        with tempfile.TemporaryDirectory(prefix="." + destination.name + "-", dir=root.parent) as temporary:
            work = Path(temporary)
            parts = work / "parts"
            parts.mkdir()
            with _lock(root):
                document = _read_progress(root)
                (work / "progress.json").write_text(json.dumps(document, ensure_ascii=False) + "\n", encoding="utf-8")
                registry = {key: info["registered_decks"] for key, info in document.get("levels", {}).items()}
                (work / "decks.json").write_text(json.dumps(registry, ensure_ascii=False) + "\n", encoding="utf-8")
            with _Volumes(parts) as writer:
                with zipfile.ZipFile(writer, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
                    for name in ("progress.json", "decks.json"):
                        bundle.write(work / name, "out/" + name)
                    for relative in sorted({e["archive"] for e in document["archives"] if e.get("archive")}):
                        for name in FILES:
                            path = root / relative / name
                            if path.is_file():
                                bundle.write(path, "out/" + relative + "/" + name,
                                             compress_type=zipfile.ZIP_STORED if name == "frames.npz" else zipfile.ZIP_DEFLATED)
                    for directory, folders, files in os.walk(root):
                        folders[:] = [name for name in folders if not name.startswith(".")
                                      and name not in ("archives", "__pycache__")]
                        directory = Path(directory)
                        if (directory / "meta.json").is_file() and (directory / "ticks.jsonl").is_file():
                            folders[:] = []
                            continue
                        for name in sorted(files):
                            path = directory / name
                            if name.endswith(".py") and not name.startswith(".") and not path.is_symlink():
                                bundle.write(path, "out/" + path.relative_to(root).as_posix())
            previous = destination.with_name("." + destination.name + ".previous")
            if previous.exists():
                shutil.rmtree(previous)
            if destination.exists():
                os.replace(destination, previous)
            try:
                os.replace(parts, destination)
            except BaseException:
                if previous.exists():
                    os.replace(previous, destination)
                raise
            shutil.rmtree(previous, ignore_errors=True)
    return sorted(destination.glob("out.zip.*"))


def record_archive(archive, *, session=None):
    root, archive = output_dir(), Path(archive)
    with _lock(root):
        relative = _store(archive, root)
        entries = [e for e in _read_progress(root)["archives"] if e["archive"] != relative]
        entries.append(_entry(archive, relative))
        return _publish(root, entries, session or budget.session())


def _extract(source, target):
    with zipfile.ZipFile(source) as bundle:
        for entry in bundle.infolist():
            path = PurePosixPath(entry.filename)
            if (path.is_absolute() or ".." in path.parts or "\\" in entry.filename
                    or (entry.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError("zip 中含不安全路径或软链：%s" % entry.filename)
        bundle.extractall(target)


# ----------------------------------------------------------------- 前次进度

CARRY_SCHEMA = "llmplay_progress_v2"
CARRY_PATH = ("carry", "progress.json")
DIAGNOSIS_SCHEMA = "llmplay_diagnosis_v1"


def carry(source=None):
    """接上一份 progress.json 开局：只读结论层，不重放、不复验、不碰存档。

    source 默认是包内 `carry/progress.json`（给目录也行）。只信任文件里的记录：
    已完成的层、用过的卡组与种子、B 层下传的卡组都从这些记录重算，存档本体不需要存在。
    out 里已经有进度时什么都不做，所以重复 import / 重开会话都安全。
    """
    root = output_dir()
    source = Path(source).expanduser() if source else ROOT.joinpath(*CARRY_PATH)
    if source.is_dir():
        source = source / CARRY_PATH[-1]
    if not source.is_file():
        return None
    with _lock(root):
        if _read_progress(root).get("archives"):
            return None
        prior = json.loads(source.read_text(encoding="utf-8"))
        if prior.get("schema") != CARRY_SCHEMA:
            raise ValueError("前次进度不是 %s：%s" % (CARRY_SCHEMA, source))
        known = workflow.plan()["tasks"]
        entries = [_carry_entry(entry) for entry in prior.get("archives") or []]
        entries = [entry for entry in entries if entry["task_id"] in known]
        if not workflow.decks_mode():
            _restore_diagnoses(root, prior)
        return _publish(root, entries, budget.session(), run_id=prior.get("run_id"))


def _carry_entry(entry) -> dict:
    """把上一份 progress 的一条记录补成当前判定要用的形状；不做任何核对。"""
    out = dict(entry)
    if workflow.decks_mode():
        out["carried"] = True
    out["seed"] = int(out["seed"])
    out["cycle"] = int(out.get("cycle") or 1)
    out["valid"] = bool(out.get("valid", True))
    out["won"] = bool(out.get("won"))
    out["robust"] = bool(out.get("robust"))
    out["abandoned"] = bool(out.get("abandoned"))
    out["deck"] = list(out.get("deck") or [])
    out["opening"] = list(out.get("opening") or [])
    for name in ("archive", "note", "strategy", "deck_reason", "deck_review"):
        out[name] = out.get(name) or ""
    for name, value in (("attempt_id", None), ("result", None), ("robustness", None),
                        ("outcome", {}), ("level", 0), ("workflow_layer", None),
                        ("manual_patch", False)):
        out.setdefault(name, value)
    return out


def _restore_diagnoses(root, prior) -> None:
    """接回 C/E 结论层中的诊断；E 诊断按任务、圈和卡组保留。"""
    detail = prior.get("detail") or {}
    cycle = int(prior.get("cycle") or 1)
    items = list((detail.get("C") or {}).values())
    items += [item for group in (detail.get("E") or {}).values()
              for item in (group.get("items") or {}).values()]
    items += prior.get("carried_diagnoses") or []
    items += [item["carried_completion"] for item in items if item.get("carried_completion")]
    for item in items:
        report = (item or {}).get("diagnosis") or {}
        if not (report.get("waves") and str(report.get("missing") or "").strip()):
            continue
        task_id = item["task_id"]
        item_cycle = int(item.get("cycle") or cycle)
        folder = root / "diagnosis" / str(item_cycle)
        folder.mkdir(parents=True, exist_ok=True)
        payload = dict(schema=DIAGNOSIS_SCHEMA, task_id=task_id, cycle=item_cycle,
                       waves=list(report["waves"]), missing=str(report["missing"]),
                       note=str(report.get("note") or ""))
        suffix = ""
        if "multiplier" in item:
            payload.update(deck=sorted(item["deck"]), multiplier=item["multiplier"])
            suffix = "__" + workflow.deck_key(item["deck"])
        (folder / (task_id + suffix + ".json")).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resume(source) -> dict:
    """按旧 progress 恢复：只搬存档和结论，不算摘要、不重放、不复验。"""
    root = output_dir()
    with tempfile.TemporaryDirectory(prefix="llmplay-resume-") as directory:
        source = Path(source).expanduser().resolve()
        if source.is_dir() and (source / "out.zip.001").is_file():
            source = source / "out.zip.001"
        if source.name == "out.zip.001":
            joined = Path(directory) / "out.zip"
            with joined.open("wb") as stream:
                for part in sorted(source.parent.glob("out.zip.[0-9]*"), key=lambda path: int(path.suffix[1:])):
                    with part.open("rb") as piece:
                        shutil.copyfileobj(piece, stream)
            source = joined
        extracted = source.is_file()
        if extracted:
            _extract(source, directory)
            source = Path(directory)
        if not source.is_dir():
            raise ValueError("续跑来源必须是 out.zip 或目录")
        missing, prior_run_ids, prior_entries, loose = [], set(), {}, []
        for path in source.rglob("progress.json"):
            if any(part.startswith(".") for part in path.relative_to(source).parts):
                continue
            prior = json.loads(path.read_text(encoding="utf-8"))
            if prior.get("run_id"):
                prior_run_ids.add(prior["run_id"])
            if prior.get("budget"):
                budget.session().import_budget(prior["budget"])
            for entry in prior.get("archives", []):
                relative = PurePosixPath(entry["archive"])
                if relative.is_absolute() or ".." in relative.parts or "\\" in entry["archive"]:
                    raise ValueError("progress.json 中含不安全存档路径")
                archive = path.parent / entry["archive"]
                prior_entries.setdefault(archive, []).append(entry)
                if not archive.is_dir():
                    if workflow.decks_mode() and entry.get("carried"):
                        # 随包沿用的结论本来就不含旧存档，续跑继续保留其登记。
                        missing.append(dict(entry))
                    else:
                        missing.append(dict(entry, valid=False, won=False, robust=False,
                                            invalid_reason="进度引用的存档缺失，无法恢复"))
        if len(prior_run_ids) > 1:
            raise ValueError("续跑来源含多个 run_id，请指定一份 out.zip 或对应的 out 目录")
        if not prior_entries:
            # 没有 progress.json：按目录里的存档直接导入（只读 meta.json，不重放）。
            for path in sorted(source.rglob("meta.json")):
                if any(part.startswith(".") for part in path.relative_to(source).parts):
                    continue
                loose.append(path.parent)
        restored = {archive: dict(records[-1]) for archive, records in prior_entries.items()
                    if archive.is_dir()}
        restored_run_id = next(iter(prior_run_ids)) if len(prior_run_ids) == 1 else None
        with _lock(root):
            entries = _read_progress(root)["archives"]
            for entry in missing:
                entries = [e for e in entries if e["archive"] != entry["archive"]]
                entries.append(entry)
            for archive, record in restored.items():
                try:
                    relative = _restore_archive(archive, root, record["archive"], consume=extracted)
                    entry = dict(record, archive=relative)
                except (OSError, KeyError, ValueError, TypeError) as exc:
                    relative = "invalid/" + uuid.uuid4().hex
                    entry = dict(task_id="unknown", seed=0, result=None, outcome={}, robustness=None,
                                 archive=relative, manual_patch=False, note="", valid=False,
                                 invalid_reason=str(exc), won=False, robust=False)
                entries = [e for e in entries if e["archive"] != relative]
                entries.append(entry)
            for archive in loose:
                try:
                    relative = _store(archive, root, consume=extracted)
                    entry = _entry(archive, relative)
                    if not restored_run_id and entry.get("run_id"):
                        restored_run_id = entry["run_id"]
                except (OSError, KeyError, ValueError, TypeError) as exc:
                    entry = dict(task_id="unknown", seed=0, result=None, outcome={}, robustness=None,
                                 archive="invalid/" + uuid.uuid4().hex, manual_patch=False, note="",
                                 valid=False, invalid_reason=str(exc), won=False, robust=False)
                entries = [e for e in entries if e["archive"] != entry["archive"]]
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
