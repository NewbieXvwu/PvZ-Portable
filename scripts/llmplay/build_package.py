"""把"LLM 打全程"沙箱包打出来：引擎 + 资源 + 接口 + 任务书。

    python3 scripts/llmplay/build_package.py --executable <静态引擎> --cache-dir <full-appdata> [--out DIR]

产物落在 ~/pvz-logs/llmplay_pkg/（不进 git、不上 HF）：
llmplay_pkg.zip（接口/引擎）与 llmplay_res.zip（资源/缓存），以及对应的 stage/ 与 resources/。

包里只有：纯 Python 接口、引擎二进制、游戏资源、任务清单、任务书、示例。
不打包 .git / artifacts / 日志 / 权重 / 密钥；打包前会扫一遍。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import os
import tempfile
import types
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

PYTHON_MODULES = ["pvz_env.py", "pvz_common.py", "pvz_frame_store.py"]
# pvz_constants.py 是**从 C++ 源码现场解析**常量表的（AGENTS 的约定：不许手抄表），
# 所以包里必须带上它读的那几个文件，否则 import 阶段就抛"常量来源文件不存在"。
CONSTANT_SOURCES = ["src/ConstEnums.h", "src/Lawn/Plant.cpp", "src/Lawn/Zombie.cpp",
                    "src/Lawn/Board.cpp", "src/LawnApp.cpp", "src/SexyAppFramework/SexyAppBase.cpp"]
SCRIPT_MODULES = ["pvz_constants.py", "pvz_tasks.py", "render_episode.py", "episode_query.py"]
RESOURCE_DIR = Path("/home/newbiexvwu/.cache/pvz-research-resources")
DEFAULT_OUT = Path.home() / "pvz-logs" / "llmplay-decks"
SEED_PLAN = ROOT / "configs" / "llmplay_seeds.json"
ROBUSTNESS_PLAN = ROOT / "configs" / "llmplay_robustness.json"

SECRET_PATTERNS = [re.compile(pattern) for pattern in
                   (rb"hf_[A-Za-z0-9]{20,}", rb"sk-[A-Za-z0-9]{20,}", rb"AIza[A-Za-z0-9_-]{30,}",
                    rb"ghp_[A-Za-z0-9]{30,}", rb"(?i)password\s*[:=]\s*\S+",
                    rb"(?i)token\s*[:=]\s*[A-Za-z0-9_\-]{16,}")]


def seed_plan() -> dict:
    """task139 的循环计划：任务顺序 + 每任务 160 个种子（`configs/llmplay_seeds.json`，机械生成）。"""
    document = json.loads(SEED_PLAN.read_text(encoding="utf-8"))
    if document.get("schema") != "llmplay_cycle_v1":
        raise SystemExit("不是 llmplay_cycle_v1：%s" % SEED_PLAN)
    return document


def task_manifest() -> dict:
    """按循环计划打包本轮 A/C/E 三层新族；参考卡组来自冻结记分板。"""
    repo = json.loads((ROOT / "configs" / "human_demo_tasks.json").read_text(encoding="utf-8"))
    if seed_plan().get("mode") == "decks":
        # 保留历史任务定义，旧存档仍能 load_archive；只派发新扩卡组族。
        return repo
    by_id = {task["task_id"]: task for task in repo["tasks"]}
    plan = seed_plan()
    order = list(plan["order"])
    missing = [task_id for task_id in order if task_id not in by_id]
    if missing:
        raise SystemExit("任务清单缺这些族，先跑 scripts/llmplay/make_llmplay3_tasks.py：%s" % missing)
    tasks = []
    for task_id in order:
        task = dict(by_id[task_id])
        task["seeds"] = [plan["tasks"][task_id][0]]      # 默认种子 = 本任务块的第一个
        tasks.append(task)
    return dict(schema=repo.get("schema", "human_demo_tasks_v1"), tasks=tasks)


def extra_seeds(manifest: dict) -> dict:
    """每个任务 160 个种子（139,000,000 起，每任务步长 1000；每圈用 40 个窗口）。"""
    plan = seed_plan()
    if plan.get("mode") == "decks":
        return plan["tasks"]
    out = {}
    for task in manifest["tasks"]:
        seeds = plan["tasks"].get(task["task_id"])
        if not seeds:
            raise SystemExit("种子块里没有任务 %s" % task["task_id"])
        out[task["task_id"]] = list(seeds)
    return out


def copy_modules(stage: Path) -> None:
    (stage / "python").mkdir(parents=True, exist_ok=True)
    (stage / "scripts").mkdir(parents=True, exist_ok=True)
    for name in PYTHON_MODULES:
        shutil.copyfile(ROOT / "python" / name, stage / "python" / name)
    for name in SCRIPT_MODULES:
        shutil.copyfile(ROOT / "scripts" / name, stage / "scripts" / name)
    # 只拷接口本体：PROMPT/README/examples 在包根目录已经有了，再拷一份会让人
    # 分不清看哪个。
    shutil.copytree(ROOT / "scripts" / "llmplay", stage / "llmplay",
                    ignore=shutil.ignore_patterns("__pycache__", "build_package.py",
                                                  "build_probe.py", "probe",
                                                  "PROMPT.md", "README.md", "examples",
                                                  "mac_check.sh",
                                                  "mac_bundle.sh", "make_llmplay_tasks.py",
                                                  "check_tasks.py", "robustness_check.py",
                                                  "cache_prepare.py", "cache_profile.py",
                                                  "make_llmplay2_tasks.py", "make_llmplay3_tasks.py"))


def read_partial(source: Path) -> dict:
    """只读进度和失败诊断结论；存档不解出、不打开。"""
    source = source.expanduser()
    if source.is_dir():
        if (source / "out.zip").is_file():
            return read_partial(source / "out.zip")
        if (source / "out/progress.json").is_file():
            source = source / "out"
        document = json.loads((source / "progress.json").read_text(encoding="utf-8"))
        reports = [json.loads(path.read_text(encoding="utf-8"))
                   for path in (source / "diagnosis").glob("*/*.json")]
        return _with_diagnoses(document, reports)
    if source.suffix == ".zip":
        with zipfile.ZipFile(source) as bundle:
            name = next(name for name in bundle.namelist()
                        if name == "progress.json" or name.endswith("/progress.json"))
            document = json.loads(bundle.read(name))
            prefix = name.removesuffix("progress.json") + "diagnosis/"
            reports = [json.loads(bundle.read(path)) for path in bundle.namelist()
                       if path.startswith(prefix) and path.endswith(".json")]
            return _with_diagnoses(document, reports)
    return json.loads(source.read_text(encoding="utf-8"))


def _with_diagnoses(document, reports):
    items = list(document.get("carried_diagnoses") or [])
    for report in reports:
        item = dict(task_id=report["task_id"], cycle=report["cycle"], diagnosis=report)
        if "deck" in report:
            item.update(deck=report["deck"], multiplier=report["multiplier"])
        items.append(item)
    return dict(document, carried_diagnoses=items)


def merge_carry(prior: dict, latest: dict) -> dict:
    """合并已完成记录与诊断；不读取存档，不重算进度、不生成中间文件。"""
    fresh = latest.get("archives") or []
    keys = {entry["archive"] for entry in fresh}
    result = dict(prior, **latest)
    result["archives"] = [entry for entry in prior.get("archives", [])
                          if entry["archive"] not in keys] + fresh
    reports = {}
    for document in (prior, latest):
        items = list(document.get("carried_diagnoses") or [])
        detail = document.get("detail") or {}
        for item in list((detail.get("C") or {}).values()) + [
                item for group in (detail.get("E") or {}).values()
                for item in (group.get("items") or {}).values()]:
            if item.get("diagnosis"):
                items.append(dict(item, cycle=document.get("cycle", 1)))
            completed = item.get("carried_completion") or {}
            if completed.get("diagnosis"):
                items.append(dict(completed, cycle=document.get("cycle", 1)))
        for item in items:
            key = (item["cycle"], item["task_id"], tuple(sorted(item.get("deck") or [])))
            reports[key] = item
    result["carried_diagnoses"] = list(reports.values())
    return result


def copy_carry(stage: Path, carry: Path | None, partial: Path | None = None) -> None:
    """把前次进度放进 carry/：首次 import 时接口自动接上，不用再传别的文件。"""
    if carry is None:
        return
    target = stage / "carry" / "progress.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if partial is None:
        shutil.copyfile(carry, target)
    else:
        merged = merge_carry(read_partial(carry), read_partial(partial))
        from llmplay import workflow
        reports = {}
        for item in merged["carried_diagnoses"]:
            key = (item["task_id"], item["cycle"])
            if "multiplier" in item:
                key += (workflow.deck_key(item["deck"]),)
            reports[key] = item["diagnosis"]
        document = workflow.derive(merged["archives"], merged.get("run_id"), diagnosis=reports)
        document["carried_diagnoses"] = merged["carried_diagnoses"]
        target.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(dict(carry_cycle=document["cycle"], carry_layer=document["layer"],
                              layers=document.get("layers"), archives=len(document["archives"])),
                         ensure_ascii=False))


def copy_reference(stage: Path) -> None:
    """包里的静态参考表（task144）：阳光、冷却、可见动作时长，全部现解析源码。"""
    from llmplay import reference
    (stage / "PLANTS.md").write_text(reference.render_markdown(), encoding="utf-8")


def copy_constant_sources(stage: Path) -> None:
    for relative in CONSTANT_SOURCES:
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)


def copy_resources(stage: Path, resource_dir: Path) -> None:
    target = stage / "res"
    (target / "properties").mkdir(parents=True, exist_ok=True)
    for source in (resource_dir / "main.pak", resource_dir / "properties" / "partner.xml",
                   resource_dir / "properties" / "partner_logo.jpg"):
        shutil.copyfile(source.resolve(), target / source.relative_to(resource_dir))


def scan(stage: Path) -> list:
    hits = []
    for path in sorted(stage.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix in (".pak", ".jpg", ".png") or path.name == "pvz-portable":
            continue
        data = path.read_bytes()
        for pattern in SECRET_PATTERNS:
            for match in pattern.finditer(data):
                # 误报过滤：token = some_function(action) 这类是函数调用，不是凭据。
                # 命中的下一字符是 '(' 就跳过。
                if data[match.end():match.end() + 1] == b"(":
                    continue
                hits.append("%s: %s" % (path.relative_to(stage),
                                        pattern.pattern.decode("utf-8", "replace")))
                break
    return hits


def build(executable: Path, out: Path, *, cache_dir: Path, resource_dir: Path = RESOURCE_DIR,
          carry: Path | None = None) -> Path:
    """分别打接口包与资源包；使用临时构建树，不能提前删除旧包/输入引擎。"""
    from llmplay.cache_prepare import audit_cache
    from llmplay.resources import PACKAGE
    executable, cache_dir = Path(executable).resolve(), Path(cache_dir).resolve()
    audit = audit_cache(cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".package-build-", dir=out) as temporary:
        return _build(executable, out, cache_dir, resource_dir, Path(temporary), audit, PACKAGE,
                      carry=carry)


def repack_interface(previous: Path, out: Path, carry: Path | None = None,
                     partial: Path | None = None, resource_zip: Path | None = None) -> list[Path]:
    """复用已交付引擎及资源契约，只重打接口，不重建资源、不重复逐关 reset。

    previous 是上一包的 stage；大文件留在本机，不进入 git。py7zr 仅构建时需要。
    """
    import py7zr
    from llmplay.resources import PACKAGE
    previous, out = previous.resolve(), out.resolve()
    carry = carry or previous / "carry" / "progress.json"
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".interface-build-", dir=out) as temporary:
        stage = Path(temporary) / "llmplay_pkg"
        stage.mkdir()
        copy_modules(stage)
        copy_constant_sources(stage)
        (stage / "build").mkdir()
        engine = previous / "build" / "pvz-portable"
        shutil.copyfile(engine, stage / "build" / "pvz-portable")
        (stage / "build" / "pvz-portable").chmod(0o755)
        (stage / "configs").mkdir()
        manifest = task_manifest()
        for name, document in (("human_demo_tasks.json", manifest),
                               ("extra_seeds.json", extra_seeds(manifest))):
            (stage / "configs" / name).write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
        for path in (SEED_PLAN, ROBUSTNESS_PLAN,
                     ROOT / "configs" / "adventure_scoreboard.json", ROOT / "configs" / "gate_v5_panel.json",
                     ROOT / "configs" / "deploy17_deck_holdouts.json"):
            shutil.copyfile(path, stage / "configs" / path.name)
        (stage / "configs" / PACKAGE).write_text(json.dumps(dict(schema="llmplay_package_v1"),
                                                             indent=2) + "\n")
        for name in ("PROMPT.md", "README.md"):
            shutil.copyfile(ROOT / "scripts" / "llmplay" / name, stage / name)
        copy_carry(stage, carry, partial)
        copy_reference(stage)
        shutil.copytree(ROOT / "scripts" / "llmplay" / "examples", stage / "examples")
        hits = scan(stage)
        if hits:
            raise ValueError("接口包命中密钥模式：%s" % hits)
        archive = Path(temporary) / "llmplay_pkg.7z"
        with py7zr.SevenZipFile(archive, "w", filters=[dict(id=py7zr.FILTER_LZMA2, preset=6)]) as bundle:
            bundle.writeall(stage, arcname="llmplay_pkg")
        volumes = []
        # 标准 7z 分卷为连续字节切片；体积远低于每卷 100 MB 的上限。
        with archive.open("rb") as stream:
            index = 1
            while block := stream.read(20_000_000):
                target = out / ("llmplay_pkg.7z.%03d" % index)
                target.write_bytes(block)
                volumes.append(target)
                index += 1
        destination = out / "llmplay_pkg"
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(stage, destination)
        resources = resource_zip or previous.parent / "llmplay_res.zip"
        if resources.resolve() != (out / "llmplay_res.zip").resolve():
            shutil.copyfile(resources, out / "llmplay_res.zip")
        print(json.dumps([dict(file=str(path), bytes=path.stat().st_size) for path in volumes],
                         indent=2))
        return volumes


def _build(executable, out, cache_dir, resource_dir, temporary, audit, package_name, carry=None):
    stage = temporary / "stage"
    stage.mkdir()
    copy_modules(stage)
    copy_constant_sources(stage)
    (stage / "build").mkdir()
    shutil.copyfile(executable, stage / "build" / "pvz-portable")
    (stage / "build" / "pvz-portable").chmod(0o755)
    resource_stage = temporary / "resources"
    copy_resources(resource_stage, resource_dir)
    shutil.copytree(cache_dir / "cache64", resource_stage / "res" / "cache64")
    resource_zip = temporary / "llmplay_res.zip"
    _zip_tree(resource_stage, resource_zip)

    manifest = task_manifest()
    (stage / "configs").mkdir()
    (stage / "configs" / "human_demo_tasks.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (stage / "configs" / "extra_seeds.json").write_text(
        json.dumps(extra_seeds(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(SEED_PLAN, stage / "configs" / "llmplay_seeds.json")
    shutil.copyfile(ROBUSTNESS_PLAN, stage / "configs" / "llmplay_robustness.json")
    (stage / "configs" / package_name).write_text(json.dumps(dict(
        schema="llmplay_package_v1"), indent=2) + "\n")
    shutil.copyfile(ROOT / "scripts" / "llmplay" / "PROMPT.md", stage / "PROMPT.md")
    shutil.copyfile(ROOT / "scripts" / "llmplay" / "README.md", stage / "README.md")
    copy_carry(stage, carry)
    copy_reference(stage)
    shutil.copytree(ROOT / "scripts" / "llmplay" / "examples", stage / "examples")

    hits = scan(stage) + scan(resource_stage)
    (out / "secret_scan.txt").write_text(
        ("命中 %d 处\n" % len(hits)) + "".join(hit + "\n" for hit in hits), encoding="utf-8")
    if hits:
        raise SystemExit("包里命中密钥模式，停止打包（见 secret_scan.txt）：\n" + "\n".join(hits))

    interface_zip = temporary / "llmplay_pkg.zip"
    _zip_tree(stage, interface_zip, prefix="llmplay_pkg")
    for path in (interface_zip, resource_zip):
        if path.stat().st_size >= 512 * 1024 * 1024:
            raise ValueError("上传包超出512MB：%s" % path.name)
    # 所有构建/扫描/压缩完成后才替换正式文件；原 stage 内引擎作输入也安全。
    old_stage = out / "stage.previous"
    if old_stage.exists(): shutil.rmtree(old_stage)
    if (out / "stage").exists(): os.replace(out / "stage", old_stage)
    os.replace(stage, out / "stage")
    old_resources = out / "resources.previous"
    if old_resources.exists(): shutil.rmtree(old_resources)
    if (out / "resources").exists(): os.replace(out / "resources", old_resources)
    os.replace(resource_stage, out / "resources")
    os.replace(resource_zip, out / "llmplay_res.zip")
    os.replace(interface_zip, out / "llmplay_pkg.zip")
    for old in (old_stage, old_resources):
        if old.exists(): shutil.rmtree(old)
    archive = out / "llmplay_pkg.zip"
    print(json.dumps(dict(packages=[dict(zip=str(out / name), size=(out / name).stat().st_size)
                                   for name in ("llmplay_pkg.zip", "llmplay_res.zip")],
                          compiled_definitions=audit["actual"], secret_hits=0), ensure_ascii=False, indent=2))
    return archive


def _zip_tree(directory, archive, prefix=""):
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                bundle.write(path, Path(prefix) / path.relative_to(directory))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--reuse-package", type=Path, help="复用上一包 stage，仅重打接口为7z分卷")
    parser.add_argument("--carry", type=Path, help="前次 progress.json，打进包内 carry/ 供开局自动接上")
    parser.add_argument("--partial", type=Path, help="合入新会话 out.zip / out 目录 / progress.json；只读进度")
    parser.add_argument("--resource-zip", type=Path, help="原样沿用的资源 zip，默认上一包旁的 llmplay_res.zip")
    parser.add_argument("--resource-dir", type=Path, default=RESOURCE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-dir", type=Path,
                        help="cache_prepare.py 生成的 full-appdata（旁边须有 cache-audit.json）")
    args = parser.parse_args()
    if args.reuse_package:
        repack_interface(args.reuse_package, args.out, carry=args.carry,
                         partial=args.partial, resource_zip=args.resource_zip)
        return 0
    if not args.executable or not args.cache_dir:
        parser.error("首次构建需要 --executable 和 --cache-dir")
    build(args.executable, args.out, resource_dir=args.resource_dir, cache_dir=args.cache_dir,
          carry=args.carry)
    return 0


if __name__ == "__main__":
    # 构建只导入模块，不执行游戏包的开局初始化，不创建临时 out/ 进度或安装资源。
    package = types.ModuleType("llmplay")
    package.__path__ = [str(ROOT / "scripts" / "llmplay")]
    sys.modules["llmplay"] = package
    raise SystemExit(main())
