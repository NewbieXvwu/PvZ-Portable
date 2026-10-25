"""把"LLM 打全程"沙箱包打出来：引擎 + 资源 + 接口 + 任务书。

    python3 scripts/llmplay/build_package.py --executable <静态引擎> [--out DIR]

产物落在 ~/pvz-logs/llmplay_pkg/（不进 git、不上 HF）：
llmplay_pkg.zip 与解压后的 stage/（方便自查与复现）。

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
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

PYTHON_MODULES = ["pvz_env.py", "pvz_common.py"]
# pvz_constants.py 是**从 C++ 源码现场解析**常量表的（AGENTS 的约定：不许手抄表），
# 所以包里必须带上它读的那几个文件，否则 import 阶段就抛"常量来源文件不存在"。
CONSTANT_SOURCES = ["src/ConstEnums.h", "src/Lawn/Plant.cpp", "src/Lawn/Zombie.cpp",
                    "src/Lawn/Board.cpp", "src/SexyAppFramework/SexyAppBase.cpp"]
SCRIPT_MODULES = ["pvz_constants.py", "pvz_tasks.py", "render_episode.py", "episode_query.py"]
RESOURCE_DIR = Path("/home/newbiexvwu/.cache/pvz-research-resources")
DEFAULT_OUT = Path.home() / "pvz-logs" / "llmplay_pkg"
SEED_PLAN = ROOT / "configs" / "llmplay_seeds.json"
ROBUSTNESS_PLAN = ROOT / "configs" / "llmplay_robustness.json"
LAYERS = ("A", "B", "C", "D", "E")

SECRET_PATTERNS = [re.compile(pattern) for pattern in
                   (rb"hf_[A-Za-z0-9]{20,}", rb"sk-[A-Za-z0-9]{20,}", rb"AIza[A-Za-z0-9_-]{30,}",
                    rb"ghp_[A-Za-z0-9]{30,}", rb"(?i)password\s*[:=]\s*\S+",
                    rb"(?i)token\s*[:=]\s*[A-Za-z0-9_\-]{16,}")]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_plan() -> dict:
    """task81c 的五层任务与 40 种子块（`configs/llmplay_seeds.json`，机械生成）。"""
    document = json.loads(SEED_PLAN.read_text(encoding="utf-8"))
    if document.get("schema") != "llmplay_seeds_v1":
        raise SystemExit("不是 llmplay_seeds_v1：%s" % SEED_PLAN)
    return document


def task_manifest() -> dict:
    """包里给的任务：task81c 的五层（A 屋顶 L46–49 → B L42–44 变体 + L41 →
    C 3.0/4.0 倍 → D 雾夜 L36–39 2.0 倍 → E 原生传送带）。

    任务定义与种子全部来自 `configs/llmplay_seeds.json`（`scripts/llmplay/
    make_llmplay_tasks.py` 生成）；卡组不在这里手写（AGENTS §8）。
    """
    repo = json.loads((ROOT / "configs" / "human_demo_tasks.json").read_text(encoding="utf-8"))
    by_id = {task["task_id"]: task for task in repo["tasks"]}
    plan = seed_plan()
    order = [task_id for layer in LAYERS for task_id in plan["layers"][layer]]
    missing = [task_id for task_id in order if task_id not in by_id]
    if missing:
        raise SystemExit("任务清单缺这些族，先跑 scripts/llmplay/make_llmplay_tasks.py：%s" % missing)
    tasks = []
    for task_id in order:
        task = dict(by_id[task_id])
        task["seeds"] = [plan["tasks"][task_id][0]]      # 默认种子 = 本任务块的第一个
        tasks.append(task)
    return dict(schema=repo.get("schema", "human_demo_tasks_v1"), tasks=tasks)


def extra_seeds(manifest: dict) -> dict:
    """每个任务 40 个种子（新独立块 81,000,000 起，每任务步长 1000）。"""
    plan = seed_plan()
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
    # 分不清看哪个；verify_archive.py 依赖仓库里的 human_demo/pvz_event_env，包里没有。
    shutil.copytree(ROOT / "scripts" / "llmplay", stage / "llmplay",
                    ignore=shutil.ignore_patterns("__pycache__", "build_package.py",
                                                  "build_probe.py", "probe",
                                                  "PROMPT.md", "README.md", "examples",
                                                  "verify_archive.py", "mac_check.sh",
                                                  "mac_bundle.sh"))


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


def build(executable: Path, out: Path, *, resource_dir: Path = RESOURCE_DIR) -> Path:
    # 只清自己的产物：out 下面还住着探路包（probe/），整个 rmtree 会把它删掉
    # ——2026-10-25 已经删过一次，探路包的 zip 就这么没的。
    stage = out / "stage"
    if stage.exists():
        shutil.rmtree(stage)
    for leftover in (out / "llmplay_pkg.zip", out / "secret_scan.txt"):
        if leftover.exists():
            leftover.unlink()
    stage.mkdir(parents=True)
    copy_modules(stage)
    copy_constant_sources(stage)
    (stage / "build").mkdir()
    shutil.copyfile(executable, stage / "build" / "pvz-portable")
    (stage / "build" / "pvz-portable").chmod(0o755)
    copy_resources(stage, resource_dir)

    manifest = task_manifest()
    (stage / "configs").mkdir()
    (stage / "configs" / "human_demo_tasks.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (stage / "configs" / "extra_seeds.json").write_text(
        json.dumps(extra_seeds(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(SEED_PLAN, stage / "configs" / "llmplay_seeds.json")
    shutil.copyfile(ROBUSTNESS_PLAN, stage / "configs" / "llmplay_robustness.json")
    shutil.copyfile(ROOT / "scripts" / "llmplay" / "PROMPT.md", stage / "PROMPT.md")
    shutil.copyfile(ROOT / "scripts" / "llmplay" / "README.md", stage / "README.md")
    shutil.copyfile(ROOT / "scripts" / "llmplay" / "selfcheck.py", stage / "selfcheck.py")
    shutil.copytree(ROOT / "scripts" / "llmplay" / "examples", stage / "examples")

    hits = scan(stage)
    (out / "secret_scan.txt").write_text(
        ("命中 %d 处\n" % len(hits)) + "".join(hit + "\n" for hit in hits), encoding="utf-8")
    if hits:
        raise SystemExit("包里命中密钥模式，停止打包（见 secret_scan.txt）：\n" + "\n".join(hits))

    archive = out / "llmplay_pkg.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                bundle.write(path, Path("llmplay_pkg") / path.relative_to(stage))
    print(json.dumps(dict(zip=str(archive), size=archive.stat().st_size,
                          sha256=sha256(archive),
                          files=sum(1 for p in stage.rglob("*") if p.is_file()),
                          secret_hits=0), ensure_ascii=False, indent=2))
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--resource-dir", type=Path, default=RESOURCE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build(args.executable, args.out, resource_dir=args.resource_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
