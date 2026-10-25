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

import pvz_tasks  # noqa: E402

PYTHON_MODULES = ["pvz_env.py", "pvz_common.py"]
# pvz_constants.py 是**从 C++ 源码现场解析**常量表的（AGENTS 的约定：不许手抄表），
# 所以包里必须带上它读的那几个文件，否则 import 阶段就抛"常量来源文件不存在"。
CONSTANT_SOURCES = ["src/ConstEnums.h", "src/Lawn/Plant.cpp", "src/Lawn/Zombie.cpp",
                    "src/Lawn/Board.cpp", "src/SexyAppFramework/SexyAppBase.cpp"]
SCRIPT_MODULES = ["pvz_constants.py", "pvz_tasks.py", "render_episode.py", "episode_query.py"]
RESOURCE_DIR = Path("/home/newbiexvwu/.cache/pvz-research-resources")
DEFAULT_OUT = Path.home() / "pvz-logs" / "llmplay_pkg"
TARGET_LEVELS = (41, 42, 43, 44, 45, 46, 47, 48, 49, 50)

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


def task_manifest() -> dict:
    """包里给的任务：屋顶各族（41..50 里已登记的）+ L41 冒烟任务。

    卡组不手写：L41 用 pvz_tasks.deck_for_level(41) 现解析（AGENTS §8）。
    """
    repo = json.loads((ROOT / "configs" / "human_demo_tasks.json").read_text(encoding="utf-8"))
    tasks = [task for task in repo["tasks"]
             if task["level"] in TARGET_LEVELS and task["task_id"].startswith("demo_roof_")]
    if not any(task["level"] == 41 for task in tasks):
        tasks.append(dict(task_id="smoke_roof_41", level=41, seeds=[20261041],
                          zombie_count_multiplier=1.0,
                          deck=list(pvz_tasks.deck_for_level(41))))
    tasks.sort(key=lambda task: (task["level"], task["task_id"]))
    return dict(schema=repo.get("schema", "human_demo_tasks_v1"), tasks=tasks)


def extra_seeds(manifest: dict) -> dict:
    """同一关卡再打几个种子的清单（默认种子 +1 / +2）。"""
    out = {}
    for task in manifest["tasks"]:
        base = task["seeds"][0]
        out[task["task_id"]] = [base, base + 1, base + 2]
    return out


def copy_modules(stage: Path) -> None:
    (stage / "python").mkdir(parents=True, exist_ok=True)
    (stage / "scripts").mkdir(parents=True, exist_ok=True)
    for name in PYTHON_MODULES:
        shutil.copyfile(ROOT / "python" / name, stage / "python" / name)
    for name in SCRIPT_MODULES:
        shutil.copyfile(ROOT / "scripts" / name, stage / "scripts" / name)
    shutil.copytree(ROOT / "scripts" / "llmplay", stage / "llmplay",
                    ignore=shutil.ignore_patterns("__pycache__", "build_package.py",
                                                  "build_probe.py", "probe"))


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
    if out.exists():
        shutil.rmtree(out)
    stage = out / "stage"
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
