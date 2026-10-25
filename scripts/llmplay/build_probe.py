"""打"探路包"：源码 + 资源 + build.sh + selfcheck.py + 4 局示范数值存档。

    python3 scripts/llmplay/build_probe.py --prebuilt <静态引擎> [--out DIR]

产物：~/pvz-logs/llmplay_pkg/probe/llmplay_probe.zip（不进 git、不上 HF）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path.home() / "pvz-logs" / "llmplay_pkg" / "probe"
RESOURCE_DIR = Path("/home/newbiexvwu/.cache/pvz-research-resources")
DEMO_ROOT = Path("/home/newbiexvwu/PvZ-search/.claude/worktrees/cq/artifacts/human_demo/raw")
DEMO_NAMES = [
    ("seed_20261031", "demo_fog_31_snow_m200"),
    ("seed_20261032", "demo_fog_32_snow_m200"),
    ("seed_20261033", "demo_fog_33_cactus_snow_m200"),
    ("seed_20261033", "demo_fog_33_cattail_snow_m200"),
]
SOURCE_REVISION = "8bf65b19"   # deploy4 源码树 + task81 的 summon_counter 字段
MIMALLOC_SOURCE = Path("/home/newbiexvwu/llmplay-build/hostbuild2/_deps/mimalloc-src")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export_source(target: Path) -> None:
    target.mkdir(parents=True)
    archive = subprocess.run(["git", "-C", str(ROOT), "archive", SOURCE_REVISION,
                              "src", "CMakeLists.txt", "CMake"],
                             check=True, capture_output=True)
    subprocess.run(["tar", "-x", "-C", str(target)], input=archive.stdout, check=True)


def build(prebuilt: Path, out: Path, *, mimalloc: Path = MIMALLOC_SOURCE) -> Path:
    if out.exists():
        shutil.rmtree(out)
    stage = out / "stage"
    stage.mkdir(parents=True)
    export_source(stage / "src")
    shutil.copytree(mimalloc, stage / "vendor" / "mimalloc",
                    ignore=shutil.ignore_patterns(".git", "test", "docs", "ide"))
    (stage / "res" / "properties").mkdir(parents=True)
    for source in (RESOURCE_DIR / "main.pak", RESOURCE_DIR / "properties" / "partner.xml",
                   RESOURCE_DIR / "properties" / "partner_logo.jpg"):
        shutil.copyfile(source.resolve(), stage / "res" / source.relative_to(RESOURCE_DIR))
    (stage / "demos").mkdir()
    for seed_dir, name in DEMO_NAMES:
        source = DEMO_ROOT / seed_dir / name
        target = stage / "demos" / name
        target.mkdir()
        for filename in ("meta.json", "ticks.jsonl", "frames.jsonl"):
            if (source / filename).is_file():
                shutil.copyfile(source / filename, target / filename)
    (stage / "build_prebuilt").mkdir()
    shutil.copyfile(prebuilt, stage / "build_prebuilt" / "pvz-portable")
    (stage / "build_prebuilt" / "pvz-portable").chmod(0o755)
    for name in ("build.sh", "selfcheck.py", "README.md"):
        shutil.copyfile(ROOT / "scripts" / "llmplay" / "probe" / name, stage / name)
    (stage / "build.sh").chmod(0o755)
    (stage / "SOURCE.txt").write_text(
        "engine source revision: %s (deploy4, binary sha256 16989f23...)\n" % SOURCE_REVISION,
        encoding="utf-8")

    archive = out / "llmplay_probe.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                bundle.write(path, Path("llmplay_probe") / path.relative_to(stage))
    print(json.dumps(dict(zip=str(archive), size=archive.stat().st_size,
                          sha256=sha256(archive),
                          files=sum(1 for p in stage.rglob("*") if p.is_file()),
                          source_revision=SOURCE_REVISION), ensure_ascii=False, indent=2))
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prebuilt", type=Path, required=True)
    parser.add_argument("--mimalloc", type=Path, default=MIMALLOC_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build(args.prebuilt, args.out, mimalloc=args.mimalloc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
