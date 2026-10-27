"""task81f：自动安装独立资源包，并给全部引擎入口提供持久 app-data。

只有正式包中的 configs/llmplay_package.json 启用自动安装；仓库普通 import 无副作用。
缓存按引擎 SHA 隔离，资源模板只读；每个会话的全部 Game/branch/robustness/verify 共用缓存。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
import zipfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path, PurePosixPath

from . import budget

MANIFEST = "llmplay_resources.json"
PACKAGE = "llmplay_package.json"
_READY = set()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _engine_sha(path, size, modified):
    return sha256(path)


@contextmanager
def _lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        if budget.fcntl is not None:
            budget.fcntl.flock(handle.fileno(), budget.fcntl.LOCK_EX)
        try:
            yield
        finally:
            if budget.fcntl is not None:
                budget.fcntl.flock(handle.fileno(), budget.fcntl.LOCK_UN)


def validate(directory, expected=None):
    """检查资源清单的每份文件，拒绝缺失/损坏或引擎不匹配的缓存。"""
    directory = Path(directory)
    manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("schema") != "llmplay_resources_v1":
        raise ValueError("资源清单 schema 不正确")
    if expected and manifest["engine_sha256"] != expected["engine_sha256"]:
        raise ValueError("预编译缓存与包内引擎不匹配")
    for name, digest in manifest["files"].items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError("资源清单含不安全路径")
        if sha256(directory / name) != digest:
            raise ValueError("资源文件损坏：%s" % name)
    for name in ("main.pak", "properties/partner.xml"):
        if name not in manifest["files"]:
            raise ValueError("资源清单缺少 %s" % name)
    return manifest


def _extract(archive, target):
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            path = PurePosixPath(entry.filename)
            if (path.is_absolute() or ".." in path.parts or "\\" in entry.filename
                    or ((entry.external_attr >> 16) & 0o170000) == 0o120000):
                raise ValueError("资源包含不安全路径或软链")
        bundle.extractall(target)


def install(root, archive=None):
    """首次导入自动解资源包。用户只需将 llmplay_res.zip 放在接口包旁边。"""
    root = Path(root)
    marker = root / "configs" / PACKAGE
    if not marker.is_file():
        return None                 # 仓库模式不安装、不要求游戏资源
    expected = json.loads(marker.read_text(encoding="utf-8"))
    target = root / "res"
    with _lock(root / ".resources.lock"):
        if (target / MANIFEST).is_file():
            validate(target, expected)
            return target
        override = os.environ.get("PVZ_LLMPLAY_RES_ZIP")
        candidates = ([Path(archive)] if archive else [Path(override)] if override else
                      [root.parent / "llmplay_res.zip", root / "llmplay_res.zip",
                       Path.cwd() / "llmplay_res.zip"])
        source = next((p for p in candidates if p.is_file()), None)
        if source is None:
            raise FileNotFoundError(
                "缺少 llmplay_res.zip。请同时上传 llmplay_pkg.zip 和 llmplay_res.zip，"
                "只解压接口包，把资源 zip 留在 llmplay_pkg 目录旁；随后重新 import llmplay。"
                "也可设置 PVZ_LLMPLAY_RES_ZIP 指向资源 zip。")
        if sha256(source) != expected["resource_zip_sha256"]:
            raise ValueError("llmplay_res.zip 的 SHA256 与接口包不匹配，请上传同一批的两个文件")
        temporary = Path(tempfile.mkdtemp(prefix=".resource-install-", dir=root))
        try:
            _extract(source, temporary)
            prepared = temporary / "res"
            validate(prepared, expected)
            if target.exists():
                raise ValueError("res 目录已存在但没有本批资源清单，请使用全新解压的接口包")
            os.replace(prepared, target)
        finally:
            shutil.rmtree(temporary)
    return target


def app_data(executable, resource_dir, session=None):
    """为当前会话提供按引擎隔离的持久目录，并原子安装预编译缓存。"""
    executable, resource_dir = Path(executable), Path(resource_dir)
    stat = executable.stat()
    engine = _engine_sha(str(executable.resolve()), stat.st_size, stat.st_mtime_ns)
    path = ((session.path if session else budget.default_path()).parent / "appdata" / engine)
    manifest_path = resource_dir / MANIFEST
    manifest_sha = sha256(manifest_path) if manifest_path.is_file() else None
    key = (str(path.resolve()), engine, manifest_sha)
    if key in _READY:
        return path
    with _lock(path.parent / (engine + ".lock")):
        path.mkdir(parents=True, exist_ok=True)
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["engine_sha256"] == engine:
                for name, digest in manifest["files"].items():
                    if not name.startswith("cache64/"):
                        continue
                    source, target = resource_dir / name, path / name
                    if target.is_file() and sha256(target) == digest:
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name("." + target.name + "." + uuid.uuid4().hex)
                    try:
                        shutil.copyfile(source, temporary)
                        if sha256(temporary) != digest:
                            raise ValueError("预编译缓存损坏：%s" % name)
                        os.replace(temporary, target)
                    finally:
                        temporary.unlink(missing_ok=True)
    _READY.add(key)
    return path
