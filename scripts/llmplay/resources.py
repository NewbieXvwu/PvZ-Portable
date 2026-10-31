"""task81f：自动安装独立资源包，并给全部引擎入口提供持久 app-data。

只有正式包中的 configs/llmplay_package.json 启用自动安装；仓库普通 import 无副作用。
不做包内文件哈希校验：资源装好就直接用，缺文件由引擎自己报错。
"""
from __future__ import annotations

import os
import shutil
import tempfile
import uuid
import zipfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path, PurePosixPath

from . import budget

PACKAGE = "llmplay_package.json"
_READY = set()


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
    if not (root / "configs" / PACKAGE).is_file():
        return None                 # 仓库模式不安装、不要求游戏资源
    target = root / "res"
    if (target / "main.pak").is_file():
        return target
    with _lock(root / ".resources.lock"):
        if (target / "main.pak").is_file():
            return target
        override = os.environ.get("PVZ_LLMPLAY_RES_ZIP")
        candidates = ([Path(archive)] if archive else [Path(override)] if override else
                      [root.parent / "llmplay_res.zip", root / "llmplay_res.zip",
                       Path.cwd() / "llmplay_res.zip"])
        source = next((p for p in candidates if p.is_file()), None)
        if source is None:
            raise FileNotFoundError(
                "缺少 llmplay_res.zip。请把 llmplay_res.zip 放在接口包目录旁，然后重新 import llmplay；"
                "也可设置 PVZ_LLMPLAY_RES_ZIP 指向它。")
        temporary = Path(tempfile.mkdtemp(prefix=".resource-install-", dir=root))
        try:
            _extract(source, temporary)
            prepared = temporary / "res" if (temporary / "res").is_dir() else temporary
            if target.exists():
                shutil.rmtree(target)       # 上次装到一半的残留
            os.replace(prepared, target)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
    return target


@lru_cache(maxsize=8)
def _engine_key(path, size, modified):
    return "%x-%x" % (int(size), int(modified))


def app_data(executable, resource_dir, session=None):
    """为当前会话提供按引擎隔离的持久目录，并原子安装预编译缓存。"""
    executable, resource_dir = Path(executable), Path(resource_dir)
    stat = executable.stat()
    engine = _engine_key(str(executable.resolve()), stat.st_size, stat.st_mtime_ns)
    path = ((session.path if session else budget.default_path()).parent / "appdata" / engine)
    key = (str(path.resolve()), engine)
    if key in _READY:
        return path
    with _lock(path.parent / (engine + ".lock")):
        path.mkdir(parents=True, exist_ok=True)
        source_root = resource_dir / "cache64"
        if source_root.is_dir():
            for source in sorted(source_root.rglob("*")):
                if not source.is_file():
                    continue
                target = path / "cache64" / source.relative_to(source_root)
                if target.is_file() and target.stat().st_size == source.stat().st_size:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name("." + target.name + "." + uuid.uuid4().hex)
                try:
                    shutil.copyfile(source, temporary)
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
    _READY.add(key)
    return path
