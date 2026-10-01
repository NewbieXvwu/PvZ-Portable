"""`scripts/hf_sync.py` 的暂存目录必须被清干净，而且绝不能碰源文件。

2026-10-01 在台式机上第一次真跑 push 时，上传本身成功、收尾崩了：
早期实现只对暂存目录里的**符号链接**解链接、再逐个 `rmdir`，漏掉了脚本自己写的
`MANIFEST.json`（普通文件），于是 `staging.rmdir()` 抛 `ENOTEMPTY`。
更早一版还有更坏的可能：如果改用会跟进符号链接的删除方式，
`staging/` 里那些指向真检查点的链接会把**源文件本身**删掉。
这两件事都要被钉住。
"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import hf_sync  # noqa: E402


class StubApi:
    """记录被调用的事实，不做任何网络访问。"""

    def __init__(self) -> None:
        self.created: list[dict] = []
        self.uploaded: list[dict] = []
        self.seen: list[str] = []

    def create_repo(self, **kwargs) -> None:
        self.created.append(kwargs)

    def upload_folder(self, **kwargs) -> None:
        self.uploaded.append(kwargs)
        folder = Path(kwargs["folder_path"])
        # 模拟真实上传：逐个读一遍暂存内容，顺便确认链接指向的文件真的存在。
        for entry in sorted(folder.rglob("*")):
            if entry.is_file():
                self.seen.append(str(entry.relative_to(folder)))
                entry.read_bytes()


def make_run(base: Path, name: str = "run_x") -> Path:
    run_dir = base / name
    (run_dir / "runs" / "run_1").mkdir(parents=True)
    (run_dir / "runs" / "run_1" / "update_000001_evaluated_1.pt").write_bytes(b"weights")
    (run_dir / "learning_curve.json").write_text('{"points": []}\n', encoding="utf-8")
    return run_dir


class StagingCleanup(unittest.TestCase):
    def setUp(self) -> None:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.base = Path(holder.name)
        self.api = StubApi()
        self._saved = (hf_sync.RESEARCH_DIR, hf_sync._require_hf)
        hf_sync.RESEARCH_DIR = self.base
        hf_sync._require_hf = lambda: (lambda **kw: self.api, None)  # type: ignore[assignment]
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        hf_sync.RESEARCH_DIR, hf_sync._require_hf = self._saved

    def push(self, name: str = "run_x") -> int:
        argv = sys.argv
        sys.argv = ["hf_sync.py", "--repo", "someone/repo", "push", name]
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                return hf_sync.main()
        finally:
            sys.argv = argv

    def test_staging_directory_is_removed(self) -> None:
        make_run(self.base)
        self.assertEqual(self.push(), 0)
        self.assertFalse((self.base / "run_x" / ".hf_staging").exists())

    def test_manifest_is_uploaded_not_only_the_symlinks(self) -> None:
        make_run(self.base)
        self.push()
        self.assertIn("MANIFEST.json", self.api.seen)
        self.assertIn("runs/run_1/update_000001_evaluated_1.pt", self.api.seen)
        self.assertIn("learning_curve.json", self.api.seen)

    def test_source_files_survive(self) -> None:
        """最关键的一条：清理暂存目录不能顺着符号链接删掉真检查点。"""
        run_dir = make_run(self.base)
        checkpoint = run_dir / "runs" / "run_1" / "update_000001_evaluated_1.pt"
        curve = run_dir / "learning_curve.json"
        self.push()
        self.assertEqual(checkpoint.read_bytes(), b"weights")
        self.assertEqual(curve.read_text(encoding="utf-8"), '{"points": []}\n')

    def test_stale_staging_from_a_previous_crash_is_recovered(self) -> None:
        """上次崩在收尾会留下 .hf_staging；下次 push 要能自愈，不能卡住。"""
        run_dir = make_run(self.base)
        stale = run_dir / ".hf_staging"
        (stale / "runs" / "run_1").mkdir(parents=True)
        (stale / "MANIFEST.json").write_text("{}\n", encoding="utf-8")
        (stale / "runs" / "run_1" / "leftover.pt").write_bytes(b"junk")

        self.assertEqual(self.push(), 0)
        self.assertFalse(stale.exists())
        # 自愈过程只该删暂存目录里的东西，不能碰到 run 目录本身的文件。
        self.assertTrue((run_dir / "learning_curve.json").exists())

    def test_repo_is_created_private(self) -> None:
        make_run(self.base)
        self.push()
        self.assertEqual(len(self.api.created), 1)
        self.assertTrue(self.api.created[0]["private"])
        self.assertEqual(self.api.created[0]["repo_id"], "someone/repo")


if __name__ == "__main__":
    unittest.main()
