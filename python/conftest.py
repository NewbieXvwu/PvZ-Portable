"""task140: 会话级护栏，只覆盖曾经把机器写爆的三个文件。

2026-10-28：两个 pytest 同时整套跑 test_curriculum_queue.py /
test_curriculum_pipeline.py / test_qc_resilience.py，其中一个用例在重试循环里
不停写 curriculum.log（单文件 9.3 GB），Mac 磁盘写满、内存耗尽、系统冻结。

护栏做三件事，都只在这三个文件上生效，正常用例的代价是每个测试两次系统调用：

1. 单测试超时（默认 120 s，PVZ_TEST_TIMEOUT 可覆盖）——卡死的循环直接失败，
   不会一直写到磁盘满。
2. 单文件写入上限（RLIMIT_FSIZE 256 MB）——跑飞的日志写到 256 MB 就被内核打断，
   不会吃掉整个盘。子进程继承这个限制。
3. 测试结束后检查 tmp 目录总量（512 MB）——超了就明确失败，而不是留下一个
   没人看的巨型目录。

不做全量重放、不扫全盘、不改动其他文件。
"""
import os
import resource
import signal
from pathlib import Path

import pytest

GUARDED_FILES = frozenset({
    'test_curriculum_queue.py',
    'test_curriculum_pipeline.py',
    'test_qc_resilience.py',
})
TEST_TIMEOUT_SECONDS = int(os.environ.get('PVZ_TEST_TIMEOUT', '120'))
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_TMP_BYTES = 512 * 1024 * 1024


def _module_name(item) -> str:
    path = getattr(item, 'path', None) or getattr(item, 'fspath', None)
    return Path(str(path)).name if path is not None else ''


def _expire(signum, frame):
    raise TimeoutError(f'task140 guard: test exceeded {TEST_TIMEOUT_SECONDS}s; '
                       'a runaway retry loop is the usual cause')


def _tmp_bytes(item) -> int:
    directory = getattr(item, 'tmp_path', None)
    if not isinstance(directory, Path) or not directory.exists():
        return 0
    total = 0
    for root, _, files in os.walk(directory):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    if _module_name(item) not in GUARDED_FILES or not hasattr(signal, 'setitimer'):
        yield
        return
    previous = signal.signal(signal.SIGALRM, _expire)
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_BYTES, hard))
    signal.setitimer(signal.ITIMER_REAL, TEST_TIMEOUT_SECONDS)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))


def pytest_runtest_teardown(item):
    if _module_name(item) not in GUARDED_FILES:
        return
    written = _tmp_bytes(item)
    if written > MAX_TMP_BYTES:
        pytest.fail(f'task140 guard: {_module_name(item)} wrote {written / 1024 ** 2:.0f} MB '
                    f'into {item.tmp_path} (limit {MAX_TMP_BYTES / 1024 ** 2:.0f} MB)',
                    pytrace=False)
