"""Long-run continuation watches process identity and cannot advance on a missing file."""
import os
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
import fog_terminal_pipeline as pipeline


def test_verified_wait_checks_live_process_identity():
    identity=pipeline.process_identity(os.getpid())
    assert identity is not None and pipeline.process_alive(os.getpid(),identity)
    assert not pipeline.process_alive(os.getpid(),identity+'0')
    assert not pipeline.process_alive(999999999,identity)


def test_missing_screen_completion_cannot_start_confirmation(tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline,'OUT',tmp_path)
    pipeline.diag.save(tmp_path/'watch-screen.json',{'run':str(tmp_path/'unfinished')})
    calls=[]
    monkeypatch.setattr(pipeline.report,'confirmation_plan',lambda:calls.append('unexpected'))
    with pytest.raises(RuntimeError,match='without complete raw run'):pipeline.after_screen()
    assert calls==[]
