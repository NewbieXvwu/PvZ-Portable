"""The corpus watcher must never start the formal path before the check passes."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "python")]
import task78_watch_corpus as watch


def test_missing_corpus_does_not_start_the_formal_path(monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(watch, "STOP", tmp_path / "stop")
    monkeypatch.setattr(watch, "corpus_ready", lambda: (False, "missing"))
    monkeypatch.setattr(watch, "run_formal", lambda: started.append(1) or 0)
    monkeypatch.setattr(sys, "argv", ["watch", "--once"])
    assert watch.main() == 1
    assert started == []


def test_ready_corpus_runs_the_formal_path_once(monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(watch, "STOP", tmp_path / "stop")
    monkeypatch.setattr(watch, "corpus_ready", lambda: (True, ""))
    monkeypatch.setattr(watch, "run_formal", lambda: started.append(1) or 7)
    monkeypatch.setattr(sys, "argv", ["watch", "--once"])
    assert watch.main() == 7
    assert started == [1]


def test_stop_file_ends_the_loop_without_checking(monkeypatch, tmp_path):
    stop = tmp_path / "stop"
    stop.write_text("")
    monkeypatch.setattr(watch, "STOP", stop)
    monkeypatch.setattr(watch, "corpus_ready", lambda: (_ for _ in ()).throw(AssertionError("checked")))
    monkeypatch.setattr(sys, "argv", ["watch", "--once"])
    assert watch.main() == 0
