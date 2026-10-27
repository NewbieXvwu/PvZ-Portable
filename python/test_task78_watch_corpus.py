"""The corpus watcher must never start the formal path before the check passes."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "python")]
import task78_watch_corpus as watch


def test_missing_corpus_does_not_start_the_formal_path(monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(watch, "STOP", tmp_path / "stop")
    monkeypatch.setattr(watch, "corpus_ready", lambda *a, **k: (False, "missing"))
    monkeypatch.setattr(watch, "run_formal", lambda: started.append(1) or 0)
    monkeypatch.setattr(sys, "argv", ["watch", "--once"])
    assert watch.main() == 1
    assert started == []


def test_ready_corpus_runs_the_formal_path_once(monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(watch, "STOP", tmp_path / "stop")
    monkeypatch.setattr(watch, "corpus_ready", lambda *a, **k: (True, ""))
    monkeypatch.setattr(watch, "run_formal", lambda: started.append(1) or 7)
    monkeypatch.setattr(sys, "argv", ["watch", "--once"])
    assert watch.main() == 7
    assert started == [1]


def test_stop_file_ends_the_loop_without_checking(monkeypatch, tmp_path):
    stop = tmp_path / "stop"
    stop.write_text("")
    monkeypatch.setattr(watch, "STOP", stop)
    monkeypatch.setattr(watch, "corpus_ready", lambda *a, **k: (_ for _ in ()).throw(AssertionError("checked")))
    monkeypatch.setattr(sys, "argv", ["watch", "--once"])
    assert watch.main() == 0


def test_source_root_builds_the_descriptor_before_checking(monkeypatch, tmp_path):
    calls = []

    class Result:
        def __init__(self, code):
            self.returncode = code
            self.stdout = "{}"

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return Result(0)

    monkeypatch.setattr(watch, "DESCRIPTOR", tmp_path / "corpus.json")
    monkeypatch.setattr(watch.subprocess, "run", fake_run)
    ready, detail = watch.corpus_ready(tmp_path / "dropped")
    assert ready is True and "row counts" in detail
    assert "--write-descriptor" in calls[0] and str(tmp_path / "dropped") in calls[0]
    assert "--verify-rows" in calls[1]


def test_build_failure_keeps_the_watcher_waiting(monkeypatch, tmp_path):
    class Result:
        def __init__(self, code):
            self.returncode = code
            self.stdout = "{}"

    def fake_run(argv, **kwargs):
        return Result(1)

    monkeypatch.setattr(watch, "DESCRIPTOR", tmp_path / "corpus.json")
    monkeypatch.setattr(watch.subprocess, "run", fake_run)
    ready, detail = watch.corpus_ready(tmp_path / "dropped")
    assert ready is False and "descriptor build failed" in detail
