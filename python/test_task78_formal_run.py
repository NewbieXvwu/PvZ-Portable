"""The formal runner must gate on the corpus and resume without duplicating work."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "python")]
import task78_formal_run as runner


def test_plan_is_corpus_then_train_then_the_four_panel_steps():
    steps = runner.plan()
    assert [name for name, _ in steps] == ["corpus", "train", "freeze", "v5", "v6", "report"]
    train = dict(steps)["train"]
    assert "--corpus" in train and train[train.index("--corpus") + 1] == "original"
    corpus = steps[0][1]
    assert "--verify-rows" in corpus
    assert dict(steps)["v5"][-1] == "v5" and dict(steps)["report"][-1] == "report"


def test_missing_corpus_stops_before_training(monkeypatch):
    calls = []

    def fake(name, command, env=None):
        calls.append(name)
        return 1 if name == "corpus" else 0

    monkeypatch.setattr(runner, "run_step", fake)
    monkeypatch.setattr(runner, "trained", lambda: False)
    monkeypatch.setattr(sys, "argv", ["task78_formal_run.py"])
    assert runner.main() == 1
    assert calls == ["corpus"]


def test_training_is_skipped_when_the_candidate_exists(monkeypatch):
    calls = []

    def fake(name, command, env=None):
        calls.append(name)
        return 0

    monkeypatch.setattr(runner, "run_step", fake)
    monkeypatch.setattr(runner, "trained", lambda: True)
    monkeypatch.setattr(sys, "argv", ["task78_formal_run.py"])
    assert runner.main() == 0
    assert calls == ["corpus", "freeze", "v5", "v6", "report"]


def test_dry_run_executes_nothing(monkeypatch):
    monkeypatch.setattr(runner, "run_step", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran")))
    monkeypatch.setattr(sys, "argv", ["task78_formal_run.py", "--dry-run"])
    assert runner.main() == 0


def test_unknown_start_step_is_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["task78_formal_run.py", "--from", "nope"])
    try:
        runner.main()
    except SystemExit as error:
        assert "unknown step" in str(error)
    else:
        raise AssertionError("expected SystemExit")
