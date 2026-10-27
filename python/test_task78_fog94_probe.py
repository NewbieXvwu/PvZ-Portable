"""The 94-state probe: cell enumeration, direction signs and the stability gate."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "python")]
import task78_fog94_probe as probe


def test_cell_key_is_row_then_column():
    assert probe.cell_key(dict(type="plant", row=0, col=6)) == "0,6"
    assert probe.cell_key(dict(type="wait", ticks=60)) == "wait"


def test_cells_of_keeps_one_action_per_legal_cactus_cell():
    obs = dict(legal_actions=dict(plants=[
        dict(packet=8, row=1, col=7), dict(packet=3, row=4, col=6),
        dict(packet=8, row=0, col=6)]))
    actions = probe.cells_of(obs)
    assert [(a["row"], a["col"]) for a in actions] == [(0, 6), (1, 7)]
    assert all(a["packet"] == 8 and a["type"] == "plant" for a in actions)


def test_cells_of_is_empty_without_a_legal_cactus():
    assert probe.cells_of(dict(legal_actions=dict(plants=[dict(packet=2, row=0, col=0)]))) == []


def test_signs_marks_missing_and_tied_horizons():
    values = {"2500": {"wait": 0.1, "0,6": 0.2}, "4000": {"wait": 0.1}, "6000": {"wait": 0.1, "0,6": 0.1}}
    assert probe.signs(values, [0, 6]) == {2500: 1, 4000: None, 6000: 0}


def test_takeovers_partition_the_94_reference_states():
    reference = ROOT / "artifacts/task78/task72_rebuild_mac_branch1/n229/f101/signatures.jsonl"
    if not reference.exists():
        pytest.skip("frozen reconstruction signatures are not on this machine")
    seen = []
    for takeover in probe.TAKEOVERS:
        states = probe.takeover_states(takeover)
        assert all(key[1] == row["decision"] and key[2] == row["tick"] for key, row in states.items())
        seen.extend(states)
    assert len(seen) == 94
    assert len(set(seen)) == 94


def _state(seed, decision, tick, best, gaps):
    return dict(task="demo_fog_33_cactus_snow_m200", seed=seed, decision=decision, tick=tick, takeover=229,
                wave=7, sun=135, balloons=2, balloon_x=[764], n_cells=1, cells=[[0, 6]],
                static_best=best, chosen=json.dumps(dict(type="wait", ticks=60, until="timeout")),
                values={str(horizon): {"wait": 0.5, "%d,%d" % tuple(best): 0.5 + gap}
                        for horizon, gap in zip(probe.HORIZONS, gaps)})


def _write(out, label, rows):
    (out / label).mkdir(parents=True, exist_ok=True)
    with (out / label / "n229_f101.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + chr(10))


def test_score_counts_flips_and_skips_zero_directions(tmp_path, monkeypatch):
    rows = [_state(101, 14, 15498, [0, 6], (0.01, 0.01, 0.01)),      # no flip
            _state(101, 15, 15558, [0, 6], (0.01, -0.01, 0.01)),     # flip
            _state(101, 16, 15618, [0, 6], (0.0, 0.01, 0.01))]       # one exact zero
    monkeypatch.setattr(probe, "reference_rows", lambda: rows)
    monkeypatch.setattr(probe, "TAKEOVERS", (229,))
    _write(tmp_path, "v5r", rows)
    _write(tmp_path, "v6", rows)
    report = probe.score(out=tmp_path)
    assert report["states"] == 3 and report["measured_states"] == 3
    assert report["arms"]["v5r"]["evaluable"] == 3
    assert report["arms"]["v5r"]["zero_direction"] == 1
    assert report["arms"]["v5r"]["directional"] == 2
    assert report["arms"]["v5r"]["flips"] == 1
    assert report["paired_states"] == 2
    assert report["gate"]["passed"] is False
    assert report["gate"]["upper_bound_at_most_zero"] is True


def test_compare_reports_missing_states(tmp_path, monkeypatch):
    rows = [_state(101, 14, 15498, [0, 6], (0.01, 0.01, 0.01))]
    monkeypatch.setattr(probe, "reference_rows", lambda: rows + [_state(101, 15, 15558, [0, 6], (0, 0, 0))])
    monkeypatch.setattr(probe, "TAKEOVERS", (229,))
    _write(tmp_path, "v5r", rows)
    report = probe.compare("v5r", 2500, out=tmp_path)
    assert report["verdict"] == "state_mismatch"
    assert report["matched"] == 1 and len(report["missing"]) == 1


def test_compare_reports_a_different_candidate_set(tmp_path, monkeypatch):
    reference = _state(101, 14, 15498, [0, 6], (0.01, 0.01, 0.01))
    measured = json.loads(json.dumps(reference))
    measured["values"]["2500"]["1,7"] = 0.51
    monkeypatch.setattr(probe, "reference_rows", lambda: [reference])
    monkeypatch.setattr(probe, "TAKEOVERS", (229,))
    _write(tmp_path, "v5r", [measured])
    report = probe.compare("v5r", 2500, out=tmp_path)
    assert report["verdict"] == "candidate_mismatch"
    assert len(report["candidate_set_mismatch"]) == 1


def test_compare_reports_a_reproduced_value(tmp_path, monkeypatch):
    reference = _state(101, 14, 15498, [0, 6], (0.01, 0.01, 0.01))
    monkeypatch.setattr(probe, "reference_rows", lambda: [reference])
    monkeypatch.setattr(probe, "TAKEOVERS", (229,))
    _write(tmp_path, "v5r", [json.loads(json.dumps(reference))])
    report = probe.compare("v5r", 2500, out=tmp_path)
    assert report["verdict"] == "reproduced"
    assert report["max_abs_diff"] == 0.0
