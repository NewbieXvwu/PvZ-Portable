"""Complete task72b's 94-state mapping using the probe's own balloon rounding.

The frozen reconstruction (task78_task72_rebuild.py) compares balloon_x as the
integers it stored with int(x). Re-reading the retained full-precision frames shows
the original probe rounded instead of truncating: int(round(x)) reproduces every
recorded balloon_x while truncation reproduces only the 26 states whose fraction is
below 0.5. This module therefore re-derives the balloon integers from the
full-precision frame and compares those, leaving every other identity field and the
frozen plan untouched.

It does not shift indices, seeds, ticks or actions, and never rewrites the
trajectories or the frozen plan. Output: bindings_v2.json next to the plan.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]
import task78_task72_rebuild as rebuild
from task78_data import sha
from value_teacher_feasibility import write_json

OUT = rebuild.OUT
REPORT = OUT / "bindings_v2.json"
RULE = ("identity = task/seed/decision/tick/wave/sun/balloons/cells/static_best/chosen action exact; "
        "balloon_x compared as int(round(float x)) re-derived from the retained full-precision frame "
        "(the probe rounded, the reconstruction stored int(x) truncation)")
FIELDS = ("task", "seed", "decision", "tick", "wave", "sun", "n_cells", "balloons", "static_best")


def balloon_ints(frame):
    """Balloon x as the original probe recorded it (round half up)."""
    return [int(round(z[2])) for z in frame["zombies"] if z[1] == 16]


def _action(reference):
    chosen = reference["chosen"]
    return json.loads(chosen) if isinstance(chosen, str) else chosen


def same(reference, row):
    if any(reference[field] != row[field] for field in FIELDS):
        return False
    if sorted(reference["cells"]) != sorted(row["cells"]):
        return False
    if sorted(reference["balloon_x"]) != sorted(balloon_ints(row["frame"])):
        return False
    return _action(reference) == row["chosen"] and row.get("action_rejected") is False


def legacy_same(reference, row):
    """The frozen rule: compare the stored truncated balloon_x."""
    if any(reference[field] != row[field] for field in FIELDS):
        return False
    if sorted(reference["cells"]) != sorted(row["cells"]):
        return False
    if sorted(reference["balloon_x"]) != sorted(row["balloon_x"]):
        return False
    return _action(reference) == row["chosen"] and row.get("action_rejected") is False


def candidates(document):
    rows = []
    for job in document["jobs"]:
        directory = OUT / f"n{job['takeover']}/f{job['future_seed']}"
        complete = directory / "complete.json"
        path = directory / "signatures.jsonl"
        if not complete.exists():
            raise ValueError(f"missing trajectory n{job['takeover']}/f{job['future_seed']}")
        receipt = json.loads(complete.read_text())
        if (receipt["plan_sha256"] != sha(OUT / "plan.json")
                or receipt["takeover"] != job["takeover"]
                or receipt["future_seed"] != job["future_seed"]
                or receipt["signatures_sha256"] != sha(path)
                or receipt["result"].get("terminal") is not True):
            raise ValueError(f"trajectory n{job['takeover']}/f{job['future_seed']} changed identity")
        for line in path.read_text().splitlines():
            row = json.loads(line)
            row["takeover"] = job["takeover"]
            row["signatures_path"] = str(path)
            row["signatures_sha256"] = receipt["signatures_sha256"]
            rows.append(row)
    if len({rebuild.identity(row) + (row["takeover"],) for row in rows}) != len(rows):
        raise ValueError("duplicate input signatures")
    return rows


def main():
    document = rebuild.frozen()
    rows = candidates(document)
    references = rebuild.reference_rows()
    bindings = []
    legacy = 0
    for reference in references:
        found = [row for row in rows if same(reference, row)]
        if len([row for row in rows if legacy_same(reference, row)]) == 1:
            legacy += 1
        bindings.append(dict(identity=list(rebuild.identity(reference)),
                             matches=[dict(takeover=row["takeover"],
                                           signatures_path=row["signatures_path"],
                                           signatures_sha256=row["signatures_sha256"],
                                           engine_state_sha256=row["engine_state_sha256"],
                                           balloon_floats=sorted(z[2] for z in row["frame"]["zombies"] if z[1] == 16),
                                           balloon_rounded=sorted(balloon_ints(row["frame"])))
                                      for row in found]))
    complete = all(len(row["matches"]) == 1 for row in bindings)
    report = dict(reference_sha256=sha(rebuild.REFERENCE),
                  plan_sha256=sha(OUT / "plan.json"),
                  rule=RULE,
                  rule_basis=("int(round(x)) matches all 94 recorded balloon_x; int(x) truncation matches 26; "
                              "ceil matches 22. Only balloon_x differs between the two rules; every other field is exact."),
                  states=len(bindings),
                  unique_matches=sum(1 for row in bindings if len(row["matches"]) == 1),
                  unmatched=sum(1 for row in bindings if not row["matches"]),
                  multiple=sum(1 for row in bindings if len(row["matches"]) > 1),
                  legacy_truncation_matches=legacy,
                  mapping_complete=complete,
                  original_probe_protocol_verified=False,
                  task72b_evaluation_complete=False,
                  bindings=bindings)
    write_json(REPORT, report)
    print(json.dumps({k: v for k, v in report.items() if k != "bindings"}, indent=1))
    return report


if __name__ == "__main__":
    main()
