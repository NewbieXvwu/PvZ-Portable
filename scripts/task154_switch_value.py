"""task154 step 6: switch the production leaf value net at a group boundary, with one-command rollback.

The queue resolves its leaf value net once at start-up (curriculum_queue.py DEFAULT_VALUE ->
artifacts/value_net/v5/model.pt), so the only way to change it without editing production code is
to replace that file. This tool does exactly that, atomically, and keeps the old weights next to it:

    python3 scripts/task154_switch_value.py status
    python3 scripts/task154_switch_value.py switch --candidate artifacts/value_net/v6/model.pt
    python3 scripts/task154_switch_value.py rollback

"switch" refuses while a round is still collecting: the queue is mid-round and a game launched
after the swap would score its leaves with the new net while the rest of the round used the old one.
Wait until every group of the current round is "done", then run it; the queue picks the new file up
when the main control restarts it. --force overrides the check (emergency rollback only).

Nothing here restarts the queue, touches its state, or writes outside artifacts/value_net/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = Path("/home/newbiexvwu/PvZ-search/artifacts/value_net/v5/model.pt")
STATE = Path("/home/newbiexvwu/PvZ-search/artifacts/value_net/switch_state.json")
QUEUE_STATE = Path("/home/newbiexvwu/pvz-logs/curriculum_state.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def queue_boundary() -> dict:
    """Is the queue between rounds? Read-only view of its own state file."""
    try:
        state = json.loads(QUEUE_STATE.read_text())
    except (OSError, ValueError):
        return {"round": None, "collecting": None}
    rounds = [entry for entry in state.get("log") or [] if entry.get("round") == state.get("round")]
    if not rounds:
        return {"round": state.get("round"), "collecting": None}
    groups = rounds[-1].get("groups") or []
    pending = [group.get("label") for group in groups if group.get("status") != "done"]
    return {"round": state.get("round"), "collecting": bool(pending),
            "pending_groups": len(pending), "groups": len(groups)}


def install(source: Path, target: Path) -> None:
    """Copy through a sibling temporary file, then replace: a reader never sees a half-written net."""
    temporary = target.with_name(target.name + ".incoming")
    shutil.copy2(source, temporary)
    if sha256(temporary) != sha256(source):
        temporary.unlink(missing_ok=True)
        raise SystemExit("copy verification failed")
    os.replace(temporary, target)


def status() -> dict:
    document = {
        "production": str(PRODUCTION),
        "production_sha256": sha256(PRODUCTION) if PRODUCTION.exists() else None,
        "state": json.loads(STATE.read_text()) if STATE.exists() else None,
        "queue": queue_boundary(),
    }
    print(json.dumps(document, indent=1))
    return document


def switch(candidate: Path, force: bool) -> None:
    if not candidate.is_file():
        raise SystemExit(f"candidate not found: {candidate}")
    boundary = queue_boundary()
    if boundary.get("collecting") and not force:
        raise SystemExit(f"round {boundary.get('round')} still collecting "
                         f"({boundary.get('pending_groups')}/{boundary.get('groups')} groups open); "
                         "wait for the group boundary or pass --force")
    previous = sha256(PRODUCTION)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = PRODUCTION.with_name(f"model.pt.bak-{stamp}")
    if backup.exists():
        raise SystemExit(f"backup already exists: {backup}")
    shutil.copy2(PRODUCTION, backup)
    if sha256(backup) != previous:
        raise SystemExit("backup verification failed; nothing was changed")
    install(candidate, PRODUCTION)
    document = {
        "switched_at": stamp,
        "previous_sha256": previous,
        "previous_backup": str(backup),
        "current_sha256": sha256(PRODUCTION),
        "candidate": str(candidate),
        "queue_round_at_switch": boundary.get("round"),
        "note": "the running queue keeps the old net until the main control restarts it",
    }
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(document, indent=1))
    print(json.dumps(document, indent=1))


def rollback() -> None:
    if not STATE.exists():
        raise SystemExit("no switch recorded; nothing to roll back")
    document = json.loads(STATE.read_text())
    backup = Path(document["previous_backup"])
    if not backup.is_file():
        raise SystemExit(f"backup missing: {backup}")
    if sha256(backup) != document["previous_sha256"]:
        raise SystemExit("backup does not match the recorded sha256; refusing")
    install(backup, PRODUCTION)
    if sha256(PRODUCTION) != document["previous_sha256"]:
        raise SystemExit("rollback verification failed")
    document["rolled_back_at"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    document["current_sha256"] = sha256(PRODUCTION)
    STATE.write_text(json.dumps(document, indent=1))
    print(json.dumps(document, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("status", "switch", "rollback"))
    parser.add_argument("--candidate", type=Path, default=ROOT / "artifacts/value_net/v6/model.pt")
    parser.add_argument("--force", action="store_true", help="switch even if the round is still collecting")
    args = parser.parse_args()
    if args.action == "status":
        status()
    elif args.action == "switch":
        switch(args.candidate, args.force)
    else:
        rollback()


if __name__ == "__main__":
    main()
