"""task36 step 3: one row-weight control = frozen dw5i recipe + --row-weights, then panel v2.

Only the row multipliers differ from dw5i_s1 (same frozen corpus, same 12-epoch recipe,
same seed, same width). No new switches: --row-weights is the existing mechanism.

  python3 scripts/run_task36_mix.py --plan A [--skip-panel] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASK27 = ROOT / "artifacts/task27"
OUT = ROOT / "artifacts/task36"
PYTHON = "/home/newbiexvwu/.venvs/ml/bin/python3"


def sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--skip-panel", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=4, help="panel game processes (max 4)")
    args = parser.parse_args()

    out = OUT / ("dw5i_%s_s1" % args.plan)
    spec = OUT / ("row_weights_%s.json" % args.plan)
    if not spec.exists():
        raise SystemExit("missing %s: run scripts/task36_row_weights.py first" % spec)
    command = json.loads((TASK27 / "training_s1.command.json").read_text())["command"]
    command = list(command)
    command[command.index("--out") + 1] = str(out)
    command += ["--row-weights", str(spec)]
    command[command.index("--source-row-report") + 1] = str(out / "source_rows.json")
    out.mkdir(parents=True, exist_ok=True)
    (OUT / ("training_%s.command.json" % args.plan)).write_text(
        json.dumps(dict(command=command, cwd=str(ROOT)), indent=2) + "\n")

    stages = [("train", command, out)]
    if not args.skip_panel:
        stages.append(("panel_v2", [
            PYTHON, str(ROOT / "scripts/unified_panel.py"), "--version", "v2",
            "--manifest", str(TASK27 / "unified_v2.json"),
            "--rollout-policy", str(out / "model.pt"),
            "--workers", str(args.workers),
            "--out", str(OUT / ("panel_dw5i_%s_v2" % args.plan)),
        ], OUT / ("panel_dw5i_%s_v2" % args.plan)))
    for name, argv, target in stages:
        log = OUT / ("%s_%s.log" % (name, args.plan))
        print(json.dumps(dict(stage=name, argv=argv, log=str(log))), flush=True)
        if args.dry_run:
            continue
        started = time.time()
        with log.open("w") as handle:
            process = subprocess.run(argv, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT)
        result = dict(stage=name, plan=args.plan, returncode=process.returncode,
                      seconds=round(time.time() - started, 1), log=str(log))
        (OUT / ("%s_%s.result.json" % (name, args.plan))).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if process.returncode:
            raise SystemExit(process.returncode)
    if not args.dry_run:
        model = out / "model.pt"
        complete = dict(plan=args.plan, model=str(model),
                        model_sha256=sha(model) if model.exists() else None,
                        model_bytes=model.stat().st_size if model.exists() else None,
                        row_weights_sha256=sha(spec),
                        panel=str(OUT / ("panel_dw5i_%s_v2" % args.plan)),
                        finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        (OUT / ("complete_%s.json" % args.plan)).write_text(json.dumps(complete, indent=2) + "\n")
        print(json.dumps(complete), flush=True)


if __name__ == "__main__":
    main()
