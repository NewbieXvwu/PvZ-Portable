"""task36 step 3: one row-weight control = frozen dw5i recipe + --row-weights, then panel v2.

Only the row multipliers differ from dw5i_s1 (same frozen corpus, same 12-epoch recipe,
same seed, same width). No new switches: --row-weights is the existing mechanism.

The frozen command still carries the options of the archive loader (--cache-dir /
--read-cache-dir / --precompute-workers), which the packed loader replaced; they are removed
here. The pack itself is built from the same row options as the training command, with the
trainer's own option tuple (python/pvz_training_pack.py), so both sides cannot drift.

The training process is paused whenever MemAvailable falls below 10 GiB: the course
collector shares this machine and must keep running.

  python3 scripts/run_task36_mix.py --plan A [--skip-panel] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASK27 = ROOT / "artifacts/task27"
OUT = ROOT / "artifacts/task36"
PYTHON = "/home/newbiexvwu/.venvs/ml/bin/python3"
MEMORY_FLOOR_KIB = 10 * 1024 * 1024  # MemAvailable must stay >= 10 GiB (course collection)

sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
import dagger_experiment_runner as guard  # noqa: E402  validated process-tree memory guard
from pvz_training_pack import archive_fingerprint, pack_options, pack_path  # noqa: E402

guard.OUT = ROOT / "artifacts/task36"

REMOVED = {"--cache-dir", "--read-cache-dir", "--precompute-workers"}


def sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip_flags(command: list[str]) -> list[str]:
    """Drop the archive-loader options, with their values; nothing else changes."""
    kept, index = [], 0
    while index < len(command):
        if command[index] in REMOVED:
            index += 2
        else:
            kept.append(command[index])
            index += 1
    return kept


def value(command: list[str], flag: str, default=None):
    return command[command.index(flag) + 1] if flag in command else default


def pack_stage(command: list[str]) -> tuple[list[str], Path] | None:
    """The packer command for exactly this training command's row options, if the pack is missing."""
    data = Path(value(command, "--data"))
    labels = set(value(command, "--labels").split(","))
    wait_keep = float(value(command, "--wait-keep", 1.0))
    obs = value(command, "--obs", "v1")
    cells = bool(float(value(command, "--cell-soft-weight", 0.0)))
    aux_horizon = int(value(command, "--aux-horizon", 8)) if float(value(command, "--aux-weight", 0.0)) else 0
    packet_head = value(command, "--packet-head", "slot")
    with_context = "--context" in command
    label_weight = {k: float(v) for k, v in
                    (kv.split("=") for kv in value(command, "--label-weight", "").split(",") if kv)}
    options = pack_options(labels, wait_keep, obs, cells, aux_horizon, packet_head, with_context,
                           label_weight, sources=archive_fingerprint(data, labels))
    target = pack_path(data, options)
    if target.is_file():
        print(json.dumps(dict(pack=str(target), bytes=target.stat().st_size, build="cached")), flush=True)
        return None
    argv = [PYTHON, str(ROOT / "scripts/pack_training_rows.py"), "--data", str(data),
            "--labels", value(command, "--labels")]
    if wait_keep != 1.0:
        argv += ["--wait-keep", str(wait_keep)]
    if obs != "v1":
        argv += ["--obs", obs]
    if cells:
        argv += ["--cells"]
    if aux_horizon:
        argv += ["--aux-horizon", str(aux_horizon)]
    if packet_head != "slot":
        argv += ["--packet-head", packet_head]
    if with_context:
        argv += ["--context"]
    if label_weight:
        argv += ["--label-weight", value(command, "--label-weight")]
    argv += ["--cache-dir", str(TASK27 / "cache"), "--precompute-workers", "8"]
    return argv, target


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
    frozen = json.loads((TASK27 / "training_s1.command.json").read_text())["command"]
    command = strip_flags(list(frozen))
    command[command.index("--out") + 1] = str(out)
    command += ["--row-weights", str(spec)]
    command[command.index("--source-row-report") + 1] = str(out / "source_rows.json")
    out.mkdir(parents=True, exist_ok=True)
    (OUT / ("training_%s.command.json" % args.plan)).write_text(
        json.dumps(dict(command=command, cwd=str(ROOT)), indent=2) + "\n")

    def low_memory(_owned):
        return guard.available() < MEMORY_FLOOR_KIB

    stages = []
    pack = pack_stage(frozen)
    if pack is not None:
        stages.append(("pack_%s" % args.plan, *pack))
    stages.append(("train", command, out))
    if not args.skip_panel:
        stages.append(("panel_v2", [
            PYTHON, str(ROOT / "scripts/unified_panel.py"), "--version", "v2",
            "--manifest", str(TASK27 / "unified_v2.json"),
            "--rollout-policy", str(out / "model.pt"),
            "--workers", str(args.workers),
            "--out", str(OUT / ("panel_dw5i_%s_v2" % args.plan)),
        ], OUT / ("panel_dw5i_%s_v2" % args.plan)))
    for name, argv, target in stages:
        print(json.dumps(dict(stage=name, argv=argv,
                              log=str(OUT / ("%s_%s.txt" % (name, args.plan))))), flush=True)
        if args.dry_run:
            continue
        result = guard.run(argv, "%s_%s" % (name, args.plan), cwd=ROOT,
                           pause_when=low_memory if name == "train" else None)
        result.update(stage=name, plan=args.plan)
        print(json.dumps(result), flush=True)
        if result["returncode"]:
            raise SystemExit(result["returncode"])
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
