"""Rotate the large run logs under artifacts: gzip the head, keep the tail plain.

A rotated log keeps *all* of its bytes, split in two files:

    run.log.gz   everything before the last --keep-mb megabytes (one gzip member per rotation)
    run.log      the last --keep-mb megabytes, plain text, so tail/grep still work

Rotating again appends a new gzip member to run.log.gz, so a long-lived log ends up as one .gz that
decompresses to the whole history. Nothing is deleted.

Guards:

* files modified within --min-age-minutes are skipped -- a running collector may still be appending
  to its log, and truncating under an open append handle would lose the writes that follow;
* the plain tail starts on a line boundary (the first newline after the cut), so the tail alone is
  readable as text;
* the rewrite is a seek/write/truncate on the same inode, so a process that reopens the path sees the
  rotated file, and no rename races with a writer.

    # Mac
    python3 scripts/rotate_logs.py --root /Users/newbiexvwu/PvZAgent/artifacts --dry-run
    python3 scripts/rotate_logs.py --root /Users/newbiexvwu/PvZAgent/artifacts

    # desktop (WSL) -- same command, different root
    python3 scripts/rotate_logs.py --root ~/PvZAgent/artifacts
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import time

CHUNK = 8 << 20


def rotate(path: Path, min_bytes: int, keep_bytes: int, dry_run: bool) -> dict:
    stat = path.stat()
    size = stat.st_size
    result = {"path": str(path), "status": "skipped", "before": size, "after": size, "gz_bytes": 0}
    if size <= min_bytes:
        result["reason"] = "small"
        return result
    cut = size - keep_bytes
    if cut <= 0:
        result["reason"] = "keep covers file"
        return result
    if dry_run:
        result.update(status="would-rotate", after=keep_bytes)
        return result

    archive = path.with_name(path.name + ".gz")
    gz_before = archive.stat().st_size if archive.exists() else 0
    with path.open("rb") as source, gzip.GzipFile(archive, "ab", compresslevel=6, mtime=0) as sink:
        remaining = cut
        while remaining:
            block = source.read(min(CHUNK, remaining))
            if not block:
                break
            sink.write(block)
            remaining -= len(block)
        # start the plain tail on a line boundary: the cut may land inside a multi-byte character
        probe = source.read(4096)
        newline = probe.find(b"\n")
        if newline >= 0:
            sink.write(probe[:newline + 1])
            cut += newline + 1
            source.seek(cut)
        else:
            source.seek(cut)
        tail = source.read()
    with path.open("r+b") as handle:
        handle.seek(0)
        handle.write(tail)
        handle.truncate()
    result.update(status="rotated", after=path.stat().st_size,
                  gz_bytes=archive.stat().st_size - gz_before, gz_total=archive.stat().st_size)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, action="append", required=True, help="directory to scan (repeatable)")
    parser.add_argument("--glob", action="append", default=None,
                        help="file pattern relative to --root (default: **/*.log and **/*.out)")
    parser.add_argument("--min-mb", type=float, default=2.0, help="rotate files larger than this")
    parser.add_argument("--keep-mb", type=float, default=1.0, help="plain-text tail to keep")
    parser.add_argument("--min-age-minutes", type=float, default=120.0,
                        help="skip files modified more recently than this (live writers)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    patterns = args.glob or ["**/*.log", "**/*.out"]
    min_bytes = int(args.min_mb * 2 ** 20)
    keep_bytes = int(args.keep_mb * 2 ** 20)
    files = []
    for root in args.root:
        for pattern in patterns:
            files.extend(sorted(root.glob(pattern)))
    files = sorted(set(files))

    started = time.time()
    results, skipped_recent = [], 0
    for path in files:
        if not path.is_file() or path.suffix == ".gz":
            continue
        age_minutes = (time.time() - path.stat().st_mtime) / 60
        if age_minutes < args.min_age_minutes:
            skipped_recent += 1
            continue
        results.append(rotate(path, min_bytes, keep_bytes, args.dry_run))

    by_status: dict[str, int] = {}
    for result in results:
        by_status[result["status"]] = by_status.get(result["status"], 0) + 1
    rotated = [r for r in results if r["status"] == "rotated"]
    summary = {
        "files": len(files),
        "skipped_recent": skipped_recent,
        "by_status": by_status,
        "before_bytes": sum(r["before"] for r in rotated),
        "after_bytes": sum(r["after"] for r in rotated),
        "gz_bytes": sum(r["gz_bytes"] for r in rotated),
        "freed_bytes": sum(r["before"] - r["after"] - r["gz_bytes"] for r in rotated),
        "dry_run": bool(args.dry_run),
        "seconds": round(time.time() - started, 1),
        "rotated": [{"path": r["path"], "before": r["before"], "after": r["after"], "gz": r["gz_bytes"]}
                    for r in rotated],
    }
    print(json.dumps({key: summary[key] for key in
                      ("files", "skipped_recent", "by_status", "before_bytes", "after_bytes",
                       "gz_bytes", "freed_bytes", "seconds")}, ensure_ascii=False, indent=2))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(f"report {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
