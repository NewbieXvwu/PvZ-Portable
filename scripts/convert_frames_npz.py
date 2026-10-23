"""Convert JSONL episode archives to frames.npz, verify every frame, then delete the JSONL.

The npz writer (python/pvz_frame_store.write_frames_from_source) keeps the exact JSON text as bytes
and adds the cached training rows, so a converted archive is byte-for-byte the same source plus a
precomputed cache. This tool is the *bulk* version for cold archives:

  * it only touches archives older than --min-age-minutes (a live collector appending to its own
    frames.jsonl is never converted underneath it);
  * it writes frames.npz through a temp file and an atomic rename;
  * it verifies the result before anything is deleted -- the stored bytes must equal the JSONL bytes,
    every frame must parse to an equal dict, the row index must list exactly the labelled frames, and
    the encoder signature must be the current one;
  * it deletes the JSONL only for archives that passed, and records each one in the manifest.

Reading stays backward compatible: python/pvz_frame_store.frame_paths/frame_bytes still read JSONL
(registered as LEGACY(jsonl-archives) in python/pvz_compat.py).

    python3 scripts/convert_frames_npz.py --data /path/to/artifacts --dry-run
    python3 scripts/convert_frames_npz.py --data /path/to/artifacts --jobs 4 --delete \
        --report /path/to/report.json --manifest /path/to/manifest.txt.gz
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from pvz_frame_store import encoder_signature, save_arrays, write_frames_from_source  # noqa: E402


def parse_frames(source: bytes) -> list[dict]:
    return [json.loads(line) for line in source.splitlines() if line.strip()]


def labelled_rows(frames: list[dict]) -> list[int]:
    return [index for index in range(1, len(frames))
            if (frames[index].get("label") or frames[index].get("action"))
            and not frames[index].get("action_rejected")]


def verify(source: bytes, npz: Path) -> str | None:
    """Why the archive does not hold ``source`` losslessly, or None when it does."""
    try:
        with np.load(npz, allow_pickle=False) as stored:
            keys = set(stored.files)
            if "frames" not in keys:
                return "missing-frames"
            stored_bytes = stored["frames"].tobytes()
            if stored_bytes != source:
                return "source-bytes-differ"
            if "encoder" not in keys:
                return "missing-encoder"
            if str(stored["encoder"]) != encoder_signature():
                return "encoder-signature"
            rows = stored["rows"]
            hints = stored["hints"]
            shapes = {name: tuple(stored[name].shape)
                      for name in ("boards", "scalars", "plant", "packet", "cell", "values") if name in keys}
    except Exception as error:  # noqa: BLE001 - any failure means "not verified"
        return f"unreadable: {error}"
    frames = parse_frames(source)
    if parse_frames(stored_bytes) != frames:
        return "frames-differ"
    if len(hints) != len(frames):
        return "hints-length"
    expected = labelled_rows(frames)
    if rows.tolist() != expected:
        return "rows-mismatch"
    for name in ("boards", "scalars", "plant", "packet", "cell", "values"):
        if name not in shapes:
            return f"missing-{name}"
        if shapes[name][0] != len(expected):
            return f"{name}-rows"
    if len(expected) and (rows[0] < 1 or rows[-1] >= len(frames)):
        return "rows-range"
    return None


def convert_one(job: tuple[str, float, bool, bool]) -> dict:
    path_text, min_age, delete, dry_run = job
    source_path = Path(path_text)
    result = {"path": path_text, "status": "skipped", "jsonl_bytes": 0, "npz_bytes": 0, "reason": ""}
    try:
        stat = source_path.stat()
    except FileNotFoundError:
        result["reason"] = "missing"
        return result
    result["jsonl_bytes"] = stat.st_size
    age = time.time() - stat.st_mtime
    if age < min_age:
        result["reason"] = f"recent ({age / 60:.0f} min)"
        return result
    target = source_path.with_suffix(".npz")
    if target.exists():
        if verify(source_path.read_bytes(), target) is None:
            result.update(status="already", npz_bytes=target.stat().st_size)
            if delete and not dry_run:
                source_path.unlink()
                result["status"] = "already+deleted"
            return result
        result.update(status="failed", reason="existing npz does not verify")
        return result
    if dry_run:
        result.update(status="would-convert")
        return result
    source = source_path.read_bytes()
    if len(source) != stat.st_size:
        result.update(status="failed", reason="size changed while reading")
        return result
    temporary = source_path.with_name(f".{source_path.name}.{os.getpid()}.tmp.npz")
    try:
        write_frames_from_source(temporary, source)
        problem = verify(source, temporary)
        if problem is not None:
            result.update(status="failed", reason=problem)
            return result
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    result.update(status="converted", npz_bytes=target.stat().st_size)
    if delete:
        source_path.unlink()
        result["status"] = "converted+deleted"
    return result


def archive_arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


def arrays_equal(before: dict, after: dict) -> str | None:
    if set(before) != set(after):
        return "keys-differ"
    for name, array in before.items():
        other = after[name]
        if array.dtype != other.dtype or array.shape != other.shape:
            return f"{name}-dtype-shape"
        if not np.array_equal(array, other, equal_nan=array.dtype.kind == "f"):
            return f"{name}-values"
    return None


def recompress_one(job: tuple[str, float, bool]) -> dict:
    """Rewrite a deflated npz as LZMA, byte-identical in every array."""
    path_text, min_age, dry_run = job
    path = Path(path_text)
    result = {"path": path_text, "status": "skipped", "before": 0, "after": 0, "reason": ""}
    try:
        stat = path.stat()
    except FileNotFoundError:
        result["reason"] = "missing"
        return result
    result["before"] = stat.st_size
    if time.time() - stat.st_mtime < min_age:
        result["reason"] = "recent"
        return result
    try:
        with zipfile.ZipFile(path) as archive:
            methods = {info.compress_type for info in archive.infolist()}
    except zipfile.BadZipFile:
        result["reason"] = "not-a-zip"
        return result
    if zipfile.ZIP_LZMA in methods and zipfile.ZIP_DEFLATED not in methods:
        result["status"] = "already"
        return result
    if zipfile.ZIP_STORED in methods:
        # Training packs are ZIP_STORED so np.load(mmap_mode="r") can map them; leave them alone.
        result["reason"] = "stored (pack)"
        return result
    if dry_run:
        result["status"] = "would-recompress"
        return result
    try:
        before = archive_arrays(path)
    except Exception as error:  # noqa: BLE001
        result.update(status="failed", reason=f"unreadable: {error}")
        return result
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        save_arrays(temporary, **before)
        problem = arrays_equal(before, archive_arrays(temporary))
        if problem is not None:
            result.update(status="failed", reason=problem)
            return result
        os.replace(temporary, path)
    except Exception as error:  # noqa: BLE001
        result.update(status="failed", reason=f"write: {error}")
        return result
    finally:
        temporary.unlink(missing_ok=True)
    result.update(status="recompressed", after=path.stat().st_size)
    return result


def recompress_main(args) -> int:
    started = time.time()
    files = []
    for root in args.data:
        files.extend(sorted(root.rglob("*.npz")))
    files = sorted(set(files))
    total = sum(path.stat().st_size for path in files)
    print(f"{len(files)} npz files, {total / 2 ** 30:.2f} GiB", flush=True)
    jobs = [(str(path), args.min_age_minutes * 60, args.dry_run) for path in files]
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for index, result in enumerate(pool.map(recompress_one, jobs, chunksize=8), 1):
            results.append(result)
            if index % 500 == 0 or index == len(jobs):
                done = [r for r in results if r["status"] == "recompressed"]
                print(f"{index}/{len(jobs)}  recompressed {len(done)}  "
                      f"{sum(r['before'] for r in done) / 2 ** 20:.0f} -> "
                      f"{sum(r['after'] for r in done) / 2 ** 20:.0f} MiB", flush=True)
    by_status: dict[str, int] = {}
    for result in results:
        by_status[result["status"]] = by_status.get(result["status"], 0) + 1
    done = [r for r in results if r["status"] == "recompressed"]
    before = sum(r["before"] for r in done)
    after = sum(r["after"] for r in done)
    summary = {
        "mode": "recompress",
        "files": len(results),
        "by_status": by_status,
        "before_bytes": before,
        "after_bytes": after,
        "ratio": (after / before) if before else None,
        "freed_bytes": before - after,
        "dry_run": bool(args.dry_run),
        "seconds": round(time.time() - started, 1),
        "failures": [{"path": r["path"], "reason": r["reason"]} for r in results if r["status"] == "failed"],
    }
    print(json.dumps({key: summary[key] for key in
                      ("files", "by_status", "before_bytes", "after_bytes", "ratio", "freed_bytes", "seconds")},
                     ensure_ascii=False, indent=2), flush=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(f"report {args.report}", flush=True)
    return 1 if summary["failures"] else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, action="append", required=True,
                        help="root to scan for */*/frames.jsonl (repeatable)")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--min-age-minutes", type=float, default=60.0,
                        help="skip archives modified more recently than this (live collectors)")
    parser.add_argument("--delete", action="store_true", help="delete the JSONL after it verifies")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--recompress", action="store_true",
                        help="rewrite deflated npz archives as LZMA (every array verified equal); packs stay ZIP_STORED")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="where to write the converted/deleted list (gzip text)")
    args = parser.parse_args()

    if args.recompress:
        return recompress_main(args)

    started = time.time()
    sources = []
    for root in args.data:
        sources.extend(sorted(root.rglob("frames.jsonl")))
    print(f"{len(sources)} JSONL archives under {', '.join(str(root) for root in args.data)}", flush=True)
    total_bytes = sum(path.stat().st_size for path in sources)
    print(f"{total_bytes / 2 ** 30:.2f} GiB of JSONL", flush=True)

    jobs = [(str(path), args.min_age_minutes * 60, args.delete, args.dry_run) for path in sources]
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for index, result in enumerate(pool.map(convert_one, jobs, chunksize=4), 1):
            results.append(result)
            if index % 100 == 0 or index == len(jobs):
                done = [r for r in results if r["status"].startswith("converted")]
                print(f"{index}/{len(jobs)}  converted {len(done)}  "
                      f"{sum(r['jsonl_bytes'] for r in done) / 2 ** 20:.0f} MiB JSONL -> "
                      f"{sum(r['npz_bytes'] for r in done) / 2 ** 20:.0f} MiB npz", flush=True)

    by_status: dict[str, int] = {}
    for result in results:
        by_status[result["status"]] = by_status.get(result["status"], 0) + 1
    done = [r for r in results if r["status"].startswith("converted")]
    failures = [r for r in results if r["status"] == "failed"]
    jsonl_bytes = sum(r["jsonl_bytes"] for r in done)
    npz_bytes = sum(r["npz_bytes"] for r in done)
    summary = {
        "archives": len(results),
        "by_status": by_status,
        "jsonl_bytes": jsonl_bytes,
        "npz_bytes": npz_bytes,
        "ratio": (npz_bytes / jsonl_bytes) if jsonl_bytes else None,
        "deleted": bool(args.delete),
        "dry_run": bool(args.dry_run),
        "seconds": round(time.time() - started, 1),
        "failures": [{"path": r["path"], "reason": r["reason"]} for r in failures],
    }
    print(json.dumps({key: summary[key] for key in
                      ("archives", "by_status", "jsonl_bytes", "npz_bytes", "ratio", "seconds")},
                     ensure_ascii=False, indent=2), flush=True)

    if args.manifest is not None and done:
        lines = "".join(f"{r['jsonl_bytes']}\t{r['npz_bytes']}\t{r['status']}\t{r['path']}\n"
                        for r in sorted(done, key=lambda r: r["path"]))
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with gzip.GzipFile(args.manifest, "wb", mtime=0) as handle:
            handle.write(lines.encode())
        summary["manifest"] = str(args.manifest)
        summary["manifest_sha256"] = hashlib.sha256(lines.encode()).hexdigest()
        print(f"manifest {args.manifest} ({len(done)} archives, sha256 {summary['manifest_sha256'][:16]})", flush=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(f"report {args.report}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
