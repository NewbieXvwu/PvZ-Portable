"""task193 import handoff: reuse first-import verification, never schedule scans.

Installed beside the existing task105 importer; its SHA receipt implementation
stays authoritative. The merged import index is the existing curriculum source.
"""
import fcntl
import json
from pathlib import Path


def no_scan_plan(*args, **kwargs):
    return dict(schema='llm_scan_jobs_v1', archives=[],
                future_block=dict(base=0, stride=0, count=0))


def main_with(args, importer):
    # Both legacy.run and SHA.run_sha call this existing hook. No seed inventory,
    # allocation, takeover games, or separate scan_jobs file is needed now.
    previous_scan = importer.scan_plan
    importer.scan_plan = no_scan_plan
    try:
        if not args.no_verify:
            try:
                from llm_import_sha import run_sha
            except ImportError:
                result = importer.run(args.input, args.dest, args.manifest, args.layers,
                    args.resource_dir, args.executable, True, args.future_base, seed_ledger=args.seed_ledger)
            else:
                result = run_sha(args.input, args.dest, args.manifest, args.layers, args.resource_dir,
                    args.executable or importer.ROOT / 'build/pvz-portable', args.future_base, args.seed_ledger,
                    verify_workers=getattr(args, 'verify_workers', 2))
        else:
            result = importer.run(args.input, args.dest, args.manifest, args.layers, args.resource_dir,
                args.executable, False, args.future_base, seed_ledger=args.seed_ledger)
    finally:
        importer.scan_plan = previous_scan
    merge_rows(args.dest, result['rows'])
    report = importer.render_report(result['rows'], result['layers'], no_scan_plan())
    report += '\ntask193: 已直接入库；生产下轮读取卡组/示范，未加入离线扫描队列。\n'
    print(report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report)
    return int(any(row['verify'] in ('失败', '拒绝') for row in result['rows']))


def merge_rows(dest, new_rows):
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / 'import.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        merged = dest / 'merged_progress.json'
        previous = json.loads(merged.read_text()).get('archives', []) if merged.exists() else []
        rows = {row.get('archive_sha256') or row.get('archive') or row['source']: row for row in previous}
        for row in new_rows:
            key = row.get('archive_sha256') or row.get('archive') or row['source']
            if row.get('archive') or key not in rows:
                rows[key] = row
        temporary = merged.with_suffix('.building')
        temporary.write_text(json.dumps(dict(cache='sha_once', archives=list(rows.values())), ensure_ascii=False) + '\n')
        temporary.replace(merged)


def batch_main(root, engine):
    """Existing batch CLI becomes import-only; it never restarts scan supervision."""
    import argparse
    import subprocess
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--first-import-index', type=Path)
    parser.add_argument('--completed-index', type=Path)
    parser.add_argument('--staged-manifest', type=Path)
    args = parser.parse_args()
    inputs = args.inputs
    if args.completed_index or args.first_import_index:
        index = json.loads((args.completed_index or args.first_import_index).read_text())
        merge_rows(root / 'artifacts/task105/import', index['archives'])
        inputs = [] if args.completed_index else inputs[1:]
    for source in inputs:
        code = subprocess.run([sys.executable, str(root / 'scripts/import_llm_run.py'), str(source),
            '--dest', str(root / 'artifacts/task105/import'), '--executable', str(engine)]).returncode
        if code not in (0, 1):
            raise SystemExit(code)
    print('task193: batch imported; no takeover scan or supervisor scheduled', flush=True)
