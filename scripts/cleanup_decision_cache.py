"""Release generated token files after all sixteen decision jobs have completed."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
from pvz_research_data import FIELDS
from gpu_window_runner import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGUSR1, stop); signal.signal(signal.SIGTERM, stop)
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = args.out/'progress.json'
    report = json.loads(receipt.read_text()) if receipt.exists() else dict(removed_bytes=0, removed_files=0,
        scope='generated tokens, row mask, binary source plan and research raw hardlink view; production originals, checkpoints and JSON receipts retained')
    for name in FIELDS:
        path = args.state/'decision-data/tokens'/(name+'.bin')
        if path.exists():
            report['removed_bytes'] += path.stat().st_size
            report['removed_files'] += 1
            path.unlink()
            write_json(receipt, report)
        if stopping:
            raise SystemExit(75)
    raw = args.state/'decision-data/split/raw'
    for parent, directories, files in os.walk(raw, topdown=False):
        for name in files:
            path = Path(parent)/name
            if name=='meta.json':
                report['removed_bytes'] += path.stat().st_size
            report['removed_files'] += 1
            path.unlink()  # drops this research-owned link; never follows it to a source
        Path(parent).rmdir()
        if stopping:
            write_json(receipt, report)
            raise SystemExit(75)
    for name in ('unseen.npy','sources.sqlite','sources.sqlite-wal','sources.sqlite-shm'):
        path=args.state/'decision-data/split'/name
        if path.exists():
            report['removed_bytes'] += path.stat().st_size
            report['removed_files'] += 1
            path.unlink()
    write_json(args.out/'ready.json', report)


if __name__ == '__main__':
    main()
