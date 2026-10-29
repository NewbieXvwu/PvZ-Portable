"""Remove only the 256 abandoned task20 view links; leave immutable pack segments alone."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from pvz_corpus_policy import ABANDONED_LABELS

SEED_RANGES = {'d10orig1_fast_search': (1880000, 1880128),
               'd10orig2_fast_search': (1880128, 1880256)}


def retire_links(view: Path, apply=False):
    removed = []
    for label in sorted(ABANDONED_LABELS):
        for seed in range(*SEED_RANGES[label]):
            path = view / f'seed_{seed}' / label
            if not path.is_symlink():
                continue
            # Do not turn a recovered or unrelated physical archive into a deletion.
            if (path / 'frames.npz').is_file() or (path / 'frames.jsonl').is_file():
                continue
            removed.append(dict(path=str(path), target=str(path.readlink())))
            if apply:
                path.unlink()  # directory link only; never follows it into an archive
    return removed


def restore_links(records):
    for record in records:
        path = Path(record['path'])
        if not path.exists() and not path.is_symlink():
            path.symlink_to(record['target'], target_is_directory=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view', type=Path, default=Path.home() / 'pvz-logs/curriculum_state.pipeline/frames')
    parser.add_argument('--apply', action='store_true', help='default: report only')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    result = dict(applied=args.apply, links=retire_links(args.view, args.apply),
                  abandoned_labels=sorted(ABANDONED_LABELS))
    if args.receipt:
        args.receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(applied=args.apply, retired_links=len(result['links']))))


if __name__ == '__main__':
    main()
