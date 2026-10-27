"""Pick the student that plays best, not the one that imitates best (learn/06: same data, another training seed,
same held-out imitation loss, 140 fewer wins out of 1024).

Every candidate plays alone (C++, greedy) on selection seeds that are not evaluation seeds -- by default the
unused tail of the 1.4x and 1.5x training tasks -- and the one with the most wins is copied to --out.

Usage:
  python scripts/select_student.py --candidates 'artifacts/student/d12s*/model.pt' --out artifacts/student/v12
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_rows(path, expected_seeds):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if (len(rows) != len(expected_seeds) or {row.get('seed') for row in rows} != set(expected_seeds)
            or any(row.get('arm') != 'fast' or row.get('terminal') is not True
                   or type(row.get('won')) is not bool or row.get('searched', 0) != 0 for row in rows)):
        raise ValueError(f'incomplete/wrong selection games: {path}')
    return sum(row['won'] for row in rows)


def select(candidates, out, manifest, sets, workers, value, resource_dir, executable,
           run_command=None):
    """Pick by the existing total-win rule, preserving complete per-game evidence."""
    run_command = subprocess.run if run_command is None else run_command
    paths = sorted({str(Path(path).resolve()) for path in candidates})
    if not paths:
        raise ValueError('no candidates')
    document = json.loads(Path(manifest).read_text())
    table = {task['task_id']: task for task in document['tasks']}
    # task128: unseen-deck cells are evidence only. Their wins are recorded, but the
    # pick stays "most wins on the scored tasks" so they cannot dominate the total.
    score_only = {task['task_id'] for task in document['tasks'] if task.get('score_only')}
    work = Path(out) / 'selection'
    work.mkdir(parents=True, exist_ok=True)
    scores = {}
    for path in paths:
        model_sha = sha(path)
        wins, files = {}, {}
        for spec in sets.split(','):
            task, offset, count = spec.split(':')
            offset, count = int(offset), int(count)
            expected = table[task]['seeds'][offset:offset + count]
            if len(expected) != count or count < 1:
                raise ValueError(f'invalid selection slice: {spec}')
            folder = work / f'{model_sha[:16]}_{task}'
            command = [sys.executable, str(ROOT / 'scripts/search_play.py'), '--rollout-policy', path,
                       '--value', str(value), '--executable', str(executable), '--manifest', str(manifest),
                       '--task-id', task, '--seed-offset', str(offset), '--seeds', str(count),
                       '--arms', 'fast', '--workers', str(workers), '--out', str(folder)]
            if resource_dir:
                command += ['--resource-dir', str(resource_dir)]
            run_command(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            games = folder / 'games.jsonl'
            wins[task] = score_rows(games, expected)
            files[task] = dict(path=str(games), sha256=sha(games))
        if sha(path) != model_sha:
            raise ValueError(f'candidate changed during selection: {path}')
        scores[path] = dict(wins=wins, total=sum(value for name, value in wins.items() if name not in score_only),
                            score_only=dict(wins={name: wins[name] for name in sorted(score_only)},
                                            total=sum(wins[name] for name in score_only)),
                            model_sha256=model_sha, games=files)
        print(json.dumps(dict(candidate=path, **scores[path])), flush=True)
    best = max(paths, key=lambda path: scores[path]['total'])
    temporary = Path(out) / 'model.building'
    shutil.copyfile(best, temporary)
    temporary.replace(Path(out) / 'model.pt')
    result = dict(best=best, sets=sets, scores=scores, manifest=str(manifest),
                  manifest_sha256=sha(manifest), model_sha256=sha(best),
                  executable=str(executable), executable_sha256=sha(executable),
                  rule='maximum total wins over scored tasks (score-only unseen decks excluded); '
                       'stable candidate-path order breaks ties',
                  rejected=[path for path in paths if path != best])
    temporary = Path(out) / 'selection.building'
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(Path(out) / 'selection.json')
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates", required=True, help="comma-separated paths or globs")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--manifest", default=str(ROOT / "artifacts/stage4/day_l7_hard.json"))
    p.add_argument("--sets", default="day_l7_z14_train:768:256,day_l7_z15_train:384:256",
                   help="task:offset:count,... (keep them apart from evaluation seeds and from collection seeds)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--value", default=str(ROOT / "artifacts/value_net/v2/model.pt"), help="unused by the fast arm")
    p.add_argument("--resource-dir")
    p.add_argument("--executable", type=Path, default=ROOT / 'build/pvz-portable')
    args = p.parse_args()
    paths = sorted({q for pattern in args.candidates.split(",") for q in glob.glob(pattern)})
    if not paths:
        raise SystemExit("no candidates")
    result = select(paths, args.out, args.manifest, args.sets, args.workers, args.value,
                    args.resource_dir, args.executable)
    print(f"best {result['best']} -> {args.out / 'model.pt'}")


if __name__ == "__main__":
    main()
