"""Pick the student that plays best, not the one that imitates best (learn/06: same data, another training seed,
same held-out imitation loss, 140 fewer wins out of 1024).

Candidates and the frozen incumbent play alone on the same independent selection seeds.
Prefer predicted gate approvals, then total wins; if all fail, maximize the worst zone delta.
Historical manifests without an incumbent retain total-win selection.

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
sys.path.insert(0, str(ROOT / 'python'))
from pvz_promotion import decide_scored


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def outcomes(path, expected_seeds):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if (len(rows) != len(expected_seeds) or {row.get('seed') for row in rows} != set(expected_seeds)
            or any(row.get('arm') != 'fast' or row.get('terminal') is not True
                   or type(row.get('won')) is not bool or row.get('searched', 0) != 0 for row in rows)):
        raise ValueError(f'incomplete/wrong selection games: {path}')
    return {row['seed']: row['won'] for row in rows}


def score_rows(path, expected_seeds):
    return sum(outcomes(path, expected_seeds).values())


def selection_rank(score):
    prediction = score['prediction']
    worst = min(zone['delta_rate'] for zone in prediction['zones'].values())
    return prediction['approved'], score['total'] if prediction['approved'] else worst, score['total']


def select(candidates, out, manifest, sets, workers, value, resource_dir, executable,
           run_command=None):
    """Share one incumbent run across candidates, preserving per-game evidence."""
    run_command = subprocess.run if run_command is None else run_command
    paths = sorted({str(Path(path).resolve()) for path in candidates})
    if not paths:
        raise ValueError('no candidates')
    document = json.loads(Path(manifest).read_text())
    table = {task['task_id']: task for task in document['tasks']}
    score_only = {task['task_id'] for task in document['tasks'] if task.get('score_only')}
    incumbent = document.get('incumbent')
    if incumbent and sha(incumbent) != document['incumbent_sha256']:
        raise ValueError('frozen selection incumbent changed')
    work = Path(out) / 'selection'
    work.mkdir(parents=True, exist_ok=True)
    scores, incumbent_outcomes = {}, {}
    incumbent_score = None
    for path in ([incumbent] if incumbent else []) + paths:
        model_sha = sha(path)
        wins, files, pairs = {}, {}, []
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
            played = outcomes(games, expected)
            wins[task] = sum(played.values())
            files[task] = dict(path=str(games), sha256=sha(games))
            if incumbent_score is None and incumbent:
                incumbent_outcomes[task] = played
            elif incumbent and table[task].get('selection_group') == 'adventure':
                pairs.extend(dict(task_id=task, seed=seed, zone=table[task]['terrain'],
                                  candidate=played[seed], incumbent=incumbent_outcomes[task][seed])
                             for seed in expected)
        if sha(path) != model_sha:
            raise ValueError(f'candidate changed during selection: {path}')
        score = dict(wins=wins, total=sum(value for name, value in wins.items() if name not in score_only),
                     score_only=dict(wins={name: wins[name] for name in sorted(score_only)},
                                     total=sum(wins[name] for name in score_only)),
                     model_sha256=model_sha, games=files)
        if incumbent and incumbent_score is None:
            incumbent_score = score
            continue
        if incumbent:
            score['prediction'] = decide_scored(pairs)
        scores[path] = score
        print(json.dumps(dict(candidate=path, **score)), flush=True)
    best = max(paths, key=lambda path: selection_rank(scores[path]) if incumbent else scores[path]['total'])
    temporary = Path(out) / 'model.building'
    shutil.copyfile(best, temporary)
    temporary.replace(Path(out) / 'model.pt')
    result = dict(best=best, sets=sets, scores=scores, manifest=str(manifest),
                  manifest_sha256=sha(manifest), model_sha256=sha(best),
                  executable=str(executable), executable_sha256=sha(executable),
                  rule=('predicted gate approval, then maximum total wins; if none approved, '
                        'maximum worst-zone delta, then total wins; stable candidate-path ties'
                        if incumbent else 'maximum total wins over scored tasks; stable candidate-path ties'),
                  rejected=[path for path in paths if path != best])
    if incumbent:
        result.update(incumbent=dict(path=incumbent, **incumbent_score),
                      prediction_note='selection-seed estimate only; adventure pairs use the gate statistical '
                                      'rule; random and score-only tasks do not enter paired decisions')
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
