"""Regenerate task27's three manifests locally and compare every task field and seed."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(current, generated):
    a, b = (json.loads(p.read_text())['tasks'] for p in (current, generated))
    left, right = ({t['task_id']: t for t in rows} for rows in (a, b))
    assert len(left) == len(a) and len(right) == len(b)
    missing = sorted(left.keys()-right.keys())
    extra = sorted(right.keys()-left.keys())
    changed = {name: sorted(k for k in left[name].keys() | right[name].keys()
                           if left[name].get(k) != right[name].get(k))
               for name in left.keys() & right.keys() if left[name] != right[name]}
    return dict(current=str(current), generated=str(generated),
                current_sha256=sha(current), generated_sha256=sha(generated),
                tasks=len(a), seeds=sum(len(t['seeds']) for t in a),
                missing=missing, extra=extra, changed=changed,
                task_order_equal=[t['task_id'] for t in a] == [t['task_id'] for t in b],
                passed=not (missing or extra or changed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current', type=Path, default=Path('/home/newbiexvwu/PvZ-search/artifacts/stage4'))
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts/task27/zone_check')
    args = parser.parse_args()
    assert args.current.is_absolute() and args.out.is_absolute()
    args.out.mkdir(parents=True, exist_ok=True)
    commands = (
        ('night_panel.json', 'make_night_panel.py', ['--combined']),
        ('night_panel_n10.json', 'make_night_panel.py',
         ['--deck', '0,1,2,3,4,5,7,8,9,11', '--suffix', '_n10']),
        ('pool_panel.json', 'make_pool_panel.py', ['--combined']),
    )
    results = []
    for name, script, flags in commands:
        current, generated = args.current/name, args.out/name
        before = sha(current)
        command = [sys.executable, str(ROOT/'scripts'/script), *flags, '--out', str(generated)]
        with (args.out/(name+'.txt')).open('w') as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
        result = compare(current, generated)
        assert sha(current) == before, 'shared manifest was modified'
        result['command'] = command
        results.append(result)
    report = dict(passed=all(r['passed'] for r in results), comparisons=results,
                  note='Metadata prose may differ. Every task field, including every seed, must match. Shared files are read-only.')
    (args.out/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    assert report['passed']


if __name__ == '__main__':
    main()
