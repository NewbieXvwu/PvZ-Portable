"""Independently audit task50 A teacher evidence without launching any games.

Partial mode reports completed pairs. --strict requires all three 32-seed
before/after arms, needed shadow traces, unchanged inputs, and a passing original
task37 comparison. CPU and choice counts are recomputed from the raw evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from check_value_teacher import TASKS


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(out, manifests, strict=False):
    missing, evidence = [], {}
    plans = {}
    for phase in ('before', 'after'):
        path = out / f'{phase}-plan.json'
        if not path.is_file():
            missing.append(str(path))
            continue
        plan = plans[phase] = read(path)
        for name, expected in {**plan['contract'], **plan['implementation']}.items():
            assert sha(Path(name)) == expected, f'frozen input changed: {name}'
    if len(plans) == 2:
        assert plans['before']['contract'] == plans['after']['contract']
    report = {'status': 'in_progress', 'tasks': {}}
    for task, manifest in TASKS:
        definition = next(t for t in read(manifests / manifest)['tasks'] if t['task_id'] == task)
        result = dict(before_games=0, after_games=0, paired_games=0, shadow_games=0,
                      teacher_same=0, teacher_points=0, noise_same=0, noise_points=0,
                      before_wins=0, after_wins=0, improved=0, regressed=0,
                      before_cpu_s=0.0, after_cpu_s=0.0, search_flag_differences=0)
        seeds = set()
        for index in range(32):
            rows, traces = {}, {}
            seed = definition['seeds'][index]
            assert seed not in seeds
            seeds.add(seed)
            for phase in ('before', 'after'):
                directory = out / phase / task / f'seed_index_{index}'
                rp, tp = directory / 'result.json', directory / 'trace.json'
                if not rp.is_file() or not tp.is_file():
                    missing.append(str(rp))
                    continue
                row, trace = read(rp), read(tp)
                assert row['seed'] == seed and row['arm'] == 'fast_search'
                assert len(trace) == row['decisions'] and 0 < len(trace) < 3000
                assert sum(bool(t['searched']) for t in trace) == row['searched']
                assert sum('noise_choice' in t for t in trace) == row['noise_points']
                assert sum(t['noise_choice'] == t['chosen'] for t in trace if 'noise_choice' in t) == row['noise_agreement']
                if phase == 'before':
                    assert row['noise_points'] == (row['searched'] + 7) // 8
                else:
                    assert row['terminal'] and row['noise_points'] == 0
                for who in ('worker', 'simulator'):
                    cpu = row['host_cpu'][who]
                    assert math.isclose(cpu['total_s'], cpu['user_s'] + cpu['system_s'], abs_tol=1e-8)
                    expected = cpu['total_s'] - row['noise_cpu_s'][who]
                    assert expected > 0
                    assert math.isclose(expected, row['host_cpu_excluding_noise_s'][who], abs_tol=1e-8)
                rows[phase], traces[phase] = row, trace
                result[f'{phase}_games'] += 1
                evidence[str(rp)], evidence[str(tp)] = sha(rp), sha(tp)
            if len(rows) != 2:
                continue
            old, new = traces['before'], traces['after']
            roots = lambda trace: [(r['state_sha256'], r['future_seed']) for r in trace]
            if roots(old) != roots(new):
                directory = out / 'shadow' / task / f'seed_index_{index}'
                rp, tp = directory / 'result.json', directory / 'trace.json'
                if not rp.is_file() or not tp.is_file():
                    missing.append(str(rp))
                    continue
                assert read(out / 'shadow-plan.json') == plans['after']
                new = read(tp)
                assert len(new) == read(rp)['decisions'] and roots(new) == roots(old)
                assert all(b['executed'] == a['chosen'] for a, b in zip(old, new))
                result['shadow_games'] += 1
                evidence[str(rp)], evidence[str(tp)] = sha(rp), sha(tp)
            for a, b in zip(old, new):
                result['search_flag_differences'] += a['searched'] != b['searched']
                if a['searched']:
                    result['teacher_points'] += 1
                    result['teacher_same'] += a['chosen'] == b['chosen']
            before, after = rows['before'], rows['after']
            result['paired_games'] += 1
            result['noise_points'] += before['noise_points']
            result['noise_same'] += before['noise_agreement']
            result['before_wins'] += before['won']
            result['after_wins'] += after['won']
            result['improved'] += after['won'] and not before['won']
            result['regressed'] += before['won'] and not after['won']
            for phase in ('before', 'after'):
                result[f'{phase}_cpu_s'] += sum(rows[phase]['host_cpu_excluding_noise_s'].values())
        report['tasks'][task] = result
    comparison_path = out / 'comparison.json'
    if not comparison_path.is_file():
        missing.append(str(comparison_path))
    elif not missing:
        comparison = read(comparison_path)
        for task, result in report['tasks'].items():
            row = comparison[task]
            assert result['before_games'] == result['after_games'] == result['paired_games'] == row['games'] == 32
            for actual, expected in ((row['agreement'], result['teacher_same'] / result['teacher_points']),
                                     (row['noise_agreement'], result['noise_same'] / result['noise_points']),
                                     (row['cpu_ratio'], result['after_cpu_s'] / result['before_cpu_s'])):
                assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)
            assert row['decisions_compared'] == result['teacher_points']
            assert row['noise_points'] == result['noise_points']
            for key in ('improved', 'regressed'):
                assert row['wins'][key] == result[key]
            assert row['wins']['before'] == result['before_wins']
            assert row['wins']['after'] == result['after_wins']
            assert row['passed'] and row['agreement'] >= row['noise_agreement'] and row['cpu_ratio'] <= 1.02
        report['comparison'] = comparison
        report['status'] = 'passed'
    report['missing_count'] = len(missing)
    report['evidence_files'] = len(evidence)
    report['evidence_manifest_sha256'] = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    if strict:
        assert not missing and report['status'] == 'passed', f'incomplete teacher evidence: {len(missing)} items'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--manifest-dir', type=Path, required=True)
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = audit(args.out, args.manifest_dir, args.strict)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
