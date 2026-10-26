#!/usr/bin/env python3
"""Independently audit task105 numerical artifacts and prepare a deployment plan.

No deployment is performed. Raw archives, engine images, model files and policies
remain local. The committed evidence contains only hashes and numerical summaries.
"""
import argparse
import collections
import json
import re
import subprocess
import tempfile
from pathlib import Path

import import_llm_run as importer
import takeover_scan as scan
from llm_import_sha import archive_sha, file_sha, save, verification_context
from task105_pipeline import ENGINE, ENGINE_SHA

ROOT = Path(__file__).resolve().parents[1]
BASELINE = '4645d578aefe119cca587e743c22bb3e5dc92f2c'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    out = ROOT / 'artifacts/task105'
    require(file_sha(ENGINE) == ENGINE_SHA, 'deploy14 engine changed')
    expected_context = verification_context(ENGINE, importer.DEFAULT_RESOURCES)
    progress = json.loads((out / 'import/import_progress.json').read_text())
    rows = progress['archives']
    plan = json.loads((out / 'scan_jobs.json').read_text())
    tasks = importer.load_manifest(ROOT / 'configs/human_demo_tasks.json')
    old = json.loads(subprocess.check_output(
        ['git', '-C', str(ROOT), 'show', BASELINE + ':configs/human_demo_tasks.json']))['tasks']
    require(all(tasks[t['task_id']] == t for t in old), 'a pre-task105 task was changed')
    with tempfile.TemporaryDirectory(prefix='task105_audit_') as directory:
        source = Path(directory)
        importer.safe_extract(args.input, source)
        archives = importer.discover(source)
        by_source = {str(a.relative_to(source)): a for a in archives}
        require(len(rows) == len(archives), 'archive count differs from the input package')
        if file_sha(args.input) == '676fb355128be6742d5023f365526396612755d5da6b362144faeaedf5cd0ee7':
            require(len(archives) == 191, 'first package must have 191 archives')
        require({r['source'] for r in rows} == set(by_source), 'not every input archive has a result')
        for row in rows:
            require(archive_sha(by_source[row['source']]) == row['archive_sha256'], 'input SHA mismatch')
    accepted = {r['task_id']: r for r in rows if r['archive'] and r['grade'] != 'loss'}
    require({a['task_id'] for a in plan['archives']} == set(accepted), 'scan plan omits accepted archives')
    receipts = out / 'import/receipts'
    task_summary = collections.defaultdict(lambda: dict(archives=0, won=set(), robust=set(),
                                                         rejected=0, reasons=collections.Counter()))
    for ids in json.loads(importer.DEFAULT_LAYERS.read_text())['layers'].values():
        for task_id in ids:
            task_summary[task_id]
    for row in rows:
        cell = task_summary[row['base_task_id']]
        cell['archives'] += 1
        receipt = json.loads((receipts / f'{row["archive_sha256"]}.json').read_text())
        require(receipt['context'] == expected_context, 'verification context differs from deploy14')
        require(receipt['archive_sha256'] == row['archive_sha256'], 'receipt SHA mismatch')
        if row['verify'] == '通过':
            require(receipt['verified'], 'accepted archive has a failed receipt')
            require(row['grade'] != 'loss', 'first package contains only claimed wins')
            cell['won'].add(row['seed'])
            if row['grade'] == 'full':
                cell['robust'].add(row['seed'])
        else:
            cell['rejected'] += 1
            cell['reasons'][row['note']] += 1
    additions, rates, outcomes, seeds, fingerprints = [], {}, [], [], {}
    for entry in plan['archives']:
        task_id = entry['task_id']
        row = accepted[task_id]
        archive = ROOT / entry['archive']
        require(archive_sha(archive) == row['archive_sha256'], 'stored archive SHA mismatch')
        require(json.loads((archive / 'meta.json').read_text())['verified'], 'stored archive is unverified')
        path = out / 'scan/results' / f'{task_id}_s{entry["seed"]}.json'
        result = json.loads(path.read_text())
        scan.validate_result(result)
        require(result['complete'] and result['archive'] == entry['archive']
                and result['task_id'] == task_id and result['seed'] == entry['seed'], 'result identity mismatch')
        require(result['future_seeds'] == entry['future_seeds'] and len(entry['future_seeds']) == 8,
                'scan futures differ from the reserved block')
        trace = result['adaptive_trace']
        require(trace['complete'] and trace['algorithm'] == 'coarse4_guard_budget9_v2', 'adaptive trace missing')
        require(trace['teacher_points'] <= 9 and trace['scan_games'] < 96, 'adaptive budget exceeded')
        games = 0
        for decision, bucket in trace['points'].items():
            for arm, summary in bucket.items():
                require(summary == result['points'][str(decision)][arm], 'trace differs from result')
                scan.validate_summary(summary, result['future_seeds'])
                job = dict(task_id=task_id, seed=entry['seed'], decision=int(decision), arm=arm)
                job_dir = scan.job_out(out / 'scan', job)
                actual = [r for r in scan.read_rows(job_dir)
                          if r.get('future_seed') in entry['future_seeds']
                          and r.get('arm') == scan.ARM_SPEC[arm][0] and r.get('seed') == entry['seed']]
                actual.sort(key=lambda r: entry['future_seeds'].index(r['future_seed']))
                require(scan.point_summary(actual) == summary, 'raw games differ from scan result')
                context = json.loads((job_dir / 'takeover_context.json').read_text())
                require(context['task'] == {k: v for k, v in tasks[task_id].items() if k != 'demo_start'},
                        'task configuration changed after scan')
                require(context['inputs'][str(ENGINE)] == ENGINE_SHA, 'scan used a different engine')
                for filename, sha in context['inputs'].items():
                    fingerprints.setdefault(filename, sha)
                    require(fingerprints[filename] == sha, 'inconsistent scan input fingerprints')
                games += summary['games']
        require(games == trace['scan_games'], 'trace game budget differs from raw games')
        points = {int(d): (b['teacher']['wins'], b['teacher']['games'])
                  for d, b in trace['points'].items() if 'teacher' in b}
        ladder = scan.build_ladder(points)
        spec = tasks[task_id].get('demo_start')
        if ladder:
            require(spec and spec['archive'] == entry['archive'] and spec['decisions'] == ladder['ladder'],
                    'registered start cell differs from the scan ladder')
            require(spec['future_seed_count'] == 256, 'registered seed block differs')
            additions.append(tasks[task_id])
            seeds.append((spec['future_seed_start'], spec['future_seed_start'] + 255, task_id))
            top = trace['points'][str(ladder['top'])]['teacher']
            rates[task_id] = dict(wins=top['wins'], games=top['games'],
                                  model='r11 + v5 + branch1', source='task105 deploy14 adaptive top')
        else:
            require(not spec, 'failed scan was registered')
        seeds.append((entry['future_seed_start'], entry['future_seed_start'] + 95, task_id + ':scan'))
        outcomes.append(dict(task_id=task_id, base_task_id=row['base_task_id'], seed=entry['seed'],
                             archive_sha256=row['archive_sha256'], result_sha256=file_sha(path),
                             l50=ladder['l50'] if ladder else None,
                             decisions=ladder['ladder'] if ladder else [], games=games,
                             top_is_full=ladder['top_is_full'] if ladder else False,
                             ready=bool(ladder), deployed=False))
    for filename, sha in fingerprints.items():
        require(file_sha(Path(filename)) == sha, 'scan input changed on disk')
    seeds.sort()
    require(all(a[1] < b[0] for a, b in zip(seeds, seeds[1:])), 'task105 seed blocks overlap')
    # Compare each new block against every pre-existing global reservation, omitting only
    # this task's own exact ledger keys.
    ledger = json.loads(importer.DEFAULT_SEED_LEDGER.read_text())['reservations']
    own_archives = {r['archive'] for r in rows if r['archive']}
    for key, reservation in ledger.items():
        try:
            fields = json.loads(key)
        except ValueError:
            fields = []
        if any(isinstance(field, str) and field in own_archives for field in fields):
            continue
        lo, hi = reservation['start'], reservation['start'] + reservation['count'] - 1
        require(all(hi < a or b < lo for a, b, _ in seeds), 'seed block overlaps an older reservation')
    occupied = set()
    for path in (ROOT / 'configs').glob('*.json'):
        document = json.loads(path.read_text())
        if path.name == 'human_demo_tasks.json':
            document['tasks'] = [t for t in document['tasks'] if t['task_id'] not in accepted]
        importer._config_seed_points(document, occupied)
    baseline_design = subprocess.check_output(
        ['git', '-C', str(ROOT), 'show', BASELINE + ':DESIGN.md'], text=True)
    occupied.update(int(m.replace(',', '')) for m in re.findall(r'\b\d{1,3}(?:,\d{3}){2,}\b', baseline_design))
    for lo, hi, _ in seeds:
        require(not any(seed in occupied for seed in range(lo, hi + 1)), 'new block overlaps config/DESIGN seeds')
    inventory = json.loads((ROOT / 'MEMORY/docs/llm-import-seed-inventory.json').read_text())
    for lo, hi in inventory['occupied'] + inventory['reserved_pools']:
        require(all(hi <= a or b < lo for a, b, _ in seeds), 'new block overlaps global inventory/pools')
    summary = {task: {**cell, 'won': len(cell['won']), 'robust': len(cell['robust']),
                      'reasons': dict(cell['reasons'])} for task, cell in sorted(task_summary.items())}
    # Exercise the real loader on all new cells even where old Mac-only archives
    # make the repository-wide historical fixture skip.
    subset = out / 'audited_start_cells.json'
    save(subset, dict(tasks=additions))
    import curriculum_queue
    cells, _ = curriculum_queue.load_demo_start_cells(subset)
    require(set(cells) == {t['task_id'] for t in additions}, 'new curriculum cells cannot load')
    evidence = dict(schema='task105_evidence_v1', audit_passed=True, input_sha256=file_sha(args.input),
                    engine_sha256=ENGINE_SHA, archives=len(rows), tasks=summary, scans=outcomes,
                    registered=len(additions), scan_games=sum(r['games'] for r in outcomes), deployed=False)
    save(out / 'audit.json', evidence)
    save(ROOT / 'MEMORY/docs/task105-import-evidence.json', evidence)
    save(out / 'deployment_plan.json', dict(schema='task105_deployment_v1', audit_passed=True,
         engine_sha256=ENGINE_SHA, tasks=additions, teacher_rates=rates,
         archive_sha256={t['task_id']: accepted[t['task_id']]['archive_sha256'] for t in additions},
         audit_sha256=file_sha(out / 'audit.json')))
    print(json.dumps(dict(audit_passed=True, archives=len(rows), scanned=len(outcomes),
                          registered=len(additions), deployed=False)))


if __name__ == '__main__':
    main()
