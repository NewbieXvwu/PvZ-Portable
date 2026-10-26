"""Freeze and evaluate task92b's mechanical panel; production state is read-only."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_student_selection import assert_disjoint
from pvz_tasks import deck_for_seed, terrain_covered
from select_student import score_rows, sha, select

OUT = ROOT / 'artifacts/task92b'
PANEL = ROOT / 'configs/student_selection_v2.json'
PLAN = ROOT / 'MEMORY/docs/multiseed-select-v2-preregister.json'
R11 = Path('/home/newbiexvwu/pvz-logs/curriculum_state.pipeline/round_000011/student/model.pt')
ENGINE = Path('/home/newbiexvwu/PvZ-search/.claude/worktrees/cq/build4/pvz-portable')
VALUE = Path('/home/newbiexvwu/PvZ-search/artifacts/value_net/v5/model.pt')
RANDOM_SOURCE = Path('/home/newbiexvwu/PvZ-search/artifacts/stage4/random_deck_panel.json')
RUNS = Path('/home/newbiexvwu/PvZ-search/artifacts/hard')
NAMES = ('a', 'c', 'd2', 'r11')
QUALITY = (124, 135, 169, 166)
BLOCK = (126_000_000, 126_020_000)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.building')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def eligible_windows(state, runs=RUNS, r11=R11):
    """Use only r11's beta-zero suffix of each state window, never older students."""
    if Path(state['student']).resolve() != r11.resolve() or state['student_round'] != 11:
        raise ValueError('snapshot incumbent is not r11')
    cells, sources = [], {}
    for base, cell in sorted(state['cells'].items()):
        for multiplier, bucket in sorted(cell.get('windows', {}).items(), key=lambda item: float(item[0])):
            outcomes, task_source = [], None
            for rnd in state['log']:
                for group in rnd['groups']:
                    if (group['base'] != base or float(group['multiplier']) != float(multiplier)
                            or group.get('beta', 0) != 0 or group.get('student_round') != 11):
                        continue
                    folder = runs / f'{group["label"]}_{group["games"]}'
                    config, games = folder / 'config.json', folder / 'games.jsonl'
                    settings = json.loads(config.read_text())
                    if (Path(settings['rollout_policy']).resolve() != r11.resolve()
                            or float(settings['dagger_beta']) != 0 or settings['task_id'] != group['task_id']):
                        raise ValueError(f'group is not an r11 beta-zero sample: {config}')
                    rows = [json.loads(line) for line in games.read_text().splitlines() if line.strip()]
                    if len(rows) != group['games'] or any(type(row.get('won')) is not bool
                                                        or row.get('terminal') is not True for row in rows):
                        raise ValueError(f'incomplete curriculum group: {games}')
                    outcomes.extend(int(row['won']) for row in rows)
                    task_source = group
                    sources[str(config)], sources[str(games)] = sha(config), sha(games)
            recent = bucket.get('recent', [])
            outcomes = outcomes[-len(recent):] if recent else []
            if outcomes and outcomes != recent[-len(outcomes):]:
                raise ValueError(f'r11 outcomes are not the saved window suffix: {base}/{multiplier}')
            if outcomes and .3 <= sum(outcomes) / len(outcomes) <= .8:
                cells.append(dict(base=base, multiplier=float(multiplier), wins=sum(outcomes),
                                  games=len(outcomes), rate=sum(outcomes) / len(outcomes),
                                  window_games=len(recent), task_id=task_source['task_id'],
                                  manifest=task_source['manifest']))
    return cells, sources


def prepare():
    from make_random_deck_tasks import FAMILIES, generate
    from pvz_env import PvZEnv
    inventory_path = OUT / 'seed_inventory.full.json'
    inventory = json.loads(inventory_path.read_text())
    if (inventory['candidate']['low'], inventory['candidate']['high']) != BLOCK:
        raise ValueError('inventory audited a different block')
    if not inventory['candidate']['disjoint'] or inventory['verification']['hit_count']:
        raise ValueError('new seed block is occupied')
    if PANEL.exists() or PLAN.exists():
        raise ValueError('panel already preregistered; do not regenerate after observing scores')
    snapshot = OUT / 'curriculum_state.snapshot.json'
    state = json.loads(snapshot.read_text())
    cells, sources = eligible_windows(state)
    count = 384 // (len(cells) + len(FAMILIES))
    if count < 1 or len(cells) + len(FAMILIES) > 20:
        raise ValueError('panel does not fit the reserved 20 slots')
    tasks = []
    for index, cell in enumerate(cells):
        path = Path(cell['manifest'])
        task = copy.deepcopy(next(task for task in json.loads(path.read_text())['tasks']
                                  if task['task_id'] == cell['task_id']))
        task['seeds'] = list(range(BLOCK[0] + 1000 * index, BLOCK[0] + 1000 * index + count))
        sources[str(path)] = sha(path)
        tasks.append(task)
    old_random = json.loads(RANDOM_SOURCE.read_text())
    excluded = old_random['tasks'][0]['deck_random']['excluded_decks']
    seeds = {name: list(range(BLOCK[0] + 1000 * (len(cells) + i),
                             BLOCK[0] + 1000 * (len(cells) + i) + count))
             for i, (_, name) in enumerate(FAMILIES)}
    with PvZEnv(resource_dir=Path.home() / '.cache/pvz-research-resources', executable=ENGINE, headless=True) as env:
        random_panel = generate(env, excluded=excluded, selection_seeds=seeds)
    for task in random_panel['tasks']:
        old = next(t for t in old_random['tasks'] if t['task_id'] == task['task_id'])
        if task['deck_random']['engine_options'] != old['deck_random']['engine_options']:
            raise ValueError('random family differs from task93 engine candidate pool')
    tasks.extend(random_panel['tasks'])
    panel = dict(schema_version=1, panel_version='selection-task92b-v2', tasks=tasks,
                 purpose='r11 beta-zero 30%-80% windows plus three required task93 random-deck families; equal games')
    reserved = [(item['low'], item['high']) for item in inventory['sources']]
    from pvz_promotion import SEED_POOLS_V2
    assert_disjoint(panel, reserved + list(SEED_POOLS_V2))
    for task in tasks:
        for seed in task['seeds']:
            if not terrain_covered(task['level'], deck_for_seed(task, seed)):
                raise ValueError('terrain coverage failed')
    write(PANEL, panel)
    sources.update({str(snapshot): sha(snapshot), str(RANDOM_SOURCE): sha(RANDOM_SOURCE),
                    str(inventory_path): sha(inventory_path), str(ENGINE): sha(ENGINE), str(VALUE): sha(VALUE)})
    models = {}
    for name in NAMES:
        source = R11 if name == 'r11' else ROOT / f'artifacts/task92/models/{name}.pt'
        destination = OUT / f'models/{name}.pt'
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        models[name] = dict(source=str(source), path=str(destination), sha256=sha(destination))
    # Fixed heldout measurements are a reference, never a panel-selection input.
    heldout_files, heldout_seeds = {}, None
    for name, wins in zip(NAMES, QUALITY):
        if name == 'r11':
            folder = Path('/home/newbiexvwu/pvz-logs/curriculum_state.pipeline/round_000013/evaluations/a41fdfa96075ae608a40/incumbent')
            games = folder / 'day_l7_eval_d7p_holdout1/games.jsonl'
        else:
            folder = ROOT.parent / 'gatepower/artifacts/task84' / f'variant_{name}'
            games = folder / 'eval_day_l7_eval_d7p_holdout1/games.jsonl'
        rows = [json.loads(line) for line in games.read_text().splitlines() if line.strip()]
        if (len(rows) != 192 or len({row['seed'] for row in rows}) != 192
                or sum(row['won'] for row in rows) != wins):
            raise ValueError('heldout reference differs from task92b evidence')
        seed_set = {row['seed'] for row in rows}
        if heldout_seeds is not None and seed_set != heldout_seeds:
            raise ValueError('heldout references are not paired on identical seeds')
        heldout_seeds = seed_set
        heldout_files[name] = dict(path=str(games), sha256=sha(games))
    write(PLAN, dict(panel_sha256=sha(PANEL), seed_block=list(BLOCK), cells=cells,
                     random_families=[name for _, name in FAMILIES], games_per_cell=count,
                     games_per_model=sum(len(t['seeds']) for t in tasks), models=models,
                     random_generation_audit=random_panel['generation_audit'],
                     rule='maximum total wins; stable a,c,d2,r11 path order breaks ties',
                     threshold_scope='30%-80% applies to existing r11 beta-zero curriculum windows. '
                                     'All three required random families are added without outcome screening; '
                                     'night/pool have no qc window and cold-start student rates below 30%.',
                     acceptance='Panel ranking equals heldout ranking D2>r11>C>A, or chooses D2/r11 and never A. '
                                'No post-score panel changes. Compare v1/v2 with four identical models.',
                     bootstrap=dict(draws=50000, seed=2026102602, stratified_by_task=True, paired_by_seed=True),
                     heldout_wins=dict(zip(NAMES, QUALITY)), sources_sha256=sources,
                     heldout_files=heldout_files,
                     seed_audit=dict(candidate=inventory['candidate'], verification=inventory['verification'])))
    print(json.dumps(dict(cells=len(tasks), per_cell=count, games=count * len(tasks), panel_sha256=sha(PANEL))))


def evaluate():
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'evaluation.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = json.loads(PLAN.read_text())
        if sha(PANEL) != plan['panel_sha256'] or sha(ENGINE) != plan['sources_sha256'][str(ENGINE)]:
            raise ValueError('preregistered panel/engine changed')
        registered = subprocess.check_output(['git', 'show', 'HEAD:configs/student_selection_v2.json'], cwd=ROOT)
        if registered != PANEL.read_bytes():
            raise ValueError('panel must be committed before scoring')
        for model in plan['models'].values():
            if sha(Path(model['path'])) != model['sha256']:
                raise ValueError('frozen model changed')
        if sha(VALUE) != plan['sources_sha256'][str(VALUE)]:
            raise ValueError('value-loader input changed')
        manifest = json.loads(PANEL.read_text())
        sets = ','.join(f'{t["task_id"]}:0:{len(t["seeds"])}' for t in manifest['tasks'])
        def cached_run(command, **kwargs):
            folder = Path(command[command.index('--out') + 1])
            games, config = folder / 'games.jsonl', folder / 'config.json'
            if games.is_file() and config.is_file():
                settings = json.loads(config.read_text())
                for flag in ('--rollout-policy', '--task-id', '--manifest', '--seeds', '--seed-offset', '--executable'):
                    if str(settings[flag[2:].replace('-', '_')]) != command[command.index(flag) + 1]:
                        raise ValueError(f'cached game config differs: {folder}/{flag}')
                task = next(t for t in json.loads(Path(settings['manifest']).read_text())['tasks']
                            if t['task_id'] == settings['task_id'])
                count = int(settings['seeds'])
                existing = [line for line in games.read_text().splitlines() if line.strip()]
                if len(existing) == count:
                    start = int(settings['seed_offset'])
                    score_rows(games, task['seeds'][start:start + count])
                    return subprocess.CompletedProcess(command, 0)
            while True:
                available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                                     if line.startswith('MemAvailable:'))) * 1024
                if available >= 7 * (1 << 30):
                    break
                print('waiting for memory before next two-worker task', available, flush=True)
                time.sleep(5)
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / 'driver.out').open('a') as log:
                kwargs.update(stdout=log, stderr=subprocess.STDOUT)
                return subprocess.run(command, **kwargs, env={**os.environ, 'PVZ_NO_POPUP': '1',
                                      'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
        select([m['path'] for m in plan['models'].values()], OUT / 'v2', PANEL, sets, 2,
               VALUE, Path.home() / '.cache/pvz-research-resources',
               ENGINE, run_command=cached_run)
        v1 = ROOT / 'configs/student_selection.json'
        v1_tasks = json.loads(v1.read_text())['tasks']
        sets = ','.join(f'{t["task_id"]}:0:{len(t["seeds"])}' for t in v1_tasks)
        select([plan['models']['r11']['path']], OUT / 'v1_r11', v1, sets, 2,
               VALUE, Path.home() / '.cache/pvz-research-resources',
               ENGINE, run_command=cached_run)


def kendall_tau(scores, quality):
    pairs = [(np.sign(scores[i] - scores[j]), np.sign(quality[i] - quality[j]))
             for i, j in itertools.combinations(range(len(scores)), 2)]
    denominator = math.sqrt(sum(a != 0 for a, b in pairs) * sum(b != 0 for a, b in pairs))
    return float(sum(a * b for a, b in pairs) / denominator) if denominator else None


def summarize(outcomes, draws=50000, seed=2026102602):
    rng = np.random.default_rng(seed)
    totals = sum((np.asarray(rows).sum(axis=0) for rows in outcomes), np.zeros(4, dtype=np.int64))
    samples = np.zeros((draws, 4), dtype=np.int64)
    for rows in outcomes:
        rows = np.asarray(rows)
        samples += rows[rng.integers(len(rows), size=(draws, len(rows)))].sum(axis=1)
    winners = np.argmax(samples, axis=1)
    winner = NAMES[int(np.argmax(totals))]
    return dict(totals=dict(zip(NAMES, map(int, totals))),
                ranking=sorted(NAMES, key=lambda name: -totals[NAMES.index(name)]),
                kendall_tau_b=kendall_tau(totals, QUALITY), selected=winner,
                bootstrap_winner_probability=dict(zip(NAMES, (np.bincount(winners, minlength=4) / draws).tolist())),
                bootstrap_heldout_best_probability=float(np.mean(winners == 2)),
                bootstrap_d2_or_r11_probability=float(np.mean(np.isin(winners, [2, 3]))),
                acceptance_passed=winner in ('d2', 'r11'))


def analyse():
    plan = json.loads(PLAN.read_text())
    sources = {str(PLAN): sha(PLAN), str(PANEL): sha(PANEL)}
    all_outcomes, task_scores = {}, {}
    for version, manifest in [('v1', ROOT / 'configs/student_selection.json'), ('v2', PANEL)]:
        tasks = json.loads(manifest.read_text())['tasks']
        outcomes = []
        scores_by_task = {}
        receipt_path = OUT / ('v1_r11' if version == 'v1' else 'v2') / 'selection.json'
        receipt = json.loads(receipt_path.read_text())
        sources[str(receipt_path)] = sha(receipt_path)
        for task in tasks:
            columns = []
            for index, name in enumerate(NAMES):
                model = plan['models'][name]
                if version == 'v1' and name != 'r11':
                    folder = ROOT / 'artifacts/task92/offline/selection' / f'c{index}_{task["task_id"]}'
                    games = folder / 'games.jsonl'
                else:
                    games = Path(receipt['scores'][model['path']]['games'][task['task_id']]['path'])
                config = games.parent / 'config.json'
                settings = json.loads(config.read_text())
                if (sha(Path(settings['rollout_policy'])) != model['sha256']
                        or settings['task_id'] != task['task_id'] or settings['arms'] != 'fast'
                        or sha(Path(settings['manifest'])) != sha(manifest)
                        or sha(Path(settings['executable'])) != plan['sources_sha256'][str(ENGINE)]):
                    raise ValueError(f'played policy/task/engine differs from frozen comparison: {config}')
                score_rows(games, task['seeds'])
                rows = {row['seed']: row for row in (json.loads(line) for line in games.read_text().splitlines() if line.strip())}
                if 'deck_random' in task and any(tuple(rows[s]['deck']) != deck_for_seed(task, s) for s in task['seeds']):
                    raise ValueError('played random deck does not match preregistered receipt')
                columns.append([int(rows[s]['won']) for s in task['seeds']])
                sources[str(games)] = sha(games)
                sources[str(config)] = sha(config)
            outcome = np.asarray(columns, dtype=np.int64).T
            scores_by_task[task['task_id']] = dict(zip(NAMES, map(int, outcome.sum(axis=0))))
            outcomes.append(outcome)
        all_outcomes[version] = summarize(outcomes, **{k: plan['bootstrap'][k] for k in ('draws', 'seed')})
        task_scores[version] = scores_by_task
    report = dict(schema_version=1, heldout_wins=dict(zip(NAMES, QUALITY)),
                  bootstrap=plan['bootstrap'], panels=all_outcomes, per_task=task_scores,
                  scope='Four fixed historical models; paired resampling within tasks only. '
                        'V1 uses 192 games/model and V2 380; differences include both allocation and budget. '
                        'Selection success does not establish future training gain or replace the promotion gate.',
                  sources_sha256=sources)
    write(OUT / 'comparison.json', report)
    print(json.dumps(report['panels'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'evaluate', 'analyse'))
    args = parser.parse_args()
    {'prepare': prepare, 'evaluate': evaluate, 'analyse': analyse}[args.action]()
