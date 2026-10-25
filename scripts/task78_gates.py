"""Offline training-corpus consistency and horizon-stability diagnostics for value v6.

Both gates read the audited fog/roof decision labels that were already collected. Each
root keeps every search candidate and one common random future per candidate, and each
horizon scores the same root/candidates, so v5 and v6 can be compared on identical
states without opening another game.

These diagnostics do not replace the preregistered task72b 94-state panel or the
independent roof panel. Rebuilding task72b requires its original probe inputs or
an exactly validated reconstruction; those inputs have not yet been obtained.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]
import numpy as np
import torch
from pvz_value_net import load_value_net
from task78_data import sha

EPS = 1e-9
DEAD_ZONE = 0.001  # task72b: |Δ| below this is v5's noise floor


def load_rows(export_path, data_dir):
    """Every audited row, keyed by archive and root position."""
    export = json.loads(Path(export_path).read_text())
    if export.get('training_eligible') is False:
        raise ValueError('independent multi-future panels require task78_roof_score.py')
    if export.get('complete') is not True:
        raise ValueError('gates require the complete audited export')
    states = defaultdict(dict)
    for shard in export['shards']:
        if not shard['rows']:
            continue
        path = Path(shard['path'])
        if sha(path) != shard['sha256']:
            raise ValueError(f'label shard changed: {path}')
        part = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        archive = part['config']['archive']
        horizon = int(part['config']['horizon'])
        group = part['config']['group']
        world = json.dumps(part['config']['world'], sort_keys=True)
        touched = set()
        for i in range(len(part['won'])):
            state = states[(archive, int(part['position'][i]))]
            state.setdefault('group', group)
            state.setdefault('world', world)
            state.setdefault('rows', {}).setdefault(horizon, []).append(dict(
                candidate=int(part['candidate'][i]), packet=int(part['packet'][i]),
                won=float(part['won'][i]), index=i))
            touched.add((archive, int(part['position'][i])))
        for key in touched:
            states[key].setdefault('shards', {})[horizon] = str(path)
    return states


def score(models, states, batch=4096):
    """v5 and v6 probabilities for every stored leaf row, in one pass per shard."""
    cache = {}
    for state in states.values():
        for horizon, path in state['shards'].items():
            if path in cache:
                continue
            part = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            boards = part['boards'].float()
            scalars = part['scalars_v2'].float()
            scored = {}
            for name, model in models.items():
                model.eval()
                out = []
                with torch.inference_mode():
                    for first in range(0, len(boards), batch):
                        out.append(torch.sigmoid(model(boards[first:first+batch],
                                                       scalars[first:first+batch])).numpy())
                scored[name] = np.concatenate(out)
            cache[path] = scored
    for state in states.values():
        state['values'] = {name: {} for name in models}
        for horizon, path in state['shards'].items():
            for row in state['rows'][horizon]:
                for name in models:
                    state['values'][name][(horizon, row['candidate'])] = float(
                        cache[path][name][row['index']])
    return states


def wait_candidate(state, horizon):
    for row in state['rows'][horizon]:
        if row['packet'] == -1:
            return row['candidate']
    return None


def usable(state):
    """States whose horizons share the same candidate set, including wait."""
    horizons = sorted(state['rows'])
    if len(horizons) < 2:
        return False
    sets = {h: {r['candidate'] for r in state['rows'][h]} for h in horizons}
    if any(s != sets[horizons[0]] for s in sets.values()):
        return False
    return all(wait_candidate(state, h) is not None for h in horizons)


def consistency(states, name):
    """Same-root candidate pairs whose terminal outcomes differ by >= 0.25."""
    correct = total = ties = 0
    per_world = defaultdict(lambda: [0, 0])
    for (archive, position), state in states.items():
        for horizon in sorted(state['rows']):
            rows = state['rows'][horizon]
            for a in range(len(rows)):
                for b in range(a+1, len(rows)):
                    left, right = rows[a], rows[b]
                    gap = left['won'] - right['won']
                    if abs(gap) < 0.25:
                        continue
                    delta = (state['values'][name][(horizon, left['candidate'])] -
                             state['values'][name][(horizon, right['candidate'])])
                    total += 1
                    if abs(delta) < EPS:
                        ties += 1
                        correct += 0.5
                        per_world[state['world']][0] += 0.5
                    elif (delta > 0) == (gap > 0):
                        correct += 1
                        per_world[state['world']][0] += 1
                    per_world[state['world']][1] += 1
    return dict(correct=correct, total=total, ties=ties,
                rate=None if not total else correct/total,
                worlds={k: v for k, v in per_world.items()})


def direction(state, name, horizon):
    """Sign of best non-wait candidate minus wait, in the model's own values."""
    wait = wait_candidate(state, horizon)
    if wait is None:
        return None
    values = state['values'][name]
    best = max((r['candidate'] for r in state['rows'][horizon] if r['candidate'] != wait),
               key=lambda c: values[(horizon, c)], default=None)
    if best is None:
        return None
    delta = values[(horizon, best)] - values[(horizon, wait)]
    if abs(delta) < EPS:
        return 0.0
    return 1.0 if delta > 0 else -1.0


def flips(states, name, differentiated=None):
    flipped = dead = 0
    per_world = defaultdict(lambda: [0, 0])
    for key, state in states.items():
        if not usable(state):
            continue
        signs = [direction(state, name, h) for h in sorted(state['rows'])]
        if any(s is None for s in signs):
            continue
        if differentiated is not None:
            spread = 0.0
            for h in sorted(state['rows']):
                wait = wait_candidate(state, h)
                values = state['values'][name]
                best = max(values[(h, r['candidate'])] for r in state['rows'][h]
                           if r['candidate'] != wait)
                spread = max(spread, abs(best - values[(h, wait)]))
            if spread < differentiated:
                dead += 1
                continue
        nonzero = [s for s in signs if s != 0.0]
        per_world[state['world']][1] += 1
        if len(set(nonzero)) > 1:
            flipped += 1
            per_world[state['world']][0] += 1
    total = sum(v[1] for v in per_world.values())
    return dict(flipped=flipped, total=total, skipped_dead=dead,
                rate=None if not total else flipped/total, worlds=dict(per_world))


def bootstrap_difference(left, right, n=10000, seed=780):
    """Cluster bootstrap over worlds for a rate difference left-right."""
    worlds = sorted(set(left['worlds']) | set(right['worlds']))
    if not worlds or not any(v[1] for v in left['worlds'].values()) or not any(v[1] for v in right['worlds'].values()):
        return dict(mean=None,lower=None,upper=None,status='insufficient_evidence')
    rng = np.random.default_rng(seed)
    draws = np.empty(n)
    for i in range(n):
        pick = rng.choice(len(worlds), len(worlds), replace=True)
        def rate(stats):
            a = sum(stats['worlds'].get(worlds[j], [0, 0])[0] for j in pick)
            b = sum(stats['worlds'].get(worlds[j], [0, 0])[1] for j in pick)
            return a/b if b else 0.0
        draws[i] = rate(left) - rate(right)
    draws.sort()
    return dict(mean=float(draws.mean()),
                lower=float(draws[int(0.025*n)]), upper=float(draws[int(0.975*n)]))


PANEL_SCOPE = 'independent_roof_terminal_panel'
CORPUS_SCOPE = 'training_and_stopping_corpus_diagnostic'


def report(states, names, scope=CORPUS_SCOPE):
    fog = {k: v for k, v in states.items() if not v['group'].startswith('roof')}
    roof = {k: v for k, v in states.items() if v['group'].startswith('roof')}
    panel = scope == PANEL_SCOPE
    document = dict(states=len(states), fog_states=len(fog), roof_states=len(roof),
                    evaluation_scope=scope,
                    preregistered_gate_eligible=False,
                    missing_required_panels=(['task72b original 94 states',
                                              'independent roof action/future coverage audit'] if panel else
                                             ['task72b original 94 states',
                                              'independent roof 48 roots with three futures']),
                    note='single common future per candidate; terminal gaps are 0/1',
                    dead_zone=DEAD_ZONE)
    for terrain, subset in (('fog', fog), ('roof', roof)):
        document[terrain] = {}
        for name in names:
            document[terrain][name] = dict(
                consistency=consistency(subset, name),
                flips_all=flips(subset, name),
                flips_differentiated=flips(subset, name, DEAD_ZONE))
        document[terrain]['consistency_difference'] = bootstrap_difference(
            document[terrain][names[0]]['consistency'],
            document[terrain][names[1]]['consistency'])
        document[terrain]['flip_difference'] = bootstrap_difference(
            document[terrain][names[0]]['flips_all'],
            document[terrain][names[1]]['flips_all'])
        # A name or training_eligible=false does not prove three-future coverage.
        # This legacy single-future reader cannot certify either original panel.
        document[terrain]['preregistered_gate_eligible'] = False
    return document


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--export', default=str(ROOT/'artifacts/task78/export.json'))
    parser.add_argument('--data', default=str(ROOT/'artifacts/task78/data'))
    parser.add_argument('--v5', default='/home/newbiexvwu/PvZ-search/artifacts/value_net/v5/model.pt')
    parser.add_argument('--v6', default=str(ROOT/'artifacts/task78/v6/model.pt'))
    parser.add_argument('--out', default=str(ROOT/'artifacts/task78/gates.json'))
    args = parser.parse_args()
    models = dict(v5=load_value_net(args.v5).cpu(), v6=load_value_net(args.v6).cpu())
    states = score(models, load_rows(args.export, args.data))
    panel = json.loads(Path(args.export).read_text()).get('training_eligible') is False
    document = report(states, ['v5', 'v6'], PANEL_SCOPE if panel else CORPUS_SCOPE)
    document['inputs'] = dict(export=sha(args.export), v5=sha(args.v5), v6=sha(args.v6))
    Path(args.out).write_text(json.dumps(document, indent=1) + '\n')
    print(json.dumps({t: {n: document[t][n]['consistency']['rate']
                          for n in ('v5', 'v6')} for t in ('fog', 'roof')}, indent=1))


if __name__ == '__main__':
    main()
