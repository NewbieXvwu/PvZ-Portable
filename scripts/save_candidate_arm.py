"""Measurement arm: let the teacher evaluate cards it cannot afford yet.

The 2026-10-26 diagnosis (MEMORY/docs/task120-teacher-blind-spot.md) found that
the production search never sees an expensive card: the engine only lists plant
actions the player can pay for right now, and the student's spend-down economy
keeps the balance under 125-300 sun. The card is therefore absent from the
candidate set long before the value net could rank it.

This wrapper leaves search_play untouched and adds one step: before a decision
is finalised, for each deck card that is unaffordable but within reach, it
snapshots the root, waits in the engine until the card is affordable, scores the
best legal cell there, and rolls that plant out. If that future value beats the
best normal candidate by MARGIN, the decision becomes the wait itself. The next
decision sees the card as legal and the unchanged search can plant it.

Measurement only: the probe advances the environment, so it must not run with
--out exports that record every operation. The wrapper refuses to install when
an episode recorder is active.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]

import search_play as sp  # noqa: E402

_OPTIONS: dict[str, float] = {}


def _value(env, obs: dict, action: dict, snapshot: int, seed: int, until_tick: int) -> float:
    # Same absolute terminal tick as the normal candidates: waiting must not buy extra horizon.
    leaf = sp._rollout_policy(env, obs, action, until_tick, snapshot, seed)
    terminal = sp._score_terminal_or_value(leaf)
    if terminal is not None:
        return terminal
    return sp._leaf_values([leaf])[0]


def _probe(env, obs: dict, root: int, packet: int, cost: int, seed: int, stats: dict,
           until_tick: int) -> dict | None:
    chunk = int(_OPTIONS['chunk'])
    budget = int(_OPTIONS['max_wait'])
    waited = 0
    current = obs
    # The caller leaves the environment at the last rollout's leaf; start from the decision root.
    env.restore_fast(root)
    while current['sun'] < cost and waited < budget and not current['terminal']:
        step = min(chunk, budget - waited)
        current, _reward, _done, _truncated, info = env.step({'type': 'wait', 'ticks': step})
        advanced = int(info.get('ticks_advanced') or 0)
        if advanced <= 0:
            break
        waited += advanced
    if current['terminal'] or current['sun'] < cost:
        return dict(reason='unaffordable', sun=int(current['sun']), waited=waited,
                    tick=int(current['tick']), start=int(obs['tick']), terminal=bool(current['terminal']))
    actions = [{'type': 'plant', 'packet': entry['packet'], 'col': entry['col'], 'row': entry['row']}
               for entry in (current.get('legal_actions') or {}).get('plants') or []
               if entry['packet'] == packet]
    if not actions:
        plants_now = (current.get('legal_actions') or {}).get('plants') or []
        return dict(reason='no_cell', sun=int(current['sun']), waited=waited,
                    legal=len(plants_now), packets=sorted({p['packet'] for p in plants_now}))
    future = env.snapshot_fast()
    scores = env.static_scores(future, actions, current.get('sun_income_rate') or 0.0)
    pairs = [(score, action) for score, action in zip(scores, actions) if score is not None]
    if not pairs:
        env.release_snapshot(future)
        return dict(reason='no_score', sun=int(current['sun']), waited=waited)
    action = max(pairs, key=lambda item: item[0])[1]
    # Match the normal candidates' evidence: their override is decided after samples + verify_samples
    # rollouts, so a 3-sample plan must not win against a 12-sample estimate.
    samples = max(1, int(_OPTIONS['samples']) + int(_OPTIONS['verify']))
    values = [_value(env, current, action, future, (seed * 1000 + k) % 2 ** 32, until_tick)
              for k in range(samples)]
    env.release_snapshot(future)
    return dict(packet=packet, cost=cost, ticks=waited, cell=[action['col'], action['row']],
                value=sum(values) / len(values), sun=int(current['sun']), samples=samples)


def _install(options: dict) -> None:
    _OPTIONS.update(options)
    original = sp._verified_choice
    last: dict[int, int] = {}

    def patched(env, obs, root, candidates, totals, counts, best, policy_index, policy_action,
                search_args, decision_seed, stats, **keywords):
        _OPTIONS['horizon'] = search_args.horizon
        _OPTIONS['verify'] = getattr(search_args, 'verify_samples', 0) or 0
        present = {c.get('packet') for c in candidates if c['type'] == 'plant'}
        # obs['packets'] is in deck order: the slot index is the action's packet field, and
        # entry['type'] is the SeedType (a different numbering).
        costs = {slot: entry.get('cost') or 0 for slot, entry in enumerate(obs.get('packets') or [])
                 if entry.get('type', -1) >= 0}
        share = float(_OPTIONS['share'])
        interval = int(_OPTIONS['interval'])
        reachable = [(packet, cost) for packet, cost in costs.items()
                     if packet not in present and cost > obs['sun'] and obs['sun'] >= share * cost
                     and obs['tick'] - last.get(packet, -10 ** 9) >= interval]
        reachable.sort(key=lambda item: item[1])
        plans = []
        # The probe advances the engine; keep it out of the export operation stream so a
        # measurement run can never be mistaken for a replayable one.
        episode = env.episode
        env.episode = None
        try:
            for packet, cost in reachable[: int(_OPTIONS['cards'])]:
                last[packet] = obs['tick']
                plan = _probe(env, obs, root, packet, cost, decision_seed, stats,
                              obs['tick'] + search_args.horizon)
                env.restore_fast(root)
                if _OPTIONS.get('log'):
                    with open(str(_OPTIONS['log']), 'a') as stream:
                        finite_now = [float(v) for v in totals if math.isfinite(float(v))]
                        stream.write(json.dumps(dict(kind='probe', task=search_args.task_id,
                            seed=decision_seed // 10000, tick=obs['tick'], sun=obs['sun'],
                            packet=packet, cost=cost, best=max(finite_now) if finite_now else None,
                            plan=plan)) + '\n')
                if plan is not None and 'value' in plan:
                    plans.append(plan)
        finally:
            env.episode = episode
        finite = [float(v) for v in totals if math.isfinite(float(v))]
        if plans and finite:
            plan = max(plans, key=lambda item: item['value'])
            stats['save_probes'] = stats.get('save_probes', 0) + len(plans)
            if _OPTIONS.get('log'):
                with open(str(_OPTIONS['log']), 'a') as stream:
                    stream.write(json.dumps(dict(task=search_args.task_id, seed=decision_seed // 10000,
                        tick=obs['tick'], sun=obs['sun'], best=round(max(finite), 6),
                        plan=plan)) + '\n')
            if plan['value'] > max(finite) + sp.MARGIN:
                stats['save_plans'] = stats.get('save_plans', 0) + 1
                stats['save_wait_ticks'] = stats.get('save_wait_ticks', 0) + plan['ticks']
                stats.setdefault('save_events', []).append(dict(tick=obs['tick'], **plan))
                stats['last_targets'] = None
                return {'type': 'wait', 'ticks': plan['ticks'], 'until': 'timeout'}
        return original(env, obs, root, candidates, totals, counts, best, policy_index,
                        policy_action, search_args, decision_seed, stats, **keywords)

    sp._verified_choice = patched


def main() -> None:
    argv = sys.argv[1:]
    options = dict(share=0.5, cards=1, max_wait=1800, chunk=300, interval=600, samples=3, verify=0)
    log_path = None
    cleaned: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == '--save-log':
            log_path = argv[index + 1]
            index += 2
            continue
        if token.startswith('--save-'):
            key = token[len('--save-'):].replace('-', '_')
            if key not in options:
                raise SystemExit(f'unknown save option {token}')
            options[key] = float(argv[index + 1])
            index += 2
            continue
        cleaned.append(token)
        index += 1
    sys.argv = [sys.argv[0]] + cleaned
    if log_path:
        options['log'] = log_path
    _install(options)
    sp.main()


if __name__ == '__main__':
    main()
