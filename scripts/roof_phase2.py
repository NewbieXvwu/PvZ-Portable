"""Task46 appendix A: bounded original roof archives and decision timelines."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
OUT = ROOT / 'artifacts/task46/phase2'
MANIFEST = ROOT / 'artifacts/task46/roof_panel.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Timeline:
    def __init__(self, path):
        import pvz_constants
        ids = {d['seed']: d['type'] for d in pvz_constants.plant_defs()}
        source = (ROOT / 'src/Lawn/Plant.cpp').read_text()
        self.shooters = {ids[name] for name in re.findall(
            r'\.mSeedType\s*=\s*SeedType::SEED_(\w+),(?:(?!\},).)*?\.mSubClass\s*=\s*PlantSubClass::SUBCLASS_SHOOTER', source, re.S)}
        assert {32, 34, 39} <= self.shooters
        self.stream = Path(path).open('w')
        self.events = Counter()

    def before(self, env, obs, state, decision):
        previous = state.get('frame')
        native = env.rollout_predict(previous['tick'] if previous else None,
            bool((state.get('action') or {}).get('type') == 'plant'),
            student_history=state.get('history'))
        plants = obs.get('plants') or []
        self.row = dict(decision=decision + 1, tick=obs['tick'], wave=obs['wave'],
            sun=obs['sun'], packets=obs['packets'], plants=plants,
            zombies=obs.get('zombies', []), projectiles=obs.get('projectiles', []),
            legal_plants=(obs.get('legal_actions') or {}).get('plants', []),
            row_shooters=[dict(Counter(str(p['type']) for p in plants if p['row'] == r
                and p['type'] in self.shooters and not p.get('squished'))) for r in range(5)],
            row_pults=[sum(p['row'] == r and p['type'] in (32, 34, 39)
                and not p.get('squished') for p in plants) for r in range(5)],
            native_action=native['action'])

    def after(self, obs, action, student, rejected, info):
        self.events.update(info.get('events') or {})
        self.row.update(action=action, student_action=student,
            native_matches=student == self.row['native_action'], rejected=rejected,
            after_tick=obs['tick'], after_sun=obs['sun'], events=info.get('events') or {})
        if action['type'] == 'plant':
            packet = next(p for p in self.row['packets'] if p['index'] == action['packet'])
            occupants = [p['type'] for p in self.row['plants'] if
                (p['row'], p['col']) == (action['row'], action['col'])]
            self.row['planting'] = dict(type=packet['type'], row=action['row'], col=action['col'],
                cost=packet['cost'], occupants=occupants, on_pot=33 in occupants, accepted=not rejected)
        self.stream.write(json.dumps(self.row, separators=(',', ':')) + '\n')

    def close(self):
        self.stream.close()


def cells():
    for level in range(42, 50):
        for seed in range(45000000, 45000004):
            yield level, seed, 'teacher'
        if level <= 45:
            yield level, 45000000, 'student'


def folder(level, seed, policy):
    return OUT / f'l{level}-{seed}-{policy}'


def capture(level, seed, policy):
    import search_play as sp
    from curriculum_queue import command_for
    from roof_experiment import settings
    from pvz_frame_store import read_frames, frame_path
    out = folder(level, seed, policy)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'meta.json').exists():
        assert json.loads((out / 'meta.json').read_text())['outcome']['terminal']
        return
    config = settings()
    config.frames, config.out = out / 'raw', out / 'result'
    cmd = command_for(config, MANIFEST, f'roof_l{level}_eval_r7', 0, 1, 'capture')
    args = sp.parser().parse_args(cmd[2:])
    args.dagger_beta = 1 if policy == 'teacher' else 0
    args.leaf_probe, args.leaf_probe_limit = str(out / 'leaves'), 0
    observer = Timeline(out / 'timeline.jsonl')
    arm = 'fast_search' if policy == 'teacher' else 'fast'
    try:
        sp._load_nets(str(config.value), str(config.rollout_policy), str(config.static_value))
        sp._init_worker(str(config.value), str(config.rollout_policy), args.rollout_sim_skip,
                        str(config.static_value))
        result = sp._play(dict(seed=seed, arm=arm, args=args), observer=observer)
    finally:
        observer.close()
        sp._close_worker_env()
        if sp._W.get('rollout_tmp'):
            sp._W['rollout_tmp'].cleanup()
        sp._W.clear()
    raw = out / 'raw' / f'seed_{seed}' / f'capture_{arm}'
    frames = read_frames(frame_path(raw))
    (out / 'frames.npz').write_bytes(frame_path(raw).read_bytes())
    task = next(t for t in json.loads(MANIFEST.read_text())['tasks']
                if t['task_id'] == args.task_id)
    task = {k: v for k, v in task.items() if k != 'seeds'}
    task.update(seed=seed, policy=policy, checkpoint=str(config.rollout_policy),
        checkpoint_sha256=sha(config.rollout_policy), search_recipe=cmd[2:],
        value=str(config.value), static_value=str(config.static_value))
    outcome = dict(result, result=frames[-1]['result'], actions=result['decisions'],
        reason='通关' if result['won'] else '僵尸进屋' if result['terminal'] else '动作上限截断',
        plants_eaten=observer.events['plants_eaten'],
        rejected_actions=sum(bool(f.get('action_rejected')) for f in frames))
    assert outcome['terminal'], outcome
    meta = dict(schema='episode_archive_v1', task=task, outcome=outcome,
                totals=dict(observer.events), frame_count=len(frames))
    (out / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    print(out.name, outcome['won'], outcome['final_wave'], outcome['final_tick'], flush=True)


def run():
    def job(cell):
        out = folder(*cell)
        out.mkdir(parents=True, exist_ok=True)
        with (out / 'capture.log').open('a') as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), 'capture',
                '--level', str(cell[0]), '--seed', str(cell[1]), '--policy', cell[2]],
                stdout=log, stderr=subprocess.STDOUT, check=True,
                env={**os.environ, 'PVZ_NO_POPUP': '1', 'OMP_NUM_THREADS': '1',
                     'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        print('DONE', out.name, flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(job, cells()))


def summarize():
    summaries = []
    for cell in cells():
        out = folder(*cell)
        meta = json.loads((out / 'meta.json').read_text())
        rows = [json.loads(x) for x in (out / 'timeline.jsonl').read_text().splitlines()]
        plants = [dict(decision=x['decision'], tick=x['tick'], wave=x['wave'], sun=x['sun'],
                       **x['planting']) for x in rows if x.get('planting')]
        accepted = [p for p in plants if p['accepted']]
        produced = sum(x['events'].get('sun_produced', 0) for x in rows)
        spent = sum(x['events'].get('sun_spent', 0) for x in rows)
        assert rows[0]['sun'] + produced - spent == rows[-1]['after_sun']
        entry = dict(level=cell[0], seed=cell[1], policy=cell[2], outcome=meta['outcome'],
            archive=str(out), archive_sha256=sha(out / 'frames.npz'),
            timeline_sha256=sha(out / 'timeline.jsonl'), decisions=len(rows),
            initial_sun=rows[0]['sun'], final_sun=rows[-1]['after_sun'], produced=produced, spent=spent,
            sun_max=max(max(x['sun'], x['after_sun']) for x in rows),
            native_mismatches=[x['decision'] for x in rows if not x['native_matches']],
            types=dict(Counter(p['type'] for p in accepted)), plantings=plants,
            missing_pot=[p for p in accepted if p['type'] != 33 and not p['on_pot']],
            row_firepower_changes=[dict(tick=x['tick'], wave=x['wave'], sun=x['sun'],
                shooters=x['row_shooters'], pults=x['row_pults']) for i, x in enumerate(rows)
                if i == 0 or x['row_shooters'] != rows[i-1]['row_shooters']],
            max_pults=max(sum(x['row_pults']) for x in rows))
        # Compare existing cold start only where the original formal cell exists.
        old = ROOT / f'artifacts/task46/runs/roof_l{cell[0]}_eval_r7_{cell[2]}_16/games.jsonl'
        if old.exists():
            original = next(json.loads(x) for x in old.read_text().splitlines()
                            if json.loads(x)['seed'] == cell[1])
            keys = ['won', 'final_wave', 'final_tick', 'decisions', 'planted_types', 'searched', 'overrides']
            assert all(original[k] == meta['outcome'][k] for k in keys), (cell, original, meta['outcome'])
            entry['cold_start_match'] = dict(source=str(old), sha256=sha(old), fields=keys)
        summaries.append(entry)
    result = dict(schema_version=1, resolution='actual engine decision tick; no interpolated intermediate state',
        firepower='engine SUBCLASS_SHOOTER live counts; pults=32/34/39 separately; counts are not DPS',
        games=summaries)
    (ROOT / 'MEMORY/docs/task46-roof-timelines.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps([dict(level=x['level'], seed=x['seed'], policy=x['policy'],
        won=x['outcome']['won'], wave=x['outcome']['final_wave'], types=x['types'],
        max_pults=x['max_pults'], mismatches=len(x['native_mismatches'])) for x in summaries]))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['run', 'capture', 'summary'])
    p.add_argument('--level', type=int)
    p.add_argument('--seed', type=int)
    p.add_argument('--policy', choices=['teacher', 'student'], default='teacher')
    a = p.parse_args()
    if a.mode == 'run':
        run()
    elif a.mode == 'summary':
        summarize()
    else:
        capture(a.level, a.seed, a.policy)
