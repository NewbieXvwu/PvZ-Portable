"""Opt-in decision-boundary evidence; all state values come from engine observations."""
from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class Timeline:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('w')
        source = (ROOT / 'src/Lawn/Plant.cpp').read_text()
        import pvz_constants
        ids = {d['seed']:d['type'] for d in pvz_constants.plant_defs()}
        self.shooters = {ids[name] for name in re.findall(
            r'\.mSeedType\s*=\s*SeedType::SEED_(\w+),(?:(?!\},).)*?\.mSubClass\s*=\s*PlantSubClass::SUBCLASS_SHOOTER', source, re.S)}
        if not self.shooters:
            raise RuntimeError('failed to parse engine shooter subclass')
        self.events = Counter()

    def before(self, env, obs, state, decision):
        previous = state.get('frame')
        native = env.rollout_predict(previous['tick'] if previous else None,
                    bool((state.get('action') or {}).get('type') == 'plant'),
                    student_history=state.get('history'))
        plants = obs.get('plants') or []
        counts = Counter(p['type'] for p in plants if not p.get('squished'))
        self.row = dict(decision=decision + 1, tick=obs['tick'], wave=obs['wave'], sun=obs['sun'],
            packets=obs['packets'], grid=obs['grid'], plants=plants,
            legal_plants=(obs.get('legal_actions') or {}).get('plants', []),
            lilypads=counts[16], cattails=counts[43], sunshrooms=counts[9],
            row_shooters=[dict(Counter(str(p['type']) for p in plants if p['row']==r and
                          p['type'] in self.shooters and not p.get('squished'))) for r in range(6)],
            projectiles=obs.get('projectiles', []), native_action=native['action'])

    def after(self, obs, action, student, rejected, info):
        self.events.update(info.get('events') or {})
        self.row.update(action=action, student_action=student,
            native_matches=student == self.row['native_action'], rejected=rejected,
            after_tick=obs['tick'], after_sun=obs['sun'], events=info.get('events') or {})
        if action['type'] == 'plant':
            packet=next(p for p in self.row['packets'] if p['index']==action['packet'])
            self.row['planting'] = dict(type=packet['type'], row=action['row'], col=action['col'],
                terrain=self.row['grid'][action['row']][action['col']], cost=packet['cost'],
                occupants=[p['type'] for p in self.row['plants'] if
                           (p['row'],p['col'])==(action['row'],action['col'])], accepted=not rejected)
        self.stream.write(json.dumps(self.row, separators=(',', ':'))+'\n')

    def close(self):
        self.stream.close()
