"""task55: one persistent engine, reset/one-tick facts only, no policy or match batches."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'scripts')]
import pvz_constants as constants
from pvz_env import PvZEnv,TaskSpec
from pvz_tasks import deck_for_level,profile_for_deck


def source_types():
    text=(ROOT/'src/LawnApp.cpp').read_text()
    methods={'bowling':'IsWallnutBowlingLevel','whack':'IsWhackAZombieLevel',
             'little_trouble':'IsLittleTroubleLevel','vasebreaker':'IsScaryPotterLevel',
             'stormy_night':'IsStormyNightLevel','bungee':'IsBungeeBlitzLevel',
             'mini_boss':'IsMiniBossLevel','boss':'IsFinalBossLevel'}
    result={}
    for kind,method in methods.items():
        start,end=constants._function_span(text,f'bool LawnApp::{method}()')
        levels=sorted(set(map(int,re.findall(r'mBoard->mLevel\s*==\s*(\d+)',text[start:end]))))
        if not levels:
            raise ValueError(f'cannot resolve Adventure levels for {method}')
        result[kind]=dict(levels=levels,source='src/LawnApp.cpp',method=method,
                          line=text[:start].count('\n')+1)
    return result


def collect(executable,resources,out):
    predicates=source_types()
    conveyor=set(constants.adventure_conveyor_levels())
    rows=[]
    with PvZEnv(resources,executable=executable) as env:
        for level in range(1,51):
            original=level in conveyor
            deck=deck_for_level(level,'original' if original else None)
            task=TaskSpec(level=level,seed=55000000+level,playthrough=2,
                          profile=profile_for_deck(deck),
                          conveyor_start='original' if original else None,
                          fog_observation='human' if 31<=level<=39 and level!=35 else None)
            obs,_=env.reset(deck=deck,task=task)
            raw=env.privileged_state()
            waves=(raw.get('hidden') or {}).get('zombies_in_wave') or []
            scheduled=sorted({z for wave in waves for z in wave if z>=0})
            types=[kind for kind,p in predicates.items() if level in p['levels']]
            row=dict(level=level,requested_seed=task.seed,observed_level=obs['level'],
                     source_predicates=types,conveyor=original,probe_deck=list(deck),
                     reset_semantics='original conveyor' if original else 'task deck replay',
                     background=constants.background_for_level(level),
                     scene={k:obs.get(k) for k in ('terrain','night','pool','fog','roof','wave_count')},
                     allowed_roster=(obs.get('loadout_context') or {}).get('zombie_roster',[]),
                     scheduled_types=scheduled,initial_zombie_types=sorted({z['type'] for z in obs['zombies']}),
                     grid_items=obs.get('grid_items',[]),
                     initial_packet_types=[p['type'] for p in obs['packets']],
                     initial_plants={str(k):sum(p['type']==k for p in obs['plants'])
                                     for k in sorted({p['type'] for p in obs['plants']})})
            after,*_,info=env.step(dict(type='wait',ticks=1))
            row.update(reset_ok=True,one_tick_ok=bool(info.get('ok')),tick_after=after['tick'],
                       one_tick_zombie_types=sorted({z['type'] for z in after['zombies']}),
                       terminal_after_one_tick=after['terminal'])
            rows.append(row)
            print(json.dumps(dict(level=level,roster=row['allowed_roster'],scheduled=scheduled,
                                  predicates=types,reset=True,tick=after['tick'])),flush=True)
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    evidence=dict(as_of=datetime.now().astimezone().isoformat(),
                  method='one engine process; 50 resets and one wait tick each; no complete games or policies',
                  executable=str(executable),executable_sha256=digest(executable),
                  source_sha256={str(p.relative_to(ROOT)):digest(p) for p in
                                 [ROOT/'src/LawnApp.cpp',ROOT/'src/Lawn/Board.cpp',ROOT/'scripts/pvz_constants.py']},
                  predicates=predicates,zombie_dictionary=constants.zombie_defs(),levels=rows)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(evidence,ensure_ascii=False,indent=1)+'\n')
    return evidence


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executable',type=Path,required=True)
    p.add_argument('--resources',type=Path,default=Path('/home/newbiexvwu/.cache/pvz-research-resources'))
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/task55/engine-inventory.json')
    args=p.parse_args();collect(args.executable,args.resources,args.out)


if __name__=='__main__':main()
