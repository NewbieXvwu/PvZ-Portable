"""Summarize the existing L9 paired games and leaf probes; never launch or replay games.

Reads only small game/probe JSON files. The 20-decision view selects four evenly
spaced late decisions per v6 probe seed; all recorded decisions remain in totals.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from value_widen_ab import paired


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def describe(rows):
    if not rows:
        return dict(decisions=0)
    return dict(decisions=len(rows), seeds=len({r['seed'] for r in rows}),
        top1_top2_median=statistics.median(r['top1_top2'] for r in rows),
        gap_below_002=sum(r['top1_top2'] < .02 for r in rows),
        gap_below_002_fraction=sum(r['top1_top2'] < .02 for r in rows)/len(rows),
        candidate_mean=statistics.mean(r['mean'] for r in rows),
        top_score_median=statistics.median(r['values'][0] for r in rows),
        top_score_min=min(r['values'][0] for r in rows),
        top_score_max=max(r['values'][0] for r in rows),
        overrides=sum(r['overridden'] for r in rows))


def summarize(root, partial=False):
    directory=root/'ab_l9_32'
    plan=json.loads((directory/'plan.json').read_text())
    task=json.loads(Path(plan['manifest']).read_text())['tasks'][0]
    expected=set(task['seeds'])
    games={name:read_rows(directory/arm/task['task_id']/'games.jsonl')
           for name,arm in [('v5','v3'),('v6','v4')]}
    complete=all(len(rows)==32 and {r['seed'] for r in rows}==expected for rows in games.values())
    if not complete and not partial:
        raise RuntimeError('32 paired games are still running; leave the existing service alone')
    outcomes={name:{r['seed']:r['won'] for r in rows} for name,rows in games.items()}
    comparison=paired(outcomes['v5'],outcomes['v6'])
    first8=set(sorted(expected)[:8])
    blocks={label:paired(*[{s:won for s,won in outcomes[name].items() if s in seeds}
                           for name in ('v5','v6')])
            for label,seeds in [('original8',first8),('additional24',expected-first8)]}
    harms=sorted(s for s in outcomes['v5'].keys() & outcomes['v6'].keys()
                 if outcomes['v5'][s] and not outcomes['v6'][s])
    episode_stats={name:dict(games=len(rows),wins=sum(r['won'] for r in rows),
        mean_final_wave=statistics.mean(r['final_wave'] for r in rows),
        mean_overrides=statistics.mean(r['overrides'] for r in rows),
        mean_searches=statistics.mean(r['searched'] for r in rows),
        mean_seconds=statistics.mean(r['seconds'] for r in rows),
        loss_waves=dict(sorted(Counter(r['final_wave'] for r in rows if not r['won']).items())))
        for name,rows in games.items()}
    probes={}
    all_rows={}
    inputs=[]
    for name in ('v5','v6'):
        rows=[];probe_games=[]
        for prefix,run in [('l9','run'),('ctl','run_ctl')]:
            for path in sorted((root/'probe').glob(f'{prefix}_{name}.*.jsonl')):
                inputs.append(dict(path=str(path),bytes=path.stat().st_size))
                rows.extend(read_rows(path))
            probe_games.extend(read_rows(root/'probe'/f'{run}_{name}'/'games.jsonl'))
        final={r['seed']:r for r in probe_games}
        late=sorted((r for r in rows if r['wave']>=25),key=lambda r:(r['seed'],r['decision']))
        all_rows[name]=late
        probes[name]=dict(games=[{k:r[k] for k in ('seed','won','final_wave','wave_count')}
                                for r in sorted(probe_games,key=lambda r:r['seed'])],
            late=describe(late),one_wave_left=describe([r for r in late if r['wave']==29]),
            per_seed={str(seed):dict(won=final[seed]['won'],final_wave=final[seed]['final_wave'],
                late=describe([r for r in late if r['seed']==seed]),
                one_wave_left=describe([r for r in late if r['seed']==seed and r['wave']==29]))
                for seed in sorted(final)})
    selected=[]
    for seed in sorted({r['seed'] for r in all_rows['v6']}):
        rows=[r for r in all_rows['v6'] if r['seed']==seed]
        for i in range(4):
            row=rows[round(i*(len(rows)-1)/3)]
            selected.append({k:row[k] for k in ('seed','decision','wave','mean','top1_top2','overridden')} |
                            dict(top_score=row['values'][0]))
    return dict(complete=complete,plan=plan,task=task,paired=comparison,blocks=blocks,
        episode_stats=episode_stats,harmed_seeds=harms,probes=probes,selected20_v6=selected,
        harmed_loss_waves=dict(sorted(Counter(r['final_wave'] for r in games['v6'] if r['seed'] in harms).items())),
        input_probe_files=inputs,probe_bytes=sum(r['bytes'] for r in inputs),
        limitations=[
            'Two arms follow different trajectories after their first differing action; these are not same-root value comparisons.',
            'Top1-top2 is the gap between sorted finite candidate means, not challenger minus policy. Older probes did not save indexed policy scores.',
            'Halving candidates have different sample counts; recorded values also include terminal 0/1 outcomes, not only neural predictions.',
            'wave=29 of wave_count=30 means one wave has not spawned; it does not imply the board is safe or the win probability should be high.',
            'Repeated decisions in five selected probe games are correlated and cannot be counted as independent games.',
            'The 32-pair extension was selected after the original eight-pair warning; p values are descriptive, not a fresh preregistered confirmatory test.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('artifacts/task154'))
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    report=summarize(args.root,args.partial)
    target=args.root/'l9_followup.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(complete=report['complete'],paired=report['paired'],
        probes={k:v['late'] for k,v in report['probes'].items()},report=str(target)),indent=2))


if __name__=='__main__':
    main()
