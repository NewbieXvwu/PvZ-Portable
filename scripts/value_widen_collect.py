"""Collect result-labelled teacher games for missing cross-zone outcome support.

Only this worktree's executable/output are mutated. Runs sequential groups with
at most four simulator workers; search settings match queue103 except beta=1.
Existing search_play result rows make each group resumable.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
MAIN=Path('/home/newbiexvwu/PvZ-search')


def command(task, manifest, out, workers, games, offset, label):
    return [sys.executable,str(ROOT/'scripts/search_play.py'),
            '--rollout-policy',str(MAIN/'artifacts/student/dw4i_s1/model.pt'),
            '--verify-samples','9','--student-candidates','2',
            '--static-value',str(MAIN/'artifacts/hard/value_branch1/model.pt'),
            '--value',str(MAIN/'artifacts/value_net/v3/model.pt'),
            '--manifest',str(manifest),'--task-id',task,'--seed-offset',str(offset),'--seeds',str(games),
            '--executable',str(ROOT/'build/pvz-portable'),'--dagger-beta','1','--workers',str(workers),
            '--resource-dir','/home/newbiexvwu/.cache/pvz-research-resources',
            '--frames',str(out/'frames'),'--frames-label',label,'--out',str(out/label)]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--games',type=int,default=32)
    p.add_argument('--seed-offset',type=int,default=6144)
    p.add_argument('--only',default='',help='comma-separated labels; empty runs the three original missing-outcome groups')
    a=p.parse_args()
    if not 1<=a.workers<=4:p.error('--workers must be 1..4')
    a.out.mkdir(parents=True,exist_ok=True)
    for task,manifest,label in [
        ('pool_l24_train_p9k',MAIN/'artifacts/stage4/pool_panel.json','task30_pool24'),
        ('pool_l27_train_p9k',MAIN/'artifacts/stage4/pool_panel.json','task30_pool27'),
        ('night_l20_train_n10_orig',MAIN/'artifacts/stage4/night_panel.json','task30_night20'),
        ('pool_l21_train_p9k',MAIN/'artifacts/stage4/pool_panel.json','task30_pool21'),
        ('day_l10_orig_train',ROOT/'artifacts/task30/day_panel.json','task30_day10')]:
        if a.only and label not in a.only.split(','):continue
        if not a.only and label in ('task30_pool21','task30_day10'):continue
        cmd=command(task,manifest,a.out,a.workers,a.games,a.seed_offset,label)
        (a.out/f'{label}_command.json').write_text(json.dumps(cmd,indent=1))
        print(f'collect {label}',flush=True)
        with (a.out/f'{label}.log').open('a') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
        print((a.out/label/'summary.json').read_text(),flush=True)
if __name__=='__main__':main()
