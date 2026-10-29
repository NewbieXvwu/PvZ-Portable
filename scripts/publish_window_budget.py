"""Publish the next-window measurements without keeping an agent waiting for production.

Edits only task162 and an append-only daily note in a disposable detached checkout.
Push is ordinary fast-forward only. A clean gpuwindow workspace may then fast-forward;
the immutable code checkouts used by queued jobs are never moved.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

from gpu_window_runner import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    probe = args.state/'decision-data/throughput'
    report = json.loads((probe/'report.json').read_text())
    revision = report['revision']
    marker = f'<!-- task162-throughput-{revision} -->'
    checkout = args.state/'publication'/revision
    checkout.parent.mkdir(parents=True, exist_ok=True)
    def git(*command, cwd=args.workspace, input=None, check=True):
        return subprocess.run(['git','-C',str(cwd),*command],input=input,text=True,capture_output=True,check=check,timeout=40)
    git('fetch','origin','gpuwindow:refs/remotes/origin/gpuwindow')
    if checkout.exists():
        # Only this program owns the disposable checkout; a previous failed push
        # is retained as a commit in the shared object store before rebuilding.
        git('reset','--hard','origin/gpuwindow',cwd=checkout)
    else:
        git('worktree','add','--detach',str(checkout),'origin/gpuwindow')
    task = checkout/'MEMORY/docs/task162-gpu-window-runner.md'
    if marker not in task.read_text():
        with task.open('a') as handle:
            handle.write('\n'+marker+'\n'+(probe/'report.md').read_text())
        today = datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
        daily = checkout/'MEMORY'/(today+'.md')
        if not daily.exists():
            daily.write_text('# '+today+'\n')
        budget = report['budget']
        release = '磁盘余量不足，尚未放行训练' if budget.get('disk_blocked') else '共同预算已放行'
        with daily.open('a') as handle:
            handle.write(f"\n- task162下个生产空档实测已完成：8臂吞吐与近48小时窗口折算写任务书；{release}：{budget['epochs']}轮、行比例{budget['train_row_fraction']:.6f}，seed1约{budget['planned_seed1_days']:.2f}训练日历天，缓存≤{budget['sidecar_max_bytes']/(1<<30):.2f}GiB；代码固定{revision}。完整数值见任务书及~/pvz-logs/gpu-window/decision-data/throughput/report.json。\n")
        git('add','MEMORY/docs/task162-gpu-window-runner.md',str(daily.relative_to(checkout)),cwd=checkout)
        tracked = git('ls-files','-z','artifacts','logs',cwd=checkout).stdout
        ignored = git('check-ignore','-z','--stdin','--no-index',cwd=checkout,input=tracked,check=False)
        if ignored.stdout:
            raise RuntimeError('tracked ignored artifacts exist; publication only stages documentation')
        git('diff','--cached','--numstat',cwd=checkout)
        git('diff','--cached','--stat',cwd=checkout)
        git('commit','-m','docs(task162): publish measured window throughput and uniform budget',cwd=checkout)
        git('push','origin','HEAD:gpuwindow',cwd=checkout)
    commit = git('rev-parse','HEAD',cwd=checkout).stdout.strip()
    branch = git('branch','--show-current').stdout.strip()
    clean = git('diff','--quiet',check=False).returncode==0 and git('diff','--cached','--quiet',check=False).returncode==0
    synced = False
    if branch=='gpuwindow' and clean:
        synced = git('merge','--ff-only',commit,check=False).returncode==0
    git('worktree','remove',str(checkout))
    write_json(args.out/'ready.json',dict(commit=commit,workspace_fast_forwarded=synced,revision=revision,
        report=str(probe/'report.json'),temporary_checkout_removed=True))


if __name__ == '__main__':
    main()
