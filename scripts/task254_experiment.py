"""Task254: existing branch collector, v5 fine-tuner and paired panel, without deployment.

Run in the foreground or let the controller launch this command in tmux. Completed
games/stages are reused; original task246 features must be restored before training.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'python')]
from value_teacher_feasibility import write_json


def read(path):
    return json.loads(Path(path).read_text())


def diagnose(original, panel):
    jobs = {j['game']: j for j in read(original/'plan.json')['jobs']}
    labels = read(original/'branch-labels.json')
    result = {}
    for task in panel:
        level = task['level']
        selected = [j for j in jobs.values() if j['level'] == level]
        roots = [r for r in labels if r['game'] in {j['game'] for j in selected}]
        outcomes = [r for root in roots for r in root['labels']]
        wins = sum(r['won'] for r in outcomes)
        result[str(level)] = dict(terrain=task['terrain'], panel_multiplier=task['zombie_count_multiplier'],
            games=len(selected), roots=len(roots), labels=len(outcomes), wins=wins,
            positive_rate=wins/len(outcomes) if outcomes else None,
            mixed_roots=sum(0 < sum(r['won'] for r in root['labels']) < len(root['labels']) for root in roots),
            archives=[j['archive'] for j in selected],
            source_multiplier_verified=False)
    return result


def prepare(args):
    base = args.data.resolve()
    (base/'work').mkdir(parents=True, exist_ok=True)
    (base/'models').mkdir(exist_ok=True)
    os.environ.update(TMPDIR=str(base/'work'), PVZ_NO_POPUP='1', OMP_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    import tempfile
    tempfile.tempdir = str(base/'work')
    for name in ('student.pt', 'gptbc.pt', 'production.pt', 'roof.pt', 'static.pt',
                 'production.bin', 'roof.bin', 'static.bin', 'finetune-t3.pt'):
        if not (base/'models'/name).exists():
            shutil.copy2(args.original/'models'/name, base/'models'/name)
    task246 = read(ROOT/'MEMORY/docs/task246-vn-branch.json')
    task252 = read(ROOT/'MEMORY/docs/task252-fog-roof-route.json')
    tasks = [t for t in task246['panel']['tasks'] if t['terrain'] == 'pool']
    tasks += [t for t in task252['panel']['tasks'] if t['terrain'] in ('fog', 'roof')]
    panel_path = base/'panel.json'
    if not panel_path.exists():
        write_json(panel_path, dict(tasks=[dict(t, task_id=f"task254_l{t['level']}",
            seeds=[254200000+t['level']*1000+k for k in range(32)]) for t in tasks]))
    report_path = ROOT/'MEMORY/docs/task254-vn-balance.json'
    report = read(report_path) if report_path.exists() else dict(task='task254', branch='vnbalance')
    report.update(production_deployed=False, mechanism=diagnose(args.original,
                  [t for t in tasks if t['terrain'] in ('pool', 'fog')]))
    for level, row in report['mechanism'].items():
        if row['terrain'] == 'fog':
            paired = task252['per_level'][level]
            row.update(panel_net_wins=paired['candidate_wins']-paired['incumbent_wins'],
                       panel_worse_p=paired['mcnemar_worse_p'])
    report['mechanism_conclusion'] = (
        'L31正率83.3%（320/384），存在全正根，但未达到L21/L24的97%-98%。'
        'L36/L37没有任何原分支标签，属于关卡覆盖缺口。L38/L39/L40同样无分支标签，'
        '其中L38/L40却变好，因此覆盖缺口不能单独解释变差方向。'
        '原元数据和特征已清理，档案名中的倍率字样不能证明与task252面板倍率及卡组一致。')
    missing = [name for name in ('features', 'branches') if not (args.original/name).is_dir()]
    report['original_data'] = dict(path=str(args.original), missing=missing,
        required='task246 features/*.npz（轨迹终局特征）及 branches/game_*/ '
                 '（rows.npz、selection.json、complete.json、root_*/labels.json），另需 plan.json 与 split.json',
        recovered=False if missing else True)
    report['cost_estimate'] = dict(workers=4, levels_pool=5, levels_fog=9, roots_per_level=[8, 12],
        futures=4, candidates_assumed=12, continuation_seconds=dict(pool=123.42, fog=30.43),
        hours_including_20_percent_overhead=[28.51, 42.77],
        note='不含电池暂停，12候选是估算值。每关最多三批，每批四根，未达到平衡也在三批后停止。',
        disk_peak_estimated_gib=10, disk_limit_gib=15)
    report['protocol'] = dict(target_positive_rate=[.4, .8], root='开局根，候选由现有C++搜索给出，学生候选从2增至4',
        adaptation='仅训练种子标签决定下一批倍率：正率高于80%乘1.25，低于40%乘0.8。'
                   '从冻结的面板倍率开始，卡组不变，保留所有候选和共同未来。',
        split='每批四个种子固定为训练2个、验证1个、测试1个，生成标签前冻结。',
        training='task246_train --task task254，生产v5起点，编码2宽64，'
                 '16轮，学习率1e-4，权重衰减1e-4，丢弃率0，批次512，轨迹与分支各50%',
        temperature='按地形仅用验证集拟合对数损失，轨迹与分支各50%，温度固化进输出层。',
        panel='每关32对新种子，泳池及雾对比生产，屋顶五关对比finetune-t3。'
              '候选用绝对0.02门槛，屋顶对照用相对0.02，生产对照用绝对0.02。',
        acceptance='每关净差>=-2/32，单侧变差p>=0.05，每地形合并95%区间下界>=-2/32')
    report['reproduction'] = dict(
        collect='python3 scripts/task254_experiment.py --phase collect --workers 4',
        full='python3 scripts/task254_experiment.py --phase all --workers 4 --original <restored-task246-directory>',
        resume_training='python3 scripts/task254_experiment.py --phase train --original <restored-task246-directory>',
        panel='python3 scripts/task254_experiment.py --phase panel --workers 4',
        cwd=str(ROOT), log_directory=str(base),
        launch='前台运行或由主控使用tmux启动。本轮未启动后台长任务。')
    report.setdefault('new_labels', {})
    report.setdefault('recommendation', dict(pool=False, fog=False, reason='尚无重训及整局证据，保持现有生产配置'))
    return base, report, report_path


def summary(base, jobs):
    results = {}
    for level in sorted({j['level'] for j in jobs}):
        counts = dict(games=0, roots=0, labels=0, wins=0, rows=0, failed=0, train_labels=0, train_wins=0)
        for job in (j for j in jobs if j['level'] == level):
            target = base/'branches'/f"game_{job['game']:04d}"
            if (target/'failed.json').exists():
                counts['failed'] += 1
            if not (target/'complete.json').exists():
                continue
            row = read(target/'complete.json')
            counts['games'] += 1
            counts['roots'] += len(row['roots'])
            counts['rows'] += row['rows']
            for root in row['roots']:
                counts['labels'] += root['labels']
                counts['wins'] += root['wins']
                if job['partition'] == 'train':
                    counts['train_labels'] += root['labels']
                    counts['train_wins'] += root['wins']
        counts['positive_rate'] = counts['wins']/counts['labels'] if counts['labels'] else None
        counts['train_positive_rate'] = counts['train_wins']/counts['train_labels'] if counts['train_labels'] else None
        results[str(level)] = counts
    return results


def collect(args, base, report, report_path):
    tasks = [t for t in read(base/'panel.json')['tasks'] if t['terrain'] in ('pool', 'fog')]
    all_jobs = []
    for batch in range(3):
        plan_path = base/f'batch-{batch}.json'
        if plan_path.exists():
            jobs = read(plan_path)['jobs']
        else:
            jobs = []
            rates = summary(base, all_jobs)
            for task in tasks:
                level = task['level']
                rate = rates.get(str(level), {}).get('train_positive_rate')
                if batch >= 2 and rate is not None and .4 <= rate <= .8:
                    continue
                previous = [j for j in all_jobs if j['level'] == level]
                multiplier = previous[-1]['task']['zombie_count_multiplier'] if previous else task['zombie_count_multiplier']
                if rate is not None:
                    multiplier *= 1.25 if rate > .8 else .8 if rate < .4 else 1
                for k in range(4):
                    seed = 254000000+level*10000+batch*100+k
                    jobs.append(dict(game=10000+level*100+batch*4+k, base=str(base), level=level,
                        terrain=task['terrain'], partition=('train', 'train', 'validation', 'test')[k],
                        task=dict(task, zombie_count_multiplier=round(multiplier, 6), seeds=[seed]),
                        probe_seed=seed, samples=4, student_candidates=4,
                        executable=str(Path.home()/'pvz-mac-collect/build-vnbranch/pvz-portable')))
            write_json(plan_path, dict(jobs=jobs))
        all_jobs.extend(jobs)
        write_json(base/'new-plan.json', dict(jobs=all_jobs))
        pending = [j for j in jobs if not any((base/'branches'/f"game_{j['game']:04d}"/marker).exists()
                                             for marker in ('complete.json', 'failed.json'))]
        if pending:
            pending_path = base/f'batch-{batch}-pending.json'
            write_json(pending_path, dict(jobs=pending))
            with (base/f'batch-{batch}.log').open('a') as log:
                subprocess.run([sys.executable, str(ROOT/'scripts/task246_collect.py'), '--data', str(base),
                                '--plan', str(pending_path), '--workers', str(args.workers)],
                               stdout=log, stderr=log, check=True)
        report['new_labels'] = summary(base, all_jobs)
        report['status'] = f'补标签完成第{batch+1}批，未训练'
        write_json(report_path, report)
    report['balanced'] = all(v['games'] >= 8 and v['positive_rate'] is not None and
                            .4 <= v['positive_rate'] <= .8 for v in report['new_labels'].values())
    write_json(report_path, report)


def train(args, base, report):
    if not report.get('balanced'):
        raise RuntimeError('新标签尚未达到每关40%-80%或少于8个成功根，保留数据待分析')
    if report['original_data']['missing']:
        raise RuntimeError('缺少原task246特征：'+', '.join(report['original_data']['missing']))
    if not (base/'offline.json').exists():
        features = base/'features'
        features.mkdir(exist_ok=True)
        # Reuse original shards and root reports without duplicating their storage.
        for path in (args.original/'features').glob('*.npz'):
            target = features/path.name
            if not target.exists(): target.symlink_to(path.resolve())
        old = read(args.original/'plan.json')['jobs']
        for job in old:
            target = base/'branches'/f"game_{job['game']:04d}"
            if not target.exists(): target.symlink_to((args.original/'branches'/target.name).resolve())
        new = [j for j in read(base/'new-plan.json')['jobs'] if
               (base/'branches'/f"game_{j['game']:04d}"/'complete.json').exists()]
        write_json(base/'plan.json', dict(jobs=old+new))
        split = read(args.original/'split.json')
        for job in new:
            split[job['partition']].append([job['terrain'], job['probe_seed']])
        write_json(base/'split.json', split)
        with (base/'training.log').open('a') as log:
            subprocess.run([sys.executable, str(ROOT/'scripts/task246_train.py'), '--data', str(base),
                            '--task', 'task254'], stdout=log, stderr=log, check=True)
    report['offline'] = read(base/'offline.json')


def panel_one(job):
    from types import SimpleNamespace
    from task250_experiment import run_arm
    base, task, seed, label = Path(job['base']), job['task'], job['seed'], job['label']
    out = base/'panel'/label/f"l{task['level']}"/str(seed)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'outcomes.json').exists(): return read(out/'outcomes.json')[0]
    if (out/'failed.json').exists(): return None
    reference = 'finetune-t3' if task['terrain'] == 'roof' else 'production'
    model = base/'models'/(f"balanced-{task['terrain']}.pt" if label == 'candidate' else f'{reference}.pt')
    settings = SimpleNamespace(workers=1, margin_relative=.02 if label == 'reference' and task['terrain'] == 'roof' else 0)
    try:
        return run_arm(sys.executable, base, dict(tasks=[task]), label, dict(value=model, roof=model),
                       settings, [task['level']], {task['level']: [seed]}, {}, out)[0]
    except Exception as error:
        write_json(out/'failed.json', dict(error=str(error)))
        return None


def panel(args, base, report, report_path):
    from task250_experiment import report as paired_report
    if 'offline' not in report:
        raise RuntimeError('尚无离线训练结果，不能进入整局面板')
    if not report.get('offline', {}).get('eligible_for_panel'):
        report['whole_game'] = dict(ran=False, reason='离线排序未过关')
        return
    if report.get('whole_game', {}).get('ran') and all(
            t['complete'] for t in report['whole_game']['terrains'].values()):
        return
    tasks = read(base/'panel.json')['tasks']
    jobs = [dict(base=str(base), task=task, label=label, seed=seed)
            for task in tasks for label in ('reference', 'candidate') for seed in task['seeds']]
    rows = {label: [] for label in ('reference', 'candidate')}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
        pending = {pool.submit(panel_one, job): job for job in jobs}
        for future in as_completed(pending):
            row = future.result()
            if row is not None: rows[pending[future]['label']].append(row)
            if sum(map(len, rows.values())) % 32 == 0:
                print('panel completed', {k:len(v) for k,v in rows.items()}, flush=True)
    result = {}
    for terrain in ('pool', 'fog', 'roof'):
        levels = {t['level'] for t in tasks if t['terrain'] == terrain}
        reference = [r for r in rows['reference'] if r['level'] in levels]
        candidate = [r for r in rows['candidate'] if r['level'] in levels]
        keys = {(r['level'], r['seed']) for r in reference}
        if not any((r['level'], r['seed']) in keys for r in candidate):
            result[terrain] = dict(complete=False, eligible=False, pairs=[], reason='无完整终局配对')
            continue
        group = paired_report(candidate, reference)
        group['complete'] = len(group['pairs']) == len(levels)*32
        group['eligible'] = group['complete'] and group['pooled']['ci_noninferior_margin_2_of_32'] and all(
            r['point_noninferior_margin_2_of_32'] and r['mcnemar_worse_p'] >= .05 for r in group['per_level'].values())
        result[terrain] = group
    failures = [dict(level=j['task']['level'], seed=j['seed'], label=j['label']) for j in jobs if
                (base/'panel'/j['label']/f"l{j['task']['level']}"/str(j['seed'])/'failed.json').exists()]
    report['whole_game'] = dict(ran=True, terrains=result, failures=failures)
    report['recommendation'] = {t: result[t]['eligible'] for t in result}
    write_json(report_path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'artifacts/task254')
    parser.add_argument('--original', type=Path, default=ROOT.parent/'vnbranch/artifacts/task246')
    parser.add_argument('--phase', choices=('prepare', 'collect', 'train', 'panel', 'all'), default='prepare')
    parser.add_argument('--workers', type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()
    args.original = args.original.resolve()
    if args.original == args.data.resolve(): parser.error('--data must differ from --original')
    base, report, report_path = prepare(args)
    report.setdefault('status', '机制核对与可续跑驱动已准备，尚未补标签')
    write_json(report_path, report)
    if report.get('completed'):
        print(report['status'])
        return
    try:
        if args.phase in ('collect', 'all'): collect(args, base, report, report_path)
        if args.phase in ('train', 'all'):
            train(args, base, report)
            report['status'] = '离线训练完成，未部署'
            write_json(report_path, report)
        if args.phase in ('panel', 'all'):
            panel(args, base, report, report_path)
            report['status'] = ('整局面板结束，未部署' if report.get('whole_game', {}).get('ran') else
                                '离线未过关，未跑整局，未部署')
    except Exception as error:
        report.update(status='未完成，保留数据待续跑', error=str(error))
        write_json(report_path, report)
        raise
    whole = report.get('whole_game')
    if whole and (not whole['ran'] or all(t['complete'] for t in whole['terrains'].values())):
        report['completed'] = True
        report.pop('error', None)
        write_json(base/'final-report.json', report)
        for path in base.iterdir():
            if path.name in ('models', 'final-report.json'): continue
            if path.is_dir() and not path.is_symlink(): shutil.rmtree(path)
            else: path.unlink()
    write_json(report_path, report)
    print(json.dumps(dict(status=report['status'], original_data=report['original_data']), ensure_ascii=False))


if __name__ == '__main__':
    main()
