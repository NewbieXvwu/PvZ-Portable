#!/usr/bin/env python3
"""Assemble existing import/scan/register results and a controller deployment plan.

No games, archive extraction, replay, model hashing or audit reruns. Reading the
small result JSONs is independent of the running collection processes.
"""
import argparse
import collections
import datetime as dt
import json
import re
from pathlib import Path

from llm_import_sha import file_sha, save
from task105_pipeline import ENGINE_SHA, scan_finished
from task105_state import metadata_lock

ROOT = Path(__file__).resolve().parents[1]


def collect():
    out = ROOT / 'artifacts/task105'
    progress = out / 'import/merged_progress.json'
    if not progress.exists():
        progress = out / 'import/import_progress.json'
    rows = json.loads(progress.read_text())['archives']
    plan = json.loads((out / 'scan_jobs.json').read_text())
    tasks = {t['task_id']: t for t in json.loads((ROOT / 'configs/human_demo_tasks.json').read_text())['tasks']}
    layers = json.loads((ROOT / 'configs/llmplay_seeds.json').read_text())['layers']
    counts = collections.defaultdict(lambda: dict(archives=0, won=set(), robust=set(), rejected=0, reasons=[]))
    for ids in layers.values():
        for task in ids:
            counts[task]
    for row in rows:
        cell = counts[row['base_task_id']]
        cell['archives'] += 1
        if row['verify'] == '通过' and row['grade'] != 'loss':
            cell['won'].add(row['seed'])
            if row['grade'] == 'full':
                cell['robust'].add(row['seed'])
        if row['verify'] in ('失败', '拒绝'):
            cell['rejected'] += 1
            cell['reasons'].append(row['note'])
    summary = {task: {**c, 'won': len(c['won']), 'robust': len(c['robust'])} for task, c in counts.items()}
    production_path = ROOT.parent / 'cq/configs/human_demo_tasks.json'
    production_ids = ({t['task_id'] for t in json.loads(production_path.read_text())['tasks']}
                      if production_path.exists() else set())
    scans, additions, rates = [], [], {}
    for entry in plan['archives']:
        task_id = entry['task_id']
        path = out / 'scan/results' / f'{task_id}_s{entry["seed"]}.json'
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        # Another worker may have finished adaptive but not registered yet.
        # Publish an entry only once the full scan→register step is ready.
        if not scan_finished(entry, out, tasks, result):
            continue
        ladder = result.get('ladder')
        spec = tasks[task_id].get('demo_start')
        top = (result['points'][str(ladder['top'])]['teacher'] if ladder else {})
        registered = bool(spec)
        deployed = task_id in production_ids
        scans.append(dict(task_id=task_id, seed=entry['seed'], games=result['scan_games'],
                          l50=ladder['l50'] if ladder else None, top=ladder['top'] if ladder else None,
                          decisions=spec['decisions'] if spec else [], registered=registered, deployed=deployed,
                          future_seed_start=spec.get('future_seed_start') if spec else None,
                          future_seed_count=spec.get('future_seed_count') if spec else None,
                          teacher_top=dict(wins=top.get('wins', 0), games=top.get('games', 0)),
                          reason='' if registered else ('已测老师点均<50%' if not ladder else '等待register落盘')))
        if registered and not deployed:
            additions.append(tasks[task_id])
            rates[task_id] = dict(wins=top['wins'], games=top['games'],
                                  model='r11 + v5 + branch1', source='task105 deploy14 adaptive top')
    evidence_path = ROOT / 'MEMORY/docs/task105-import-evidence.json'
    previous = json.loads(evidence_path.read_text()) if evidence_path.exists() else {}
    input_receipt = out / 'input_receipt.json'
    source = json.loads(input_receipt.read_text()) if input_receipt.exists() else {}
    packages_path = out / 'input_packages.json'
    packages = json.loads(packages_path.read_text())['packages'] if packages_path.exists() else []
    if packages:
        source = dict(identity=packages[-1]['identity'], sha256=None)
    result = dict(schema='task105_results_v1', generated=dt.datetime.now(dt.timezone.utc).isoformat(),
                  input_sha256=source.get('sha256', previous.get('input_sha256')),
                  initial_input_sha256=previous.get('initial_input_sha256', previous.get('input_sha256')),
                  input_path=source.get('identity', {}).get('path', previous.get('input_path')),
                  input_packages=packages,
                  engine_sha256=ENGINE_SHA,
                  archives=len(rows), verified=sum(r['verify'] == '通过' for r in rows),
                  rejected=sum(r['verify'] in ('失败', '拒绝') for r in rows), tasks=summary, scans=scans,
                  scan_expected=len(plan['archives']), scanned=len(scans),
                  registered=sum(s['registered'] for s in scans), deployed=sum(s['deployed'] for s in scans),
                  scan_games=sum(s['games'] for s in scans),
                  complete=len(scans) == len(plan['archives']))
    save(out / 'delivery_summary.json', result)
    save(evidence_path, result)
    preparation = dict(schema='task105_preparation_v1', prepared=True, package_complete=result['complete'],
                       scope=[t['task_id'] for t in additions], engine_sha256=ENGINE_SHA,
                       scanned=len(scans), scan_expected=len(plan['archives']))
    save(out / 'preparation.json', preparation)
    save(out / 'deployment_plan.json', dict(schema='task105_deployment_v1', prepared=True,
         package_complete=result['complete'], engine_sha256=ENGINE_SHA, tasks=additions, teacher_rates=rates,
         archive_sha256={t['task_id']: Path(t['demo_start']['archive']).name for t in additions},
         preparation_sha256=file_sha(out / 'preparation.json')))
    write_report(result, layers)
    return result


def report_prefix(text):
    prefix, _, suffix = text.partition('## 结果\n')
    # Controller addenda may be appended after the generated result section.
    # Move them into the preserved prefix before replacing that section.
    for section in re.split(r'(?=^## )', suffix, flags=re.M):
        if section.startswith('## 追加输入'):
            prefix = prefix.rstrip() + '\n\n' + section.rstrip() + '\n\n'
    return prefix


def write_report(result, layers):
    path = ROOT / 'MEMORY/docs/task105-gptpro-run1-import.md'
    prefix = report_prefix(path.read_text())
    won = sum(c['won'] for c in result['tasks'].values())
    robust = sum(c['robust'] for c in result['tasks'].values())
    lines = ['## 结果', '',
             f"- 累计{result['archives']}份SHA去重存档：deploy14 verify通过{result['verified']}份、拒绝{result['rejected']}份；"
             f"按(任务,种子)去重为{won}个通关种子、{robust}个稳健种子。稳健按随包第10波k=8且≥5胜记录。",
             '- verify结果按存档SHA持久化。后续新SHA用2个游戏进程并行首次verify；已有成功或失败回执直接复用，绝不二次重放。',
             f"- 接手扫描完成{result['scanned']}/{result['scan_expected']}份，登记{result['registered']}格，"
             f"其中{result['deployed']}格已出现在生产qc清单。其余已登记格可由主控组边界部署；本任务未执行部署脚本。",
             '- 原adaptive→register链继续，最多2个游戏进程、qc优先。已移除额外审计、全包重导入、重复register、全机种子复查等尾链。',
             '- 策略仅留本机原zip参考；存档、游戏资源未进git、未上传HF。', '',
             '| 层 | 任务 | 通关种子 | 稳健种子 | 拒绝份数 |', '|---|---|---|---|---|']
    for layer, ids in layers.items():
        for task in ids:
            c = result['tasks'][task]
            lines.append(f"| {layer} | {task} | {c['won']} | {c['robust']} | {c['rejected']} |")
    known = {task for ids in layers.values() for task in ids}
    for task in sorted(result['tasks'].keys() - known):
        c = result['tasks'][task]
        lines.append(f"| 其他 | {task} | {c['won']} | {c['robust']} | {c['rejected']} |")
    reasons = [(task, reason) for task, c in result['tasks'].items() for reason in c['reasons']]
    lines += ['', '拒绝原因：' + ('无。' if not reasons else '；'.join(f'{t}：{r}' for t, r in reasons)), '',
              '| 已完成扫描 / 起点格 | L50 | 顶档 | 已登记阶梯（晚→早） | 老师顶档 | 课程未来种子块 | qc状态 |',
              '|---|---|---|---|---|---|---|']
    for scan in result['scans']:
        top = scan['teacher_top']
        status = ('已在qc清单' if scan['deployed'] else '已登记，待主控部署') if scan['registered'] else scan['reason']
        start, count = scan['future_seed_start'], scan['future_seed_count']
        block = f'{start}–{start + count - 1}' if start is not None and count else '—'
        lines.append(f"| `{scan['task_id']}` | {scan['l50'] or '—'} | {scan['top'] or '—'} | "
                     f"{' / '.join(map(str, scan['decisions'])) or '—'} | {top['wins']}/{top['games']} | {block} | {status} |")
    lines += ['', '起点格未来种子块见`configs/human_demo_tasks.json`中的`demo_start`；未完成扫描不预登记。', '',
              '## 继续运行与交付', '',
              '已有进程活着时不另起同一任务。入口：`scripts/task105_pipeline.py <zip>`；状态和断点在`artifacts/task105/`。',
              '同一 run_id 的续跑包直接换 zip 重跑入口即可：按存档 SHA 增量导入（旧回执复用、不重放），'
              '`adaptive --skip-complete` 跳过已完成的扫描，register 幂等。',
              '多包入口`scripts/task105_import_batch.py <zip>...`串行增量导入，按SHA合并账本和作业；'
              '每包完成后保存路径、大小和mtime回执，续跑直接复用。随后以`--skip-import`续扫并跳过已完成结果。',
              '引擎进程会被 earlyoom 打断（其他工作树的 pytest 峰值 12–17 GB）。单份失败不整链停摆，'
              '记入`artifacts/task105/failures.json` 后继续；重跑时已测点从结果 JSON 续跑，不重放。',
              '`search_play`使用ProcessPoolExecutor，worker意外退出时报错，避免替补worker卡死后无限等待。'
              '续跑直接跳过已扫描且已登记的条目；有失败项时流水线记为incomplete，不能冒充整包完成。',
              '`scripts/task105_supervise.py <zip>`可接管已有流水线，等待其退出后补跑未完成项；'
              '看守进程使用独立锁，不并起同一任务；多包入口在导入完成后直接转为看守进程。',
              '结果整理入口`scripts/task105_results.py`只读取已有小JSON，不解压、不重放、不复跑。',
              '部署脚本为`scripts/deploy_task105.py`；当前计划为`artifacts/task105/deployment_plan.json`，仅含已登记且尚未在qc清单中的格。',
              '主控在组边界排空qc后才应用；本任务不执行该脚本（含预览）。', '']
    remaining = result['scan_expected'] - result['scanned']
    lines += ['## 交付结论', '',
              f"- 已增量导入{result['archives']}个唯一SHA：{result['verified']}份通过、{result['rejected']}份拒绝。拒绝原因见上方结果节。",
              f"- 接手扫描{result['scanned']}/{result['scan_expected']}份，已登记{result['registered']}格、生产qc清单{result['deployed']}格。"
              + (f'剩余{remaining}份继续断点扫描。' if remaining else '全部接手扫描已完成。'),
              '- 未执行部署；已登记且未部署的起点格由主控在组边界应用。存档、策略及游戏资源只留本机。', '']
    text = prefix + '\n'.join(lines)
    path.write_text(text)
    brief = Path('/home/newbiexvwu/pvz-logs/task105-brief.md')
    if brief.exists():
        brief.write_text(report_prefix(brief.read_text()) + '## 结果\n'
                         + text.split('## 结果\n', 1)[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='?', help='accepted for the legacy running-chain entry; not reopened')
    parser.parse_args()
    with metadata_lock(ROOT):
        result = collect()
    print(f"task105: {result['scanned']}/{result['scan_expected']} scanned, "
          f"{result['registered']} registered, {result['deployed']} in production")


if __name__ == '__main__':
    main()
