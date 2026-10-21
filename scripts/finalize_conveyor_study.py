"""Finish task20's evidence and documents after its isolated controller exits successfully."""
import json
from datetime import datetime
import subprocess
import time
from pathlib import Path
from zoneinfo import ZoneInfo

from run_conveyor_study import OUT, ROOT, guard, read, sha, verify, write


def today():
    return datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()


def record_task(text):
    path = ROOT/'MEMORY/docs/task20-conveyor-fidelity.md'
    previous = path.read_text()
    before, after = previous.split('\n## 问题', 1)
    path.write_text(before+'\n- '+today()+'，'+text+'\n\n## 问题'+after)


def update_section(path, text):
    old = path.read_text()
    heading = '\n## 9. task20：原版 1-10 传送带（2026-10-21）'
    assert heading in old
    # This task owns the final task20 section only; never replace earlier panel history.
    prefix, section = old.split(heading, 1)
    assert '\n## ' not in section
    path.write_text(prefix+heading+'\n\n'+text+'\n')


def document_baseline():
    report = read(ROOT/'MEMORY/docs/task20-baseline.json')
    b = report['baseline']; teacher = b['teacher']; student = b['identity']
    text = ('原任务在reset写入固定初始卡组，之后仍有原生传送带补卡；该变体停止新增评估与采集，旧档案保留训练。'
            '新任务`day_l10_orig_eval`使用引擎10格空传送带开局、原生免费出卡，deck为空。'
            '旧slot输入不变；identity的Python/C++输入已按引擎原版标志对齐。\n\n'
            '| 模型 / 实际执行方式 | 胜局 | 评估种子 |\n|---|---:|---|\n'
            f"| v34学生 | {b['v34']['won']}/256 | 30000–30255 |\n"
            f"| task19 identity学生（输入对齐后） | {student['won']}/256 | 30000–30255 |\n"
            f"| v34+搜索老师（beta=1，搜索执行） | {teacher['won']}/64 | 30000–30063 |\n\n"
            f"同一前64种子中，v34学生为{teacher['student_v34_same_64_won']}/64。"
            '老师64局是该搜索策略的实测水平，样本量和学生256局不同，不能视作数学上限。'
            '旧identity 247/256使用了未对齐输入，旧老师24局使用beta=0，均只保留历史，不并入表中。'
            'v34沿用cb已完成基线，历史程序哈希不可得，未用它比较吞吐。\n\n'
            'DAgger仍按主控要求beta=0（学生执行、老师标注），同一冻结identity老师采两轮各128局，'
            '加入原82标签，从头训练seed1原12epoch，随后评原版和七卡L7–L9各256局。'
            '[基线核验](task20-baseline.json)、[冻结计划](task20-study-plan.json)、'
            '[七卡对照和原数据清单](task20-reference.json)。')
    update_section(ROOT/'MEMORY/docs/day-panel.md', text)
    record_task(f"5完成（重新核验）：原版学生v34 {b['v34']['won']}/256、对齐identity {student['won']}/256，"
                f"老师beta=1 {teacher['won']}/64（同前64种子v34 {teacher['student_v34_same_64_won']}/64）。"
                '逐seed/config/均值摘要/冻结身份重读通过，结果见task20-baseline.json与day-panel第9节；两轮采集继续beta=0。')
    with (ROOT/f'MEMORY/{today()}.md').open('a') as stream:
        stream.write(f"- task20老师执行基线核验完成：v34 {b['v34']['won']}/256、对齐identity {student['won']}/256、"
                     f"v34+搜索beta1 {teacher['won']}/64；同前64种子v34 {teacher['student_v34_same_64_won']}/64。"
                     '已写day-panel原版独立任务族，不混入旧beta0或旧输入结果，继续既定beta0采集/单种子试验。\n')


def legacy_contract(plan):
    """Task20 item1's engineering replay contract, not a new study of the retired variant."""
    marker = OUT/'legacy_contract.json'
    if marker.exists():
        assert read(marker)['executable_sha256'] == plan['executable_sha256']
        return
    verify(plan)
    source = Path.home()/'PvZ-search/artifacts/day_panel/dw3s1_l10_d7_256'
    config = read(source/'config.json')
    assert sha(Path(config['rollout_policy'])) == plan['model_sha256']['v34']
    manifest = read(Path(config['manifest']))
    task = next(t for t in manifest['tasks'] if t['task_id'] == 'day_l10_eval_d7')
    local = OUT/'legacy_contract_manifest.json'
    write(local, dict(schema_version=1, tasks=[task]))
    dest = OUT/'legacy_contract_eight'
    command = [guard.PYTHON, 'scripts/search_play.py', '--rollout-policy', plan['models']['v34'],
               '--value', plan['models']['value'], '--manifest', str(local), '--task-id', task['task_id'],
               '--executable', str(ROOT/'build/pvz-portable'), '--resource-dir', guard.ENV['PVZ_RESOURCE_DIR'],
               '--workers', '4', '--seeds', '8', '--arms', 'fast', '--out', str(dest)]
    resource = guard.run(command, 't20_legacy_contract', extra={'CUDA_VISIBLE_DEVICES': ''})
    assert resource['returncode'] == 0
    current = [json.loads(v) for v in (dest/'games.jsonl').read_text().splitlines()]
    old = {r['seed']: r for r in map(json.loads, (source/'games.jsonl').read_text().splitlines())}
    assert len(current) == 8 and {r['seed'] for r in current} == set(range(30000, 30008))
    for row in current:
        assert {k:v for k,v in row.items() if k != 'seconds'} == {k:v for k,v in old[row['seed']].items() if k != 'seconds'}
    verify(plan)
    write(marker, dict(games=8, identical_except_seconds=True, executable_sha256=plan['executable_sha256'],
                       source_games_sha256=sha(source/'games.jsonl'), current_games_sha256=sha(dest/'games.jsonl'),
                       resources=resource, scope='Mandatory default-task compatibility check only; retired variant not studied'))


def document_final():
    result = read(ROOT/'MEMORY/docs/task20-results.json')
    assert result['complete'] and result['evaluated_games'] == 1024
    rows = result['evaluation']; orig = next(r for r in rows if r['task_id'] == 'day_l10_orig_eval')
    panel = ROOT/'MEMORY/docs/day-panel.md'
    extra = ('\n\n小规模DAgger实测已完成：两轮各128份原版档案、beta=0（学生执行、老师标注），'
             '与原38,271份档案合并为38,527份/84标签；从头训练一个seed1 identity+row_pool、原12epoch配方，'
             '不挑epoch或种子。\n\n| 任务 | 旧identity | 加原版数据后 | 净胜局变化 | 净变化pp |\n'
             '|---|---:|---:|---:|---:|\n')
    for row in rows:
        extra += f"| {row['task_id']} | {row['reference_won']}/256 | {row['won']}/256 | {row['delta_wins']:+d} | {row['delta_pp']:+.2f} |\n"
    extra += ('\n完整1024局、逐seed配对得失、采集标签/实际动作差异、档案完整性和训练配方已独立重读，'
              '见[最终结果](task20-results.json)。原版基线已达256/256，本次只能观察是否保持及七卡回退，'
              '不能从单种子宣称泛化或非劣。增加数据也改变验证种子划分和优化步数；无默认/生产模型晋升。'
              '旧任务兼容性另做8局工程检查（除seconds外逐字段一致），不作退役任务新研究。')
    with panel.open('a') as stream: stream.write(extra+'\n')
    wins = '/'.join(str(r['won']) for r in rows)
    record_task('6完成：两轮beta0各128份原版归档，合并原数据38,527份/84标签，'
                f'seed1 identity+row_pool原12epoch从头训练；原版/L7/L8/L9各256局胜局{wins}。'
                f"原版对照{orig['reference_won']}→{orig['won']}；完整结果见task20-results.json与day-panel第9节。"
                '档案/模式/标签/检查点/源哈希/配方/所有种子实际重读通过；另8局旧任务兼容检查除seconds外逐字段相同。'
                '单种子描述性试验，不晋升默认/生产模型。')
    with (ROOT/f'MEMORY/{today()}.md').open('a') as stream:
        stream.write(f'- task20小试验收束：原版beta0采集2×128、原数据合并38,527档案/84标签，'
                     f'seed1原12epoch，原版及七卡L7/L8/L9各256局胜局{wins}；'
                     '独立重读完整1024局/归档/标签/模型/配方，旧默认8局除seconds外逐字段一致。'
                     '报告在task20-results.json/day-panel第9节，无默认/生产晋升，单种子不作非劣推断。\n')
    write(OUT/'finalization_complete.json', dict(complete=True, results_sha256=sha(ROOT/'MEMORY/docs/task20-results.json')))


if __name__ == '__main__':
    marker = OUT/'finalization_complete.json'
    if not marker.exists():
        controller = read(OUT/'study_controller.process.json')['pid']
        while not (OUT/'baseline_complete.json').exists():
            assert guard.live(controller), 'study controller exited before baseline completed'
            time.sleep(10)
        while guard.available() < 6*1024*1024: time.sleep(10)
        subprocess.run([guard.PYTHON, 'scripts/report_conveyor_study.py'], cwd=ROOT, env=guard.ENV, check=True)
        if not (OUT/'baseline_documented.json').exists():
            document_baseline(); write(OUT/'baseline_documented.json', dict(complete=True))
        while guard.live(controller): time.sleep(10)
        assert read(OUT/'evaluation_complete.json')['complete']
        while guard.available() < 6*1024*1024: time.sleep(10)
        legacy_contract(read(OUT/'study_plan.json'))
        subprocess.run([guard.PYTHON, 'scripts/report_conveyor_study.py', '--final'], cwd=ROOT, env=guard.ENV, check=True)
        document_final()
