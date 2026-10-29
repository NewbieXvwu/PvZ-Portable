"""Task27: executed-teacher dead-deck diagnosis, then both full v2 references (<=4 games)."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
from task27_common import ROOT, MAIN, OUT, PYTHON, read, write, sha, record, run_stage
from unified_panel import summarize
from pvz_frame_store import read_frames
from pvz_recipe import materialize_recipe

MODELS = {'dw4i_s1':MAIN/'artifacts/student/dw4i_s1/model.pt',
          'identity':MAIN/'artifacts/archive/identity/task19/identity_dw3_s1/model.pt'}
DEAD = ('day_l7_eval_d7p_holdout3','day_l8_eval_d7p_holdout1','day_l9_eval_d7p_holdout1')


def plan():
    path = OUT/'pretraining_plan.json'
    if not path.exists():
        assert read(OUT/'type_count_check/complete.json')['complete']
        sources = ['scripts/run_task27_pretraining.py','scripts/task27_common.py','scripts/unified_panel.py',
                   'scripts/search_play.py','scripts/dagger_experiment_runner.py','src/main.cpp',
                   'src/EnvironmentRollout.cpp','python/pvz_rollout_policy.py','python/pvz_event_env.py',
                   'python/pvz_env.py','scripts/render_episode.py','scripts/pvz_tasks.py']
        data = dict(registered_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                    models={k:str(v) for k,v in MODELS.items()},models_sha256={k:sha(v) for k,v in MODELS.items()},
                    manifests_sha256={v:sha(OUT/f'unified_{v}.json') for v in ('v1','v2')},
                    executable_sha256=sha(ROOT/'build/pvz-portable'),sources={s:sha(ROOT/s) for s in sources},
                    dead_tasks=list(DEAD),dead_seeds=list(range(30000,30016)),workers=4,
                    teacher=dict(beta=1,verify_samples=9,student_candidates=2,samples=3,horizon=2500,
                                 min_gap=300,margin=.02,static_value=str(MAIN/'artifacts/hard/value_branch1/model.pt')),
                    value_sha256=sha(MAIN/'artifacts/value_net/v3/model.pt'),
                    static_value_sha256=sha(MAIN/'artifacts/hard/value_branch1/model.pt'))
        write(path,data)
        target = ROOT/'MEMORY/docs/task27-pretraining-plan.json';write(target,data)
        record('2–3评估登记：v2的34项/3264局依次跑dw4i_s1与旧identity；三套全零留出先各16局beta1老师，参数照queue103（verify9、student候选2、静态branch1、3样本2500tick）。自有最多4游戏，模型/清单/源码/二进制SHA固定。',target)
    return read(path)


def verify(data):
    for name,digest in data['sources'].items(): assert sha(ROOT/name) == digest, name
    assert sha(ROOT/'build/pvz-portable') == data['executable_sha256']
    for name,path in data['models'].items(): assert sha(path) == data['models_sha256'][name]
    for version,digest in data['manifests_sha256'].items(): assert sha(OUT/f'unified_{version}.json') == digest
    assert sha(MAIN/'artifacts/value_net/v3/model.pt') == data['value_sha256']
    assert sha(MAIN/'artifacts/hard/value_branch1/model.pt') == data['static_value_sha256']
    if data.get('retained_evidence'):
        initial=read(OUT/'step0/pretraining_plan.initial.json')
        assert sha(OUT/'step0/pretraining_plan.initial.json')==data['prior_pretraining_plan_sha256']
        assert initial['models_sha256']==data['models_sha256'] and initial['manifests_sha256']==data['manifests_sha256']
        assert sha(OUT/'step0/previous-pvz-portable')==initial['executable_sha256']
        for path,digest in data['retained_evidence'].items():assert sha(OUT/path)==digest,path


def diagnose(data):
    manifest = OUT/'unified_v2.json'
    tasks = {t['task_id']:t for t in read(manifest)['tasks']}
    report = {}
    for name in DEAD:
        folder = OUT/'dead_decks'/name;tag = 'dead_'+name
        marker = folder/'diagnosis.json'
        if marker.exists():
            report[name] = read(marker);continue
        command = [PYTHON,str(ROOT/'scripts/search_play.py'),'--rollout-policy',str(MODELS['dw4i_s1']),
                   '--value',str(MAIN/'artifacts/value_net/v3/model.pt'),
                   '--static-value',str(MAIN/'artifacts/hard/value_branch1/model.pt'),
                   '--verify-samples','9','--student-candidates','2','--samples','3','--horizon','2500',
                   '--min-gap','300','--margin','0.02','--dagger-beta','1','--workers','4','--arms','fast_search',
                   '--manifest',str(manifest),'--task-id',name,'--seeds','16',
                   '--executable',str(ROOT/'build/pvz-portable'),
                   '--resource-dir','/home/newbiexvwu/.cache/pvz-research-resources',
                   '--out',str(folder),'--frames',str(OUT/'dead_frames'),'--frames-label',name]
        run_stage(command,tag);verify(data)
        rows = [read_line for read_line in map(__import__('json').loads,(folder/'games.jsonl').read_text().splitlines())]
        assert len(rows) == 16 and {r['seed'] for r in rows} == set(data['dead_seeds'])
        assert all(r['arm'] == 'fast_search' for r in rows)
        waves = Counter();front = Counter();rosters = Counter();games = []
        for row in sorted(rows,key=lambda r:r['seed']):
            archive = OUT/'dead_frames'/f'seed_{row["seed"]}'/(name+'_fast_search')
            frames = read_frames(archive/'frames.npz');meta = read(archive/'meta.json')
            assert meta['outcome']['won'] == row['won']
            assert [p[1] for p in sorted(frames[0]['packets']) if p[1] >= 0] == tasks[name]['deck']
            rosters[str(meta['context']['roster'])] += 1
            # Terminal snapshots identify threats still alive, not a definitive death attribution.
            zombies = frames[-1].get('zombies') or []
            nearest = min(zombies,key=lambda z:z[2]) if zombies else None
            if not row['won']:
                waves[row['final_wave']] += 1
                if nearest is not None: front[nearest[1]] += 1
            games.append(dict(seed=row['seed'],won=row['won'],final_wave=row['final_wave'],
                              nearest_terminal_zombie=nearest,frames_sha256=sha(archive/'frames.npz')))
        won = sum(r['won'] for r in rows)
        result = dict(task=name,deck=tasks[name]['deck'],games=16,won=won,
                      loss_waves=dict(waves),nearest_terminal_zombie_types=dict(front),rosters=dict(rosters),
                      per_game=games,games_sha256=sha(folder/'games.jsonl'),
                      conclusion='老师可赢，存在学生泛化缺口' if won else '本老师16局仍全败：结构性弱卡组嫌疑，交主控决定；有限样本不是数学上的不可赢证明',
                      caveat='Nearest surviving zombie at terminal is descriptive, not an instrumented killer ID.')
        write(marker,result);report[name]=result
        record(f'3留出诊断：{name}卡组{tasks[name]["deck"]}，beta1老师{won}/16；输局波数{dict(waves)}。'+result['conclusion']+'。')
    target = ROOT/'MEMORY/docs/task27-dead-decks.json';write(target,report)
    if not (OUT/'dead_decks_complete.json').exists():
        write(OUT/'dead_decks_complete.json',dict(complete=True,results_sha256=sha(target)))
        record('3三套留出老师诊断齐：48局及终局波数/最近存活僵尸、逐局SHA已核验；不更换留出卡组，结论与限制见证据。',target)


def references(data):
    for name,model in MODELS.items():
        target = OUT/f'baseline_v2_{name}'
        recipe = materialize_recipe(ROOT/'recipes/panel_v2.json', OUT/f'recipes/reference_{name}_v2.json',
                                    version='v2', manifest=str(OUT/'unified_v2.json'),
                                    rollout_policy=str(model), out=str(target))
        command = [PYTHON, str(ROOT/'scripts/unified_panel.py'), '--recipe', str(recipe)]
        run_stage(command,'baseline_v2_'+name);verify(data)
        panel = read(target/'panel.json'); manifest = read(OUT/'unified_v2.json')
        assert panel['complete'] and len(panel['tasks']) == 34 and sum(t['games'] for t in panel['tasks']) == 3264
        assert panel['model_sha256'] == data['models_sha256'][name]
        expected_binary=data.get('historical_reference_executable_sha256',{}).get(name,data['executable_sha256'])
        assert panel['executable_sha256']==expected_binary
        for task in manifest['tasks']:
            group = manifest['groups'][task['task_id']];count = 64 if group in ('heldout','pool') else 128
            actual = summarize(target/task['task_id'],task,task['seeds'][:count],plant_types=True)
            actual['group'] = group
            assert actual == next(t for t in panel['tasks'] if t['task_id'] == task['task_id'])
        marker = OUT/f'baseline_v2_{name}_complete.json'
        if not marker.exists():
            write(marker,dict(complete=True,panel_sha256=sha(target/'panel.json')))
            record(f'2参考面板完成：{name}的v2共34项/3264局，逐任务原始seed、胜局及动态植物类型计数核对；各区域数字将在dw5i齐后同表汇报。')


def main():
    data = plan();verify(data);diagnose(data);references(data);verify(data)
    write(OUT/'pretraining_complete.json',dict(complete=True,plan_sha256=sha(OUT/'pretraining_plan.json')))


if __name__ == '__main__':
    main()
