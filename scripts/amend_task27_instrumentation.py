"""Record the requested task0 instrumentation revision without rerunning finished games."""
from task27_common import ROOT, OUT, read, write, sha, record


def main():
    assert not (OUT/'study_plan.json').exists(), 'Never amend a training freeze in place'
    initial_path=OUT/'step0/pretraining_plan.initial.json';initial=read(initial_path)
    assert read(OUT/'step0/type_count_check/complete.json')['complete']
    assert read(OUT/'step0/type_count_check/complete.json')['executable_sha256']==sha(ROOT/'build/pvz-portable')
    assert 'Ran 12 tests' in (OUT/'step0/unit_tests.txt').read_text()
    assert (OUT/'step0/unit_tests.txt').read_text().rstrip().endswith('OK')
    panel_path=OUT/'baseline_v2_dw4i_s1/panel.json';panel=read(panel_path)
    assert panel['complete'] and len(panel['tasks'])==34 and sum(t['games'] for t in panel['tasks'])==3264
    assert panel['executable_sha256']==initial['executable_sha256']==sha(OUT/'step0/previous-pvz-portable')
    evidence=[panel_path,OUT/'baseline_v2_dw4i_s1.result.json',OUT/'step0/reference_recovery.json']
    for task in panel['tasks']:
        name=task['task_id'];folder=OUT/'baseline_v2_dw4i_s1'/name
        assert sha(folder/'games.jsonl')==task['games_sha256']
        evidence += [folder/'games.jsonl',folder/'config.json',
                     OUT/'baseline_v2_dw4i_s1/resources'/(name+'.contract.json'),
                     OUT/'baseline_v2_dw4i_s1/resources'/(name+'.result.json')]
    for task in initial['dead_tasks']:
        evidence += [OUT/'dead_decks'/task/'games.jsonl',OUT/'dead_decks'/task/'diagnosis.json',
                     OUT/('dead_'+task+'.result.json')]
    sources=set(initial['sources']) | {'python/pvz_episode_stats.py','scripts/dw4_night_ab_report.py',
            'scripts/sim_skip_agreement.py','scripts/cuda_root_corpus.py','scripts/cuda_leaf_compare.py',
            'scripts/check_panel_type_counts.py','scripts/amend_task27_instrumentation.py'}
    updated={**initial,'sources':{p:sha(ROOT/p) for p in sorted(sources)},
             'executable_sha256':sha(ROOT/'build/pvz-portable'),
             'prior_pretraining_plan_sha256':sha(initial_path),
             'retained_evidence':{str(p.relative_to(OUT)):sha(p) for p in evidence},
             'historical_reference_executable_sha256':{'dw4i_s1':initial['executable_sha256']},
             'instrumentation_revision':'User-requested task0: canonical type/slot keys, shared legacy reader, same_type fix. No policy or deck changes.',
             'step0_smoke_sha256':sha(OUT/'step0/type_count_check/complete.json')}
    write(OUT/'pretraining_plan.json',updated)
    from run_task27_pretraining import verify
    verify(updated)
    target=ROOT/'MEMORY/docs/task27-pretraining-plan.json';write(target,updated)
    proof=ROOT/'MEMORY/docs/task27-step0.json'
    write(proof,dict(complete=True,unit_tests=12,smoke=read(OUT/'step0/type_count_check/complete.json'),
                     prior_plan_sha256=sha(initial_path),amended_plan_sha256=sha(OUT/'pretraining_plan.json'),
                     retained_reference_games=3264,retained_reference_executable_sha256=initial['executable_sha256'],
                     unavailable_consumer='analyze_l20_conveyor.py is absent in dw5 and main; no unrelated worktree imported.',
                     note='Old raw records are immutable; exact executable/source lineage retained. The requested protocol change is isolated from policy behavior.'))
    record('0统计协议实测完成：新C++/Python只写planted_types与planted_slots，执行前取真实类型，same_type比较已修；共用读取器及夜间/统一面板/模拟跳步/CUDA语料与比较读者已接通，旧传送带无法恢复时标未知。12项单测通过；夜间/原版L10/泳池各一组完整C++与Python对局，逐动作类型计数、新旧构建结果完全一致。已完成的dw4i参考3264局保持原始记录和旧构建SHA，不重跑；新计划明确保留阶段来源，随后恢复旧identity评估，训练仍等q103第2轮。',target,proof)


if __name__=='__main__':main()
