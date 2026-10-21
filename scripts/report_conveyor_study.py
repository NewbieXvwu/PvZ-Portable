"""Re-read task20's actual games, archives and checkpoint before publishing conclusions."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from run_conveyor_study import OUT, ROOT, read, sha, verify, write


def games(folder, task, seeds, model, *, teacher=False, beta=None, workers=True):
    rows = [json.loads(line) for line in (folder/'games.jsonl').read_text().splitlines()]
    assert len(rows) == len(seeds) and Counter(r['seed'] for r in rows) == Counter(seeds), folder
    arm = 'fast_search' if teacher else 'fast'
    assert all(r['arm'] == arm for r in rows)
    config = read(folder/'config.json')
    assert config['task_id'] == task and config['rollout_policy'] == str(model)
    if workers:
        assert config['workers'] == '4'
    assert config['max_actions'] == '3000'
    if teacher:
        assert float(config['dagger_beta']) == beta
        plan = read(OUT/'study_plan.json')
        assert config['value'] == plan['models']['value'] and config['static_value'] == plan['models']['static']
        assert config['executable'] == str(ROOT/'build/pvz-portable')
        assert config['manifest'] == plan['manifest']
        expected = dict(samples='3', halving='True', verify_samples='9', student_candidates='2',
                        candidate_generator='static', per_card='1', horizon='2500', margin='0.02',
                        min_gap='300', rollout_sim_skip='61', rollout_backend='cpp', cpu_sparse_tracks='False')
        for name, value in expected.items():
            assert config[name] == value, (folder, name, config[name])
    else:
        assert all(r['searched'] == 0 for r in rows)
    won = sum(r['won'] for r in rows)
    assert read(folder/'summary.json')[arm]['won'] == won
    result = dict(games=len(rows), won=won, win_rate=won/len(rows), games_sha256=sha(folder/'games.jsonl'),
                  config_sha256=sha(folder/'config.json'), searched=sum(r['searched'] for r in rows))
    if teacher:
        result['dagger_beta'] = beta
    return result, {r['seed']: r for r in rows}


def paired(after, before):
    assert set(after) == set(before)
    gain = sum(after[s]['won'] and not before[s]['won'] for s in after)
    loss = sum(before[s]['won'] and not after[s]['won'] for s in after)
    return dict(delta_wins=gain-loss, delta_pp=100*(gain-loss)/len(after),
                pilot_only_wins=gain, reference_only_wins=loss)


def baselines(plan):
    proof = read(OUT/'baseline_complete.json')
    assert proof['plan_sha256'] == sha(OUT/'study_plan.json')
    output = {}; indexed = {}
    for name, folder, model, teacher in (
        ('v34', 'base_v34', 'v34', False),
        ('identity', 'base_identity_aligned', 'identity', False),
        ('teacher', 'base_teacher_executed', 'v34', True)):
        seeds = plan['teacher_seeds'] if teacher else plan['baseline_seeds']
        result, rows = games(OUT/folder, 'day_l10_orig_eval', seeds, plan['models'][model],
                             teacher=teacher, beta=1 if teacher else None)
        assert result['games_sha256'] == proof[name]['games_sha256']
        assert result['won'] == proof[name]['won']
        if name == 'v34':
            result['provenance'] = proof[name]['provenance']
        else:
            resource = read(OUT/('t20_'+folder+'.result.json'))
            assert resource['returncode'] == 0
            result['resources'] = resource
        output[name] = result; indexed[name] = rows
    # Compare the teacher to the same 64 student seeds, rather than unequal denominators.
    output['teacher']['student_v34_same_64_won'] = sum(indexed['v34'][s]['won'] for s in plan['teacher_seeds'])
    return output, indexed


def collection(plan):
    from pvz_frame_store import read_frames
    from pvz_rollout_policy import packet_features
    proof = read(OUT/'collection_complete.json')
    assert proof['archives'] == 256 and proof['plan_sha256'] == sha(OUT/'study_plan.json')
    rounds = []; archive_records = []; frames_total = differences = labels = free_packets = 0
    mirror = OUT/'frames_v2'
    assert not mirror.is_symlink() and mirror.resolve().is_relative_to(ROOT)
    for index, (seeds, label) in enumerate(zip(plan['collection_seeds'], plan['collection_labels'])):
        folder = OUT/f'collect_round{index+1}'
        result, rows = games(folder, 'day_l10_orig_train', seeds, plan['models']['identity'], teacher=True, beta=0)
        assert result['games_sha256'] == proof['rounds'][index]['games_sha256']
        result['resources'] = read(OUT/f't20_collect_round{index+1}.result.json')
        assert result['resources']['returncode'] == 0
        rounds.append(result)
        for seed in seeds:
            source = OUT/f'frames/seed_{seed}/{label}_fast_search'
            target = mirror/f'seed_{seed}/{label}_fast_search'
            assert target.resolve().is_relative_to(ROOT)
            meta = read(source/'meta.json'); task = meta['task']
            assert task['seed'] == seed and task['level'] == 10 and task['deck'] == []
            assert task['conveyor_start'] == 'original' and task['task_id'] == 'day_l10_orig_train'
            assert task['checkpoint_sha256'] == plan['model_sha256']['identity']
            assert task['checkpoint'] == plan['models']['identity']
            assert meta['outcome']['won'] == rows[seed]['won']
            for name in ('meta.json', 'frames.npz'):
                assert sha(source/name) == sha(target/name)
            frames = read_frames(source/'frames.npz')
            assert len(frames) == rows[seed]['decisions']+1
            assert len(frames[0]['packets']) == 10 and all(p[1] < 0 for p in frames[0]['packets'])
            assert all(f.get('conveyor_start') == 'original' for f in frames)
            frames_total += len(frames)
            for frame in frames[1:]:
                assert 'label' in frame and 'action' in frame
                labels += 1; differences += frame['label'] != frame['action']
            # Verify every archived free-card feature against the engine's actual-mode flag.
            for frame in frames:
                costly = [p for p in frame['packets'] if p[1] >= 0 and p[5] > frame['sun']]
                if costly:
                    features = packet_features(frame)
                    assert all(float(features[p[0], 3]) == 1 for p in costly)
                    free_packets += len(costly)
            archive_records.append(dict(seed=seed, label=label, meta_sha256=sha(source/'meta.json'),
                                        frames_sha256=sha(source/'frames.npz'), frames=len(frames)))
    assert len(archive_records) == 256 and labels > 0 and free_packets > 0
    # Keep the per-archive integrity ledger local; publish its hash and aggregate counts.
    ledger = OUT/'collection_audit.json'
    write(ledger, archive_records)
    return dict(rounds=rounds, archives=256, frames=frames_total, labelled_decisions=labels,
                teacher_student_action_differences=differences, free_packet_observations=free_packets,
                ledger_sha256=sha(ledger))


def training(plan):
    import math
    import torch
    path = OUT/'identity_dagger_s1/model.pt'; proof = read(OUT/'training_complete.json')
    assert sha(path) == proof['model_sha256'] and proof['plan_sha256'] == sha(OUT/'study_plan.json')
    command = read(OUT/'t20_training.command.json')['command']
    assert command == plan['training_command'] and '--init' not in command and '--max-steps' not in command
    model = torch.load(path, map_location='cpu', weights_only=True)
    reference = torch.load(plan['models']['identity'], map_location='cpu', weights_only=True)
    inventory = read(ROOT/'MEMORY/docs/task20-reference.json')
    from pvz_frame_store import frame_paths
    counts = Counter(p.parent.name for p in frame_paths(OUT/'frames_v2') if p.parent.name in plan['labels'])
    assert all(counts[label] == count for label, count in inventory['inherited_archive_counts'].items())
    assert all(counts[label+'_fast_search'] == 128 for label in plan['collection_labels'])
    assert sum(counts.values()) == 38271+256
    assert Counter(model['labels'].split(',')) == Counter(plan['labels'])
    assert len(plan['labels']) == 84 and model['packet_head'] == 'identity' and model['row_pool']
    recipe = ('width', 'pool', 'label_weight', 'packet_balance', 'row_pool', 'aux_weight', 'aux_horizon',
              'packet_head', 'plant_embedding_dim', 'context', 'obs', 'soft_weight', 'spread_weight',
              'spread_cap', 'lr_schedule', 'ema', 'cell_soft_weight')
    assert all(model[k] == reference[k] for k in recipe)
    assert all(torch.isfinite(v).all() for v in model['state_dict'].values())
    metrics = read(OUT/'identity_dagger_s1/metrics.json')
    assert model['history'] == metrics['history'] and [r['epoch'] for r in model['history']] == list(range(1, 13))
    assert all(math.isfinite(float(row[k])) for row in model['history'] for k in ('train_loss', 'val_loss'))
    assert metrics['data_memory']['rows'] > 11051782
    assert proof['resources']['returncode'] == 0
    return dict(model_sha256=sha(path), seed=1, epochs=12, labels=84, archives=sum(counts.values()), recipe_matches_reference=list(recipe),
                data_memory=metrics['data_memory'], final_epoch=model['history'][-1], resources=proof['resources'])


def report(final):
    plan = read(OUT/'study_plan.json'); verify(plan)
    assert read(ROOT/'MEMORY/docs/task20-study-plan.json') == plan
    baseline, indexed = baselines(plan)
    result = dict(completed_at=datetime.now(timezone.utc).isoformat(), complete=False,
                  study_plan_sha256=sha(OUT/'study_plan.json'), executable_sha256=plan['executable_sha256'],
                  original_models_sha256=plan['model_sha256'], baseline=baseline,
                  preflight=read(OUT/'preflight_complete.json'))
    assert result['preflight']['plan_sha256'] == result['study_plan_sha256']
    if final:
        result['collection'] = collection(plan)
        result['training'] = training(plan)
        reference = read(ROOT/'MEMORY/docs/task20-reference.json')
        assert reference['study_plan_sha256'] == result['study_plan_sha256']
        assert reference['model_sha256'] == plan['model_sha256']['identity']
        proof = read(OUT/'evaluation_complete.json')
        assert proof['complete'] and proof['games'] == 1024
        assert proof['model_sha256'] == result['training']['model_sha256']
        evaluation = []
        for task in plan['evaluation_tasks']:
            folder = OUT/('pilot_'+task)
            record, rows = games(folder, task, plan['baseline_seeds'], OUT/'identity_dagger_s1/model.pt')
            assert record['games_sha256'] == proof['tasks'][task]['games_sha256']
            if task == 'day_l10_orig_eval':
                before = indexed['identity']
            else:
                old = next(r for r in reference['tasks'] if r['task_id'] == task)
                old_folder = Path(old['folder'])
                assert sha(old_folder/'games.jsonl') == old['games_sha256']
                assert sha(old_folder/'config.json') == old['config_sha256']
                old_result, before = games(old_folder, task, plan['baseline_seeds'], plan['models']['identity'])
                assert old_result['won'] == old['won']
            record.update(task_id=task, reference_won=sum(r['won'] for r in before.values()), **paired(rows, before))
            record['resources'] = read(OUT/('t20_pilot_'+task+'.result.json'))
            assert record['resources']['returncode'] == 0
            evaluation.append(record)
        compatibility = read(OUT/'legacy_contract.json')
        assert compatibility['games'] == 8 and compatibility['identical_except_seconds']
        assert compatibility['executable_sha256'] == plan['executable_sha256']
        tests = OUT/'review_fix_tests.txt'
        assert 'Ran 31 tests' in tests.read_text() and tests.read_text().rstrip().endswith('OK')
        result.update(complete=True, evaluation=evaluation, evaluated_games=1024,
                      engineering=dict(legacy_contract=compatibility, targeted_tests=31, test_log_sha256=sha(tests)),
                      limitations=['One fresh training seed: descriptive pilot, no population-level noninferiority claim.',
                                   'More data also changes the seed-based validation split and optimizer-step count.',
                                   'v34 baseline inherited from cb; historical executable hash unavailable.',
                                   'No production/default promotion; previous L10 variant is retired from new studies.'])
    verify(plan)
    target = ROOT/'MEMORY/docs'/('task20-results.json' if final else 'task20-baseline.json')
    write(target, result)
    print(json.dumps(dict(path=str(target), complete=result['complete'], baseline={k:v['won'] for k,v in baseline.items()})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    report(args.final)
