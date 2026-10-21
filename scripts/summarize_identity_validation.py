"""Audit the completed task19 pilot and write its small, reproducible results record."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/task19'
SOURCE = Path.home() / 'PvZ-search'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def games(folder, task, seeds):
    rows = [json.loads(line) for line in (folder / 'games.jsonl').read_text().splitlines()]
    assert len(rows) == len(seeds) and {r['seed'] for r in rows} == set(seeds), task
    assert all(r['arm'] == 'fast' and r['searched'] == 0 for r in rows), task
    config = read(folder / 'config.json')
    assert config['task_id'] == task and config['arms'] == 'fast', task
    wins = sum(bool(r['won']) for r in rows)
    assert read(folder / 'summary.json')['fast']['won'] == wins, task
    return {r['seed']: bool(r['won']) for r in rows}, config


def summarize():
    # Check evidence rather than treating an existing completion marker as sufficient.
    import numpy as np
    import torch
    torch.set_num_threads(1)
    plan_path = OUT / 'validation_plan.json'
    plan = read(plan_path)
    assert plan_path.read_bytes() == (ROOT / 'MEMORY/docs/task19-validation-amended-plan.json').read_bytes()
    expected_tasks = [f'day_l{level}_eval{suffix}' for suffix in ('_d7', '') for level in (7, 8, 9, 10)]
    assert plan['task_ids'] == expected_tasks and plan['evaluation_seeds'] == list(range(30000, 30256))
    assert plan['training_seed'] == 1 and plan['workers'] == 4
    train = read(OUT / 'identity_training_complete.json')
    evaluation = read(OUT / 'identity_evaluation_complete.json')
    frozen = read(OUT / 'evaluation_freeze.json')
    assert evaluation['complete'] and set(evaluation['tasks']) == set(expected_tasks)
    assert train['plan_sha256'] == sha(plan_path)
    model_path = OUT / 'identity_dw3_s1/model.pt'
    assert sha(model_path) == train['model_sha256'] == evaluation['model_sha256'] == frozen['model_sha256']
    assert sha(ROOT / 'build/pvz-portable') == evaluation['executable_sha256'] == frozen['executable_sha256']
    assert sha(plan['manifest']) == plan['manifest_sha256'] == frozen['manifest_sha256']
    assert sha(OUT / 'evaluation_freeze.json') == evaluation['evaluation_freeze_sha256']
    for name, digest in {**plan['training_sources'], **frozen['sources']}.items():
        assert sha(ROOT / name) == digest, name
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    baseline_path = SOURCE / 'artifacts/student/dw3s1/model.pt'
    baseline = torch.load(baseline_path, map_location='cpu', weights_only=True)
    reference_path = ROOT / 'MEMORY/docs/task19-dw3s1-reference.json'
    reference_proof = read(reference_path)
    assert sha(baseline_path) == reference_proof['checkpoint_sha256_at_audit']
    assert checkpoint['packet_head'] == 'identity' and checkpoint['row_pool']
    assert checkpoint['width'] == checkpoint['plant_embedding_dim'] == 64
    assert len(checkpoint['history']) == 12 and 'max_steps' not in checkpoint
    assert set(checkpoint['labels'].split(',')) == set(plan['labels']) == set(baseline['labels'].split(','))
    assert all(torch.isfinite(v).all() for v in checkpoint['state_dict'].values())
    for item in checkpoint['history']:
        assert all(not isinstance(v, float) or math.isfinite(v) for v in item.values())
    recipe_keys = ['width', 'label_weight', 'pool', 'packet_balance', 'row_pool', 'aux_weight',
                   'context', 'obs', 'soft_weight', 'spread_weight', 'spread_cap',
                   'lr_schedule', 'ema', 'cell_soft_weight']
    assert all(checkpoint[k] == baseline[k] for k in recipe_keys)
    rows_line = next(line for line in (OUT / 'identity_training.txt').read_text().splitlines()
                     if line.startswith('{"data_memory":'))
    data_memory = json.loads(rows_line)['data_memory']
    assert data_memory['rows'] == 11051782
    results = []
    for task in expected_tasks:
        folder = OUT / 'eval' / task
        actual, config = games(folder, task, plan['evaluation_seeds'])
        assert config['rollout_policy'] == str(model_path) and config['workers'] == '4'
        assert config['manifest'] == plan['manifest']
        assert config['executable'] == str(ROOT / 'build/pvz-portable')
        record = evaluation['tasks'][task]
        assert record['games_sha256'] == sha(folder / 'games.jsonl')
        assert record['games'] == len(actual) == 256 and record['won'] == sum(actual.values())
        resource = read(OUT / ('eval_' + task + '.result.json'))
        assert resource['returncode'] == 0
        row = dict(task_id=task, games=256, won=record['won'], win_rate=record['won']/256,
                   games_sha256=record['games_sha256'], resources=resource)
        if task.endswith('_d7'):
            level = int(task.split('_')[1][1:])
            base_folder = SOURCE / f'artifacts/day_panel/dw3s1_l{level}_d7_256'
            reference, base_config = games(base_folder, task, plan['evaluation_seeds'])
            assert base_config['rollout_policy'] == str(baseline_path)
            assert sum(reference.values()) == plan['reference_dw3s1_d7_wins'][level-7]
            assert sha(base_folder / 'games.jsonl') == reference_proof['tasks'][task]['games_sha256']
            assert sha(base_folder / 'config.json') == reference_proof['tasks'][task]['config_sha256']
            row['reference_dw3s1'] = dict(won=sum(reference.values()),
                delta_wins=sum(actual.values())-sum(reference.values()),
                delta_pp=100*(sum(actual.values())-sum(reference.values()))/256,
                identity_only_wins=sum(actual[s] and not reference[s] for s in actual),
                baseline_only_wins=sum(reference[s] and not actual[s] for s in actual),
                games_sha256=sha(base_folder / 'games.jsonl'), workers=base_config['workers'],
                executable=base_config['executable'])
        results.append(row)
    assert train['result']['returncode'] == 0
    smoke_path = ROOT / 'MEMORY/docs/task19-perturbation-smoke.json'
    smoke = read(smoke_path)
    assert smoke['complete'] and len(smoke['games']) == 8 and smoke['training_loader']['rows'] == 6192
    smoke_archives = []
    for game in smoke['games']:
        archive = Path(game['archive'])
        meta = read(archive / 'meta.json')
        actual_task = meta['task']
        assert actual_task['deck'] == game['actual_deck']
        assert actual_task['seed'] == game['seed'] and actual_task['level'] == game['level']
        assert actual_task['checkpoint_sha256'] == smoke['model_sha256']
        assert meta['outcome']['won'] == game['won']
        with np.load(archive / 'frames.npz', allow_pickle=False) as stored:
            frames = [json.loads(line) for line in stored['frames'].tobytes().splitlines()]
        assert len(frames) == game['frames']
        assert [p[1] for p in frames[0]['packets']] == actual_task['deck']
        changed = sum([p[1] for p in f['packets']] != actual_task['deck'] for f in frames)
        assert changed == game['frames_with_changed_seed_bank']
        smoke_archives.append(dict(level=game['level'], seed=game['seed'], frames=len(frames),
            meta_sha256=sha(archive / 'meta.json'), frames_sha256=sha(archive / 'frames.npz')))
    assert sum(g['frames'] for g in smoke_archives) == 6200
    manifest = read(ROOT / 'artifacts/stage4/day_panel_identity.json')
    assert sha(ROOT / 'artifacts/stage4/day_panel_identity.json') == smoke['manifest_sha256']
    engineering_path = ROOT / 'MEMORY/docs/task19-engineering-validation.json'
    engineering = read(engineering_path)
    for name, digest in engineering['source_sha256'].items():
        assert sha(ROOT / name) == digest, name
    assert engineering['identity_unique_tests'] == 34
    fix = engineering['numeric_fix']
    assert fix['after']['source_sha256'] == sha(ROOT / 'src/EnvironmentRollout.cpp')
    assert fix['after']['executable_sha256'] == frozen['executable_sha256']
    assert fix['slot_bit_exact']['bit_mismatches'] == 0 and fix['slot_bit_exact']['decisions'] == 72
    assert fix['evaluation']['games'] == 2048 and fix['evaluation']['identical_outcomes']
    assert not fix['trained_parity']['argmax_mismatches'] and not fix['trained_parity']['action_mismatches']
    assert engineering['legacy_slot']['configurations'] == 24
    assert engineering['perturbation']['training_seeds'] == 32768
    parity_path = ROOT / 'MEMORY/docs/task19-trained-parity.json'
    parity = read(parity_path)
    assert parity['model_sha256'] == sha(model_path) and parity['executable_sha256'] == frozen['executable_sha256']
    assert parity['checker_sha256'] == sha(ROOT / 'scripts/check_identity_checkpoint.py')
    assert parity['exported_sha256'] == sha(model_path.with_suffix('.pvzrn'))
    assert parity['format'] == 'PVZRN006' and parity['header_fields'] == [64, 1, 0, 3, 1, 1, 64]
    assert parity['all_argmax_and_actions_equal'] and parity['passed']
    assert parity['tolerance_absolute'] == 1e-5 and parity['tolerance_relative'] == 1e-5
    assert parity['worst_tolerance_ratio'] < 1 and parity['argmax_mismatches'] == 0
    assert parity['real_decisions'] == 128
    return dict(completed_at=datetime.now(timezone.utc).isoformat(), complete=True,
        scope='one corrected seed1 dw3+identity+row_pool model; all eight predeclared 256-game groups',
        model_sha256=sha(model_path), baseline_model_sha256=sha(baseline_path),
        baseline_reference_sha256=sha(reference_path),
        plan_sha256=sha(plan_path), evaluation_freeze_sha256=sha(OUT / 'evaluation_freeze.json'),
        training=dict(epochs=12, labels=len(plan['labels']), archives=sum(plan['archive_counts'].values()),
                      data_memory=data_memory, final_epoch=checkpoint['history'][-1], resources=train['result'],
                      recipe_matches_dw3s1=recipe_keys),
        evaluation=dict(games=sum(r['games'] for r in results), tasks=results),
        perturbation=dict(generation_audit=manifest['generation_audit'], smoke_sha256=sha(smoke_path),
                          smoke_games=8, loaded_rows=6192, heldouts='initial unordered card sets',
                          smoke_archives=smoke_archives,
                          live_replay_scope='one archived eight-action prefix, plus archived-deck replay unit tests'),
        engineering=dict(validation_sha256=sha(engineering_path),
                         unique_identity_tests=engineering['identity_unique_tests'],
                         slot_exact_configurations=24, trained_parity_sha256=sha(parity_path),
                         trained_cpu_max_abs_error=parity['maximum_abs_error'],
                         trained_cpu_tolerance=[parity['tolerance_absolute'], parity['tolerance_relative']],
                         trained_cpu_worst_tolerance_ratio=parity['worst_tolerance_ratio'],
                         trained_cpu_reference_noise=parity['per_head_reference_noise'],
                         trained_cpu_absolute_only_exceedances=parity['absolute_only_exceedances'],
                         trained_cpu_real_decisions=parity['real_decisions'],
                         numeric_fix=engineering['numeric_fix']),
        limitations=[
            'Single training seed; descriptive comparison, no population-level noninferiority claim.',
            'Identity also repairs the empty-slot public wave scalar and balances packets by plant identity.',
            'L10 keeps RESET_V2 initial stock plus native conveyor shifts, refills and free cards.',
            'The eight perturbation smoke archives are not added to this existing-data training pilot.',
            'Baseline used six workers and a separate native build; counts are not a throughput comparison.',
            'Full interactive whole-deck warning checks and optional trunk embedding sharing are omitted.',
            'No default or production model promotion; CUDA explicitly rejects PVZRN006.'
        ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'MEMORY/docs/task19-identity-results.json')
    args = parser.parse_args()
    result = summarize()
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(complete=True, games=result['evaluation']['games'],
                          tasks=[(r['task_id'], r['won']) for r in result['evaluation']['tasks']])), flush=True)
