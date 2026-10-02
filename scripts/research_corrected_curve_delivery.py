"""Validate and export all completed corrected evaluations without changing originals."""
from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import gzip
import hashlib
import json
from pathlib import Path

from research_comparison_summary import outcomes, summarize_evaluation
from research_packet_cost_reevaluation import validate_curve

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data := stream.read(1 << 20):
            result.update(data)
    return result.hexdigest()


def save_fresh(path, value):
    path = Path(path)
    if path.exists():
        raise ValueError(f'fresh derived evidence required: {path}')
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def ordinary_cohorts(raw, tasks, modes, conveyor_ids):
    result = {}
    for mode in modes:
        groups = defaultdict(list)
        for task in tasks:
            if task['evaluation_role'] == 'training_probe' or task['task_id'] in conveyor_ids:
                continue
            rows = raw['seed_results'][mode][task['task_id']]
            groups[f"validation/cap{task['wave_cap']}"].extend(rows)
            groups[f"validation/terrain/{task['terrain']}"].extend(rows)
        result[mode] = {key: outcomes(rows) for key, rows in groups.items()}
    return result


def baseline_pairs(baselines, tasks, modes):
    pairs = []
    for seed in (0, 1, 2):
        reference = baselines[seed]['R0']
        for reward in ('R1', 'R2', 'R3'):
            changes = wins = 0
            examples = []
            for mode in modes:
                for task in tasks:
                    key = task['task_id']
                    before = {r['seed']: r for r in reference[mode][key]}
                    after = {r['seed']: r for r in baselines[seed][reward][mode][key]}
                    if set(before) != set(after) or set(before) != set(task['seeds']):
                        raise ValueError('paired baseline seed inventory differs')
                    for env_seed in task['seeds']:
                        if before[env_seed] != after[env_seed]:
                            changes += 1
                            wins += before[env_seed]['won'] != after[env_seed]['won']
                            if len(examples) < 8:
                                examples.append({'mode': mode, 'task_id': key, 'seed': env_seed})
            pairs.append({'initialization_seed': seed, 'reward_pair': ['R0', reward],
                          'rows_compared': len(tasks)*64*len(modes), 'full_row_differences': changes,
                          'win_label_differences': wins, 'examples': examples})
    return pairs


def validation_wins(raw, tasks, modes, conveyor_ids):
    """Keep task/seed identity for paired changes between frozen nodes."""
    result = {}
    for mode in modes:
        for cap in (1, 3, 5):
            for scope in ('all_original_tasks', 'ordinary_tasks'):
                rows = {(task['task_id'], row['seed']): bool(row['won'])
                        for task in tasks
                        if task['evaluation_role'] != 'training_probe' and task['wave_cap'] == cap
                        and (scope == 'all_original_tasks' or task['task_id'] not in conveyor_ids)
                        for row in raw['seed_results'][mode][task['task_id']]}
                if not rows:
                    raise ValueError('empty paired validation cohort')
                result[(mode, cap, scope)] = rows
    return result


def paired_transitions(previous, current):
    result = []
    if set(previous) != set(current):
        raise ValueError('paired validation cohort inventory changed')
    for (mode, cap, scope), before in previous.items():
        after = current[(mode, cap, scope)]
        if set(before) != set(after):
            raise ValueError('paired validation task/seed inventory changed')
        result.append({'mode': mode, 'wave_cap': cap, 'scope': scope, 'count': len(before),
                       'before_wins': sum(before.values()), 'after_wins': sum(after.values()),
                       'retained_wins': sum(before[k] and after[k] for k in before),
                       'lost_wins': sum(before[k] and not after[k] for k in before),
                       'gained_wins': sum(not before[k] and after[k] for k in before)})
    return result


def descriptive_progress(curves, modes):
    """Describe repeated above-baseline wins; do not invent an acceptance test."""
    result = []
    for curve in curves:
        for mode in modes:
            for scope in ('all_original_tasks', 'ordinary_tasks'):
                source = 'summary' if scope == 'all_original_tasks' else 'ordinary_cohorts'
                points = []
                for point in curve['points']:
                    cohorts = point[source][mode]
                    if source == 'summary':
                        cohorts = cohorts['cohorts']
                    points.append({'node_index': point['node_index'], 'decisions': point['counters']['decisions'],
                                   'actual_original_wall_seconds': point['original_training_wall_seconds'],
                                   **{f'cap{cap}': cohorts[f'validation/cap{cap}'] for cap in (1, 3, 5)}})
                repeated = [p['node_index'] for p in points[1:]
                            if all(p[f'cap{cap}']['won'] > points[0][f'cap{cap}']['won'] for cap in (3, 5))]
                result.append({'experiment_id': curve['experiment_id'], 'initialization_seed': curve['initialization_seed'],
                               'reward': curve['reward']['name'], 'mode': mode, 'scope': scope, 'points': points,
                               'nodes_with_both_cap3_and_cap5_above_own_baseline': repeated,
                               'multiple_frozen_nodes_above_baseline': len(repeated) >= 2})
    return {'interpretation': 'Descriptive paired fixed-node results, without a new statistical or learning acceptance threshold. Training practiced cap1 only; cap3/cap5 are validation transfer. All original tasks retained alongside ordinary-only diagnostics.',
            'equal_wall_comparison_performed': False, 'B_frozen': False, 'series': result}


def validate_and_prepare(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    state_path, inventory_path = output/'state.json', output/'inventory.json'
    state = json.loads(state_path.read_text())
    report = json.loads((output/'report.json').read_text())
    inventory = json.loads(inventory_path.read_text())
    if (state['status'] != 'complete' or len(state['results']) != 60
            or len(inventory['nodes']) != 60 or report != state):
        raise ValueError('all 60 terminal corrected nodes required before exporting curves')
    if state['protocol_sha256'] != digest(protocol_path) or state['inventory_sha256'] != digest(inventory_path):
        raise ValueError('protocol/inventory identity changed')
    for name, expected in protocol['required_fingerprints'].items():
        path = Path(name) if Path(name).is_absolute() else ROOT/name
        if digest(path) != expected:
            raise ValueError(f'preregistered source/evidence changed: {name}')
    for name, expected in inventory['original_states'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('original training state changed')
    tasks = json.loads((ROOT/protocol['evaluation_manifest']).read_text())['tasks']
    if len(tasks) != 35 or any(len(t['seeds']) != 64 for t in tasks):
        raise ValueError('complete original 35-task/64-seed evaluation inventory required')
    modes = protocol['modes']
    grouped, baselines, previous_wins, total_jobs = defaultdict(list), defaultdict(dict), {}, 0
    for expected, record in zip(inventory['nodes'], state['results'], strict=True):
        for key in ('experiment_id', 'node_index', 'initialization_seed', 'reward', 'counters', 'updates', 'checkpoint_sha256'):
            if record[key] != expected[key]:
                raise ValueError(f'node identity/order changed: {key}')
        if record['jobs'] != 4480 or record['changes']['unexpected_changed_rows']:
            raise ValueError('incomplete node or true non-conveyor drift')
        if (digest(ROOT/expected['checkpoint']) != expected['checkpoint_sha256']
                or digest(ROOT/expected['original_raw']) != expected['original_raw_sha256']
                or digest(ROOT/record['raw_path']) != record['raw_sha256']):
            raise ValueError('original/corrected raw or checkpoint changed')
        with gzip.open(ROOT/record['raw_path'], 'rt') as stream:
            raw = json.load(stream)
        if (raw['experiment_id'] != record['experiment_id'] or raw['counters'] != record['counters']
                or raw['checkpoint_sha256'] != record['checkpoint_sha256']
                or raw['model_state_sha256'] != record['model_state_sha256']):
            raise ValueError('raw node metadata differs')
        if summarize_evaluation(raw, tasks, modes) != record['summary']:
            raise ValueError('derived counts, Wilson or terminal summaries disagree with raw rows')
        if raw['simulator_sha256'] != digest(protocol['executable']):
            raise ValueError('raw simulator differs from guarded native')
        point = {key: record[key] for key in ('node_index', 'counters', 'updates', 'checkpoint_sha256',
                 'model_state_sha256', 'original_training_wall_seconds', 'reevaluation_seconds',
                 'resources', 'raw_path', 'raw_sha256', 'changes', 'summary')}
        point['checkpoint_path'] = expected['checkpoint']
        point['ordinary_cohorts'] = ordinary_cohorts(raw, tasks, modes, protocol['allowed_changed_task_ids'])
        current_wins = validation_wins(raw, tasks, modes, protocol['allowed_changed_task_ids'])
        if record['experiment_id'] in previous_wins:
            point['paired_changes_from_previous_node'] = paired_transitions(previous_wins[record['experiment_id']], current_wins)
        previous_wins[record['experiment_id']] = current_wins
        grouped[record['experiment_id']].append(point)
        if record['node_index'] == 0:
            baselines[record['initialization_seed']][record['reward']] = raw['seed_results']
            if record['model_state_sha256'] != inventory['paired_initial_states'][str(record['initialization_seed'])]:
                raise ValueError('actual untrained parameters differ from original paired initialization')
        total_jobs += record['jobs']
    queue = json.loads((ROOT/protocol['queue']).read_text())
    curves = []
    for entry in queue['order']:
        config = json.loads((ROOT/entry['config']).read_text())
        original_path = ROOT/entry['output_dir']/'training_state.json'
        original = json.loads(original_path.read_text())
        points = grouped[config['experiment_id']]
        old_points = validate_curve(original, config)
        if (len(points) != 5 or [p['node_index'] for p in points] != list(range(5))
                or any(p['counters'] != old['counters'] or p['updates'] != old['updates']
                       for p, old in zip(points, old_points, strict=True))):
            raise ValueError('corrected curve does not preserve each original node')
        curves.append({'schema_version': 1, 'scope': 'readonly guarded-native reevaluation; original training preserved',
                       'experiment_id': config['experiment_id'], 'reward': config['reward'],
                       'initialization_seed': config['initialization_seed'],
                       'initial_state_sha256': original['initial_state_sha256'],
                       'original_state_path': str(original_path.relative_to(ROOT)),
                       'original_state_sha256': digest(original_path), 'config_path': entry['config'],
                       'config_sha256': digest(ROOT/entry['config']), 'model_config': config['model'],
                       'original_budget': config['budget'], 'original_training_counters': original['counters'],
                       'original_training_wall_seconds': original['wall_seconds'],
                       'reevaluation_protocol_sha256': digest(protocol_path), 'modes': modes,
                       'tasks': len(tasks), 'environment_seeds_per_task': 64,
                       'status': 'corrected_evaluation_complete', 'points': points,
                       'interpretation': 'Original and corrected rows remain available. No new training, learning gate, reward winner or B set by export. Reevaluation wall includes cached resumption only as actually recorded.'})
    if len(curves) != 12 or total_jobs != 268800:
        raise ValueError('complete four-reward/three-initialization matrix required')
    return curves, {'schema_version': 1, 'scope': 'complete corrected curve delivery verification',
                    'status': 'validated_locally', 'curves': 12, 'nodes': 60, 'jobs': total_jobs,
                    'protocol_path': str(protocol_path.relative_to(ROOT)), 'protocol_sha256': digest(protocol_path),
                    'state_sha256': digest(state_path), 'inventory_sha256': digest(inventory_path),
                    'raw_nodes_verified': 60, 'checkpoint_files_verified': 60, 'original_states_verified': 12,
                    'all_seed_inventories_and_Wilson_verified': True,
                    'non_conveyor_changed_rows': sum(r['changes']['unexpected_changed_rows'] for r in state['results']),
                    'conveyor_changed_rows': sum(r['changes']['conveyor_changed_rows'] for r in state['results']),
                    'win_label_changes': sum(r['changes']['win_label_changes'] for r in state['results']),
                    'paired_actual_baselines': baseline_pairs(baselines, tasks, modes),
                    'descriptive_learning_progress': descriptive_progress(curves, modes),
                    'original_states_unchanged': True, 'raw_delivery_channel': 'HF; not uploaded yet',
                    'git_delivery_complete': False, 'T5_E_passed': False, 'B_frozen': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--reevaluation-dir', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    summary = args.summary.resolve()
    if summary.parent != ROOT/'artifacts/t5/perf':
        raise ValueError('derived conclusion must use the Git-whitelisted perf directory')
    with (args.reevaluation_dir/'.execution.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        curves, audit = validate_and_prepare(args.protocol.resolve(), args.reevaluation_dir.resolve())
        paths = [ROOT/'artifacts/t5/curves'/f"{c['experiment_id']}_packet_cost_v2.json" for c in curves]
        if summary.exists() or any(path.exists() for path in paths):
            raise ValueError('preserve earlier derived evidence; fresh curve/audit paths required')
        audit['curve_files'] = []
        for path, curve in zip(paths, curves, strict=True):
            save_fresh(path, curve)
            audit['curve_files'].append({'path': str(path.relative_to(ROOT)), 'sha256': digest(path), 'bytes': path.stat().st_size})
        audit['publisher_source_sha256'] = digest(Path(__file__))
        save_fresh(summary, audit)
        print(f"validated and saved {len(curves)} corrected curves, 60 nodes, {audit['jobs']} raw jobs", flush=True)


if __name__ == '__main__':
    main()
