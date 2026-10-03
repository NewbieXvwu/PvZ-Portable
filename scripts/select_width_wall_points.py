"""Retain actual width policies at the frozen wall caps without model inference.

Only paused candidates that reached the common interaction budget are inspected.
Selection uses checkpoint wall time and update count, never evaluation outcomes.
Sparse historical checkpoints leave an explicit unused-time gap.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(1)
    started = time.monotonic()
    matrix = json.loads((ROOT/'experiments/t6/informative_width_v1/comparison_v1.json').read_text())
    output = args.output_dir.resolve()
    output.relative_to(ROOT/'artifacts/research')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, status='running',
        created_utc=datetime.now(timezone.utc).isoformat(),
        matrix='experiments/t6/informative_width_v1/comparison_v1.json',
        common_decision_budget=matrix['common_decision_budget'],
        wall_caps_seconds=matrix['common_wall_budgets_seconds'], candidates=[],
        selection_rule='Greatest actual (wall_seconds, updates) at/before cap; filename breaks identical-state ties. No outcome selection.',
        scope='CPU checkpoint headers only, zero model/Native cases. Hardlinks preserve selected policies without changing resume pointers. Unused wall gaps remain; this does not certify equal consumed time or completed width comparison.')

    def save() -> None:
        temporary = output/'report.json.tmp'
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        temporary.replace(output/'report.json')

    save()
    try:
        for entry in matrix['order']:
            config = json.loads((ROOT/entry['config']).read_text())
            run = ROOT/entry['output_dir']
            row = dict(experiment_id=config['experiment_id'],
                initialization_seed=config['initialization_seed'], width=config['model']['width'],
                run_dir=entry['output_dir'], config=entry['config'], points=[])
            report['candidates'].append(row)
            state_path = run/'training_state.json'
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            row['observed_counters'] = state.get('counters')
            if state.get('counters', {}).get('decisions', 0) < matrix['common_decision_budget']:
                row['status'] = 'interaction_budget_pending'
                save()
                continue
            with (run/'.execution.lock').open('a+') as lock:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                state = json.loads(state_path.read_text())
                if state.get('phase') != 'ready' or json.loads((run/'experiment_config.json').read_text()) != config:
                    raise ValueError(f'Candidate must be paused at a complete boundary: {run.name}')
                origin = state['initial_state_sha256']
                provenance = json.loads((run/'provenance.json').read_text())
                headers = []
                for path in sorted(run.glob('runs/run_*/*.pt')):
                    checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
                    saved = checkpoint['training_state']
                    if (checkpoint.get('research_version') != 1
                            or checkpoint['experiment_config'] != config
                            or saved['initial_state_sha256'] != origin
                            or saved.get('initialization_provenance')
                            or saved['invocations'][0]['kind'] != 'random_initialization'
                            or checkpoint['provenance']['fingerprints']['simulator'] != provenance['fingerprints']['simulator']
                            or saved['counters']['decisions'] > state['counters']['decisions']):
                        raise ValueError(f'Checkpoint does not belong to candidate origin: {path}')
                    headers.append(dict(source_checkpoint=str(path.relative_to(ROOT)),
                        wall_seconds=saved['wall_seconds'], updates=saved['updates'],
                        counters=saved['counters'], source_revision=checkpoint['provenance']['commit'],
                        simulator_sha256=checkpoint['provenance']['fingerprints']['simulator']))
                    del checkpoint
                for cap in matrix['common_wall_budgets_seconds']:
                    eligible = [h for h in headers if h['wall_seconds'] <= cap]
                    if not eligible:
                        raise ValueError(f'No actual checkpoint at/before cap: {run.name}, {cap}')
                    selected = max(eligible, key=lambda h: (h['wall_seconds'], h['updates'], h['source_checkpoint']))
                    target = output/'points'/run.name/f'cap_{cap:.6f}.pt'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.link(ROOT/selected['source_checkpoint'], target)
                    row['points'].append(dict(**selected, wall_cap_seconds=cap,
                        unused_wall_seconds=cap-selected['wall_seconds'],
                        unused_fraction=(cap-selected['wall_seconds'])/cap,
                        retained_checkpoint=str(target.relative_to(ROOT))))
                row.update(status='selected_pending_score', inspected_checkpoint_count=len(headers),
                           initial_state_sha256=origin)
                save()
        report.update(status='complete', seconds=time.monotonic()-started,
                      cuda_initialized=torch.cuda.is_initialized(), extra_model_cases=0, extra_native_cases=0)
    except BaseException as error:
        report.update(status='failed_preserved', error=repr(error), seconds=time.monotonic()-started)
        save()
        raise
    save()
    print(json.dumps(dict(status=report['status'], seconds=report['seconds'],
                         candidates=len(report['candidates']), cuda_initialized=report['cuda_initialized'])))


if __name__ == '__main__':
    main()
