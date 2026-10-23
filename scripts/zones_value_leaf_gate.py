"""Check 64 archive roots with the same v4 weights on two engine builds.

Both full leaves/logical statistics and scored-leaf responses must be bit-identical.
Each scored leaf is also checked against Python v4 at a 1e-5 probability tolerance.
Engines run sequentially, so this gate owns at most one game process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
import bench_engine as bench


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args, executable, student_binary, value_binary, value):
    import torch
    from pvz_value_net import encode_frame, observation_context
    records = []
    max_error = 0.0
    roots = 0
    manifest = json.loads(args.manifest.read_text())
    with bench._open_env(str(executable)) as env:
        env.load_rollout_net(student_binary)
        env.load_value_net(value_binary)
        for family, entries in manifest['families'].items():
            for entry in entries:
                archive = Path(entry['archive'])
                rows = bench._archive_rows(archive / ('frames.npz' if (archive / 'frames.npz').is_file()
                                                       else 'frames.jsonl'))
                snapshots, mismatches = bench.replay_episode(env, entry, rows, entry['decisions'])
                assert not mismatches and len(snapshots) == len(entry['decisions'])
                for index, snapshot in snapshots.items():
                    roots += 1
                    start = env.restore(snapshot)['tick']
                    for seed in range(3):
                        leaf = env.rollout(None, start + 2500, sim_skip=61, snapshot_id=snapshot, seed=seed)
                        full_digest = bench._rollout_digest(leaf, env._last_response)
                        if 'rollout_result' in leaf:
                            expected = float(leaf['rollout_result'])
                        elif leaf['terminal']:
                            expected = float(leaf['result'] == 1)
                        else:
                            board, scalars = encode_frame(bench._frame(leaf), value.encoder_version)
                            if value.context:
                                scalars = torch.cat([scalars, torch.from_numpy(observation_context(leaf))])
                            with torch.inference_mode():
                                expected = torch.sigmoid(value(board[None], scalars[None])).item()
                        scored = env.rollout(None, start + 2500, sim_skip=61, snapshot_id=snapshot,
                                             seed=seed, score_value=True)
                        error = abs(scored['value'] - expected)
                        max_error = max(max_error, error)
                        assert error < 1e-5, (family, index, seed, error)
                        records.append(dict(family=family, archive=str(archive), frame=index, seed=seed,
                                            full_leaf=full_digest, scored=scored,
                                            decisions=env._last_response['decisions']))
                    env.release_snapshot(snapshot)
            print(f'{executable}: {family} passed', flush=True)
    assert roots == 64 and len(records) == 192
    return dict(executable=str(executable), executable_sha256=sha(executable), roots=roots,
                leaves=len(records), max_python_value_error=max_error,
                full_and_scored_digest=bench._digest(records)), records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, action='append', required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--student', type=Path, required=True)
    parser.add_argument('--value', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert len(args.executable) == 2
    import torch
    from export_value_net import export_value_net
    from pvz_value_net import load_value_net
    from pvz_rollout_policy import load_student, export_student
    torch.set_num_threads(1)
    value = load_value_net(args.value)
    assert value.encoder_version == 2
    runs = []
    with tempfile.TemporaryDirectory(prefix='zones-value-leaves-') as tmp:
        student_binary, value_binary = Path(tmp) / 'student.bin', Path(tmp) / 'value.bin'
        export_student(load_student(args.student), student_binary)
        export_value_net(value, value_binary)
        reference = None
        for executable in args.executable:
            summary, records = run(args, executable, student_binary, value_binary, value)
            if reference is not None:
                assert records == reference, 'full/scored leaves or logical statistics differ'
            reference = records
            runs.append(summary)
    report = dict(value_sha256=sha(args.value), student_sha256=sha(args.student),
                  manifest_sha256=sha(args.manifest), encoder_version=2, identical=True, runs=runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
