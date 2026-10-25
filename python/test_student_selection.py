"""task92: the pre-gate student selection panel -- seed isolation and pick bookkeeping."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
import pvz_student_selection as panel
from pvz_promotion import SEED_POOLS_V2 as POOLS
from select_student import score_rows, select


def test_training_seeds_are_two_distinct_integers():
    assert panel.TRAINING_SEEDS == (1, 2)
    assert len(set(panel.TRAINING_SEEDS)) == len(panel.TRAINING_SEEDS)


def test_seed_spans_covers_every_registered_shape():
    document = dict(a=dict(seeds=[5, 9]), b=[dict(future_seeds=[100, 104])],
                    c=dict(future_seed_start=200, future_seed_count=3), ignored=dict(seeds='no'))
    assert sorted(panel.seed_spans(document)) == [(5, 10), (100, 105), (200, 203)]


def test_assert_disjoint_rejects_duplicates_and_registered_overlap():
    make = lambda seeds: dict(tasks=[dict(task_id='t', seeds=seeds)])
    panel.assert_disjoint(make([1, 2, 3]), [(10, 20)])
    try:
        panel.assert_disjoint(make([1, 1]), [])
        assert False, 'duplicate panel seeds must fail'
    except ValueError as error:
        assert 'repeats a seed' in str(error)
    try:
        panel.assert_disjoint(make([15]), [(10, 20)])
        assert False, 'overlap with a registered interval must fail'
    except ValueError as error:
        assert 'overlap' in str(error)


def test_committed_panel_is_mechanically_disjoint_from_gate_and_demo_seeds():
    selection = json.loads((ROOT / 'configs/student_selection.json').read_text())
    reserved = list(POOLS)
    reserved.extend(panel.seed_spans(json.loads((ROOT / 'configs/human_demo_tasks.json').read_text())))
    panel.assert_disjoint(selection, reserved)
    assert selection['panel_version'] == 'selection-task92-v1'
    assert len(selection['tasks']) == len(panel.SELECTION_TASKS)
    for task in selection['tasks']:
        assert len(task['seeds']) == panel.SELECTION_GAMES


def manifest_document():
    return dict(schema_version=1, tasks=[
        dict(task_id='t1', seeds=[10, 11, 12, 13]),
        dict(task_id='t2', seeds=[20, 21, 22])])


def fake_runner(wins_by_candidate):
    """Return a run_command that plays won/lost rows per (candidate, task)."""
    calls = []
    def run(command, check=True, stdout=None, stderr=None):
        args = command
        model = args[args.index('--rollout-policy') + 1]
        out = Path(args[args.index('--out') + 1])
        manifest = json.loads(Path(args[args.index('--manifest') + 1]).read_text())
        task = args[args.index('--task-id') + 1]
        offset = int(args[args.index('--seed-offset') + 1])
        count = int(args[args.index('--seeds') + 1])
        seeds = next(t['seeds'] for t in manifest['tasks'] if t['task_id'] == task)[offset:offset + count]
        won = wins_by_candidate[model][task]
        out.mkdir(parents=True, exist_ok=True)
        rows = [dict(seed=seed, arm='fast', terminal=True, won=bool(i < won), searched=0)
                for i, seed in enumerate(seeds)]
        (out / 'games.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
        calls.append((model, task, offset, count))
        class Result:
            returncode = 0
        return Result()
    run.calls = calls
    return run


def test_select_picks_the_best_total_and_records_the_rejected_candidates():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        manifest = root / 'panel.json'
        manifest.write_text(json.dumps(manifest_document()))
        weak, strong = root / 'weak.pt', root / 'strong.pt'
        weak.write_bytes(b'weak')
        strong.write_bytes(b'strong weights')
        (root / 'engine').write_bytes(b'engine')
        wins = {str(weak): dict(t1=1, t2=1), str(strong): dict(t1=3, t2=2)}
        run = fake_runner(wins)
        result = select([str(weak), str(strong)], root / 'chosen', str(manifest), 't1:0:4,t2:0:3',
                        workers=2, value=root / 'value.pt', resource_dir=root / 'res',
                        executable=root / 'engine', run_command=run)
        assert result['best'] == str(strong)
        assert result['rejected'] == [str(weak)]
        assert result['scores'][str(strong)]['total'] == 5
        assert result['scores'][str(weak)]['wins'] == dict(t1=1, t2=1)
        assert result['manifest_sha256'] == __import__('hashlib').sha256(manifest.read_bytes()).hexdigest()
        assert (root / 'chosen/model.pt').read_bytes() == b'strong weights'
        # Every candidate played every requested slice exactly once.
        assert len(run.calls) == 4
        assert sorted({c[0] for c in run.calls}) == sorted([str(weak), str(strong)])
        assert (root / 'chosen/selection.json').is_file()


def test_select_rejects_incomplete_or_wrongly_played_games():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        manifest = root / 'panel.json'
        manifest.write_text(json.dumps(manifest_document()))
        model = root / 'm.pt'
        model.write_bytes(b'm')
        def bad_runner(rows):
            def run(command, check=True, stdout=None, stderr=None):
                out = Path(command[command.index('--out') + 1])
                out.mkdir(parents=True, exist_ok=True)
                (out / 'games.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
                class Result:
                    returncode = 0
                return Result()
            return run
        for rows, note in [
            ([dict(seed=10, arm='fast', terminal=True, won=True, searched=0)], 'missing games'),
            ([dict(seed=99, arm='fast', terminal=True, won=True, searched=0) for _ in range(4)], 'foreign seed'),
            ([dict(seed=s, arm='greedy', terminal=True, won=True, searched=0) for s in (10, 11, 12, 13)], 'wrong arm'),
            ([dict(seed=s, arm='fast', terminal=False, won=True, searched=0) for s in (10, 11, 12, 13)], 'non terminal'),
            ([dict(seed=s, arm='fast', terminal=True, won=True, searched=7) for s in (10, 11, 12, 13)], 'searched fast arm'),
        ]:
            try:
                select([str(model)], root / 'out', str(manifest), 't1:0:4', workers=1,
                       value=root / 'v.pt', resource_dir='', executable=root / 'e',
                       run_command=bad_runner(rows))
                assert False, f'{note} must not be accepted'
            except ValueError:
                pass


def test_score_rows_sums_wins_over_exact_seed_set():
    rows = [dict(seed=s, arm='fast', terminal=True, won=s % 2 == 0, searched=0)
            for s in (10, 11, 12, 13)]
    assert score_rows_lines(rows, {10, 11, 12, 13}) == 2


def score_rows_lines(rows, expected):
    # score_rows reads a file; exercise through a temp file for the real path.
    import tempfile as tf
    with tf.NamedTemporaryFile('w', suffix='.jsonl', delete=False) as handle:
        handle.write('\n'.join(json.dumps(r) for r in rows))
        path = Path(handle.name)
    try:
        return score_rows(path, expected)
    finally:
        path.unlink()
