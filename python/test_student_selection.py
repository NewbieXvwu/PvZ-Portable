"""task92: the pre-gate student selection panel -- seed isolation and pick bookkeeping."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
import pvz_student_selection as panel
from select_student import score_rows, select, selection_rank


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


def manifest_document(incumbent):
    return dict(schema_version=1, incumbent=str(incumbent), tasks=[
        dict(task_id='t1', terrain='day', selection_group='adventure', seeds=[10, 11, 12, 13]),
        dict(task_id='t2', terrain='night', selection_group='adventure', seeds=[20, 21, 22])])


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
        rows = [dict(seed=seed, arm='fast', terminal=True, won=bool(i < won), searched=0, final_wave=10)
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
        incumbent = root / 'incumbent.pt'
        incumbent.write_bytes(b'incumbent weights')
        manifest = root / 'panel.json'
        manifest.write_text(json.dumps(manifest_document(incumbent)))
        weak, strong = root / 'weak.pt', root / 'strong.pt'
        weak.write_bytes(b'weak')
        strong.write_bytes(b'strong weights')
        (root / 'engine').write_bytes(b'engine')
        wins = {str(incumbent): dict(t1=2, t2=2),
                str(weak): dict(t1=1, t2=1), str(strong): dict(t1=3, t2=2)}
        run = fake_runner(wins)
        result = select([str(weak), str(strong)], root / 'chosen', str(manifest), 't1:0:4,t2:0:3',
                        workers=2, value=root / 'value.pt', resource_dir=root / 'res',
                        executable=root / 'engine', run_command=run)
        assert result['best'] == str(strong)
        assert result['rejected'] == [str(weak)]
        assert result['scores'][str(strong)]['total'] == 5
        assert result['scores'][str(weak)]['wins'] == dict(t1=1, t2=1)
        assert result['manifest'] == str(manifest)
        assert (root / 'chosen/model.pt').read_bytes() == b'strong weights'
        # Incumbent plus both candidates played every requested slice exactly once.
        assert len(run.calls) == 6
        assert sorted({c[0] for c in run.calls}) == sorted([str(incumbent), str(weak), str(strong)])
        assert (root / 'chosen/selection.json').is_file()


def test_select_rejects_incomplete_or_wrongly_played_games():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        incumbent = root / 'incumbent.pt'
        incumbent.write_bytes(b'incumbent weights')
        manifest = root / 'panel.json'
        manifest.write_text(json.dumps(manifest_document(incumbent)))
        model = root / 'm.pt'
        model.write_bytes(b'm')
        def bad_runner(rows):
            def run(command, check=True, stdout=None, stderr=None):
                out = Path(command[command.index('--out') + 1])
                out.mkdir(parents=True, exist_ok=True)
                played = rows
                if command[command.index('--rollout-policy') + 1] == str(incumbent):
                    task = command[command.index('--task-id') + 1]
                    offset = int(command[command.index('--seed-offset') + 1])
                    count = int(command[command.index('--seeds') + 1])
                    seeds = next(t['seeds'] for t in manifest_document(incumbent)['tasks']
                                 if t['task_id'] == task)[offset:offset + count]
                    played = [dict(seed=s, arm='fast', terminal=True, won=True, searched=0) for s in seeds]
                (out / 'games.jsonl').write_text('\n'.join(json.dumps(r) for r in played) + '\n')
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


def test_r26_selection_marginals_replay_with_one_shared_synthetic_incumbent(tmp_path):
    zones = ('day', 'night', 'pool', 'fog', 'roof')
    counts = (72, 72, 72, 72, 16, 24, 16)
    names = (*zones, 'random', 'unseen')
    recorded = ((58, 65, 48, 28, 0, 3, 8), (63, 68, 50, 37, 1, 4, 7),
                (52, 51, 37, 34, 0, 2, 8), (47, 59, 42, 39, 1, 3, 8))
    tasks, cursor = [], 130_000_000
    for name, count in zip(names, counts):
        tasks.append(dict(task_id=name, terrain=name, seeds=list(range(cursor, cursor + count)),
                          selection_group='adventure' if name in zones else name,
                          score_only=name == 'unseen'))
        cursor += count
    incumbent = tmp_path / 'incumbent.pt'
    incumbent.write_bytes(b'synthetic reference, not the missing r11 selection outcomes')
    manifest = tmp_path / 'panel.json'
    manifest.write_text(json.dumps(dict(tasks=tasks, incumbent=str(incumbent))))
    models, wins = [], {str(incumbent): dict(zip(names, (60, 60, 45, 35, 1, 24, 16)))}
    for seed, values in enumerate(recorded, 1):
        model = tmp_path / f'seed_{seed:06}.pt'
        model.write_bytes(str(seed).encode())
        models.append(str(model))
        wins[str(model)] = dict(zip(names, values))
    engine = tmp_path / 'engine'
    engine.write_bytes(b'engine')
    run = fake_runner(wins)
    result = select(models, tmp_path / 'out', manifest,
                    ','.join(f'{name}:0:{count}' for name, count in zip(names, counts)),
                    2, '', '', engine, run_command=run)
    assert [result['scores'][model]['total'] for model in models] == [202, 223, 176, 191]
    assert result['best'] == models[1]
    assert len(run.calls) == 5 * len(tasks)
    assert sum(call[0] == str(incumbent) for call in run.calls) == len(tasks)
    prediction = result['scores'][models[1]]['prediction']
    assert prediction['approved'] and prediction['overall']['pairs'] == 304
    assert prediction['overall']['gain'] == 18 and prediction['overall']['harm'] == 0
    assert set(prediction['zones']) == set(zones)
    assert all(zone['holm_adjusted_p'] == 1 for name, zone in prediction['zones'].items() if name != 'roof')
    first = result['scores'][models[0]]['prediction']
    assert first['zones']['fog']['gain'] == 0 and first['zones']['fog']['harm'] == 7
    assert first['zones']['fog']['holm_adjusted_p'] == 4 / 128
    assert 'significant_zone_regression:fog' in first['reasons']
    assert result['incumbent']['path'] == str(incumbent)


def test_selection_prefers_approval_then_total_or_least_worst_zone():
    from pvz_promotion import decide_scored
    scores = {}
    for name, wins in dict(approved=(80, 80, 80), better=(81, 81, 81),
                           high_total=(60, 100, 100), least_regression=(70, 70, 70)).items():
        pairs = [dict(zone=zone, candidate=seed < count, incumbent=seed < 80)
                 for zone, count in zip(('day', 'night', 'pool'), wins) for seed in range(100)]
        scores[name] = dict(total=sum(wins), prediction=decide_scored(pairs))
    assert scores['high_total']['total'] > scores['better']['total']
    assert max(scores, key=lambda name: selection_rank(scores[name])) == 'better'
    rejected = ('high_total', 'least_regression')
    assert all(not scores[name]['prediction']['approved'] for name in rejected)
    assert max(rejected, key=lambda name: selection_rank(scores[name])) == 'least_regression'


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
