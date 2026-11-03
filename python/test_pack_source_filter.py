import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]

from pvz_training_pack import filter_sources, pack_options, segment_contracts, segment_path
from pvz_training_rows import source_enabled, source_kind


def archive(root, relative, meta):
    path = root / relative
    path.mkdir(parents=True)
    (path / 'frames.npz').write_bytes(b'')
    (path / 'meta.json').write_text(json.dumps(meta))
    return path / 'frames.npz'


def test_source_enabled_is_the_union_of_active_recipes():
    old = dict(label_source='teacher', task=dict(checkpoint='/runs/round_000002/student/model.pt'))
    r52 = dict(label_source='teacher', task=dict(checkpoint='/runs/round_000052/student/model.pt'))
    weights = [dict(default=0., teacher_r52=1.), dict(default=0.)]
    assert not source_enabled(old, weights)
    assert source_enabled(r52, weights)
    assert source_enabled(old, [dict(default=0.), dict(default=.1)])
    assert source_enabled(old, [])


def test_filter_sources_drops_only_rows_no_recipe_trains(tmp_path):
    old = archive(tmp_path, 'seed_1/old', dict(label_source='teacher'))
    r52 = archive(tmp_path, 'seed_2/new', dict(label_source='teacher',
                                              task=dict(checkpoint='/runs/round_000052/model.pt')))
    human = archive(tmp_path, 'seed_3/human', dict(label_source='human'))
    weights = [dict(default=0., teacher_r52=1., human=1.)]
    assert filter_sources([old, r52, human], weights) == [r52, human]
    assert filter_sources([old, r52], None) == [old, r52]


def test_demo_segments_split_by_import_batch_and_chunk(tmp_path, monkeypatch):
    paths = []
    for batch in ('a' * 64, 'b' * 64):
        for seed in (1, 2):
            paths.append(archive(
                tmp_path,
                f'.llm_demo_rows/{batch}/seed_{seed}/llm_demo_decisions',
                dict(label_source='llm_gpt_pro', llm_demo_sha256=batch)))
    options = pack_options({'llm_demo_decisions'}, obs='v2')
    contracts = segment_contracts(tmp_path, options, 'llm_demo_decisions', paths)
    assert len(contracts) == 2
    assert len({segment_path(tmp_path, contract) for contract, _ in contracts}) == 2
    assert sum(len(group) for _, group in contracts) == 4

    monkeypatch.setattr('pvz_training_pack.DEMO_SEGMENT_ARCHIVES', 1)
    chunked = segment_contracts(tmp_path, options, 'llm_demo_decisions', paths)
    assert len(chunked) == 4
    assert len({segment_path(tmp_path, contract) for contract, _ in chunked}) == 4


def test_pack_omits_a_label_every_recipe_zeroes(tmp_path):
    import numpy as np
    from pack_training_rows import pack
    from pvz_training_pack import archive_fingerprint
    from test_incremental_pack import fixture

    data = tmp_path / 'data'
    fixture(data, ['keep', 'drop'])
    for meta in data.glob('seed_*/keep/meta.json'):
        meta.write_text(json.dumps(dict(
            label_source='teacher', task=dict(checkpoint='/runs/round_000052/student/model.pt'))))
    labels = {'keep', 'drop'}
    options = pack_options(labels, obs='v2', sources=archive_fingerprint(data, labels))
    index = pack(data, options, tmp_path / 'cache',
                 source_weights=[dict(default=0., teacher_r52=1.)])
    with np.load(index, allow_pickle=False) as stored:
        segments = [Path(index).parent / name for name in stored['segments'].tolist()]
        assert int(stored['rows']) > 0
    packed = []
    for segment in segments:
        with np.load(segment, allow_pickle=False) as stored:
            packed.extend(json.loads(str(stored['meta']))['labels'])
    assert packed == ['keep']


def test_teacher_line_is_a_promoted_new_line_student():
    def kind(round_number):
        return source_kind(dict(label_source='teacher', task=dict(
            checkpoint=f'/runs/round_{round_number:06d}/student/model.pt')))

    assert kind(72) == 'teacher_line'
    assert kind(94) == 'teacher_line'
    assert kind(101) == 'teacher_line'
    assert kind(71) == 'teacher'  # never promoted, never a collection prior
    assert kind(52) == 'teacher_r52'  # existing categories keep their meaning
    assert source_kind(dict(label_source='teacher',
                            task=dict(policy='artifacts/student/gptbc_s1/model.pt'))) == 'teacher_gptbc'
    assert source_kind(dict(label_source='teacher',
                            task=dict(checkpoint='round_000094/student/model.pt.bak'))) == 'teacher'
    assert source_kind(dict(label_source='human', task=dict(
        checkpoint='round_000094/student/model.pt'))) == 'human'


def packed_labels(index):
    import numpy as np
    labels = []
    with np.load(index, allow_pickle=False) as stored:
        for name in stored['segments'].tolist():
            with np.load(Path(index).parent / name, allow_pickle=False) as segment:
                labels.extend(json.loads(str(segment['meta']))['labels'])
    return labels


def test_pack_recovers_teacher_line_rows_on_the_next_recipe_change(tmp_path):
    import numpy as np
    from pack_training_rows import pack
    from pvz_training_pack import archive_fingerprint
    from test_incremental_pack import fixture

    data = tmp_path / 'data'
    fixture(data, ['line', 'r52'])
    for meta in data.glob('seed_*/line/meta.json'):
        meta.write_text(json.dumps(dict(
            label_source='teacher', task=dict(checkpoint='/runs/round_000094/student/model.pt'))))
    for meta in data.glob('seed_*/r52/meta.json'):
        meta.write_text(json.dumps(dict(
            label_source='teacher', task=dict(checkpoint='/runs/round_000052/student/model.pt'))))
    labels = {'line', 'r52'}
    options = pack_options(labels, obs='v2', sources=archive_fingerprint(data, labels))
    before = pack(data, options, tmp_path / 'cache',
                  source_weights=[dict(default=0., teacher_r52=1., teacher_gptbc=1.)])
    assert sorted(packed_labels(before)) == ['r52']
    with np.load(before, allow_pickle=False) as stored:
        rows_before = int(stored['rows'])
    after = pack(data, options, tmp_path / 'cache',
                 source_weights=[dict(default=0., teacher_r52=1., teacher_gptbc=1., teacher_line=1.)])
    assert sorted(packed_labels(after)) == ['line', 'r52']
    with np.load(after, allow_pickle=False) as stored:
        assert int(stored['rows']) > rows_before
