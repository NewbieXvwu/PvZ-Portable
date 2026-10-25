import json

import gate_seed_inventory as inventory


def spans(document):
    return {(row['low'], row['high']) for row in inventory.json_segments(document, 'fixture')}


def test_manifest_task_list_includes_current_and_demo_future_seeds():
    assert spans({'tasks': [{'seeds': [20_261_031], 'demo_start': {
        'future_seed_start': 44_000_000, 'future_seed_count': 256}},
        {'seeds': [81_000_000, 81_000_127]}]}) == {
            (20_261_031, 20_261_032), (44_000_000, 44_000_256), (81_000_000, 81_000_128)}


def test_closed_prose_ranges_keep_the_last_seed(tmp_path):
    design = tmp_path / 'DESIGN.md'
    design.write_text('30000–31023; [31024, 65536); 81,000,000..81,000,127')
    assert {(row['low'], row['high']) for row in inventory.design_segments(design)} == {
        (30000, 31024), (31024, 65536), (81_000_000, 81_000_128)}


def test_manifest_dictionary_ledger_and_wide_last_task_are_not_dropped():
    assert (100_000_000, 100_000_002) in spans({'tasks': {'probe': [100_000_000, 100_000_001]}})
    assert (100_000_000, 100_000_002) in spans({'tasks': {'probe': {'seeds': [100_000_000, 100_000_001]}}})
    assert spans({'pool': [31024, 65536], 'allocated': [[31024, 32560]]}) == {
        (31024, 65536), (31024, 32560)}
    assert (49_000_000, 49_008_192) in spans({'seed_blocks': {'train': [49_000_000, 49_008_191]}})
    # per_task may exceed stride: do not leave the final task's tail "free".
    assert (100_000_000, 100_000_040) in spans(
        {'base': 100_000_000, 'stride': 10, 'per_task': 30, 'tasks': ['a', 'b']})


def test_literal_review_excludes_float_and_hex_but_retains_actual_seed_and_key(tmp_path):
    (tmp_path / 'game.json').write_text(json.dumps({
        'score': .100845682, 'hash': '100387332e90fcf1',
        'seed': 100_000_123, 'rss_bytes': 100_500_000}))
    review = inventory.verify_window([str(tmp_path)], 100_000_000, 101_000_000)
    assert [(hit['key'], hit['value']) for hit in review['hits']] == [
        ('seed', 100_000_123), ('rss_bytes', 100_500_000)]


def test_scanning_overlapping_roots_reads_each_file_once(tmp_path):
    nested = tmp_path / 'configs'
    nested.mkdir()
    (nested / 'tasks.json').write_text(json.dumps({'tasks': [{'seeds': [100_000_123]}]}))
    scan = inventory.scan_json([str(tmp_path), str(nested)])
    assert scan['files_seen'] == 1
    assert [(s['low'], s['high']) for s in scan['segments']] == [(100_000_123, 100_000_124)]


def test_free_gaps_do_not_include_registered_endpoints():
    occupied = inventory.merge([(30000, 31024), (31024, 65536)])
    assert occupied == [[30000, 65536]]
    assert inventory.free_gaps(occupied, 65537, 1) == [
        {'low': 0, 'high': 30000, 'size': 30000}, {'low': 65536, 'high': 65537, 'size': 1}]
