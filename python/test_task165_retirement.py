"""Small task165 fixtures: no production files, real games or training runs."""
import json
from pathlib import Path
import re
import sys
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
from pvz_corpus_policy import ABANDONED_LABELS
from retire_task20 import retire_links, restore_links
from stabilize_production_paths import recipe_text, rewrite


def test_task165_retirement_is_exact_and_reversible(tmp_path):
    view = tmp_path / 'view'
    for seed, label in [(1880000, 'd10orig1_fast_search'), (1880128, 'd10orig2_fast_search'),
                        (7, 'unrelated')]:
        link = view / f'seed_{seed}' / label
        link.parent.mkdir(parents=True)
        link.symlink_to(tmp_path / 'missing' / label, target_is_directory=True)
    before = retire_links(view)
    assert len(before) == 2 and all(Path(x['path']).is_symlink() for x in before)
    assert retire_links(view, True) == before
    assert retire_links(view, True) == []
    assert (view / 'seed_7/unrelated').is_symlink()
    restore_links(before)
    assert len(retire_links(view)) == 2


def test_task165_retirement_reuses_segments_and_training_input(tmp_path, monkeypatch):
    from test_incremental_pack import fixture, options
    from pack_training_rows import pack
    from pvz_frame_store import frame_paths
    from pvz_recipe import load_recipe, write_recipe
    import train_rollout_policy as trainer
    data = tmp_path / 'data'
    fixture(data, ['fresh'])
    index = pack(data, options(data, ['fresh']))
    stamps = {p: p.stat().st_mtime_ns for p in (data / 'packs/segments').glob('*.npz')}
    # Directory links still exist, but both frame files inside point at deleted originals.
    for i, label in enumerate(sorted(ABANDONED_LABELS)):
        original = tmp_path / label
        original.mkdir()
        (original / 'frames.npz').symlink_to(tmp_path / 'deleted.npz')
        link = data / f'seed_{1880000 + i * 128}' / label
        link.parent.mkdir()
        link.symlink_to(original, target_is_directory=True)
    labels = ['fresh', *sorted(ABANDONED_LABELS)]
    assert {p.parent.name for p in frame_paths(data)} == {'fresh'}
    with patch('pack_training_rows._build_segment', side_effect=AssertionError('must reuse segment')):
        assert pack(data, options(data, labels)) == index
    assert all(p.stat().st_mtime_ns == stamp for p, stamp in stamps.items())
    assert len(trainer.load(data, set(labels))[2]) > 0

    recipe = tmp_path / 'old_recipe.json'
    write_recipe(recipe, dict(schema_version=1, kind='training', data=str(data), labels=['fresh'],
                             out=str(tmp_path / 'student'), obs='v1', wait_keep=1.0))
    document = json.loads(recipe.read_text())
    document['labels'] = labels  # emulate a saved job written before retirement approval
    recipe.write_text(json.dumps(document))
    assert load_recipe(recipe)['labels'] == ['fresh']
    monkeypatch.setattr(sys, 'argv', ['train', '--recipe', str(recipe)])
    monkeypatch.setattr(trainer, 'Metrics', Mock())
    monkeypatch.setattr(trainer, 'set_oom_score_adj', lambda *_: None)
    original_load = trainer.load
    observed = {}
    class Loaded(Exception):
        pass
    def load(*args, **kwargs):
        observed['rows'] = len(original_load(*args, **kwargs)[2])
        observed['labels'] = args[1] if len(args) > 1 else kwargs['labels']
        raise Loaded
    monkeypatch.setattr(trainer, 'load', load)
    with pytest.raises(Loaded):
        trainer.main()
    assert observed['rows'] > 0 and not set(observed['labels']) & ABANDONED_LABELS
    # Unapproved losses continue to fail rather than silently disappear.
    unknown = data / 'seed_9/not_abandoned'
    unknown.mkdir(parents=True)
    (unknown / 'frames.npz').symlink_to(tmp_path / 'another_deleted.npz')
    with pytest.raises(FileNotFoundError):
        options(data, ['fresh', 'not_abandoned'])


def test_task165_stable_panels_and_path_rewrites():
    from pvz_recipe import load_recipe
    panel = json.loads((ROOT / 'configs/panels/day_orig_panel.json').read_text())
    task = next(x for x in panel['tasks'] if x['task_id'] == 'day_l10_orig_train')
    assert task['level'] == 10 and task['deck'] == [] and task['conveyor_start'] == 'original'
    assert task['seeds'] == list(range(1880000, 1888192))
    resolved = load_recipe(ROOT / 'recipes/panel_v2.json', 'panel')
    assert Path(resolved['day_orig_manifest']) == ROOT / 'configs/panels/day_orig_panel.json'
    old = '.claude/' + 'worktrees/'
    recipe = dict(kind='training', data=old + 'dw4/artifacts/task23/training_frames',
                  labels=['fresh', *sorted(ABANDONED_LABELS)])
    changed = json.loads(recipe_text(json.dumps(recipe)))
    assert changed['labels'] == ['fresh'] and changed['data'] == 'artifacts/student/frames_v2'
    assert rewrite('PACKAGE=$R/' + old + 'prodint16\nEXE16=$PACKAGE/build16/pvz-portable\n', '.sh') == (
        'PACKAGE=$R\nEXE16=$R/artifacts/production/deploy16/pvz-portable\n')


def test_task165_no_runtime_worktree_defaults():
    pattern = re.compile(r'\.claude/worktrees/([A-Za-z0-9_-]+)(?=/|[\'"\s])')
    bad = []
    for folder in ('scripts', 'python', 'recipes', 'configs'):
        for path in (ROOT / folder).rglob('*'):
            if path.suffix not in ('.py', '.sh', '.json') or not path.is_file() or path.name.startswith('test_'):
                continue
            for match in pattern.finditer(path.read_text()):
                if match[1] not in ('cq', 'obsup'):
                    bad.append((str(path.relative_to(ROOT)), match[0]))
    assert bad == []


def test_task165_update_backup_restore_and_immutable_engine(tmp_path):
    from apply_deploy17_paths import save_and_write, restore, stage_engine
    old = tmp_path / 'old.sh'
    old.write_text('previous')
    old.chmod(0o755)
    added = tmp_path / 'added.json'
    segment = tmp_path / 'existing-segment.npz'
    segment.write_bytes(b'existing segment')
    stamp = segment.stat().st_mtime_ns
    backup = tmp_path / 'backup'
    save_and_write({old: b'next', added: b'{}'}, backup)
    assert old.read_text() == 'next' and old.stat().st_mode & 0o111
    restore(backup)
    assert old.read_text() == 'previous' and not added.exists()
    assert segment.read_bytes() == b'existing segment' and segment.stat().st_mtime_ns == stamp
    source, target = tmp_path / 'source-engine', tmp_path / 'stable/engine'
    source.write_bytes(b'immutable executable')
    source.chmod(0o755)
    assert stage_engine(source, target) == stage_engine(source, target)
    source.write_bytes(b'different')
    with pytest.raises(RuntimeError, match='different bytes'):
        stage_engine(source, target)
    assert target.read_bytes() == b'immutable executable'


def test_task165_update_watchdog_bridge_is_extractable():
    import subprocess
    # Extract only; never invoke the function or open a tmux session.
    result = subprocess.run(['bash', '-c', 'source "$1"; watchdog_extract_start "$2"',
                             'fixture', str(ROOT / 'scripts/qc_watchdog.sh'),
                             str(ROOT / 'scripts/qc_deploy17_paths.sh')],
                            capture_output=True, text=True, check=True)
    assert 'watchdog_extract_start "$2" | /bin/bash' in result.stdout
    assert 'qc_deploy17.sh' in result.stdout and '--at-boundary' not in result.stdout
