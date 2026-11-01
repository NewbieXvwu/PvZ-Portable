import json
from concurrent.futures import Future
from pathlib import Path

from test_curriculum_pipeline import ROOT, setup
import curriculum_queue as queue
from human_demo import export_verified
from pvz_frame_store import read_frames, write_frames
from pvz_recipe import load_recipe
from pvz_training_rows import source_factor
from test_compact_loader import game_frames


def test_verified_human_export_enters_both_clean_recipe_plans(tmp_path, monkeypatch):
    args, state = setup(tmp_path)
    raw = tmp_path / 'raw'
    for label, verified, source in (('user_l28', True, 'human'),
                                    ('unfinished', False, 'human'),
                                    ('whatif', True, 'counterfactual')):
        archive = raw / 'seed_271' / label
        archive.mkdir(parents=True)
        (archive / 'meta.json').write_text(json.dumps(dict(
            verified=verified, label_source=source, task=dict(id=label, seed=271))))
        if verified and source == 'human':
            write_frames(archive / 'frames.npz', game_frames(3))
    exported = export_verified([raw], args.frames)
    assert len(exported) == 1 and exported[0].name == 'user_l28'
    assert len(read_frames(exported[0] / 'frames.npz')) == 3
    assert export_verified([raw], args.frames) == []
    args.recipe = ROOT / 'recipes/dw4i2_v2timer_clean_a.json'
    args.alternate_recipe = ROOT / 'recipes/dw4i2_v2timer_clean_b.json'
    pipeline = queue.TrainingPipeline(args, state)
    monkeypatch.setattr(pipeline.executor, 'submit', lambda *a: Future())
    try:
        pipeline.enqueue()
        recipe = Path(state['training_jobs'][0]['recipe'])
        for path in (recipe, recipe.with_name('recipe_b.json')):
            planned = load_recipe(path, 'training')
            assert 'user_l28' in planned['labels']
            assert source_factor({'label_source': 'human'}, planned['source_weight']) == 1.
            assert source_factor({'label_source': 'teacher'}, planned['source_weight']) == 0.
            print(path.name, dict(human_labels=['user_l28'], source_weight=planned['source_weight']))
        pipeline._prepare_corpus(['user_l28'])
        assert (pipeline.directory / 'frames/seed_271/user_l28').resolve() == exported[0].resolve()
    finally:
        pipeline.executor.shutdown(wait=True)
