"""task126: selection-only decks, four frozen candidates, serial resume and deployment isolation."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
import curriculum_queue as queue
from pvz_recipe import load_recipe, write_recipe
from pvz_selection_refresh import make_plan, selection_templates, SELECTION_POOLS, SELECTION_BLOCK_SIZE
from pvz_student_selection import assert_disjoint, seed_spans
from test_curriculum_pipeline import setup, write_evaluation, write_selection

HOLDOUT = ROOT / 'configs/student_selection_holdout.json'


def test_adventure_body_and_score_only_unseen_decks(tmp_path):
    source = ROOT / 'configs/gate_v5_panel.json'
    document = json.loads(source.read_text())
    templates = selection_templates(HOLDOUT, source)
    adventure = [t for t in templates['tasks'] if t.get('selection_group') == 'adventure']
    assert {t['task_id'] for t in adventure} == {t['task_id'] for t in document['tasks']} - set(document['monitor'])
    assert len(adventure) == 36 and all(t['seeds'] == [] for t in adventure)
    assert len([t for t in templates['tasks'] if 'deck_random' in t]) == 3
    holds = [t for t in templates['tasks'] if t.get('selection_group') == 'unseen_deck']
    assert len(holds) == 2 and all(t['score_only'] for t in holds)
    # d7q cannot draw the umbrella set (37 is outside its pool); the starfruit
    # set is explicitly excluded. d7p exclusions are documented from its source manifest.
    registered = json.loads((ROOT / 'configs/student_selection_v2.json').read_text())
    q = next(t for t in registered['tasks'] if t['task_id'] == 'day_l8_train_d7q')['deck_perturb']
    for signature in {tuple(sorted(t['deck'])) for t in holds}:
        assert signature in {tuple(sorted(d)) for d in q['excluded_decks']} or not set(signature) <= set(q['pool'])
    model = tmp_path / 'model.pt'
    model.write_bytes(b'incumbent')
    block = (SELECTION_POOLS[0][0], SELECTION_POOLS[0][0] + SELECTION_BLOCK_SIZE)
    panel = make_plan(templates, block, model)
    assert panel['panel_version'] == 'selection-task128-v4'
    assert len(panel['tasks']) == 41 and len(panel['adventure_tasks']) == 36
    assert {len(t['seeds']) for t in panel['tasks']} == {SELECTION_BLOCK_SIZE // 41}
    assert sum(len(t['seeds']) for t in panel['tasks']) <= SELECTION_BLOCK_SIZE
    assert panel['score_only'] == panel['selection_holdout_tasks'] == [t['task_id'] for t in holds]
    assert_disjoint(panel, [(31024, 65536), (100_000_000, 101_000_000),
                            (103_000_000, 103_500_000), (200_000_000, 201_000_000)])


def test_selection_requires_the_gate_v5_adventure_panel(tmp_path):
    legacy = tmp_path / 'legacy.json'
    legacy.write_text(json.dumps(dict(panel_version='v2', groups={}, tasks=[])))
    with pytest.raises(ValueError, match='gate-v5'):
        selection_templates(HOLDOUT, legacy)
    with pytest.raises(ValueError, match='gate-v5'):
        selection_templates()


def test_gate_card_sets_are_rejected_even_under_new_names(tmp_path):
    gate = tmp_path / 'gate.json'
    task = json.loads(HOLDOUT.read_text())['tasks'][0]
    gate.write_text(json.dumps(dict(panel_version='v5', monitor=[], groups={'adv_l7': 'day'},
                                    tasks=[dict(task, task_id='adv_l7')])))
    with pytest.raises(ValueError, match='distinct'):
        selection_templates(HOLDOUT, gate)


def test_two_new_decks_reset_in_the_engine_without_playing_a_game():
    from pvz_env import PvZEnv, TaskSpec
    from pvz_tasks import profile_for_deck
    executable = Path('/home/newbiexvwu/PvZ-search/.claude/worktrees/cq/build6/pvz-portable')
    resources = Path.home() / '.cache/pvz-research-resources'
    if not executable.exists() or not resources.exists():
        pytest.skip('requires local build6 and resources for two legality-only resets')
    with PvZEnv(resource_dir=resources, executable=executable, headless=True) as env:
        for task in json.loads(HOLDOUT.read_text())['tasks']:
            deck = tuple(task['deck'])
            env.reset(deck=deck, task=TaskSpec(level=task['level'], seed=0,
                playthrough=2, profile=profile_for_deck(deck)))
            assert not env.deck_warnings(deck)['warnings']


def test_four_candidates_are_serial_frozen_and_resume_without_retraining(tmp_path, monkeypatch):
    args, state = setup(tmp_path)
    alternate = tmp_path / 'b.json'
    write_recipe(alternate, dict(load_recipe(args.recipe), source_weight=dict(level_balance=.25)))
    args.alternate_recipe = alternate
    args.candidates_per_recipe = 2
    args.training_parallelism = 1
    args.selection_holdout_manifest = HOLDOUT
    state.update(round=1, log=[dict(groups=[dict(label='c1a')])])
    monkeypatch.setattr(queue, 'mem_available_bytes', lambda: 64 * queue.GIB)
    monkeypatch.setattr(queue, 'mem_total_bytes', lambda: 64 * queue.GIB)
    monkeypatch.setattr(queue, 'gpu_free_bytes', lambda: 16 * queue.GIB)
    def prepare(self, job):
        path = Path(job['selection_manifest'])
        if not path.exists():
            templates = json.loads(Path(job['selection_refresh']['templates']).read_text())
            panel = make_plan(templates, job['selection_refresh']['seed_block'], args.rollout_policy)
            path.write_text(json.dumps(panel))
        return dict(manifest_sha256=queue.file_sha(path))
    monkeypatch.setattr(queue.TrainingPipeline, '_prepare_selection', prepare)
    seen, calls = [], []
    interrupted = False
    def command(self, command, logfile):
        nonlocal interrupted
        calls.append(logfile.name)
        if logfile.name == 'train.log':
            assert self.train_parallelism == 1
            recipe = load_recipe(Path(command[command.index('--recipe') + 1]))
            seen.append(recipe)
            model = logfile.parent / 'student/model.pt'
            model.parent.mkdir(parents=True, exist_ok=True)
            model.write_bytes(str(recipe['seed']).encode())
        elif logfile.name == 'selection.log':
            if not interrupted:
                interrupted = True
                raise RuntimeError('selection interrupted')
            write_selection(self, command)
            out = Path(command[command.index('--out') + 1]) / 'selection.json'
            receipt = json.loads(out.read_text())
            panel = json.loads(Path(command[command.index('--manifest') + 1]).read_text())
            for score in receipt['scores'].values():
                score['wins'] = {t['task_id']: 1 for t in panel['tasks']}
            out.write_text(json.dumps(receipt))
        elif logfile.name == 'evaluation.log':
            write_evaluation(self, command)
        return queue.GIB
    monkeypatch.setattr(queue.TrainingPipeline, '_command', command)
    pipeline = queue.TrainingPipeline(args, state)
    try:
        pipeline.enqueue()
        job = state['training_jobs'][0]
        with pytest.raises(RuntimeError, match='selection interrupted'):
            pipeline.futures[0][1].result(timeout=10)
        # Deployment/default changes cannot rewrite an already enqueued job.
        args.candidates_per_recipe, args.training_parallelism = 1, 2
        pipeline.poll()
        pipeline.futures[0][1].result(timeout=10)
        pipeline.poll()
    finally:
        pipeline.close()
    assert job['training_seeds'] == [1, 2, 3, 4]
    assert [r['seed'] for r in seen] == [1, 2, 3, 4]
    assert [r['source_weight'].get('level_balance', 0) for r in seen] == [0, 0, .25, .25]
    assert calls.count('pack.log') == 1 and calls.count('train.log') == 4
    assert [c['recipe_family'] for c in job['timing']['candidates']] == ['A', 'A', 'B', 'B']
    assert [c['seed'] for c in job['timing']['candidates']] == [1, 2, 3, 4]
    assert all(c['selection_unseen_deck']['wins'] == 2 for c in job['timing']['candidates'])
    assert all(len(c['selection_unseen_deck']['tasks']) == 2 for c in job['timing']['candidates'])
    assert job['timing']['training_mode']['attempt_history'][0]['mode'] == 'serial'
    assert job['status'] == 'done'
    assert len(json.loads((tmp_path / 'selection_seed_ledger.json').read_text())['allocated']) == 1


def test_deploy14h_activation_rollback_and_watchdog_extraction(tmp_path):
    script = ROOT / 'scripts/qc_deploy14h.sh'
    source = script.read_text().split('# --- preflight:', 1)[0].replace('/home/newbiexvwu', str(tmp_path))
    functions = tmp_path / 'functions.sh'
    functions.write_text(source)
    logs = tmp_path / 'pvz-logs'
    logs.mkdir()
    result = subprocess.run(['bash', '-c', '''
source "$1"
tmux() { if [ "$1" = has-session ]; then return 1; fi; printf '%s\n' "$*" >> "$MODE_CHOICE.calls"; }
code_state() { :; }
start_qc
choose_mode rebalance
start_qc
SAVED=1; STARTED=1; rollback injected
''', 'test', str(functions)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 1, result.stderr
    calls = (logs / 'qc_deploy14h.mode.calls').read_text().splitlines()
    assert len(calls) == 3
    for call in (calls[0], calls[2]):
        assert 'dw4i2_rebalanced.json' in call
        assert '--selection-holdout-manifest' not in call
        assert '--candidates-per-recipe' not in call and '--training-parallelism' not in call
    assert 'dw4i2_rebalanced_p05.json' in calls[1]
    assert '--candidates-per-recipe 2' in calls[1]
    assert '--training-parallelism 1' in calls[1]
    assert '--selection-holdout-manifest' in calls[1]
    result = subprocess.run(['bash', '-c', 'source "$1"; watchdog_extract_start "$2"',
                              'test', str(ROOT / 'scripts/qc_watchdog.sh'), str(script)],
                             capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    subprocess.run(['bash', '-n'], input=result.stdout, text=True, check=True)
