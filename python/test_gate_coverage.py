"""task247: 门禁卡组整局覆盖、低倍率与存量格迁移。"""
import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]

import curriculum_queue as queue
import make_gate_coverage_panel as panel
import make_multiplier_tasks as variants
import make_user_roof_panel as roof
from pvz_promotion import SEED_POOLS_V2, START_SEED_POOL
from pvz_tasks import profile_for_deck, terrain_covered


def coverage_tasks():
    bases = panel.tasks() + roof.tasks()
    tables = {task['task_id']: task for task in bases}
    families = tuple(f for f in variants.FAMILIES if f[0] in variants.GATE_COVERAGE_FAMILIES)
    with patch.object(variants, 'FAMILIES', families):
        return bases + variants.build_variants({'gate_coverage': tables, 'roof_user': tables})


def loaded(tmp_path):
    path = tmp_path / 'coverage.json'
    path.write_text(json.dumps({'tasks': coverage_tasks()}))
    # Other source panels are local production artifacts and outside this test.
    with patch.object(queue, 'source_paths', return_value={}):
        return queue.load_panel(path)


def test_gate_coverage_loads_exact_gate_decks_and_low_ladders(tmp_path):
    tasks, values, names, manifests = loaded(tmp_path)
    state = queue.make_state(tasks, names, {}, {}, 'coverage.json')
    gate = json.loads((ROOT / 'configs/gate_v8_panel.json').read_text())['tasks']
    bases = panel.tasks() + roof.tasks()
    assert len(state['cells']) == len(bases) == 9
    assert values == [1.0] + [m for m, _ in variants.MULTIPLIERS]
    for task in bases:
        assert any(t['level'] == task['level'] and t['deck'] == json.loads(json.dumps(task['deck']))
                   for t in gate)
        assert terrain_covered(task['level'], task['deck'])
        assert profile_for_deck(tuple(task['deck'])).seed_slot_count == len(task['deck'])
        cell = state['cells'][task['task_id']]
        assert cell['ladder'] == variants.GATE_COVERAGE_LADDERS[queue.family_of(task['task_id'])]
        assert queue.cell_level(cell, values) == (.25 if task['task_id'] == 'pool_user_l29_user1' else .5)
        for multiplier in cell['ladder']:
            task_id = queue.task_id_for(task['task_id'], multiplier, tasks, names)
            assert tasks[task_id]['zombie_count_multiplier'] == multiplier
            assert task_id in manifests


def test_gate_coverage_seeds_isolate_gate_and_preserve_issued_roof_bands():
    tasks = coverage_tasks()
    used = set()
    for task in tasks:
        seeds = set(task['seeds'])
        assert len(seeds) == 2048
        assert not used & seeds
        assert not any(low <= seed < high for low, high in (*SEED_POOLS_V2, START_SEED_POOL)
                       for seed in seeds)
        used.update(seeds)
    table = {task['task_id']: task for task in tasks}
    for level in (47, 49):
        for index, (_, suffix) in enumerate(variants.MULTIPLIERS):
            task = table[f'roof_user_l{level}_user_{suffix}']
            assert task['seeds'][0] == 84_100_000 + index * 100_000 + (level - 47) * 2048


def test_gate_coverage_migrates_roof_ladder_without_reusing_offsets(tmp_path):
    tasks, values, names, _ = loaded(tmp_path)
    base = 'roof_user_l47_user'
    state = dict(round=123, cells={base: dict(level=47, mult_index=0, last_round=116,
                 windows={'1.0': dict(recent=[0] * 8, wins=0, games=8)})},
                 offsets={base: 72, base + '_m125': 48})
    before = copy.deepcopy(state)
    queue.reconcile_cells(state, tasks, names, {})
    assert queue.cell_level(state['cells'][base], values) == 1.
    assert state['offsets'] == before['offsets']
    assert state['cells'][base]['windows'] == before['cells'][base]['windows']
    queue.adjust(state['cells'], values)
    assert queue.cell_level(state['cells'][base], values) == .75
    state['cells'][base]['windows']['0.75'] = dict(recent=[0] * 8, wins=0, games=8)
    queue.adjust(state['cells'], values)
    assert queue.cell_level(state['cells'][base], values) == .5
    queue.reconcile_cells(state, tasks, names, {})
    assert queue.cell_level(state['cells'][base], values) == .5


def test_gate_coverage_new_cells_rotate_inside_existing_multiplier_budget(tmp_path):
    tasks, values, names, _ = loaded(tmp_path)
    state = queue.make_state(tasks, names, {}, {}, 'coverage.json')
    # Production has 45 multiplier cells, including the two roof_user cells.
    cells = {f'old_{i}': dict(level=7, mult_index=0, last_round=120, windows={}) for i in range(43)}
    cells.update(state['cells'])
    assert len(cells) == 52
    seen = set()
    # 768 - 192 specialists - 192 random - 192 demo - 36 drills = 156.
    for round_index in range(124, 133):
        picks = queue.allocate(cells, values, budget=156, per_group=48, round_index=round_index)
        assert sum(games for _, _, games in picks) <= 156
        for base, _, _ in picks:
            cells[base]['last_round'] = round_index
            seen.add(base)
    assert set(cells) <= seen
