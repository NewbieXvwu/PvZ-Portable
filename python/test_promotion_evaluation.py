"""Frozen real-runner contracts, resumability and corrupted episode rejection (no engines)."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pvz_promotion import SEED_POOLS_V2, initial_panel
from test_curriculum_pipeline import setup
import promotion_gate as gate


V5_UNITS = 80      # 40 tasks x 2 sides


def arguments(root):
    common, _ = setup(root)
    candidate = root / 'candidate.pt'
    candidate.write_bytes(b'candidate')
    return SimpleNamespace(candidate=candidate, incumbent=common.rollout_policy,
        source_panel=root / 'panel.json', executable=common.executable, value=common.value,
        resource_dir=common.resource_dir, workers=1, out=root / 'evaluation',
        candidates={'A': str(candidate)}, calibration=initial_panel(gate.read(root / 'panel.json')),
        eval_units=1, seed_block=None, seed_ledger=root / 'gate_ledger.json')


def test_systemic_unit_failure_keeps_engine_diagnostics(tmp_path):
    args = arguments(tmp_path)
    task = dict(task_id='failed', seeds=[1])
    marker = 'Inconsistency detected by ld.so: Assertion failed'

    def crash(command, **kwargs):
        kwargs['stdout'].write(marker + '\n')
        return SimpleNamespace(returncode=127)

    with (tmp_path / 'lock').open('w') as lock, \
         patch.object(gate.subprocess, 'run', side_effect=crash):
        with pytest.raises(RuntimeError, match='Inconsistency detected by ld.so'):
            gate._run_unit(args, lock, args.source_panel, task, 'candidate', args.candidate)


# ---- ledger allocation, exhaustion, resume ----


def test_seed_pool_exhaustion_aborts_instead_of_wrapping():
    import pvz_promotion
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        args = arguments(root)
        tiny = (31024, 31024 + 1920 + 100)
        # A single-pool registration still has to abort, not continue somewhere else.
        with patch.object(pvz_promotion, 'SEED_POOLS_V2', (tiny,)):
            gate.allocate_seed_block(args.seed_ledger, 1920)
            with pytest.raises(gate.SeedPoolExhausted, match='refusing to reuse'):
                gate.allocate_seed_block(args.seed_ledger, 1920)
            args.out = root / 'evaluation3'
            with patch.object(gate.subprocess, 'run', side_effect=AssertionError('gate must not run')):
                with pytest.raises(gate.SeedPoolExhausted):
                    gate.evaluate(args)
        assert not (args.out / 'registration.json').exists()


def test_ledger_refuses_a_different_pool_list_and_overlaps():
    with tempfile.TemporaryDirectory() as folder:
        ledger = Path(folder) / 'gate_ledger.json'
        ledger.write_text(json.dumps(dict(pools=[[31024, 65536], [200_000_000, 201_000_000]],
                                          allocated=[[31024, 32560]])))
        with pytest.raises(ValueError, match='was created for pools'):
            gate.allocate_seed_block(ledger, 1920)
        ledger.write_text(json.dumps(dict(pools=[list(p) for p in SEED_POOLS_V2],
                                          allocated=[[31024, 32560], [32000, 34000]])))
        with pytest.raises(ValueError, match='overlapping'):
            gate.allocate_seed_block(ledger, 1920)


def test_two_hundred_fifty_rounds_are_pairwise_disjoint_across_both_windows():
    """Dry-run shape of task88 step 4: 250 rounds (1920 seeds each) from a copy."""
    import pvz_promotion
    with tempfile.TemporaryDirectory() as folder:
        ledger = Path(folder) / 'gate_ledger.json'
        first = pvz_promotion.SEED_POOLS_V2[0]
        # Start near the end of the first window, as the live ledger is on 2026-10-26.
        ledger.write_text(json.dumps(dict(pools=[list(p) for p in SEED_POOLS_V2],
                                          allocated=[[first[0], 65000]])))
        blocks = [gate.allocate_seed_block(ledger, 1920) for _ in range(250)]
        spans = sorted(blocks)
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
        assert len({low for low, _ in blocks}) == 250
        assert sum(high - low for low, high in blocks) == 250 * 1920
        assert any(low >= pvz_promotion.SEED_POOLS_V2[1][0] for low, _ in blocks)
        assert gate.read(ledger)['allocated'][-1] == list(blocks[-1])
