#!/usr/bin/env python3
"""task49: full vs incremental packs, real deterministic two-epoch trainer comparison.

Use an isolated --out under artifacts/task49. Never infer parity from losses alone:
compare every input tensor byte, per-step losses, final parameters and source/split report.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pack_training_rows import assert_payload_equal, pack, pack_monolithic
from pvz_recipe import load_recipe, training_row_options, validate, write_recipe
from pvz_training_pack import archive_fingerprint, pack_options
import train_rollout_policy as trainer


def train_pair(root, data, contract, device='cpu', recipe=None):
    """The real trainer, identical RNG/kernel/split/optimizer, full two-epoch traces and weights."""
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    index = pack(data, contract, root / 'cache')
    full = pack_monolithic(data, contract, root / 'cache', out=root / 'full.npz')
    assert_payload_equal(trainer.read_pack(full), trainer.read_pack(index))
    traces, models, reports = [], [], []
    try:
        for name, path in [('full', full), ('incremental', index)]:
            out = root / name
            settings = dict(schema_version=1, kind='training', data=str(data), labels=contract['labels'],
                          out=str(out), device=device, threads=1, epochs=2, width=8, seed=7,
                          wait_keep=contract['wait_keep'], obs=contract['obs'],
                          packet_head='identity', pool='meanmax', row_pool=True,
                          packet_balance=.5, cell_soft_weight=3., spread_weight=8.,
                          label_weight=contract['label_weight'], lr_schedule='cosine',
                          source_row_report=True, batch_size=32, validation_batch_size=32,
                          validation=dict(seed=0, fraction=.34))
            if recipe is not None:
                settings = dict(recipe, out=str(out), data=str(data), labels=contract['labels'],
                                device=device, epochs=2)
            settings = validate(settings, 'training')
            if pack_options(**training_row_options(settings), sources=contract['sources']) != contract:
                raise AssertionError('trainer recipe and compared pack row options differ')
            recipe_path = root / f'{name}.json'
            write_recipe(recipe_path, settings)
            losses = []
            def load(_data, **kwargs):
                return trainer.read_pack(path, data, kwargs.get('row_observer'))
            with patch.object(sys, 'argv', ['trainer', '--recipe', str(recipe_path)]), \
                 patch.object(trainer, 'load', load):
                trainer.main(step_observer=lambda step, loss: losses.append(loss))
            traces.append(losses)
            models.append(torch.load(out / 'model.pt', map_location='cpu', weights_only=False))
            reports.append((out / 'source_rows.json').read_bytes())
        assert traces[0] == traces[1] and traces[0]
        assert len(models[0]['history']) == len(models[1]['history']) == 2
        for key, value in models[0]['state_dict'].items():
            assert torch.equal(value.view(torch.uint8), models[1]['state_dict'][key].view(torch.uint8)), key
        assert reports[0] == reports[1]
        for a, b in zip(models[0]['history'], models[1]['history']):
            assert {k: v for k, v in a.items() if k != 'seconds'} == {k: v for k, v in b.items() if k != 'seconds'}
        return dict(passed=True, epochs=2, steps=len(traces[0]), max_loss_error=0,
                    model_tensors=len(models[0]['state_dict']), source_rows_identical=True,
                    rows=len(trainer.read_pack(index)[2]), device=device,
                    losses_sha256=hashlib.sha256(np.asarray(traces[0], np.float64).tobytes()).hexdigest(),
                    source_rows_sha256=hashlib.sha256(reports[0]).hexdigest())
    finally:
        torch.use_deterministic_algorithms(deterministic)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if not args.recipe.is_absolute() or not args.out.is_absolute():
        parser.error('use absolute paths')
    args.out.mkdir(parents=True, exist_ok=True)
    recipe = load_recipe(args.recipe, 'training')
    recipe['seed'] = recipe['seed'] if recipe['seed'] is not None else 1
    data = Path(recipe['data'])
    contract = pack_options(**training_row_options(recipe),
                            sources=archive_fingerprint(data, recipe['labels']))
    evidence = train_pair(args.out, data, contract, args.device, recipe)
    (args.out / 'parity.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence), flush=True)


if __name__ == '__main__':
    main()
