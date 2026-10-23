#!/usr/bin/env python3
"""Compare original training CLI and recipe losses on the same full historical corpus.

No shortened schedule, subset, or max_steps override: interrupt after observing
200 losses, retaining the original full-run cosine length. Only outputs, caches,
and feature-preparation worker count are relocated/limited. One trainer at a time.
"""
import argparse
import ast
from collections import Counter
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import shutil
import tempfile
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]
from pvz_recipe import load_recipe
from pvz_metrics import Metrics


class TraceComplete(Exception):
    pass


def foreign_trainers():
    found = []
    for entry in Path('/proc').glob('[0-9]*/cmdline'):
        if int(entry.parent.name) == os.getpid():
            continue
        try:
            args = entry.read_bytes().split(b'\0')
        except OSError:
            continue
        if any(Path(v.decode(errors='replace')).name.startswith(('train_rollout_policy', 'train_value'))
               and v.endswith(b'.py') for v in args):
            found.append(int(entry.parent.name))
    return found


def legacy_module(revision, parse_only=False):
    source = subprocess.check_output(['git','-C',str(ROOT),'show',f'{revision}:scripts/train_rollout_policy.py'], text=True)
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(v, ast.Name) and v.id == 'ROOT' for v in node.targets):
            node.value = ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(ROOT))],keywords=[])
        if isinstance(node, ast.FunctionDef) and node.name == 'main':
            if parse_only:
                for index, statement in enumerate(node.body):
                    if isinstance(statement, ast.Assign) and any(isinstance(v, ast.Name) and v.id == 'args' for v in statement.targets):
                        node.body = node.body[:index+1] + [ast.Return(value=ast.Name(id='args',ctx=ast.Load()))]
                        break
                continue
            for statement in ast.walk(node):
                if isinstance(statement, ast.For) and isinstance(statement.target, ast.Name) and statement.target.id == 'start':
                    # Observer at the same point as the recipe trainer: immediately after loss calculation.
                    statement.body.insert(1, ast.parse('loss_observer(completed_steps + 1, float(loss.detach()))').body[0])
    scope = {'__file__':str(ROOT/'scripts/train_rollout_policy.py'), '__name__':'historical_training'}
    exec(compile(ast.fix_missing_locations(tree), '<historical-training>', 'exec'), scope)
    return scope


def audit_source(recipe):
    path = Path(recipe['provenance']['source_path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == recipe['provenance']['source_sha256']
    record = json.loads(path.read_text())
    parser = legacy_module('6fdee2af', parse_only=True)
    previous = sys.argv
    try:
        sys.argv = record['command'][1:]
        original = vars(parser['main']())
    finally:
        sys.argv = previous
    relocated = {'out', 'cache_dir', 'source_row_report'}
    checked = {}
    for name, value in original.items():
        if name in ('ema', 'soft_weight'):
            assert value == 0, f'retired behavior was active: {name}'
            checked[name] = 'retired; original was zero'
            continue
        if name in relocated:
            assert str(value or '') == recipe['provenance']['original_' + name]
            if name == 'source_row_report':
                assert bool(value) == recipe[name]
            else:
                assert Path(recipe[name]).is_relative_to(ROOT)
            checked[name] = 'writable path relocated' if name != 'source_row_report' else 'report moved with out'
            continue
        if name == 'labels':
            value = value.split(',')
        elif name == 'label_weight':
            value = {k:float(v) for k,v in (item.split('=') for item in value.split(',') if item)}
        elif isinstance(value, Path):
            value = str(value)
        assert value == recipe[name], (name, value, recipe[name])
        checked[name] = 'equal'
    return checked


def seed_private_cache(recipe):
    """Reuse verified historical cache bytes without writing to their source tree."""
    source = Path(recipe['provenance']['original_cache_dir'])
    target = Path(recipe['cache_dir'])
    copied = size = 0
    for folder in ('v2', 'identity_scene', 'packets', 'cells'):
        base = source/folder
        if not base.is_dir():
            continue
        (target/folder).mkdir(parents=True,exist_ok=True)
        for path in base.glob('*.npz'):
            destination = target/folder/path.name
            if destination.exists():
                continue
            with tempfile.NamedTemporaryFile(dir=target/folder,suffix='.seed',delete=False) as handle:
                temporary = Path(handle.name)
            try:
                shutil.copyfile(path,temporary)
                try:
                    # Link our private COPY atomically; never hard-link the source cache.
                    os.link(temporary,destination)
                    copied += 1
                    size += path.stat().st_size
                except FileExistsError:
                    pass
            finally:
                temporary.unlink(missing_ok=True)
    return dict(copied_files=copied,copied_bytes=size)


class QuietMetrics:
    def __init__(self, *args, **kwargs): pass
    def start(self): return self
    def update(self, **kwargs): pass
    def close(self): pass


def verify(recipe_path, out):
    foreign = foreign_trainers()
    if foreign:
        raise RuntimeError(f'GPU comparison requires no other trainers: {foreign}')
    import torch
    import train_rollout_policy as current
    torch.set_num_threads(1)
    recipe = load_recipe(recipe_path, 'training')
    field_audit = audit_source(recipe)
    cache_seed = seed_private_cache(recipe)
    print(json.dumps(dict(cache_seed=cache_seed)),flush=True)
    record_path = Path(recipe['provenance']['source_path'])
    assert hashlib.sha256(record_path.read_bytes()).hexdigest() == recipe['provenance']['source_sha256']
    record = json.loads(record_path.read_text())
    baseline = legacy_module('6fdee2af')
    baseline['Metrics'] = QuietMetrics
    original_metrics, original_load, original_argv = current.Metrics, current.load, sys.argv
    current.Metrics = QuietMetrics
    command = record['command'][1:]
    command[0] = str(ROOT/'scripts/train_rollout_policy.py')
    for flag, value in {'--out':str(out/'legacy'), '--cache-dir':recipe['cache_dir'],
                        '--precompute-workers':'1'}.items():
        command[command.index(flag)+1] = value
    if '--source-row-report' in command:
        command[command.index('--source-row-report')+1] = str(out/'legacy/source_rows.json')
    (out/'legacy').mkdir(parents=True,exist_ok=True)
    payload = None
    payload_key = None
    observations = []
    legacy_signature = inspect.signature(baseline['load'])
    current_signature = inspect.signature(current.load)
    def freeze(value):
        if isinstance(value, dict):
            return tuple(sorted((key,freeze(item)) for key,item in value.items()))
        if isinstance(value, (set,frozenset)):
            return tuple(sorted(freeze(item) for item in value))
        if isinstance(value, (list,tuple)):
            return tuple(freeze(item) for item in value)
        return str(value) if isinstance(value, Path) else value
    def cached_load(mode, *args, **kwargs):
        nonlocal payload, payload_key
        signature = legacy_signature if mode == 'legacy' else current_signature
        request = signature.bind(*args, **kwargs)
        request.apply_defaults()
        fields = dict(request.arguments)
        observer = fields.pop('row_observer', None)
        # Compare every semantic loader argument across both interfaces. Cache and
        # representation controls disappeared with task33; they do not select rows.
        for field in ('cache_dir', 'read_cache_dir', 'precompute_workers', 'dense_boards'):
            fields.pop(field, None)
        key = freeze(fields)
        if payload is None:
            payload_key = key
            from pack_training_rows import pack
            from pvz_training_pack import archive_fingerprint, pack_options
            data = fields.pop('data')
            options = pack_options(**fields, sources=archive_fingerprint(data, fields['labels']))
            packed = out / 'rows.npz'
            # A restart never trusts an incomplete pack; build to a temporary name,
            # then publish only after np.savez has closed it successfully.
            temporary = out / 'rows.building.npz'
            pack(data, options, Path(recipe['cache_dir']), 1,
                 Path(recipe['read_cache_dir']) if recipe['read_cache_dir'] else None, temporary)
            os.replace(temporary, packed)
            def capture(path, rows):
                observations.append((path, rows.copy()))
                if observer is not None:
                    observer(path, rows)
            payload = current.read_pack(packed, data, capture)
        else:
            assert key == payload_key, 'recipe and legacy requested different training rows'
            if observer is not None:
                for path, rows in observations:
                    observer(path, rows)
        result = list(payload)
        result[5] = result[5].clone()  # each run starts before in-place weighting
        if mode == 'legacy':
            # Storage-only adapters reconstruct the historical tensor dtypes and
            # dense value matrix. No historical loss, weights or optimizer is edited.
            import numpy as np
            result[2] = result[2].float()
            result[3] = result[3].long()
            result[4] = result[4].long()
            values = payload[7]
            dense = np.full((len(result[2]), current.PACKETS + 1), np.nan, np.float32)
            for start in range(0, len(dense), 65536):
                dense[start:start + 65536] = values[np.arange(start, min(start + 65536, len(dense)))].numpy()
            result[7] = torch.from_numpy(dense)
            class HistoricalPackets:
                def __init__(self, store): self.store = store
                def __getitem__(self, index): return self.store[index]
                def numel(self): return len(result[2]) * current.PACKETS * 4
                def element_size(self): return 4
            if recipe['packet_head'] == 'identity':
                result[-1] = HistoricalPackets(result[-1])
        return tuple(result)
    baseline['load'] = lambda *a, **kw: cached_load('legacy', *a, **kw)
    current.load = lambda *a, **kw: cached_load('recipe', *a, **kw)
    traces = {}
    sampler = Metrics('recipe_parity_'+recipe_path.stem, training=True).start()
    sampler.update(step=0, phase='loading')
    try:
        for name in ('legacy','recipe'):
            values = []
            def observe(step, loss):
                if foreign_trainers():
                    raise RuntimeError('another trainer started; stop only this comparison')
                values.append(loss)
                sampler.update(step=step, phase=name)
                if step % 20 == 0:
                    print(json.dumps(dict(phase=name,step=step,loss=loss)),flush=True)
                if step == 200:
                    raise TraceComplete
            sys.argv = command if name == 'legacy' else [str(ROOT/'scripts/train_rollout_policy.py'),
                '--recipe',str(recipe_path),'--out',str(out/'recipe'),'--precompute-workers','1']
            started = time.monotonic()
            try:
                if name == 'legacy':
                    baseline['loss_observer'] = observe
                    baseline['main']()
                else:
                    current.main(step_observer=observe)
            except TraceComplete:
                pass
            assert len(values) == 200, f'{name}: fewer than 200 steps'
            traces[name] = dict(losses=values,seconds=time.monotonic()-started)
            gc.collect()
            torch.cuda.empty_cache()
        errors = [abs(a-b) for a,b in zip(traces['legacy']['losses'],traces['recipe']['losses'])]
        result = dict(recipe=str(recipe_path), recipe_sha256=hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
            legacy_revision='6fdee2af', source_command_sha256=recipe['provenance']['source_sha256'],
            steps=200,rows=len(payload[2]),archives=len(observations),
            loader_call_sha256=hashlib.sha256(repr(payload_key).encode()).hexdigest(),
            labels=dict(Counter(path.parent.name for path,_ in observations)),
            full_corpus=True,original_schedule=True,shared_storage='task33 compact pack; historical tensor views restored',field_audit=field_audit,cache_seed=cache_seed,device=recipe['device'],seed=recipe['seed'],
            maximum_loss_error=max(errors),passed=max(errors)<=1e-6,
            torch_version=str(torch.__version__),cuda_version=torch.version.cuda,traces=traces)
        (out/'parity.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ('traces','labels')}),flush=True)
        assert result['passed']
    finally:
        sampler.close()
        current.Metrics, current.load, sys.argv = original_metrics, original_load, original_argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recipe',type=Path)
    parser.add_argument('out',type=Path)
    args = parser.parse_args()
    if not args.recipe.is_absolute() or not args.out.is_absolute():
        parser.error('all paths must be absolute')
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'parity.json').unlink(missing_ok=True)
    verify(args.recipe,args.out)


if __name__ == '__main__': main()
