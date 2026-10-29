"""Versioned recipes. Unknown fields (including nested ones) are errors.

Paths in a recipe file are relative to the repository root and are resolved when the
recipe is loaded, so a recipe means the same thing in every worktree and on every
machine. Defaults live here, never in a second argparse training interface. Saved
recipe.resolved.json is the run contract and holds the resolved absolute paths.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

SCHEMA_VERSION = 1
TRAIN_DEFAULTS = dict(
    epochs=12, device='cpu', threads=1, precompute_workers=1,
    wait_keep=1.0, seed=None, width=64, arch='cnn', label_weight={}, source_weight={}, packet_balance=0.0,
    pool='mean', row_pool=False, packet_head='slot', aux_weight=0.0,
    aux_horizon=8, context=False, obs='v1', soft_temperature=.05,
    cell_soft_weight=0.0, spread_weight=0.0, spread_cap=5.0,
    demo_row_weight=1.0, demo_start_row_weight=1.0, llm_demo_import=None, llm_demo_results=[1],
    save_epochs=0, lr_schedule='constant', init=None, cache=True,
    cache_dir=None, read_cache_dir=None, dense_boards=False, row_weights=None,
    source_row_report=False, max_steps=None, micro_batch_size=None, train_row_fraction=1.0, batch_size=2048, validation_batch_size=4096,
    optimizer=dict(name='AdamW', lr=.002, weight_decay=.0001, betas=[.9, .999], eps=1e-8, amsgrad=False),
    validation=dict(seed=0, fraction=.1), provenance={})

PANEL_DEFAULTS = dict(version='v1',
    generate=False, generate_pressure=False, pressure=False, pressure_seeds=64,
    pressure_tasks='', tasks='', day_seeds=128, heldout_seeds=64, night_seeds=128,
    pool_seeds=40, task_seed_overrides={}, workers=2, provenance={})

_ROOT: Path | None = None


def repository_root() -> Path:
    """The checkout that owns the corpus, the build and the production outputs.

    Agent worktrees live at <checkout>/.claude/worktrees/<name> (AGENTS.md §1), so a run
    started from one still describes the same artifacts/ tree the mainline uses; a
    normal checkout is its own root. No switch: one convention, resolved on load.
    """
    global _ROOT
    if _ROOT is None:
        root = Path(__file__).resolve().parents[1]
        if root.parent.name == 'worktrees' and root.parent.parent.name == '.claude':
            root = root.parent.parent.parent
        _ROOT = root
    return _ROOT


def resolve_paths(document):
    """Resolve every path field against the repository root; absolute values stay as given."""
    root = repository_root()
    result = copy.deepcopy(document)
    for name in PATH_FIELDS:
        value = result.get(name)
        if isinstance(value, str) and value and not Path(value).is_absolute():
            result[name] = str(root / value)
    return result


PATH_FIELDS = ('data', 'out', 'init', 'cache_dir', 'read_cache_dir', 'row_weights', 'llm_demo_import',
               'manifest', 'rollout_policy', 'value', 'executable', 'resource_dir',
               'day_manifest', 'perturb_manifest', 'night_manifest', 'day_orig_manifest',
               'pool_manifest', 'pressure_manifest')


def _object(value, allowed, context):
    if not isinstance(value, dict):
        raise ValueError(f'{context}: expected object')
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f'{context}: unknown fields: {sorted(unknown)}')


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name}: expected integer >= {minimum}')


def _number(value, name, minimum=0, maximum=None, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum or (
            maximum is not None and value > maximum) or (positive and value <= 0):
        raise ValueError(f'{name}: invalid number {value!r}')


def validate(document, kind=None):
    if not isinstance(document, dict):
        raise ValueError('recipe: expected object')
    if document.get('schema_version') != SCHEMA_VERSION or type(document.get('schema_version')) is not int:
        raise ValueError(f'unsupported schema_version: {document.get("schema_version")}')
    kind = kind or document.get('kind')
    if kind not in ('training', 'panel') or document.get('kind') != kind:
        raise ValueError(f'expected recipe kind {kind!r}')
    defaults = TRAIN_DEFAULTS if kind == 'training' else PANEL_DEFAULTS
    required = {'data', 'labels', 'out'} if kind == 'training' else {
        'out', 'manifest', 'value', 'executable', 'resource_dir', 'day_manifest',
        'perturb_manifest', 'night_manifest', 'day_orig_manifest', 'pool_manifest', 'pressure_manifest'}
    allowed = set(defaults) | required | {'schema_version', 'kind'}
    if kind == 'panel':
        allowed.add('rollout_policy')
    _object(document, allowed, 'recipe')
    missing = required - document.keys()
    if missing:
        raise ValueError(f'missing fields: {sorted(missing)}')
    result = {**copy.deepcopy(defaults), **copy.deepcopy(document)}
    for name in required - {'labels'}:
        if result[name] is None:
            raise ValueError(f'{name}: required field cannot be null')
    for name in PATH_FIELDS:
        value = result.get(name)
        # A recipe file names paths relative to the repository root; they become absolute on
        # load. Absolute values are still accepted for machine-local data outside the repo
        # (the game resources) and for an already resolved recipe.
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f'{name}: expected a path string')
    # Provenance is descriptive, never used as training configuration.
    _object(result['provenance'], {'source_path', 'source_sha256', 'source_revision',
                                 'original_out', 'original_cache_dir', 'original_source_row_report', 'notes'}, 'provenance')
    if not all(isinstance(v, str) for v in result['provenance'].values()):
        raise ValueError('provenance: expected string values')
    if kind == 'training':
        for name in ('epochs', 'width', 'aux_horizon', 'batch_size', 'validation_batch_size', 'threads', 'precompute_workers'):
            _integer(result[name], name, 1)
        _integer(result['save_epochs'], 'save_epochs')
        for name in ('seed', 'max_steps', 'micro_batch_size'):
            if result[name] is not None:
                _integer(result[name], name, 0 if name == 'seed' else 1)
        for name in ('row_pool', 'context', 'cache', 'dense_boards', 'source_row_report'):
            if type(result[name]) is not bool:
                raise ValueError(f'{name}: expected boolean')
        for name, choices in dict(pool=('mean', 'meanmax'), packet_head=('slot', 'identity'),
                                  obs=('v1', 'v2lite', 'v2', 'v2fog', 'v2fogstack4', 'v2bungee', 'v2sleep',
                                       'v2timer'), lr_schedule=('constant', 'cosine')).items():
            if result[name] not in choices:
                raise ValueError(f'{name}: expected one of {choices}')
        if not isinstance(result['device'], str) or not result['device']:
            raise ValueError('device: expected nonempty string')
        if not isinstance(result['arch'], str) or not result['arch'] or (
                result['arch'] != 'cnn' and ':' not in result['arch']):
            raise ValueError('arch: expected cnn or module:factory')
        labels = result['labels']
        if not isinstance(labels, list) or not labels or not all(isinstance(v, str) and v and ',' not in v for v in labels):
            raise ValueError('labels: expected nonempty list of label names')
        if len(labels) != len(set(labels)):
            raise ValueError('labels: duplicate label')
        _object(result['label_weight'], set(labels), 'label_weight')
        for name, weight in result['label_weight'].items():
            _number(weight, f'label_weight.{name}')
        _object(result['source_weight'], {'human', 'teacher', 'llm_gpt_pro', 'llm_agent', 'level_balance',
                                          'level_balance_power', 'level_balance_max_multiplier'},
                'source_weight')
        for name, weight in result['source_weight'].items():
            _number(weight, f'source_weight.{name}')
        if 'level_balance' in result['source_weight']:
            _number(result['source_weight']['level_balance'], 'source_weight.level_balance',
                    positive=True, maximum=1)
        for name in ('level_balance_power', 'level_balance_max_multiplier'):
            if name in result['source_weight']:
                if 'level_balance' not in result['source_weight']:
                    raise ValueError(f'source_weight.{name}: requires source_weight.level_balance')
                if name == 'level_balance_power':
                    # 1 is the exact inverse of retained rows, 0 leaves the weights untouched.
                    _number(result['source_weight'][name], f'source_weight.{name}', maximum=1)
                else:
                    _number(result['source_weight'][name], f'source_weight.{name}', positive=True)
        _number(result['wait_keep'], 'wait_keep', maximum=1, positive=True)
        _number(result['train_row_fraction'], 'train_row_fraction', maximum=1, positive=True)
        outcomes = result['llm_demo_results']
        if (not isinstance(outcomes, list) or not outcomes
                or any(type(value) is not int or value not in (0, 1, 2) for value in outcomes)
                or len(set(outcomes)) != len(outcomes)):
            raise ValueError('llm_demo_results: expected unique results from [0, 1, 2]')
        for name in ('packet_balance', 'aux_weight', 'cell_soft_weight', 'spread_weight',
                     'demo_row_weight', 'demo_start_row_weight'):
            _number(result[name], name)
        for name in ('soft_temperature', 'spread_cap'):
            _number(result[name], name, positive=True)
        opt = {**TRAIN_DEFAULTS['optimizer'], **result['optimizer']} if isinstance(result['optimizer'], dict) else result['optimizer']
        _object(opt, set(TRAIN_DEFAULTS['optimizer']), 'optimizer')
        if opt['name'] != 'AdamW':
            raise ValueError('optimizer.name: only AdamW is supported')
        for name in ('lr', 'eps'):
            _number(opt[name], f'optimizer.{name}', positive=True)
        _number(opt['weight_decay'], 'optimizer.weight_decay')
        if type(opt['amsgrad']) is not bool:
            raise ValueError('optimizer.amsgrad: expected boolean')
        if not isinstance(opt['betas'], list) or len(opt['betas']) != 2:
            raise ValueError('optimizer.betas: expected two numbers')
        for value in opt['betas']:
            _number(value, 'optimizer.betas', maximum=1)
            if value == 1:
                raise ValueError('optimizer.betas: must be below 1')
        result['optimizer'] = opt
        split = result['validation']
        _object(split, {'seed', 'fraction'}, 'validation')
        result['validation'] = {**TRAIN_DEFAULTS['validation'], **split}
        _integer(result['validation']['seed'], 'validation.seed')
        _number(result['validation']['fraction'], 'validation.fraction', maximum=1, positive=True)
        if result['validation']['fraction'] >= 1:
            raise ValueError('validation.fraction: must be below 1')
        if (result['row_weights'] or result['source_row_report']) and not result['cache']:
            raise ValueError('row_weights/source_row_report require cache')
    else:
        if result['version'] not in ('v1', 'v2'):
            raise ValueError('version: expected v1 or v2')
        for name in ('generate', 'generate_pressure', 'pressure'):
            if type(result[name]) is not bool:
                raise ValueError(f'{name}: expected boolean')
        for name in ('workers', 'pressure_seeds', 'day_seeds', 'heldout_seeds', 'night_seeds', 'pool_seeds'):
            _integer(result[name], name, 1)
        if result['workers'] > 4:
            raise ValueError('workers: panel maximum is 4')
        for name in ('tasks', 'pressure_tasks'):
            if not isinstance(result[name], str):
                raise ValueError(f'{name}: expected comma-separated string')
        if not isinstance(result['task_seed_overrides'], dict):
            raise ValueError('task_seed_overrides: expected object')
        for name, value in result['task_seed_overrides'].items():
            _integer(value, f'task_seed_overrides.{name}', 1)
    return result


def load_recipe(path, kind=None):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('recipe path must be absolute')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate JSON field: {key}')
            result[key] = value
        return result
    return resolve_paths(validate(json.loads(path.read_text(), object_pairs_hook=unique), kind))


def write_recipe(path, document):
    document = validate(document)
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('recipe path must be absolute')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + '\n')
    return document


def training_row_options(recipe, labels=None):
    """The row contract shared by training and curriculum packing."""
    return dict(labels=set(recipe['labels'] if labels is None else labels),
                wait_keep=recipe['wait_keep'], label_weight=recipe['label_weight'],
                with_context=recipe['context'], obs=recipe['obs'],
                cells=bool(recipe['cell_soft_weight']),
                aux_horizon=recipe['aux_horizon'] if recipe['aux_weight'] else 0,
                packet_head=recipe['packet_head'],
                **({'source_weight': recipe['source_weight']} if recipe.get('source_weight') else {}),
                **({key: recipe[key] for key in ('llm_demo_import', 'llm_demo_results')}
                   if recipe.get('llm_demo_import') else {}))


def training_args(parser, argv=None):
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--out', type=Path, help='override output directory')
    parser.add_argument('--device', help='override compute device')
    parser.add_argument('--threads', type=int, help='override torch CPU threads')
    parser.add_argument('--precompute-workers', type=int, help='override feature preparation workers')
    cli = parser.parse_args(argv)
    try:
        resolved = load_recipe(cli.recipe, 'training')
        for name in ('out', 'device', 'threads', 'precompute_workers'):
            if getattr(cli, name) is not None:
                resolved[name] = str(cli.out) if name == 'out' else getattr(cli, name)
        resolved = validate(resolved, 'training')
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args = argparse.Namespace(**resolved)
    # Script-specific execution controls are not part of the frozen training recipe.
    for name, value in vars(cli).items():
        if not hasattr(args, name):
            setattr(args, name, value)
    for name in PATH_FIELDS:
        if getattr(args, name, None) is not None:
            setattr(args, name, Path(getattr(args, name)))
    args.labels = ','.join(resolved['labels'])
    args.label_weight = ','.join(f'{k}={v}' for k, v in resolved['label_weight'].items())
    args.source_row_report = args.out / 'source_rows.json' if resolved['source_row_report'] else None
    args.recipe = cli.recipe
    args.resolved_recipe = resolved
    args.recipe_sha256 = hashlib.sha256(cli.recipe.read_bytes()).hexdigest()
    return args


def materialize_recipe(base, path, **updates):
    """Freeze a study's corpus/output variant as a reviewable recipe before launch."""
    document = load_recipe(base)
    document.update({key: str(value) if isinstance(value, Path) else value for key, value in updates.items()})
    document = validate(document)
    path = Path(path)
    if path.exists():
        if load_recipe(path) != document:
            raise ValueError(f'cannot overwrite a different frozen recipe: {path}')
    else:
        write_recipe(path, document)
    return path


def panel_args(parser, argv=None):
    parser.add_argument('--recipe', type=Path, required=True)
    for name in ('out', 'rollout-policy', 'executable', 'resource-dir'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--workers', type=int)
    cli = parser.parse_args(argv)
    try:
        resolved = load_recipe(cli.recipe, 'panel')
        for name in ('out', 'rollout_policy', 'executable', 'resource_dir', 'workers'):
            value = getattr(cli, name)
            if value is not None:
                resolved[name] = str(value) if isinstance(value, Path) else value
        resolved = validate(resolved, 'panel')
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args = argparse.Namespace(**resolved)
    for name in PATH_FIELDS:
        if getattr(args, name, None) is not None:
            setattr(args, name, Path(getattr(args, name)))
    args.rollout_policy = getattr(args, 'rollout_policy', None)
    args.resolved_recipe = resolved
    return args
