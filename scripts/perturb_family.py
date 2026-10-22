"""task28：扰动卡组族的共用核心（引擎整组检查 + 疯狂戴夫候选池）。

两件事都只问引擎，不在 Python 里另写规则：
1. 替换池 = `SeedChooserScreen::CrazyDavePickSeeds` 的候选（`PvZEnv.deck_pool`）；
2. 整组检查 = `SeedChooserScreen::OnStartButton`（`PvZEnv.deck_warnings`）。
`deck_for_seed` 运行时是纯函数，所以把“会触发警告的卡组”在生成时穷举出来写进清单
（`engine_warned_decks`），运行时只做集合排除。
"""
import copy
from itertools import combinations


def candidate_decks(base, pool):
    """底卡组替换恰好 1 或 2 张后的全部**不同卡组集合**（槽位顺序无关，先到先得）。"""
    base = tuple(base)
    additions = [p for p in pool if p not in base]
    seen = set()
    for slot in range(len(base)):
        for plant in additions:
            deck = list(base)
            deck[slot] = plant
            key = tuple(sorted(deck))
            if key in seen:
                continue
            seen.add(key)
            yield deck
    for slot_a, slot_b in combinations(range(len(base)), 2):
        for plant_a, plant_b in combinations(additions, 2):
            deck = list(base)
            deck[slot_a] = plant_a
            deck[slot_b] = plant_b
            key = tuple(sorted(deck))
            if key in seen:
                continue
            seen.add(key)
            yield deck


def engine_family(env, level, base_deck, playthrough, seed, profile_for_deck, task_spec):
    """在 reset 后的真实 Board 上查两件事：逐卡选项与戴夫候选池。"""
    env.reset(deck=tuple(base_deck),
              task=task_spec(level=level, seed=seed, profile=profile_for_deck(tuple(base_deck)),
                             playthrough=playthrough))
    options = env.deck_options()
    pool = [p for p in env.deck_pool()['crazy_dave_pool'] if 0 <= p < 40]
    warnings = dict(options['not_recommended'])
    pool = sorted(p for p in pool if not warnings.get(p, 1))
    if not pool:
        raise ValueError(f'level {level}: empty Crazy Dave pool')
    return options, pool


def warned_decks(env, base_deck, pool):
    """穷举这一族能生成的全部卡组，返回触发任意整组警告的集合（排序后的列表）。"""
    warned = []
    checked = 0
    for deck in candidate_decks(base_deck, pool):
        checked += 1
        if env.deck_warnings(deck)['warnings']:
            warned.append(sorted(deck))
    return warned, checked


def build_family(source, catalog, levels, source_format, train_format, holdout_format, base_seeds,
                 train_count, holdouts_each, holdout_seed_start, holdout_eval_seeds,
                 deck_for_seed, required_plants=()):
    """生成训练族 + 留出族；`catalog[level] = (options, pool, warned)`。"""
    tasks = []
    heldout = []
    signatures = set()
    for level in levels:
        original = next(t for t in source['tasks'] if t['task_id'] == source_format.format(level=level))
        options, pool, warned, _checked = catalog[level]
        missing = [p for p in required_plants if p not in pool]
        if missing:
            raise ValueError(f'level {level}: Crazy Dave pool misses required plants {missing}')
        task = copy.deepcopy(original)
        start = base_seeds[level]
        task.update(task_id=train_format.format(level=level), seeds=list(range(start, start + train_count)))
        task['deck_perturb'] = dict(swaps=[1, 2], pool=pool, engine_options=options, excluded_decks=[],
                                     require_no_warnings=True, engine_warned_decks=warned)
        tasks.append(task)
        attempt = 0
        for index in range(holdouts_each):
            while True:
                deck = deck_for_seed(task, holdout_seed_start + 10000 * level + attempt)
                attempt += 1
                signature = tuple(sorted(deck))
                if signature not in signatures:
                    break
                if attempt > 4096:
                    raise ValueError('not enough distinct held-out decks')
            signatures.add(signature)
            held = copy.deepcopy(original)
            held.update(task_id=holdout_format.format(level=level, index=index + 1), deck=list(deck),
                        seeds=list(holdout_eval_seeds), split='heldout_deck',
                        holdout_semantics='unordered plant set excluded from every perturbation training family')
            heldout.append(held)
    for task in tasks:
        task['deck_perturb']['excluded_decks'] = [list(v) for v in sorted(signatures)]
    return tasks, heldout


def run_family(*, source_path, out_path, levels, base_seeds, train_count, source_format, train_format,
               holdout_format, holdouts_each, holdout_seed_start, holdout_eval_seeds, required_plants,
               scan_paths, resource_dir, executable, purpose, task_spec, profile_for_deck,
               deck_for_seed, terrain_covered, env_factory):
    import hashlib
    import json
    from pathlib import Path

    source = json.loads(Path(source_path).read_text())
    catalog = {}
    with env_factory(resource_dir, executable=executable) as env:
        for level in levels:
            original = next(t for t in source['tasks'] if t['task_id'] == source_format.format(level=level))
            options, pool = engine_family(env, level, original['deck'], original.get('playthrough', 2),
                                          base_seeds[level], profile_for_deck, task_spec)
            warned, checked = warned_decks(env, original['deck'], pool)
            catalog[level] = (options, pool, warned, checked)
            print(f'level {level}: pool {len(pool)} cards, {checked} candidate decks, '
                  f'{len(warned)} warned ({len(warned) / checked:.1%})', flush=True)
    tasks, heldout = build_family(source, catalog, levels, source_format, train_format, holdout_format,
                                  base_seeds, train_count, holdouts_each, holdout_seed_start,
                                  holdout_eval_seeds, deck_for_seed, required_plants)
    result = dict(schema_version=1, purpose=purpose,
                  warning_scope='engine SeedChooserScreen::CheckDeckWarnings (OnStartButton) + '
                                'SeedChooserScreen::CrazyDaveSeedPool; no Python card rules',
                  tasks=tasks + heldout)
    excluded = {tuple(sorted(t['deck'])) for t in result['tasks'] if t.get('split') == 'heldout_deck'}
    decks = set()
    coverage = {p: 0 for p in required_plants}
    seeds = []
    for task in tasks:
        warned = {tuple(sorted(d)) for d in task['deck_perturb']['engine_warned_decks']}
        for seed in task['seeds']:
            deck = deck_for_seed(task, seed)
            assert len(set(deck) - set(task['deck'])) in (1, 2)
            assert len(deck) == len(task['deck']) and len(set(deck)) == len(deck)
            assert not (set(deck) & set(range(40, 48))), 'upgrade plant in deck'
            assert terrain_covered(task['level'], deck)
            assert tuple(sorted(deck)) not in excluded
            assert tuple(sorted(deck)) not in warned, 'training deck would trigger a whole-deck warning'
            for plant in required_plants:
                if plant in deck:
                    coverage[plant] += 1
            decks.add(tuple(sorted(deck)))
            seeds.append(seed)
    seen = {}
    for path in scan_paths:
        data = json.loads(Path(path).read_text())
        entries = data['tasks'] if isinstance(data, dict) and 'tasks' in data else data
        for task in entries:
            if not isinstance(task, dict):
                continue
            for seed in task.get('seeds') or []:
                seen.setdefault(int(seed), task.get('task_id'))
    clashes = sorted({(s, seen[s]) for s in seeds if s in seen})
    if clashes:
        raise SystemExit(f'seed overlap with existing manifests: {clashes[:5]} ({len(clashes)} seeds)')
    if required_plants and not all(coverage[p] for p in required_plants):
        raise SystemExit(f'required plants missing from training decks: {coverage}')
    result['generation_audit'] = dict(
        training_seeds=len(seeds), distinct_training_card_sets=len(decks),
        heldout_card_sets=len(excluded), all_heldout_sets_excluded=True,
        required_pool=list(required_plants), required_pool_deck_counts=coverage,
        per_level_candidates={str(level): dict(pool=len(catalog[level][1]),
                                              candidates=catalog[level][3],
                                              warned=len(catalog[level][2]))
                              for level in levels},
        scanned_manifests=[str(p) for p in scan_paths], scanned_existing_seeds=len(seen),
        seed_overlap=0,
        source_manifest_sha256=hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
        executable_sha256=hashlib.sha256(Path(executable).read_bytes()).hexdigest())
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result['generation_audit'], indent=1), flush=True)
    return result

