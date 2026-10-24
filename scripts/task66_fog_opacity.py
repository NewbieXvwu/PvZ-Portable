"""task66: measure how much fog a zombie really stands in, and how far the cell rule is from it.

Reads an episode archive (a human demonstration, read-only), replays its action tape tick by
tick against the engine, and for every zombie-tick records the composite fog alpha over the
zombie (the engine's EnvironmentZombieFogOpacity) together with the verdict of the pre-task66
rule ("the cell value is non-zero, so hide it").

The engine prints both numbers only in the privileged observation, so no policy and no public
frame is touched by this tool.

    python3 scripts/task66_fog_opacity.py replay --archive <dir> --out <json>
    python3 scripts/task66_fog_opacity.py summarize --out <json> <per-archive json>...
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from pvz_env import PlayerProfileContext, TaskSpec, fog_diagnostics
from pvz_event_env import EventWaitEnv

DEFAULT_RESOURCE_DIR = Path(os.environ.get(
    'PVZ_RESOURCE_DIR', str(Path.home() / 'Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN')))
DEMO_ROOT = Path('/Users/newbiexvwu/PvZAgent/.claude/worktrees/demo/artifacts/human_demo/raw')
DEMOS = (
    DEMO_ROOT / 'seed_20261031/demo_fog_31_fume_rebaselined',
    DEMO_ROOT / 'seed_20261031/demo_fog_31_snow_m200',
    DEMO_ROOT / 'seed_20261033/demo_fog_33_cactus_snow_m200',
)


def task_from_meta(meta: dict) -> TaskSpec:
    config = dict(meta['environment_task'])
    config['profile'] = PlayerProfileContext(**config['profile'])
    config['preplanted'] = tuple(tuple(plant) for plant in config.get('preplanted', ()))
    return TaskSpec(**config)


def replay(archive: Path, resource_dir: Path, executable: Path, limit: int | None = None) -> dict:
    """Replay one archive, verifying every recorded state hash, and collect per-tick fog data."""
    meta = json.loads((archive / 'meta.json').read_text())
    task = task_from_meta(meta)
    rows: list[dict] = []
    hidden: list[dict] = []
    verified = 0
    with EventWaitEnv(resource_dir, executable=executable, headless=True) as env:
        env.begin_demo()
        env.reset(deck=meta['task']['deck'], task=task)
        if env.state_hash() != meta['initial_state_hash']:
            raise RuntimeError(f'{archive.name}: initial state hash differs')
        with (archive / 'ticks.jsonl').open() as stream:
            for operation, line in enumerate(stream, 1):
                if limit is not None and operation > limit:
                    break
                entry = json.loads(line)
                observation, _, _, _, info = env.step(entry['action'])
                if info['ok'] != entry['ok'] or observation['tick'] != entry['end_tick'] or \
                        env._last_response['state_hash'] != entry['state_hash']:
                    raise RuntimeError(f'{archive.name}: replay diverged at operation {operation}, '
                                       f'tick {observation["tick"]}')
                verified += 1
                if observation['terminal']:
                    break
                privileged = env.privileged_state()
                fog = privileged.get('fog_amount')
                if fog is None:
                    continue
                tick = privileged['tick']
                diagnostics = fog_diagnostics(privileged)
                for zombie in privileged['zombies']:
                    if not zombie['on_board']:
                        continue
                    col = max(0, min(8, int((zombie['x'] - 40) // 80)))
                    detail = diagnostics[zombie['id']]
                    entry = dict(tick=tick, id=zombie['id'], row=zombie['row'], x=zombie['x'],
                                 type=zombie['type'], col=col, opacity=detail['opacity'],
                                 head=detail['head'], body=detail['body'],
                                 box=detail['box'], cell=fog[zombie['row']][col],
                                 cell_hidden=detail['cell_hidden'])
                    rows.append(entry)
                    if detail['cell_hidden']:
                        hidden.append(entry)
    return dict(archive=str(archive), label=meta['task']['id'], level=meta['task']['level'],
                operations=verified, zombie_ticks=len(rows), hidden_ticks=len(hidden),
                rows=rows, hidden=hidden)


def histogram(values: list[int], width: int = 20) -> list[int]:
    """Counts of opacity in permille, one bucket per 1/width (0.05 by default)."""
    buckets = [0] * (width + 1)
    for value in values:
        buckets[min(width, value * width // 1000)] += 1
    return buckets


def summarize(paths: list[Path]) -> dict:
    archives = [json.loads(path.read_text()) for path in paths]
    hidden = [entry for archive in archives for entry in archive['hidden']]
    rows = [entry for archive in archives for entry in archive['rows']]
    on_board = [entry for entry in hidden if entry['x'] <= 780]
    opacities = [entry['opacity'] for entry in hidden]
    below = {f'below_{limit}': sum(value < limit for value in opacities)
             for limit in (200, 300, 400, 500, 600, 700, 800, 900)}
    by_column: dict[int, int] = {}
    by_cell: dict[int, int] = {}
    for entry in hidden:
        by_column[entry['col']] = by_column.get(entry['col'], 0) + 1
        by_cell[entry['cell']] = by_cell.get(entry['cell'], 0) + 1
    # "Only the head shows": the head band is under half-covered while the body is not.  This is
    # the case the user pointed at in their L33 screenshot (row 0, head above the fog's edge).
    head_only = sum(1 for entry in hidden if entry['head'] < 500 <= entry['body'])
    # Mean coverage against x for zombies whose body is on screen: where the fog wall really is.
    profile: dict[int, list[int]] = {}
    for entry in rows:
        if not 400 <= entry['x'] <= 780:
            continue
        profile.setdefault(int(entry['x'] // 20) * 20, []).append(entry['opacity'])
    return dict(
        archives=[dict(archive=archive['archive'], label=archive['label'], level=archive['level'],
                       operations=archive['operations'], zombie_ticks=archive['zombie_ticks'],
                       hidden_ticks=archive['hidden_ticks']) for archive in archives],
        zombie_ticks=len(rows), hidden_ticks=len(hidden),
        hidden_ticks_on_screen=len(on_board),
        hidden_share=len(hidden) / len(rows) if rows else 0.0,
        opacity_histogram_permille=histogram(opacities),
        opacity_min=min(opacities) if opacities else None,
        opacity_median=sorted(opacities)[len(opacities) // 2] if opacities else None,
        opacity_max=max(opacities) if opacities else None,
        **below,
        head_only_ticks_at_500=head_only,
        hidden_by_column=dict(sorted(by_column.items())),
        hidden_by_cell_value=dict(sorted(by_cell.items())),
        opacity_by_x=dict(sorted((str(key), round(sum(value) / len(value)))
                                 for key, value in profile.items())),
        distinct_hidden_zombies=len({(entry['tick'] // 100000, entry['id']) for entry in hidden}),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    a = sub.add_parser('replay')
    a.add_argument('--archive', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--resource-dir', type=Path, default=DEFAULT_RESOURCE_DIR)
    a.add_argument('--executable', type=Path, default=ROOT / 'build/pvz-portable')
    a.add_argument('--limit', type=int, default=None)

    s = sub.add_parser('summarize')
    s.add_argument('--out', type=Path, required=True)
    s.add_argument('inputs', type=Path, nargs='+')

    args = parser.parse_args()
    if args.command == 'replay':
        result = replay(args.archive, args.resource_dir, args.executable, args.limit)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result))
        print(json.dumps({key: value for key, value in result.items() if key not in ('rows', 'hidden')},
                         indent=2))
    else:
        result = summarize(args.inputs)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
