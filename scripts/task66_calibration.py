"""task66: build the human calibration sheet for the fog-visibility threshold.

The threshold has to come from what a person can actually see, not from a number picked here.
This tool does three things:

  select   pick hidden zombie-ticks spanning fog opacity 0.15-0.95, plus the "only the head
           shows" cases the user pointed at, plus controls
  capture  replay each demonstration in a windowed engine and save the frame at each chosen
           tick (the frame comes from the engine's own framebuffer, so a sleeping display or a
           locked desktop cannot black it out)
  panel    lay the crops out on one page, numbered, with no opacity printed anywhere

    python3 scripts/task66_calibration.py select  --out artifacts/task66/calibration/selection.json
    python3 scripts/task66_calibration.py capture --selection ... --out-dir artifacts/task66/calibration
    python3 scripts/task66_calibration.py panel   --shots ... --out artifacts/task66/calibration/panel.png
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from pvz_env import PlayerProfileContext, TaskSpec, fog_diagnostics
from pvz_event_env import EventWaitEnv

DEFAULT_RESOURCE_DIR = Path(os.environ.get(
    'PVZ_RESOURCE_DIR', str(Path.home() / 'Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN')))
REPLAY_DIR = ROOT / 'artifacts/task66/replay'
BOARD_WIDTH = 800


def band_centres(low: int = 200, high: int = 950, step: int = 50) -> list[int]:
    """One centre per 0.05 band of composite opacity, 0.20 through 0.95."""
    return list(range(low, high + 1, step))


def select(replay_paths: list[Path], head_only: int = 3) -> dict:
    """Pick the zombies the user will look at.

    The pool is *every* zombie-tick, not only the ones the old cell rule hides: the whole point
    is to find out where the human boundary sits, and the interesting cases are exactly the ones
    the rule and the eye may disagree about.  Each zombie id appears at most once, and two picks
    from one replay are never within 90 ticks of each other, so the sheet cannot be answered by
    "same zombie, next frame".

    Where a band can be shown at the fog's *left* edge (x around 460-560, zombie fully on the
    board, fog wall rising across its body) that is preferred: it is the case a player actually
    meets.  The right-edge family (x > 740, zombie half off the board, fog thinning again in the
    last cel's single-coverage tail) is only a fallback, because a zombie that is 80% outside the
    board is not something anyone can judge.
    """
    archives = [json.loads(path.read_text()) for path in replay_paths]
    chosen: list[dict] = []
    used_ids: set = set()
    used_ticks: dict = {}

    def key(entry):
        return (entry['archive'], entry['id'])

    def on_board(entry) -> bool:
        box_x, _, box_width, _ = entry['box']
        return box_x >= 0 and box_x + box_width <= BOARD_WIDTH

    def wall(entry) -> bool:
        return 460 <= entry['x'] <= 560

    def take(entry, note: str) -> None:
        used_ids.add(key(entry))
        used_ticks.setdefault(entry['archive'], []).append(entry['tick'])
        chosen.append(dict(entry, note=note))

    def free(entry) -> bool:
        if key(entry) in used_ids:
            return False
        return all(abs(entry['tick'] - tick) >= 90 for tick in used_ticks.get(entry['archive'], ()))

    pool = [dict(entry, archive=archive['archive'], label=archive['label'])
            for archive in archives for entry in archive['rows']]

    # Head-only cases get reserved slots: the head band is under half covered while the body is
    # buried, which is the case the user pointed at (a head sticking out above the fog's top edge).
    head_cases = [entry for entry in pool
                  if entry['head'] < 400 and entry['body'] > 800 and on_board(entry)
                  and 60 <= entry['head'] <= 380]
    for target in (120, 220, 320)[:head_only]:
        candidates = [entry for entry in head_cases if free(entry)]
        if not candidates:
            continue
        candidates.sort(key=lambda entry: (abs(entry['head'] - target), not wall(entry),
                                           -entry['body']))
        take(candidates[0], f'head-only head={target}')

    # One sample per 0.05 band of composite opacity.
    for centre in band_centres():
        candidates = [entry for entry in pool
                      if abs(entry['opacity'] - centre) <= 25 and free(entry)]
        if not candidates:
            continue
        candidates.sort(key=lambda entry: (not wall(entry), not on_board(entry),
                                           not entry['cell_hidden'],
                                           abs(entry['opacity'] - centre), entry['x']))
        take(candidates[0], f'band {centre}')

    def control(predicate, note: str) -> None:
        candidates = [entry for entry in pool if free(entry) and predicate(entry)]
        if not candidates:
            return
        candidates.sort(key=lambda entry: (not on_board(entry), entry['tick']))
        take(candidates[0], note)

    for _ in range(2):
        control(lambda entry: entry['opacity'] >= 980 and entry['cell_hidden'] and on_board(entry),
                'control: deep fog, expected invisible')
    for _ in range(2):
        control(lambda entry: entry['opacity'] == 0 and not entry['cell_hidden'] and entry['x'] < 450,
                'control: no fog, expected clearly visible')

    # Number the page in a shuffled order: sorting by opacity would hand the answer to the user.
    random.Random(66).shuffle(chosen)
    fields = ('archive', 'label', 'tick', 'id', 'row', 'x', 'type', 'col', 'opacity', 'head',
              'body', 'box', 'cell', 'cell_hidden', 'note')
    items = [{key: entry[key] for key in fields} for entry in chosen]
    for index, item in enumerate(items, 1):
        item['index'] = index
    return dict(items=items)


def capture(selection: Path, out_dir: Path, resource_dir: Path, executable: Path,
            replays: list[Path]) -> None:
    data = json.loads(selection.read_text())
    by_archive: dict[str, list[dict]] = {}
    for item in data['items']:
        by_archive.setdefault(item['archive'], []).append(item)
    records = {}
    for path in replays:
        loaded = json.loads(path.read_text())
        records[loaded['archive']] = loaded
    out_dir.mkdir(parents=True, exist_ok=True)
    shots = []
    for archive, items in by_archive.items():
        meta = json.loads((Path(archive) / 'meta.json').read_text())
        config = dict(meta['environment_task'])
        config['profile'] = PlayerProfileContext(**config['profile'])
        config['preplanted'] = tuple(tuple(plant) for plant in config.get('preplanted', ()))
        targets = {item['tick']: item for item in items}
        last = max(targets)
        # The recorded zombie-tick table is the reference for "is this the same trajectory".
        # The state hash cannot be used here: a visible run leaves three reanimation fields at
        # 0 where a headless run leaves them at -1 (measured, see the task note), so the digest
        # differs from the one a headless recording produced even though the game state matches.
        expected: dict[int, list[tuple]] = {}
        record = records.get(archive)
        if record is not None:
            for entry in record['rows']:
                if entry['tick'] in targets:
                    expected.setdefault(entry['tick'], []).append(
                        (entry['id'], entry['row'], entry['type'], entry['x'], entry['opacity']))
        os.environ['PVZ_ENV_RENDER_EVERY'] = '0'
        os.environ['PVZ_ENV_RENDER_MS'] = '0'
        # Not begin_demo(): demo mode forces the app headless and then nothing is ever drawn.
        with EventWaitEnv(resource_dir, executable=executable, headless=False) as env:
            env.reset(deck=meta['task']['deck'], task=TaskSpec(**config))
            if env.state_hash() != meta['initial_state_hash']:
                raise RuntimeError(f'{archive}: initial state hash differs')
            with (Path(archive) / 'ticks.jsonl').open() as stream:
                for line in stream:
                    entry = json.loads(line)
                    observation, _, _, _, info = env.step(entry['action'])
                    if info['ok'] != entry['ok']:
                        raise RuntimeError(f'{archive}: action result differs at tick '
                                           f'{observation["tick"]}')
                    tick = observation['tick']
                    if tick not in targets:
                        if tick > last:
                            break
                        continue
                    item = targets[tick]
                    state = env.privileged_state()
                    diagnostics = fog_diagnostics(state)
                    zombie = next((z for z in state['zombies'] if z['id'] == item['id']), None)
                    if zombie is None:
                        raise RuntimeError(f'{archive}: zombie {item["id"]} gone at tick {tick}')
                    if tick in expected:
                        want = sorted(expected[tick])
                        got = sorted((z['id'], z['row'], z['type'], z['x'],
                                      diagnostics[z['id']]['opacity'])
                                     for z in state['zombies'] if z['on_board'])
                        if got != want:
                            raise RuntimeError(f'{archive}: zombie table differs at tick {tick}: '
                                               f'{got} != {want}')
                    detail = diagnostics[item['id']]
                    path = out_dir / f"{item['index']:02d}.png"
                    env.screenshot(path)
                    shots.append(dict(item, screenshot=str(path), box=detail['box'],
                                      opacity=detail['opacity'], head=detail['head'],
                                      body=detail['body']))
                    if tick >= last:
                        break
    (out_dir / 'shots.json').write_text(json.dumps(dict(shots=shots), indent=2))
    print(f'{len(shots)} shots')


def panel(shots: Path, out: Path, crop: int, columns: int = 5) -> None:
    """One numbered page of crops.  No opacity, no head/body split, no archive name: the only
    thing on the page is the picture and its number."""
    from PIL import Image, ImageDraw, ImageFont
    data = json.loads(shots.read_text())
    tiles = []
    for shot in data['shots']:
        image = Image.open(shot['screenshot']).convert('RGB')
        scale = image.width / BOARD_WIDTH
        box_x, box_y, box_width, box_height = shot['box']
        centre_x = box_x + box_width / 2
        centre_y = box_y + box_height / 2
        half = crop / 2
        left = max(0, min(BOARD_WIDTH - crop, centre_x - half))
        top = max(0, min(image.height / scale - crop, centre_y - half))
        tiles.append((shot['index'], image.crop((int(left * scale), int(top * scale),
                                                 int((left + crop) * scale),
                                                 int((top + crop) * scale)))))
    tile_size = max(tile.width for _, tile in tiles)
    tile_height = max(tile.height for _, tile in tiles)
    rows = (len(tiles) + columns - 1) // columns
    label_height = 48
    border = 2
    page = Image.new('RGB', (columns * (tile_size + border), rows * (tile_height + label_height + border)),
                     'black')
    draw = ImageDraw.Draw(page)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 30)
    except OSError:
        font = ImageFont.load_default()
    for position, (index, tile) in enumerate(tiles):
        column, row = position % columns, position // columns
        x = column * (tile_size + border)
        y = row * (tile_height + label_height + border)
        page.paste(tile, (x, y))
        text = f'{index:02d}'
        width = draw.textlength(text, font=font)
        draw.text((x + (tile_size - width) / 2, y + tile_height + 8), text, fill='white', font=font)
    out.parent.mkdir(parents=True, exist_ok=True)
    page.save(out)
    print(f'{out}: {len(tiles)} tiles, {page.width}x{page.height}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('select')
    s.add_argument('--out', type=Path, required=True)
    s.add_argument('--replays', type=Path, nargs='+',
                   default=sorted(REPLAY_DIR.glob('demo_fog_*.json')))
    c = sub.add_parser('capture')
    c.add_argument('--selection', type=Path, required=True)
    c.add_argument('--out-dir', type=Path, required=True)
    c.add_argument('--resource-dir', type=Path, default=DEFAULT_RESOURCE_DIR)
    c.add_argument('--executable', type=Path, default=ROOT / 'build/pvz-portable')
    c.add_argument('--replays', type=Path, nargs='+',
                   default=sorted(REPLAY_DIR.glob('demo_fog_*.json')))
    p = sub.add_parser('panel')
    p.add_argument('--shots', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--crop', type=int, default=200, help='crop side in board pixels')
    p.add_argument('--columns', type=int, default=5)
    args = parser.parse_args()
    if args.command == 'select':
        result = select(args.replays)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        for item in result['items']:
            print(f"{item['index']:02d}", item['note'], item['label'], item['tick'], item['id'],
                  'op', item['opacity'], 'head', item['head'], 'body', item['body'],
                  'row', item['row'], 'col', item['col'], 'x', round(item['x'], 1),
                  'hidden', item['cell_hidden'])
    elif args.command == 'capture':
        capture(args.selection, args.out_dir, args.resource_dir, args.executable, args.replays)
    else:
        panel(args.shots, args.out, args.crop, args.columns)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
