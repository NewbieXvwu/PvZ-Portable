"""task66: build the human calibration sheet for the fog-visibility threshold.

The threshold has to come from what a person can actually see, not from a number picked here.
This tool does three things:

  select   pick hidden zombie-ticks spanning fog opacity 0.15-0.95 plus controls, from the
           three replayed demonstrations
  capture  replay each demonstration in a windowed engine, screenshot the frame at each chosen
           tick and crop around the target zombie
  panel    lay the crops out on one page, numbered, with no opacity printed anywhere

    python3 scripts/task66_calibration.py select  --out artifacts/task66/calibration/selection.json
    python3 scripts/task66_calibration.py capture --selection ... --out-dir artifacts/task66/calibration/shots
    python3 scripts/task66_calibration.py panel   --selection ... --out artifacts/task66/calibration/panel.png
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from pvz_env import PlayerProfileContext, TaskSpec
from pvz_event_env import EventWaitEnv

DEFAULT_RESOURCE_DIR = Path(os.environ.get(
    'PVZ_RESOURCE_DIR', str(Path.home() / 'Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN')))
REPLAY_DIR = ROOT / 'artifacts/task66/replay'
WINDOWLIST = ROOT / 'artifacts/task66/windowlist'


def band_centres(low: int = 150, high: int = 950, step: int = 50) -> list[int]:
    return list(range(low + step // 2, high + 1, step))


def select(replay_paths: list[Path], head_only: int = 3) -> dict:
    archives = [json.loads(path.read_text()) for path in replay_paths]
    pool = []
    for archive in archives:
        for entry in archive['hidden']:
            pool.append(dict(entry, archive=archive['archive'], label=archive['label']))
    chosen, used = [], set()

    def key(entry):
        return (entry['archive'], entry['tick'], entry['id'])

    # The head-only cases first: the head band is under half covered while the body is buried.
    # They are the ones the user pointed at, so they get reserved slots instead of competing
    # with the opacity bands.
    head_cases = [entry for entry in pool
                  if entry['head'] < 400 <= 800 < entry['body'] and 560 <= entry['x'] <= 780]
    head_cases.sort(key=lambda entry: (entry['head'], -entry['body']))
    spread = max(1, len(head_cases) // head_only) if head_cases else 1
    for entry in head_cases[::spread][:head_only]:
        used.add(key(entry))
        chosen.append(entry)

    for centre in band_centres():
        candidates = [entry for entry in pool
                      if abs(entry['opacity'] - centre) <= 25
                      and 560 <= entry['x'] <= 780
                      and key(entry) not in used]
        if not candidates:
            continue
        # Prefer a zombie whose whole body is on screen, whose head is not the only visible
        # part (those are covered above), and whose cell is the leftmost fog column -- the case
        # the user reported -- then the one closest to the band centre.
        candidates.sort(key=lambda entry: (entry['col'] != 6, entry['head'] < entry['body'],
                                           abs(entry['opacity'] - centre)))
        pick = candidates[0]
        used.add(key(pick))
        chosen.append(pick)

    def control(predicate, note: str) -> dict | None:
        for archive in archives:
            for entry in archive['rows']:
                if predicate(entry) and (archive['archive'], entry['tick'], entry['id']) not in used:
                    used.add((archive['archive'], entry['tick'], entry['id']))
                    return dict(entry, archive=archive['archive'], label=archive['label'], note=note)
        return None

    controls = [
        control(lambda entry: entry['cell_hidden'] and entry['opacity'] >= 980 and 560 <= entry['x'] <= 780,
                'deep fog, expected invisible'),
        control(lambda entry: entry['cell_hidden'] and entry['opacity'] >= 980 and 560 <= entry['x'] <= 780,
                'deep fog, expected invisible'),
        control(lambda entry: not entry['cell_hidden'] and entry['opacity'] == 0 and entry['col'] <= 4,
                'no fog at all, expected clearly visible'),
        control(lambda entry: not entry['cell_hidden'] and entry['opacity'] == 0 and entry['col'] <= 4,
                'no fog at all, expected clearly visible'),
    ]
    controls = [entry for entry in controls if entry is not None]
    fields = ('archive', 'label', 'tick', 'id', 'row', 'x', 'type', 'col', 'opacity', 'head',
              'body', 'cell', 'cell_hidden')
    items = [{key: entry[key] for key in fields} | dict(kind='sample') for entry in chosen]
    items += [{key: entry[key] for key in fields} | dict(kind='control', note=entry['note'])
              for entry in controls]
    # Numbering runs in tick order per archive so one replay pass can serve several items.
    items.sort(key=lambda entry: (entry['archive'], entry['tick']))
    for index, item in enumerate(items, 1):
        item['index'] = index
    return dict(items=items)


def window_id(pid: int) -> int | None:
    for _ in range(40):
        out = subprocess.run([str(WINDOWLIST), str(pid)], capture_output=True, text=True).stdout.split()
        if out:
            return int(out[0])
        time.sleep(0.1)
    return None


def capture(selection: Path, out_dir: Path, resource_dir: Path, executable: Path) -> None:
    data = json.loads(selection.read_text())
    by_archive: dict[str, list[dict]] = {}
    for item in data['items']:
        by_archive.setdefault(item['archive'], []).append(item)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'shots.json').unlink(missing_ok=True)
    shots = []
    for archive, items in by_archive.items():
        meta = json.loads((Path(archive) / 'meta.json').read_text())
        config = dict(meta['environment_task'])
        config['profile'] = PlayerProfileContext(**config['profile'])
        config['preplanted'] = tuple(tuple(plant) for plant in config.get('preplanted', ()))
        targets = {item['tick']: item for item in items}
        last = max(targets)
        os.environ['PVZ_ENV_RENDER_EVERY'] = '1'
        os.environ['PVZ_ENV_RENDER_MS'] = '0'
        with EventWaitEnv(resource_dir, executable=executable, headless=False) as env:
            identifier = window_id(env._process.pid)
            if identifier is None:
                raise RuntimeError('no window for the visible engine')
            env.begin_demo()
            env.reset(deck=meta['task']['deck'], task=TaskSpec(**config))
            with (Path(archive) / 'ticks.jsonl').open() as stream:
                for operation, line in enumerate(stream, 1):
                    entry = json.loads(line)
                    observation, *_ = env.step(entry['action'])
                    tick = observation['tick']
                    if tick not in targets:
                        if tick > last:
                            break
                        continue
                    item = targets[tick]
                    time.sleep(0.25)
                    path = out_dir / f"{item['index']:02d}.png"
                    subprocess.run(['screencapture', '-x', '-o', '-l', str(identifier), str(path)],
                                   check=True)
                    state = env.privileged_state()
                    zombie = next((z for z in state['zombies'] if z['id'] == item['id']), None)
                    shots.append(dict(item, screenshot=str(path),
                                      box=zombie['fog_box'] if zombie else None,
                                      opacity=zombie['fog_opacity'] if zombie else None,
                                      head=zombie['fog_head_opacity'] if zombie else None,
                                      body=zombie['fog_body_opacity'] if zombie else None))
                    if tick >= last:
                        break
    (out_dir / 'shots.json').write_text(json.dumps(dict(shots=shots), indent=2))


def panel(shots: Path, out: Path, scale: float, offset_x: float, offset_y: float,
          crop: int, columns: int = 5) -> None:
    """One numbered page of crops.  No opacity, no head/body split, no archive name: the only
    thing on the page is the picture and its number."""
    from PIL import Image, ImageDraw
    data = json.loads(shots.read_text())
    tiles = []
    for shot in data['shots']:
        image = Image.open(shot['screenshot']).convert('RGB')
        if shot['box'] is None:
            continue
        box_x, box_y, box_width, box_height = shot['box']
        centre_x = box_x + box_width / 2
        centre_y = box_y + box_height / 2
        half = crop / 2
        left = (offset_x + (centre_x - half) * scale)
        top = (offset_y + (centre_y - half) * scale)
        right = (offset_x + (centre_x + half) * scale)
        bottom = (offset_y + (centre_y + half) * scale)
        tiles.append((shot['index'], image.crop((int(left), int(top), int(right), int(bottom)))))
    tile_size = max(tile.width for _, tile in tiles)
    tile_height = max(tile.height for _, tile in tiles)
    rows = (len(tiles) + columns - 1) // columns
    label_height = 34
    page = Image.new('RGB', (columns * tile_size, rows * (tile_height + label_height)), 'white')
    draw = ImageDraw.Draw(page)
    for position, (index, tile) in enumerate(tiles):
        column, row = position % columns, position // columns
        x = column * tile_size
        y = row * (tile_height + label_height)
        page.paste(tile.resize((tile_size, tile_height)), (x, y))
        draw.text((x + 10, y + tile_height + 8), f'{index:02d}', fill='black')
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

    p = sub.add_parser('panel')
    p.add_argument('--shots', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--scale', type=float, required=True,
                   help='window pixels per board pixel in the screenshot')
    p.add_argument('--offset-x', type=float, required=True,
                   help='screenshot x of board x=0')
    p.add_argument('--offset-y', type=float, required=True,
                   help='screenshot y of board y=0')
    p.add_argument('--crop', type=int, default=220, help='crop side in board pixels')
    p.add_argument('--columns', type=int, default=5)
    args = parser.parse_args()
    if args.command == 'select':
        result = select(args.replays)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        for item in result['items']:
            print(item['index'], item['kind'], item['label'], item['tick'], item['id'],
                  item['opacity'], item['col'], round(item['x'], 1))
    elif args.command == 'capture':
        capture(args.selection, args.out_dir, args.resource_dir, args.executable)
    else:
        panel(args.shots, args.out, args.scale, args.offset_x, args.offset_y, args.crop,
              args.columns)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
