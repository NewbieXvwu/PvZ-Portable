"""task66: recompute the composite fog alpha in Python, from the game's own resources.

An independent second implementation of what EnvironmentFogOpacityAt does: it reads the fog
alpha straight out of main.pak (images/fog_.jpg's blue channel -- the channel
ResourceManager::LoadAlphaImage copies into the image's alpha), walks the quads exactly as
Board::DrawFog places them, and composes them with the usual alpha-over rule.  Then it compares
against the engine's own per-zombie number for the same tick.

If the two agree, the engine's number is not just "something deterministic" but the same
coverage the renderer paints.

    python3 scripts/task66_fog_reference.py --level 31 --seed 660031 --ticks 9000

Needs Pillow (local verification tool only; the test suite does not import this).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from pvz_env import TaskSpec, fog_diagnostics
from pvz_event_env import EventWaitEnv
from pvz_tasks import deck_for_level, profile_for_deck

DEFAULT_RESOURCE_DIR = Path(os.environ.get(
    'PVZ_RESOURCE_DIR', str(Path.home() / 'Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN')))


def pak_entries(pak: Path) -> dict[str, bytes]:
    """The pak's files, de-obfuscated (PakInterface XORs every byte with 0xF7)."""
    data = bytes(byte ^ 0xF7 for byte in pak.read_bytes())
    magic, version = struct.unpack_from('<II', data, 0)
    if magic != 0xBAC04AC0 or version != 0:
        raise ValueError(f'{pak} is not a PvZ pak')
    offset = 8
    records = []
    while True:
        flags = data[offset]
        offset += 1
        if flags & 0x80:
            break
        width = data[offset]
        offset += 1
        name = data[offset:offset + width].decode('latin-1').replace('\\', '/')
        offset += width
        size = struct.unpack_from('<i', data, offset)[0]
        offset += 4 + 8
        records.append((name, size))
    entries, position = {}, offset
    for name, size in records:
        entries[name] = data[position:position + size]
        position += size
    return entries


def fog_alpha(resource_dir: Path) -> np.ndarray:
    """The alpha DrawFog paints with: blue channel of images/fog_.jpg, shape (190, 1680)."""
    from PIL import Image
    entries = pak_entries(resource_dir / 'main.pak')
    with __import__('io').BytesIO(entries['images/fog_.jpg']) as buffer:
        return np.asarray(Image.open(buffer).convert('RGB'))[:, :, 2].astype(np.float64)


def trunc_div(numerator: int, denominator: int) -> int:
    """C++ integer division: truncates toward zero, unlike Python's floor division."""
    return -((-numerator) // denominator) if numerator < 0 else numerator // denominator


def opacity_at(alpha: np.ndarray, cols: int, rows: int, grid: np.ndarray, look: np.ndarray,
               offset: float, x: int, y: int) -> float:
    cel_width = alpha.shape[1] // cols
    cel_height = alpha.shape[0] // rows
    transmittance = 1.0
    for grid_x in range(9):
        position_x = int(grid_x * 80 + offset - 15)  # C++ float -> int truncates toward zero
        for grid_y in range(7):
            fade = int(grid[grid_x][grid_y])
            if fade == 0:
                continue
            position_y = grid_y * 85 + 20
            if not position_y <= y < position_y + cel_height:
                continue
            cel_col = int(look[grid_y % 6][grid_x]) % 8
            for copy in range(2 if grid_x == 8 else 1):
                left = position_x + copy * 80
                if not left <= x < left + cel_width:
                    continue
                texel = cel_col * cel_width + (x - left)
                texel_alpha = alpha[y - position_y][texel] / 255.0
                if texel_alpha == 0.0:
                    continue
                transmittance *= 1.0 - texel_alpha * (fade / 255.0)
    return 1.0 - transmittance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--level', type=int, default=31)
    parser.add_argument('--seed', type=int, default=660031)
    parser.add_argument('--ticks', type=int, default=9000)
    parser.add_argument('--every', type=int, default=250)
    parser.add_argument('--resource-dir', type=Path, default=DEFAULT_RESOURCE_DIR)
    parser.add_argument('--executable', type=Path, default=ROOT / 'build/pvz-portable')
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()

    alpha = fog_alpha(args.resource_dir)
    deck = deck_for_level(args.level)
    compared, worst, mismatched = 0, 0.0, []
    with EventWaitEnv(args.resource_dir, executable=args.executable, headless=True) as env:
        env.reset(deck=deck, task=TaskSpec(level=args.level, seed=args.seed,
                                           profile=profile_for_deck(deck)))
        for tick in range(1, args.ticks + 1):
            env.step(dict(type='wait', ticks=1))
            if tick % args.every:
                continue
            state = env.privileged_state()
            if state.get('fog_amount') is None:
                continue
            offset = state['fog_offset']
            shift = int((offset + 50.0) / 100.0)
            grid = np.zeros((9, 7), dtype=np.int64)
            for grid_x in range(9):
                for row in range(6):
                    col = grid_x + shift
                    grid[grid_x][row] = state['fog_amount'][row][col] if 0 <= col < 9 else 0
                grid[grid_x][6] = grid[grid_x][0]
            look = np.asarray(state['fog_look'], dtype=np.int64)   # [row][col]
            diagnostics = fog_diagnostics(state)
            for zombie in state['zombies']:
                if not zombie['on_board'] or zombie['id'] not in diagnostics:
                    continue
                detail = diagnostics[zombie['id']]
                box_x, box_y, box_width, box_height = detail['box']
                head_values, body_values = [], []
                for row in range(5):
                    y = box_y + trunc_div((2 * row + 1) * box_height, 2 * 5)
                    for column in range(3):
                        x = box_x + trunc_div((2 * column + 1) * box_width, 2 * 3)
                        value = opacity_at(alpha, 8, 1, grid, look, offset, x, y)
                        (head_values if row < 2 else body_values).append(value)
                pairs = [(min(head_values + body_values), detail['opacity'] / 1000.0, 'all'),
                         (min(head_values), detail['head'] / 1000.0, 'head'),
                         (min(body_values), detail['body'] / 1000.0, 'body')]
                for reference, engine, band in pairs:
                    compared += 1
                    worst = max(worst, abs(reference - engine))
                    if abs(reference - engine) > 0.02:
                        mismatched.append(dict(tick=state['tick'], id=zombie['id'], row=zombie['row'],
                                               x=zombie['x'], band=band, engine=engine,
                                               reference=reference))
    result = dict(compared=compared, worst_abs_error=worst, mismatched=mismatched[:10],
                  mismatched_count=len(mismatched))
    print(json.dumps(result, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
    return 0 if not mismatched else 1


if __name__ == '__main__':
    raise SystemExit(main())
