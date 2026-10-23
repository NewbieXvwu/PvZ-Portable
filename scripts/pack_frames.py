"""Copy episode archives as lossless, training-ready NPZ; retain originals.

Sources may be JSONL or NPZ (`pvz_frame_store.frame_paths` lists both); the JSON text is copied
byte-for-byte and the training rows are rebuilt from it, so a packed archive verifies against its
source the same way `scripts/convert_frames_npz.py` verifies in place.
"""
import argparse
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from pvz_frame_store import frame_bytes, frame_paths, write_frames_from_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = frame_paths(args.data)
    before = after = 0
    for index, source in enumerate(paths, 1):
        target = (args.out / source.relative_to(args.data)).with_suffix(".npz")
        write_frames_from_source(target, frame_bytes(source))
        meta = source.parent / "meta.json"
        if meta.is_file() and meta.resolve() != (target.parent / "meta.json").resolve():
            shutil.copy2(meta, target.parent / "meta.json")
        before += source.stat().st_size
        after += target.stat().st_size
        if index % 100 == 0 or index == len(paths):
            print(f"{index}/{len(paths)} episodes, {before} -> {after} bytes", flush=True)


if __name__ == "__main__":
    main()
