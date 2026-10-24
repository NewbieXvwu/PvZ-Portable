"""The chunked archive writer must be the whole-game writer, byte for byte.

search_play hands the frame JSON over as bytes and pvz_frame_store encodes it a chunk at a
time (task18: the archive write was the tallest point of a collection worker).  That is only
allowed to change the peak, never the archive, so both encoders are compared here on a game
long enough to span several chunks, with the frame patterns the trainer cares about:
rejected actions, wait labels, search targets and a verification block.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from pvz_frame_store import (ENCODE_CHUNK, encode_frames, encode_frames_from_source, frame_bytes,
                             write_frames, write_frames_from_source)


def game(frames: int = 3 * ENCODE_CHUNK + 7) -> list[dict]:
    """A synthetic game: plants come and go, every third decision is a rejected action."""
    base = dict(sun=50, wave=0, scene={"wave_count": 10}, packets=[], zombies=[], lanes=list(range(5)))
    out = [dict(base, tick=0, plants=[[0, 0, 0, 100, 100]], mowers=[[2, 1], [3, 1]])]
    for index in range(1, frames):
        action = ({"type": "wait", "ticks": 60} if index % 3 == 0
                  else {"type": "plant", "packet": index % 7, "row": index % 5, "col": index % 9})
        frame = dict(base, tick=index * 60, sun=50 + index,
                     plants=[[index % 5, index % 9, index % 7, 100, 100]] if index % 2 else [],
                     mowers=[[2, 1] if index % 5 else [2, 2], [3, 1]],
                     action=action, action_rejected=index % 3 == 0)
        if index % 4 == 0:
            frame["targets"] = [[index % 7, index % 5, index % 9, round(0.1 * (index % 9), 4)],
                                [-1, -1, -1, 0.5]]
        if index % 6 == 0:
            frame["verification"] = {"samples": 9, "student": action, "challenger": action,
                                     "before_values": [0.4], "after_values": [0.5],
                                     "before_override": True, "after_override": False,
                                     "original_samples": [3, 3], "decision_seed": index,
                                     "mode": "reproduced"}
        out.append(frame)
    return out


def source_of(frames: list[dict]) -> bytes:
    return "".join(json.dumps(frame, separators=(",", ":")) + "\n" for frame in frames).encode()


def test_chunked_encode_matches_whole_game_encode():
    frames = game()
    source = source_of(frames)
    whole = encode_frames(frames, source)
    chunked = encode_frames_from_source(source, chunk=ENCODE_CHUNK)
    assert list(whole) == list(chunked)
    assert whole["hints"] == chunked["hints"]
    for name in ("rows", "boards", "scalars", "plant", "packet", "cell", "values"):
        assert whole[name].dtype == chunked[name].dtype, name
        assert whole[name].shape == chunked[name].shape, name
        np.testing.assert_array_equal(whole[name], chunked[name], err_msg=name)


def test_chunked_encode_is_independent_of_the_chunk_size():
    frames = game()
    source = source_of(frames)
    reference = encode_frames(frames, source)
    for chunk in (1, 2, 3, ENCODE_CHUNK, len(frames) + 5):
        encoded = encode_frames_from_source(source, chunk=chunk)
        for name in ("rows", "boards", "scalars", "plant", "packet", "cell", "values"):
            np.testing.assert_array_equal(reference[name], encoded[name],
                                          err_msg=f"{name} at chunk={chunk}")
        assert reference["hints"] == encoded["hints"]


def test_source_writer_writes_the_same_archive(tmp_path, monkeypatch):
    import zipfile
    frames = game()
    source = source_of(frames)
    reference = tmp_path / "whole" / "frames.npz"
    reference.parent.mkdir()
    monkeypatch.setattr(zipfile.time, "localtime", lambda *args: (2026, 1, 2, 3, 4, 6, 4, 2, -1))
    write_frames(reference, frames)
    streamed = tmp_path / "streamed" / "frames.npz"
    streamed.parent.mkdir()
    monkeypatch.setattr(zipfile.time, "localtime", lambda *args: (2027, 5, 6, 7, 8, 10, 3, 126, -1))
    write_frames_from_source(streamed, source)
    assert frame_bytes(reference) == frame_bytes(streamed)
    with np.load(reference, allow_pickle=False) as a, np.load(streamed, allow_pickle=False) as b:
        assert a.files == b.files
        for name in a.files:
            np.testing.assert_array_equal(a[name], b[name], err_msg=name)
    # ZIP envelope dates are allowed to differ; the source and every array above
    # must remain identical across the two writers and across write dates.
