"""Lossless episode archives with precomputed student training rows."""
from functools import lru_cache
import hashlib
import inspect
import json
from pathlib import Path
import tempfile

import numpy as np

def frame_paths(data: Path):
    paths = {path.parent: path for path in data.glob("*/*/frames.jsonl")}
    paths.update({path.parent: path for path in data.glob("*/*/frames.npz")})
    return sorted(paths.values(), key=lambda path: str(path.parent))


def frame_path(archive: Path) -> Path:
    binary = archive / "frames.npz"
    return binary if binary.is_file() else archive / "frames.jsonl"


def frame_bytes(path: Path) -> bytes:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as stored:
            return stored["frames"].tobytes()
    return path.read_bytes()


def read_frames(path: Path) -> list[dict]:
    return [json.loads(line) for line in frame_bytes(path).splitlines() if line.strip()]


def row_damage_labels(frames: list[dict], horizon: int) -> np.ndarray:
    """Per input frame: plant disappearance or first mower activation in the next K decisions.

    Archives expose disappearance, not a per-plant cause of death. Executed shovels are excluded;
    natural consumption/expiry remains indistinguishable from destruction in these archives.
    The final window ends at the last recorded frame. Sampling training rows never changes K.
    """
    from collections import Counter
    events = np.zeros((len(frames), 6), dtype=np.int64)
    for index in range(1, len(frames)):
        previous, current = frames[index - 1], frames[index]
        before = Counter(tuple(plant[:3]) for plant in previous["plants"])
        after = Counter(tuple(plant[:3]) for plant in current["plants"])
        action = current.get("action") or {}
        for row, col, _kind in before - after:
            if (action.get("type") == "shovel" and not current.get("action_rejected")
                    and (row, col) == (action.get("row"), action.get("col"))):
                continue
            events[index, row] = 1
        old_mowers, new_mowers = dict(previous["mowers"]), dict(current["mowers"])
        for row, state in new_mowers.items():
            if state == 2 and old_mowers.get(row) != 2:
                events[index, row] = 1
        for row, state in old_mowers.items():
            if state != 2 and row not in new_mowers:
                events[index, row] = 1
    cumulative = events.cumsum(axis=0)
    starts = np.arange(len(frames))
    ends = np.minimum(starts + horizon, len(frames) - 1)
    return (cumulative[ends] > cumulative[starts]).astype(np.float32)


def target_values(targets) -> list[float]:
    from pvz_rollout_policy import PACKETS
    out = [float("nan")] * (PACKETS + 1)
    for packet, _row, _col, value in targets or []:
        index = PACKETS if packet < 0 else packet
        if not out[index] >= value:
            out[index] = value
    return out


# Per training row, the plant candidates the search compared at cell level: (packet, cell, value), best value per
# (packet, cell), best first, padded with (-1, -1, nan). `target_values` keeps only the best cell of each card; this
# keeps the cells, so a student can be taught which of two cells of the same card the search preferred.
CELL_CANDIDATES = 12


def candidate_cells(targets) -> list[list[float]]:
    from pvz_rollout_policy import LAWN_COLS
    best: dict[tuple[int, int], float] = {}
    for packet, row, col, value in targets or []:
        if packet < 0:
            continue
        key = (packet, row * LAWN_COLS + col)
        if not best.get(key, float("-inf")) >= value:
            best[key] = value
    ranked = sorted(((value, packet, cell) for (packet, cell), value in best.items()), reverse=True)[:CELL_CANDIDATES]
    return [[packet, cell, value] for value, packet, cell in ranked] + [[-1, -1, float("nan")]] * (CELL_CANDIDATES - len(ranked))


def encode_candidates(frames: list[dict]) -> dict:
    """Same rows as `encode_frames` (frame i-1 -> labelled action at frame i), cell-level candidates only."""
    rows, cands = [], []
    for index in range(1, len(frames)):
        action = frames[index].get("label") or frames[index].get("action")
        if not action or frames[index].get("action_rejected"):
            continue
        rows.append(index)
        cands.append(candidate_cells(frames[index].get("targets")))
    return dict(rows=np.array(rows, dtype=np.int64),
                cands=np.array(cands, dtype=np.float32).reshape(-1, CELL_CANDIDATES, 3))


def cache_candidates(path: Path, directory: Path) -> Path:
    """Side cache of `encode_candidates` beside the feature cache (built only when a trainer asks for cell targets)."""
    signature = hashlib.sha256("".join(inspect.getsource(f) for f in (candidate_cells, encode_candidates)).encode()
                               + str(CELL_CANDIDATES).encode()).hexdigest()
    source = frame_bytes(path)
    digest = hashlib.sha256(signature.encode() + source).hexdigest()
    target = directory / "cells" / (hashlib.sha256(str(path.resolve()).encode()).hexdigest() + ".npz")
    if target.exists():
        with np.load(target, allow_pickle=False) as stored:
            if str(stored["digest"]) == digest:
                return target
    _save(target, digest=digest, **encode_candidates([json.loads(line) for line in source.splitlines() if line.strip()]))
    return target


def cache_aux(path: Path, directory: Path, horizon: int) -> Path:
    source = frame_bytes(path)
    signature = inspect.getsource(row_damage_labels) + inspect.getsource(cache_aux) + str(horizon)
    digest = hashlib.sha256(signature.encode() + source).hexdigest()
    target = directory / f"aux_v1_k{horizon}" / (hashlib.sha256(str(path.resolve()).encode()).hexdigest() + ".npz")
    if target.exists():
        with np.load(target, allow_pickle=False) as stored:
            if str(stored["digest"]) == digest:
                return target
    frames = [json.loads(line) for line in source.splitlines() if line.strip()]
    rows = np.array([index for index in range(1, len(frames))
                     if (frames[index].get("label") or frames[index].get("action"))
                     and not frames[index].get("action_rejected")], dtype=np.int64)
    _save(target, digest=digest, rows=rows, aux=row_damage_labels(frames, horizon)[rows - 1].astype(np.uint8))
    return target


def encode_frames(frames: list[dict], source: bytes) -> dict:
    from pvz_rollout_policy import LAWN_COLS, PACKETS, features
    from pvz_value_net import BOARD_CHANNELS, COLS, ROWS, SCALARS
    rows, boards, scalars, plant, packet, cell, values = [], [], [], [], [], [], []
    for index in range(1, len(frames)):
        action = frames[index].get("label") or frames[index].get("action")
        if not action or frames[index].get("action_rejected"):
            continue
        board, scalar = features(frames[index - 1], frames[index - 2] if index >= 2 else None)
        rows.append(index)
        boards.append(board.numpy().astype(np.float16))
        scalars.append(scalar.numpy())
        is_plant = action["type"] == "plant"
        plant.append(float(is_plant))
        packet.append(action["packet"] if is_plant else -1)
        cell.append(action["row"] * LAWN_COLS + action["col"] if is_plant else -1)
        values.append(target_values(frames[index].get("targets")))
    hints = [b'"type":"plant"' in line[max(line.rfind(b'"label":'), line.rfind(b'"action":')):]
             or b'"targets":' in line for line in source.splitlines()]
    return dict(hints=hints, rows=np.array(rows, dtype=np.int64),
                boards=np.array(boards, dtype=np.float16).reshape(-1, BOARD_CHANNELS, ROWS, COLS),
                scalars=np.array(scalars, dtype=np.float32).reshape(-1, SCALARS + 2),
                plant=np.array(plant, dtype=np.float32), packet=np.array(packet, dtype=np.int64),
                cell=np.array(cell, dtype=np.int64),
                values=np.array(values, dtype=np.float32).reshape(-1, PACKETS + 1))


def encode_v2lite_frames(frames: list[dict], source: bytes) -> dict:
    return encode_student_frames(frames, source, "v2lite")


def encode_student_frames(frames: list[dict], source: bytes, obs: str) -> dict:
    from pvz_student_observation import OBS_EXTRA, StudentHistory
    encoded = encode_frames(frames, source)
    history = StudentHistory(obs)
    boards, scalars = [], []
    rows = iter(encoded["rows"])
    wanted = next(rows, None)
    for index, frame in enumerate(frames[:-1], 1):
        board, scalar = history.extra(frame)
        if index == wanted:
            boards.append(board)
            scalars.append(scalar)
            wanted = next(rows, None)
    count = len(encoded["rows"])
    channels, extra_scalars = OBS_EXTRA[obs]
    encoded["boards"] = np.concatenate([encoded["boards"], np.asarray(boards, dtype=np.float16).reshape(count, channels, 6, 10)], axis=1)
    encoded["scalars"] = np.concatenate([encoded["scalars"], np.asarray(scalars, dtype=np.float32).reshape(count, extra_scalars)], axis=1)
    return encoded


@lru_cache(maxsize=3)
def encoder_signature(obs: str = "v1") -> str:
    import pvz_rollout_policy
    import pvz_value_net
    functions = (pvz_rollout_policy.features, pvz_value_net.featurize_numpy, pvz_value_net._col_of,
                 target_values, encode_frames)
    constants = {module.__name__: {name: value for name, value in vars(module).items()
                                  if name.isupper() and isinstance(value, (int, float))}
                 for module in (pvz_rollout_policy, pvz_value_net)}
    source = "".join(inspect.getsource(function) for function in functions)
    signature = hashlib.sha256((source + json.dumps(constants, sort_keys=True)).encode()).hexdigest()
    if obs == "v1":
        return signature
    if obs not in ("v2lite", "v2"):
        raise ValueError(f"unknown student observation version: {obs}")
    import pvz_student_observation
    return hashlib.sha256((signature + obs + inspect.getsource(pvz_student_observation)
                           + inspect.getsource(encode_student_frames)).encode()).hexdigest()


def _save(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            np.savez_compressed(temporary, **arrays)
            temporary.close()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)


# Frames per encode chunk. One chunk of boards is what the archive writer holds at a time
# (28 KB a frame) instead of a whole game's; see encode_frames_from_source.
ENCODE_CHUNK = 128


def encode_frames_from_source(source: bytes, chunk: int = ENCODE_CHUNK) -> dict:
    """``encode_frames`` over the frame JSON in ``source``, one chunk of frames at a time.

    Same arrays, same order, same dtype as ``encode_frames(frames, source)`` for the frames
    ``source`` holds: each chunk carries the two frames of context ``features`` reads, and its
    rows are re-based to absolute frame indices. Only the peak allocation differs -- one
    chunk of boards instead of the whole game's, which is what made the archive write the
    tallest point of a collection worker (task18).
    """
    from pvz_rollout_policy import PACKETS
    from pvz_value_net import BOARD_CHANNELS, COLS, ROWS, SCALARS
    lines = source.splitlines(keepends=True)
    if not lines:
        return encode_frames([], source)
    # A row is a frame index from 1 to len(lines)-1, so the arrays can be allocated once and
    # filled chunk by chunk: accumulating one array per chunk and concatenating them at the end
    # would hold the whole game's boards twice, which is the spike this function exists to avoid.
    limit = len(lines) - 1
    out = {"rows": np.empty(limit, dtype=np.int64),
           "boards": np.empty((limit, BOARD_CHANNELS, ROWS, COLS), dtype=np.float16),
           "scalars": np.empty((limit, SCALARS + 2), dtype=np.float32),
           "plant": np.empty(limit, dtype=np.float32),
           "packet": np.empty(limit, dtype=np.int64),
           "cell": np.empty(limit, dtype=np.int64),
           "values": np.empty((limit, PACKETS + 1), dtype=np.float32)}
    hints: list[bool] = []
    filled = 0
    for start in range(0, len(lines), chunk):
        end = min(start + chunk, len(lines))
        base = max(0, start - 2)                 # features() reads frame i-1 and frame i-2
        encoded = encode_frames([json.loads(line) for line in lines[base:end]], b"".join(lines[base:end]))
        keep = encoded["rows"] >= start - base    # rows are chunk-local; drop the overlap
        count = int(keep.sum())
        out["rows"][filled:filled + count] = encoded["rows"][keep] + base   # re-base to the whole game
        for name in ("boards", "scalars", "plant", "packet", "cell", "values"):
            out[name][filled:filled + count] = encoded[name][keep] if not keep.all() else encoded[name]
        hints.extend(encoded["hints"][start - base:])
        filled += count
    for name in out:
        out[name] = out[name][:filled]
    return dict(hints=hints, **out)


def write_frames_from_source(path: Path, source: bytes) -> None:
    """``write_frames`` for a caller that already holds the frame JSON as bytes.

    A collection worker writes one JSON line per decision anyway; keeping the parsed frames
    alive for the whole game only to re-encode them costs ~18 MB of dicts and makes the
    writer hold every board twice (task18).
    """
    if path.suffix == ".jsonl":
        path.write_bytes(source)
        path.with_suffix(".npz").unlink(missing_ok=True)
        return
    _save(path, frames=np.frombuffer(source, dtype=np.uint8), encoder=encoder_signature(),
          **encode_frames_from_source(source))


def write_frames(path: Path, frames: list[dict]):
    source = "".join(json.dumps(frame, separators=(",", ":")) + "\n" for frame in frames).encode()
    if path.suffix == ".jsonl":
        path.write_bytes(source)
        path.with_suffix(".npz").unlink(missing_ok=True)
        return
    _save(path, frames=np.frombuffer(source, dtype=np.uint8), encoder=encoder_signature(),
          **encode_frames(frames, source))


def cache_episode(path: Path, directory: Path, obs: str = "v1") -> Path:
    signature = encoder_signature(obs)
    if obs != "v1":
        directory = directory / obs
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as stored:
            if "encoder" in stored and str(stored["encoder"]) == signature:
                return path
    source = frame_bytes(path)
    digest = hashlib.sha256(signature.encode() + source).hexdigest()
    target = directory / (hashlib.sha256(str(path.resolve()).encode()).hexdigest() + ".npz")
    if target.exists():
        with np.load(target, allow_pickle=False) as stored:
            if str(stored["digest"]) == digest:
                return target
    frames = [json.loads(line) for line in source.splitlines()]
    encoded = encode_frames(frames, source) if obs == "v1" else encode_student_frames(frames, source, obs)
    _save(target, digest=digest, **encoded)
    return target
