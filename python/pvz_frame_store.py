"""Lossless episode archives with precomputed student training rows."""
from functools import lru_cache
from contextlib import contextmanager
import hashlib
import inspect
import io
import json
from pathlib import Path
import tempfile
import re
import zipfile

import numpy as np

# LEGACY(jsonl-archives): the JSONL half of the glob is for archives written before frames.npz.
def frame_paths(data: Path):
    paths = {path.parent: path for path in data.glob("*/*/frames.jsonl")}
    paths.update({path.parent: path for path in data.glob("*/*/frames.npz")})
    return sorted(paths.values(), key=lambda path: str(path.parent))


def archive_seed(directory: str) -> int:
    """Seed of a `seed_<n>` archive directory; demo-start games are `seed_<demo>_f<future>` and go by the future seed."""
    tag = directory.removeprefix("seed_").rsplit("_f", 1)[-1]
    return int(tag)


def frame_path(archive: Path) -> Path:
    binary = archive / "frames.npz"
    return binary if binary.is_file() else archive / "frames.jsonl"


# LEGACY(jsonl-archives): text archives are read, never written, from here on.
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


# LEGACY(aux-labels): side cache of the auxiliary damage head (task16 experiment).
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


# LEGACY(obs-versions): the signature freezes which encoder wrote an archive, so v1/v2lite/v2 stay readable.
@lru_cache(maxsize=5)
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
    if obs not in ("v2lite", "v2", "v2fog", "v2size", "v2boss", "v2bungee", "v2sleep",
                   "v2timer"):
        raise ValueError(f"unknown student observation version: {obs}")
    # v2lite/v2 are frozen encoder semantics. Adding a new version in their shared
    # module must not invalidate all historical archives and rewrite every pack.
    # Keep the original module-derived fingerprints for the unchanged base encoder;
    # future feature changes must introduce a new observation version, as v2fog does.
    if signature == '5c34482620514a7434e638058f9b75d05e51a8aa9c18b634a8f80e4dd424f008':
        frozen = {
            'v2lite': 'b6020e4131d35d7ff9ef0fb0996f877e6f3ffd2122045a65bd5e87be2c065aa5',
            'v2': '5031d93614db9ecff271dcaa6aa8645d7a799452f170e5ebf699cc2fb8c1df91',
            'v2fog': '17aba0780afb4005daeb1cff6f50ca6048179a7c83b4fae0bbde35437e4036e8',
            'v2size': '400d14ae3def3b970f2e19593b11c5f0a54a9d00da596fa2f5ce19710be75886',
            'v2boss': 'b5d77ca232300a2d7331ae572ace07ef1c58757f3ef5e2818c3fb5a691df3bc4',
            'v2bungee': '82609c46ed14655cead5910457ee9c0a627d77948b896c8afef09b012a2bdb6f',
            # task145: v2sleep 冻结在这里，新增的 v2timer 分支不能改写已有 v2sleep 档案的签名。
            'v2sleep': '309266015e12322c291071ddbf1c2e952fb1e7bc2f2248f4cac1ca9b38f41029',
        }
        if obs in frozen:
            return frozen[obs]
    import pvz_student_observation
    # 每个版本把自己的特征函数源码也放进指纹：只有它变了，这个版本才该重建缓存。
    extras = {'v2sleep': (pvz_value_net.sleep_features,),
              'v2timer': (pvz_value_net.sleep_features, pvz_value_net.timer_features)}.get(obs, ())
    return hashlib.sha256((signature + obs + inspect.getsource(pvz_student_observation)
                           + inspect.getsource(encode_student_frames)
                           + "".join(inspect.getsource(function) for function in extras)).encode()).hexdigest()


# LZMA instead of zlib: on a sample of 12 archives, ZIP_LZMA is 59.6% of np.savez_compressed
# (235 KB -> 140 KB per episode), because the frames JSON is the bulk and LZMA models its repetition
# across the whole file. Reading costs 12 ms instead of 4 ms per archive and writing 0.63 s instead
# of 0.06 s, both negligible next to one game. np.load reads it without any change (zipfile handles
# the codec); packs keep ZIP_STORED in scripts/pack_training_rows.py so they can be memory-mapped.
def _save(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_LZMA) as archive:
                for name, array in arrays.items():
                    buffer = io.BytesIO()
                    np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
                    archive.writestr(name + ".npy", buffer.getvalue())
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


def save_arrays(path: Path, **arrays) -> None:
    """Write an archive/cache npz verbatim -- the same container _save uses (LZMA)."""
    _save(path, **arrays)


def write_frames(path: Path, frames: list[dict], *, cache: bool = True):
    source = "".join(json.dumps(frame, separators=(",", ":")) + "\n" for frame in frames).encode()
    if path.suffix == ".jsonl":
        path.write_bytes(source)
        path.with_suffix(".npz").unlink(missing_ok=True)
        return
    if not cache:
        # Interface-only archives use the same lossless container; training builds its cache downstream.
        _save(path, frames=np.frombuffer(source, dtype=np.uint8))
        return
    _save(path, frames=np.frombuffer(source, dtype=np.uint8), encoder=encoder_signature(),
          **encode_frames(frames, source))


class _ZeroPaddedV2Cache:
    def __init__(self, stored):
        self.stored = stored

    def __getitem__(self, name):
        value = self.stored[name]
        if name == 'boards':
            return np.pad(value, ((0, 0), (0, 12), (0, 0), (0, 0)))
        if name == 'scalars':
            return np.pad(value, ((0, 0), (0, 2)))
        return value

    def __contains__(self, name):
        return name in self.stored


class _TimerCache:
    """v2timer rows derived from a v2sleep cache: 4 computed channels appended on read.

    task147: the only difference between the two versions is those four channels, and they can
    be computed from the frame JSON without touching the 113 channels underneath. Storing them
    next to a pointer avoids rewriting the 113 parent channels. Keep the parent cache
    alongside the view, just as for the existing zero-padding views.
    """
    def __init__(self, stored, timer):
        self.stored, self.timer = stored, timer

    def __getitem__(self, name):
        if name == 'timer':
            return self.timer
        value = self.stored[name]
        return np.concatenate([value, self.timer], axis=1) if name == 'boards' else value

    def __contains__(self, name):
        if name == 'timer':
            return True
        return name in self.stored


@contextmanager
def read_episode_cache(path: Path):
    """Read ordinary rows, a zero-padding view over a v2 cache, or a v2timer timer view."""
    with np.load(path, allow_pickle=False) as stored:
        if 'timer_from' in stored.files:
            with read_episode_cache(Path(str(stored['timer_from']))) as parent:
                yield _TimerCache(parent, stored['timer'])
            return
        if 'zero_pad_from' not in stored.files:
            yield stored
            return
        with np.load(str(stored['zero_pad_from']), allow_pickle=False) as parent:
            if str(parent['digest']) != str(stored['parent_digest']):
                raise ValueError(f'zero-padding cache parent changed: {path}')
            yield _ZeroPaddedV2Cache(parent)


def _derive_timer_cache(source: bytes, digest: str, target: Path,
                        base_directory: Path, read_directory: Path | None) -> Path | None:
    """Write the v2timer view over an existing v2sleep cache, or None if there is no parent."""
    from pvz_value_net import timer_features
    parent_digest = hashlib.sha256(encoder_signature('v2sleep').encode() + source).hexdigest()
    roots = [base_directory] if read_directory is None else [read_directory, base_directory]
    for root in roots:
        parent = root / 'v2sleep' / target.name
        if not parent.is_file():
            continue
        with np.load(parent, allow_pickle=False) as stored:
            if str(stored['digest']) != parent_digest:
                continue
        try:
            with read_episode_cache(parent) as stored:
                rows = np.asarray(stored['rows'], dtype=np.int64)
        except (FileNotFoundError, ValueError):
            continue
        lines = source.splitlines()
        channels = np.empty((len(rows), 4, 6, 10), dtype=np.float16)
        for offset, index in enumerate(rows):
            channels[offset] = timer_features(json.loads(lines[index - 1]))
        _save(target, digest=digest, timer_from=str(parent.resolve()), timer=channels)
        return target
    return None


def cache_episode(path: Path, directory: Path, obs: str = "v1", read_directory: Path | None = None) -> Path:
    signature = encoder_signature(obs)
    base_directory = directory
    if obs != "v1":
        directory = directory / obs
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as stored:
            if "encoder" in stored and str(stored["encoder"]) == signature:
                return path
    source = frame_bytes(path)
    digest = hashlib.sha256(signature.encode() + source).hexdigest()
    target = directory / (hashlib.sha256(str(path.resolve()).encode()).hexdigest() + ".npz")
    candidates = [target]
    if read_directory is not None:
        read_root = read_directory if obs == 'v1' else read_directory / obs
        candidates.insert(0, read_root / target.name)
    for candidate in candidates:
        if candidate.exists():
            with np.load(candidate, allow_pickle=False) as stored:
                if str(stored["digest"]) == digest:
                    if 'timer_from' in stored.files:
                        try:
                            with read_episode_cache(Path(str(stored['timer_from']))):
                                pass
                        except (FileNotFoundError, ValueError):
                            continue
                    if 'zero_pad_from' in stored.files:
                        parent = Path(str(stored['zero_pad_from']))
                        if not parent.is_file():
                            continue
                        with np.load(parent, allow_pickle=False) as previous:
                            if str(previous['digest']) != str(stored['parent_digest']):
                                continue
                    return candidate
    # Old v2 archives have none of the newly observed fields. Reuse their
    # already encoded rows, adding zero channels, instead of featurizing every
    # decision again. Bytes are read here anyway to compute the cache digest.
    # Escaped JSON keys and archives with any new field take the normal encoder.
    if obs == 'v2sleep' and not any(token in source for token in
                                   (b'"boss"', b'"bungee"', b'"sleep"', b'\\u')):
        old_digest = hashlib.sha256(encoder_signature('v2').encode() + source).hexdigest()
        old_roots = [base_directory] if read_directory is None else [read_directory, base_directory]
        for root in old_roots:
            candidate = root / 'v2' / target.name
            if not candidate.is_file():
                continue
            with np.load(candidate, allow_pickle=False) as stored:
                if str(stored['digest']) != old_digest:
                    continue
            # A small view avoids recompressing all dense historical boards.
            # The old cache remains untouched; absent/changed parents rebuild.
            _save(target, digest=digest, zero_pad_from=str(candidate.resolve()), parent_digest=old_digest)
            return target
    # task147: v2timer is v2sleep plus four channels that come from the frame JSON alone. When the
    # v2sleep cache exists, copy its row indices, compute only those four channels, and store a
    # small view. The alternative -- featurizing and LZMA-compressing 117 channels per archive --
    # is what made the cold rebuild 7-15 h (see MEMORY/docs/task145-obs-timer.md).
    if obs == 'v2timer':
        derived = _derive_timer_cache(source, digest, target, base_directory, read_directory)
        if derived is not None:
            return derived
    frames = [json.loads(line) for line in source.splitlines()]
    encoded = encode_frames(frames, source) if obs == "v1" else encode_student_frames(frames, source, obs)
    _save(target, digest=digest, **encoded)
    return target


def encode_packets(frames: list[dict]) -> dict:
    """Aligned identity inputs for frame i-1 -> labelled action at i; never infer deck from slot number."""
    from pvz_rollout_policy import PACKETS, packet_features
    rows, packets = [], []
    for i in range(1, len(frames)):
        if not (frames[i].get('label') or frames[i].get('action')) or frames[i].get('action_rejected'):
            continue
        rows.append(i)
        packets.append(packet_features(frames[i - 1]).numpy())
    return dict(rows=np.asarray(rows, np.int64),
                packets=np.asarray(packets, np.float32).reshape(-1, PACKETS, 4))


def cache_packets(path: Path, directory: Path, source: bytes | None = None) -> Path:
    """Identity-only side cache; old feature signatures/archives stay unchanged."""
    from pvz_rollout_policy import PACKETS, PlantTypes, packet_features
    source = frame_bytes(path) if source is None else source
    signature = inspect.getsource(packet_features) + inspect.getsource(encode_packets) + str((PACKETS, PlantTypes))
    digest = hashlib.sha256(signature.encode() + source).hexdigest()
    target = directory / 'packets' / (hashlib.sha256(str(path.resolve()).encode()).hexdigest() + '.npz')
    if target.exists():
        with np.load(target, allow_pickle=False) as stored:
            if str(stored['digest']) == digest:
                return target
    _save(target, digest=digest, **encode_packets([json.loads(line) for line in source.splitlines()]))
    return target


def identity_scene_scalars(source: bytes) -> np.ndarray:
    """Read every frame's scene exactly; flat, uniquely named scene objects avoid decoding entities.

    Nested/escaped/ambiguous scenes fall back to the full JSON parser. No assumption that wave
    count is constant, and no inference from a scalar already corrupted by an empty seed slot.
    """
    from pvz_rollout_policy import identity_wave_scalar
    pattern = re.compile(rb'"scene"\s*:\s*(\{[^{}\n]*\})')
    result = []
    for line in source.splitlines():
        match = pattern.search(line) if b'\\u' not in line and line.count(b'"scene"') == 1 else None
        scene = json.loads(match[1]) if match else json.loads(line)['scene']
        result.append(identity_wave_scalar({'scene': scene}))
    return np.asarray(result, dtype=np.float32)


def cache_identity_scene(path: Path, directory: Path) -> Path:
    from pvz_rollout_policy import identity_wave_scalar
    source = frame_bytes(path)
    signature = inspect.getsource(identity_scene_scalars) + inspect.getsource(identity_wave_scalar)
    digest = hashlib.sha256(signature.encode() + source).hexdigest()
    target = directory / 'identity_scene' / (hashlib.sha256(str(path.resolve()).encode()).hexdigest() + '.npz')
    if target.exists():
        with np.load(target, allow_pickle=False) as stored:
            if str(stored['digest']) == digest:
                return target
    with np.load(cache_packets(path, directory, source=source), allow_pickle=False) as stored:
        rows = stored['rows']
    scalars = identity_scene_scalars(source)
    _save(target, digest=digest, rows=rows, wave_scalar=scalars[rows - 1])
    return target
