"""The CUDA rollout server as the search sees it (``cuda/RolloutCuda.cu --serve``).

Why a server and not a subprocess per continuation: a continuation costs ~49 rounds, and a round costs
~10 ms of *device* time whether the batch holds 8 lanes or 1000 (``~/pvz-logs/cuda_status.md`` round 9).
One search decision is 10-40 lanes, which is under the point where the GPU beats even a single CPU
thread -- so the only way this backend can pay is to put several workers' decisions into one wave.  A
process that outlives the request is what makes that possible: starting one costs 1.4 s, more than the
decision it would serve.

Protocol (the server's comment block restates it): one ``AF_UNIX`` stream socket, ``RUN`` lines with an
inline pair array, ``OK <lanes> <resultBytes> <leafBytes>`` answers.  Nothing here is a second
simulator: the same ``RunBatch`` serves both front doors, and ``/tmp/smoke_serve.py`` checks that the
records the two doors return are byte-identical.

Leaves come back as the value net's own input (5668 float32 per lane: board[92][6][10], scalars[113],
task context[35], the order of ``pvz_value_net.featurize_numpy``), so scoring a GPU continuation does not
need an observation round trip.
"""
from __future__ import annotations

import os
import socket
import struct
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

RECORD = struct.Struct("<" + "i" * 33)
PAIR = struct.Struct("<" + "i" * 7)             # root, future seed, kind, packet, row, col, ticks
VALUE_FLOATS = 5668
BOARD_CHANNELS, ROWS, COLS = 92, 6, 10
SCALAR_FEATURES = 113
CONTEXT_FEATURES = 35
# RolloutResult, in cuda/CudaRollout.h declaration order: 17 scalars, plantedByType[10], rowMinZombieX[6].
FIELDS = ("rootIndex", "decisions", "netCalls", "ticks", "endTick", "result", "sun", "wave",
          "zombiesAlive", "zombiesKilled", "plantsAlive", "plantsEaten", "plantsPlanted",
          "mowersLeft", "mowersTriggered", "sunProduced", "errorFlags")

# cuda/Rollout.inc's RolloutBegin reads the forced first action as five int32: kind, packet, row, col,
# ticks.  kind 0 lets the student decide, 1 plants, 2 waits.
KIND_STUDENT, KIND_PLANT, KIND_WAIT = 0, 1, 2


class CudaServerError(RuntimeError):
    """The server refused a request or went away.  Callers fall back to the CPU rollout."""


def first_action_words(action: dict[str, Any] | None) -> tuple[int, int, int, int, int]:
    """The five lane words for a teacher candidate, or "the student decides" for ``None``."""
    if not action or action.get("type") != "plant":
        ticks = (action or {}).get("ticks") or 60
        return (KIND_WAIT if action else KIND_STUDENT), -1, 0, 0, int(ticks)
    return KIND_PLANT, int(action["packet"]), int(action["row"]), int(action["col"]), 0


def _int32(value: int) -> int:
    """Wrap to the signed 32 bits the lane fields hold, keeping the bit pattern.

    A future seed is `decision_seed * 1000 + k` modulo 2**32, which is routinely above INT32_MAX; the
    generator's stream is the *uint32* of it (cuda/SimCore.h's Rand::Seed), so wrapping here is the same
    future, while refusing to wrap would be a struct error.
    """
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value >= 0x80000000 else value


def encode_pairs(pairs: Sequence[tuple[int, int, dict[str, Any] | None]]) -> bytes:
    """`(root index, future seed, candidate action)` triples as the server's wire format."""
    return b"".join(PAIR.pack(*[_int32(word) for word in (root, seed, *first_action_words(action))])
                    for root, seed, action in pairs)


class CudaRolloutClient:
    """One connection to a running server.

    ``run`` is the synchronous form the search uses.  ``begin``/``answer`` split it in two, because two
    connections with requests outstanding at once are what the server's gather window merges into one
    wave -- and that merge is the only reason the seam beats running the batch door per decision.
    """

    def __init__(self, socket_path: str | os.PathLike[str], timeout: float = 600.0):
        self.socket_path = str(socket_path)
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._buffer = bytearray()
        self._pending = 0
        self._started = 0.0
        self.capacity = 0
        self.batches = 0
        self.lanes = 0
        self.seconds = 0.0

    def connect(self) -> int:
        if self._sock is not None:
            return self.capacity
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(self.timeout)
        self._sock.connect(self.socket_path)
        self._send(b"PING\n")
        line = self._read_line()
        if not line.startswith("PONG"):
            raise CudaServerError(f"unexpected greeting {line!r} from {self.socket_path}")
        self.capacity = int(line.split()[1])
        return self.capacity

    def close(self) -> None:
        if self._sock is None:
            return
        try:
            self._send(b"QUIT\n")
            self._read_line()
        except OSError:
            pass
        self._sock.close()
        self._sock = None

    def status(self) -> dict[str, int]:
        """Server counters, for verifying that concurrent clients actually shared GPU waves."""
        self.connect()
        if self._pending:
            raise CudaServerError("cannot query status while a rollout answer is pending")
        self._send(b"PING\n")
        words = self._read_line().split()
        if len(words) != 4 or words[0] != "PONG":
            raise CudaServerError(f"unexpected status {words!r}")
        return dict(zip(("capacity", "lanes", "waves"), map(int, words[1:])))

    def run(self, roots_path: str | os.PathLike[str],
            pairs: Sequence[tuple[int, int, dict[str, Any] | None]], *, horizon: int,
            max_actions: int = 10 ** 9, want_leaf: bool = True
            ) -> tuple[list[dict[str, Any]], Any]:
        """Continue every pair from the corpus at ``roots_path``; returns (records, leaf features).

        ``leaf features`` is a ``(lanes, VALUE_FLOATS)`` float32 numpy array, or ``None`` when
        ``want_leaf`` is false.  A record's ``result`` is 1 / -1 / 0 for won / lost / still playing when
        the horizon cut it off, and ``errorFlags`` is nonzero for a lane whose root would not import.
        """
        if not pairs:
            return [], None
        self.begin(roots_path, pairs, horizon=horizon, max_actions=max_actions, want_leaf=want_leaf)
        return self.answer()

    def begin(self, roots_path: str | os.PathLike[str],
              pairs: Sequence[tuple[int, int, dict[str, Any] | None]], *, horizon: int,
              max_actions: int = 10 ** 9, want_leaf: bool = True) -> int:
        """Send a request without waiting for the answer, and return how many pairs it holds.

        One connection carries one request at a time (the server answers requests in the order their
        waves finish, not the order they arrived), but two connections with outstanding requests are
        what the server's gather window merges into a single wave -- which is the whole point of the
        seam, so the send and the read have to be separable.
        """
        if self._sock is None:
            self.connect()
        if len(pairs) > max(self.capacity, 1) * 64:
            # The server slices a request across waves, so this is only about keeping one worker's
            # backlog from outrunning its own memory; the search never comes close.
            raise CudaServerError(f"{len(pairs)} pairs is too many for one request")
        header = ("RUN horizon=%d max_actions=%d leaf=%d pairs=%d roots=%s\n"
                  % (horizon, max_actions, 1 if want_leaf else 0, len(pairs), roots_path)).encode()
        self._pending = len(pairs)
        self._started = time.time()
        self._send(header + encode_pairs(pairs))
        return self._pending

    def answer(self) -> tuple[list[dict[str, Any]], Any]:
        """Collect the request `begin` sent on this connection."""
        expected, started = self._pending, self._started
        self._pending = 0
        line = self._read_line()
        answer = line.split()
        if not answer or answer[0] != "OK":
            raise CudaServerError(f"server refused the request: {line}")
        lanes, result_bytes, leaf_bytes = int(answer[1]), int(answer[2]), int(answer[3])
        if lanes != expected or result_bytes != lanes * RECORD.size:
            raise CudaServerError(f"server answered {line} for {expected} pairs")
        blob = self._read_exactly(result_bytes + leaf_bytes)
        records = [_record(blob[index * RECORD.size:(index + 1) * RECORD.size]) for index in range(lanes)]
        leaves = None
        if leaf_bytes:
            import numpy as np
            if leaf_bytes != lanes * VALUE_FLOATS * 4:
                raise CudaServerError(f"{leaf_bytes} leaf bytes for {lanes} lanes")
            leaves = np.frombuffer(blob[result_bytes:result_bytes + leaf_bytes], dtype=np.float32)
            leaves = leaves.reshape(lanes, VALUE_FLOATS).copy()
        self.batches += 1
        self.lanes += lanes
        self.seconds += time.time() - started
        return records, leaves

    # --- framing -----------------------------------------------------------

    def _send(self, payload: bytes) -> None:
        assert self._sock is not None
        self._sock.sendall(payload)

    def _read_line(self) -> str:
        while b"\n" not in self._buffer:
            piece = self._sock.recv(65536)
            if not piece:
                raise CudaServerError(f"server closed the connection ({self.socket_path})")
            self._buffer += piece
        end = self._buffer.index(b"\n")
        line = self._buffer[:end].decode(errors="replace")
        del self._buffer[:end + 1]
        return line

    def _read_exactly(self, count: int) -> bytes:
        while len(self._buffer) < count:
            piece = self._sock.recv(max(65536, count - len(self._buffer)))
            if not piece:
                raise CudaServerError(f"server closed the connection mid-answer ({self.socket_path})")
            self._buffer += piece
        blob = bytes(self._buffer[:count])
        del self._buffer[:count]
        return blob


def _record(blob: bytes) -> dict[str, Any]:
    words = RECORD.unpack(blob)
    record = dict(zip(FIELDS, words[:len(FIELDS)]))
    record["plantedByType"] = list(words[len(FIELDS):len(FIELDS) + 10])
    record["rowMinZombieX"] = list(words[len(FIELDS) + 10:])
    return record


def leaf_tensors(leaves):
    """The GPU leaf features as the value net's own inputs: (board, scalars, context) torch tensors."""
    import torch
    array = torch.from_numpy(leaves)
    board = array[:, :BOARD_CHANNELS * ROWS * COLS].reshape(-1, BOARD_CHANNELS, ROWS, COLS)
    scalars = array[:, BOARD_CHANNELS * ROWS * COLS:BOARD_CHANNELS * ROWS * COLS + SCALAR_FEATURES]
    context = array[:, BOARD_CHANNELS * ROWS * COLS + SCALAR_FEATURES:]
    return board, scalars, context


def launch_server(binary: str | os.PathLike[str], net: str | os.PathLike[str],
                  socket_path: str | os.PathLike[str], *, lanes: int = 2048, gather_ms: int = 80,
                  start_timeout_ms: int = 120000, env: dict[str, str] | None = None,
                  log: Any = None) -> subprocess.Popen:
    """Start ``--serve`` and wait for its READY line.  The caller owns the process."""
    Path(socket_path).unlink(missing_ok=True)
    process = subprocess.Popen(
        [str(binary), "--serve", "--net", str(net), "--socket", str(socket_path),
         "--lanes", str(lanes), "--gather-ms", str(gather_ms),
         "--start-timeout-ms", str(start_timeout_ms)],
        stdout=subprocess.PIPE, stderr=(subprocess.PIPE if log is None else log), text=True,
        env={**os.environ, **(env or {})})
    line = ""
    deadline = time.time() + 120.0
    while time.time() < deadline:
        line = process.stdout.readline()
        if line.startswith("READY"):
            return process
        if not line and process.poll() is not None:
            break
    message = process.stderr.read() if process.stderr else ""
    process.kill()
    raise CudaServerError(f"server did not come up ({line!r}): {message}")


def serve_policy(binary: str | os.PathLike[str], policy_path: str | os.PathLike[str],
                 workdir: str | os.PathLike[str], *, lanes: int = 2048, gather_ms: int = 80,
                 start_timeout_ms: int = 600000, log: Any = None) -> tuple[subprocess.Popen, str]:
    """Export a torch student checkpoint to ``.pvzrn`` and start a server for it in ``workdir``.

    The caller passes this socket to its workers.  Both the game driver and the search-agreement probe
    need exactly this, and the GPU arm is only comparable to the CPU arm if both roll out with the *same*
    student -- so the export belongs next to the launch, not in each caller.
    """
    from pvz_rollout_policy import export_student, load_student
    directory = Path(workdir)
    net_path = directory / "student.pvzrn"
    export_student(load_student(policy_path), str(net_path))
    socket_path = str(directory / "serve.sock")
    return launch_server(binary, net_path, socket_path, lanes=lanes, gather_ms=gather_ms,
                         start_timeout_ms=start_timeout_ms, log=log), socket_path
