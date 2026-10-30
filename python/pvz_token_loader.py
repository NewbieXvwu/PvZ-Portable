"""Block shuffled token batches, decoded by workers into bounded shared buffers.

Only the existing token_data mmap is opened by workers. No persistent loader files.
Workers use NumPy; Torch is imported only by the consuming process.
"""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context, shared_memory
import struct

import numpy as np

from pvz_token_blocks import decode

_data = None


def _open_data(path):
    global _data
    _data = np.load(path, mmap_mode='r', allow_pickle=False)


def _decode_job(work):
    job, batch_size = work
    pieces = []
    for address, local, indices in job:
        size, = struct.unpack_from('<I', _data, address)
        block = decode(_data[address+4:address+4+size]).arrays
        lengths = np.diff(block['offsets'])[local]
        pieces.append((block, local, indices, lengths))
    rows = sum(len(p[1]) for p in pieces)
    lengths = np.concatenate([p[3] for p in pieces])
    capacities = np.maximum.reduceat(lengths, np.arange(0, rows, batch_size))
    widths = np.repeat(capacities, batch_size)[:rows]
    offsets = np.r_[0, np.cumsum(widths)]
    padded = int(offsets[-1])
    fields = dict(ids=(np.int64, (padded, 5)),
                  features=(np.float32, (padded, 32)),
                  target_indices=(np.int64, (padded,)), mask=(np.bool_, (padded,)),
                  packet_index=(np.int64, (rows, 10)), cell_index=(np.int64, (rows, 54)),
                  terrain=(np.int64, (rows,)), indices=(np.int64, (rows,)),
                  offsets=(np.int64, (rows+1,)))
    layout, size = {}, 0
    for key, (dtype, shape) in fields.items():
        size = (size+7)//8*8
        layout[key] = (np.dtype(dtype).str, shape, size)
        size += np.dtype(dtype).itemsize * int(np.prod(shape))
    memory = shared_memory.SharedMemory(create=True, size=size)
    try:
        arrays = _views(memory, layout)
        arrays['offsets'][:] = offsets
        pos = 0
        for a, local, indices, lengths in pieces:
            stop = pos + len(local)
            # Expand sparse features once for the whole block, never once per row.
            features = np.zeros((len(a['ids']), 32), np.float32)
            rr = np.repeat(np.arange(len(features)), a['counts'])
            features[rr, a['columns']] = a['values']
            lo, hi = offsets[pos], offsets[stop]
            repeat = widths[pos:stop]
            within = np.arange(lo, hi) - np.repeat(offsets[pos:stop], repeat)
            valid = within < np.repeat(lengths, repeat)
            take = np.repeat(a['offsets'][local], repeat) + within
            take = np.minimum(take, np.repeat(a['offsets'][local+1]-1, repeat))
            ids = arrays['ids'][lo:hi]
            ids[:] = a['ids'][take]
            ids[~valid] = [0, 0, 0, -1, -1]
            values = arrays['features'][lo:hi]
            values[:] = features[take]
            values[~valid] = 0
            targets = arrays['target_indices'][lo:hi]
            targets[:] = a['target_indices'][take]
            targets[~valid] = -1
            arrays['mask'][lo:hi] = valid & ~((ids[:, 0] == 8) & (ids[:, 1] == 0))
            for key in ('packet_index', 'cell_index', 'terrain'):
                arrays[key][pos:stop] = a[key][local]
            arrays['indices'][pos:stop] = indices
            pos = stop
        return memory.name, layout
    except BaseException:
        memory.unlink()
        raise
    finally:
        memory.close()


def _views(memory, layout):
    return {key: np.ndarray(shape, dtype=np.dtype(dtype), buffer=memory.buf, offset=offset)
            for key, (dtype, shape, offset) in layout.items()}


def _release(result):
    memory = shared_memory.SharedMemory(name=result[0])
    memory.unlink()
    memory.close()


class TokenBlockLoader:
    """A persistent worker pool; each pending job holds about 512 selected rows.

    Selection is supplied by the trainer, so validation and zero-weight exclusions
    cannot be shuffled into training. A block is sequential inside a shuffled block
    stream; selected rows are shuffled only within their original 32-row block.
    """
    def __init__(self, store, workers=4, prefetch=2):
        self.store = store
        self.starts = np.flatnonzero(store.arrays['rows'] == 0)
        self.ends = np.r_[self.starts[1:], len(store.arrays['rows'])]
        self.workers, self.prefetch = workers, prefetch
        self.pool = None

    def __enter__(self):
        self.pool = ProcessPoolExecutor(self.workers, mp_context=get_context('spawn'),
            initializer=_open_data, initargs=(str(self.store.arrays['data'].filename),))
        return self

    def __exit__(self, *exc):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def _jobs(self, indices, batch_size, seed, rows):
        keep = np.zeros(len(self.store.arrays['rows']), bool)
        keep[np.asarray(indices)] = True
        active = np.flatnonzero(np.logical_or.reduceat(keep, self.starts))
        rng = np.random.default_rng(seed)
        target = max(1, 512//batch_size) * batch_size
        job, count, remaining = [], 0, rows
        while remaining:
            for number in rng.permutation(active):
                start, end = self.starts[number], self.ends[number]
                selected = rng.permutation(np.flatnonzero(keep[start:end]))
                while len(selected) and remaining:
                    n = min(target-count, len(selected), remaining)
                    local, selected = selected[:n], selected[n:]
                    job.append((int(self.store.arrays['offsets'][start]), local, local+start))
                    count += n
                    remaining -= n
                    if count == target:
                        yield job, batch_size
                        job, count = [], 0
                if not remaining:
                    break
        if job:
            yield job, batch_size

    def batches(self, indices, batch_size, seed, rows=None):
        """Close this generator when an epoch stops early to release prefetched buffers."""
        import torch
        jobs = iter(self._jobs(indices, batch_size, seed, len(indices) if rows is None else rows))
        pending = deque()
        try:
            for _ in range(self.workers*self.prefetch):
                job = next(jobs, None)
                if job is not None:
                    pending.append(self.pool.submit(_decode_job, job))
            while pending:
                result = pending.popleft().result()
                job = next(jobs, None)
                if job is not None:
                    pending.append(self.pool.submit(_decode_job, job))
                memory = shared_memory.SharedMemory(name=result[0])
                try:
                    arrays = _views(memory, result[1])
                    for at in range(0, len(arrays['indices']), batch_size):
                        stop = min(at+batch_size, len(arrays['indices']))
                        lo, hi = arrays['offsets'][at], arrays['offsets'][stop]
                        capacity = int((hi-lo)//(stop-at))
                        tokens = {key: torch.from_numpy(value[lo:hi].reshape(stop-at, capacity, *value.shape[1:])
                                  if key in ('ids', 'features', 'mask', 'target_indices') else value[at:stop])
                                  for key, value in arrays.items() if key not in ('indices', 'offsets')}
                        yield torch.from_numpy(arrays['indices'][at:stop]), tokens
                finally:
                    memory.unlink()
                    memory.close()
        finally:
            for future in pending:
                if not future.cancel():
                    _release(future.result())
