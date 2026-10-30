"""Memory-mapped sparse token sidecar; original float32 public features, no replay."""
import json
from pathlib import Path
import numpy as np

FIELDS = dict(ids=(np.int16, (5,)), counts=(np.uint8, ()), columns=(np.uint8, ()), values=(np.float32, ()),
              offsets=(np.int64, ()), feature_offsets=(np.int64, ()), stored_row=(np.int32, ()),
              packet_index=(np.int16, (10,)), terrain=(np.uint8, ()))


class ResidentRows:
    """Only the selected training rows resident in RAM; callers retain global row IDs."""
    def __init__(self, source, rows):
        import copy
        self.rows = np.asarray(rows)
        if hasattr(source, 'offsets') and hasattr(source, 'indices'):
            local = copy.copy(source)
            starts = source.offsets[self.rows]
            counts = source.offsets[self.rows + 1] - starts
            local.offsets = np.r_[0, counts.cumsum()]
            positions = (np.repeat(starts - local.offsets[:-1], counts)
                         + np.arange(local.offsets[-1]))
            local.indices = source.indices[positions].copy()
            local.values = source.values[positions].copy()
            local.plane_of = None if source.plane_of is None else source.plane_of[self.rows].copy()
            local.shape = (len(rows), *source.shape[1:])
            local.nnz = len(local.values)
            local._plane_device = {}
            self.data = local
        else:
            self.data = source[self.rows].clone()

    def batch(self, rows, device):
        return self.data.batch(np.searchsorted(self.rows, np.asarray(rows)), device)

    def __getitem__(self, rows):
        return self.data[np.searchsorted(self.rows, np.asarray(rows))]


class ArchiveTokens:
    """Sequential CPU evaluation reads one source at a time, without a disk token cache."""
    def __init__(self, sources):
        import sqlite3
        self.directory = Path(sources)
        self.db = sqlite3.connect(self.directory/'sources.sqlite')
        records = list(self.db.execute('SELECT source,n FROM sources ORDER BY source'))
        self.names = [name for name, n in records]
        self.offsets = np.r_[0, np.cumsum([n for name,n in records])]
        self.current, self.frames, self.rows = None, None, None

    def sample(self, i):
        from pvz_frame_store import read_frames
        from pvz_student_tokens import frame_tokens
        archive = int(np.searchsorted(self.offsets, i, side='right')-1)
        if archive != self.current:
            self.frames = read_frames(self.directory/'raw'/self.names[archive])
            self.rows = np.frombuffer(self.db.execute('SELECT rows FROM sources WHERE source=?', (self.names[archive],)).fetchone()[0], np.int32)
            self.current = archive
        return frame_tokens(self.frames[int(self.rows[i-self.offsets[archive]])-1])


class SparseTokens:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / 'ready.json').read_text())
        self.fallback = None
        self.arrays = {}
        for name, (dtype, shape) in FIELDS.items():
            path = self.directory / (name + '.bin')
            self.arrays[name] = (np.memmap(path, dtype=dtype, mode='r').reshape(-1, *shape) if path.stat().st_size
                                 else np.empty((0, *shape), dtype=dtype))

    def sample(self, i):
        a = self.arrays
        stored = int(np.searchsorted(a['stored_row'], i))
        if stored == len(a['stored_row']) or a['stored_row'][stored] != i:
            if self.fallback is None:
                self.fallback = ArchiveTokens(self.directory.parent/'split')
            return self.fallback.sample(i)
        lo, hi = a['offsets'][stored:stored+2]
        flo, fhi = a['feature_offsets'][stored:stored+2]
        features = np.zeros((hi-lo, 32), np.float32)
        rows = np.repeat(np.arange(hi-lo), a['counts'][lo:hi])
        features[rows, a['columns'][flo:fhi]] = a['values'][flo:fhi]
        ids = a['ids'][lo:hi]
        cells = np.zeros((54,5), np.int16)
        cells[:,0] = 2; cells[:,3] = np.arange(54)//9; cells[:,4] = np.arange(54)%9
        cell_features = np.zeros((54,32), np.float32)
        cell_features[:,0] = cells[:,3]/5; cell_features[:,1] = cells[:,4]/8
        cell_features[:,8] = np.isin(cells[:,3], ids[ids[:,0]==10,3])
        ids = np.concatenate([ids[:2],cells,ids[2:]])
        features = np.concatenate([features[:2],cell_features,features[2:]])
        packet_index = a['packet_index'][stored].astype(np.int32)
        packet_index[packet_index>=2] += 54
        return dict(ids=ids, features=features, target_indices=np.full(len(ids), -1, np.int32),
                    packet_index=packet_index, cell_index=np.arange(2,56,dtype=np.int32),
                    terrain=np.asarray(a['terrain'][stored], np.int64))

    def batch(self, indices, device='cpu'):
        from pvz_student_tokens import collate_tokens
        return collate_tokens([self.sample(int(i)) for i in indices], device)


def extra_validation(mask_path, rows):
    if mask_path is None:
        return None
    mask = np.load(mask_path, mmap_mode='r', allow_pickle=False)
    if mask.dtype != np.bool_ or mask.shape != (rows,):
        raise ValueError('heldout deck mask shape/dtype differs from frozen pack')
    return mask


def keep_training_rows(indices, fraction):
    """Same global-row hash for every architecture and model seed; no RNG mutation."""
    if fraction >= 1:
        return np.ones(len(indices), dtype=bool)
    value = np.asarray(indices, dtype=np.uint64) + np.uint64(162)
    value = (value ^ (value >> 16)) * np.uint64(0x45d9f3b)
    value = (value ^ (value >> 16)) * np.uint64(0x45d9f3b)
    value = value ^ (value >> 16)
    return (value & np.uint64(0xffffffff)) < int(fraction * (1 << 32))
