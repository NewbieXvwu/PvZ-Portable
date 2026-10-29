"""Memory-mapped sparse token sidecar; original float32 public features, no replay."""
import json
from pathlib import Path
import numpy as np

FIELDS = dict(ids=(np.int16, (5,)), counts=(np.uint8, ()), columns=(np.uint8, ()), values=(np.float32, ()),
              offsets=(np.int64, ()), feature_offsets=(np.int64, ()), packet_index=(np.int32, (10,)),
              cell_index=(np.int32, (54,)), terrain=(np.int64, ()))


class SparseTokens:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / 'ready.json').read_text())
        self.arrays = {}
        for name, (dtype, shape) in FIELDS.items():
            path = self.directory / (name + '.bin')
            self.arrays[name] = np.memmap(path, dtype=dtype, mode='r').reshape(-1, *shape)

    def sample(self, i):
        a = self.arrays
        lo, hi = a['offsets'][i:i+2]
        flo, fhi = a['feature_offsets'][i:i+2]
        features = np.zeros((hi-lo, 32), np.float32)
        rows = np.repeat(np.arange(hi-lo), a['counts'][lo:hi])
        features[rows, a['columns'][flo:fhi]] = a['values'][flo:fhi]
        return dict(ids=a['ids'][lo:hi], features=features, target_indices=np.full(hi-lo, -1, np.int32),
                    **{name: a[name][i] for name in ('packet_index', 'cell_index', 'terrain')})

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
