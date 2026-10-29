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
        self.arrays = {}
        if self.directory.is_file():
            with np.load(self.directory, allow_pickle=False) as stored:
                self.meta = json.loads(str(stored['meta']))
                for name, (dtype, shape) in RICH_FIELDS.items():
                    if name in stored:
                        self.arrays[name] = np.asarray(stored[name], dtype=dtype).reshape(-1, *shape)
        else:
            self.meta = json.loads((self.directory / 'ready.json').read_text())
            for name, (dtype, shape) in RICH_FIELDS.items():
                path = self.directory / (name + '.bin')
                if path.is_file():
                    self.arrays[name] = (np.memmap(path, dtype=dtype, mode='r') if path.stat().st_size else np.empty(0,dtype)).reshape(-1, *shape)

    def sample(self, i):
        a = self.arrays
        lo, hi = a['offsets'][i:i+2]
        flo, fhi = a['feature_offsets'][i:i+2]
        features = np.zeros((hi-lo, 32), np.float32)
        rows = np.repeat(np.arange(hi-lo), a['counts'][lo:hi])
        features[rows, a['columns'][flo:fhi]] = a['values'][flo:fhi]
        return dict(ids=a['ids'][lo:hi], features=features, target_indices=(a['target_indices'][lo:hi] if 'target_indices' in a else np.full(hi-lo, -1, np.int32)),
                    **{name: a[name][i] for name in ('packet_index', 'cell_index', 'terrain')})

    def batch(self, indices, device='cpu'):
        from pvz_student_tokens import collate_tokens
        return collate_tokens([self.sample(int(i)) for i in indices], device)

    @property
    def nbytes(self):
        return sum(a.nbytes for a in self.arrays.values())


def extra_validation(mask_path, rows):
    if mask_path is None:
        return None
    mask = np.load(mask_path, mmap_mode='r', allow_pickle=False)
    if mask.dtype != np.bool_ or mask.shape != (rows,):
        raise ValueError('heldout deck mask shape/dtype differs from frozen pack')
    return mask

# Version 2 adds public rich target links. Version 1 compact files remain readable.
RICH_FIELDS = {**FIELDS, 'target_indices': (np.int32, ())}
SIDECAR_NAME = 'tokens_v3.npz'


def frame_crc(path):
    """Constant-size ZIP-directory lookup; never decompress/replay an archive to bind it."""
    import zipfile
    import zlib
    path = Path(path)
    if path.suffix == '.npz':
        with zipfile.ZipFile(path) as archive:
            member = archive.getinfo('frames.npy')
            return [member.CRC, member.file_size]
    raw = path.read_bytes()
    return [zlib.crc32(raw), len(raw)]


class SparseTokenWriter:
    """Streaming task162-compatible columns, compressed per episode or mmap-ready per pack."""
    def __init__(self, parent, *, index_kind='frame', token_input='obs_v3'):
        import tempfile
        self.parent = Path(parent)
        self.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='.tokens-building-', dir=self.parent)
        self.directory = Path(self.temporary.name)
        self.handles = {name: (self.directory/(name+'.bin')).open('wb') for name in RICH_FIELDS}
        self.rows = self.tokens = self.features = 0
        self.max_tokens = 0
        self.index_kind, self.token_input = index_kind, token_input
        for name in ('offsets', 'feature_offsets'):
            np.zeros(1, np.int64).tofile(self.handles[name])

    def append(self, sample):
        features = np.asarray(sample['features'], np.float32)
        # Preserve negative zero as well as every other float32 bit pattern.
        rr, cc = np.nonzero(features.view(np.uint32))
        self.rows += 1
        self.tokens += len(features)
        self.features += len(rr)
        self.max_tokens = max(self.max_tokens, len(features))
        arrays = dict(ids=sample['ids'], counts=np.bincount(rr, minlength=len(features)),
                      columns=cc, values=features[rr, cc], packet_index=sample['packet_index'],
                      cell_index=sample['cell_index'], terrain=sample['terrain'],
                      target_indices=sample['target_indices'], offsets=[self.tokens], feature_offsets=[self.features])
        for name, value in arrays.items():
            np.asarray(value, dtype=RICH_FIELDS[name][0]).tofile(self.handles[name])

    def finish(self, target, *, source_frame=None, compressed=True):
        import os
        import zipfile
        for handle in self.handles.values():
            handle.close()
        meta = dict(version=2, token_input=self.token_input, index_kind=self.index_kind,
                    rows=self.rows, tokens=self.tokens, features=self.features, max_tokens=self.max_tokens,
                    sparse_float32=True)
        if source_frame is not None:
            meta['frame_crc'] = frame_crc(source_frame)
        (self.directory/'ready.json').write_text(json.dumps(meta)+'\n')
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if compressed:
            building = self.directory/'tokens.npz'
            with zipfile.ZipFile(building, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
                def write(name, array):
                    with archive.open(name+'.npy', 'w', force_zip64=True) as stream:
                        np.lib.format.write_array(stream, array, allow_pickle=False)
                write('meta', np.asarray(json.dumps(meta)))
                for name, (dtype, _) in RICH_FIELDS.items():
                    path = self.directory/(name+'.bin')
                    value = np.memmap(path, dtype=dtype, mode='r') if path.stat().st_size else np.empty(0, dtype)
                    write(name, value)
            os.replace(building, target)
        else:
            if target.exists():
                raise FileExistsError(target)
            os.replace(self.directory, target)
        self.temporary.cleanup()
        return meta

    def abort(self):
        for handle in self.handles.values():
            if not handle.closed:
                handle.close()
        self.temporary.cleanup()


def episode_tokens(frame_path):
    """Read an optional attachment; bind using the compact member's recorded CRC only."""
    path = Path(frame_path).parent/SIDECAR_NAME
    if not path.is_file():
        return None
    store = SparseTokens(path)
    if store.meta.get('index_kind') != 'frame' or store.meta.get('frame_crc') != frame_crc(frame_path):
        raise ValueError(f'{path}: token attachment belongs to different compact frames')
    return store


def decision_tokens(directory, rows, token_input):
    """Bind a prebuilt decision attachment without opening any source game."""
    store = SparseTokens(directory)
    if (store.meta['rows'] != rows or store.meta.get('index_kind', 'decision') != 'decision'
            or store.meta.get('token_input', 'compact') != token_input):
        raise ValueError('decision token rows/input differ from the frozen pack recipe')
    return store


def decision_contract(directory):
    """Small resume identity: the producer publishes immutable columns before ready.json."""
    import hashlib
    directory = Path(directory)
    return dict(path=str(directory.resolve()), ready_sha256=hashlib.sha256((directory/'ready.json').read_bytes()).hexdigest())


def capture_frame(observation, writer=None):
    """Collection-only boundary: inference can still read inline v3; compact storage cannot."""
    from render_episode import _frame
    from pvz_student_tokens import frame_tokens
    frame = _frame(observation)
    if writer is not None:
        writer.append(frame_tokens(frame, token_input='obs_v3'))
    frame.pop('v3', None)
    return frame
