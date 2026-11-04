import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from mac_collect_import import receive


def test_receive_retry_after_lost_reply_keeps_completed_batch(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub.utils import EntryNotFoundError

    target = tmp_path / 'mac197' / 'batch'
    target.mkdir(parents=True)
    frames = target / 'frames.npz'
    frames.write_bytes(b'completed batch')
    deletes = []

    class Api:
        def delete_file(self, **kwargs):
            deletes.append(kwargs)
            if len(deletes) > 1:
                raise EntryNotFoundError('the previous receive already deleted this package')

    def download(**kwargs):
        pytest.fail('a completed batch must not be downloaded again')

    monkeypatch.setattr(huggingface_hub, 'HfApi', Api)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    package = 'transfer/mac197/frames/batch.tar.gz'
    assert receive('owner/repo', package, tmp_path) == target
    assert receive('owner/repo', package, tmp_path) == target
    assert frames.read_bytes() == b'completed batch'
    assert len(deletes) == 2


def test_receive_retry_preserves_transient_delete_errors(tmp_path, monkeypatch):
    import huggingface_hub

    (tmp_path / 'mac197' / 'batch').mkdir(parents=True)

    class Api:
        def delete_file(self, **kwargs):
            raise RuntimeError('429 retry later')

    monkeypatch.setattr(huggingface_hub, 'HfApi', Api)
    with pytest.raises(RuntimeError, match='429'):
        receive('owner/repo', 'transfer/mac197/frames/batch.tar.gz', tmp_path)
