"""Mac import regression tests: tiny local tgz packages and a fake HF snapshot."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import import_mac_labels as mac


def package(tmp_path, label, seed=1, extra=None):
    # 5 frames, 2 usable rows: exclude first frame, missing action, rejected action.
    frames = [{'action': {'type': 'wait'}}, {'action': {'type': 'wait'}}, {},
              {'action': {'type': 'plant'}, 'action_rejected': True},
              {'label': {'type': 'wait'}, 'action': None}]
    stream = io.BytesIO()
    np.savez_compressed(stream, frames=np.frombuffer(
        ''.join(json.dumps(frame) + '\n' for frame in frames).encode(), dtype=np.uint8))
    base = f'artifacts/student/frames_v2/seed_{seed}/{label}_fast_search'
    members = [(base + '/frames.npz', stream.getvalue()),
               (base + '/meta.json', json.dumps({'task': {'seed': seed}}).encode())]
    members.extend(extra or [])
    path = tmp_path / (label + '.tgz')
    with tarfile.open(path, 'w:gz') as archive:
        for name, raw in members:
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    return path


class FakeHF:
    def __init__(self, packages):
        self.packages = packages
        self.downloads = []
        self.siblings = [SimpleNamespace(rfilename=f'transfer/{label}.tgz', size=path.stat().st_size,
                                        lfs=SimpleNamespace(sha256=mac.sha(path)))
                         for label, path in packages.items()]

    def model_info(self, repo, files_metadata):
        assert repo == 'fake/repo' and files_metadata
        return SimpleNamespace(sha='fixed-snapshot', siblings=self.siblings)

    def download(self, repo, filename, revision, local_dir, token):
        assert revision == 'fixed-snapshot'
        self.downloads.append(filename)
        return self.packages[Path(filename).stem]


def refresh(tmp_path, hf, **kwargs):
    return mac.refresh(destination=tmp_path / 'frames_v2', state=tmp_path / 'state',
                       api=hf, download=hf.download, repo='fake/repo', legacy_reports=(),
                       disk_check=lambda *args: None, **kwargs)


def test_two_packages_incremental_journal_and_missing_local_repair(tmp_path):
    hf = FakeHF({label: package(tmp_path, label) for label in ('d1dag1m', 'p21w1m')})
    first = refresh(tmp_path, hf, labels=['d1dag1m'])
    assert first['added_files_this_refresh'] == 2
    # Simulate report loss after a checkpoint: the per-package journal suffices.
    (tmp_path / 'state/mac_import.json').unlink()
    second = refresh(tmp_path, hf)
    assert hf.downloads == ['transfer/d1dag1m.tgz', 'transfer/p21w1m.tgz']
    assert second['added_files_this_refresh'] == 2
    assert [(r['episodes'], r['archives'], r['rows']) for r in second['rounds']] == [(1, 1, 2)] * 2
    assert all(r['archive_sha256'] == mac.sha(hf.packages[r['label']]) for r in second['rounds'])
    third = refresh(tmp_path, hf)
    assert len(hf.downloads) == 2 and third['added_files_this_refresh'] == 0
    assert all(r['reused_verified_package'] for r in third['rounds'])
    (tmp_path / 'frames_v2/seed_1/p21w1m_fast_search/meta.json').unlink()
    assert refresh(tmp_path, hf)['added_files_this_refresh'] == 1
    assert len(hf.downloads) == 3


def test_sha_mismatch_no_extraction(tmp_path):
    hf = FakeHF({'p21w1m': package(tmp_path, 'p21w1m')})
    hf.siblings[0].lfs.sha256 = '0' * 64
    with pytest.raises(ValueError, match='LFS SHA256 mismatch'):
        refresh(tmp_path, hf)
    assert not (tmp_path / 'frames_v2').exists()
    assert not json.loads((tmp_path / 'state/mac_package_journal.json').read_text())['rounds']


def test_conflicting_local_bytes_are_preserved(tmp_path):
    hf = FakeHF({'p21w1m': package(tmp_path, 'p21w1m')})
    refresh(tmp_path, hf)
    target = tmp_path / 'frames_v2/seed_1/p21w1m_fast_search/meta.json'
    target.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='existing content differs'):
        refresh(tmp_path, hf)
    assert target.read_text() == '{"changed":true}'


@pytest.mark.parametrize('member', ['../escape', '/absolute',
                                  'seed_2/p22w1m_fast_search/meta.json',
                                  'seed_1/p21w1m_fast_search/meta.json'])
def test_bad_layout_is_rejected_before_writing(tmp_path, member):
    archive = package(tmp_path, 'p21w1m', extra=[(member, b'{}')])
    with pytest.raises(ValueError):
        mac.import_archive(archive, tmp_path / 'frames', 'p21w1m', disk_check=lambda *a: None)
    assert not list((tmp_path / 'frames').glob('seed_*'))


def test_symlink_members_and_incomplete_pairs_rejected(tmp_path):
    archive = tmp_path / 'bad.tgz'
    with tarfile.open(archive, 'w:gz') as stream:
        member = tarfile.TarInfo('seed_1/p21w1m_fast_search/meta.json')
        member.type = tarfile.SYMTYPE
        member.linkname = '/outside'
        stream.addfile(member)
    with pytest.raises(ValueError, match='nonregular'):
        mac.import_archive(archive, tmp_path / 'frames', 'p21w1m', disk_check=lambda *a: None)
    with tarfile.open(archive, 'w:gz') as stream:
        member = tarfile.TarInfo('seed_1/p21w1m_fast_search/meta.json')
        stream.addfile(member, io.BytesIO())
    with pytest.raises(ValueError, match='incomplete'):
        mac.import_archive(archive, tmp_path / 'frames', 'p21w1m', disk_check=lambda *a: None)


def test_legacy_receipts_and_frozen_z_selection(tmp_path):
    hf = FakeHF({label: package(tmp_path, label) for label in ('d7p1m', 'z15dag6b')})
    destination = tmp_path / 'frames_v2'
    old = mac.import_archive(hf.packages['d7p1m'], destination, 'd7p1m', disk_check=lambda *a: None)
    mac.import_archive(hf.packages['z15dag6b'], destination, 'z15dag6b', disk_check=lambda *a: None)
    # A historical partial z selection must not be expanded from the remote package.
    hf.packages['z15dag6b'] = package(tmp_path, 'z15dag6b', seed=2)
    hf.siblings[1].lfs.sha256 = mac.sha(hf.packages['z15dag6b'])
    legacy = tmp_path / 'legacy.json'
    mac.write(legacy, dict(repo='fake/repo', destination=str(destination),
                           rounds=[dict(label='d7p1m', **old)],
                           excluded_archival_variants=['z15dag12bv_fast_search']))
    result = mac.refresh(destination=destination, state=tmp_path / 'state', api=hf,
                         download=hf.download, repo='fake/repo', legacy_reports=[legacy],
                         disk_check=lambda *a: None)
    assert hf.downloads == []
    assert result['rounds'][1]['status'] == 'frozen_selection'
    assert not (destination / 'seed_2').exists()
    assert result['excluded_archival_variants'] == ['z15dag12bv_fast_search']


def test_patterns_environment_paths_and_disk_guard(tmp_path, monkeypatch):
    for label in ('z17dag43b', 'z15dag6c', 'd10dag5b', 'd1dag4m', 'd7p1m', 'n19dag8m', 'p29w8m'):
        assert mac.label_from_remote(f'transfer/{label}.tgz') == label
    for label in ('z15dag12bv', 'z15v2fill', 'p21w1d', 'p21dag1', '../p21w1m'):
        assert mac.label_from_remote(f'transfer/{label}.tgz') is None
    monkeypatch.delenv('PVZ_HF_REPO', raising=False)
    with pytest.raises(ValueError, match='export PVZ_HF_REPO'):
        mac.hf_environment()
    monkeypatch.setenv('PVZ_HF_REPO', 'fake/repo')
    assert mac.hf_environment() == 'fake/repo'
    with pytest.raises(ValueError, match='absolute path'):
        mac.absolute(Path('relative'))
    monkeypatch.setattr(mac.shutil, 'disk_usage', lambda p: SimpleNamespace(free=mac.MIN_FREE - 1))
    with pytest.raises(RuntimeError, match='15 GiB reserve'):
        mac.check_disk()
    monkeypatch.setattr(mac.shutil, 'disk_usage', lambda p: SimpleNamespace(free=mac.MIN_FREE + 100))
    with pytest.raises(RuntimeError):
        mac.check_disk(101)
    assert mac.check_disk(100) == mac.MIN_FREE + 100


def test_missing_requested_label_and_import_lock(tmp_path):
    with pytest.raises(ValueError, match='missing from HF'):
        refresh(tmp_path, FakeHF({}), labels=['p21w1m'])
    with mac.import_lock(tmp_path / 'state'):
        with pytest.raises(RuntimeError, match='another Mac importer'):
            refresh(tmp_path, FakeHF({}))


def test_recovered_local_receipt_blocks_unverifiable_remote_without_download(tmp_path):
    hf = FakeHF({'p21w1m': package(tmp_path, 'p21w1m')})
    receipt = mac.import_archive(hf.packages['p21w1m'], tmp_path / 'frames_v2',
                                 'p21w1m', disk_check=lambda *a: None)
    receipt.pop('archive_sha256')
    mac.write(tmp_path / 'state/mac_package_journal.json',
              dict(repo='fake/repo', destination=str(tmp_path / 'frames_v2'),
                   rounds=[dict(label='p21w1m', recovered_local_receipt=True, **receipt)]))
    with pytest.raises(ValueError, match='automatic re-import refused'):
        refresh(tmp_path, hf)
    assert hf.downloads == []


def test_rebuilt_receipts_reuse_known_package_and_preserve_unknown_hash(tmp_path):
    import rebuild_mac_import_journal as recovery
    hf = FakeHF({label: package(tmp_path, label) for label in ('d1dag1m', 'p21w1m')})
    destination = tmp_path / 'frames_v2'
    known = mac.import_archive(hf.packages['d1dag1m'], destination, 'd1dag1m',
                               disk_check=lambda *a: None)
    mac.import_archive(hf.packages['p21w1m'], destination, 'p21w1m', disk_check=lambda *a: None)
    legacy = tmp_path / 'legacy.json'
    mac.write(legacy, dict(repo='fake/repo', destination=str(destination),
                           rounds=[dict(label='d1dag1m', **known)]))
    markers, labels = tmp_path / 'paths.txt', tmp_path / 'labels.txt'
    markers.write_text('transfer/d1dag1m.tgz\ntransfer/p21w1m.tgz\n')
    labels.write_text('')
    report = recovery.rebuild(remote_paths=markers, labels_file=labels,
                              destination=destination, state=tmp_path / 'state',
                              legacy_reports=[legacy])
    assert [row['archives'] for row in report['rounds']] == [1, 1]
    assert 'archive_sha256' not in report['rounds'][1]
    before = (tmp_path / 'state/mac_package_journal.json').read_bytes()
    with pytest.raises(ValueError, match='refuse to overwrite'):
        recovery.rebuild(remote_paths=markers, labels_file=labels,
                         destination=destination, state=tmp_path / 'state',
                         legacy_reports=[legacy])
    assert (tmp_path / 'state/mac_package_journal.json').read_bytes() == before
    assert refresh(tmp_path, hf, labels=['d1dag1m'])['added_files_this_refresh'] == 0
    with pytest.raises(ValueError, match='automatic re-import refused'):
        refresh(tmp_path, hf, labels=['p21w1m'])
    assert hf.downloads == []
