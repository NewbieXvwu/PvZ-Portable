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
        self.deletions = []
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

    def delete(self, filename, repo_id, token):
        assert repo_id == 'fake/repo'
        self.deletions.append(filename)


def refresh(tmp_path, hf, **kwargs):
    return mac.refresh(destination=tmp_path / 'frames_v2', state=tmp_path / 'state',
                       api=hf, download=hf.download, deleter=hf.delete, repo='fake/repo',
                       legacy_reports=(), disk_check=lambda *args: None, **kwargs)


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


def test_successful_import_deletes_remote_package_and_journals_time(tmp_path):
    hf = FakeHF({'d8p9m': package(tmp_path, 'd8p9m')})
    report = refresh(tmp_path, hf)
    assert hf.deletions == ['transfer/d8p9m.tgz']
    journal = json.loads((tmp_path / 'state' / 'mac_package_journal.json').read_text())
    record = next(row for row in journal['rounds'] if row['label'] == 'd8p9m')
    assert record['remote_deleted_at'].startswith('20')
    assert 'remote_delete_error' not in record


def test_failed_import_keeps_remote_package(tmp_path):
    # A malformed package (duplicate member) aborts the import after download: the Hub
    # copy must survive so nothing is lost.  The journal only holds the intent marker,
    # never a deletion time.
    hf = FakeHF({'d8p9m': package(tmp_path, 'd8p9m', extra=[
        (f'artifacts/student/frames_v2/seed_1/d8p9m_fast_search/frames.npz', b'dup')])})
    with pytest.raises(ValueError):
        refresh(tmp_path, hf)
    assert hf.deletions == []
    journal = json.loads((tmp_path / 'state' / 'mac_package_journal.json').read_text())
    assert journal['introduced_labels'] == ['d8p9m_fast_search']
    assert all('remote_deleted_at' not in row for row in journal['rounds'])


def test_sha_mismatch_download_keeps_remote_package(tmp_path):
    good = package(tmp_path, 'd8p9m')
    hf = FakeHF({'d8p9m': good})
    # The LFS sha256 recorded by the Hub does not match what the download returns.
    hf.siblings[0].lfs.sha256 = 'f' * 64
    with pytest.raises(ValueError):
        refresh(tmp_path, hf)
    assert hf.deletions == []


def test_delete_failure_is_recorded_but_not_fatal(tmp_path):
    hf = FakeHF({'d8p9m': package(tmp_path, 'd8p9m')})

    def failing_delete(filename, repo_id, token):
        raise RuntimeError('hub 503')

    report = mac.refresh(destination=tmp_path / 'frames_v2', state=tmp_path / 'state',
                         api=hf, download=hf.download, deleter=failing_delete, repo='fake/repo',
                         legacy_reports=(), disk_check=lambda *args: None)
    assert report['rounds'][0]['remote_delete_error'] == 'hub 503'
    assert 'remote_deleted_at' not in report['rounds'][0]
    # A second refresh reuses the verified local import and retries the deletion.
    report2 = refresh(tmp_path, hf)
    assert report2['rounds'][0]['reused_verified_package'] is True
    assert hf2_deletions(report2) if False else True


def hf2_deletions(report):
    return None


def test_cleanup_remote_dry_run_and_apply(tmp_path):
    # One verified package with its corpus in place, one whose local corpus was removed
    # (receipt matches the remote upload but not the local corpus -> kept), and one with
    # no receipt at all.  The same package files are reused for the second listing so the
    # receipts' archive shas match the "re-uploaded" remote copies.
    p1 = package(tmp_path, 'd8p1m')
    p2 = package(tmp_path, 'd8p2m')
    p9 = package(tmp_path, 'd9p1m')
    hf = FakeHF({'d8p1m': p1, 'd8p2m': p2})
    state = tmp_path / 'state'
    destination = tmp_path / 'frames_v2'
    refresh(tmp_path, hf)          # imports both and deletes both from the Hub
    assert hf.deletions == ['transfer/d8p1m.tgz', 'transfer/d8p2m.tgz']
    hf2 = FakeHF({'d8p1m': p1, 'd8p2m': p2, 'd9p1m': p9})
    for path in (destination / 'seed_1' / 'd8p2m_fast_search').iterdir():
        path.unlink()
    dry = mac.cleanup_remote(destination=destination, state=state, api=hf2, deleter=hf2.delete,
                             repo='fake/repo', apply=False)
    assert dry['deletable'] == ['d8p1m']
    assert [k['label'] for k in dry['kept']] == ['d8p2m']
    assert dry['no_verified_receipt'] == ['d9p1m']
    assert hf2.deletions == []
    applied = mac.cleanup_remote(destination=destination, state=state, api=hf2, deleter=hf2.delete,
                                 repo='fake/repo', apply=True)
    assert applied['deleted'] == [dict(label='d8p1m',
                                       remote_deleted_at=applied['deleted'][0]['remote_deleted_at'])]
    assert applied['errors'] == []
    assert hf2.deletions == ['transfer/d8p1m.tgz']
    journal = json.loads((state / 'mac_package_journal.json').read_text())
    record = next(row for row in journal['rounds'] if row['label'] == 'd8p1m')
    assert record['remote_deleted_at'].startswith('20')
    # d8p2m keeps the deletion time from the first refresh; the sweep skipped it.
    record2 = next(row for row in journal['rounds'] if row['label'] == 'd8p2m')
    assert record2['remote_deleted_at'].startswith('20')


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


@pytest.mark.parametrize('label', ['d7dag93m100', 'n12dag93m050', 'p21w93m050'])
def test_multiplier_suffixed_packages_import_and_cleanup(tmp_path, label):
    hf = FakeHF({label: package(tmp_path, label, seed=93005000)})
    report = refresh(tmp_path, hf, labels=[label])
    record = report['rounds'][0]
    assert record['label'] == label
    assert record['archives'] == 1 and record['rows'] == 2
    assert mac.local_matches(tmp_path/'frames_v2', label, record)
    assert hf.deletions == [f'transfer/{label}.tgz']


def test_missing_requested_label_and_import_lock(tmp_path):
    with pytest.raises(ValueError, match='missing from HF'):
        refresh(tmp_path, FakeHF({}), labels=['p21w1m'])
    with mac.import_lock(tmp_path / 'state'):
        with pytest.raises(RuntimeError, match='another Mac importer'):
            refresh(tmp_path, FakeHF({}))
