"""Add the requested Mac rounds from HF transfer/ without replacing existing frames."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import tarfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/task23'
PATTERN = re.compile(r'transfer/(d(?:9|10)dag[45]b|d(?:7|8|9|10)p[1-4]m)\.tgz')


def hf_environment():
    """Read only the two literal assignments; never execute a shell profile or log credentials."""
    path = Path('/home/newbiexvwu/.bashrc')
    if path.exists():
        for line in path.read_text().splitlines():
            match = re.match(r'\s*(?:export\s+)?(HF_TOKEN|PVZ_HF_REPO)=(.*)', line)
            if match and not os.environ.get(match[1]):
                value = shlex.split(match[2], comments=True)
                if len(value) == 1: os.environ[match[1]] = value[0]
    return os.environ['PVZ_HF_REPO']


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()


def import_archive(path, destination, label):
    """Accept only per-seed frame/meta members; exclusive writes protect the shared data root."""
    added = existing = 0; archives = set(); conflicts = []; content = hashlib.sha256()
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            if member.isdir(): continue
            if not member.isfile(): raise ValueError('nonregular archive entry: '+member.name)
            parts = Path(member.name).parts
            if '..' in parts or Path(member.name).is_absolute(): raise ValueError('unsafe archive path')
            indices = [i for i, part in enumerate(parts) if re.fullmatch(r'seed_\d+', part)]
            if len(indices) != 1: raise ValueError('missing/ambiguous seed directory: '+member.name)
            relative = Path(*parts[indices[0]:])
            if len(relative.parts) != 3 or relative.parts[1] != label+'_fast_search' or relative.name not in ('frames.npz','meta.json'):
                raise ValueError('unexpected archive layout: '+member.name)
            target = destination/relative
            if not target.resolve().is_relative_to(destination.resolve()): raise ValueError('escaping existing symlink')
            raw = archive.extractfile(member).read()
            digest = hashlib.sha256(raw).hexdigest(); content.update((str(relative)+digest).encode())
            archives.add(relative.parts[0])
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with target.open('xb') as output: output.write(raw)
                added += 1
            except FileExistsError:
                existing += 1
                if sha(target) != digest: conflicts.append(str(relative))
    if conflicts: raise ValueError('existing content differs (never replaced): '+','.join(conflicts[:5]))
    for seed in archives:
        folder = destination/seed/(label+'_fast_search')
        assert (folder/'meta.json').is_file() and (folder/'frames.npz').is_file()
    return dict(archives=len(archives), added_files=added, existing_identical_files=existing,
                archive_sha256=sha(path), member_ledger_sha256=content.hexdigest())


def main():
    from huggingface_hub import HfApi, hf_hub_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=Path('/home/newbiexvwu/PvZ-search/artifacts/student/frames_v2'))
    args = parser.parse_args()
    assert args.destination.is_absolute()
    OUT.mkdir(parents=True, exist_ok=True)
    repo = hf_environment(); api = HfApi()
    info = api.model_info(repo, files_metadata=True)
    selected = [entry for entry in info.siblings if PATTERN.fullmatch(entry.rfilename)]
    records = []
    for entry in selected:
        label = PATTERN.fullmatch(entry.rfilename)[1]
        path = Path(hf_hub_download(repo, entry.rfilename, revision=info.sha, local_dir=OUT/'hf_downloads'))
        if entry.lfs: assert sha(path) == entry.lfs.sha256
        result = dict(label=label, remote_path=entry.rfilename, **import_archive(path, args.destination, label))
        records.append(result); print(json.dumps(result), flush=True)
    required = {'d9dag4b','d10dag4b','d9dag5b','d10dag5b'}
    assert required <= {r['label'] for r in records}, 'required completed Mac rounds missing'
    result = dict(repo=repo, revision=info.sha, destination=str(args.destination),
                  rounds=records, archives=sum(r['archives'] for r in records),
                  missing_requested=[f'd{l}p{r}m' for r in range(1,5) for l in (7,8,9,10)
                                     if f'd{l}p{r}m' not in {v['label'] for v in records}])
    (OUT/'mac_import.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__': main()
