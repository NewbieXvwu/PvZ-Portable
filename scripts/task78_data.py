"""Read-only source inventory and outcome-blind root selection for task78.

Writes exclusively below this worktree's ignored artifacts/task78. Never repairs,
recaches or edits qc archives. Large identity manifests stay out of git.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path('/home/newbiexvwu/PvZ-search')
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def order(path):
    return hashlib.sha256(str(path).encode()).hexdigest()


def group_name(name):
    if 'roofl41r7m100' in name:
        return 'roof_l41'
    if 'foguserl31snowm100' in name:
        return 'fog_l31_human'
    for marker, group in [('demofog31snow', 'fog_demo_l31'),
                          ('demofog32snow', 'fog_demo_l32'),
                          ('demofog33cactussnow', 'fog_demo_l33_cactus'),
                          ('demofog33cattailsnow', 'fog_demo_l33_cattail')]:
        if marker in name:
            return group
    return None


def world_identity(archive):
    match = re.fullmatch(r'seed_(\d+)(?:_f(\d+))?', archive.parent.name)
    if not match:
        raise ValueError(f'unknown seed identity: {archive}')
    return dict(seed=int(match[1]), future_seed=int(match[2]) if match[2] else None)


def midpoints(indices, quota=2):
    if len(indices) <= quota:
        return list(indices)
    return [indices[(2*i+1)*len(indices)//(2*quota)] for i in range(quota)]


def select(paths):
    """One archive per world per cohort, hash order, before opening frame outcomes."""
    groups = {}
    for path in sorted(paths, key=order):
        group = group_name(path.name)
        if group:
            identity = json.dumps(world_identity(path), sort_keys=True)
            groups.setdefault(group, {}).setdefault(identity, path)
    result = []
    for group, worlds in sorted(groups.items()):
        quota = 24 if group == 'roof_l41' else 8
        if len(worlds) < quota:
            raise ValueError(f'{group}: need {quota} worlds, found {len(worlds)}')
        result.extend((group, path) for path in list(worlds.values())[:quota])
    if len(result) != 64:
        raise ValueError(f'expect 64 selected archives in six cohorts, got {len(result)}')
    return result


def dependency_inventory():
    import torch
    checkpoint = MAIN/'artifacts/value_net/v5/model.pt'
    model = torch.load(checkpoint, map_location='cpu', weights_only=False)
    sources = list(model['branches'])
    sources += [model['decision_data'], model['audit_manifest']]
    return dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                trained_on=sorted(model['trained_on']),
                declared_sources=[dict(path=p, exists=Path(p).exists()) for p in sources])


def inventory():
    from pvz_frame_store import frame_path, read_frames
    source = MAIN/'artifacts/student/frames_v2'
    paths = [p for seed in source.iterdir() if seed.is_dir() and seed.name.startswith('seed_')
             for p in seed.iterdir() if p.is_dir() and group_name(p.name)]
    selected = select(paths)
    rows = []
    for group, archive in selected:
        path = frame_path(archive)
        before = sha(path), sha(archive/'meta.json')
        frames = read_frames(path)
        meta = json.loads((archive/'meta.json').read_text())
        source_type = meta.get('label_source')
        legacy_teacher = source_type is None and meta.get('task', {}).get('arm') == 'fast_search'
        if source_type != 'teacher' and not legacy_teacher:
            raise ValueError(f'not a teacher source: {archive}')
        # Frame i contains targets computed at input frame i-1.
        eligible = [i-1 for i in range(1, len(frames))
                    if frames[i].get('targets') and not frames[i-1].get('terminal')]
        roots = midpoints(eligible)
        if not roots:
            raise ValueError(f'no actual search roots: {archive}')
        deck = [int(p[1]) for p in frames[0]['packets']]
        scene = frames[0]['scene']
        if scene.get('pool') and not ({16, 43} & set(deck)):
            raise ValueError(f'deck does not cover water: {archive}')
        if scene.get('roof') and 33 not in deck:
            raise ValueError(f'deck lacks roof pots: {archive}')
        if before != (sha(path), sha(archive/'meta.json')):
            raise ValueError(f'source changed while reading: {archive}')
        row = dict(group=group, archive=str(archive), frames_path=str(path),
                   label_source='legacy_fast_search' if legacy_teacher else 'teacher',
                   frames_sha256=before[0], meta_sha256=before[1],
                   world=world_identity(archive), deck=deck, level=scene['level'],
                   multiplier=meta['context']['multiplier'],
                   roots=[dict(frame=i, tick=frames[i]['tick'], wave=frames[i]['wave']) for i in roots],
                   search_roots=len(eligible), demo_archive=meta.get('demo_archive'),
                   demo_decision=meta.get('demo_decision'))
        rows.append(row)
    dep = dependency_inventory()
    trained = set(dep['trained_on'])
    # Fixed demo environment seeds may repeat; independent futures must not.
    reserved = set(range(78000000, 78002000)) | set(range(78100000, 78102000))
    source_seeds = {v for r in rows for v in r['world'].values() if v is not None}
    collision = sorted(reserved & (source_seeds | trained))
    if collision:
        raise ValueError(f'reserved evaluation seeds already used: {collision}')
    output = ROOT/'artifacts/task78'
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/'sources.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old['archives'] != rows:
            raise ValueError('frozen source inventory differs; do not overwrite')
    manifest.write_text(json.dumps(dict(archives=rows, dependencies=dep), indent=1)+'\n')
    summary = dict(status='sources_selected_not_replay_verified',
                   available=dict(Counter(group_name(p.name) for p in paths)),
                   selected=dict(Counter(r['group'] for r in rows)),
                   selected_archives=len(rows), selected_roots=sum(len(r['roots']) for r in rows),
                   source_manifest=str(manifest), source_manifest_sha256=sha(manifest),
                   baseline_sha256=dep['checkpoint_sha256'],
                   missing_v5_declared_sources=sum(not r['exists'] for r in dep['declared_sources']),
                   v5_declared_sources=len(dep['declared_sources']),
                   direct_reserved_seed_collisions=collision,
                   full_panel_seed_audit_complete=False, replay_verified=False,
                   trained=False, evaluated=False,
                   demo_sources_missing=sum(bool(r['demo_archive']) and not Path(r['demo_archive']).exists()
                                            for r in rows))
    (output/'inventory.json').write_text(json.dumps(summary, indent=1)+'\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    inventory()
