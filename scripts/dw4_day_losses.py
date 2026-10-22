"""Read-only, common-data hard-label head losses for task23 step 5 (no optimizer)."""
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]
import numpy as np
import torch
import torch.nn.functional as F
from pvz_frame_store import cache_episode, cache_packets, cache_identity_scene
from pvz_rollout_policy import load_student


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finish(c):
    result = dict(c)
    for name, numerator, denominator in (
        ('plant_bce', 'plant_loss', 'rows'), ('plant_positive_bce', 'positive_loss', 'plants'),
        ('plant_negative_bce', 'negative_loss', 'waits'), ('packet_ce', 'packet_loss', 'plants'),
        ('cell_ce_teacher_packet', 'cell_loss', 'plants'), ('plant_accuracy', 'plant_correct', 'rows'),
        ('packet_accuracy', 'packet_correct', 'plants'), ('cell_accuracy_teacher_packet', 'cell_correct', 'plants')):
        result[name] = c[numerator]/c[denominator] if c[denominator] else None
    return result


def evaluate(plan_path, selection_path, destination):
    torch.set_num_threads(1)
    plan = json.loads(plan_path.read_text())
    selection = json.loads(selection_path.read_text())
    models = {name: load_student(Path(path)).eval() for name, path in plan['comparison_models'].items()}
    assert all(m.packet_head == 'identity' and m.obs == 'v2' and not m.context for m in models.values())
    assert all(digest(path) == plan['comparison_sha256'][name] for name, path in plan['comparison_models'].items())
    counters = {name: defaultdict(lambda: defaultdict(float)) for name in models}
    archives = defaultdict(int)
    cache = ROOT/'artifacts/task23/cache'
    for number, entry in enumerate(selection):
        path = Path(entry['path'])
        assert digest(path) == entry['frames_sha256'], str(path)
        encoded = cache_episode(path, cache, 'v2', read_directory=Path(plan['read_cache']))
        with np.load(encoded, allow_pickle=False) as f:
            rows = f['rows']; b = f['boards'].astype(np.float32); s = f['scalars'].copy()
            y = f['plant']; pk = f['packet']; cells = f['cell']
        with np.load(cache_packets(path, cache), allow_pickle=False) as f:
            assert np.array_equal(rows, f['rows']); info = f['packets']
        with np.load(cache_identity_scene(path, cache), allow_pickle=False) as f:
            assert np.array_equal(rows, f['rows']); s[:, 4] = f['wave_scalar']
        keys = ('all', 'level_'+str(entry['level']), 'label_'+entry['label'])
        for key in keys: archives[key] += 1
        for start in range(0, len(rows), 256):
            end = start+256
            board = torch.from_numpy(b[start:end]); scalars = torch.from_numpy(s[start:end])
            target = torch.from_numpy(y[start:end]); packet = torch.from_numpy(pk[start:end])
            cell = torch.from_numpy(cells[start:end]); packets = torch.from_numpy(info[start:end])
            planted = target > .5
            with torch.inference_mode():
                for name, net in models.items():
                    out = net(board, scalars, packet.clamp(min=0), packet_info=packets)
                    plant_loss = F.binary_cross_entropy_with_logits(out['plant'], target, reduction='none')
                    values = dict(rows=len(target), plants=int(planted.sum()), waits=int((~planted).sum()),
                                  plant_loss=float(plant_loss.sum()), positive_loss=float(plant_loss[planted].sum()),
                                  negative_loss=float(plant_loss[~planted].sum()),
                                  plant_correct=int(((out['plant'] > 0) == planted).sum()))
                    if planted.any():
                        values.update(packet_loss=float(F.cross_entropy(out['packet'][planted], packet[planted], reduction='sum')),
                                      cell_loss=float(F.cross_entropy(out['cell'][planted], cell[planted], reduction='sum')),
                                      packet_correct=int((out['packet'][planted].argmax(1) == packet[planted]).sum()),
                                      cell_correct=int((out['cell'][planted].argmax(1) == cell[planted]).sum()))
                    for key in keys:
                        for field, value in values.items(): counters[name][key][field] += value
        if number % 25 == 0:
            print(json.dumps(dict(archives_done=number+1, archives_total=len(selection))), flush=True)
    result = dict(selection_sha256=digest(selection_path), archives=len(selection),
                  models={name: {key: dict(archives=archives[key], **finish(c)) for key, c in groups.items()}
                          for name, groups in counters.items()},
                  methodology='All valid cached rows, no wait subsampling or model-specific weights. '
                              'Plant: unweighted BCE; packet/cell: CE conditional on teacher planting; '
                              'cell uses the teacher packet. These are common evaluation losses, not the weighted training objective.',
                  checkpoint_sha256=plan['comparison_sha256'])
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(completed=True, archives=len(selection), output=str(destination))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.plan, args.selection, args.out)
