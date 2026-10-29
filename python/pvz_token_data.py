"""富输入训练行，复用现行标签、等待抽样与权重语义。"""
import json
import random
import numpy as np
import torch

from pvz_frame_store import (archive_seed, candidate_cells, frame_bytes, frame_paths,
                             row_damage_labels, target_values)
from pvz_token_student import token_array, token_batch
from pvz_training_pack import training_rows


class TokenStore:
    def __init__(self, arrays):
        self.arrays = arrays
        self.nbytes = sum(a.nbytes for a in arrays)

    def batch(self, indices, device):
        return token_batch([self.arrays[int(i)] for i in indices], device)


def load_token_rows(data, labels, wait_keep=1., label_weight=None, with_context=False,
                    obs="v3", cells=False, aux_horizon=0, row_observer=None,
                    packet_head="identity", source_weight=None, archive_paths=None):
    from train_rollout_policy import CandidateStore, CompactBoards, PacketStore, ValueStore
    from pvz_rollout_policy import packet_features
    paths = sorted(archive_paths if archive_paths is not None else
                   (p for p in frame_paths(data) if p.parent.name in labels))
    streams, arrays, targets, packet_arrays, seed_of, damage = {}, [], [], [], [], []
    values, candidates = ValueStore(), CandidateStore() if cells else None
    for path in paths:
        rng = streams.setdefault(path.parent.name, random.Random(0))
        lines = frame_bytes(path).splitlines()
        frames = [json.loads(line) for line in lines]
        # Keep exactly the existing label-local RNG stream, including the unlabelled first frame.
        keep = []
        for line in lines:
            tail = line[max(line.rfind(b'"label":'), line.rfind(b'"action":')):]
            keep.append(b'"type":"plant"' in tail or b'"targets":' in line or rng.random() < wait_keep)
        explicit = training_rows(path)
        if explicit is not None:
            wanted = set(explicit)
            keep = [i in wanted for i in range(len(frames))]
        meta = json.loads((path.parent / "meta.json").read_text())
        scale = (label_weight or {}).get(path.parent.name, 1.) * (source_weight or {}).get(meta.get("label_source", "teacher"), 1.)
        future = row_damage_labels(frames, aux_horizon) if aux_horizon else None
        rows = []
        for i in range(1, len(frames)):
            record = frames[i]
            action = record.get("label") or record.get("action")
            if not keep[i] or not action or record.get("action_rejected"):
                continue
            planted = action["type"] == "plant"
            arrays.append(token_array(frames[i - 1]))
            packet_arrays.append(packet_features(frames[i - 1]).numpy())
            retention = wait_keep if explicit is None else 1.
            weight = scale * (1. if planted or record.get("targets") else 1. / retention)
            targets.append((int(planted), action["packet"] if planted else -1,
                            action["row"] * 9 + action["col"] if planted else -1, weight))
            values.append(target_values(record.get("targets")))
            if cells:
                candidates.append(candidate_cells(record.get("targets")))
            if future is not None:
                damage.append(future[i - 1])
            seed_of.append(archive_seed(path.parent.parent.name))
            rows.append(i)
        if row_observer is not None:
            row_observer(path, np.asarray(rows, np.int64))
    if not targets:
        raise ValueError("no obs-v3 labelled rows selected")
    dense = np.asarray(targets, np.float32)
    scalars = CompactBoards((1,), np.float32)
    scalars.append(np.zeros((len(targets), 1), np.float32))
    packets = PacketStore(len(targets), 10)
    packets.set(0, len(targets), np.stack(packet_arrays))
    result = (TokenStore(arrays), scalars.finish(), torch.tensor(dense[:, 0], dtype=torch.uint8),
              torch.tensor(dense[:, 1], dtype=torch.int16), torch.tensor(dense[:, 2], dtype=torch.int16),
              torch.tensor(dense[:, 3]), seed_of, values.finish(),
              candidates.finish() if candidates is not None else None)
    if aux_horizon:
        result += (torch.from_numpy(np.stack(damage)),)
    return (*result, packets.finish(len(targets)))
