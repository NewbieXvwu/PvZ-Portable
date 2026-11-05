"""Same-root candidate ranking metric, restored from task43 (c2db2193^)."""
from collections import defaultdict
import torch


def pair_accuracy(pred, d, idx, min_gap: float, by_packet: bool = False) -> dict:
    """Mean prediction vs mean result per (seed, position, candidate), compared within each position
    (``by_packet``: within the cells of one card)."""
    groups = defaultdict(lambda: defaultdict(list))
    for k in idx.tolist():
        key = (int(d["seed"][k]), int(d["position"][k])) + ((int(d["packet"][k]),) if by_packet else ())
        groups[key][int(d["candidate"][k])].append(k)
    agree = pairs = 0
    for cands in groups.values():
        stats = [(float(d["won"][ks].mean()), float(pred[ks].mean())) for ks in map(torch.tensor, cands.values())]
        for i in range(len(stats)):
            for j in range(i + 1, len(stats)):
                if abs(stats[i][0] - stats[j][0]) >= min_gap:
                    pairs += 1
                    agree += (stats[i][0] - stats[j][0]) * (stats[i][1] - stats[j][1]) > 0
    return {"pair_accuracy": agree / max(1, pairs), "pairs": pairs, "positions": len(groups)}

