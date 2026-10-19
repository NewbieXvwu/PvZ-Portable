"""Paired agreement intervals with whole-game cluster resampling (no search-point pseudoreplication)."""
from collections import defaultdict
import json
import math


def wilson(successes, total, z=1.959963984540054):
    if total <= 0:
        raise ValueError("at least one observation required")
    p = successes / total
    scale = 1 + z*z / total
    centre = (p + z*z / (2*total)) / scale
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / scale
    return [max(0.0, centre-half), min(1.0, centre+half)]


def agreement_intervals(rows, *, margin=.03, draws=20000, seed=20261019, minimum_games=8):
    import numpy as np
    if not 0 <= margin <= 1 or draws < 100:
        raise ValueError("invalid margin or too few bootstrap draws")
    groups = defaultdict(lambda: [0, 0, 0])
    key = lambda a: json.dumps(a, sort_keys=True)
    for row in rows:
        g = groups[row["seed"]]
        g[0] += 1
        g[1] += key(row["approx"]) == key(row["exact"])
        g[2] += key(row["exact_other_futures"]) == key(row["exact"])
    if not groups:
        raise ValueError("no paired search points")
    counts = np.array([groups[s] for s in sorted(groups)], dtype=np.int64)
    total, a, b = map(int, counts.sum(axis=0))
    rng = np.random.default_rng(seed)
    picked = rng.integers(len(counts), size=(draws, len(counts)))
    sampled = counts[picked].sum(axis=1)
    approx = sampled[:, 1] / sampled[:, 0]
    noise = sampled[:, 2] / sampled[:, 0]
    interval = lambda values: np.quantile(values, [.025, .975]).tolist()
    difference = a/total - b/total
    difference_ci = interval(approx-noise)
    return {"games": len(groups), "searches": total,
            "approx_agreement": a/total, "noise_agreement": b/total,
            "approx_wilson_ci95_pooled": wilson(a, total), "noise_wilson_ci95_pooled": wilson(b, total),
            "approx_cluster_ci95": interval(approx), "noise_cluster_ci95": interval(noise),
            "difference": difference, "difference_cluster_ci95": difference_ci,
            "noninferiority_margin": margin, "bootstrap_draws": draws, "bootstrap_seed": seed,
            "bootstrap_unit": "whole game, ratio of resampled agreement counts to resampled search counts",
            "gate_passed": len(groups) >= minimum_games and difference_ci[0] >= -margin,
            "minimum_games": minimum_games,
            "per_game": [{"seed": s, "searches": groups[s][0], "approx_matches": groups[s][1],
                          "noise_matches": groups[s][2]} for s in sorted(groups)]}
