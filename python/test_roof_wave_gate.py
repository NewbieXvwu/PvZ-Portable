"""task168: exact signed ranks and paired wave recording, no games."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_promotion import wilcoxon_worse


@pytest.mark.parametrize('differences,statistic,p', [
    ([0, 0], 0, 1), ([-1, -1], 0, .25), ([1, 1], 3, 1),
    ([-1, 0, 1], 1.5, .75), ([-1, 2, -3], 2, .375),
    # Wilcoxon's published corn example, sign reversed for the lower-tail alternative.
    ([-6, -8, -14, -16, -23, -24, -28, -29, -41, 48, -49, -56, -60, 67, -75],
     24, .0206298828125),
])
def test_exact_wilcoxon_roof_ties_zeros_and_direction(differences, statistic, p):
    result = wilcoxon_worse(differences)
    assert result['positive_rank_sum'] == statistic
    assert result['wilcoxon_worse_p'] == p
    assert result['nonzero_pairs'] == sum(d != 0 for d in differences)
