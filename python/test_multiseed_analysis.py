"""Statistical checks for the task92 offline audit, without playing games."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_multiseed_selection import expected_gain, paired_analysis


def test_gain_enumerates_actual_misranking_and_ties():
    # Half the K=2 draws contain both models. A reversed panel loses 10 on
    # those draws against their average, hence -5 over all draws.
    gain, optimal = expected_gain([[2, 1], [1, 2], [1, 1]], [0, 20], 2)
    np.testing.assert_allclose(gain, [-5, 5, 0])
    np.testing.assert_allclose(optimal, [.5, 1, .75])


def test_paired_bootstrap_retains_covariance_and_task_strata():
    # Perfectly correlated policies: every panel resample must still tie.
    # Independent per-model resampling would spuriously report nonzero SD.
    outcomes = [np.array([[0, 0, 0], [1, 1, 1]]), np.ones((2, 3))]
    report = paired_analysis(outcomes, draws=100, seed=7)
    assert report['totals'] == [3, 3, 3]
    assert all(sd == 0 for sd in report['paired_difference_sd'].values())
    assert report['bootstrap_winner_probability'] == [1, 0, 0]
    for values in report['empirical_candidate_draws'].values():
        assert abs(values['bootstrap_mean_gain']) < 1e-10
        np.testing.assert_allclose(values['bootstrap_gain_interval95'], [0, 0], atol=1e-10)
