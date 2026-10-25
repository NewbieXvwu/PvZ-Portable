"""Evidence checks for task77: a shortened or relabelled run must not pass identity."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from task77_search_hotpath import compare_frames, paired_interval


class SearchHotpathEvidenceTest(unittest.TestCase):
    def test_length_difference_at_end_is_reported(self):
        self.assertEqual(compare_frames([{'tick': 1}], [])['reason'], 'length')
        self.assertEqual(compare_frames([], [{'tick': 1}])['index'], 0)

    def test_checks_labels_targets_and_observations_even_when_action_matches(self):
        frame = {'tick': 1, 'action': {'type': 'wait'}, 'label': {'type': 'plant'},
                 'targets': [[0, 0, 0, 0.1234]], 'sun': 25}
        for key, replacement in [('label', {'type': 'wait'}),
                                 ('targets', [[0, 0, 0, 0.1235]]), ('sun', 50)]:
            mismatch = compare_frames([frame], [dict(frame, **{key: replacement})])
            self.assertIn(key, mismatch['fields'])

    def test_retains_float_bit_relevant_sign_and_verification_values(self):
        self.assertIsNotNone(compare_frames([{'value': 0.0}], [{'value': -0.0}]))
        self.assertIsNotNone(compare_frames([{'verification': {'gap': 0.1}}],
                                             [{'verification': {'gap': 0.2}}]))

    def test_only_verification_wall_time_and_dict_order_are_excluded(self):
        left = {'tick': 1, 'verification': {'gap': 0.1, 'seconds': 2}}
        right = {'verification': {'seconds': 3, 'gap': 0.1}, 'tick': 1}
        self.assertIsNone(compare_frames([left], [right]))

    def test_interval_rejects_invalid_or_empty_evidence(self):
        for ratios in ([], [0], [-1], [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError):
                paired_interval(ratios, draws=100)
        result = paired_interval([.8] * 8, draws=100)
        self.assertAlmostEqual(result['geomean'], .8)
        self.assertAlmostEqual(result['bootstrap95'][0], .8)


if __name__ == '__main__':
    unittest.main()
