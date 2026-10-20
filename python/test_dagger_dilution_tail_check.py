"""Check tiny Student tails against independent elementary closed forms."""
from pathlib import Path
import sys
import unittest

import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dagger_dilution_tail_check import holm, student_two_sided


class TailPrecisionTest(unittest.TestCase):
    def test_cauchy_extreme_tail(self):
        with mp.workdps(100):
            t = mp.mpf('1e30')
            expected = 2 * mp.atan(1 / t) / mp.pi
            self.assertLess(abs(student_two_sided(t, 1) / expected - 1), mp.mpf('1e-65'))

    def test_df2_closed_form(self):
        with mp.workdps(100):
            for t in (mp.mpf('.1'), mp.mpf(4), mp.mpf('1e20')):
                expected = 2 / (mp.sqrt(2 + t*t) * (mp.sqrt(2 + t*t) + t))
                self.assertLess(abs(student_two_sided(t, 2) / expected - 1), mp.mpf('1e-65'))

    def test_holm_manual_family(self):
        with mp.workdps(100):
            result = holm({k: mp.mpf(v) for k, v in dict(a='.01', b='.04', c='.03').items()})
            self.assertEqual(set(result), {'a', 'b', 'c'})
            for name, expected in dict(a='.03', b='.06', c='.06').items():
                self.assertLess(abs(result[name] - mp.mpf(expected)), mp.mpf('1e-90'))


if __name__ == '__main__':
    unittest.main()
