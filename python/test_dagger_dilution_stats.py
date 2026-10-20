import unittest
import numpy as np
from dagger_dilution_stats import holm,summarize
from teacher_audit_stats import t_critical


class DilutionInferenceTest(unittest.TestCase):
    def test_holm_is_order_correct_and_monotone(self):
        adjusted=holm({'c':.2,'a':.01,'b':.02})
        self.assertAlmostEqual(adjusted['a'],.03)
        self.assertAlmostEqual(adjusted['b'],.04)
        self.assertAlmostEqual(adjusted['c'],.2)

    def test_cluster_unit_and_two_training_seed_gate_prevent_spurious_support(self):
        n=64
        def arm(counts):
            a=np.zeros((2,6,n),float)
            for s,c in enumerate(counts):a[s,:,:c]=1
            return a
        wins={'F':arm([32,32]),'B':arm([48,48]),'M':arm([58,26]),
              'T':arm([51,51]),'I':arm([61,61]),'R':arm([32,32]),'S':arm([32,32])}
        report=summarize(wins)
        c=report['contrasts']['B-F'];vec=(wins['B']-wins['F']).mean(axis=(0,1))
        se=float(vec.std(ddof=1)/np.sqrt(n));expected=float(vec.mean())
        self.assertEqual(c['df'],n-1)
        self.assertAlmostEqual(c['t_ci95'][0],expected-t_critical(n-1)*se)
        self.assertTrue(c['supports_positive_effect'])
        # M has a positive aggregate but opposite effects across initializations.
        mixed=report['contrasts']['M-F']
        self.assertGreater(mixed['difference'],0)
        self.assertFalse(mixed['both_training_seeds_positive'])
        self.assertFalse(mixed['supports_positive_effect'])
        self.assertEqual(report['contrasts']['S-F']['env_seed_cluster_ci95'],[0,0])
        self.assertEqual(report['contrasts']['S-F']['holm_p'],1)


if __name__=='__main__':unittest.main()
