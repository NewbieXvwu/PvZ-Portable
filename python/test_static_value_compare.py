import unittest
from static_value_compare import key,pool_actions


class CommonPoolTest(unittest.TestCase):
    def test_pool_contains_both_teacher_sets_and_is_legal_and_unique(self):
        actions=[dict(type='plant',packet=p,row=0,col=c) for p in range(2) for c in range(6)]
        wait={'type':'wait','ticks':60}
        scores={'mac':[(1-i/100,a) for i,a in enumerate(actions)],
                'desktop':[(i/100,a) for i,a in enumerate(actions)]}
        arms={'mac':{'candidates':[actions[0],actions[6],wait]},
              'desktop':{'candidates':[actions[5],actions[-1],wait]}}
        result=pool_actions(scores,arms,actions,20261020)
        self.assertEqual(result,pool_actions(scores,arms,actions,20261020))
        keys={key(a) for a in result}
        self.assertEqual(len(keys),len(result))
        self.assertTrue(all(key(a) in {key(x) for x in actions+[wait]} for a in result))
        self.assertTrue(all(key(a) in keys for v in arms.values() for a in v['candidates']))
        # Two model choices each, then one random as yet uncovered cell per card.
        self.assertEqual(len(result),11)

    def test_wait_identity_ignores_timeout_spelling_only_for_fixed_teacher_waits(self):
        self.assertEqual(key({'type':'wait','ticks':60}),key({'type':'wait','ticks':60,'until':'timeout'}))


if __name__=='__main__':unittest.main()
