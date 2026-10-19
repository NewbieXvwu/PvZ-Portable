import unittest
from teacher_counterfactual_audit import alternatives, category, physical


class AuditTest(unittest.TestCase):
    def test_frame_excludes_only_archive_annotations(self):
        frame = {'tick':3,'v2':{'wave_timer':2},'action':{},'label':{},'targets':[],
                 'verification':{},'action_rejected':False}
        self.assertEqual(physical(frame), {'tick':3,'v2':{'wave_timer':2}})

    def test_gap_uses_frontmost_on_board_zombie(self):
        state={'packets':[[0,4]],'wave':4,'zombies':[[2,0,400,0,1,True],[2,0,500,0,1,True]]}
        found=category(state, {'type':'plant','packet':0,'row':2,'col':5})
        self.assertEqual(found[0], 'A')
        self.assertLessEqual(found[1],0)
        state['zombies'][0][5]=False
        state['zombies'][1][5]=False
        self.assertIsNone(category(state, {'type':'plant','packet':0,'row':2,'col':5}))

    def test_sunflower_wave_and_column_boundaries(self):
        s={'packets':[[0,1]],'wave':3}
        a={'type':'plant','packet':0,'row':2,'col':4}
        self.assertEqual(category(s,a), ('B',None))
        s['wave']=4
        self.assertIsNone(category(s,a))

    def test_alternatives_cover_legal_cells_and_retain_origin(self):
        legal=[{'packet':0,'row':r,'col':c} for r in range(2) for c in range(6)]
        obs={'legal_actions':{'plants':legal}}
        label={'type':'plant','packet':0,'row':1,'col':5}
        choices,_=alternatives(obs,{'category':'B','label':label},{})
        self.assertEqual(choices[0],label)
        self.assertEqual(len(choices),10)
        self.assertTrue(all(a['col']<=3 for a in choices[1:-1]))
        choices,runner=alternatives(obs,{'category':'A','label':label},
                                    {'targets':[[0,1,5,.8],[0,0,2,.7]]})
        self.assertIn(runner,choices)
        self.assertEqual(len(choices),8)

    def test_illegal_archived_label_stops_audit(self):
        with self.assertRaises(RuntimeError):
            alternatives({'legal_actions':{'plants':[]}},
                         {'category':'B','label':{'type':'plant','packet':0,'row':0,'col':4}}, {})


if __name__=='__main__':
    unittest.main()
