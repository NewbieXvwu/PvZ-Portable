import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from value_sparse_acceptance_check import game_rows,probe_rows,paired_gate,check_config


class SparseAcceptanceEvidenceTest(unittest.TestCase):
    def test_game_evidence_requires_exact_seeds_true_booleans_and_terminal_before_cap(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'games.jsonl'
            valid=[dict(seed=s,won=True,decisions=100) for s in (31032,31033)]
            p.write_text(''.join(json.dumps(r)+'\n' for r in valid))
            self.assertEqual(set(game_rows(p,[31032,31033])),{31032,31033})
            for changed in ([valid[0],valid[0]], [dict(valid[0],won='false'),valid[1]],
                            [dict(valid[0],decisions=3000),valid[1]], [valid[0]]):
                p.write_text(''.join(json.dumps(r)+'\n' for r in changed))
                with self.assertRaises(ValueError):game_rows(p,[31032,31033])

    def test_probe_evidence_rejects_duplicates_missing_counts_and_unplanned_seeds(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'leaves.1.jsonl'
            row=dict(task='pool',seed=31032,decision=2,std=.03,top1_top2=.021,mean=.4,
                     candidates=5,scored=5,overridden=True)
            p.write_text(json.dumps(row)+'\n')
            self.assertEqual(len(probe_rows([p],'pool',[31032],{31032:1})),1)
            with self.assertRaises(ValueError):probe_rows([p],'pool',[31032],{31032:2})
            with self.assertRaises(ValueError):probe_rows([p],'pool',[31033],{31033:1})
            p.write_text((json.dumps(row)+'\n')*2)
            with self.assertRaises(ValueError):probe_rows([p],'pool',[31032],{31032:2})

    def test_paired_gate_preserves_both_declared_thresholds(self):
        self.assertTrue(paired_gate(dict(delta_wins=-2,mcnemar_exact_two_sided=.5)))
        self.assertFalse(paired_gate(dict(delta_wins=-3,mcnemar_exact_two_sided=.9)))
        self.assertFalse(paired_gate(dict(delta_wins=-1,mcnemar_exact_two_sided=.01)))
        self.assertTrue(paired_gate(dict(delta_wins=5,mcnemar_exact_two_sided=.01)))

    def test_wrong_teacher_arm_or_changed_recipe_cannot_pass_config_audit(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);teacher=dict(rollout_policy='/student',static_value='/static',executable='/engine')
            p=dict(teacher=teacher)
            c=dict(**teacher,value='/new',manifest='/frozen',task_id='pool',seeds='4',seed_offset='0',
                workers='4',per_card='1',student_candidates='2',samples='3',horizon='2500',
                verify_samples='9',min_gap='300',rollout_sim_skip='61',dagger_beta='1.0',
                halving='True',arms='fast_search')
            f=out/'config.json';f.write_text(json.dumps(c));check_config(out,p,'pool',Path('/new'),Path('/frozen'),4)
            c['arms']='fast';f.write_text(json.dumps(c))
            with self.assertRaises(ValueError):check_config(out,p,'pool',Path('/new'),Path('/frozen'),4)
            c['arms']='fast_search';c['samples']='9';f.write_text(json.dumps(c))
            with self.assertRaises(ValueError):check_config(out,p,'pool',Path('/new'),Path('/frozen'),4)


if __name__=='__main__':unittest.main()
