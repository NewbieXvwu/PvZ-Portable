"""Descriptive comparisons must retain real cohorts and actual timing evidence."""
from pathlib import Path
import copy
import importlib.util
import unittest

spec = importlib.util.spec_from_file_location(
    'research_comparison_summary', Path(__file__).resolve().parents[1]/'scripts/research_comparison_summary.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class ComparisonSummaryTests(unittest.TestCase):
    def config(self):
        return dict(experiment_id='reference', initialization_seed=0, purpose='reference',
                    model={'width':256}, reward={'gamma':0.99}, ppo={'learning_rate':0.0001},
                    evaluation={'modes':['greedy']}, sampling={'manifest':'same'})

    def point(self, decisions, seconds):
        return dict(counters={'decisions':decisions}, active_wall_seconds=seconds)

    def candidate(self, name, points):
        return dict(experiment_id=name, initialization_seed=0, evaluations=points)

    def task(self, task_id, cap, seeds):
        return dict(task_id=task_id, wave_cap=cap, seeds=seeds, terrain='roof',
                    evaluation_role='heldout', zombie_count_multiplier=1)

    def outcome(self, seed, won=False):
        return dict(seed=seed, won=won, result=1 if won else 2,
                    truncated=False, terminal_wave=3)

    def test_model_axis_accepts_dimension_change_but_keeps_reward_and_ppo(self):
        base=self.config(); variant=copy.deepcopy(base)
        variant.update(experiment_id='small',purpose='small encoder',initialization_seed=1)
        variant['model']['width']=128
        self.assertEqual(summary.common_config(base,'model'),summary.common_config(variant,'model'))
        variant['reward']['gamma']=1
        self.assertNotEqual(summary.common_config(base,'model'),summary.common_config(variant,'model'))
        variant['reward']=base['reward'];variant['ppo']['learning_rate']=0.0003
        self.assertNotEqual(summary.common_config(base,'model'),summary.common_config(variant,'model'))

    def test_reward_axis_keeps_model_and_original_purpose_contract(self):
        base=self.config();variant=copy.deepcopy(base);variant['reward']['gamma']=1
        self.assertEqual(summary.common_config(base,'reward'),summary.common_config(variant,'reward'))
        variant['model']['width']=128
        self.assertNotEqual(summary.common_config(base,'reward'),summary.common_config(variant,'reward'))

    def test_transferred_initialization_cannot_count_as_fresh_model_trial(self):
        config=self.config();config['initialization']={'checkpoint':'trained.pt'}
        with self.assertRaises(ValueError):summary.common_config(config,'model')

    def test_missing_interaction_node_is_missing_even_when_bracketed(self):
        candidates=[self.candidate('fast',[self.point(2000,10),self.point(25000,30)]),
                    self.candidate('slow',[self.point(2000,20)])]
        views=summary.budget_views(candidates,[2000,10000,25000],[])
        self.assertTrue(views['interaction_points'][0]['all_candidates_observed'])
        self.assertEqual(views['interaction_points'][1]['observed'],[])
        self.assertEqual(views['interaction_points'][2]['missing'],['slow'])

    def test_shared_wall_cap_never_uses_a_future_score_or_interpolates(self):
        candidates=[self.candidate('fast',[self.point(0,2),self.point(25000,30)]),
                    self.candidate('slow',[self.point(0,5),self.point(2000,20),self.point(25000,40)])]
        view=summary.budget_views(candidates,[],[30])['wall_budget_views'][0]
        fast,slow=view['observed']
        self.assertEqual((fast['actual_decisions'],fast['unused_budget_seconds']),(25000,0))
        self.assertEqual((slow['actual_decisions'],slow['unused_budget_seconds']),(2000,10))
        self.assertEqual(slow['evaluation_index'],1)

    def test_cold_start_beyond_cap_and_unstarted_candidates_remain_missing(self):
        candidates=[self.candidate('cold',[self.point(0,500)]),self.candidate('new',[])]
        view=summary.budget_views(candidates,[],[100])['wall_budget_views'][0]
        self.assertFalse(view['all_candidates_observed'])
        self.assertEqual(view['missing'],['cold','new'])
        self.assertEqual(view['observed'],[])

    def test_duplicate_nodes_and_nonpositive_wall_caps_are_rejected(self):
        candidate=self.candidate('duplicate',[self.point(2000,10),self.point(2000,12)])
        with self.assertRaises(ValueError):summary.budget_views([candidate],[2000],[])
        with self.assertRaises(ValueError):summary.budget_views([],[],[0])

    def test_repeat_or_deleted_failed_seed_cannot_raise_the_rate(self):
        task=self.task('roof3',3,[1,2])
        for rows in ([self.outcome(1,True)], [self.outcome(1,True),self.outcome(1)]):
            with self.assertRaises(ValueError):
                summary.summarize_evaluation({'seed_results':{'greedy':{'roof3':rows}}},[task],['greedy'])

    def test_full_aid_and_original_results_remain_distinct_per_task(self):
        tasks=[self.task('ordinary_full',None,[1]),self.task('aided_full',None,[1])]
        payload={'seed_results':{'greedy':{'ordinary_full':[self.outcome(1)],'aided_full':[self.outcome(1,True)]}}}
        result=summary.summarize_evaluation(payload,tasks,['greedy'])['greedy']
        self.assertEqual(result['per_task']['ordinary_full']['won'],0)
        self.assertEqual(result['per_task']['aided_full']['won'],1)
        self.assertEqual(result['cohorts']['validation/capNone']['count'],2)

    def test_budget_truncation_cannot_be_reported_as_a_win(self):
        row=self.outcome(1,True);row['truncated']=True
        with self.assertRaises(ValueError):summary.outcomes([row])

    def test_retained_unlabelled_task_inherits_explicit_manifest_split(self):
        task=self.task('retained_roof',3,[1]);del task['evaluation_role']
        payload={'seed_results':{'greedy':{'retained_roof':[self.outcome(1,True)]}}}
        with self.assertRaises(ValueError):summary.summarize_evaluation(payload,[task],['greedy'])
        result=summary.summarize_evaluation(payload,[task],['greedy'],'heldout')['greedy']
        self.assertEqual(result['cohorts']['validation/cap3']['won'],1)

    def test_another_model_or_trial_checkpoint_cannot_supply_equal_time_cost(self):
        config=self.config();state={'experiment_identity':'own'}
        provenance={'model_config':config['model']};payload={'model_config':config['model']}
        checkpoint=dict(experiment_config=config,experiment_identity='own',config=config['model'],
                        training_state=dict(experiment_identity='own',experiment_id=config['experiment_id']))
        summary.validate_model_checkpoint(checkpoint,config,provenance,state,payload)
        wrong=copy.deepcopy(checkpoint);wrong['experiment_identity']='other'
        with self.assertRaises(ValueError):summary.validate_model_checkpoint(wrong,config,provenance,state,payload)
        wrong=copy.deepcopy(checkpoint);wrong['config']={'width':128}
        with self.assertRaises(ValueError):summary.validate_model_checkpoint(wrong,config,provenance,state,payload)


if __name__=='__main__':unittest.main()
