import importlib.util
from pathlib import Path
import json
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('qc_log_throughput',ROOT/'scripts/qc_log_throughput.py')
log=importlib.util.module_from_spec(spec);spec.loader.exec_module(log)


class ReconstructionTest(unittest.TestCase):
    def segment(self,hour='10',label='reuse',cumulative=48,workers=2,rc=0,final=48):
        return [f'48 games on {workers} workers','INFO: BuildNum: 2074',f'{final}/48 (600s)',
                f'2026-10-23T{hour}:10:00 round 1 {label} day_l7_train_d7 mult=1 offset=0 games={cumulative} wins=1 rate=0.5 rc={rc} 600s']

    def test_reused_label_totals_are_not_new_output(self):
        d=log.parse(self.segment(cumulative=48)+self.segment(hour='11',cumulative=96),log.stamp('2026-10-23T12:00:00'))
        self.assertEqual([r['new_valid_games'] for r in d['runs']],[48,48])
        self.assertEqual(log.aggregate(d['runs'])[0]['games'],96)

    def test_workers_come_from_each_start_not_last_config(self):
        d=log.parse(self.segment(workers=2)+self.segment(hour='11',workers=8,cumulative=96),0)
        self.assertEqual([r['workers'] for r in d['runs']],[2,8])
        self.assertAlmostEqual(log.aggregate(d['runs'])[0]['worker_hours'],6000/3600)

    def test_background_headroom_log_preserves_memory_and_live_start_boundary(self):
        worker='2026-10-23T11:00:00 round 2 next: workers=2/16 (MemAvailable 10.1 GiB, per-worker 0.86 GiB, background headroom 1.64 GiB)'
        lines=self.segment()+['2026-10-23T10:10:00 round 1 done: 1 groups, 48 games, changes=0',
                              worker,'48 games on 2 workers']
        d=log.parse(lines,log.stamp('2026-10-23T12:00:00'))
        self.assertEqual(d['orphan_starts'][0]['worker_log']['background'],'1.64')
        self.assertTrue(d['rounds'][0]['cycle_complete'])
        self.assertEqual(d['rounds'][0]['cycle_wall_seconds'],3600)

    def test_failed_partial_is_not_complete(self):
        d=log.parse(self.segment(rc=1,final=30),0)
        self.assertFalse(d['runs'][0]['complete']);self.assertEqual(log.aggregate(d['runs']),[])

    def test_missing_start_or_final_progress_is_not_inferred(self):
        for lines in (self.segment()[1:],self.segment()[:2]+self.segment()[3:]):
            self.assertFalse(log.parse(lines,0)['runs'][0]['complete'])

    def test_duplicate_completion_line_does_not_double_count(self):
        s=self.segment();d=log.parse(s+[s[-1]],0)
        self.assertEqual(len(d['runs']),1)

    def test_live_pack_cycle_has_upper_bound_not_finished_rate(self):
        lines=self.segment()+['2026-10-23T10:10:00 round 1 done: 1 groups, 48 games, changes=0']
        cutoff=log.stamp('2026-10-23T11:00:00');d=log.parse(lines,cutoff);r=d['rounds'][0]
        self.assertTrue(r['complete_collection']);self.assertFalse(r['cycle_complete'])
        self.assertEqual(r['cycle_games_per_hour_or_upper_bound'],48)
        self.assertEqual(r['noncollection_seconds_lower_bound'],3000)

    def test_next_group_closes_cycle_with_inter_round_time(self):
        lines=self.segment()+['2026-10-23T10:10:00 round 1 done: 1 groups, 48 games, changes=0']+self.segment(hour='11',label='next')
        r=log.parse(lines,0)['rounds'][0]
        self.assertTrue(r['cycle_complete']);self.assertEqual(r['cycle_wall_seconds'],3600)

    def test_failed_pack_and_restart_close_calendar_without_claiming_pack_success(self):
        lines=self.segment()+['2026-10-23T10:10:00 round 1 done: 1 groups, 48 games, changes=0',
                              '2026-10-23T10:50:00 pack 202 labels 2400s rc=-15 ',
                              '2026-10-23T11:00:00 round 2 next: workers=2/16 (MemAvailable 8.0 GiB, per-worker 0.5 GiB)',
                              '48 games on 2 workers']
        r=log.parse(lines,log.stamp('2026-10-23T11:10:00'))['rounds'][0]
        self.assertTrue(r['cycle_complete']);self.assertTrue(r['pack_failed'])
        self.assertEqual(r['cycle_status'],'closed_calendar_with_pack_failure')
        self.assertEqual(r['pack_events'][0]['returncode'],-15)
        self.assertEqual(r['cycle_wall_seconds'],3600)

    def test_failed_pack_without_restart_is_still_censored_and_failed(self):
        lines=self.segment()+['2026-10-23T10:10:00 round 1 done: 1 groups, 48 games, changes=0',
                              '2026-10-23T10:50:00 pack 202 labels 2400s rc=1 error']
        r=log.parse(lines,log.stamp('2026-10-23T11:00:00'))['rounds'][0]
        self.assertFalse(r['cycle_complete']);self.assertTrue(r['pack_failed'])
        self.assertEqual(r['cycle_status'],'right_censored_with_pack_failure')

    def test_later_pack_failure_does_not_pollute_prior_cycle(self):
        lines=self.segment()+['2026-10-23T10:10:00 round 1 done: 1 groups, 48 games, changes=0']
        lines+=self.segment(hour='11',label='next')
        lines+=['2026-10-23T11:50:00 pack 202 labels 2400s rc=-15']
        r=log.parse(lines,log.stamp('2026-10-23T12:00:00'))['rounds'][0]
        self.assertFalse(r['pack_failed']);self.assertEqual(r['pack_events'],[])

    def test_durable_duplicate_rows_refute_a_complete_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'reuse_48';folder.mkdir()
            (folder/'config.json').write_text(json.dumps(dict(frames_label='reuse',task_id='day_l7_train_d7')))
            (folder/'games.jsonl').write_text('\n'.join(json.dumps(dict(seed=1,won=False)) for _ in range(48)))
            runs=log.parse(self.segment(),0)['runs']
            with self.assertRaisesRegex(ValueError,'durable game rows'):
                log.audit_raw_rows(runs,Path(tmp))

    def test_next_live_group_start_closes_prior_cycle_without_waiting_for_its_end(self):
        lines=self.segment()+['2026-10-23T10:10:00 round 1 done: 1 groups, 48 games, changes=0',
                              '2026-10-23T11:00:00 round 2 next: workers=2/16 (MemAvailable 8.0 GiB, per-worker 0.5 GiB)',
                              '48 games on 2 workers']
        r=log.parse(lines,log.stamp('2026-10-23T11:10:00'))['rounds'][0]
        self.assertTrue(r['cycle_complete']);self.assertEqual(r['cycle_wall_seconds'],3600)
        self.assertEqual(r['cycle_end_source'],'next_worker_log')

    def test_unmatched_game_start_inside_round_invalidates_whole_round_claim(self):
        lines=['48 games on 2 workers']+self.segment()+[
            '2026-10-23T10:10:00 round 1 done: 1 groups, 48 games, changes=0']
        # An unmatched attempt predating the selected first group is outside this round.
        self.assertTrue(log.parse(lines,log.stamp('2026-10-23T11:00:00'))['rounds'][0]['complete_collection'])
        lines=self.segment()+['48 games on 2 workers','48 games on 2 workers',
            '48/48 (600s)',
            '2026-10-23T10:20:00 round 1 second day_l7_train_d7 mult=1 offset=48 games=48 wins=1 rate=0.5 rc=0 600s',
            '2026-10-23T10:20:00 round 1 done: 2 groups, 96 games, changes=0']
        r=log.parse(lines,log.stamp('2026-10-23T11:00:00'))['rounds'][0]
        self.assertFalse(r['complete_collection'])


if __name__=='__main__':unittest.main()
