"""task70 B7: a failing background job must not stop collection.

Injects a stub background job into TrainingPipeline and drives poll() the way the collection
loop does (once per group boundary, run_round). Pinned: the first failure is only recorded
(job failed, error persisted in state, logged) while collection keeps going; exactly one retry
follows; a second consecutive failure of the same round isolates the snapshot while
collection keeps going; close() never retries and never hangs on a failed job.
"""
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from curriculum_queue import MAX_BACKGROUND_ATTEMPTS, TrainingPipeline  # noqa: E402


def make_pipeline(tmp: str) -> TrainingPipeline:
    """The fields poll()/close() touch; _run is stubbed per test, so no recipe machinery."""
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    pipeline.args = type('Args', (), {'state': Path(tmp) / 'curriculum_state.json',
                                      'log': Path(tmp) / 'curriculum.log'})()
    pipeline.state = {'training_jobs': []}
    pipeline.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='test-background')
    pipeline.futures = []
    pipeline.failed = False
    return pipeline


class BackgroundFailureRetry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def tearDown(self):
        # tests that do not call close() themselves still must not leak the executor
        self.pipeline.executor.shutdown(wait=True)

    def submit(self, run):
        self.pipeline = make_pipeline(self.tmp.name)
        job = {'round': 13, 'status': 'queued',
               'recipe': str(Path(self.tmp.name) / 'recipe.json')}
        self.pipeline._run = run
        self.pipeline.futures = [(job, self.pipeline.executor.submit(run))]
        return job

    def drive(self, until, timeout=5.0):
        """poll() per group boundary; must not raise while retries remain."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.pipeline.poll()
            if until():
                return
            time.sleep(0.02)
        self.fail('condition not reached before timeout')

    def test_first_failure_is_recorded_and_retried_exactly_once(self):
        calls = []

        def run(job=None):
            calls.append(1)
            raise RuntimeError('pack exploded on seed_20261031_f44000000')

        job = self.submit(run)
        # first failure: recorded, collection continues across group boundaries
        self.drive(lambda: job.get('attempts') == 1)
        self.assertEqual(job['status'], 'queued')          # re-queued for the retry
        self.assertIn('pack exploded', job['error'])
        # exactly one retry, at the next group boundary (runs in the background while
        # collection keeps going; no poll here -- the next poll would see the stop)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not (len(calls) >= MAX_BACKGROUND_ATTEMPTS
                                                   and self.pipeline.futures[0][1].done()):
            time.sleep(0.02)
        self.assertEqual(len(calls), MAX_BACKGROUND_ATTEMPTS)
        # task98: the second consecutive failure of the same round isolates the
        # snapshot; collection continues and qc is not stopped.
        self.drive(lambda: job.get('status') == 'failed' and job.get('isolated'), timeout=5.0)
        self.assertEqual(job['attempts'], 2)
        log = (Path(self.tmp.name) / 'curriculum.log').read_text()
        self.assertIn('retrying once', log)
        self.assertIn('snapshot isolated, collection continues', log)
        self.assertNotIn('stopping qc', log)

    def test_retry_success_completes_the_round(self):
        calls = []
        report = {'round': 13, 'promotion': {'version': 0, 'approved': False, 'reasons': ['stub']}}

        def run(job=None):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError('transient failure')
            return report

        job = self.submit(run)
        self.pipeline.evaluation_contract = lambda candidate: {'stub': True}
        self.drive(lambda: job['status'] == 'done')
        self.assertEqual(len(calls), 2)
        self.assertFalse(job['promotion_approved'])
        self.assertEqual(job['attempts'], 1)

    def test_close_never_retries_and_does_not_hang(self):
        calls = []

        def run(job=None):
            calls.append(1)
            raise RuntimeError('always fails')

        job = self.submit(run)
        self.pipeline.close()          # must terminate without raising or retrying
        self.assertEqual(len(calls), 1)
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(job['attempts'], 1)


if __name__ == "__main__":
    unittest.main()
