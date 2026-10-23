"""Evidence gates must reject false completion without launching games."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('qc_collection_bench', ROOT / 'scripts/bench_qc_collection.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class EvidenceGateTest(unittest.TestCase):
    def rows(self, seeds):
        return [dict(seed=s, won=False, arm='fast_search') for s in seeds]

    def test_wrong_matching_count_is_not_frozen_coverage(self):
        with self.assertRaisesRegex(ValueError, 'exactly cover'):
            bench.validate_rows(self.rows([11, 12]), [10, 11])

    def test_duplicate_rows_cannot_replace_a_missing_game(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            bench.validate_rows(self.rows([10, 10]), [10, 11])

    def test_cumulative_previous_games_are_not_this_batch(self):
        with self.assertRaisesRegex(ValueError, 'exactly cover'):
            bench.validate_rows(self.rows(list(range(16))), list(range(8, 16)))

    def test_string_false_is_not_a_valid_game_outcome(self):
        rows=self.rows([10]);rows[0]['won']='False'
        with self.assertRaisesRegex(ValueError, 'invalid'):
            bench.validate_rows(rows,[10])

    def test_valid_unordered_complete_batch(self):
        bench.validate_rows(self.rows([12, 10, 11]), [10, 11, 12])

    def test_changed_frozen_model_fails_before_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model';p.write_bytes(b'old')
            frozen=dict(input_sha256={str(p):bench.sha(p)},source_python_sha256={})
            p.write_bytes(b'new')
            with self.assertRaisesRegex(ValueError, 'frozen input changed'):
                bench.require_inputs(frozen)


if __name__=='__main__':unittest.main()
