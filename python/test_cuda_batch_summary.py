import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cuda_batch_bench import read_batch_summary


class BatchSummaryTest(unittest.TestCase):
    def test_structured_timing_keeps_legacy_rounded_wave_units(self):
        data=dict(schema_version=1,wall_ms=1235,wall_ms_unrounded=1234.625,
                  completed_rollouts=24,requested_rollouts=24,batches=3,error_rollouts=0)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'timing.json';path.write_text(json.dumps(data))
            parsed=read_batch_summary(path,rollouts=24,batches=3)
            self.assertEqual(parsed['wall_ms']/3,1235/3)
            for field,value in [('schema_version',2),('schema_version',True),('completed_rollouts',23),
                                ('requested_rollouts',25),('batches',2),('error_rollouts',1),
                                ('wall_ms',-1),('wall_ms',1.5),('wall_ms',True)]:
                with self.subTest(field=field,value=value):
                    path.write_text(json.dumps(data|{field:value}))
                    with self.assertRaises(ValueError):read_batch_summary(path,rollouts=24,batches=3)

    def test_missing_or_malformed_summary_is_not_replaced_by_console_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'timing.json'
            with self.assertRaises(FileNotFoundError):read_batch_summary(path,rollouts=1,batches=1)
            path.write_text('wall 10 ms for 1 rollouts')
            with self.assertRaises(json.JSONDecodeError):read_batch_summary(path,rollouts=1,batches=1)


if __name__=='__main__':unittest.main()
