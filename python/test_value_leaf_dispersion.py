from argparse import Namespace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import value_leaf_dispersion as probe


class LeafProbeResumeTest(unittest.TestCase):
    def test_resume_keeps_completed_probes_and_drops_only_unfinished_prefixes(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'games.jsonl').write_text('{"seed":1,"searched":20}\n')
            rows=[dict(task='task',seed=s,decision=3) for s in (1,2)]
            f=out/'leaves.99.jsonl';f.write_text(''.join(json.dumps(x)+'\n' for x in rows))
            probe.prepare_probe_resume(out,'task',{1,2})
            self.assertEqual([json.loads(x) for x in f.read_text().splitlines()],[rows[0]])

    def test_missing_completed_probes_and_wrong_seeds_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);f=out/'games.jsonl';f.write_text('{"seed":1,"searched":20}\n')
            with self.assertRaises(ValueError):probe.prepare_probe_resume(out,'task',{1,2})
            f.write_text('{"seed":3,"searched":0}\n')
            with self.assertRaises(ValueError):probe.prepare_probe_resume(out,'task',{1,2})

    def test_duplicate_completed_probe_is_not_silently_covered(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'games.jsonl').write_text('{"seed":1,"searched":20}\n')
            row=json.dumps(dict(task='task',seed=1,decision=3))+'\n'
            (out/'leaves.1.jsonl').write_text(row);(out/'leaves.2.jsonl').write_text(row)
            with self.assertRaises(ValueError):probe.prepare_probe_resume(out,'task',{1})

    def test_internal_frozen_manifest_is_used_without_changing_teacher_flags(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);manifest=r/'frozen.json';manifest.write_text(json.dumps(dict(tasks=[
                dict(task_id=t,seeds=[31032,31033,31034,31035]) for t in probe.TASKS])))
            args=Namespace(out=r/'out',manifest=manifest,v3=r/'base.pt',v4=r/'new.pt',
                games=4,seed_offset=0,workers=4,executable=r/'engine',limit=80)
            with patch.object(probe.subprocess,'run') as run:probe.run(args)
            self.assertEqual(run.call_count,4)
            for call in run.call_args_list:
                cmd=call.args[0]
                self.assertEqual(cmd[cmd.index('--manifest')+1],str(manifest))
                self.assertEqual(cmd[cmd.index('--verify-samples')+1],'9')
                self.assertEqual(cmd[cmd.index('--student-candidates')+1],'2')


if __name__=='__main__':unittest.main()
