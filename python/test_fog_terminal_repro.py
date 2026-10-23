"""Reproducibility compares physics and decisions, not incidental gzip timestamps."""
import gzip
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
from fog_terminal_repro_check import semantic_compare


def test_semantic_repeat_rejects_changed_outcome_or_teacher_decision(tmp_path):
    fields=('identity','outcome','v4_short_value','short_leaf','teacher_leaf','terminal_state_sha256','checks')
    original={k:{'value':1} for k in fields};repeated={k:dict(v) for k,v in original.items()}
    a,b=tmp_path/'a.gz',tmp_path/'b.gz'
    for path,mtime in ((a,1),(b,2)):
        with gzip.GzipFile(filename=str(path),mode='wb',mtime=mtime) as f:f.write(b'{"decision":0}\n')
    assert a.read_bytes()!=b.read_bytes()
    assert semantic_compare(original,repeated,a,b)['teacher_trace_equal']
    repeated['outcome']={'value':2}
    with pytest.raises(ValueError,match='outcome'):semantic_compare(original,repeated,a,b)
    repeated['outcome']={'value':1}
    with gzip.open(b,'wb') as f:f.write(b'{"decision":1}\n')
    with pytest.raises(ValueError,match='teacher_trace'):semantic_compare(original,repeated,a,b)
