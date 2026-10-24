"""A teacher archive needs a matched student-continuation baseline."""
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import episode_query as query


def test_teacher_variants_keep_prefix_checkpoint_and_student_continuation():
    meta = {'task': dict(seed=17, level=42, policy='teacher', deck=[1, 32, 33],
                        checkpoint='/frozen/model.pt')}
    action = {'type': 'wait', 'ticks': 60}
    record = dict(outcome={'final_wave': 8}, frames=[{}, {'action_rejected': True}])
    with patch.object(query, 'collect', return_value=record) as collect:
        result = query._run_variant(meta, {1: action}, '/resources', prefix=[action])
    assert collect.call_args.args[3] == 'student'
    assert collect.call_args.kwargs['checkpoint'] == '/frozen/model.pt'
    assert collect.call_args.kwargs['prefix_actions'] == [action]
    assert result['override_rejected']


def test_teacher_whatif_compares_rerun_baseline_and_reports_ticks():
    wait = {'type': 'wait', 'ticks': 60}
    meta = dict(task=dict(policy='teacher', deck=[1, 32, 33]),
                outcome=dict(final_wave=30))
    frames = [dict(tick=0), dict(tick=60, action=wait)]
    baseline = dict(final_wave=8, wave_count=30, final_tick=20000, reason='僵尸进屋',
                    mowers_used=0, plants_eaten=9)
    variant = dict(baseline, final_wave=9, final_tick=19000)
    with patch.object(query, '_run_variant', side_effect=[baseline, variant]) as replay:
        result = query._whatif(Path('/archive'), meta, frames, 1, ['wait:60'],
                               '/resources', False, None, 3, None)
    assert replay.call_count == 2
    assert replay.call_args_list[0].args[1] == {1: wait}
    assert replay.call_args_list[0].kwargs['prefix'] == []
    assert '同一学生续走' in result
    assert 'tick 20000' in result and 'tick 19000' in result
    assert '+1 波' in result  # Compared with the matched 8-wave baseline, not archive wave 30.
