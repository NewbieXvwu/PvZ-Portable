"""Check actual-type counters against per-action archives and legacy game outcomes."""
from collections import Counter
from task27_common import ROOT, MAIN, OUT, PYTHON, read, write, sha, guard
import subprocess
import sys
sys.path.insert(0, str(ROOT/'python'))
from pvz_env import TaskSpec
from pvz_event_env import EventWaitEnv
from pvz_rollout_policy import export_student, load_student
from pvz_frame_store import read_frames
from pvz_tasks import profile_for_deck
from render_episode import load_training_task, training_task_kwargs
from pvz_episode_stats import planted_slots,planted_by_type


def native(binary, model, manifest, name, seed):
    definition = load_training_task(manifest, name)
    kw = training_task_kwargs(definition, seed)
    task = TaskSpec(level=kw['level'], seed=seed, playthrough=2,
                    profile=profile_for_deck(tuple(kw['deck'])),
                    **{k:v for k,v in kw['task_extra'].items() if v is not None})
    with EventWaitEnv('/home/newbiexvwu/.cache/pvz-research-resources', executable=binary) as env:
        env.load_rollout_net(model)
        env.reset(deck=tuple(kw['deck']), task=task)
        obs = env.rollout(None, 0, max_actions=3000)
        details = env._last_response
    types=planted_by_type(details,definition)
    if isinstance(details.get('planted_by_type'),list) and len(details['planted_by_type'])==54:
        types={str(i):c for i,c in enumerate(details['planted_by_type']) if c}
    if binary==ROOT/'build/pvz-portable':
        assert isinstance(details['planted_types'],dict) and 'planted_slots' in details
        assert 'planted' not in details and 'planted_by_type' not in details
    return dict(won=obs.get('result') == 1, final_wave=obs['wave'],
                decisions=details['decisions'], mowers_left=details['mowers_left'],
                planted_slots=planted_slots(details), planted_types=types)


def main():
    import torch
    torch.set_num_threads(1)
    out = OUT/'step0/type_count_check'; out.mkdir(parents=True, exist_ok=True)
    checkpoint = MAIN/'artifacts/student/dw4i_s1/model.pt'
    exported = out/'s1.pvzrn'; export_student(load_student(checkpoint).eval(), exported)
    manifest = OUT/'unified_v2.json'
    checks = []
    for name in ('night_l12_eval_n10', 'day_l10_orig_eval', 'pool_l21_eval_p9k'):
        old = native(OUT/'step0/previous-pvz-portable', exported, manifest, name, 30000)
        new = native(ROOT/'build/pvz-portable', exported, manifest, name, 30000)
        assert old == new
        folder = out/name
        command = [PYTHON,str(ROOT/'scripts/search_play.py'),'--rollout-policy',str(checkpoint),
                   '--value',str(MAIN/'artifacts/value_net/v3/model.pt'),
                   '--manifest',str(manifest),'--task-id',name,'--arms','fast','--workers','1','--seeds','1',
                   '--out',str(folder),'--frames',str(out/'frames'),'--frames-label',name,
                   '--executable',str(ROOT/'build/pvz-portable'),
                   '--resource-dir','/home/newbiexvwu/.cache/pvz-research-resources']
        with (out/(name+'.txt')).open('w') as stream:
            subprocess.run(command,env=guard.ENV,check=True,stdout=stream,stderr=subprocess.STDOUT)
        result = read(folder/'games.jsonl')
        assert result['won'] == new['won'] and result['final_wave'] == new['final_wave']
        assert result['decisions'] == new['decisions']
        frames = read_frames(out/'frames/seed_30000'/(name+'_fast')/'frames.npz')
        types = Counter(); slots = Counter()
        for before, after in zip(frames,frames[1:]):
            action = after.get('action',{})
            if action.get('type') == 'plant' and not after.get('action_rejected'):
                kind = dict((p[0],p[1]) for p in before['packets'])[action['packet']]
                types[str(kind)] += 1; slots[str(action['packet'])] += 1
        assert dict(types) == result['planted_types'] == new['planted_types']
        assert dict(slots) == result['planted_slots'] == new['planted_slots']
        assert 'planted' not in result and 'planted_by_type' not in result
        checks.append(dict(task=name, seed=30000, legacy_outcome_identical=True,
                           native_vs_archived_actions_equal=True, plants=sum(types.values()),
                           planted_types=dict(types)))
    write(out/'complete.json',dict(complete=True,model_sha256=sha(checkpoint),
          executable_sha256=sha(ROOT/'build/pvz-portable'), checks=checks))
    print(read(out/'complete.json'))


if __name__ == '__main__':
    main()
