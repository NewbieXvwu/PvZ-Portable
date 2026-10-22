"""Trained-width native parity on day/night/conveyor; task19's measured-noise tolerance."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]
import torch
from pvz_env import TaskSpec
from pvz_event_env import EventWaitEnv
from pvz_rollout_policy import act, load_student, export_student, identity_scalars, packet_features
from pvz_student_observation import StudentHistory
from pvz_tasks import profile_for_deck
from render_episode import _frame, load_training_task, training_task_kwargs


def check(model_path, out):
    torch.set_num_threads(1)
    net = load_student(model_path).eval(); reference = copy.deepcopy(net).double().eval()
    exported = out.with_suffix('.pvzrn'); export_student(net,exported)
    manifest = ROOT/'artifacts/task23/unified_manifest.json'
    heads = ('plant','packet','cell'); samples = {name:[] for name in heads}
    actions = argmax = 0; cases = []
    with EventWaitEnv('/home/newbiexvwu/.cache/pvz-research-resources',executable=ROOT/'build/pvz-portable') as env:
        env.load_rollout_net(exported)
        for task_id in ('day_l9_eval_d7','night_l11_eval_n9','day_l10_orig_eval'):
            task = load_training_task(manifest,task_id); kw = training_task_kwargs(task,30000)
            spec = TaskSpec(level=kw['level'],seed=30000,playthrough=2,
                            profile=profile_for_deck(tuple(kw['deck']),kw.get('first_aid_owned',False)),
                            **{k:v for k,v in kw['task_extra'].items() if v is not None})
            obs, _ = env.reset(deck=tuple(kw['deck']),task=spec)
            history = StudentHistory(net.obs); previous = previous_action = None; count = 0
            for _ in range(64):
                assert not obs['terminal']
                frame = _frame(obs); frame['action'] = previous_action
                native = env.rollout_predict(previous['tick'] if previous else None,
                                             bool(previous_action and previous_action['type']=='plant'),student_history=history)
                board,scalars = history.features(frame,previous); info = packet_features(frame)[None]
                with torch.inference_mode():
                    h,g = net.encode(board[None],identity_scalars(frame,scalars)[None],packet_info=info)
                    pairs = [('plant',net.plant(g).flatten(),torch.tensor([native['plant']]),reference.plant(g.double()).flatten()),
                             ('packet',net.packet_logits(g,info)[0],torch.tensor(native['packet']),reference.packet_logits(g.double(),info)[0])]
                    if native['cell_packet'] >= 0:
                        pk = torch.tensor([native['cell_packet']])
                        pairs.append(('cell',net.cell_logits(h,g,pk,info)[0],torch.tensor(native['cell']),
                                      reference.cell_logits(h.double(),g.double(),pk,info)[0]))
                    for name,python,cpp,exact in pairs:
                        assert torch.isfinite(python).all() and torch.isfinite(cpp).all()
                        argmax += int(python.argmax()) != int(cpp.argmax())
                        samples[name].append((python.double(),cpp.double(),(python.double()-exact).abs()))
                chosen = act(net,frame,previous,obs['legal_actions']['plants'],history=history)
                actions += chosen != native['action']
                previous,previous_action = frame,chosen
                obs,*_,info_step = env.step(chosen); assert info_step.get('ok'); count += 1
            cases.append(dict(task=task_id,seed=30000,states=count))
    noise = {name:max(float(s[2].max()) for s in samples[name]) for name in heads}
    maxima = {}; ratios = {}; over = 0
    for name in heads:
        maxima[name] = max(float((a-b).abs().max()) for a,b,_ in samples[name])
        ratios[name] = max(float(((a-b).abs()/(1e-5+1e-5*a.abs()+2*noise[name])).max()) for a,b,_ in samples[name])
        over += sum(int(((a-b).abs() >= 1e-5+1e-5*a.abs()+2*noise[name]).sum()) for a,b,_ in samples[name])
    sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    result = dict(complete=True,passed=not(over or actions or argmax),width=net.trunk[0].out_channels,cases=cases,
                  states=sum(c['states'] for c in cases),model_sha256=sha(model_path),checker_sha256=sha(Path(__file__)),
                  executable_sha256=sha(ROOT/'build/pvz-portable'),max_abs_error=maxima,per_head_reference_noise=noise,
                  worst_tolerance_ratio=ratios,tolerance_overruns=over,action_mismatches=actions,argmax_mismatches=argmax,
                  tolerance='abs error < 1e-5 + 1e-5*abs(float32 Python logit) + 2*measured per-head float32-vs-float64 head noise; all actions/argmax exact')
    out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)
    assert result['passed'], 'Trained native parity failed; preserve evidence before interpreting capacity results'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args();check(args.model,args.out)
