"""Compare a completed identity checkpoint's CPU logits/actions in Python and native code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

import torch
from pvz_env import TaskSpec
from pvz_event_env import EventWaitEnv
from pvz_rollout_policy import act, export_student, identity_scalars, load_student, packet_features
from pvz_student_observation import StudentHistory
from pvz_tasks import profile_for_deck
from render_episode import _frame


def check(model_path, binary, resource_dir):
    torch.set_num_threads(1)
    model = load_student(model_path).eval()
    assert model.packet_head == 'identity' and model.obs == 'v2' and not model.context
    exported = model_path.with_suffix('.pvzrn')
    export_student(model, exported)
    assert exported.read_bytes()[:8] == b'PVZRN006'
    cases = [(9, (0, 1, 2, 3, 4, 5, 7)), (9, (7, 5, 4, 3, 2, 1, 0)),
             (9, (0, 1, 2, 3, 4, 5)), (10, (0, 1, 2, 3, 4, 5, 7))]
    maximum = 0.
    records = []
    with EventWaitEnv(resource_dir, executable=binary) as env:
        env.load_rollout_net(exported)
        for level, deck in cases:
            live, _ = env.reset(deck=deck, task=TaskSpec(level=level, seed=30000, profile=profile_for_deck(deck)))
            history = StudentHistory(model.obs)
            previous = action = None
            planted = empty = free = 0
            for _ in range(32):
                frame = _frame(live)
                frame['action'] = action
                legal = live['legal_actions']['plants']
                empty += any(p[1] < 0 for p in frame['packets'])
                free += any(frame['packets'][p['packet']][5] > frame['sun'] for p in legal)
                native = env.rollout_predict(previous['tick'] if previous else None,
                    bool(action and action['type'] == 'plant'), student_history=history)
                board, scalars = history.features(frame, previous)
                info = packet_features(frame)[None]
                with torch.inference_mode():
                    h, g = model.encode(board[None], identity_scalars(frame, scalars)[None], packet_info=info)
                    pairs = [(model.plant(g).flatten(), torch.tensor([native['plant']], dtype=torch.float32)),
                             (model.packet_logits(g, info)[0], torch.tensor(native['packet'], dtype=torch.float32))]
                    if native['cell_packet'] >= 0:
                        pairs.append((model.cell_logits(h, g, torch.tensor([native['cell_packet']]), info)[0],
                                      torch.tensor(native['cell'], dtype=torch.float32)))
                    for python, cpp in pairs:
                        maximum = max(maximum, float((python - cpp).abs().max()))
                        torch.testing.assert_close(python, cpp, rtol=0, atol=1e-5)
                        assert int(python.argmax()) == int(cpp.argmax())
                chosen = act(model, frame, previous, legal, history=history)
                assert chosen == native['action'], (level, frame['tick'], chosen, native['action'])
                planted += chosen['type'] == 'plant'
                previous, action = frame, chosen
                live, *_ = env.step(chosen)
            records.append(dict(level=level, initial_deck=deck, decisions=32, planted=planted,
                                empty_slot_positions=empty, free_card_positions=free))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(model_sha256=sha(model_path), exported_sha256=sha(exported),
                executable_sha256=sha(binary), checker_sha256=sha(Path(__file__)),
                maximum_abs_error=maximum, tolerance=1e-5, all_argmax_and_actions_equal=True,
                real_decisions=sum(r['decisions'] for r in records), cases=records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--executable', type=Path, default=ROOT / 'build/pvz-portable')
    parser.add_argument('--resource-dir', type=Path, default=Path.home() / '.cache/pvz-research-resources')
    args = parser.parse_args()
    assert not os.environ.get('CUDA_VISIBLE_DEVICES'), 'CPU parity only; use CUDA_VISIBLE_DEVICES=""'
    result = check(args.model.resolve(), args.executable.resolve(), args.resource_dir.resolve())
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
