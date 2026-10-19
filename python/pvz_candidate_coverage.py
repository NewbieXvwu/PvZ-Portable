"""Optional, learned marginal-coverage proposer. No plant identities or board rules.

Training uses paired continuation values, conditioned on the existing candidate set.
The teacher still evaluates every proposed action and owns the final choice.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

SCHEMA = 1


def action_key(action):
    if action.get('type', 'plant') == 'plant':
        return ('plant', int(action['packet']), int(action['row']), int(action['col']))
    return (action['type'],)


def features(frame, context, actions, static_scores, student_scores):
    from pvz_value_net import featurize, ROWS, COLS
    board, scalars = featurize(frame)
    state = np.concatenate((board.numpy().reshape(-1), scalars.numpy(), np.asarray(context))).astype('float32')
    packets = {int(p[0]): p for p in frame['packets']}
    student = {action_key(a): float(p) for p, a in student_scores}
    static = {action_key(a): float(v) for v, a in static_scores}
    by_card = {}
    for i, a in enumerate(actions):
        if a['type'] == 'plant':
            by_card.setdefault(a['packet'], []).append(i)
    ranks = [{}, {}]
    gaps = [{}, {}]
    for ids in by_card.values():
        for dim, scores in enumerate((student, static)):
            ordered = sorted(ids, key=lambda i: -scores.get(action_key(actions[i]), 0.))
            best = scores.get(action_key(actions[ordered[0]]), 0.)
            for rank, i in enumerate(ordered):
                ranks[dim][i] = rank / max(1, len(ids)-1)
                gaps[dim][i] = best - scores.get(action_key(actions[i]), 0.)
    rows = []
    for i, a in enumerate(actions):
        cell = np.zeros(ROWS * COLS, dtype='float32')
        packet = np.zeros(11, dtype='float32')  # seedbank max10, plus wait
        plant = a['type'] == 'plant'
        if plant:
            cell[int(a['row']) * COLS + int(a['col'])] = 1
            packet[int(a['packet'])] = 1
        else:
            packet[-1] = 1
        k = action_key(a)
        numeric = [float(packets[a['packet']][-1]) if plant else 0.,
                   student.get(k, max(0., 1.-sum(student.values())) if not plant else 0.),
                   static.get(k, 0.), ranks[0].get(i, 0.), ranks[1].get(i, 0.),
                   gaps[0].get(i, 0.), gaps[1].get(i, 0.), float(not plant)]
        rows.append(np.concatenate((cell, packet, numeric)))
    return state, np.asarray(rows, dtype='float32')


class CoverageNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU())
        self.action = nn.Sequential(nn.Linear(action_dim, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(160, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        for name, size in [('state_mean', state_dim), ('state_scale', state_dim),
                           ('action_mean', action_dim), ('action_scale', action_dim)]:
            self.register_buffer(name, torch.ones(size) if name.endswith('scale') else torch.zeros(size))

    def forward(self, state, actions, selected):
        s = self.state((state-self.state_mean)/self.state_scale)
        a = self.action((actions-self.action_mean)/self.action_scale)
        chosen = a[selected]
        pooled = torch.cat((chosen.mean(0), chosen.max(0).values))
        return self.head(torch.cat((s.expand(len(a), -1), a, pooled.expand(len(a), -1)), dim=1)).flatten()


def load(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['schema_version'] != SCHEMA:
        raise ValueError('unsupported candidate coverage schema')
    model = CoverageNet(saved['state_dim'], saved['action_dim'])
    model.load_state_dict(saved['state_dict'], strict=True)
    model.eval()
    return model


def propose(model, state, action_features, actions, candidates, budget):
    if budget < 0:
        raise ValueError('candidate coverage budget must be nonnegative')
    lookup = {action_key(a): i for i, a in enumerate(actions)}
    selected = [lookup[action_key(a)] for a in candidates if action_key(a) in lookup]
    if not selected:
        raise ValueError('coverage requires at least one existing candidate')
    output = []
    with torch.inference_mode():
        state, x = torch.as_tensor(state), torch.as_tensor(action_features)
        for _ in range(budget):
            available = [i for i, a in enumerate(actions) if a['type'] == 'plant' and i not in selected]
            if not available:
                break
            scores = model(state, x, selected)
            winner = max(available, key=lambda i: float(scores[i]))
            selected.append(winner)
            output.append(actions[winner])
    return output
