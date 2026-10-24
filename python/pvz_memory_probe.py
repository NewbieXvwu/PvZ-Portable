"""Offline task58 models; no policy/export integration. T,B,C,H,W convention."""
from __future__ import annotations

import math

import torch
from torch import nn

CONFIGS = ('current', 'stack4', 'stack8', 'convgru', 'spatial_ssm')


class MemoryProbe(nn.Module):
    def __init__(self, kind, width, board_channels=102, scalar_channels=157):
        super().__init__()
        self.kind, self.width = kind, width
        self.k = int(kind[-1]) if kind.startswith('stack') else 1
        self.scalar = nn.Linear(scalar_channels, 16)
        self.stem = nn.Conv2d(board_channels + 16, 32, 1)
        if kind in ('current', 'stack4', 'stack8'):
            self.core = nn.Sequential(nn.Conv2d(32 * self.k, width, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(width, width, 3, padding=1), nn.ReLU())
        elif kind == 'convgru':
            self.input = nn.Conv2d(32, width, 3, padding=1)
            self.gates = nn.Conv2d(2 * width, 2 * width, 3, padding=1)
            self.candidate = nn.Conv2d(2 * width, width, 3, padding=1)
        elif kind == 'spatial_ssm':
            # Diagonal stable complex A, convolutional B/C; ZOH at actual elapsed ticks.
            self.log_rate = nn.Parameter(torch.linspace(math.log(.001), math.log(1.), width))
            self.frequency = nn.Parameter(torch.linspace(0., math.pi, width))
            self.drive = nn.Conv2d(32, 2 * width, 3, padding=1)
            self.read = nn.Conv2d(2 * width, width, 3, padding=1)
            self.skip = nn.Conv2d(32, width, 1)
        else:
            raise ValueError(kind)
        self.head = nn.Conv2d(width, 35, 1)

    def forward(self, board, scalar, dt, state=None):
        t, batch, _, rows, cols = board.shape
        scalars = torch.relu(self.scalar(scalar)).reshape(t * batch, 16, 1, 1)
        inputs = torch.cat((board.flatten(0, 1), scalars.expand(-1, -1, rows, cols)), 1)
        x = torch.relu(self.stem(inputs)).reshape(t, batch, 32, rows, cols)
        if self.kind in ('current', 'stack4', 'stack8'):
            if self.k > 1:
                if state is None:
                    state = x.new_zeros(self.k - 1, batch, 32, rows, cols)
                history = torch.cat((state, x))
                stacked = torch.cat([history[i:i+t] for i in range(self.k)], 2)
                state = history[-(self.k-1):]
            else:
                stacked = x
            hidden = self.core(stacked.flatten(0, 1))
            return self.head(hidden).reshape(t, batch, 35, rows, cols)[..., :9], state
        if state is None:
            channels = self.width * (2 if self.kind == 'spatial_ssm' else 1)
            state = x.new_zeros(batch, channels, rows, cols)
        outputs = []
        for index in range(t):
            current = x[index]
            if self.kind == 'convgru':
                u = torch.relu(self.input(current))
                reset, update = torch.sigmoid(self.gates(torch.cat((u, state), 1))).chunk(2, 1)
                candidate = torch.tanh(self.candidate(torch.cat((u, reset * state), 1)))
                state = (1 - update) * state + update * candidate
                hidden = state
            else:
                # Time unit = 60 engine ticks. No guessed velocity or zombie tracking.
                delta = dt[index, :, None, None, None] / 60.
                a = -self.log_rate.exp()[None, :, None, None]
                w = self.frequency[None, :, None, None]
                magnitude, angle = torch.exp(a * delta), w * delta
                er, ei = magnitude * torch.cos(angle), magnitude * torch.sin(angle)
                # (exp(lambda*dt)-1)/lambda, evaluated in real arithmetic.
                denom = a.square() + w.square()
                pr = ((er - 1) * a + ei * w) / denom
                pi = (ei * a - (er - 1) * w) / denom
                hr, hi = state.chunk(2, 1)
                br, bi = self.drive(current).chunk(2, 1)
                state = torch.cat((er * hr - ei * hi + pr * br - pi * bi,
                                   ei * hr + er * hi + pi * br + pr * bi), 1)
                hidden = torch.relu(self.read(state) + self.skip(current))
            outputs.append(self.head(hidden)[..., :9])
        return torch.stack(outputs), state


def make_probe(kind, budget=100_000, **kwargs):
    # All instantiated parameters participate in forward; no dummy padding parameters.
    choices = []
    for width in range(8, 160):
        model = MemoryProbe(kind, width, **kwargs)
        count = sum(p.numel() for p in model.parameters())
        choices.append((abs(count-budget), width))
    return MemoryProbe(kind, min(choices)[1], **kwargs)
