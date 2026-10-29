"""entity1 public tokens: group, kind, x, y, row, vx, vy, health, state, active, progress, exact.

Coordinates are screen pixels / (800, 600), row / 5, velocity / 10,
health or damage / 200, and current animation state / 100. No entity IDs.
Old archives retain their recorded information and mark missing precise fields.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ENTITY_LIMITS = (10, 54, 40, 24)
ENTITY_TOKENS = sum(ENTITY_LIMITS)
ENTITY_FIELDS = 12
ENTITY_LAYERS = 2
ENTITY_TYPES = 174


def entity_features(frame):
    rich = frame.get('v2', {}).get('entities')
    if rich is not None:
        result = np.zeros((ENTITY_TOKENS, ENTITY_FIELDS), np.float32)
        tokens = np.asarray(rich, np.float32).reshape(-1, ENTITY_FIELDS)[:ENTITY_TOKENS]
        result[:len(tokens)] = tokens
        return torch.from_numpy(result)
    from pvz_rollout_policy import packet_features
    groups = [[], [], [], []]
    for kind, active, progress, affordable in packet_features(frame).tolist():
        if kind >= 0:
            groups[0].append([1, kind, 0, 0, 0, 0, 0, 0, 0, active, progress, affordable])
    states = {(p[0], p[1], p[2]): p[3] for p in frame.get('v2', {}).get('plants', [])}
    for row, col, kind, health, maximum in frame['plants']:
        groups[1].append([2, min(kind, 53), (40 + col * 80) / 800, 0, row / 5,
                          0, 0, health / max(1, maximum), states.get((row, col, kind), -1) / 100,
                          0, 0, 0])
    for row, kind, position, speed, health, on_board, eating, chilled, ice, *_ in frame['zombies']:
        if on_board:
            groups[2].append([3, min(kind, 33), position / 800, 0, row / 5, speed / 10,
                              0, health / 200, -0.01, eating, bool(chilled or ice), 0])
    for row, position, damage in frame.get('v2', {}).get('projectiles', []):
        groups[3].append([4, 0, position / 800, 0, row / 5, 0, 0, damage / 200, -0.01, 0, 0, 0])
    result = np.zeros((ENTITY_TOKENS, ENTITY_FIELDS), np.float32)
    tokens = [token for group, limit in zip(groups, ENTITY_LIMITS) for token in sorted(group)[:limit]]
    if tokens:
        result[:len(tokens)] = tokens
    return torch.from_numpy(result)


class EntityLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.relative = nn.Parameter(torch.zeros(2))
        self.output = nn.Linear(width, width)
        self.ffnorm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.ReLU(), nn.Linear(2 * width, width))

    def forward(self, tokens, positions, valid):
        query, key, value = self.qkv(self.norm(tokens)).chunk(3, dim=-1)
        distance = (positions[:, :, None] - positions[:, None, :]).abs()
        score = query @ key.transpose(1, 2) / query.shape[-1] ** .5
        score = score + (distance * self.relative).sum(-1)
        score = score.masked_fill(~valid[:, None, :], -1e9)
        tokens = tokens + self.output(score.softmax(-1) @ value)
        return tokens + self.ff(self.ffnorm(tokens))


class EntityEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.embedding = nn.Embedding(ENTITY_TYPES, width)
        self.input = nn.Linear(ENTITY_FIELDS - 2, width)
        self.layers = nn.ModuleList([EntityLayer(width) for _ in range(ENTITY_LAYERS)])
        self.output = nn.Linear(width, width)

    def forward(self, entities):
        valid = entities[:, :, 0] > 0
        length = max(1, int(valid.sum(1).max()))
        entities, valid = entities[:, :length], valid[:, :length]
        group = entities[:, :, 0].long()
        offsets = torch.tensor([0, 0, 54, 108, 142], device=entities.device)
        kinds = (entities[:, :, 1].long() + offsets[group]).clamp(0, ENTITY_TYPES - 1)
        tokens = F.relu(self.embedding(kinds) + self.input(entities[:, :, 2:]))
        positions = torch.stack((entities[:, :, 4] * 5, entities[:, :, 2] * 10), dim=-1)
        for layer in self.layers:
            tokens = layer(tokens, positions, valid)
        pooled = (tokens * valid[:, :, None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        return self.output(pooled)
