"""entity1 public tokens: group, kind, x, y, row, vx, vy, health, state, active, progress, exact.

Rows 0-9 are the ten seed-bank slots, in slot order, whether or not the slot holds a card.
Rows 10-127 are board entities: plants (54), zombies (40) and projectiles (24), sorted inside
each group so the encoding is deterministic.  Coordinates are screen pixels / (800, 600),
row / 5, velocity / 10, health or damage / 200, current animation state / 100.  No entity IDs.
Cards carry no coordinates and are never treated as board entities: they neither enter the
board pool nor take a relative-position bias.  Only the archived public frame fields are read.

The 2026-10-29 review found that pooling cards with board entities fed "which deck" into the
plant-or-wait context (an empty opening board pooled to the deck's mean embedding) and that
cards shared kind embedding rows with plants.  Groups now have their own embedding and kind
namespace, the card tokens are a separate output for the packet head, and training hides one
or two card tokens at random so the packet head cannot lean on the deck's composition.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ENTITY_CARDS = 10
ENTITY_LIMITS = (ENTITY_CARDS, 54, 40, 24)
ENTITY_TOKENS = sum(ENTITY_LIMITS)
ENTITY_FIELDS = 12
ENTITY_LAYERS = 2
ENTITY_HEADS = 4
ENTITY_TYPES = 174
ENTITY_GROUPS = 5          # 0 padding, 1 card, 2 plant, 3 zombie, 4 projectile
ENTITY_DECK_DROPOUT = 0.5  # training-only probability of hiding 1-2 card tokens
ROW_BUCKETS = 6            # |row difference| 0..5
COL_BUCKETS = 5            # |x difference| in cells: 0, <1, <2, <4, >=4


def entity_features(frame):
    result = np.zeros((ENTITY_TOKENS, ENTITY_FIELDS), np.float32)
    from pvz_rollout_policy import packet_features
    packets = packet_features(frame).numpy()
    for slot in range(ENTITY_CARDS):
        kind, active, progress, affordable = packets[slot]
        result[slot] = [1, kind, 0, 0, 0, 0, 0, 0, 0, active, progress, affordable]
    groups = ([], [], [])
    states = {(p[0], p[1], p[2]): p[3] for p in frame.get('v2', {}).get('plants', [])}
    for row, col, kind, health, maximum in frame['plants']:
        groups[0].append([2, min(kind, 53), (40 + col * 80) / 800, 0, row / 5,
                          0, 0, health / max(1, maximum), states.get((row, col, kind), -1) / 100,
                          0, 0, 0])
    for row, kind, position, speed, health, on_board, eating, chilled, ice, *_ in frame['zombies']:
        if on_board:
            groups[1].append([3, min(kind, 33), position / 800, 0, row / 5, speed / 10,
                              0, health / 200, -0.01, eating, bool(chilled or ice), 0])
    for row, position, damage in frame.get('v2', {}).get('projectiles', []):
        groups[2].append([4, 0, position / 800, 0, row / 5, 0, 0, damage / 200, -0.01, 0, 0, 0])
    tokens = [token for group, limit in zip(groups, ENTITY_LIMITS[1:]) for token in sorted(group)[:limit]]
    if tokens:
        result[ENTITY_CARDS:ENTITY_CARDS + len(tokens)] = tokens
    return torch.from_numpy(result)


class EntityLayer(nn.Module):
    """Multi-head self-attention over board tokens with learned relative-position tables.

    Card queries attend to board tokens only (cross-attention); board queries see board tokens
    only.  Padding and hidden cards are neither queries nor keys in practice: queries are
    zeroed after the layer, and an all-masked key row renormalizes to zero attention.
    """

    def __init__(self, width, heads=ENTITY_HEADS):
        super().__init__()
        if width % heads:
            raise ValueError('entity width must be divisible by the head count')
        self.heads, self.head_width = heads, width // heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.row_same = nn.Parameter(torch.zeros(heads, 2))
        self.row_bucket = nn.Parameter(torch.zeros(heads, ROW_BUCKETS))
        self.col_bucket = nn.Parameter(torch.zeros(heads, COL_BUCKETS))
        self.output = nn.Linear(width, width)
        self.ffnorm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.ReLU(), nn.Linear(2 * width, width))

    def forward(self, tokens, positions, board, valid):
        batch, length, width = tokens.shape
        query, key, value = self.qkv(self.norm(tokens)).reshape(
            batch, length, 3, self.heads, self.head_width).permute(2, 0, 3, 1, 4)
        score = query @ key.transpose(-1, -2) / self.head_width ** .5           # [B, H, N, N]
        row, col = positions[:, :, 0], positions[:, :, 1]
        same = (row[:, :, None] == row[:, None, :]).long()
        drow = (row[:, :, None] - row[:, None, :]).abs().round().long().clamp(max=ROW_BUCKETS - 1)
        dcol = torch.bucketize((col[:, :, None] - col[:, None, :]).abs(),
                               torch.tensor([.5, 1.5, 3.5, 6.0], device=col.device))
        bias = (self.row_same.t()[same] + self.row_bucket.t()[drow] + self.col_bucket.t()[dcol])
        bias = bias.permute(0, 3, 1, 2)                                          # [B, H, N, N]
        pair_board = (board[:, :, None] & board[:, None, :])[:, None]
        score = score + bias * pair_board
        keys = valid & board                                                    # [B, N]
        score = score.masked_fill(~keys[:, None, None, :], -1e9)
        attention = score.softmax(-1) * keys[:, None, None, :]
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-6)
        update = (attention @ value).transpose(1, 2).reshape(batch, length, width)
        tokens = tokens + self.output(update)
        return tokens + self.ff(self.ffnorm(tokens))


class EntityEncoder(nn.Module):
    def __init__(self, width, heads=ENTITY_HEADS, layers=ENTITY_LAYERS,
                 deck_dropout=ENTITY_DECK_DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(ENTITY_TYPES, width)
        self.group = nn.Embedding(ENTITY_GROUPS, width)
        self.input = nn.Linear(ENTITY_FIELDS - 2, width)
        self.layers = nn.ModuleList([EntityLayer(width, heads) for _ in range(layers)])
        self.output = nn.Linear(width, width)
        self.deck_dropout = deck_dropout

    def hidden_cards(self, valid, card):
        """Training-only deck augmentation: hide one or two card tokens per sample."""
        hidden = torch.zeros_like(valid)
        if not self.training or self.deck_dropout <= 0:
            return hidden
        batch = valid.shape[0]
        dropped = torch.rand(batch, device=valid.device) < self.deck_dropout
        count = torch.randint(1, 3, (batch,), device=valid.device)
        scores = torch.rand(batch, ENTITY_CARDS, device=valid.device).masked_fill(
            ~card[:, :ENTITY_CARDS], 2.0)
        ranks = scores.argsort(1).argsort(1)
        hidden[:, :ENTITY_CARDS] = (ranks < count[:, None]) & card[:, :ENTITY_CARDS] & dropped[:, None]
        return hidden

    def forward(self, entities):
        group = entities[:, :, 0].long().clamp(0, ENTITY_GROUPS - 1)
        kind = entities[:, :, 1].long()
        card = group == 1
        valid = (group > 0) & (kind >= 0)
        valid = valid & ~self.hidden_cards(valid, card)
        # Board rows 10+ are truncated at the last valid token, not at the valid count: the ten
        # card slots are reserved even when some of them are empty.
        last = int(valid.nonzero()[:, 1].max()) + 1 if valid.any() else ENTITY_CARDS
        length = min(ENTITY_TOKENS, max(ENTITY_CARDS, last))
        if length < entities.shape[1]:
            group, kind, card = group[:, :length], kind[:, :length], card[:, :length]
            valid, entities = valid[:, :length], entities[:, :length]
        length = entities.shape[1]
        offsets = torch.tensor([0, 0, 54, 108, 142], device=entities.device)
        kinds = (kind + offsets[group]).clamp(0, ENTITY_TYPES - 1)
        tokens = F.relu(self.embedding(kinds) + self.group(group) + self.input(entities[:, :, 2:]))
        tokens = tokens * valid[:, :, None]
        positions = torch.stack((entities[:, :, 4] * 5, entities[:, :, 2] * 10), dim=-1)
        board = valid & ~card
        for layer in self.layers:
            tokens = layer(tokens, positions, board, valid)
        tokens = tokens * valid[:, :, None]
        pooled = (tokens * board[:, :, None]).sum(1) / board.sum(1, keepdim=True).clamp_min(1)
        cards = tokens[:, :ENTITY_CARDS] * valid[:, :ENTITY_CARDS, None]
        return self.output(pooled), cards
