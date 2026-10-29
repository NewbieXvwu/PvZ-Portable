"""无记忆的 GameplayModelV1 编码器与学生三头，输入由 C++ obs-v3 产生。"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

CELL_START, CARD_START, CARD_END = 2, 56, 66
TOKEN_FIELDS = 38


def token_array(frame):
    v3 = frame.get("v3")
    if not v3 or v3.get("version") != 3:
        raise ValueError("token_v3 requires C++ obs-v3 frames; backfill old archives first")
    return np.asarray(v3["tokens"], dtype=np.float32)


def token_batch(arrays, device="cpu"):
    # No corpus-wide padding or entity truncation. Padding is local to each minibatch.
    out = np.zeros((len(arrays), max(len(a) for a in arrays), TOKEN_FIELDS), np.float32)
    out[:, :, 0] = -1
    out[:, :, 3:6] = -1
    for i, array in enumerate(arrays):
        out[i, :len(array)] = array
    return torch.from_numpy(out).to(device)


class RelationLayer(nn.Module):
    def __init__(self, width, heads, ff_width):
        super().__init__()
        self.heads, self.head_width = heads, width // heads
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.kind_pair_bias = nn.Parameter(torch.zeros(heads, 12, 12))
        self.row_bias = nn.Embedding(12, heads)
        self.col_bias = nn.Embedding(18, heads)
        self.same_cell_bias = nn.Embedding(2, heads)
        self.target_bias = nn.Parameter(torch.zeros(heads))
        self.ff_norm = nn.LayerNorm(width)
        self.gate = nn.Linear(width, ff_width)
        self.value = nn.Linear(width, ff_width)
        self.down = nn.Linear(ff_width, width)

    def forward(self, x, kinds, relation, key_valid):
        batch, count, width = x.shape
        row, col, same, spatial, linked = relation
        bias = self.kind_pair_bias.permute(1, 2, 0)[kinds[:, :, None], kinds[:, None, :]]
        position = self.row_bias(row) + self.col_bias(col) + self.same_cell_bias(same)
        bias = bias + position * spatial[:, :, :, None]
        bias = bias.permute(0, 3, 1, 2) + self.target_bias[None, :, None, None] * linked[:, None]
        bias = bias.masked_fill(~key_valid[:, None, None, :], -1e9)
        qkv = self.qkv(self.attention_norm(x)).reshape(batch, count, 3, self.heads, self.head_width)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.projection(attended.transpose(1, 2).reshape(batch, count, width))
        norm = self.ff_norm(x)
        return x + self.down(F.silu(self.gate(norm)) * self.value(norm))


class TokenStudent(nn.Module):
    arch, obs, packet_head = "token_v3", "v3", "identity"

    def __init__(self, width=192, layers=4, heads=6, ff_width=768, aux=False):
        super().__init__()
        if min(width, layers, heads, ff_width) < 1 or width % heads:
            raise ValueError("token width/layers/heads/ff_width must be positive; width divisible by heads")
        self.config = dict(width=width, layers=layers, heads=heads, ff_width=ff_width)
        self.plant_embedding_dim = width
        self.kind_embedding = nn.Embedding(12, width)
        self.category_embedding = nn.Embedding(128, width)
        self.variant_embedding = nn.Embedding(128, width)
        self.feature_projection = nn.Sequential(nn.Linear(32, width), nn.SiLU(), nn.Linear(width, width))
        self.row_embedding = nn.Embedding(8, width)
        self.col_embedding = nn.Embedding(11, width)
        self.encoder = nn.ModuleList(RelationLayer(width, heads, ff_width) for _ in range(layers))
        self.encoder_norm = nn.LayerNorm(width)
        self.plant = nn.Linear(width, 1)
        self.packet_query = nn.Linear(width, width)
        self.packet_key = nn.Linear(width, width)
        self.cell_key = nn.Linear(width, width)
        self.plant_cell_query = nn.Linear(2 * width, width)
        self.aux = nn.Linear(width, 1) if aux else None

    def encode(self, tokens):
        ids = tokens[:, :, :6].long()
        kinds, categories, variants, rows, cols, targets = ids.unbind(2)
        valid = kinds >= 0
        kinds = kinds.clamp_min(0)
        x = (self.kind_embedding(kinds) + self.category_embedding(categories)
             + self.variant_embedding(variants) + self.feature_projection(tokens[:, :, 6:])
             + self.row_embedding(rows + 1) + self.col_embedding(cols + 1))
        row_known = (rows[:, :, None] >= 0) & (rows[:, None, :] >= 0)
        col_known = (cols[:, :, None] >= 0) & (cols[:, None, :] >= 0)
        row_delta = (rows[:, :, None] - rows[:, None, :]).clamp(-5, 5) + 5
        col_delta = (cols[:, :, None] - cols[:, None, :]).clamp(-8, 8) + 8
        same = (row_known & col_known & (row_delta == 5) & (col_delta == 8)).long()
        # Cards read the board but can never enter any board token, including the global query.
        card = kinds == 8
        spatial = ~card[:, :, None] & ~card[:, None, :]
        linked = targets[:, :, None] == torch.arange(tokens.shape[1], device=tokens.device)[None, None, :]
        relation = (torch.where(row_known, row_delta, 11), torch.where(col_known, col_delta, 17),
                    same, spatial, linked)
        for layer in self.encoder:
            x = layer(x, kinds, relation, valid & ~card)
        return self.encoder_norm(x)

    def forward(self, board, scalars=None, packet=None, auxiliary=False, packet_info=None):
        x = self.encode(board)
        global_state, cards, cells = x[:, 0], x[:, CARD_START:CARD_END], x[:, CELL_START:CARD_START]
        scale = self.config["width"] ** -.5
        packet_logits = (self.packet_query(global_state)[:, None] * self.packet_key(cards)).sum(2) * scale
        packet_logits = packet_logits.masked_fill(board[:, CARD_START:CARD_END, 1] == 0, -1e9)
        result = {"plant": self.plant(global_state).squeeze(1), "packet": packet_logits}
        if packet is not None:
            selected = cards[torch.arange(len(cards), device=cards.device), packet.long()]
            query = self.plant_cell_query(torch.cat((global_state, selected), dim=1))
            result["cell"] = (query[:, None] * self.cell_key(cells)).sum(2) * scale
        if auxiliary:
            result["aux"] = self.aux(cells).reshape(-1, 6, 9).mean(2)
        return result

    @torch.inference_mode()
    def act(self, frame, legal_plants):
        tokens = token_batch([token_array(frame)], next(self.parameters()).device)
        out = self(tokens)
        if out["plant"].item() <= 0 or not legal_plants:
            return {"type": "wait", "ticks": 60, "until": "timeout"}
        slots = sorted({a["packet"] for a in legal_plants})
        packet = max(slots, key=lambda i: out["packet"][0, i].item())
        cells = self(tokens, packet=torch.tensor([packet], device=tokens.device))["cell"][0]
        chosen = max((a for a in legal_plants if a["packet"] == packet),
                     key=lambda a: cells[a["row"] * 9 + a["col"]].item())
        return {"type": "plant", **{k: chosen[k] for k in ("packet", "row", "col")}}

    def checkpoint(self):
        return dict(arch=self.arch, obs=self.obs, token_config=self.config,
                    state_dict={k: v.detach().cpu() for k, v in self.state_dict().items()})
