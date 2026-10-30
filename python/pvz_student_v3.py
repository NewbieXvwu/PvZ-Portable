"""Feed-forward GameplayModelV1 encoder and three supervised pointer heads.

The retired model supplies the embedding/relations/pre-LN/SwiGLU design; no PPO,
recurrent state, critic, or event-wait modules are retained.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from pvz_student_tokens import FEATURE_COUNT
from pvz_value_net import BOARD_CHANNELS

from pvz_student_config import NETWORK_DEFAULTS, network_config

class SwiGLU(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.gate = nn.Linear(width, hidden)
        self.value = nn.Linear(width, hidden)
        self.down = nn.Linear(hidden, width)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.value(x))


class ExpertFFN(nn.Module):
    def __init__(self, width, config):
        super().__init__()
        self.mode, self.top_k = config['ffn'], config['top_k']
        self.experts = nn.ModuleList([SwiGLU(width, config['ff_width']) for _ in range(config['experts'])])
        self.shared = nn.ModuleList([SwiGLU(width, config['ff_width']) for _ in range(config['shared_experts'])])
        self.router = nn.Linear(width, config['experts']) if self.mode == 'moe' else None

    def forward(self, x, terrain):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        result = torch.zeros_like(flat)
        if self.mode == 'terrain':
            selected = terrain[:, None].expand(shape[:2]).reshape(-1, 1)
            weights = torch.ones_like(selected, dtype=x.dtype)
        else:
            logits = self.router(flat)
            selected = logits.argsort(dim=-1, descending=True, stable=True)[:, :self.top_k]
            weights = logits.gather(1, selected).softmax(-1)
        # Dispatch selected tokens only; shared experts see every token.
        for i, expert in enumerate(self.experts):
            positions, route = torch.where(selected == i)
            if positions.numel():
                result = result.index_add(0, positions, expert(flat[positions]) * weights[positions, route, None])
        for expert in self.shared:
            result = result + expert(flat)
        return result.reshape(shape)


class RelationAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_width = heads, width//heads
        self.qkv = nn.Linear(width, width*3)
        self.projection = nn.Linear(width, width)
        self.kind_pair_bias = nn.Parameter(torch.zeros(heads, 12, 12))
        self.row_bias = nn.Embedding(12, heads)
        self.col_bias = nn.Embedding(18, heads)
        self.same_cell_bias = nn.Embedding(2, heads)
        self.target_bias = nn.Parameter(torch.zeros(heads))

    def forward(self, x, tokens, relations):
        batch, count, width = x.shape
        q, k, v = self.qkv(x).reshape(batch, count, 3, self.heads, self.head_width).permute(2, 0, 3, 1, 4).unbind(0)
        kinds = tokens['ids'][:, :, 0]
        rb, cb, same, linked = relations
        bias = self.kind_pair_bias.permute(1, 2, 0)[kinds[:, :, None], kinds[:, None, :]]
        bias = bias + self.row_bias(rb) + self.col_bias(cb) + self.same_cell_bias(same)
        bias = bias.permute(0, 3, 1, 2) + self.target_bias[None, :, None, None]*linked[:, None]
        bias = bias.masked_fill(~tokens['mask'][:, None, None, :], float('-inf'))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.projection(out.transpose(1, 2).reshape(batch, count, width))


class RelationLayer(nn.Module):
    def __init__(self, width, config):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = RelationAttention(width, config['heads'])
        self.ff_norm = nn.LayerNorm(width)
        self.ff = (SwiGLU(width, config['ff_width']) if config['ffn'] == 'dense' else ExpertFFN(width, config))

    def forward(self, x, tokens, relations):
        x = x + self.attention(self.attention_norm(x), tokens, relations)
        z = self.ff_norm(x)
        return x + (self.ff(z) if isinstance(self.ff, SwiGLU) else self.ff(z, tokens['terrain']))


class StudentV3(nn.Module):
    def __init__(self, width=192, network=None, context=False, aux=False, token_input="compact"):
        super().__init__()
        self.token_input = token_input
        self.width = width
        self.network = network_config({'architecture': 'token', **(network or {})}, width)
        self.obs, self.context, self.packet_head = 'v3', context, 'identity'
        self.plant_embedding_dim = width
        self.kind_embedding = nn.Embedding(12, width)
        self.category_embedding = nn.Embedding(128, width)
        self.variant_embedding = nn.Embedding(128, width)
        self.row_embedding = nn.Embedding(8, width)
        self.col_embedding = nn.Embedding(11, width)
        self.feature_projection = nn.Sequential(nn.Linear(FEATURE_COUNT, width), nn.SiLU(), nn.Linear(width, width))
        self.encoder = nn.ModuleList([RelationLayer(width, self.network) for _ in range(self.network['layers'])])
        self.encoder_norm = nn.LayerNorm(width)
        self.plant = nn.Linear(width, 1)
        self.packet_query, self.packet_key = nn.Linear(width, width), nn.Linear(width, width)
        self.cell_key, self.plant_cell_query = nn.Linear(width, width), nn.Linear(width*2, width)
        self.cnn = (nn.Sequential(nn.Conv2d(BOARD_CHANNELS, width, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(width, width, 3, padding=1), nn.ReLU()) if self.network['hybrid'] else None)
        self.aux = nn.Linear(width, 1) if aux else None

    def encode_tokens(self, tokens, board=None):
        ids = tokens['ids']
        kinds, categories, variants, rows, cols = ids.unbind(-1)
        x = (self.kind_embedding(kinds) + self.category_embedding(categories) + self.variant_embedding(variants)
             + self.row_embedding((rows+1).clamp(0, 7)) + self.col_embedding((cols+1).clamp(0, 10))
             + self.feature_projection(tokens['features']))
        rk = (rows[:, :, None]>=0) & (rows[:, None, :]>=0)
        ck = (cols[:, :, None]>=0) & (cols[:, None, :]>=0)
        rb = torch.where(rk, (rows[:, :, None]-rows[:, None, :]).clamp(-5, 5)+5, 11)
        cb = torch.where(ck, (cols[:, :, None]-cols[:, None, :]).clamp(-8, 8)+8, 17)
        same = (rk & ck & (rb==5) & (cb==8)).long()
        linked = tokens['target_indices'][:, :, None] == torch.arange(x.shape[1], device=x.device)[None, None, :]
        for layer in self.encoder:
            x = layer(x, tokens, (rb, cb, same, linked))
        x = self.encoder_norm(x)
        # Pool public board state only: packet slots never enter the arithmetic mean.
        pool = tokens['mask'] & (kinds != 8)
        g = (x * pool[:, :, None]).sum(1) / pool.sum(1, keepdim=True).clamp_min(1)
        def gather(index):
            return x.gather(1, index.clamp_min(0)[:, :, None].expand(-1, -1, self.width))
        packets, cells = gather(tokens['packet_index']), gather(tokens['cell_index'])
        if self.cnn is not None:
            if board is None:
                raise ValueError('hybrid student requires board')
            grid = self.cnn(board).flatten(2).transpose(1, 2)
            spatial = tokens['mask'] & (kinds >= 3) & (kinds <= 7) & (rows>=0) & (rows<6) & (cols>=0) & (cols<9)
            index = (rows*10+cols).clamp(0, 59)
            grid = grid.scatter_add(1, index[:, :, None].expand_as(x), x*spatial[:, :, None])
            grid = grid.reshape(-1, 6, 10, self.width)
            cells = cells + grid[:, :, :9].reshape(-1, 54, self.width)
            g = g + grid.mean((1, 2))
        return dict(global_=g, packets=packets, cells=cells, packet_mask=tokens['packet_index']>=0)

    def packet_logits(self, encoded):
        logits = torch.einsum('bw,bpw->bp', self.packet_query(encoded['global_']), self.packet_key(encoded['packets'])) / math.sqrt(self.width)
        return logits.masked_fill(~encoded['packet_mask'], -1e9)

    def cell_logits(self, encoded, packet):
        selected = encoded['packets'].gather(1, packet[:, None, None].expand(-1, 1, self.width)).squeeze(1)
        q = self.plant_cell_query(torch.cat([encoded['global_'], selected], -1))
        return torch.einsum('bw,bcw->bc', q, self.cell_key(encoded['cells'])) / math.sqrt(self.width)

    def forward(self, board, scalars, packet=None, auxiliary=False, packet_info=None, tokens=None):
        if tokens is None:
            raise ValueError('v3 student requires public tokens')
        enc = self.encode_tokens(tokens, board)
        out = dict(plant=self.plant(enc['global_']).squeeze(1), packet=self.packet_logits(enc))
        if packet is not None:
            out['cell'] = self.cell_logits(enc, packet)
        if auxiliary:
            if self.aux is None:
                raise ValueError('auxiliary head disabled')
            out['aux'] = self.aux(enc['cells']).reshape(-1, 6, 9).mean(2)
        return out


def _frame_encoding(net, frame, previous, context):
    from pvz_rollout_policy import features
    from pvz_student_tokens import frame_tokens, collate_tokens
    if net.context and context is None:
        raise ValueError('v3 context student requires task context')
    context = context if net.context else None
    device = next(net.parameters()).device
    tokens = collate_tokens([frame_tokens(frame, context, net.token_input)], device)
    board = None
    if net.cnn is not None:
        board = features(frame, previous, context)[0][None].to(device)
    return net.encode_tokens(tokens, board)


@torch.inference_mode()
def choose_action(net, frame, previous, legal, context=None):
    from pvz_rollout_policy import WAIT
    if not legal:
        return dict(WAIT)
    encoded = _frame_encoding(net, frame, previous, context)
    if net.plant(encoded['global_']).item() <= 0:
        return dict(WAIT)
    packets = sorted({a['packet'] for a in legal})
    scores = net.packet_logits(encoded)[0]
    kinds = {int(p[0]): int(p[1]) for p in frame['packets']}
    packet = max(packets, key=lambda p: (float(scores[p]), -kinds[p]))
    cells = net.cell_logits(encoded, torch.tensor([packet], device=scores.device))[0]
    best = max((a for a in legal if a['packet'] == packet), key=lambda a: float(cells[a['row']*9+a['col']]))
    return dict(type='plant', packet=packet, row=best['row'], col=best['col'])


@torch.inference_mode()
def rank_actions(net, frame, previous, legal, context=None):
    if not legal:
        return []
    encoded = _frame_encoding(net, frame, previous, context)
    plant = float(net.plant(encoded['global_']).sigmoid().item())
    packets = sorted({a['packet'] for a in legal})
    scores = net.packet_logits(encoded)[0]
    probabilities = scores[packets].softmax(-1).tolist()
    result = []
    for packet, probability in zip(packets, probabilities):
        actions = [a for a in legal if a['packet'] == packet]
        cells = net.cell_logits(encoded, torch.tensor([packet], device=scores.device))[0]
        index = [a['row']*9+a['col'] for a in actions]
        for action, p in zip(actions, cells[index].softmax(-1).tolist()):
            result.append((plant*probability*p, action))
    return sorted(result, key=lambda item: -item[0])


def export_v3(net, path, graph=None, capacity=256, batch=1):
    from pvz_student_v3_export import export_student_v3
    if graph is not None:
        from pvz_generic_forward import export_graph
        return export_graph(net, graph, path)
    return export_student_v3(net, path, capacity, batch)
