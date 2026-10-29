"""task162 controls; token encoder from studentv3net 2a866783, r42 tokens alongside boards."""
import torch
from torch import nn
import torch.nn.functional as F
from pvz_rollout_policy import RolloutNet
from pvz_value_net import SCALARS, CONTEXT, BOARD_CHANNELS
from pvz_student_observation import OBS_EXTRA


def terrain(scalars, context=False):
    start = SCALARS + 2 + (CONTEXT if context else 0)
    night, pool, fog, roof = scalars[:, start:start + 4].bool().unbind(1)
    return torch.where(roof, 4, torch.where(fog, 3, torch.where(pool, 2, night.long())))


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.ReLU(), nn.Conv2d(width, width, 3, padding=1))

    def forward(self, x):
        return F.relu(x + self.conv(x))


def resnet6(**options):
    net = RolloutNet(**options)
    net.trunk = nn.Sequential(net.trunk[0], nn.ReLU(), *[ResidualBlock(options['width']) for _ in range(6)])
    return net


class TerrainExperts(nn.Module):
    """Shared first two convolutions; independent third convolution/scalars/heads."""
    def __init__(self, **options):
        super().__init__()
        first = RolloutNet(**options)
        self.shared = first.trunk[:4]
        self.experts = nn.ModuleList([first] + [RolloutNet(**options) for _ in range(4)])
        for expert in self.experts:
            expert.trunk = expert.trunk[4:]
        for name in ('obs', 'context', 'packet_head', 'plant_embedding_dim', 'pool', 'row_pool'):
            setattr(self, name, getattr(first, name))

    def forward(self, board, scalars, packet=None, auxiliary=False, packet_info=None):
        hidden = self.shared(board)
        selected = terrain(scalars, self.context)
        result = {}
        for i, expert in enumerate(self.experts):
            rows = (selected == i).nonzero().flatten()
            if not len(rows):
                continue
            out = expert(hidden[rows], scalars[rows], None if packet is None else packet[rows], auxiliary,
                         None if packet_info is None else packet_info[rows])
            for key, value in out.items():
                if key not in result:
                    result[key] = value.new_zeros((len(board), *value.shape[1:]))
                result[key] = result[key].index_copy(0, rows, value)
        return result


class BoardMoE(nn.Module):
    """One shared and top-2 of four spatial experts; whole-board dispatch."""
    def __init__(self, width):
        super().__init__()
        self.shared = ResidualBlock(width)
        self.experts = nn.ModuleList([ResidualBlock(width) for _ in range(4)])
        self.router = nn.Linear(width, 4)
        self.routing_loss = None
        self.route_counts = torch.zeros(5, 4, dtype=torch.int64)
        self.current_terrain = None

    def forward(self, x):
        logits = self.router(x.mean((2, 3)))
        selected = logits.argsort(dim=1, descending=True, stable=True)[:, :2]
        weights = logits.gather(1, selected).softmax(1)
        result = self.shared(x)
        for i, expert in enumerate(self.experts):
            rows, slots = torch.where(selected == i)
            if len(rows):
                result = result.index_add(0, rows, expert(x[rows]) * weights[rows, slots, None, None, None])
        usage = F.one_hot(selected, 4).float().mean((0, 1))
        self.routing_loss = 4 * (usage.detach() * logits.softmax(1).mean(0)).sum()
        if not self.training and self.current_terrain is not None:
            with torch.no_grad():
                self.route_counts.index_add_(0, self.current_terrain.cpu(), F.one_hot(selected, 4).sum(1).cpu())
        return result


class MoEPolicy(RolloutNet):
    def __init__(self, **options):
        super().__init__(**options)
        self.trunk = nn.Sequential(*self.trunk[:4], BoardMoE(options['width']))

    @property
    def routing_loss(self):
        return self.trunk[-1].routing_loss

    def encode(self, board, scalars, **kwargs):
        self.trunk[-1].current_terrain = terrain(scalars, self.context)
        return super().encode(board, scalars, **kwargs)


def token_policy(size='small', hybrid=False, **options):
    from pvz_student_v3 import StudentV3
    width = options['width']
    net = StudentV3(width, dict(architecture='token', layers=2 if size == 'small' else 4,
        heads=4 if size == 'small' else 6, ff_width=width * 4, hybrid=hybrid),
        context=options.get('context', False), aux=options.get('aux', False))
    net.obs = options.get('obs', 'v2timer')
    net.requires_tokens = True
    if hybrid:
        net.cnn[0] = nn.Conv2d(BOARD_CHANNELS + OBS_EXTRA[net.obs][0], width, 3, padding=1)
    return net


def token_small(**options):
    return token_policy('small', **options)


def token_medium(**options):
    return token_policy('medium', **options)


def cnn_entity(**options):
    return token_policy('small', hybrid=True, **options)


def action(net, frame, previous, legal, context=None, history=None, ranked=False):
    """Python path for Mac bench_play; same public feature/token encoders as training."""
    from pvz_rollout_policy import identity_scalars, packet_features, WAIT
    from pvz_student_observation import StudentHistory
    if not legal:
        return [] if ranked else dict(WAIT)
    if history is None:
        if net.obs != 'v1':
            raise ValueError('research student requires StudentHistory updated at every decision')
        from pvz_rollout_policy import features
        board, scalars = features(frame, previous, context if net.context else None)
    else:
        board, scalars = history.features(frame, previous, context if net.context else None)
    info = packet_features(frame)[None]
    scalars = identity_scalars(frame, scalars)[None]
    args = dict(packet_info=info)
    if getattr(net, 'requires_tokens', False):
        from pvz_student_tokens import frame_tokens, collate_tokens
        args['tokens'] = collate_tokens([frame_tokens(frame, context if net.context else None)])
    out = net(board[None], scalars, **args)
    if not ranked and out['plant'].item() <= 0:
        return dict(WAIT)
    packets = sorted({a['packet'] for a in legal})
    probability = out['packet'][0, packets].softmax(0)
    chosen = packets if ranked else [packets[int(probability.argmax())]]
    result = []
    for p in chosen:
        cell = net(board[None], scalars, torch.tensor([p]), **args)['cell'][0]
        actions = [a for a in legal if a['packet'] == p]
        weights = cell[[a['row'] * 9 + a['col'] for a in actions]].softmax(0)
        result.extend((float(out['plant'].sigmoid().item() * probability[packets.index(p)] * w), a)
                      for w, a in zip(weights, actions))
    result.sort(key=lambda pair: -pair[0])
    return result if ranked else result[0][1]
