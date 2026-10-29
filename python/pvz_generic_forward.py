"""PVZGN001 graph description, nn.Module exporter and independent PyTorch reference.

Shapes specify capacity, including fixed batch. Native obs-v3 can execute shorter token lengths.
Tensor layouts are explicit; no tracing or Python
is embedded in the artifact. See MEMORY/docs/task161-cpp-generic-forward.md for ABI.
"""
from __future__ import annotations

import struct

import torch
import torch.nn.functional as F

MAGIC = b'PVZGN001'


class GraphSpec:
    def __init__(self, metadata=()):
        self.metadata = list(metadata)
        self.tensors = {}
        self.nodes = []

    def tensor(self, name, shape, role=1):
        if name in self.tensors:
            raise ValueError(f'duplicate tensor: {name}')
        self.tensors[name] = {'shape': list(shape), 'role': role}
        return name

    def weight(self, module, name):
        if name not in self.tensors:
            self.tensor(name, module.state_dict()[name].shape, 2)
        return name

    def node(self, name, op, inputs, shape, ints=(), floats=(), stage=0):
        self.tensor(name, shape, 0)
        self.nodes.append(dict(name=name, op=op, inputs=list(inputs), ints=list(ints),
                               floats=list(floats), stage=stage))
        return name


def export_graph(module: torch.nn.Module, graph: GraphSpec, path):
    """Export only state_dict tensors referenced by a supplied fixed-shape graph."""
    state = module.state_dict()
    ids = {name: i for i, name in enumerate(graph.tensors)}
    def string(f, value):
        raw = value.encode('utf-8')
        f.write(struct.pack('<I', len(raw)) + raw)
    def vector(f, values, kind='i'):
        f.write(struct.pack('<I', len(values)))
        if values:
            f.write(struct.pack('<' + kind * len(values), *values))
    with open(path, 'wb') as f:
        f.write(MAGIC)
        vector(f, graph.metadata)
        f.write(struct.pack('<I', len(ids)))
        for name, desc in graph.tensors.items():
            string(f, name)
            f.write(struct.pack('<i', desc['role']))
            vector(f, desc['shape'])
            if desc['role'] == 2:
                value = state[name].detach().cpu().to(torch.float32).contiguous()
                if list(value.shape) != desc['shape']:
                    raise ValueError(f'weight shape changed: {name}')
                f.write(value.numpy().astype('<f4', copy=False).tobytes())
        f.write(struct.pack('<I', len(graph.nodes)))
        for node in graph.nodes:
            string(f, node['op'])
            f.write(struct.pack('<ii', node['stage'], ids[node['name']]))
            vector(f, [ids[i] for i in node['inputs']])
            vector(f, node['ints'])
            vector(f, node['floats'], 'f')


def read_metadata(path):
    with open(path, 'rb') as f:
        if f.read(8) != MAGIC:
            raise ValueError('expected PVZGN001')
        count, = struct.unpack('<I', f.read(4))
        if count > 32:
            raise ValueError('invalid metadata length')
        return list(struct.unpack('<' + 'i' * count, f.read(4 * count)))


def _ffn(x, weights, swiglu):
    up, ub, down, db = weights
    z = F.linear(x, up, ub)
    z = F.silu(z.chunk(2, -1)[0]) * z.chunk(2, -1)[1] if swiglu else F.relu(z)
    return F.linear(z, down, db)


@torch.inference_mode()
def reference(module, graph, feeds, *, intermediates=False):
    """Independent tensor implementation for parity; not used by production inference."""
    values = dict(module.state_dict())
    values.update(feeds)
    for n in graph.nodes:
        a = [values[i] for i in n['inputs']]
        p, fp, op = n['ints'], n['floats'], n['op']
        x = a[0]
        shape = graph.tensors[n['name']]['shape']
        if op == 'linear': y = F.linear(*a)
        elif op in ('conv', 'conv_relu'):
            y = F.conv2d(*a, stride=tuple(p[2:4]), padding=tuple(p[:2]))
            if op == 'conv_relu': y = F.relu(y)
        elif op == 'relu': y = F.relu(x)
        elif op == 'silu': y = F.silu(x)
        elif op == 'gather': y = x.gather(1, a[1].long().clamp_min(0)[:, :, None].expand(-1, -1, x.shape[-1]))
        elif op == 'masked_pool': y = (x*a[1][:, :, None]).sum(1)/a[1].sum(1, keepdim=True).clamp_min(1)
        elif op == 'row_context':
            rows=torch.cat([x.mean(3,keepdim=True),x.amax(3,keepdim=True)],1)
            y=F.relu(x+F.conv2d(rows,a[1],a[2]))
        elif op == 'add': y = a[0] + a[1]
        elif op == 'concat': y = torch.cat(a, dim=p[0])
        elif op == 'reshape': y = x.reshape(shape)
        elif op == 'permute': y = x.permute(p)
        elif op == 'broadcast':
            present = [i for i in p if i >= 0]
            unused = [i for i in range(x.ndim) if i not in present]
            z = x.permute(present + unused).reshape([x.shape[i] if i >= 0 else 1 for i in p])
            y = z.expand(shape)
        elif op == 'pool':
            axes = tuple(range(x.ndim-p[0], x.ndim))
            mean, maximum = x.mean(axes), x.amax(axes)
            y = mean if p[1] == 0 else maximum if p[1] == 1 else torch.cat([mean, maximum], -1)
        elif op == 'norm':
            y = (F.layer_norm(x, (x.shape[-1],), a[1], a[2], fp[0]) if not p[0] else
                 x * torch.rsqrt(x.square().mean(-1, keepdim=True) + fp[0]) * a[1] + a[2])
        elif op == 'attention':
            B, T, D = x.shape
            H = p[0]
            q, k, v = [t.reshape(B, T, H, D//H).transpose(1, 2) for t in a[:3]]
            scores = (q @ k.transpose(-1, -2)) / (D//H)**0.5 + a[3]
            scores = scores.masked_fill(~(a[4] > 0)[:, None, None, :], -torch.inf)
            prob = scores.softmax(-1).nan_to_num(0)
            y = (prob @ v).transpose(1, 2).reshape(B, T, D) * (a[4] > 0)[:, :, None]
        elif op == 'relation':
            kinds, rows, cols = x.unbind(-1)
            kinds = kinds.long()
            dr = (rows[:, :, None]-rows[:, None, :]).clamp(-p[0], p[0]).long()+p[0]
            dc = (cols[:, :, None]-cols[:, None, :]).clamp(-p[1], p[1]).long()+p[1]
            dr = dr.masked_fill((rows[:, :, None]<0) | (rows[:, None, :]<0), 2*p[0]+1)
            dc = dc.masked_fill((cols[:, :, None]<0) | (cols[:, None, :]<0), 2*p[1]+1)
            same = ((rows[:, :, None] == rows[:, None, :]) & (cols[:, :, None] == cols[:, None, :]) &
                    (rows[:, :, None] >= 0) & (cols[:, :, None] >= 0))
            y = (a[1][:, kinds[:, :, None], kinds[:, None, :]] + a[2][:, dr] + a[3][:, dc] +
                 (same[None] * a[4][:, None, None, None] if a[4].ndim == 1 else a[4][:, same.long()])).permute(1, 0, 2, 3)
            if len(a) == 7:
                links = a[5][:, :, None] == torch.arange(x.shape[1], device=x.device)[None, None, :]
                y = y + a[6][None, :, None, None]*links[:, None]
        elif op == 'ffn': y = _ffn(x, a[1:], p[0])
        elif op == 'moe':
            swiglu, hard, experts, topk, shared = p
            flat = x.reshape(-1, x.shape[-1])
            y = torch.zeros_like(flat)
            weights = [a[2+4*i:6+4*i] for i in range(experts+shared)]
            for i in range(shared): y += _ffn(flat, weights[experts+i], swiglu)
            if hard:
                ids = a[1].reshape(-1).long()
                for i in range(experts): y[ids == i] += _ffn(flat[ids == i], weights[i], swiglu)
            else:
                # Stable sort defines ties by ascending expert id in both implementations.
                logits = a[1].reshape(-1, experts)
                ids = logits.argsort(dim=-1, descending=True, stable=True)[:, :topk]
                gates = logits.gather(1, ids).softmax(-1)
                for r in range(len(flat)):
                    for k in range(topk):
                        y[r] += gates[r, k] * _ffn(flat[r:r+1], weights[ids[r, k]], swiglu)[0]
            y = y.reshape(x.shape)
        elif op == 'scatter':
            y = x.clone()
            for b in range(x.shape[0]):
                for t in range(a[1].shape[1]):
                    r, c = a[2][b, t].tolist()
                    if 0 <= r < x.shape[2] and 0 <= c < x.shape[3]:
                        y[b, :, int(r), int(c)] += a[1][b, t]
        elif op == 'pointer': y = torch.einsum('bd,btd->bt', x, a[1]) * fp[0]
        elif op == 'embedding': y = F.embedding((x.long()+(p[0] if p else 0)).clamp(min=0, max=a[1].shape[0]-1) if p else x.long().clamp(min=0), a[1])
        elif op == 'select': y = x[torch.arange(x.shape[0]), a[1].long()]
        elif op == 'slice': y = x.narrow(*p)
        elif op == 'mask': y = x.masked_fill((a[1] < 0).reshape(a[1].shape + (1,)*(x.ndim-a[1].ndim)).expand_as(x), fp[0])
        else: raise ValueError(op)
        if list(y.shape) != shape:
            raise ValueError(f"{n['name']}: {list(y.shape)} != {shape}")
        values[n['name']] = y.contiguous()
    return values if intermediates else {n: values[n] for n in ('plant', 'packet', 'cell') if n in values}


class Builder:
    """Small explicit graph construction helpers; weights keep their state_dict names."""
    def __init__(self, module, graph):
        self.m, self.g = module, graph

    def layer(self, name, x, path, stage=0):
        layer = self.m.get_submodule(path)
        w = self.g.weight(self.m, path+'.weight')
        b = self.g.weight(self.m, path+'.bias') if layer.bias is not None else None
        shape = self.g.tensors[x]['shape'].copy()
        if isinstance(layer, torch.nn.Conv2d):
            shape = [shape[0], layer.out_channels] + [(shape[i+2]+2*layer.padding[i]-layer.kernel_size[i])//layer.stride[i]+1 for i in range(2)]
            params = [*layer.padding, *layer.stride]
            if x == 'board': params.append(1)  # sparse public feature channels; cached packed weights
            return self.g.node(name, 'conv', [x, w]+([b] if b else []), shape, params, stage=stage)
        shape[-1] = layer.out_features
        return self.g.node(name, 'linear', [x, w]+([b] if b else []), shape, stage=stage)

    def relu(self, name, x, stage=0):
        source = next((n for n in reversed(self.g.nodes) if n['name'] == x), None)
        if source and source['op'] == 'conv' and self.g.nodes[-1] is source:
            # Keep the graph's logical output explicit, including in intermediate parity dumps.
            shape=self.g.tensors.pop(x)['shape']
            self.g.nodes.pop()
            return self.g.node(name,'conv_relu',source['inputs'],shape,source['ints'],source['floats'],stage)
        return self.g.node(name, 'relu', [x], self.g.tensors[x]['shape'], stage=stage)


def rollout_graph(net, batch=1):
    """Represent all native RolloutNet variants with ordinary graph nodes.

    Native feature extraction already supplies identity-aggregated scalars. Python
    callers must use identity_scalars(frame, s) before passing identity-model feeds.
    """
    if net.obs == 'v2fogstack4':
        raise ValueError('native feature adapter does not support fog history stacks')
    versions = ('v1', 'v2lite', 'v2', 'v2fog', 'v2size', 'v2boss', 'v2bungee', 'v2sleep', 'v2timer')
    g = GraphSpec([versions.index(net.obs)+1, int(net.context), int(net.packet_head == 'identity')])
    b = Builder(net, g)
    W, E = net.trunk[0].out_channels, net.plant_embedding_dim
    G = W * (3 if net.pool == 'meanmax' else 2)
    g.tensor('board', [batch, net.trunk[0].in_channels, 6, 10])
    g.tensor('scalars', [batch, net.scalar[0].in_features])
    g.tensor('packet_info', [batch, 10, 4])
    g.tensor('selected_packet', [batch])
    h = 'board'
    for i in (0, 2, 4): h = b.relu(f'trunk_relu{i}', b.layer(f'trunk_conv{i}', h, f'trunk.{i}'))
    if net.row_pool:
        h = g.node('hidden', 'row_context', [h, g.weight(net, 'row_projection.weight'),
                    g.weight(net, 'row_projection.bias')], [batch,W,6,10])
    pooled = g.node('global_pool', 'pool', [h], [batch, G-W], [2, int(net.pool == 'meanmax')*2])
    scalar = b.relu('scalar_relu', b.layer('scalar_projection', 'scalars', 'scalar.0'))
    ctx = g.node('context', 'concat', [pooled, scalar], [batch, G], [1])
    b.layer('plant', ctx, 'plant', 1)
    table = g.weight(net, 'packet_embedding.weight')
    if net.packet_head == 'identity':
        kinds = g.node('packet_kind_3d', 'slice', ['packet_info'], [batch, 10, 1], [2, 0, 1], stage=1)
        kinds = g.node('packet_kind', 'reshape', [kinds], [batch, 10], stage=1)
        emb = g.node('packet_embeddings', 'embedding', [kinds, table], [batch, 10, E], stage=1)
        info = g.node('packet_public', 'slice', ['packet_info'], [batch, 10, 3], [2, 1, 3], stage=1)
        kernel=g.weight(net,'packet.0.weight'); bias=g.weight(net,'packet.0.bias')
        wk=g.node('packet_global_kernel','slice',[kernel],[W,G],[1,0,G],stage=1)
        ek=g.node('packet_local_kernel','slice',[kernel],[W,E+3],[1,G,E+3],stage=1)
        glob=g.node('packet_global_projected','linear',[ctx,wk,bias],[batch,W],stage=1)
        glob=g.node('packet_global_broadcast','broadcast',[glob],[batch,10,W],[0,-1,1],stage=1)
        pin = g.node('packet_input', 'concat', [emb, info], [batch, 10, E+3], [2], stage=1)
        local=g.node('packet_local_projected','linear',[pin,ek],[batch,10,W],stage=1)
        activation=g.node('packet_projection','add',[glob,local],[batch,10,W],stage=1)
        logits = b.layer('packet_score', b.relu('packet_relu', activation, 1), 'packet.2', 1)
        logits = g.node('packet_unmasked', 'reshape', [logits], [batch, 10], stage=1)
        g.node('packet', 'mask', [logits, kinds], [batch, 10], floats=[-1e9], stage=1)
        emb = g.node('selected_embedding', 'select', [emb, 'selected_packet'], [batch, E], stage=2)
    else:
        b.layer('packet', ctx, 'packet', 1)
        emb = g.node('selected_embedding', 'embedding', ['selected_packet', table], [batch, E], stage=2)
    # Split the first 1x1 head into local and global contributions. Global/card projections
    # are identical at every cell; compute them once. Constant kernel slices fold at load.
    kernel = g.weight(net, 'cell.0.weight')
    bias = g.weight(net, 'cell.0.bias')
    local_kernel = g.node('cell_local_kernel', 'slice', [kernel], [W, W, 1, 1], [1, 0, W], stage=2)
    global_kernel = g.node('cell_global_kernel4d', 'slice', [kernel], [W, E+G, 1, 1], [1, W, E+G], stage=2)
    global_kernel = g.node('cell_global_kernel', 'reshape', [global_kernel], [W, E+G], stage=2)
    global_input = g.node('cell_global_input', 'concat', [emb, ctx], [batch, E+G], [1], stage=2)
    glob = g.node('cell_global_projected', 'linear', [global_input, global_kernel, bias], [batch, W], stage=2)
    glob = g.node('cell_global_broadcast', 'broadcast', [glob], [batch, W, 6, 10], [0, 1, -1, -1], stage=2)
    local = g.node('cell_local_projected', 'conv', [h, local_kernel], [batch, W, 6, 10], [0, 0, 1, 1], stage=2)
    cin = g.node('cell_projection', 'add', [local, glob], [batch, W, 6, 10], stage=2)
    logits = b.layer('cell_score', b.relu('cell_relu', cin, 2), 'cell.2', 2)
    logits = g.node('cell_crop', 'slice', [logits], [batch, 1, 6, 9], [3, 0, 9], stage=2)
    g.node('cell', 'reshape', [logits], [batch, 54], stage=2)
    return g


def run_driver(module, graph, feeds, folder, driver, iterations=1, token_count=None):
    """Write bounded parity artifacts and run the standalone tool (development only)."""
    import json
    import subprocess
    from pathlib import Path
    import numpy as np
    folder = Path(folder)
    export_graph(module, graph, folder/'model.bin')
    with open(folder/'inputs.f32', 'wb') as f:
        for name, desc in graph.tensors.items():
            if desc['role'] == 1: f.write(feeds[name].numpy().astype('<f4').tobytes())
    command = [str(driver), str(folder/'model.bin'), str(folder/'inputs.f32'), str(folder/'outputs.f32'), str(iterations)]
    if token_count is not None: command.append(f'tokens={token_count}')
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    timing = json.loads(result.stdout)
    raw = np.fromfile(folder/'outputs.f32', dtype='<f4')
    outputs, pos = {}, 0
    for i, n in enumerate(graph.nodes):
        shape = timing['shapes'][i] if token_count is not None else graph.tensors[n['name']]['shape']
        size = int(np.prod(shape))
        outputs[n['name']] = torch.from_numpy(raw[pos:pos+size].copy().reshape(shape))
        pos += size
    assert pos == len(raw)
    timing.pop('shapes', None)
    return outputs, timing
