"""Untrained, reproducible example graphs for task161 parity and CPU timings."""
import torch
from torch import nn
from pvz_generic_forward import GraphSpec, Builder, rollout_graph


def token_example(layers=2, tokens=64, width=128, batch=1, hybrid=False, moe=False):
    m = nn.Module()
    g = GraphSpec()
    b = Builder(m, g)
    def module(name, layer):
        m.add_module(name, layer)
        return name
    def weight(name, value):
        m.register_buffer(name, value)
        return g.weight(m, name)
    def norm(name, x):
        module(name, nn.LayerNorm(width))
        return g.node(name+'_out', 'norm', [x, g.weight(m, name+'.weight'), g.weight(m, name+'.bias')],
                      [batch, tokens, width], [0], [1e-5])
    def ffn_weights(name, activation=1):
        up = module(name+'_up', nn.Linear(width, width*4*(2 if activation else 1)))
        down = module(name+'_down', nn.Linear(width*4, width))
        return [g.weight(m, up+'.weight'), g.weight(m, up+'.bias'),
                g.weight(m, down+'.weight'), g.weight(m, down+'.bias')]
    x = g.tensor('tokens', [batch, tokens, width])
    valid = g.tensor('valid', [batch, tokens])
    meta = g.tensor('token_meta', [batch, tokens, 3])
    selected = g.tensor('selected_packet', [batch])
    H = 4
    tables = [weight('kind_bias', torch.randn(H, 12, 12)*.02),
              weight('row_bias', torch.randn(H, 14)*.02),
              weight('col_bias', torch.randn(H, 22)*.02),
              weight('same_bias', torch.randn(H)*.02)]
    bias = g.node('relation', 'relation', [meta, *tables], [batch, H, tokens, tokens], [6, 10])
    for layer in range(layers):
        pre = f'block{layer}'
        z = norm(pre+'_ln1', x)
        qkv = [b.layer(pre+'_'+name, z, module(pre+'_'+name+'_proj', nn.Linear(width, width))) for name in ('q', 'k', 'v')]
        a = g.node(pre+'_attn', 'attention', [*qkv, bias, valid], [batch, tokens, width], [H])
        a = b.layer(pre+'_proj', a, module(pre+'_out_proj', nn.Linear(width, width)))
        x = g.node(pre+'_res1', 'add', [x, a], [batch, tokens, width])
        z = norm(pre+'_ln2', x)
        if moe:
            route = g.tensor(pre+'_terrain', [batch, tokens])
            weights = sum([ffn_weights(pre+f'_expert{i}') for i in range(5)], [])
            z = g.node(pre+'_moe', 'moe', [z, route, *weights], [batch, tokens, width], [1, 1, 5, 1, 0])
        else:
            z = g.node(pre+'_ffn', 'ffn', [z, *ffn_weights(pre+'_ffn')], [batch, tokens, width], [1])
        x = g.node(pre+'_res2', 'add', [x, z], [batch, tokens, width])
    if hybrid:
        board = g.tensor('board', [batch, 92, 6, 10])
        grid = b.relu('cnn_relu', b.layer('cnn', board, module('cnn_proj', nn.Conv2d(92, width, 3, padding=1))))
        coords = g.tensor('coords', [batch, tokens, 2])
        grid = g.node('scatter', 'scatter', [grid, x, coords], [batch, width, 6, 10])
        query = g.node('hybrid_global', 'pool', [grid], [batch, width], [2, 0])
        keys = g.node('grid_channels_last', 'permute', [grid], [batch, 6, 10, width], [0, 2, 3, 1])
        keys = g.node('grid_lawn', 'slice', [keys], [batch, 6, 9, width], [2, 0, 9])
        keys = g.node('grid_tokens', 'reshape', [keys], [batch, 54, width])
    else:
        keys = g.node('cell_keys', 'slice', [x], [batch, 54, width], [1, 0, 54])
        # Packet tokens are excluded from global pooling, matching the task157 contract.
        pooled = g.node('pool_transpose', 'permute', [keys], [batch, width, 54], [0, 2, 1])
        query = g.node('global', 'pool', [pooled], [batch, width], [1, 0])
    packets = g.node('packet_keys', 'slice', [x], [batch, 10, width], [1, 54, 10])
    b.layer('plant', query, module('plant_head', nn.Linear(width, 1)), 1)
    g.node('packet', 'pointer', [query, packets], [batch, 10], floats=[width**-.5], stage=1)
    card = g.node('selected_key', 'select', [packets, selected], [batch, width], stage=2)
    cq = g.node('cell_query', 'add', [query, card], [batch, width], stage=2)
    g.node('cell', 'pointer', [cq, keys], [batch, 54], floats=[width**-.5], stage=2)
    return m.eval(), g


def residual_example(blocks=6, width=64, batch=1):
    # A regular RolloutNet graph, replacing its trunk with a configurable residual stack.
    from pvz_rollout_policy import RolloutNet
    m = RolloutNet(width, 'meanmax', obs='v2timer', row_pool=True, packet_head='identity').eval()
    original = rollout_graph(m, batch)
    g = GraphSpec(original.metadata)
    for name, desc in original.tensors.items():
        if desc['role'] == 1: g.tensor(name, desc['shape'])
    b = Builder(m, g)
    h = b.relu('stem_relu', b.layer('stem', 'board', 'trunk.0'))
    for i in range(blocks):
        for j in range(2): m.add_module(f'res{i}_{j}', nn.Conv2d(width, width, 3, padding=1))
        z = b.relu(f'res{i}_relu', b.layer(f'res{i}_conv0', h, f'res{i}_0'))
        z = b.layer(f'res{i}_conv1', z, f'res{i}_1')
        h = b.relu(f'res{i}_out', g.node(f'res{i}_add', 'add', [h, z], [batch, width, 6, 10]))
    # The remaining row-pool/scalar/three-head graph is unchanged.
    start = next(i for i, n in enumerate(original.nodes) if n['op'] == 'row_context')
    for n in original.nodes[start:]:
        for name in n['inputs']:
            if name in original.tensors and original.tensors[name]['role'] == 2:
                g.weight(m, name)
        inputs = [h if name == 'trunk_relu4' else name for name in n['inputs']]
        g.node(n['name'], n['op'], inputs, original.tensors[n['name']]['shape'], n['ints'], n['floats'], n['stage'])
    return m, g


def example(name, batch=1):
    from pvz_rollout_policy import RolloutNet
    if name in ('cnn64', 'cnn128'):
        m = RolloutNet(int(name[3:]), 'meanmax', obs='v2timer', row_pool=True, packet_head='identity').eval()
        return m, rollout_graph(m, batch)
    if name == 'res6': return residual_example(batch=batch)
    if name == 'hybrid': return token_example(hybrid=True, batch=batch)
    if name == 'hard_moe5': return token_example(layers=1, tokens=64, moe=True, batch=batch)
    _, layers, tokens = name.split('_')
    return token_example(int(layers), int(tokens), batch=batch)


EXAMPLES = ('cnn64', 'cnn128', 'res6', 'token_2_64', 'token_2_128', 'token_4_64', 'token_4_128', 'hybrid', 'hard_moe5')


def random_feeds(graph):
    feeds = {}
    for name, desc in graph.tensors.items():
        if desc['role'] != 1: continue
        shape = desc['shape']
        x = torch.randn(shape)*.1
        if name == 'valid':
            x = torch.ones(shape)
            x[:, -3:] = 0
        elif name == 'token_meta':
            x[..., 0] = torch.randint(12, shape[:-1])
            x[..., 1] = torch.randint(-1, 6, shape[:-1])
            x[..., 2] = torch.randint(-1, 10, shape[:-1])
        elif name == 'coords':
            x[..., 0] = torch.randint(-1, 6, shape[:-1])
            x[..., 1] = torch.randint(-1, 10, shape[:-1])
        elif name == 'selected_packet': x = torch.zeros(shape)
        elif name.endswith('_terrain'): x = torch.randint(5, shape).float()
        elif name == 'packet_info':
            x = torch.rand(shape)
            x[..., 0] = torch.randint(54, shape[:-1])
            x[:, -2:, 0] = -1
        feeds[name] = x.float()
    if graph.metadata and graph.metadata[2] and 'scalars' in feeds and 'packet_info' in feeds:
        scalars, info = feeds['scalars'], feeds['packet_info']
        scalars[:, 5:113] = 0
        kinds = info[..., 0].long()
        valid = kinds >= 0
        for start, values in ((5, info[..., 1]*info[..., 3]), (59, info[..., 2])):
            scalars[:, start:start+54].scatter_reduce_(1, kinds.clamp(min=0),
                values.masked_fill(~valid, 0), reduce='amax', include_self=True)
    return feeds


V3_EXAMPLES = ('v3_dense128_2', 'v3_dense192_4', 'v3_terrain5', 'v3_top2', 'v3_hybrid128_2')


def v3_example(name, capacity=128):
    """Actual task163 networks over task157-shaped synthetic public token arrays."""
    import numpy as np
    from pvz_student_v3 import StudentV3
    from pvz_student_tokens import frame_tokens, collate_tokens
    from pvz_student_v3_export import student_graph, graph_feeds
    width=192 if name=='v3_dense192_4' else 128
    layers=4 if width==192 else 2
    mode='terrain' if name=='v3_terrain5' else 'moe' if name=='v3_top2' else 'dense'
    hybrid=name=='v3_hybrid128_2'
    net=StudentV3(width,dict(layers=layers,heads=6 if width==192 else 4,
        ff_width=256 if mode!='dense' else width*4,ffn=mode,hybrid=hybrid),token_input='obs_v3').eval()
    raw=np.zeros((96,38),np.float32); raw[:,3:6]=-1
    raw[1,0]=1; raw[2:56,0]=2
    raw[2:56,3]=np.arange(54)//9; raw[2:56,4]=np.arange(54)%9
    raw[56:66,0]=8; raw[56:63,1]=[1,2,3,4,5,6,17]
    rng=np.random.default_rng(161)
    raw[66:,0]=rng.choice([3,4,5,6,7,9,10],30)
    raw[66:,1]=rng.integers(1,20,30)
    raw[66:,3]=rng.integers(-1,6,30); raw[66:,4]=rng.integers(-1,9,30)
    raw[66,5]=67
    raw[:,6:]=rng.normal(0,.1,(96,32))
    frame=dict(scene=dict(pool=True),v3=dict(version=3,tokens=raw.tolist()))
    tokens=collate_tokens([frame_tokens(frame,token_input='obs_v3')],capacity=capacity)
    board=torch.randn(1,92,6,10)*.1
    selected=torch.tensor([0])
    module,graph=student_graph(net,capacity=capacity)
    return module,graph,graph_feeds(tokens,selected,board),net,tokens,board,selected
