"""Full StudentV3 -> task161 PVZGN001, including embeddings and conditional heads.

Graphs have a maximum capacity; native execution uses actual token count without padding.
The capacity is allocated at load and tokens are never truncated. Metadata observation id 10
means task157 public rich tokens, not the unrelated legacy observation id 3 (v2).
"""
import torch
from torch import nn
from pvz_generic_forward import GraphSpec, Builder, export_graph
from pvz_value_net import BOARD_CHANNELS


def graph_feeds(tokens, selected_packet, board=None):
    ids, valid = tokens['ids'], tokens['mask']
    raw = torch.cat([ids.float(), tokens['target_indices'][:, :, None].float(), tokens['features']], -1)
    spatial = valid & (ids[:, :, 0]>=3) & (ids[:, :, 0]<=7) & (ids[:, :, 3]>=0) & (ids[:, :, 3]<6) & (ids[:, :, 4]>=0) & (ids[:, :, 4]<9)
    coords = ids[:, :, 3:5].float().masked_fill(~spatial[:, :, None], -1)
    feeds = dict(tokens=raw, token_meta=ids[:, :, [0, 3, 4]].float(), valid=valid.float(),
                 pool_valid=(valid & (ids[:, :, 0]!=8)).float(), coords=coords,
                 packet_index=tokens['packet_index'].float(), cell_index=tokens['cell_index'].float(),
                 terrain=tokens['terrain'].float(), selected_packet=selected_packet.float())
    if board is not None:
        feeds['board'] = board
    return {k: v.detach().cpu().contiguous() for k, v in feeds.items()}


def student_graph(net, capacity=256, batch=1):
    if capacity < 66 or batch < 1:
        raise ValueError('v3 graph requires capacity >= 66 and batch >= 1')
    module = nn.Module()
    module.add_module('source', net)
    g = GraphSpec([10, int(net.context), 1])
    b = Builder(module, g)
    W, T, B = net.width, capacity, batch
    g.tensor('tokens', [B, T, 38])
    g.tensor('token_meta', [B, T, 3])
    g.tensor('valid', [B, T])
    g.tensor('pool_valid', [B, T])
    g.tensor('packet_index', [B, 10])
    g.tensor('cell_index', [B, 54])
    g.tensor('selected_packet', [B])

    def weight(path): return g.weight(module, 'source.'+path)
    def packed(name, value):
        module.register_buffer(name, value.detach().contiguous())
        return g.weight(module, name)
    def add(name, left, right, shape, stage=0):
        return g.node(name, 'add', [left, right], shape, stage=stage)
    def norm(name, x, path):
        return g.node(name, 'norm', [x, weight(path+'.weight'), weight(path+'.bias')], [B,T,W], [0], [net.get_submodule(path).eps])
    def ffn_weights(name, ff):
        return [packed(name+'_up_w', torch.cat([ff.gate.weight, ff.value.weight], 0)),
                packed(name+'_up_b', torch.cat([ff.gate.bias, ff.value.bias], 0)),
                packed(name+'_down_w', ff.down.weight), packed(name+'_down_b', ff.down.bias)]

    embedded = []
    for col, name in enumerate(('kind', 'category', 'variant', 'row', 'col')):
        v = g.node(name+'_id3', 'slice', ['tokens'], [B,T,1], [2,col,1])
        v = g.node(name+'_id', 'reshape', [v], [B,T])
        embedded.append(g.node(name+'_emb', 'embedding', [v, weight(name+'_embedding.weight')], [B,T,W], [1] if col>=3 else []))
    x = embedded[0]
    for i, emb in enumerate(embedded[1:]): x = add('embedding_sum'+str(i), x, emb, [B,T,W])
    f = g.node('features', 'slice', ['tokens'], [B,T,32], [2,6,32])
    f = b.layer('features_up', f, 'source.feature_projection.0')
    f = g.node('features_silu', 'silu', [f], [B,T,W])
    f = b.layer('features_down', f, 'source.feature_projection.2')
    x = add('input_embedded', x, f, [B,T,W])
    targets = g.node('targets3', 'slice', ['tokens'], [B,T,1], [2,5,1])
    targets = g.node('targets', 'reshape', [targets], [B,T])
    if net.network['ffn'] == 'terrain':
        g.tensor('terrain', [B])
        terrain = g.node('terrain_per_token', 'broadcast', ['terrain'], [B,T], [0,-1])
    for i, layer in enumerate(net.encoder):
        pre = f'layer{i}'
        path = f'encoder.{i}'
        z = norm(pre+'_ln1', x, path+'.attention_norm')
        qkv = b.layer(pre+'_qkv', z, 'source.'+path+'.attention.qkv')
        q, k, v = [g.node(pre+'_'+name, 'slice', [qkv], [B,T,W], [2,j*W,W]) for j, name in enumerate(('q','k','v'))]
        att = layer.attention
        relation = g.node(pre+'_relation', 'relation', ['token_meta', weight(path+'.attention.kind_pair_bias'),
            packed(pre+'_rows', att.row_bias.weight.T), packed(pre+'_cols', att.col_bias.weight.T),
            packed(pre+'_same', att.same_cell_bias.weight.T), targets, weight(path+'.attention.target_bias')],
            [B,net.network['heads'],T,T], [5,8])
        a = g.node(pre+'_attention', 'attention', [q,k,v,relation,'valid'], [B,T,W], [net.network['heads']])
        a = b.layer(pre+'_attention_out', a, 'source.'+path+'.attention.projection')
        x = add(pre+'_res1', x, a, [B,T,W])
        z = norm(pre+'_ln2', x, path+'.ff_norm')
        if net.network['ffn'] == 'dense':
            z = g.node(pre+'_ffn', 'ffn', [z, *ffn_weights(pre+'_ff', layer.ff)], [B,T,W], [1])
        else:
            ff = layer.ff
            route = terrain if ff.mode == 'terrain' else b.layer(pre+'_route', z, 'source.'+path+'.ff.router')
            weights = []
            for j, expert in enumerate([*ff.experts, *ff.shared]): weights += ffn_weights(pre+f'_e{j}', expert)
            z = g.node(pre+'_moe', 'moe', [z, route, *weights], [B,T,W],
                       [1,int(ff.mode=='terrain'),len(ff.experts),ff.top_k,len(ff.shared)])
        x = add(pre+'_res2', x, z, [B,T,W])
    x = norm('encoded', x, 'encoder_norm')
    state = g.node('state', 'masked_pool', [x,'pool_valid'], [B,W])
    packets = g.node('packets', 'gather', [x,'packet_index'], [B,10,W])
    cells = g.node('cells', 'gather', [x,'cell_index'], [B,54,W])
    if net.cnn is not None:
        g.tensor('board', [B, BOARD_CHANNELS, 6, 10])
        g.tensor('coords', [B,T,2])
        h = b.relu('cnn_relu0', b.layer('cnn0','board','source.cnn.0'))
        h = b.relu('cnn_relu2', b.layer('cnn2',h,'source.cnn.2'))
        h = g.node('grid', 'scatter', [h,x,'coords'], [B,W,6,10])
        state = add('hybrid_state', state, g.node('cnn_state','pool',[h],[B,W],[2,0]), [B,W])
        crop = g.node('grid_crop','slice',[h],[B,W,6,9],[3,0,9])
        crop = g.node('grid_channels_last','permute',[crop],[B,6,9,W],[0,2,3,1])
        crop = g.node('grid_cells','reshape',[crop],[B,54,W])
        cells = add('hybrid_cells', cells, crop, [B,54,W])
    b.layer('plant', state, 'source.plant', 1)
    pq = b.layer('packet_query', state, 'source.packet_query', 1)
    pk = b.layer('packet_key', packets, 'source.packet_key', 1)
    logits = g.node('packet_scores','pointer',[pq,pk],[B,10],floats=[W**-.5],stage=1)
    g.node('packet','mask',[logits,'packet_index'],[B,10],floats=[-1e9],stage=1)
    card = g.node('selected_card','select',[packets,'selected_packet'],[B,W],stage=2)
    query = g.node('cell_query_input','concat',[state,card],[B,W*2],[1],stage=2)
    query = b.layer('cell_query',query,'source.plant_cell_query',2)
    key = b.layer('cell_key',cells,'source.cell_key',2)
    g.node('cell','pointer',[query,key],[B,54],floats=[W**-.5],stage=2)
    return module, g


def export_student_v3(net, path, capacity=256, batch=1):
    if net.token_input != 'obs_v3':
        raise ValueError('native v3 requires token_input=obs_v3; compact prototype checkpoints use a different input contract')
    module, graph = student_graph(net, capacity, batch)
    export_graph(module, graph, path)
