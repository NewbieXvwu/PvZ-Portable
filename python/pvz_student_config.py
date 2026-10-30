"""Lightweight student network recipe contract (no torch import)."""

NETWORK_DEFAULTS = dict(architecture='cnn', layers=4, heads=6, ff_width=768,
                        ffn='dense', experts=5, top_k=2, shared_experts=1, hybrid=False)


def network_config(config=None, width=64):
    if config is None:
        config = {}
    if not isinstance(config, dict) or config.keys() - NETWORK_DEFAULTS.keys():
        raise ValueError('network: unknown fields or non-object')
    c = {**NETWORK_DEFAULTS, **config}
    if c['architecture'] not in ('cnn', 'token') or c['ffn'] not in ('dense', 'terrain', 'moe'):
        raise ValueError('network: invalid architecture/ffn')
    for key in ('layers', 'heads', 'ff_width', 'experts', 'top_k', 'shared_experts'):
        if type(c[key]) is not int or c[key] < (0 if key == 'shared_experts' else 1):
            raise ValueError(f'network.{key}: invalid integer')
    if type(c['hybrid']) is not bool or c['top_k'] > c['experts']:
        raise ValueError('network: invalid hybrid/top_k')
    if c['ffn'] == 'terrain' and c['experts'] != 5:
        raise ValueError('terrain FFN requires five background experts')
    if c['architecture'] == 'token' and width % c['heads']:
        raise ValueError('token width must be divisible by heads')
    if c['architecture'] == 'cnn' and c != NETWORK_DEFAULTS:
        raise ValueError('network options require architecture=token')
    return c
