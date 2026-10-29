"""obs-v3 到 task161/163 图输入的数组适配，不组装实体或推断游戏语义。"""
import numpy as np

OBS_VERSION = 3
TOKEN_FIELDS = 38
FEATURE_COUNT = 32
CELL_START, CARD_START, PREFIX_COUNT = 2, 56, 66


def graph_inputs(frame, capacity=None):
    """一个决策帧 → 连续 float32 的 batch=1 张量，保持原 token 索引。"""
    observation = frame.get("v3")
    if not observation or observation.get("version") != OBS_VERSION:
        raise ValueError("requires C++ obs-v3; backfill old archives first")
    raw = np.asarray(observation["tokens"], dtype=np.float32)
    count = len(raw)
    capacity = count if capacity is None else capacity
    if capacity < count:
        raise ValueError(f"obs-v3 has {count} tokens, graph capacity is {capacity}; truncation is unsupported")
    tokens = np.zeros((1, capacity, TOKEN_FIELDS), np.float32)
    tokens[:, :, 3:6] = -1
    tokens[0, :count] = raw
    valid = np.zeros((1, capacity), np.float32)
    valid[0, :count] = ~((raw[:, 0] == 8) & (raw[:, 1] == 0))
    return dict(tokens=tokens,
                token_meta=np.ascontiguousarray(tokens[:, :, [0, 3, 4]]),
                coords=np.ascontiguousarray(tokens[:, :, 3:5]), valid=valid)
