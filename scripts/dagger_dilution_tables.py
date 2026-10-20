"""Render factual task15 tables only after complete, identity-checked inference."""
import argparse
import hashlib
import json
from pathlib import Path


def render(stats, execution):
    assert execution['complete'] is True
    assert stats['models'] == 14 and stats['total_games'] == 86016
    assert stats['environment_seed_clusters'] == 1024 and stats['training_seeds'] == [1, 2]
    assert stats['primary_family'] == ['B-F', 'M-F', 'T-F', 'I-R', 'I-F', 'R-F', 'S-F']
    assert {v['name'] for v in execution['models']} == {a + '_s' + str(s) for a in 'BFMTIRS' for s in (1, 2)}
    rows = ['| 臂 | 训练种子 | z15 | z16 | z17 | z18 | z19 | z20 | 六档均值/1024 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in 'BFMTIRS':
        values = stats['arms'][arm]['wins_by_training_seed_and_tier']
        assert len(values) == 2 and all(len(v) == 6 for v in values)
        for seed, counts in enumerate(values, 1):
            assert all(isinstance(v, int) and 0 <= v <= 1024 for v in counts)
            rows.append('| ' + ' | '.join([arm, str(seed), *map(str, counts), f'{sum(counts)/6:.2f}']) + ' |')
    rows += ['', '| 对比 | seed1差/pp | seed2差/pp | 平均差/pp | 整簇bootstrap 95% CI/pp | Holm p | 预登记判定 |',
             '|---|---:|---:|---:|---|---:|---|']
    for name in stats['primary_family']:
        value = stats['contrasts'][name]
        delta = value['per_training_seed_difference']
        ci = value['env_seed_cluster_ci95']
        assert len(delta) == 2 and len(ci) == 2 and value['df'] == 1023
        decision = ('支持正效应' if value['supports_positive_effect'] else
                    '支持负效应' if value['supports_negative_effect'] else '未通过')
        rows.append('| ' + ' | '.join([name, *[f'{v*100:+.2f}' for v in delta],
            f'{value["difference"]*100:+.2f}', f'[{ci[0]*100:+.2f}, {ci[1]*100:+.2f}]',
            f'{value["holm_p"]:.6g}', decision]) + ' |')
    rows += ['', '判定同时要求两个训练种子方向一致、整簇区间排除0及Holm7通过；“未通过”不表示等价。',
             '六档和两个初始化按共同环境种子联合聚类，推断仅对这两个已训练初始化成立；不外推训练种子总体。',
             '逐档区间、t区间及未校正p保留在完整统计JSON，机制解释与训练集推荐另行结合全部证据填写。', '']
    return '\n'.join(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stats', type=Path, required=True)
    parser.add_argument('--execution', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    stats = json.loads(args.stats.read_text())
    execution = json.loads(args.execution.read_text())
    assert execution['evaluation']['results_sha256'] == hashlib.sha256(args.stats.read_bytes()).hexdigest()
    args.out.write_text(render(stats, execution))
