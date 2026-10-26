"""Task96: what does the deterministic switch cost on the op that diverges?

The epoch-level A/B was dominated by cache warm-up and host load (default 13.5 vs
20.9 s for the same configuration). This times only the embedding backward at the
shape packet_logits uses, alternating default and deterministic blocks so drift
cancels, and reports per-iteration medians. GPU use is a few seconds of small
tensors; no training process is started.
"""
import argparse
import json
import statistics
import time
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/task96/embedding_probe'
ITERATIONS = 200
BLOCKS = 6


def gpu_state():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    used, free, util = [int(x.strip()) for x in result.stdout.splitlines()[0].split(',')]
    return dict(used_mib=used, free_mib=free, utilization=util)


def main():
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    ids = torch.randint(0, 8, (2048, 10), device='cuda')
    upstream = torch.randn(2048, 10, 64, device='cuda')
    embedding = torch.nn.Embedding(41, 64, device='cuda')
    report = dict(task=96, iterations=ITERATIONS, blocks=BLOCKS, gpu_before=gpu_state(),
                  environment=dict(torch=torch.__version__, cuda=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name(0)), blocks_measured=[])
    for block in range(BLOCKS):
        for label, deterministic in (('default', False), ('deterministic', True)):
            torch.use_deterministic_algorithms(deterministic)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(ITERATIONS):
                embedding.zero_grad(set_to_none=True)
                embedding(ids).backward(upstream)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            report['blocks_measured'].append(dict(block=block, mode=label,
                                                  seconds=elapsed,
                                                  per_iteration_ms=1000*elapsed/ITERATIONS))
    default = [entry['per_iteration_ms'] for entry in report['blocks_measured'] if entry['mode'] == 'default']
    strict = [entry['per_iteration_ms'] for entry in report['blocks_measured'] if entry['mode'] == 'deterministic']
    report['summary'] = dict(default_median_ms=statistics.median(default),
                             deterministic_median_ms=statistics.median(strict),
                             ratio=statistics.median(strict)/statistics.median(default),
                             default_range=[min(default), max(default)],
                             deterministic_range=[min(strict), max(strict)])
    torch.use_deterministic_algorithms(False)
    report['gpu_after'] = gpu_state()
    (OUT/'cost.json').write_text(json.dumps(report, ensure_ascii=False, indent=1)+'\n')
    print(json.dumps(report['summary'], indent=1))


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
