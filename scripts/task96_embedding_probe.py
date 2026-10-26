"""Task96 embedding-gradient micro probe.

The 1.05M-row control localised the first divergence to packet_embedding.weight:
step 1 forward and loss are bitwise identical, only that gradient differs. This
micro probe asks the same question in isolation, in seconds of GPU time, so the
mechanism can be stated at op level rather than inferred from a full epoch:

  * does nn.Embedding backward vary run to run when a batch repeats ids;
  * does torch.use_deterministic_algorithms change or reject it;
  * same for a bare index_add_ of duplicated rows.

No training is started and no production process is touched; the probe allocates
a few MB and runs well under a second per repeat.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/task96/embedding_probe'
REPEATS = 5


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().to('cpu').contiguous().numpy().tobytes()).hexdigest()


def gpu_state():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    used, free, util = [int(x.strip()) for x in result.stdout.splitlines()[0].split(',')]
    return dict(used_mib=used, free_mib=free, utilization=util)


def main():
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    ids = torch.arange(2048, device='cuda') % 8          # 8 distinct ids in 2048 rows
    ids = ids[torch.randperm(len(ids), device='cuda')]
    upstream = torch.randn(2048, 64, device='cuda')
    report = dict(task=96, rows=len(ids), distinct_ids=int(ids.unique().numel()),
                  table_rows=41, embedding_dim=64, repeats=REPEATS, gpu_before=gpu_state(),
                  environment=dict(torch=torch.__version__, cuda=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name(0)))
    for label, deterministic in (('default', False), ('deterministic', True)):
        torch.use_deterministic_algorithms(deterministic)
        entry = dict(deterministic=deterministic, embedding=[], index_add=[], errors=[])
        for _ in range(REPEATS):
            embedding = torch.nn.Embedding(41, 64, device='cuda')
            with torch.no_grad():
                embedding.weight.copy_(torch.randn(41, 64, device='cuda'))
            try:
                embedding.zero_grad(set_to_none=True)
                embedding(ids).backward(upstream)
                entry['embedding'].append(tensor_sha(embedding.weight.grad))
            except Exception as error:
                entry['errors'].append('embedding: '+repr(error))
        for _ in range(REPEATS):
            try:
                table = torch.zeros(41, 64, device='cuda')
                table.index_add_(0, ids, upstream)
                entry['index_add'].append(tensor_sha(table))
            except Exception as error:
                entry['errors'].append('index_add: '+repr(error))
        entry['embedding_distinct'] = len(set(entry['embedding']))
        entry['index_add_distinct'] = len(set(entry['index_add']))
        report[label] = entry
    torch.use_deterministic_algorithms(False)
    report['gpu_after'] = gpu_state()
    report['finished'] = time.time()
    (OUT/'probe.json').write_text(json.dumps(report, ensure_ascii=False, indent=1)+'\n')
    print(json.dumps({key: (value if key in ('default', 'deterministic') else value)
                      for key, value in report.items() if key != 'environment'}, indent=1))


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
