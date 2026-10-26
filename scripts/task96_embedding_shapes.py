"""Task96 embedding-gradient micro probe, at the shapes the student actually uses.

packet_logits looks up [batch, 10] ids and cell_logits looks up [batch] ids whose
gradient arrives through a 6x54 expansion, both into a 41 x 64 table with many
repeated ids. This repeats each shape several times with and without
deterministic algorithms and reports how many distinct gradient hashes appear.
No training, no production process, a few MB of GPU.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/task96/embedding_probe'
REPEATS = 10
TABLE = 41
DIM = 64
BATCH = 2048
SHAPES = [('flat', (BATCH,), (BATCH, DIM)),
          ('packet_head', (BATCH, 10), (BATCH, 10, DIM)),
          ('cell_expand', (BATCH,), (BATCH, DIM, 6, 54))]


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
    base_ids = torch.randint(0, 8, (BATCH,), device='cuda')      # 8 distinct types, like the real batches
    report = dict(task=96, table_rows=TABLE, dim=DIM, batch=BATCH, repeats=REPEATS,
                  distinct_types=8, gpu_before=gpu_state(),
                  environment=dict(torch=torch.__version__, cuda=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name(0)))
    for label, deterministic in (('default', False), ('deterministic', True)):
        torch.use_deterministic_algorithms(deterministic)
        section = {}
        for name, ids_shape, grad_shape in SHAPES:
            ids = base_ids.view(-1, 1).expand(*ids_shape) if len(ids_shape) > 1 else base_ids
            ids = ids.contiguous()
            upstream = torch.randn(grad_shape, device='cuda')
            hashes, errors = [], []
            for _ in range(REPEATS):
                embedding = torch.nn.Embedding(TABLE, DIM, device='cuda')
                with torch.no_grad():
                    embedding.weight.copy_(torch.randn(TABLE, DIM, device='cuda'))
                try:
                    embedding.zero_grad(set_to_none=True)
                    out = embedding(ids)
                    if len(grad_shape) == out.dim() + 2:   # the cell head expands the lookup over 6x54 cells
                        out = out[:, :, None, None].expand(*grad_shape)
                    out.backward(upstream)
                    hashes.append(tensor_sha(embedding.weight.grad))
                except Exception as error:
                    errors.append(repr(error))
            section[name] = dict(ids=list(ids_shape), grad=list(grad_shape),
                                 distinct=len(set(hashes)), hashes=hashes[:3], errors=errors)
        report[label] = section
    torch.use_deterministic_algorithms(False)
    report['gpu_after'] = gpu_state()
    report['finished'] = time.time()
    (OUT/'shapes.json').write_text(json.dumps(report, ensure_ascii=False, indent=1)+'\n')
    print(json.dumps({key: {name: {k: v for k, v in value.items() if k != 'hashes'}
                            for name, value in section.items()}
                      for key, section in report.items() if key in ('default', 'deterministic')},
                     indent=1))


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
