"""Task58 stdlib-only queue; importing torch is reserved for the bounded child."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/task58'
CONFIGS = ('current', 'stack4', 'stack8', 'convgru', 'spatial_ssm')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mem_available():
    return next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))


def run_all(epochs):
    for seed in (1,2,3):
        for kind in CONFIGS:
            for arm in ('human','xray'):
                folder = OUT / 'models' / f'{kind}-{arm}-s{seed}'
                if (folder/'result.json').exists():
                    result = json.loads((folder/'result.json').read_text())
                    assert result['corpus_sha256'] == digest(OUT/'corpus.json')
                    assert result['epochs'] == epochs
                    assert result['checkpoint_sha256'] == digest(folder/'model.pt')
                    continue
                # Child checks again after imports and waits before initializing CUDA.
                while mem_available() < 10 * 1024**3:
                    print(json.dumps(dict(wait='MemAvailable',bytes=mem_available())),flush=True)
                    time.sleep(30)
                folder.mkdir(parents=True,exist_ok=True)
                command = [sys.executable,str(ROOT/'scripts/memory_probe_train.py'),'train-one',
                           '--kind',kind,'--arm',arm,'--seed',str(seed),'--epochs',str(epochs)]
                print('START ' + ' '.join(command),flush=True)
                with (folder/'train.log').open('w') as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,
                        timeout=3600,env={**os.environ,'PVZ_NO_POPUP':'1',
                                         'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
                print('DONE ' + folder.name,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs',type=int,default=20)
    run_all(parser.parse_args().epochs)
