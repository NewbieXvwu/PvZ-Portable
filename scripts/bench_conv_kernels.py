#!/usr/bin/env python3
"""Isolated task35 SSE/BLAS comparison on the fixed manifest's real observations.

Uses no simulator. Generates ignored fixtures/binaries in artifacts/perf. This
kernel experiment does not implement batched rollouts or change the game path.
"""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT / 'scripts'))
import torch
from bench_engine import STUDENT, _archive_rows
from pvz_rollout_policy import load_student
from pvz_student_observation import StudentHistory


def main():
    torch.set_num_threads(1)
    net = load_student(str(STUDENT))
    if net.obs != 'v2' or net.context:
        raise ValueError('this frozen experiment expects dw4i_s1 v2 without task context')
    out = ROOT / 'artifacts/perf'
    out.mkdir(parents=True, exist_ok=True)
    fixture = out / 'conv_kernel_inputs.bin'
    manifest = json.loads((out / 'engine_roots.json').read_text())
    roots = sum(len(e['decisions']) for group in manifest['families'].values() for e in group)
    with fixture.open('wb') as stream, torch.inference_mode():
        stream.write(b'PVZKERN1' + struct.pack('<I', roots * 3))
        for entries in manifest['families'].values():
            for entry in entries:
                archive = Path(entry['archive'])
                frames = _archive_rows(archive / ('frames.npz' if (archive / 'frames.npz').exists() else 'frames.jsonl'))
                history = StudentHistory(net.obs)
                for index, frame in enumerate(frames[:max(entry['decisions']) + 1]):
                    board, _ = history.features(frame, frames[index - 1] if index else None)
                    if index not in entry['decisions']:
                        continue
                    if 'v2' not in frame:
                        raise ValueError(f'missing v2 observation: {archive} frame {index}')
                    hidden = board[None]
                    for layer in range(3):
                        conv = net.trunk[layer * 2]
                        channels, width = conv.in_channels, conv.out_channels
                        kr, kc = conv.kernel_size
                        stream.write(struct.pack('<5i', channels, width, kr, kc, int(layer == 0)))
                        for tensor in (hidden, conv.weight, conv.bias):
                            stream.write(tensor.detach().cpu().contiguous().numpy().astype('<f4', copy=False).tobytes())
                        hidden = torch.relu(conv(hidden))
    binary = out / 'bench-conv-kernels'
    subprocess.run(['c++', '-O3', '-std=c++20', '-ffp-contract=off',
                    str(ROOT / 'scripts/bench_conv_kernels.cpp'), '-lopenblas', '-o', str(binary)], check=True)
    result = subprocess.run([str(binary), str(fixture)], check=True, text=True, capture_output=True,
                            env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})
    parsed = json.loads(result.stdout)
    (out / 'conv_kernel_compare.json').write_text(json.dumps(parsed, indent=1) + '\n')
    print(json.dumps(parsed, indent=1), flush=True)


if __name__ == '__main__':
    main()
