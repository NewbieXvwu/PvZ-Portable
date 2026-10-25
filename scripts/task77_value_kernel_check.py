#!/usr/bin/env python3
"""Compare value logits as float32 bytes before/after task77 without a game process.

Exercises empty/dense/changing sparse channels, padding, reuse and reloading changed
weights. The compiler flags and actual production exports come from the task77 freeze.
"""
import hashlib
import json
from pathlib import Path
import shlex
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
from task77_search_hotpath import OUT, save

DRIVER = r'''
#include "EnvironmentValueNet.h"
#include <fstream>
#include <stdexcept>
int main(int argc, char** argv) {
    if (argc != 5) return 2;
    EnvironmentValueNet net;
    std::ofstream out(argv[4], std::ios::binary);
    for (int pass = 0; pass < 3; ++pass) {
        net.Load(argv[pass == 1 ? 2 : 1]);
        std::ifstream in(argv[3], std::ios::binary);
        std::vector<float> board(net.Channels * net.Cells), scalars(net.ScalarCount());
        while (in.read(reinterpret_cast<char*>(board.data()), board.size() * sizeof(float))) {
            if (!in.read(reinterpret_cast<char*>(scalars.data()), scalars.size() * sizeof(float))) return 3;
            const float logit = net.Logit(board, scalars);
            out.write(reinterpret_cast<const char*>(&logit), sizeof(logit));
        }
    }
    return out.good() ? 0 : 4;
}
'''


def compile_driver(folder, source, tag):
    units = json.loads((ROOT / 'build/compile_commands.json').read_text())
    unit = next(u for u in units if u['file'].endswith('/EnvironmentValueNet.cpp'))
    original = shlex.split(unit['command'])
    flags = []
    skip = False
    for word in original[1:]:
        if skip:
            skip = False
        elif word == '-o':
            skip = True
        elif word != '-c' and word != unit['file']:
            flags.append(word)
    executable = folder / tag
    cmd = [original[0], '-I' + str(source), *flags, '-ffunction-sections', '-fdata-sections',
           str(folder / 'driver.cpp'), str(source / 'EnvironmentValueNet.cpp'),
           '-Wl,--gc-sections', '-lopenblas', '-o', str(executable)]
    with (folder / f'compile_{tag}.log').open('w') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    return executable


def main():
    import numpy as np
    from export_value_net import export_value_net
    from pvz_value_net import load_value_net
    folder = OUT / 'value_kernel_check'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'driver.cpp').write_text(DRIVER)
    executables = {tag: compile_driver(folder, source, tag) for tag, source in (
        ('before', OUT / 'baseline/src'), ('after', ROOT / 'src'))}
    frozen = json.loads((OUT / 'freeze.json').read_text())
    evidence = {'models': {}, 'pass': True}
    rng = np.random.default_rng(77)
    for kind in ('value', 'static'):
        model = folder / f'{kind}.bin'
        export_value_net(load_value_net(frozen[kind]['path']), str(model))
        data = bytearray(model.read_bytes())
        width, encoder, context = struct.unpack_from('<III', data, 8)
        scalar_count = 113 + (6 if encoder == 2 else 0) + context
        # Perturb every first-layer channel; an unchanged active set after Load must
        # never reuse packed weights from the previous model.
        weights = np.frombuffer(data, dtype='<f4', count=width * 92 * 9, offset=20)
        weights += np.float32(.03125)
        changed = folder / f'{kind}_changed.bin'
        changed.write_bytes(data)
        samples = []
        for index in range(160):
            board = rng.normal(size=(92, 60)).astype('<f4')
            if index % 8 == 0:
                board.fill(0)
            elif index % 8 == 1:
                board.fill(-0.0)
            elif index % 8 != 2:
                active = rng.choice(92, index % 23 + 1, replace=False)
                mask = np.ones(92, dtype=bool)
                mask[active] = False
                board[mask] = 0
            scalars = rng.normal(size=scalar_count).astype('<f4')
            samples.append(board.tobytes() + scalars.tobytes())
        features = folder / f'{kind}_features.bin'
        features.write_bytes(b''.join(samples))
        outputs = {}
        for tag, exe in executables.items():
            output = folder / f'{kind}_{tag}.bin'
            subprocess.run([str(exe), str(model), str(changed), str(features), str(output)], check=True)
            outputs[tag] = output.read_bytes()
        ok = outputs['before'] == outputs['after'] and len(outputs['before']) == 160 * 3 * 4
        evidence['pass'] &= ok
        evidence['models'][kind] = {'pass': ok, 'logits': len(outputs['before']) // 4,
            'sha256': {k: hashlib.sha256(v).hexdigest() for k, v in outputs.items()}}
    save(folder / 'evidence.json', evidence)
    print(json.dumps(evidence, indent=1))
    if not evidence['pass']:
        raise RuntimeError('value logits differ bit for bit')


if __name__ == '__main__':
    main()
