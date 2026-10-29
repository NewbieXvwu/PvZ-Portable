#!/usr/bin/env python3
"""Bounded task161 parity/timing sweep (standalone executable, no simulator runs)."""
import argparse
import json
import os
import platform
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import torch
from pvz_generic_examples import EXAMPLES, example, random_feeds
from pvz_generic_forward import reference, run_driver


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--example', choices=EXAMPLES, action='append')
    p.add_argument('--driver', type=Path, default=ROOT/'build-cppnet/generic-forward')
    p.add_argument('--sparse-cnn', action='store_true', help='also measure 8 active input channels on CNN examples')
    args = p.parse_args()
    if args.iterations < 1: p.error('--iterations must be positive')
    torch.set_num_threads(1)
    records = []
    with tempfile.TemporaryDirectory(prefix='pvz-generic-bench-') as folder:
        cases = [(n, False) for n in args.example or EXAMPLES]
        if args.sparse_cnn: cases += [(n, True) for n in ('cnn64', 'cnn128', 'res6')]
        for name, sparse in cases:
            torch.manual_seed(161)
            module, graph = example(name)
            feeds = random_feeds(graph)
            if sparse: feeds['board'][:, 8:] = 0
            expected = reference(module, graph, feeds, intermediates=True)
            actual, timing = run_driver(module, graph, feeds, folder, args.driver, args.iterations)
            maximum = 0.
            for key, value in actual.items():
                maximum = max(maximum, (value-expected[key]).abs().max().item())
                torch.testing.assert_close(value, expected[key], rtol=0, atol=1e-5,
                                           msg=lambda msg: name+'/'+key+': '+msg)
            if name in ('cnn64', 'cnn128'):
                with torch.inference_mode():
                    heads = module(feeds['board'], feeds['scalars'], feeds['selected_packet'].long(),
                                   packet_info=feeds['packet_info'])
                for key in ('plant', 'packet', 'cell'):
                    torch.testing.assert_close(actual[key], heads[key].reshape_as(actual[key]), rtol=0, atol=1e-5)
            record = dict(example=name, input='8_active_channels' if sparse else 'dense_random',
                          **timing, max_abs_error=maximum,
                          nodes=len(graph.nodes), parameters=sum(t.numel() for t in module.parameters()),
                          ratio_to_historical_61us=timing['us']/61)
            records.append(record)
            print(json.dumps(record), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(dict(seed=161, batch=1, threads=1,
                environment=dict(machine=platform.machine(), torch=torch.__version__,
                                 timing='C++ warm graph; mean wall and process CPU microseconds; excludes load/I/O/features'),
                records=records), indent=2)+'\n')


if __name__ == '__main__': main()
