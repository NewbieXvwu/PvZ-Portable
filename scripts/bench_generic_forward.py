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
from pvz_generic_examples import EXAMPLES, V3_EXAMPLES, example, random_feeds, v3_example
from pvz_generic_forward import reference, run_driver, export_graph


def native_comparison(args, folder):
    """Matched weights/board, interleaved old/new order; never uses a production executable."""
    from pvz_event_env import EventWaitEnv
    from pvz_env import TaskSpec
    from pvz_rollout_policy import export_student
    torch.manual_seed(161)
    module,graph=example('cnn64')
    folder=Path(folder)
    export_student(module,folder/'legacy.bin')
    export_graph(module,graph,folder/'generic.bin')
    with EventWaitEnv(args.resource_dir,executable=args.executable) as env:
        obs,_=env.reset(task=TaskSpec(level=9,seed=161006))
        # Build a fixed occupied fixture, independent of the random policy's predictions.
        while obs['tick']<6000:
            legal=obs['legal_actions']['plants']
            obs,*_=env.step({'type':'plant',**legal[0]} if legal else {'type':'wait','ticks':60})
        values={'legacy':[],'generic':[]}
        for name in ('legacy','generic','generic','legacy'):
            env.load_rollout_net(folder/(name+'.bin'))
            measured=env._command(f'ROLLOUT_BENCH {args.iterations}')
            if not measured.get('ok'): raise RuntimeError('isolated engine needs ROLLOUT_BENCH support')
            values[name].append(measured['us'])
        means={name:sum(samples)/len(samples) for name,samples in values.items()}
        return dict(level=9,seed=161006,tick=obs['tick'],iterations=args.iterations,
                    order=['legacy','generic','generic','legacy'],samples_us=values,mean_us=means,
                    generic_over_legacy=means['generic']/means['legacy'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--example', choices=(*EXAMPLES,*V3_EXAMPLES), action='append')
    p.add_argument('--suite', choices=('generic','v3','all'), default='generic')
    p.add_argument('--driver', type=Path, default=ROOT/'build-cppnet/generic-forward')
    p.add_argument('--sparse-cnn', action='store_true', help='also measure 8 active input channels on CNN examples')
    p.add_argument('--native', action='store_true', help='also compare old/new formats on the same live board')
    p.add_argument('--executable', type=Path, default=ROOT/'build-cppnet/pvz-portable')
    p.add_argument('--resource-dir', type=Path, default=Path.home()/'.cache/pvz-research-resources')
    args = p.parse_args()
    if args.iterations < 1: p.error('--iterations must be positive')
    torch.set_num_threads(1)
    records = []
    report=dict(seed=161,batch=1,threads=1,
        environment=dict(machine=platform.machine(),torch=torch.__version__,
            timing='C++ warm graph; mean wall and process CPU microseconds; excludes load/I/O/features'),records=records)
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    with tempfile.TemporaryDirectory(prefix='pvz-generic-bench-') as folder:
        names=args.example or (EXAMPLES if args.suite=='generic' else V3_EXAMPLES if args.suite=='v3' else (*EXAMPLES,*V3_EXAMPLES))
        cases = [(n, False) for n in names]
        if args.sparse_cnn: cases += [(n, True) for n in names if n in ('cnn64', 'cnn128', 'res6')]
        for name, sparse in cases:
            torch.manual_seed(161)
            is_v3=name in V3_EXAMPLES
            if is_v3: module,graph,feeds,net,tokens,board,selected=v3_example(name)
            else:
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
            if is_v3:
                with torch.inference_mode(): heads=net(board,None,selected,tokens=tokens)
                for key in ('plant','packet','cell'):
                    torch.testing.assert_close(actual[key],heads[key].reshape_as(actual[key]),rtol=0,atol=1e-5)
            record = dict(example=name, input='raw_obs_v3_96of128' if is_v3 else '8_active_channels' if sparse else 'dense_random',
                          **timing, max_abs_error=maximum,
                          nodes=len(graph.nodes), parameters=sum(t.numel() for t in module.parameters()),
                          ratio_to_historical_61us=timing['us']/61)
            records.append(record)
            print(json.dumps(record), flush=True)
            save()
        if args.native:
            report['native']=native_comparison(args,folder)
            print(json.dumps(dict(native=report['native'])),flush=True)
            save()


if __name__ == '__main__': main()
