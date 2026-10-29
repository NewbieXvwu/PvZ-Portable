#!/usr/bin/env python3
"""Same-input ABBA token benchmark, including weight-only precision experiments."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import torch
from pvz_generic_examples import EXAMPLES,V3_EXAMPLES,example,random_feeds,v3_example
from pvz_generic_forward import reference,run_driver
from pvz_student_v3_export import student_graph,graph_feeds
from pvz_student_tokens import collate_tokens


def matrix_flops(g):
    """2 per multiply-add. Excludes exp/norm/lookup/moves; not an instruction count."""
    total=0
    for n in g.nodes:
        shape=lambda name:g.tensors[name]['shape']
        x=shape(n['inputs'][0]); y=shape(n['name']); op=n['op']
        at=lambda i:shape(n['inputs'][i])
        if op=='linear': total+=2*math.prod(x[:-1])*at(1)[0]*at(1)[1]
        elif op in ('conv','conv_relu'): total+=2*math.prod(y)*math.prod(at(1)[1:])
        elif op=='attention': total+=4*x[0]*x[1]*x[1]*x[2]
        elif op=='pointer': total+=2*math.prod(at(1))
        elif op in ('ffn','moe'):
            offset=1 if op=='ffn' else 2
            replicas=1 if op=='ffn' else (1 if n['ints'][1] else n['ints'][3])+n['ints'][4]
            total+=2*math.prod(x[:-1])*(math.prod(at(offset))+math.prod(at(offset+2)))*replicas
        elif op=='row_context': total+=4*x[0]*x[2]*x[1]*x[1]
    return total


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,default=ROOT/'build-cppnet/generic-forward-before-token')
    p.add_argument('--after',type=Path,default=ROOT/'build-cppnet/generic-forward')
    p.add_argument('--iterations',type=int,default=80)
    p.add_argument('--example',action='append',choices=(*EXAMPLES,*V3_EXAMPLES))
    p.add_argument('--quantized',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    report=dict(seed=161,threads=1,batch=1,iterations=args.iterations,
        cpu='i7-12700F; unpinned, production services running, nice10',
        peak=dict(gflops=156.8,basis='P-core turbo upper bound: 4.9 GHz * 2 FMA/cycle * 8 lanes * 2 FLOPs; E-core turbo upper bound 57.6 GF/s; no frequency residency claim'),
        flop_definition='matrix multiply-adds only / full graph wall time, including embedding/normalization/softmax/heads; excludes load and features',records=[])
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    with tempfile.TemporaryDirectory(prefix='pvz-token-bench-') as folder:
        for name in args.example or (*EXAMPLES,*V3_EXAMPLES):
            torch.manual_seed(161)
            is_v3=name in V3_EXAMPLES
            if is_v3:
                module,g,feeds,net,tokens,board,selected=v3_example(name)
                # The same 96 real tokens; padded baseline capacity128, compact execution96.
                compact_tokens={k:(v[:,:96].contiguous() if k in ('ids','mask','features','target_indices') else v) for k,v in tokens.items()}
                compact=graph_feeds(compact_tokens,selected,board)
                _,compact_g=student_graph(net,capacity=96)
                actual_count=96
            else:
                module,g=example(name);feeds=random_feeds(g);compact=feeds;compact_g=g;actual_count=None
            expected=reference(module,compact_g,compact,intermediates=True)
            before_expected=reference(module,g,feeds,intermediates=True)
            measurements={k:[] for k in ('before','after','blas')}
            errors={}
            order=['before','after','blas','blas','after','before']
            if args.quantized and is_v3:
                order+=['after','bf16','int8','int8','bf16','after']
                measurements.update(bf16=[],int8=[])
            for mode in order:
                os.environ['PVZ_GENERIC_LINEAR']='blas' if mode=='blas' else 'packed'
                os.environ['PVZ_GENERIC_PRECISION']=mode if mode in ('int8','bf16') else 'fp32'
                before=mode=='before'
                outputs,timing=run_driver(module,g,feeds if before else compact,folder,
                    args.before if before else args.after,args.iterations,
                    token_count=actual_count if not before else None)
                ref=before_expected if before else expected
                error=max((outputs[k]-ref[k]).abs().max().item() for k in outputs)
                head_error=max((outputs[k]-ref[k]).abs().max().item() for k in ('plant','packet','cell'))
                if mode not in ('int8','bf16') and error>=1e-5:
                    raise AssertionError(f'{name}/{mode} max error {error}')
                measurements[mode].append(dict(us=timing['us'],cpu_us=timing['cpu_us']))
                errors[mode]=dict(all_nodes=error,heads=head_error)
            flops=matrix_flops(compact_g)
            means={m:sum(t['us'] for t in v)/len(v) for m,v in measurements.items()}
            record=dict(example=name,actual_tokens=actual_count,capacity=128 if is_v3 else None,
                order=order,samples=measurements,mean_us=means,max_abs_error=errors,
                matrix_flops=flops,gflops=flops/means['after']/1000,
                percent_p_core_turbo_peak=flops/means['after']/1000/156.8*100,
                speedup=means['before']/means['after'])
            report['records'].append(record);save()
            print(json.dumps(record),flush=True)


if __name__=='__main__': main()
