"""Small numeric-only storage design benchmark; no replay, source copies or cold-cache audit.

Temporary block/episode files are removed before returning. Reads existing task157
rich archives, with one short consecutive window per selected terrain.
"""
import argparse
import json
from pathlib import Path
import random
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python')]
import numpy as np
from pvz_frame_store import read_frames
from pvz_student_tokens import frame_tokens, collate_tokens
from pvz_research_data import SparseTokenWriter, SparseTokens
from pvz_token_blocks import encode, decode


def measure(load, indices, batch=256):
    start = time.perf_counter()
    pending = []
    for index in indices:
        pending.append(load(index))
        if len(pending) == batch:
            collate_tokens(pending)
            pending.clear()
    if pending:
        collate_tokens(pending)
    seconds = time.perf_counter()-start
    return dict(rows=len(indices), seconds=seconds, rows_per_second=len(indices)/seconds,
                includes='seek/read/decompress/selected-row reconstruction/CPU batch padding and tensor conversion')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rows-per-game', type=int, default=512)
    p.add_argument('--lookups', type=int, default=4096)
    args = p.parse_args()
    import torch
    torch.set_num_threads(1)
    archives = json.loads((args.source/'manifest.json').read_text())['archives']
    selected = [next(a for a in archives if a['level'] == level) for level in (7,30,39,42)]
    samples, spans = [], []
    for a in selected:
        frames = read_frames(args.source/a['path'])
        start = max(0,(len(frames)-args.rows_per_game)//2)
        before = len(samples)
        samples.extend(frame_tokens(f,token_input='obs_v3') for f in frames[start:start+args.rows_per_game])
        spans.append(dict(source=a['path'],level=a['level'],lo=before,hi=len(samples),first_frame=start))
        del frames
    rng = random.Random(157)
    random_rows = [rng.randrange(len(samples)) for _ in range(args.lookups)]
    report = dict(rows=len(samples),sources=spans,single_cpu_thread=True,
        cache='OS page-cache warm, application decoded cache disabled for random reads; no cold-cache claim',
        reference_rows=23000000,exact_r42_rows=23631235,reference_sparse_bytes_per_row=5544.54325398948,
        decoder='zlib level3, exact sparse float32 plus target links; experimental TKB1, not installed in production')
    cases = []
    generated = 0
    with tempfile.TemporaryDirectory(prefix='obsrec-storage-') as temporary:
        directory = Path(temporary)
        for block_size in (1,8,32,128):
            path = directory/f'blocks{block_size}.bin'
            offsets = [0]; raw_bytes = 0
            with path.open('wb') as handle:
                for at in range(0,len(samples),block_size):
                    compressed, raw = encode(samples[at:at+block_size])
                    handle.write(compressed); offsets.append(handle.tell()); raw_bytes += raw
            size = path.stat().st_size
            generated += size
            with path.open('rb') as handle:
                def load(i):
                    block = i//block_size
                    handle.seek(offsets[block])
                    stored = decode(handle.read(offsets[block+1]-offsets[block]))
                    return stored.sample(i%block_size)
                shuffled = list(range(len(offsets)-1)); rng.shuffle(shuffled)
                sequential = [i for block in shuffled for i in range(block*block_size,min(len(samples),(block+1)*block_size))]
                random_result = measure(load,random_rows)
                # Decode once per block for the locality-aware sampler; preserve row order on return.
                cached_block, cached_store = -1, None
                def local(i):
                    nonlocal cached_block, cached_store
                    block = i//block_size
                    if block != cached_block:
                        handle.seek(offsets[block])
                        cached_store = decode(handle.read(offsets[block+1]-offsets[block])); cached_block = block
                    return cached_store.sample(i%block_size)
                local_result = measure(local,sequential)
            ratio = size/raw_bytes
            projected = ratio*report['reference_sparse_bytes_per_row']*report['reference_rows']
            index_bytes = 8*((report['reference_rows']+block_size-1)//block_size+1)
            block_bytes = size/(len(offsets)-1)
            cases.append(dict(block_rows=block_size,sample_bytes=size,raw_bytes=raw_bytes,compression_ratio=ratio,
                projected_23m_bytes=projected+index_bytes,index_23m_bytes=index_bytes,
                projected_r42_bytes=ratio*report['reference_sparse_bytes_per_row']*report['exact_r42_rows']+
                    8*((report['exact_r42_rows']+block_size-1)//block_size+1),
                mean_block_bytes=block_bytes,random=random_result,locality=local_result,
                disk_budget_random_rows_s={str(mbps):1/(1/random_result['rows_per_second']+block_bytes/(mbps*1e6)) for mbps in (50,200)}))
            path.unlink()
        episodes = []
        raw_sizes = []
        for number, span in enumerate(spans):
            writer = SparseTokenWriter(directory)
            for sample in samples[span['lo']:span['hi']]: writer.append(sample)
            target = directory/f'episode{number}.npz'
            meta = writer.finish(target)
            episodes.append(target)
            raw_sizes.append(15*meta['tokens']+5*meta['features']+280*meta['rows']+16)
        episode_bytes = sum(path.stat().st_size for path in episodes); generated += episode_bytes
        def episode_load(i):
            which = next(j for j,span in enumerate(spans) if span['lo'] <= i < span['hi'])
            return SparseTokens(episodes[which]).sample(i-spans[which]['lo'])
        episode_random = measure(episode_load,random_rows[:256])
        average_bytes = episode_bytes/len(episodes)
        cached_episode, cached_store = -1, None
        def episode_local(i):
            nonlocal cached_episode, cached_store
            which = next(j for j,span in enumerate(spans) if span['lo'] <= i < span['hi'])
            if which != cached_episode:
                cached_store = SparseTokens(episodes[which]); cached_episode = which
            return cached_store.sample(i-spans[which]['lo'])
        episode_local_result = measure(episode_local,list(range(len(samples))))
        report['episode_on_demand'] = dict(sample_bytes=episode_bytes,raw_bytes=sum(raw_sizes),
            compression_ratio=episode_bytes/sum(raw_sizes),
            projected_23m_bytes=episode_bytes/sum(raw_sizes)*report['reference_sparse_bytes_per_row']*report['reference_rows'],
            projected_r42_bytes=episode_bytes/sum(raw_sizes)*report['reference_sparse_bytes_per_row']*report['exact_r42_rows'],
            random=episode_random,locality=episode_local_result,
            mean_episode_rows=len(samples)/len(spans),mean_episode_bytes=average_bytes,
            disk_budget_random_rows_s={str(mbps):1/(1/episode_random['rows_per_second']+average_bytes/(mbps*1e6)) for mbps in (50,200)})
    report.update(blocks=cases,temporary_compressed_written_bytes=generated,temporary_raw_columns_written_bytes=sum(raw_sizes),
        temporary_total_written_bytes=generated+sum(raw_sizes),temporary_removed=True,
        projection_note='Disk projections normalize compression ratios to the 52-game task157 sample, not just these four windows. Frozen row plan is reused; a new optional uint32 row map adds 92 MB for 23M rows. Cold disk budgets assume effective 50/200 MB/s, not measured SSD throughput; CPU and assumed read times are added without overlap.')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(rows=len(samples),temporary_written=generated+sum(raw_sizes),removed=True)))
    for case in cases:
        print(case['block_rows'],round(case['projected_23m_bytes']/1e9,2),round(case['random']['rows_per_second']),round(case['locality']['rows_per_second']))
    print('episode',round(report['episode_on_demand']['projected_23m_bytes']/1e9,2),round(episode_random['rows_per_second']),round(episode_local_result['rows_per_second']))


if __name__ == '__main__':
    main()
