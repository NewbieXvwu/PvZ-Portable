"""Bounded stratified replay sample for task157; separate from collection/training.

Only the explicitly sampled archives are replayed in an isolated engine. Restart uses
its journal, including recorded failures. --report-only reads existing results and
attachment metadata; neither mode copies original frame files.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]
from pvz_frame_store import frame_paths, archive_seed
from pvz_event_env import EventWaitEnv
from backfill_observation_v3 import replay_archive


def sample_plan(args):
    plan_path = args.out/'plan.json'
    if plan_path.exists():
        return json.loads(plan_path.read_text())
    strata = defaultdict(list)
    for path in frame_paths(args.data):
        match = re.match(r'c(\d+)', path.parent.name)
        if match and args.first <= int(match[1]) <= args.last and (path.parent/'meta.json').is_file():
            strata[int(match[1]), 'demo' if '_f' in path.parent.parent.name else 'normal'].append(path)
    folders = {p.name.rsplit('_', 1)[0]: p for p in args.groups.iterdir() if p.is_dir()}
    plan = []
    for (round_id, kind), paths in sorted(strata.items()):
        paths = sorted(paths)
        random.Random(157000+round_id*2+(kind == 'demo')).shuffle(paths)
        for path in paths[:args.samples]:
            group = folders.get(path.parent.name.removesuffix('_fast_search'))
            config = json.loads((group/'config.json').read_text()) if group and (group/'config.json').is_file() else {}
            plan.append(dict(round=round_id, stratum=kind, population=len(paths), source=str(path),
                             manifest=config.get('manifest'), task_id=config.get('task_id'), group=str(group)))
    plan_path.write_text(json.dumps(plan, indent=2)+'\n')
    return plan


def projection(args, results):
    import numpy as np
    strata = defaultdict(list)
    for result in results:
        if not result['mismatch'] and 'raw_bytes' not in result:
            source = Path(result['source'])
            attachment = args.out/'backfill/frames'/source.parent.parent.name/source.parent.name/'tokens_v3.npz'
            with np.load(attachment, allow_pickle=False) as arrays:
                meta = json.loads(str(arrays['meta']))
            result.update(raw_bytes=15*meta['tokens']+5*meta['features']+280*meta['rows']+16,
                          nonzero_features=meta['features'], compact_bytes=source.stat().st_size)
        strata[result['round'], result['stratum']].append(result)
    rounds = []
    for round_id in range(args.first, args.last+1):
        row = dict(round=round_id, population=0, sample_successes=0, sample_games=0,
                   estimated_successes=0., estimated_backfill_seconds=0.,
                   estimated_accepted_attachment_bytes=0., estimated_capture_attachment_bytes=0.,
                   estimated_compact_frame_bytes=0., estimated_raw_attachment_bytes=0., estimated_frames=0.)
        for kind in ('normal', 'demo'):
            selected = strata[round_id, kind]
            if not selected:
                continue
            population = selected[0]['population']
            good = [r for r in selected if not r['mismatch']]
            row['population'] += population
            row['sample_games'] += len(selected)
            row['sample_successes'] += len(good)
            row['estimated_successes'] += population*len(good)/len(selected)
            row['estimated_backfill_seconds'] += population*sum(r.get('projected_seconds', r['seconds']) for r in selected)/len(selected)
            row['estimated_accepted_attachment_bytes'] += population*sum(r.get('bytes_written', 0) for r in selected)/len(selected)
            if good:
                for key, source in (('estimated_capture_attachment_bytes', 'bytes_written'),
                                    ('estimated_compact_frame_bytes', 'compact_bytes'),
                                    ('estimated_raw_attachment_bytes', 'raw_bytes'), ('estimated_frames', 'frames_written')):
                    row[key] += population*sum(r[source] for r in good)/len(good)
        rounds.append(row)
    good = [r for r in results if not r['mismatch']]
    bytes_per_row = sum(r['estimated_raw_attachment_bytes'] for r in rounds)/max(1, sum(r['estimated_frames'] for r in rounds))
    return dict(rounds=rounds, results=results, sample_games=len(results), successes=len(good),
                sparse_bytes_per_decision=bytes_per_row, frozen_r42_rows=23631235,
                projected_frozen_r42_sparse_bytes=bytes_per_row*23631235,
                temporary_raw_peak_bytes=max((r['raw_bytes'] for r in good), default=0),
                generated_attachments_bytes=sum(r['bytes_written'] for r in good),
                sampling='seed157000+2*round+is_demo; two archives per round/normal-demo stratum',
                executable=str(args.executable),
                caution='Small stratified sample, not a full replay. Time includes demo prefixes, startup, compression and contention; empty rounds have no retained matching archives.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--groups', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--first', type=int, default=36)
    p.add_argument('--last', type=int, default=46)
    p.add_argument('--samples', type=int, default=2, help='per round and normal/demo stratum')
    p.add_argument('--paired', type=int, default=4, help='successful samples also timed without tokens')
    p.add_argument('--report-only', action='store_true')
    p.add_argument('--executable', type=Path, default=ROOT/'build-studentv3net/pvz-portable')
    p.add_argument('--resource-dir', type=Path, default=Path.home()/'.cache/pvz-research-resources')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    journal = args.out/'sample.jsonl'
    results = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    if not args.report_only:
        plan = sample_plan(args)
        completed = {r['source'] for r in results}
        paired = sum('control' in r for r in results)
        with EventWaitEnv(args.resource_dir, executable=args.executable) as env:
            for item in plan:
                if item['source'] in completed:
                    continue
                if env._process is None or env._process.poll() is not None:
                    env.close()
                path = Path(item['source'])
                result = replay_archive(env, path, args.out/'backfill', manifest=item['manifest'], task_id=item['task_id'])
                result.update({k: item[k] for k in ('round', 'stratum', 'population')})
                if not result['mismatch'] and paired < args.paired:
                    result['control'] = replay_archive(env, path, args.out/'control', manifest=item['manifest'], task_id=item['task_id'], capture=False)
                    paired += 1
                group = Path(item['group'])
                if (group/'games.jsonl').is_file():
                    future = archive_seed(path.parent.parent.name)
                    for line in (group/'games.jsonl').read_text().splitlines():
                        game = json.loads(line)
                        if (game.get('future_seed') or game.get('seed')) == future:
                            result['original_collect_seconds'] = game.get('seconds')
                            break
                results.append(result)
                with journal.open('a') as stream:
                    stream.write(json.dumps(result, ensure_ascii=False)+'\n')
                print(json.dumps({k: result[k] for k in ('round', 'stratum', 'mismatch', 'seconds')}), flush=True)
    (args.out/'summary.json').write_text(json.dumps(projection(args, results), ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
