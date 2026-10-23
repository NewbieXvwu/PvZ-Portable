"""Read-only task52 evidence audit; never starts a game or calls an API."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'),str(ROOT/'python')]
from llm_candidates import canonical, digest, filter_actions, parse_payload, atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_event(event, trace, cache, settings, threshold):
    """Reparse the actual cached response and its exact engine-table prompt."""
    saved = read(cache/f'{event["position"]}.json')
    assert digest(saved['identity']) == event['position'], 'wrong cache key'
    response = saved['response']
    assert digest(response) == event['response_sha256'], 'changed cached response'
    text = ''.join(p.get('text','') for p in response['candidates'][0]['content']['parts']
                   if not p.get('thought'))
    payload = parse_payload(text)
    prompt = saved['identity']['body']['contents'][0]['parts'][0]['text']
    legal = json.loads(prompt.rsplit('\n引擎合法种植动作：',1)[1])
    actions, counts = filter_actions(payload,legal)
    assert actions == event['actions'] and counts == event['counts'], 'legality/count mismatch'
    assert payload == event['payload'] and response.get('usageMetadata',{}) == event['usage']
    index = event['decision_seed'] % 10000
    root = trace[index]
    assert root['tick'] == event['tick'] and root['wave'] == event['wave'] and root['searched']
    assert root['chosen'] == event['chosen']
    valid = {canonical(a) for a in actions}
    novel = {canonical(a) for a in event['novel_actions']}
    assert novel <= valid and len(novel) == len(event['novel_actions'])
    selected = canonical(event['chosen'])
    assert event['suggestion_selected'] == (selected in valid)
    assert event['novel_selected'] == (selected in novel)
    if event['trigger'] == 'B_low':
        values = event['evidence']['policy_leaf_values']
        assert len(values) == 3 and all(math.isfinite(v) and v < threshold for v in values)
    else:
        evidence = event['evidence']
        assert evidence['new_types']
        detail = json.loads(prompt.split('\n公开实体：',1)[1].split('\n引擎合法种植动作：',1)[0])
        for kind in evidence['new_types']:
            assert any(z['type']==kind and abs(z['row']-evidence['reference_row']) <= settings['adjacent_rows']
                       for z in detail.get('zombies',[]))
        if root['policy']['type']=='plant':
            assert evidence['reference_row'] == root['policy']['row']
    assert len(event['actions']) <= 3
    if event['cache_hit']:
        assert not event['attempts'], 'cache hit charged as a request'
    else:
        assert event['attempts'] == saved['attempts']
    if event['novel_selected']:
        scores = {canonical(a):v for a,v in event['scores']}
        advantage = scores[selected]-event['anchor_value']
        assert math.isclose(advantage,event['selected_leaf_advantage'],abs_tol=1e-12)
        # The same original margin protects every proposer; terminal diagnostics cannot
        # retroactively choose an action, and are reported separately from the real game.
        assert advantage > .02
        pairs = event['terminal_pairs']
        assert len(pairs) == settings['terminal_futures']
        assert event['diagnostic_cpu_s'] >= 0
        for k,pair in enumerate(pairs):
            assert pair['future'] == (event['decision_seed']*1000+9000001+k)%2**32
            for name in ('selected','anchor'):
                assert type(pair[name]['terminal']) is bool and type(pair[name]['won']) is bool
    return counts


def audit(out, allow_incomplete=False):
    out=Path(out)
    plan=read(out/'plan.json');contract=digest(plan)
    evidence=[]
    for filename,expected in plan['files'].items():
        assert sha(filename)==expected, f'frozen input changed: {filename}'
    evidence.append(('plan.json',sha(out/'plan.json')))
    calibrated=read(out/'calibration.json')
    assert calibrated['contract']==contract
    all_results={};missing=[];games=events_count=0
    for family_index,family in enumerate(('day','roof','fog')):
        values=[]
        for seed in range(52010000+family_index*1000,52010002+family_index*1000):
            dest=out/'calibration'/family/'A'/str(seed)
            r=read(dest/'result.json');trace=read(dest/'trace.json')
            assert r['contract']==contract and r['terminal'] and r['llm_queries']==0
            assert len(trace)==r['decisions'] and sha(dest/'trace.json')==r['trace_sha256']
            values.extend(r['calibration_leaf_values'])
            for p in (dest/'result.json',dest/'trace.json'):evidence.append((str(p.relative_to(out)),sha(p)))
        # Independent linear quantile calculation, not the original numpy call.
        values.sort();location=(len(values)-1)*plan['settings']['quantile']
        lower=math.floor(location);upper=math.ceil(location)
        threshold=values[lower]+(values[upper]-values[lower])*(location-lower)
        assert math.isclose(threshold,calibrated['thresholds'][family],rel_tol=1e-14)
        assert len(values)==calibrated['counts'][family]
        arms={}
        for arm in ('A','B_low','B_novel'):
            arms[arm]=[]
            for seed in range(52000000+family_index*1000,52000016+family_index*1000):
                dest=out/'games'/family/arm/str(seed)
                if not (dest/'result.json').exists():
                    missing.append((family,arm,seed));continue
                r=read(dest/'result.json');trace=read(dest/'trace.json');events=read(dest/'events.json')
                assert r['contract']==contract and r['terminal']
                assert r['seed']==seed and r['family']==family and r['comparison_arm']==arm
                assert sha(dest/'trace.json')==r['trace_sha256'] and sha(dest/'events.json')==r['events_sha256']
                assert len(trace)==r['decisions'] and sum(x['searched'] for x in trace)==r['searched']
                assert len(events)==r['llm_queries'] <= plan['settings']['max_calls']
                assert sum(e['cache_hit'] for e in events)==r['llm_cache_hits']
                assert arm!='A' or not events
                assert r['cpu_s_including_diagnostics']>=r['diagnostic_cpu_s']>=0
                previous=-10**12;handled=set()
                for event in events:
                    assert event['source']=='llm' and event['trigger']==arm
                    assert event['decision_seed']//10000==seed
                    assert event['tick']-previous>=plan['settings']['interval']
                    previous=event['tick']
                    if arm=='B_novel':
                        new=set(event['evidence']['new_types']);assert not (new&handled);handled.update(new)
                    check_event(event,trace,ROOT/'artifacts/task52/cache',plan['settings'],threshold)
                for p in (dest/'result.json',dest/'trace.json',dest/'events.json'):
                    evidence.append((str(p.relative_to(out)),sha(p)))
                arms[arm].append(r);games+=1;events_count+=len(events)
        all_results[family]=arms
    comparison={}
    if not missing:
        for family,arms in all_results.items():
            comparison[family]={}
            for arm in ('B_low','B_novel'):
                before={r['seed']:r for r in arms['A']};after={r['seed']:r for r in arms[arm]}
                assert len(before)==len(after)==16 and before.keys()==after.keys()
                win_diff=[int(after[s]['won'])-int(before[s]['won']) for s in sorted(before)]
                wave_diff=[after[s]['final_wave']-before[s]['final_wave'] for s in sorted(before)]
                comparison[family][arm]=dict(before_wins=sum(r['won'] for r in before.values()),
                    after_wins=sum(r['won'] for r in after.values()),net_wins=sum(win_diff),
                    improved=win_diff.count(1),regressed=win_diff.count(-1),
                    wave_deltas=wave_diff,mean_wave_delta=statistics.mean(wave_diff))
    result=dict(status='incomplete' if missing else 'complete',contract=contract,
                auditor_sha256=sha(__file__),games_verified=games,llm_events_verified=events_count,
                missing_games=missing,paired_comparison=comparison,
                evidence_files=len(evidence),evidence_bundle_sha256=digest(sorted(evidence)))
    atomic_json(out/'independent-audit.json',result)
    if missing and not allow_incomplete:
        raise ValueError(f'{len(missing)} formal games still missing; audit is incomplete')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/task52/study-v1')
    p.add_argument('--allow-incomplete',action='store_true')
    args=p.parse_args();r=audit(args.out,args.allow_incomplete)
    print(canonical({k:v for k,v in r.items() if k!='missing_games'}))


if __name__=='__main__':main()
