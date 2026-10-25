"""Audit label coverage and ranking-signal sufficiency in the original task72b rows.

Every labelled candidate enters the audit, including the h4000-best arm when it
differs from static-best. This checks the supplied summary, not engine replay or
candidate leaf scores, and starts no game or training process.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'MEMORY/docs/task72b-probe-rows.jsonl'
TARGET=ROOT/'MEMORY/docs/task78-task72-truth-audit.json'
MIN_GAP=.25
FUTURES=8
HORIZONS=('2500','4000','6000')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyse(rows, expected_count=94):
    """A zero eligible-pair upper bound requires no unknown nonterminal leaf flags."""
    identity=lambda r:(r['task'],r['seed'],r['decision'],r['tick'])
    if len(rows)!=expected_count or len({identity(row) for row in rows})!=expected_count:
        raise ValueError('missing or duplicate original state identities')
    candidate_counts=Counter()
    gaps=Counter()
    eligible=[]
    static_gaps=Counter()
    labelled_futures=0
    all_loss=0
    for row in rows:
        cells={f'{r},{c}' for r,c in row['cells']}
        if len(cells)!=row['n_cells'] or len(cells)!=len(row['cells']):
            raise ValueError('original candidate cell coverage is inconsistent')
        static=','.join(map(str,row['static_best']))
        truth=row['truth']
        if not {'wait',static}<=set(truth) or not set(truth)<=cells|{'wait'}:
            raise ValueError('truth candidate identity is outside the original legal set')
        if set(row['values'])!=set(HORIZONS) or any(
                set(row['values'][h])!=cells|{'wait'} for h in HORIZONS):
            raise ValueError('original horizon/candidate coverage is incomplete')
        for values in row['values'].values():
            if any(not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1
                   for v in values.values()):
                raise ValueError('invalid supplied leaf value')
        rates={}
        for action,record in truth.items():
            wins,finals=record['wins'],record['finals']
            if len(wins)!=FUTURES or len(finals)!=FUTURES:
                raise ValueError('every supplied truth arm requires eight recorded futures')
            if any(v not in (0,1) for v in wins+finals) or wins!=finals:
                raise ValueError('recorded final values and binary wins differ')
            rate=record['win_rate']
            if not isinstance(rate,(int,float)) or not math.isfinite(rate) or abs(rate-sum(wins)/FUTURES)>1e-12:
                raise ValueError('win rate differs from its eight recorded wins')
            rates[action]=rate
            labelled_futures+=FUTURES
        candidate_counts[len(truth)]+=1
        all_loss+=int(all(rate==0 for rate in rates.values()))
        static_gaps[abs(rates[static]-rates['wait'])]+=1
        for left,right in combinations(sorted(rates),2):
            gap=abs(rates[left]-rates[right])
            gaps[gap]+=1
            if gap>=MIN_GAP:
                eligible.append(dict(identity=identity(row),left=left,right=right,absolute_truth_gap=gap))
    return dict(scope='arithmetic_and_candidate_coverage_audit_of_supplied_task72b_summary',
        states=len(rows),unique_state_identities=len(rows),
        labelled_candidates_per_state={str(k):v for k,v in sorted(candidate_counts.items())},
        recorded_candidate_future_outcomes=labelled_futures,
        all_labelled_pairs=sum(gaps.values()),
        absolute_gap_counts={str(k):v for k,v in sorted(gaps.items())},
        maximum_absolute_gap=max(gaps,default=None),
        static_vs_wait_gap_counts={str(k):v for k,v in sorted(static_gaps.items())},
        minimum_registered_gap=MIN_GAP,eligible_truth_pairs=len(eligible),
        eligible_state_count=len({tuple(row['identity']) for row in eligible}),
        eligible_pairs=eligible,all_candidates_lose_states=all_loss,
        nonterminal_eligible_pairs_upper_bound=len(eligible),
        ranking_signal_status='insufficient_evidence' if not eligible else 'leaf_scores_and_terminal_classification_required',
        value_consistency=dict(v5=None,v6=None,improvement=None,bootstrap_interval=None),
        engine_replay_verified=False,leaf_scores_recomputed=False,horizon_stability_evaluated=False,
        original_takeover_mapping_verified=False,original_26_state_mask_restored=False,
        overall_task_complete=False,
        note='The supplied wins/finals and candidate coverage are internally consistent. This does not verify original trajectories or terminal traces. With zero truth-gap-eligible pairs, nonterminal filtering cannot produce an evaluable ranking pair; no zero agreement rate or fabricated confidence interval is reported.')


def main():
    rows=[json.loads(line) for line in REFERENCE.read_text().splitlines() if line.strip()]
    report=analyse(rows)
    report.update(reference_sha256=digest(REFERENCE),audit_script_sha256=digest(Path(__file__)),
                  engine_processes_started=0)
    TARGET.write_text(json.dumps(report,ensure_ascii=False,indent=1)+'\n')
    print(json.dumps({k:report[k] for k in ('states','all_labelled_pairs','maximum_absolute_gap',
                    'eligible_truth_pairs','ranking_signal_status')},indent=1))


if __name__=='__main__':main()
