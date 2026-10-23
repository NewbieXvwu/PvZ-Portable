"""Reserve 20% of supplemental teacher seeds for A6 before training sees them."""
import argparse
import json
from pathlib import Path
from value_widen_audit import seed_key


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--supplement',action='append',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();games=json.loads(a.audit.read_text());reserved=[]
    for directory in a.supplement:
        for meta_path in sorted((directory/'frames').glob('*/*/meta.json')):
            meta=json.loads(meta_path.read_text());seed=int(meta['task']['seed'])
            # Remainder 0 belongs to early stopping; 1/2 are never used for fitting/stopping.
            if int(seed_key(seed)[:8],16)%10 not in (1,2):continue
            label=meta_path.parent.name
            group='pool_teacher_transfer' if 'pool' in label else 'day_original_teacher_transfer' if 'day10' in label else 'night_original_teacher_transfer'
            reserved.append(dict(cohort=group,seed=seed,archive=str(meta_path.parent)))
    all_games=games+reserved
    train_overlap={g['seed'] for g in games}&{g['seed'] for g in reserved}
    if train_overlap:raise ValueError('supplement duplicates original external seed; inspect before freezing')
    a.out.write_text(json.dumps(all_games,indent=1));print(f'{len(games)} original + {len(reserved)} supplemental external games',flush=True)
if __name__=='__main__':main()
