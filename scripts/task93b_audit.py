"""Audit the three imported task93b labels; launches no games or training."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]
from import_mac_labels import ledger_digest, sha, training_rows
from pvz_promotion import _wilson
from pvz_tasks import profile_for_deck, random_deck_covered, random_deck_draws

PANEL = Path('/home/newbiexvwu/PvZ-search/artifacts/stage4/random_deck_panel.json')
FAMILIES = [('day_l7_train_rand', 'd7dag93m100', 93000500, 1., 104),
            ('night_l12_train_rand', 'n12dag93m050', 93010500, .5, 75),
            ('pool_l21_train_rand', 'p21w93m050', 93020500, .5, 114)]


def audit():
    tasks = {t['task_id']: t for t in json.loads(PANEL.read_text())['tasks']}
    receipt = json.loads((ROOT/'artifacts/task93/mac_import.json').read_text())
    corpus = Path(receipt['destination'])
    records = {r['label']: r for r in receipt['rounds']}
    original_seeds = {s for t in tasks.values() for s in t['seeds']}
    result = dict(source_git_commit='ec5dfc9b24226b224faae68088f6c59d9ee3a305',
        source_hf_revision=receipt['revision'], repo=receipt['repo'], destination=str(corpus),
        original_panel_sha256=sha(PANEL), families=[],
        use='followup curriculum plan; frozen 144-game validation unchanged',
        limitations=['Mac engine-validation manifests and evaluation configs were not transferred; '
                    'sampler and archive metadata checked locally, not new engine-validation receipts.',
                    'Outcomes come from archive metadata, not paired student evaluations.',
                    'Row counts are usable rows before wait sampling; measure retained rows with a training pack.'])
    for family, label, start, multiplier, expected_wins in FAMILIES:
        task = tasks[family]
        seeds = set(range(start, start+192))
        assert seeds.isdisjoint(original_seeds), 'original-panel seed collision'
        counts, lengths, attempts = Counter(), Counter(), Counter()
        ledger = []
        won = rows = frames = 0
        for seed in sorted(seeds):
            folder = corpus/f'seed_{seed}/{label}_fast_search'
            meta = json.loads((folder/'meta.json').read_text())
            t, outcome = meta['task'], meta['outcome']
            deck = t['deck']
            assert meta['label_source'] == 'teacher' and t['arm'] == 'fast_search' and t['policy'] == 'student'
            assert t['seed'] == seed and t['task_id'] == family+'_expand'
            assert t['level'] == task['level'] and t['playthrough'] == 2
            assert t['checkpoint_sha256'] == '1cf82f5e6cf7994ca961d91e878d4ac65a4563eea2be50a1e96a9787c22dfd4c'
            assert t['zombie_count_multiplier'] == multiplier == meta['context']['multiplier']
            assert 6 <= len(deck) <= 10 and len(set(deck)) == len(deck) and random_deck_covered(task, deck)
            assert profile_for_deck(tuple(deck)).seed_slot_count == len(deck)
            attempt = next((a for a, draw in random_deck_draws(task, seed) if tuple(deck) == draw), None)
            assert attempt is not None, 'archived deck differs from frozen candidate-pool sampler'
            attempts[attempt] += 1
            lengths[len(deck)] += 1
            counts.update(deck)
            assert type(outcome['won']) is bool and outcome['final_tick'] > 0
            assert 0 <= outcome['final_wave'] <= outcome['wave_count']
            won += outcome['won']
            path = folder/'frames.npz'
            rows += training_rows(path.read_bytes())
            with np.load(path, allow_pickle=False) as stored:
                assert set(stored.files) <= {'frames', 'encoder', 'hints', 'rows', 'boards', 'scalars',
                                             'plant', 'packet', 'cell', 'values'}
                assert stored['frames'].dtype == np.uint8
                frames += len(stored['frames'].tobytes().splitlines())
            ledger.extend((str(p.relative_to(corpus)), sha(p)) for p in (path, folder/'meta.json'))
        record = records[label]
        assert record['archives'] == 192 and rows == record['rows'] and won == expected_wins
        assert ledger_digest(ledger) == record['local_ledger_sha256'], 'local bytes differ from import receipt'
        row = dict(task_id=family+'_expand', label=label+'_fast_search', seed_range=[start, start+191],
            archives=192, multiplier=multiplier, won=won, win_rate=won/192, wilson95=_wilson(won, 192),
            frames=frames, unsampled_usable_rows=rows, deck_length_counts=dict(sorted(lengths.items())),
            plant_deck_counts={str(p): counts[p] for p in range(48)}, sampler_attempt_counts=dict(sorted(attempts.items())),
            heldout_deck_collisions=0, original_panel_seed_collisions=0,
            archive_sha256=record['archive_sha256'], local_ledger_sha256=record['local_ledger_sha256'],
            remote_path=record['remote_path'])
        result['families'].append(row)
        print(f'{label}: {won}/192; {rows} unsampled usable rows', flush=True)
    result['archives'] = sum(r['archives'] for r in result['families'])
    result['unsampled_usable_rows'] = sum(r['unsampled_usable_rows'] for r in result['families'])
    return result


if __name__ == '__main__':
    path = ROOT/'artifacts/task93/mac_import_audit.json'
    report = audit()
    previous = json.loads(path.read_text()) if path.exists() else {}
    if previous.get('families') == report['families'] and 'remote_cleanup' in previous:
        report['remote_cleanup'] = previous['remote_cleanup']
    path.write_text(json.dumps(report, indent=2)+'\n')
