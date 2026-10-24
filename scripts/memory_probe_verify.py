"""Task58 completion audit: fail on missing data, runs, predictions or evidence."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_frame_store import read_frames
from pvz_memory_probe import CONFIGS, make_probe
from memory_probe_data import OUT, digest, write_json
from memory_probe_metrics import masks, stats


def verify():
    torch.set_num_threads(1)
    corpus_path = OUT / 'corpus.json'
    corpus_hash = digest(corpus_path)
    corpus = json.loads(corpus_path.read_text())
    entries = json.loads((OUT/'inventory.json').read_text())
    assert corpus['inventory_sha256'] == digest(OUT/'inventory.json')
    assert len(entries) == len(corpus['episodes']) == 164
    declared = json.loads((ROOT/'MEMORY/docs/memory-probe-data.json').read_text())
    assert declared['corpus_sha256'] == corpus_hash
    input_audit = json.loads((ROOT/'MEMORY/docs/memory-probe-input-audit.json').read_text())
    assert input_audit['corpus_sha256'] == corpus_hash and input_audit['bit_mismatches'] == 0
    expected_ids, groups = {}, {}
    frames, positives, returns = 0, 0, {}
    for entry, row in zip(entries, corpus['episodes']):
        assert entry['id'] == row['id']
        group = (entry['task']['level'],entry['task']['seed'])
        assert groups.setdefault(group,row['split']) == row['split']
        expected_ids.setdefault(row['split'],set()).add(row['id'])
        assert digest(entry['source']) == entry['source_sha256'] == row['source_sha256']
        source = read_frames(Path(entry['source']))
        path = OUT/'episodes'/f"{row['id']}.npz"
        assert digest(path) == row['pack_sha256']
        with np.load(path,allow_pickle=False) as pack:
            n = row['frames']
            assert n == row['checked_public_frames'] == len(source) and row['mismatches'] == 0
            assert pack['human_board'].shape == pack['xray_board'].shape == (n,102,6,10)
            assert pack['human_scalar'].shape == pack['xray_scalar'].shape == (n,157)
            assert pack['target'].shape == (n,35,6,9)
            assert np.isin(pack['target'],[0,1]).all()
            np.testing.assert_array_equal(pack['target'][:,0],pack['target'][:,1:].any(1))
            ticks = np.asarray([f['tick'] for f in source])
            np.testing.assert_array_equal(pack['tick'],ticks)
            np.testing.assert_array_equal(pack['dt'],np.diff(ticks,prepend=ticks[0]))
            np.testing.assert_array_equal(pack['fog'],np.asarray([f['fog_amount'] for f in source])>0)
            assert int(pack['fog'].sum()) == row['hidden_cells']
            assert int((pack['target'][:,0]*pack['fog']).sum()) == row['hidden_occupied']
            positives += row['hidden_occupied']
        frames += n
        returns[row['split']] = returns.get(row['split'],0) + len(row['returns'])
    assert frames == input_audit['checked_frames'] == 66092
    assert {k:len(v) for k,v in expected_ids.items()} == dict(train=100,validation=20,test=44)
    assert len(set.union(*expected_ids.values())) == 164
    assert len(groups) == 164
    evaluation_path = OUT/'evaluation.json'
    evaluation = json.loads(evaluation_path.read_text())
    summary = json.loads((ROOT/'MEMORY/docs/memory-probe-results.json').read_text())
    assert summary['full_evaluation_sha256'] == digest(evaluation_path)
    assert evaluation['corpus_sha256'] == summary['corpus_sha256'] == corpus_hash
    assert evaluation['test_episodes'] == 44 and evaluation['test_returns'] == returns['test']
    runs = {(r['kind'],r['arm'],r['seed']):r for r in evaluation['runs']}
    assert len(runs) == 30
    resource_rows, prediction_hashes, max_difference = [], {}, 0.
    test_rows = [r for r in corpus['episodes'] if r['split']=='test']
    for seed in (1,2,3):
        for kind in CONFIGS:
            for arm in ('human','xray'):
                folder = OUT/'models'/f'{kind}-{arm}-s{seed}'
                result = json.loads((folder/'result.json').read_text())
                history = result['history']
                assert [r['epoch'] for r in history] == list(range(1,21))
                selected = min(history,key=lambda r:r['validation_hidden_logloss'])
                assert selected['epoch'] == result['best_epoch']
                assert selected['validation_hidden_logloss'] == result['validation_hidden_logloss']
                assert result['mem_available_bytes'] >= 10*1024**3
                assert result['elapsed_seconds'] < 3600 and result['time_limit_seconds'] == 3600
                assert result['corpus_sha256'] == corpus_hash
                assert result['checkpoint_sha256'] == digest(folder/'model.pt')
                assert result['train_episodes'] == 100 and result['validation_episodes'] == 20
                assert {p.stem for p in (folder/'test').glob('*.npz')} == expected_ids['test']
                run = runs[kind,arm,seed]
                assert {e['id'] for e in run['episodes']} == expected_ids['test']
                resource_rows.append(dict(kind=kind,arm=arm,seed=seed,
                    mem_available_bytes=result['mem_available_bytes'],elapsed_seconds=result['elapsed_seconds'],
                    checkpoint_sha256=result['checkpoint_sha256'],parameters=result['parameters']))
                # Independently run one full held-out episode on CPU from selected weights.
                row = test_rows[seed-1]
                model = make_probe(kind).eval()
                checkpoint = torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
                model.load_state_dict(checkpoint['state_dict'])
                for test in test_rows:
                    path = folder/'test'/f"{test['id']}.npz"
                    with np.load(path,allow_pickle=False) as prediction:
                        probability = prediction['probabilities']
                        assert str(prediction['checkpoint_sha256']) == result['checkpoint_sha256']
                        assert str(prediction['corpus_sha256']) == corpus_hash
                        assert str(prediction['episode_pack_sha256']) == test['pack_sha256']
                        assert probability.shape == (test['frames'],35,6,9)
                        assert np.isfinite(probability).all() and (probability>=0).all() and (probability<=1).all()
                        assert (probability[:,1:] <= probability[:,:1]+1e-7).all()
                        if test['id'] == row['id']:
                            with np.load(OUT/'episodes'/f"{row['id']}.npz",allow_pickle=False) as pack:
                                hidden = pack['fog']
                                expected = stats(pack['target'][:,0][hidden],probability[:,0][hidden])
                                recorded = next(e for e in run['episodes'] if e['id']==row['id'])['hidden']
                                assert expected == recorded
                                state, values = None, []
                                with torch.inference_mode():
                                    for base in range(0,row['frames'],17):
                                        sl = slice(base,base+17)
                                        output,state = model(torch.from_numpy(pack[f'{arm}_board'][sl,None]),
                                            torch.from_numpy(pack[f'{arm}_scalar'][sl,None]),
                                            torch.from_numpy(pack['dt'][sl,None].astype(np.float32)),state)
                                        p = output.sigmoid()[:,0]
                                        p[:,1:] *= p[:,:1]
                                        values.append(p.numpy())
                                cpu_probability = np.concatenate(values)
                                difference = float(np.max(np.abs(cpu_probability-probability)))
                                max_difference = max(max_difference,difference)
                                np.testing.assert_allclose(cpu_probability,probability,atol=2e-5,rtol=2e-4)
                    prediction_hashes[str(path.relative_to(OUT))] = digest(path)
    params = [r['parameters'] for r in resource_rows]
    assert max(params)/min(params) < 1.05
    assert set(evaluation['cpu']['configurations']) == set(CONFIGS)
    assert evaluation['cpu']['threads'] == evaluation['cpu']['batch'] == 1
    for row in evaluation['cpu']['configurations'].values():
        assert 0 < row['p50_us'] <= row['p95_us'] and row['state_bytes'] >= 0
    for bucket in ('hidden','age_0_120','age_121_600','age_over_600','return_after'):
        assert set(evaluation['comparisons'][bucket]) == set(CONFIGS[1:])
    tests = subprocess.run([sys.executable,'-m','pytest','-q',str(ROOT/'python/test_memory_probe.py')],
                           check=True,capture_output=True,text=True)
    changed = subprocess.run(['git','-C',str(ROOT),'diff','--name-only','8cd67c31'],
                              check=True,capture_output=True,text=True).stdout.splitlines()
    allowed = ('MEMORY/','scripts/memory_probe_','python/pvz_memory_probe.py','python/test_memory_probe.py')
    assert all(p.startswith(allowed) for p in changed)
    assert not any(Path(p).suffix in ('.pt','.npz','.log') for p in changed)
    tracked = subprocess.run(['git','-C',str(ROOT),'ls-files','-z',str(ROOT/'artifacts'),str(ROOT/'logs')],
                              check=True,capture_output=True).stdout
    ignored = subprocess.run(['git','-C',str(ROOT),'check-ignore','-z','--stdin','--no-index'],
                              input=tracked,capture_output=True)
    assert not ignored.stdout
    branch = subprocess.run(['git','-C',str(ROOT),'branch','--show-current'],
                             check=True,capture_output=True,text=True).stdout.strip()
    assert branch == 'memprobe'
    full = dict(corpus_sha256=corpus_hash,episodes=164,frames=frames,split_counts={k:len(v) for k,v in expected_ids.items()},
        hidden_occupied=positives,returns=returns,runs=resource_rows,prediction_files=len(prediction_hashes),
        cpu_recomputed_episodes=30,cpu_prediction_max_abs_error=max_difference,
        prediction_hashes=prediction_hashes,tests=tests.stdout.strip(),
        allowed_changed_paths=changed,tracked_ignored_files=0,branch=branch)
    write_json(OUT/'completion-audit.json',full)
    concise = {k:v for k,v in full.items() if k not in ('prediction_hashes','runs')}
    concise.update(full_audit_sha256=digest(OUT/'completion-audit.json'),runs=30,
        min_start_mem_available_gib=min(r['mem_available_bytes'] for r in resource_rows)/1024**3,
        max_training_seconds=max(r['elapsed_seconds'] for r in resource_rows),
        status='all machine-verifiable data/model/metric/resource gates passed; report and git delivery audited separately')
    write_json(ROOT/'MEMORY/docs/memory-probe-completion-audit.json',concise)
    print(json.dumps(concise,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    verify()
