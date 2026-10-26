"""CPU-only task96 deployment/source and pack-metadata audit; no content rehash."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LOGS = Path('/home/newbiexvwu/pvz-logs')
FILES = ['scripts/train_rollout_policy.py', 'scripts/pack_training_rows.py',
         'python/pvz_frame_store.py', 'python/pvz_training_pack.py', 'python/pvz_recipe.py',
         'python/pvz_rollout_policy.py', 'python/pvz_student_observation.py',
         'python/pvz_training_rows.py', 'python/pvz_value_net.py']


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def sha(value):
    return hashlib.sha256(value).hexdigest()


def function(source, name):
    tree = ast.parse(source)
    node = next(n for n in tree.body if getattr(n, 'name', '') == name)
    return ast.dump(node, include_attributes=False)


def main():
    import numpy as np
    before, after = '4f4d217b', 'bec2b9e6'
    setup = (LOGS/'qc_setup.log').read_text().splitlines()
    evidence = [line for line in setup if any(key in line for key in
                ('deploy11:', 'restart qc after stage4 restore', 'deploy12:'))]
    r11 = LOGS/'curriculum_state.pipeline/round_000011'
    training = json.loads((r11/'trained.json').read_text())
    evaluation = json.loads((r11/'complete.json').read_text())['promotion']['contract']
    evaluation_files = [('scripts/search_play.py', 'runner_sha256'),
                        ('python/pvz_promotion.py', 'policy_module_sha256'),
                        ('scripts/promotion_gate.py', 'evaluator_sha256')]
    receipts = [dict(path=f, sha256=evaluation[key],
                     matches_deployment=sha(git('show', before+':'+f)) == evaluation[key])
                for f, key in evaluation_files]
    assert all(item['matches_deployment'] for item in receipts)
    files = []
    for path in FILES:
        a, b = (git('show', rev+':'+path) for rev in (before, after))
        files.append(dict(path=path, r11_deployment_sha256=sha(a), a_sha256=sha(b), same=a==b,
                          diff=git('diff', '--unified=2', before, after, '--', path).decode()))
    semantic_same = []
    for path, names in [('python/pvz_rollout_policy.py', ['RolloutNet', 'features']),
                        ('python/pvz_frame_store.py', ['encode_frames', 'encode_student_frames',
                                                     'target_values']),
                        ('python/pvz_value_net.py', ['featurize_numpy', '_col_of'])]:
        a, b = (git('show', rev+':'+path).decode() for rev in (before, after))
        for name in names:
            semantic_same.append(dict(path=path, symbol=name,
                                      same_ast=function(a, name)==function(b, name)))
    assert all(row['same_ast'] for row in semantic_same)
    output = ROOT/'MEMORY/docs/task96-code-provenance.json'
    result = dict(task=96, r11_deployment_revision=git('rev-parse', before).decode().strip(),
                  a_revision=git('rev-parse', after).decode().strip(),
                  deployment_log=str(LOGS/'qc_setup.log'), deployment_evidence=evidence,
                  training_receipt=training, evaluation_receipts=receipts,
                  files=files, selected_semantics=semantic_same,
                  inference='Deployment log identifies 4f4d217b during r11, corroborated by three evaluation source hashes. Trainer bytes are identical to A. This is stronger than the CQ HEAD reflog; an unrecorded per-file override during training cannot be excluded without a per-run receipt.')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=1)+'\n')
    # Read only the small corpus index and stat its referenced segments. CodeBuddy
    # already hashed all 9.6GB; repeating that I/O would add no new evidence.
    audit=json.loads((ROOT/'MEMORY/docs/task96-existing-runs.json').read_text())
    recipe=audit['completed_existing_runs'][0]['recipe']
    a_root=Path(recipe['data']); r11_root=Path(json.loads((r11/'recipe.json').read_text())['data'])
    index=a_root/'packs/7d15b915a2192805.npz'
    a_output=a_root.parent/'variant_a/student'
    row_files=[r11/'student/source_rows.json', a_output/'source_rows.json']
    reports=[json.loads(path.read_text()) for path in row_files]
    assert reports[0]==reports[1], 'per-label source rows changed'
    rows=[]
    with np.load(index, allow_pickle=False) as data:
        segments=[str(name) for name in data['segments']]
    for segment in segments:
        a_path=a_root/'packs'/segment; old_path=r11_root/'packs'/segment
        assert a_path.is_file() and old_path.is_file()
        a=a_path.stat(); old=old_path.stat()
        with np.load(a_path,allow_pickle=False) as stored:
            options=json.loads(str(stored['meta']))
            assert len(options['labels'])==1
            label=options['labels'][0]
            assert int(stored['rows'])==reports[0]['sources'][label]['rows']
        rows.append(dict(segment=segment,label=label,source_counts=reports[0]['sources'][label],r11_exists=True,a_exists=True,r11_bytes=old.st_size,
                         a_bytes=a.st_size,r11_mtime_ns=old.st_mtime_ns,a_mtime_ns=a.st_mtime_ns,
                         same_inode=(old.st_dev,old.st_ino)==(a.st_dev,a.st_ino),
                         r11_mtime_after_training=old.st_mtime>training['train_started']))
    assert set(row['label'] for row in rows)==set(reports[0]['sources'])
    counts=dict(segments=len(rows),missing=0,size_differences=sum(r['r11_bytes']!=r['a_bytes'] for r in rows),
                same_inode=sum(r['same_inode'] for r in rows),
                r11_mtime_after_training=sum(r['r11_mtime_after_training'] for r in rows))
    metadata=dict(task=96,scope='Metadata-only follow-up; CodeBuddy already compared all 348 segment SHA256, so no repeat full-data hashing.',
                  index=str(index),index_sha256=sha(index.read_bytes()),r11_root=str(r11_root),segments=rows,
                  source_rows=[dict(path=str(path),sha256=sha(path.read_bytes())) for path in row_files],
                  source_rows_equal=True,total_rows=reports[0]['rows'],
                  counts=counts,content_hash_evidence='CodeBuddy result committed in 9ffc611a; 348/348 referenced segment SHA256 identical. No per-segment digest manifest was committed then.',
                  limitation='Current surviving files and pre-training mtimes support unchanged inputs; there is no immutable per-run content manifest recorded at r11 launch.')
    (ROOT/'MEMORY/docs/task96-data-metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=1)+'\n')
    print(json.dumps(dict(counts=counts,trainer_bytes_identical=files[0]['same'],evaluation_receipts=receipts)))


if __name__ == '__main__':
    main()
