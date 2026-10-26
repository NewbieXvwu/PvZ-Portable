"""Audit CodeBuddy's completed task96 runs without starting another training job."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path('/tmp/exp96/detexp')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name,str(tensor.dtype),list(tensor.shape)]).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main():
    import torch
    documents=[];models=[]
    for run in ('run1','run2'):
        folder=RUNS/run
        recipe=json.loads((folder/'recipe.resolved.json').read_text())
        metrics=json.loads((folder/'metrics.json').read_text())
        model=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
        models.append(model['state_dict'])
        documents.append(dict(run=run,folder=str(folder),seed=recipe['seed'],
            epochs=recipe['epochs'],model_file_sha256=sha(folder/'model.pt'),
            tensor_sha256=tensor_digest(models[-1]),
            source_rows_sha256=sha(folder/'source_rows.json'),
            recipe={k:v for k,v in recipe.items() if k!='out'},metrics=metrics))
    assert documents[0]['recipe']==documents[1]['recipe']
    assert set(models[0])==set(models[1])
    differences=[]
    for name,tensor in models[0].items():
        other=models[1][name]
        assert tensor.shape==other.shape and tensor.dtype==other.dtype
        different=int((tensor!=other).sum())
        differences.append(dict(name=name,elements=tensor.numel(),different=different,
            max_abs=float((tensor-other).abs().max())))
    files=['scripts/train_rollout_policy.py','scripts/pack_training_rows.py',
           'python/pvz_rollout_policy.py','python/pvz_frame_store.py',
           'python/pvz_training_pack.py','python/pvz_recipe.py','python/pvz_student_observation.py']
    source=[]
    for name in files:
        expected=subprocess.check_output(['git','-C',str(ROOT),'show','bec2b9e6:'+name])
        path=Path('/tmp/exp96')/name
        source.append(dict(path=name,sha256=sha(path),
            matches_bec2b9e6=path.read_bytes()==expected))
    old_trainer=subprocess.check_output(['git','-C',str(ROOT),'show',
        '930d776d:scripts/train_rollout_policy.py'],text=True)
    old_model=subprocess.check_output(['git','-C',str(ROOT),'show',
        '930d776d:python/pvz_rollout_policy.py'],text=True)
    r11=Path('/home/newbiexvwu/pvz-logs/curriculum_state.pipeline/round_000011')
    r11_recipe=json.loads((r11/'student/recipe.resolved.json').read_text())
    report=dict(task=96,schema_version=1,completed_existing_runs=documents,
        recipe_equal_except_output=True,tensor_comparison=differences,
        different_tensors=sum(d['different']>0 for d in differences),
        total_tensors=len(differences),source_snapshot=source,
        runtime_provenance_correction=dict(
            rejected_revision='930d776d',old_trainer_has_recipe='--recipe' in old_trainer,
            old_model_has_identity_head="packet_head" in old_model,
            r11_packet_head=r11_recipe['packet_head'],r11_uses_resolved_recipe=True,
            explanation='The main checkout HEAD is not runtime provenance. r11 uses recipe and identity-head functionality absent at 930d776d; its exact runtime revision remains unproven.'),
        trainer_observations=dict(dataloader_used=False,
            shuffle='CPU torch.randperm after torch.manual_seed(recipe.seed)',
            validation='sorted seed set shuffled with private random.Random(validation.seed)',
            precompute_workers='Completed runs read existing packs; this option is not a DataLoader worker count.'),
        limitations=['Both runs finished before this audit; their launch environment and deterministic-algorithm flags were not captured.',
            'Model file hashes include metadata; tensor comparison independently proves weight differences.',
            'Runtime contention differs; the two elapsed times do not measure deterministic-mode overhead.',
            'Recipe equality and nominal pack name do not alone establish a content-immutable training input.'],
        next='Locate runtime code provenance and test deterministic algorithms on a bounded frozen-pack diagnostic, yielding GPU to qc.')
    output=ROOT/'MEMORY/docs/task96-existing-runs.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=1)+'\n')
    print(json.dumps(dict(output=str(output),different_tensors=report['different_tensors'],
        total_tensors=report['total_tensors'],tensor_sha256=[d['tensor_sha256'] for d in documents])))


if __name__=='__main__':main()
