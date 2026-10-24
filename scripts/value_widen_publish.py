"""Upload a candidate value checkpoint to HF, then verify downloaded bytes.

Reads HF_TOKEN/PVZ_HF_REPO from the environment. No secrets, model files or data
shards are written into git; the optional receipt contains only public metadata.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--audit-manifest',type=Path,default=None);p.add_argument('--verify-dir',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True);p.add_argument('--repo',default=os.environ.get('PVZ_HF_REPO'))
    p.add_argument('--path-in-repo',default='transfer/value_v4.pt')
    p.add_argument('--message',default='task30: candidate value v4, versioned cross-zone encoding')
    a=p.parse_args()
    if not a.repo:p.error('--repo or $PVZ_HF_REPO is required')
    import torch
    from huggingface_hub import HfApi,hf_hub_download
    from pvz_value_net import load_value_net
    net=load_value_net(a.checkpoint)
    if net.encoder_version!=2:raise ValueError('expected candidate with value encoder v2')
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    audit_checked=False
    if a.audit_manifest:
        audit={g['seed'] for g in json.loads(a.audit_manifest.read_text())}
        if set(ck['trained_on'])&audit:raise ValueError('checkpoint saw external audit seeds')
        audit_checked=True
    digest=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest();token=os.environ.get('HF_TOKEN')
    commit=HfApi(token=token).upload_file(repo_id=a.repo,repo_type='model',path_or_fileobj=str(a.checkpoint),
        path_in_repo=a.path_in_repo,commit_message=a.message)
    downloaded=Path(hf_hub_download(repo_id=a.repo,repo_type='model',filename=a.path_in_repo,
        revision=commit.oid,local_dir=str(a.verify_dir),token=token))
    if hashlib.sha256(downloaded.read_bytes()).hexdigest()!=digest:raise ValueError('HF roundtrip checksum mismatch')
    receipt=dict(repo=a.repo,path=a.path_in_repo,revision=commit.oid,url=str(commit.commit_url),sha256=digest,
        bytes=a.checkpoint.stat().st_size,verified_download=True,encoder=ck['encoder'],default_changed=False,
        audit_seeds_checked=audit_checked)
    a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps(receipt,indent=1))
    print(json.dumps(receipt,indent=1))
if __name__=='__main__':main()
