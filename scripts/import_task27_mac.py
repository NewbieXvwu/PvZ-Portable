"""Refresh uploaded breadth-stage Mac rounds, exclusively adding missing frame/meta files."""
import re
from collections import Counter
from task27_common import ROOT, MAIN, OUT, read, write, sha
from import_dw4_transfer import hf_environment, import_archive

PATTERN = re.compile(r'transfer/((?:d(?:1|3|4|6|7|8|9|10)(?:dag\d+[bm]|p\d+m))|(?:n(?:11|12|13|14|16|17|18|19)dag\d+m))\.tgz')
OLD_MAC = re.compile(r'transfer/(z\d+dag\d+[a-z]*)\.tgz')


def refresh():
    from huggingface_hub import HfApi, hf_hub_download
    repo = hf_environment(); info = HfApi().model_info(repo, files_metadata=True)
    target = OUT/'mac_import.json'
    previous = read(target) if target.exists() else {}
    known = {r['label']:r for r in previous.get('rounds',[])}
    journal_path=OUT/'mac_package_journal.json'
    journal=read(journal_path) if journal_path.exists() else {}
    known.update({r['label']:r for r in journal.get('rounds',[])})
    # These packages were already imported and hashed in task23; do not download them again.
    for row in read(MAIN/'.claude/worktrees/dw4/artifacts/task23/mac_import.json')['rounds']:
        known.setdefault(row['label'],row)
    introduced = set(previous.get('introduced_labels',[])) | set(journal.get('introduced_labels',[]));records = []
    destination = MAIN/'artifacts/student/frames_v2'
    existing_old = Counter(p.parent.name for p in destination.glob('*/*/frames.npz')
                           if p.parent.name.startswith('z'))
    for entry in info.siblings:
        match = PATTERN.fullmatch(entry.rfilename) or OLD_MAC.fullmatch(entry.rfilename)
        if not match: continue
        label = match[1]; old = known.get(label)
        remote_sha = entry.lfs.sha256 if entry.lfs else None
        if old and remote_sha and old.get('archive_sha256') == remote_sha:
            result = {**old, 'refresh_added_files':0, 'reused_verified_package':True}
        elif label.startswith('z') and existing_old[label+'_fast_search'] and label+'_fast_search' not in introduced:
            result = dict(label=label,remote_path=entry.rfilename,refresh_added_files=0,
                          archives=existing_old[label+'_fast_search'],
                          already_present_old_round=True,remote_sha256=remote_sha,
                          note='Existing old high-difficulty round: no re-import or expansion of its frozen selection.')
        else:
            if label.startswith('z') and not existing_old[label+'_fast_search']:
                introduced.add(label+'_fast_search')
                write(journal_path,dict(introduced_labels=sorted(introduced),rounds=list(known.values())))
            path = __import__('pathlib').Path(hf_hub_download(repo,entry.rfilename,revision=info.sha,
                                                           local_dir=OUT/'hf_mac'))
            if remote_sha: assert sha(path) == remote_sha
            result = dict(label=label,remote_path=entry.rfilename,
                          **import_archive(path,destination,label))
            result['refresh_added_files'] = result['added_files']
            if result['added_files']: introduced.add(label+'_fast_search')
        known[label]=result
        write(journal_path,dict(introduced_labels=sorted(introduced),rounds=list(known.values())))
        records.append(result)
        print({'label':label,'archives':result['archives'],'added':result['refresh_added_files']},flush=True)
    result = dict(repo=repo,revision=info.sha,destination=str(destination),rounds=records,
                  introduced_labels=sorted(introduced),
                  package_scope='All uploaded d*/n* breadth Mac rounds, plus previously unimported z* rounds; v2fill is legacy feature backfill, not a new round.',
                  added_files_this_refresh=sum(r['refresh_added_files'] for r in records))
    write(target,result)
    return result


if __name__ == '__main__':
    refresh()
