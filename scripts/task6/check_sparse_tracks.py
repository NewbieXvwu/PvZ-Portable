"""Link/run the focused track-cache contracts against an independently built simulator.

Configure the CPU build with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON first. This reuses
its objects and compiler/link flags, replacing only main; it never rewrites pvz-portable.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    build = args.build_dir.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    commands = json.loads((build/'compile_commands.json').read_text())
    entry = next(c for c in commands if Path(c['file']).resolve() == ROOT/'src/main.cpp')
    compile_args = entry.get('arguments') or shlex.split(entry['command'])
    assert compile_args.count('-o') == 1
    compile_args[compile_args.index('-o')+1] = str(out/'contracts.o')
    source_indices = [i for i,v in enumerate(compile_args) if v == entry['file']]
    assert len(source_indices) == 1
    compile_args[source_indices[0]] = str(ROOT/'scripts/task6/test_sparse_tracks.cpp')
    link_args = shlex.split((build/'CMakeFiles/pvz-portable.dir/link.txt').read_text())
    main_indices = [i for i,v in enumerate(link_args) if v.endswith('/src/main.cpp.o')]
    assert len(main_indices) == 1 and link_args.count('-o') == 1
    link_args[main_indices[0]] = str(out/'contracts.o')
    link_args[link_args.index('-o')+1] = str(out/'contracts')
    steps = [compile_args,link_args,[str(out/'contracts')]]
    (out/'commands.json').write_text(json.dumps({'cwd':str(build),'commands':steps},indent=2)+'\n')
    env = {**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    for command in steps:
        subprocess.run(command,cwd=build,env=env,check=True)


if __name__ == '__main__':
    main()
