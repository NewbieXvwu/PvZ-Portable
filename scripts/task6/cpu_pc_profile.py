"""Summarize cpu_pc_sample.c profiles using executable symbols and per-process ELF maps."""
import argparse
from array import array
from bisect import bisect_right
from collections import Counter
import json
from pathlib import Path
import subprocess


def summarize(directory, executable):
    executable = executable.resolve()
    with executable.open('rb') as stream:
        header = stream.read(20)
    if header[:4] != b'\x7fELF':
        raise ValueError('ELF executable required')
    pie = int.from_bytes(header[16:18], 'little' if header[5] == 1 else 'big') == 3
    symbols = []
    for line in subprocess.check_output(['nm','-n','-C','--defined-only',str(executable)],text=True).splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[1] in 'tTwW':
            symbols.append((int(parts[0],16), parts[2]))
    symbols.sort()
    addresses = [a for a, _ in symbols]
    modules, functions = Counter(), Counter()
    files = sorted(directory.glob('*.pcs'))
    if not files:
        raise ValueError('no completed profiles')
    total = 0
    truncated = []
    for path in files:
        maps = []
        for line in path.with_suffix('.maps').read_text().splitlines():
            parts = line.split(None, 5)
            if len(parts) < 6 or 'x' not in parts[1]:
                continue
            start, end = (int(p,16) for p in parts[0].split('-'))
            maps.append((start, end, start-int(parts[2],16), parts[5]))
        maps.sort()
        starts = [m[0] for m in maps]
        pcs = array('Q')
        pcs.frombytes(path.read_bytes())
        if len(pcs) == 1 << 20:
            truncated.append(path.name)
        for pc in pcs:
            index = bisect_right(starts, pc)-1
            if index < 0 or pc >= maps[index][1]:
                modules['unmapped'] += 1
                continue
            start, end, base, filename = maps[index]
            modules[filename] += 1
            if filename == str(executable):
                symbol = bisect_right(addresses, pc-(base if pie else 0))-1
                functions[symbols[symbol][1] if symbol >= 0 else 'before first symbol'] += 1
        total += len(pcs)
    if not total:
        raise ValueError('empty CPU profiles')
    ranked = lambda counts: [{'name':name,'samples':n,'percent_native_cpu':n/total*100}
                             for name,n in counts.most_common()]
    return {'profiles':len(files),'samples':total,'interval_cpu_us':10000,
            'method':'ITIMER_PROF program counters; executable functions via nm/ELF mappings; C++ processes only',
            'executable':str(executable),'truncated_profiles':truncated,
            'modules':ranked(modules),'executable_functions':ranked(functions)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    result = summarize(args.directory,args.executable)
    text = json.dumps(result,indent=2)
    if args.out:
        args.out.write_text(text+'\n')
    print(text)


if __name__ == '__main__':
    main()
