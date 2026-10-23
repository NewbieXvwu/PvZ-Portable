#!/usr/bin/env python3
"""Build task35's isolated C++ lockstep prototype from a committed source export.

The export/build/binary live in ignored artifacts/perf/batch_prototype. No source
or object in the production build is modified. Does not start a game process.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/perf/batch_prototype'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    source = OUT / revision / 'source'
    source.mkdir(parents=True, exist_ok=True)
    archive = OUT / revision / 'source.tar'
    subprocess.run(['git', '-C', str(ROOT), 'archive', '--format=tar', '-o', str(archive), revision], check=True)
    with tarfile.open(archive) as stream:
        stream.extractall(source, filter='data')
    header = source / 'src/EnvironmentRollout.h'
    text = header.read_text()
    assert text.count('private:') == 1
    header.write_text(text.replace('private:', 'public: // isolated prototype access; layout unchanged'))
    rollout = source / 'src/EnvironmentRollout.cpp'
    text = rollout.read_text()
    marker = '\tConvolve(board, Channels + (v2 ? 9 : v2lite ? 4 : 0), 3, 3, 0, hidden, true);'
    end = '\tConvolve(scratch, width, 1, 5, 4, hidden, false);'
    first, last = text.index(marker), text.index(end) + len(end)
    original = text[first:last]
    hook = ('\tif (gPrototypeHidden) hidden = *gPrototypeHidden;\n\telse\n\t{\n' + original + '\n\t}')
    text = text[:first] + hook + text[last:]
    text = text.replace('void EnvironmentRollout::Encode(',
        'extern thread_local const std::vector<float>* gPrototypeHidden;\n\nvoid EnvironmentRollout::Encode(', 1)
    rollout.write_text(text)
    main_file = source / 'src/main.cpp'
    text = main_file.read_text()
    marker = 'static void RunEnvironment(LawnApp* app)'
    assert text.count(marker) == 1
    helper = (ROOT / 'scripts/prototypes/batch_rollout.inc').read_text()
    text = text.replace(marker, helper + '\n' + marker, 1)
    marker = '\t\telse if (command == "ROLLOUT")'
    assert text.count(marker) == 1
    branch = ('\t\telse if (command == "PROTOTYPE_BATCH")\n\t\t{\n'
              '\t\t\ttry { PrototypeBatchCommand(app, rollout, snapshots, input); }\n'
              '\t\t\tcatch (const std::exception& error) {\n'
              '\t\t\t\tstd::cerr << error.what() << std::endl;\n'
              '\t\t\t\tstd::cout << "PVZENV {\\"protocol_version\\":" << kEnvironmentProtocolVersion << ",\\"ok\\":false}" << std::endl;\n'
              '\t\t\t}\n\t\t\tcontinue;\n\t\t}\n')
    text = text.replace(marker, branch + marker, 1)
    # Measure the native arm with the same process CPU clock as the lockstep arm.
    # Production uses steady_clock for these diagnostics; only the exported copy changes.
    first = text.index(marker)
    last = text.index('\t\telse if (command == "RESET_V2")', first)
    native = text[first:last]
    native = native.replace('const auto start = std::chrono::steady_clock::now();',
                            'const auto start = PrototypeCpu();')
    native = native.replace('std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()',
                            '(PrototypeCpu() - start)')
    restore = '\t\t\tif (ok && rootId) ok = RestoreEnvironmentSnapshot(app, snapshots.at(rootId));'
    assert native.count(restore) == 1
    native = native.replace(restore, '\t\t\tconst auto snapshotStart = PrototypeCpu();\n' + restore +
                            '\n\t\t\tconst double snapshotCpu = PrototypeCpu() - snapshotStart;')
    output = '\t\t\t\t<< ",\\"net_calls\\":" << rollout.NetworkCalls()'
    assert native.count(output) == 1
    native = native.replace(output,
        '\t\t\t\t<< ",\\"network_cpu_seconds\\":" << networkSeconds'
        '\n\t\t\t\t<< ",\\"simulation_cpu_seconds\\":" << simulationSeconds'
        '\n\t\t\t\t<< ",\\"snapshot_cpu_seconds\\":" << snapshotCpu\n' + output)
    main_file.write_text(text[:first] + native + text[last:])
    build = OUT / revision / 'build'
    subprocess.run(['cmake', '-S', str(source), '-B', str(build), '-G', 'Ninja',
                    '-DCMAKE_BUILD_TYPE=Release', '-DFETCHCONTENT_UPDATES_DISCONNECTED=ON',
                    '-DFETCHCONTENT_SOURCE_DIR_MIMALLOC=' + str(ROOT / 'build/_deps/mimalloc-src')], check=True)
    subprocess.run(['cmake', '--build', str(build), '--target', 'pvz-portable', '--parallel', '2'], check=True)
    (OUT / 'build.json').write_text(json.dumps({'revision': revision, 'binary': str(build / 'pvz-portable'),
        'source': str(source), 'helper_sha256': hashlib.sha256(helper.encode()).hexdigest(),
        'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=1) + '\n')
    print(build / 'pvz-portable')


if __name__ == '__main__':
    main()
