"""task81f 本机缓存生成与审计；不进入用户包，不改引擎。

使用同一 ELF 的调试符号对原生函数做预加载，覆盖只 reset 不会触发的片尾等定义。
--symbols 必须与引擎 .text/.rodata 完全一致。GDB 只用于本机生成及分阶段诊断。
用户包和沙箱运行不依赖 GDB。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "python"))
from llmplay.resources import sha256


def definitions():
    expected = set()
    for filename, pattern in (
            ("Reanimator.cpp", r'\.mReanimFileName\s*=\s*"([^"]+)"'),
            ("PvzpParticle.cpp", r'\.mParticleFileName\s*=\s*"([^"]+)"'),
            ("Trail.cpp", r'"(particles/[^"\n]+\.trail)"')):
        expected.update("compiled/" + name + ".compiled" for name in
                        re.findall(pattern, (ROOT / "src/PvzpLib" / filename).read_text()))
    return expected


def audit_cache(directory):
    directory = Path(directory) / "cache64"
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*.compiled")}
    expected = definitions()
    if expected - actual:
        raise ValueError("缓存定义缺失：%s" % sorted(expected - actual))
    for name in actual:
        blob = (directory / name).read_bytes()
        cookie, size = struct.unpack("<II", blob[:8])
        if cookie != 0xDEADFED4 or len(zlib.decompress(blob[8:])) != size:
            raise ValueError("缓存 cookie/解压大小不符：%s" % name)
    return dict(expected=len(expected), actual=len(actual), missing=[],
                files={"cache64/" + name: sha256(directory / name) for name in sorted(actual)})


def sections(path):
    blob = Path(path).read_bytes()
    if blob[:6] != b"\x7fELF\x02\x01":
        raise ValueError("仅接受 x86_64 Linux 小端 ELF")
    offset = struct.unpack_from("<Q", blob, 40)[0]
    width, count, strings = struct.unpack_from("<HHH", blob, 58)
    entries = [struct.unpack_from("<IIQQQQIIQQ", blob, offset+i*width) for i in range(count)]
    header = entries[strings]
    names = blob[header[4]:header[4]+header[5]]
    result = {}
    for entry in entries:
        name = names[entry[0]:].split(b"\0", 1)[0].decode()
        if name in (".text", ".rodata"):
            result[name] = hashlib.sha256(blob[entry[4]:entry[4]+entry[5]]).hexdigest()
    return result


def symbol_addresses(symbols):
    output = subprocess.check_output(["nm", "-C", str(symbols)], text=True)
    patterns = dict(ready=r"RunEnvironment\(LawnApp\*\)$",
                    preload=r"ReanimatorEnsureDefinitionLoaded\(.*\)(?: \[clone .constprop.0\])?$",
                    compile=r"DefinitionCompileAndLoad\(.*\)(?: \[clone .isra.0\])?$",
                    images=r"Sexy::ResourceManager::DoLoadImage\(.*\)$",
                    fonts=r"Sexy::ResourceManager::DoLoadFont\(.*\)$")
    found = {}
    for name, pattern in patterns.items():
        lines = [line for line in output.splitlines() if re.search(pattern, line)]
        if len(lines) != 1:
            raise ValueError("不能唯一定位 %s：%s" % (name, lines))
        found[name] = int(lines[0].split()[0], 16)
    return found


def precompile(executable, resources, cache, symbols, gdb, work):
    if sections(executable) != sections(symbols):
        raise ValueError("调试符号的 .text/.rodata 与包内引擎不一致")
    addresses = symbol_addresses(symbols)
    script = work / "precompile.gdb"
    script.write_text("\n".join([
        "set pagination off", "set confirm off", "set debuginfod enabled off",
        "symbol-file " + str(symbols), "break *0x%x" % addresses["ready"], "run", "python",
        "count = int(gdb.parse_and_eval('gReanimatorDefCount'))",
        "for i in range(count):",
        "    gdb.execute('call ((void (*)(int,bool))0x%x)(%%d,true)' %% i, to_string=True)" % addresses["preload"],
        "print('PRECOMPILE OK: %d reanimations' % count)", "end", "quit", ""]))
    with (work / "precompile.txt").open("w") as output:
        subprocess.run([str(gdb), "-nx", "-nh", "-batch", "-x", str(script), "--args",
                        str(executable), "-env", "-resdir", str(resources), "-savedir", str(cache)],
                       stdout=output, stderr=subprocess.STDOUT, check=True)
    if "PRECOMPILE OK:" not in (work / "precompile.txt").read_text():
        raise RuntimeError("未完成原生全定义预加载")
    return addresses


def reset_all(executable, resources, cache, work):
    from pvz_env import PvZEnv, TaskSpec
    from pvz_tasks import deck_for_level, profile_for_deck
    from llmplay.game import load_tasks, task_spec
    plan = json.loads((ROOT / "configs/llmplay_seeds.json").read_text())
    rows = []
    with PvZEnv(resources, executable=executable, save_dir=cache) as env:
        for level in range(1, 51):
            deck = deck_for_level(level)
            spec = TaskSpec(level=level, seed=0, profile=profile_for_deck(deck))
            obs, _ = env.reset(task=spec, deck=deck)
            obs, _, _, _, _ = env.step({"type": "wait", "ticks": 3000})
            rows.append(dict(level=level, tick=obs["tick"], wave=obs["wave"]))
        tasks = load_tasks()
        for task_id in plan["tasks"]:
            spec, deck, seed = task_spec(tasks[task_id], plan["tasks"][task_id][0])
            obs, _ = env.reset(task=spec, deck=deck)
            obs, _, _, _, _ = env.step({"type": "wait", "ticks": 3000})
            rows.append(dict(task_id=task_id, seed=seed, tick=obs["tick"], wave=obs["wave"]))
    (work / "reset-all.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def start_once(executable, resources, save, log):
    save.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stderr:
        start = time.perf_counter()
        process = subprocess.Popen([str(executable), "-env", "-resdir", str(resources), "-savedir", str(save)],
                                   cwd=executable.parent, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=stderr, text=True)
        try:
            for line in process.stdout:
                if line.startswith("PVZENV "):
                    ready = json.loads(line[7:])
                    if not ready.get("ready"):
                        raise RuntimeError("启动未进入 ready")
                    break
            else:
                raise RuntimeError("引擎未发出 ready")
            elapsed = time.perf_counter() - start
            process.communicate("QUIT\n", timeout=30)
            if process.returncode:
                raise RuntimeError("引擎启动退出码 %s" % process.returncode)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    text = log.read_text()
    return dict(seconds=elapsed, compile_count=len(re.findall(r"INFO: compile \d+ ms:", text)),
                compile_ms=sum(map(int, re.findall(r"INFO: compile (\d+) ms:", text))),
                native_loading_ms={key: int(value) for key, value in re.findall(r"loading '([^']+)' (\d+) ms", text)})


def benchmark(executable, resources, cache, work):
    rows = []
    for mode in ("cold", "hot", "precompiled"):
        hot = work / "benchmark-hot"
        if mode == "hot":
            if hot.exists(): shutil.rmtree(hot)
            start_once(executable, resources, hot, work / "hot-prime.stderr")
        for index in range(5):
            save = hot if mode == "hot" else work / ("benchmark-%s-%d" % (mode, index))
            if mode != "hot" and save.exists(): shutil.rmtree(save)
            if mode == "precompiled": shutil.copytree(cache / "cache64", save / "cache64")
            row = start_once(executable, resources, save, work / ("%s-%d.stderr" % (mode, index)))
            if mode != "cold" and row["compile_count"]:
                raise RuntimeError("热/预编译启动仍在编译定义")
            rows.append(dict(mode=mode, index=index, **row))
    result = dict(samples=rows, medians={mode: statistics.median(r["seconds"] for r in rows if r["mode"] == mode)
                                       for mode in ("cold", "hot", "precompiled")})
    (work / "startup-benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("executable", "symbols", "gdb", "resource-dir", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "full-appdata"
    precompile(args.executable, args.resource_dir, cache, args.symbols, args.gdb, args.out)
    rows = reset_all(args.executable, args.resource_dir, cache, args.out)
    audit = audit_cache(cache)
    result = dict(schema="llmplay_cache_audit_v1", engine_sha256=sha256(args.executable),
                  executable_sections=sections(args.executable), cache=audit,
                  levels_reset=50, tasks_reset=len(rows)-50,
                  benchmark=benchmark(args.executable, args.resource_dir, cache, args.out))
    (args.out / "cache-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("cache", "benchmark")}, indent=2))
    print("CACHE OK: %d definitions; medians %s" % (audit["actual"], result["benchmark"]["medians"]))


if __name__ == "__main__":
    main()
