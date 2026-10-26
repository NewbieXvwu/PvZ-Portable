"""同一未修改 ELF 的启动阶段诊断；用户包不携带此脚本。

总启动墙时由 cache_prepare.py 无调试器计时；此处单独用匹配符号设置断点，读取目标
线程 /proc/.../schedstat CPU 纳秒（不含 GDB 等待时间）。断点有开销，阶段数只作归因，
不拿调试器下的墙时作为加速结果。compile 包括其内部引用图片/字体的时间，不可简单相加。
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import re
import subprocess
from pathlib import Path

from cache_prepare import sections, symbol_addresses

GDB_SCRIPT = r'''
import gdb, json
from pathlib import Path
config = json.loads(Path(CONFIG_PATH).read_text())
stats = {key: dict(cpu_ms=0.0, calls=0) for key in ('definitions', 'images', 'fonts', 'main_pak')}
def cpu():
    pid, tid, _ = gdb.selected_thread().ptid
    return int(Path('/proc/%s/task/%s/schedstat' % (pid, tid)).read_text().split()[0])
class Finished(gdb.FinishBreakpoint):
    def __init__(self, name):
        super().__init__(gdb.newest_frame(), internal=True)
        self.silent = True
        self.name, self.started = name, cpu()
    def stop(self):
        stats[self.name]['cpu_ms'] += (cpu()-self.started)/1e6
        stats[self.name]['calls'] += 1
        return False
class Started(gdb.Breakpoint):
    def __init__(self, name, address):
        super().__init__('*0x%x' % address, internal=True)
        self.silent, self.name = True, name
    def stop(self):
        Finished(self.name)
        return False
for name in ('compile', 'images', 'fonts'):
    Started('definitions' if name == 'compile' else name, config['addresses'][name])
class Pak(gdb.Breakpoint):
    def __init__(self, line, begin):
        _, locations = gdb.decode_line('PakInterface.cpp:%s' % line)
        super().__init__('*0x%x' % locations[0].pc, internal=True)
        self.silent, self.begin = True, begin
    def stop(self):
        if self.begin:
            config['pak_start'] = cpu()
        else:
            stats['main_pak']['cpu_ms'] += (cpu()-config['pak_start'])/1e6
            stats['main_pak']['calls'] += 1
        return False
Pak(160, True)
Pak(283, False)
gdb.Breakpoint('*0x%x' % config['addresses']['ready'], internal=True)
gdb.execute('run', to_string=True)
Path(config['output']).write_text(json.dumps(stats, indent=2)+'\n')
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("executable", "symbols", "gdb", "resource-dir", "cache-dir", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if sections(args.executable) != sections(args.symbols):
        raise ValueError("符号与引擎不同")
    args.out.mkdir(parents=True, exist_ok=True)
    addresses = symbol_addresses(args.symbols)
    samples = []
    for mode in ("cold", "precompiled"):
        for index in range(5):
            save = args.out / (mode+"-"+str(index))
            if save.exists(): shutil.rmtree(save)
            if mode == "precompiled": shutil.copytree(args.cache_dir / "cache64", save / "cache64")
            config = args.out / "config.json"
            output = args.out / (mode+"-"+str(index)+".json")
            config.write_text(json.dumps(dict(addresses=addresses, output=str(output))))
            script = args.out / "profile.gdb"
            script.write_text("\n".join([
                "set pagination off", "set confirm off", "set debuginfod enabled off",
                "symbol-file "+str(args.symbols), "python", "CONFIG_PATH = "+repr(str(config)),
                GDB_SCRIPT, "end", "quit", ""]))
            with (args.out / (mode+"-"+str(index)+".txt")).open("w") as log:
                subprocess.run([str(args.gdb), "-nx", "-nh", "-batch", "-x", str(script),
                                "--args", str(args.executable), "-env", "-resdir", str(args.resource_dir),
                                "-savedir", str(save)], stdout=log, stderr=subprocess.STDOUT, check=True)
            row = json.loads(output.read_text())
            if row["main_pak"]["calls"] != 1 or row["images"]["calls"] < 1 or row["fonts"]["calls"] < 1:
                raise ValueError("阶段断点未覆盖目标调用")
            native_compiles = len(re.findall(r"INFO: compile \d+ ms:",
                                            (args.out / (mode+"-"+str(index)+".txt")).read_text()))
            if mode == "precompiled" and native_compiles:
                raise ValueError("预编译启动仍编译")
            samples.append(dict(mode=mode, index=index, native_compiles=native_compiles, phases=row))
            print(mode, index, row, flush=True)
    report = dict(metric="target_thread_scheduler_cpu_ms_under_gdb; diagnostic_only",
                  note="definitions = load compiled cache or compile XML, includes nested image/font lookups; phases are not additive",
                  samples=samples, medians={mode: {
                      name: statistics.median(row["phases"][name]["cpu_ms"] for row in samples if row["mode"] == mode)
                      for name in ("definitions", "images", "fonts", "main_pak")}
                      for mode in ("cold", "precompiled")})
    (args.out / "phase-profile.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
