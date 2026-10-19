"""Controller for distributed collection: one Mac drives every friend machine over ssh.

Machines live in ~/.pvz-dist/machines.json, the segment ledger in ~/.pvz-dist/ledger.json.
Offsets 16384-65535 of day_l7_z17_train belong to this ledger (DESIGN.md); segments never overlap.

    python scripts/dist/controller.py status
    python scripts/dist/controller.py bootstrap pvz-f1
    python scripts/dist/controller.py push pvz-f1 --all
    python scripts/dist/controller.py assign pvz-f1 artifacts/student/v29/model.pt 32 \\
        --label z17distproto --offset 14736 --start
    python scripts/dist/controller.py collect --label z17distproto
    python scripts/dist/controller.py web pvz-f1

Nothing here needs a token on the friend machine: code, models, resources and tasks go out over
ssh, results come back the same way, and only this Mac talks to Hugging Face.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = Path.home() / ".pvz-dist"
MACHINES = STATE / "machines.json"
LEDGER = STATE / "ledger.json"
RESULTS = STATE / "results"
WORKER_ROOT = "pvz-worker"
DIST_FIRST = 16384
DIST_LAST = 65536
DEFAULT_TASK = "day_l7_z17_train"
def _main_repo() -> Path:
    """The worktree shares .git with the main checkout, where the untracked artifacts live."""
    proc = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute",
                           "--git-common-dir"], capture_output=True, text=True)
    if proc.returncode == 0 and proc.stdout.strip():
        return Path(proc.stdout.strip()).parent
    return ROOT


MAIN = _main_repo()


def artifact(relative: str) -> Path:
    for base in (ROOT, MAIN):
        path = base / relative
        if path.exists():
            return path
    return ROOT / relative


DEFAULT_MANIFEST = artifact("artifacts/stage4/day_l7_hard.json")
VALUE = artifact("artifacts/hard_mac/value_v3/model.pt")
STATIC_VALUE = artifact("artifacts/hard_mac/value_branch1/model.pt")
RESOURCES = Path("/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN")
FLAGS_FILE = artifact("artifacts/hard_mac/round15_flags")


def now() -> str:
    return datetime.now().strftime("%F %T")


def say(msg: str) -> None:
    print(f"{now()} {msg}", flush=True)


def load_json(path: Path, default):
    if not path.is_file():
        return default
    return json.loads(path.read_text())


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")


def machines() -> dict:
    data = load_json(MACHINES, None)
    if data is None:
        data = {"pvz-f1": {"host": "pvz-f1", "cores": 10, "gpu": False, "enabled": True,
                           "note": "friend Mac (Apple M5, 32 GB), ssh alias in ~/.ssh/config"}}
        save_json(MACHINES, data)
        say(f"created {MACHINES}")
    return data


def machine(name: str) -> dict:
    table = machines()
    if name not in table:
        raise SystemExit(f"unknown machine {name!r}; known: {', '.join(table)}")
    return table[name]


def ssh_args(entry: dict) -> list:
    # UU 远程的转发会在网络抖动时缓冲几十秒（实测 SYN_ACK/主机密钥晚到 ~19 s），
    # 15 秒的 ConnectTimeout 会在对端回答之前就把连接掐掉。给足时间，失败由 --wait 重试兜底。
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=180",
            "-o", "ServerAliveInterval=60", "-o", "ServerAliveCountMax=10",
            "-o", "TCPKeepAlive=yes", entry["host"]]


def ssh(entry: dict, command: str, check: bool = True, stdin: str | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(ssh_args(entry) + [command], input=stdin, text=True,
                          capture_output=True)
    if check and proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"ssh {entry['host']} failed: {command}")
    return proc


def rsync_try(entry: dict, source: str, target: str, extra: list | None = None) -> bool:
    # macOS ships openrsync (rsync 2.6.9-compatible): --partial/--inplace exist, --append-verify does
    # NOT (it prints the usage text and fails). Large files over a flaky tunnel are therefore sent by
    # repeated --partial runs: whatever arrived stays on the far side and the next run replaces it.
    cmd = ["rsync", "-az", "--stats", "--partial", "--inplace", "--timeout=600", "-e",
           "ssh -o BatchMode=yes -o ConnectTimeout=180 -o ServerAliveInterval=60 -o ServerAliveCountMax=10"]
    cmd += (extra or []) + [source, f"{entry['host']}:{target}"]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        say(f"rsync to {entry['host']} failed")
        return False
    tail = [line.strip() for line in proc.stdout.splitlines() if line.startswith("Number of")]
    say(f"rsync {source} -> {entry['host']}:{target} ({' '.join(tail)})")
    return True


def rsync(entry: dict, source: str, target: str, extra: list | None = None) -> None:
    if not rsync_try(entry, source, target, extra):
        raise SystemExit(f"rsync to {entry['host']} failed")


def rsync_retry(entry: dict, source: str, target: str, extra: list | None = None,
                tries: int = 3, delay: int = 10) -> bool:
    """The tunnel drops mid-transfer often; retry the same --partial run instead of failing the job."""
    for attempt in range(1, tries + 1):
        if rsync_try(entry, source, target, extra):
            return True
        if attempt < tries:
            say(f"retry {attempt}/{tries} in {delay}s")
            time.sleep(delay)
    return False


# --------------------------------------------------------------------------- push
def push_code(entry: dict, commit: str | None) -> None:
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    if dirty and not commit:
        say("WARNING: uncommitted changes are NOT pushed (git archive HEAD):")
        for line in dirty.splitlines()[:5]:
            say("  " + line)
    staging = Path(tempfile.mkdtemp(prefix="pvz-dist-push-"))
    try:
        ref = commit or "HEAD"
        with (staging / "src.tar").open("wb") as fh:
            proc = subprocess.run(["git", "-C", str(ROOT), "archive", ref], stdout=fh)
        if proc.returncode != 0:
            raise SystemExit(f"git archive {ref} failed (commit everything first)")
        with tarfile.open(staging / "src.tar") as tar:
            tar.extractall(staging / "tree")
        rsync(entry, str(staging / "tree") + "/", f"~/{WORKER_ROOT}/repo/",
              ["--delete", "--exclude", "build/", "--exclude", ".git/", "--exclude", ".pvz-pushed"])
        # Marker: bootstrap.sh sees this and does not try to reach GitHub (unreachable from China).
        ssh(entry, f"touch {WORKER_ROOT}/repo/.pvz-pushed")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def push_models(entry: dict, student: Path | None) -> None:
    ssh(entry, f"mkdir -p {WORKER_ROOT}/models")
    for path, name in ((VALUE, "value_v3.pt"), (STATIC_VALUE, "value_branch1.pt")):
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        rsync(entry, str(path), f"~/{WORKER_ROOT}/models/{name}")
    if student:
        if not student.is_file():
            raise SystemExit(f"missing student {student}")
        rsync(entry, str(student), f"~/{WORKER_ROOT}/models/{student.parent.name}.pt")
        rsync(entry, str(student), f"~/{WORKER_ROOT}/models/selfcheck.pt")


def push_resources(entry: dict) -> None:
    if not (RESOURCES / "main.pak").is_file():
        raise SystemExit(f"missing {RESOURCES}/main.pak")
    ssh(entry, f"mkdir -p {WORKER_ROOT}/resources")
    rsync(entry, str(RESOURCES / "main.pak"), f"~/{WORKER_ROOT}/resources/")
    rsync(entry, str(RESOURCES / "properties") + "/", f"~/{WORKER_ROOT}/resources/properties/")


def push_manifest(entry: dict) -> None:
    if not DEFAULT_MANIFEST.is_file():
        raise SystemExit(f"missing {DEFAULT_MANIFEST}")
    ssh(entry, f"mkdir -p {WORKER_ROOT}/manifest")
    rsync(entry, str(DEFAULT_MANIFEST), f"~/{WORKER_ROOT}/manifest/day_l7_hard.json")


def rebuild(entry: dict) -> None:
    jobs = max(1, int(entry.get("cores", 4)))
    say(f"rebuild on {entry['host']} (-j{jobs})")
    cmd = (f"cd {WORKER_ROOT}/repo && PATH=$HOME/{WORKER_ROOT}/env/bin:$PATH "
           f"cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release "
           f"-DCMAKE_PREFIX_PATH=$HOME/{WORKER_ROOT}/env -DPVZ_ROLLOUT_BLAS=ON "
           f">$HOME/{WORKER_ROOT}/logs/cmake-configure.log 2>&1 && "
           f"cmake --build build -j{jobs} >$HOME/{WORKER_ROOT}/logs/cmake-build.log 2>&1 && "
           f"echo BUILD_OK")
    proc = ssh(entry, cmd, check=False)
    if "BUILD_OK" not in proc.stdout:
        sys.stderr.write(proc.stdout[-2000:] + proc.stderr[-2000:])
        raise SystemExit("remote build failed (see ~/pvz-worker/logs/cmake-build.log)")


def cmd_push(args) -> None:
    entry = machine(args.machine)
    if args.all or args.code:
        push_code(entry, args.commit)
        if args.build or args.all:
            rebuild(entry)
    if args.all or args.models:
        push_models(entry, Path(args.student).expanduser() if args.student else None)
    if args.all or args.resources:
        push_resources(entry)
    if args.all or args.manifest:
        push_manifest(entry)
    say("push done")


def cmd_bootstrap(args) -> None:
    entry = machine(args.machine)
    script = ROOT / "scripts/dist/bootstrap.sh"
    subprocess.run(["scp", "-q", str(script), f"{entry['host']}:~/bootstrap.sh"], check=True)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    say(f"bootstrap {entry['host']} at {commit[:8]} (this takes a while; log: ~/{WORKER_ROOT}/logs/bootstrap-run.log)")
    push_code(entry, None)          # friend machines never need GitHub
    proc = ssh(entry, f"mkdir -p ~/{WORKER_ROOT}/logs && PVZ_SKIP_GIT=1 PVZ_COMMIT={commit} "
                      f"bash ~/bootstrap.sh 2>&1 | tail -40", check=False)
    print(proc.stdout)
    if proc.returncode != 0:
        raise SystemExit("bootstrap failed")


# --------------------------------------------------------------------------- assign
def teacher_flags() -> list:
    """round15_flags minus the two model paths, plus the 29b DAgger settings."""
    flags = shlex.split(FLAGS_FILE.read_text()) if FLAGS_FILE.is_file() else ["--verify-samples", "9"]
    out, skip = [], False
    for token in flags:
        if skip:
            skip = False
            continue
        if token in ("--value", "--static-value"):
            skip = True
            continue
        out.append(token)
    return out + ["--dagger-beta", "0", "--student-candidates", "2"]


def next_offset(games: int) -> int:
    ledger = load_json(LEDGER, {"segments": []})
    taken = sorted((s["seed_offset"], s["seed_offset"] + s["seeds"]) for s in ledger["segments"])
    cursor = DIST_FIRST
    for start, end in taken:
        if cursor + games <= start:
            return cursor
        cursor = max(cursor, end)
    if cursor + games <= DIST_LAST:
        return cursor
    raise SystemExit(f"no free offset for {games} games in [{DIST_FIRST}, {DIST_LAST})")


def cmd_assign(args) -> None:
    entry = machine(args.machine)
    student = Path(args.student).expanduser().resolve()
    if not student.is_file():
        raise SystemExit(f"missing student {student}")
    offset = args.offset if args.offset is not None else next_offset(args.games)
    if offset < DIST_FIRST and not args.allow_reserved:
        say(f"note: offset {offset} is outside the distributed range [{DIST_FIRST}, {DIST_LAST}) — "
            "allowed only for registered prototypes")
    ledger = load_json(LEDGER, {"segments": []})
    for seg in ledger["segments"]:
        if seg["seed_offset"] < offset + args.games and offset < seg["seed_offset"] + seg["seeds"]:
            if seg["label"] != args.label:
                raise SystemExit(f"segment overlaps {seg['label']} "
                                 f"[{seg['seed_offset']}, {seg['seed_offset'] + seg['seeds']})")
    label = args.label or f"z17dist{offset}"
    name = student.parent.name
    spec = {
        "label": label,
        "manifest": "manifest/day_l7_hard.json",
        "task_id": args.task_id,
        "seed_offset": offset,
        "seeds": args.games,
        "student": f"models/{name}.pt",
        "value": "models/value_v3.pt",
        "static_value": "models/value_branch1.pt",
        "resource_dir": "resources",
        "teacher_flags": teacher_flags(),
        "workers": 1 if args.gpu else (args.workers or max(1, min(int(entry.get("cores", 4)) - 2, 16))),
        "chunk": 1 if args.gpu else (args.workers or max(1, min(int(entry.get("cores", 4)) - 2, 16))),
        "assigned_at": now(),
    }
    if args.gpu:
        spec["gpu"] = True
        if not entry.get("gpu"):
            say(f"note: {args.machine} is not marked \"gpu\": true in machines.json — the worker will "
                "check for build/rollout_cuda_small and refuse to start if it is missing")
    ssh(entry, f"mkdir -p {WORKER_ROOT}/tasks/{label}")
    ssh(entry, f"cat > {WORKER_ROOT}/tasks/{label}/task.json", stdin=json.dumps(spec, indent=1))
    entry_ledger = {"label": label, "machine": args.machine, "student": str(student),
                    "seed_offset": offset, "seeds": args.games, "status": "assigned",
                    "assigned_at": now(), "workers": spec["workers"]}
    ledger["segments"] = [s for s in ledger["segments"] if s["label"] != label] + [entry_ledger]
    save_json(LEDGER, ledger)
    say(f"assigned {label}: {args.games} games at offset {offset} on {args.machine} "
        f"({spec['workers']} workers)")
    if args.push:
        push_models(entry, student)
        push_manifest(entry)
        push_code(entry, None)
        rebuild(entry)
    if args.start:
        start_worker(entry, label)


# --------------------------------------------------------------------------- run control
def start_worker(entry: dict, label: str | None) -> None:
    root = f"$HOME/{WORKER_ROOT}"
    label_arg = f" --label {label}" if label else ""
    wake = "caffeinate -i " if entry.get("os", "darwin") == "darwin" else ""
    cmd = (f"pkill -f 'dist/worker.py' 2>/dev/null; sleep 2; "
           f"pkill -9 -f 'dist/worker.py' 2>/dev/null; sleep 1; cd {root} && mkdir -p logs && "
           f"nohup {wake}{root}/env/bin/python {root}/repo/scripts/dist/worker.py --root {root} "
           f"run{label_arg} > logs/worker.out 2>&1 & echo started $!")
    proc = ssh(entry, cmd, check=False)
    say(proc.stdout.strip() or proc.stderr.strip())


def cmd_start(args) -> None:
    start_worker(machine(args.machine), args.label)


def cmd_stop(args) -> None:
    entry = machine(args.machine)
    proc = ssh(entry, "curl -s -m 5 -X POST http://127.0.0.1:8765/stop_now || true", check=False)
    say(proc.stdout.strip() or "no response (worker not running?)")


def cmd_status(args) -> None:
    ledger = load_json(LEDGER, {"segments": []})
    enabled = [(name, entry) for name, entry in machines().items() if entry.get("enabled", True)]

    def probe(item):
        # One ssh per machine, in parallel: a dead tunnel blocks for minutes, and with several
        # machines a sequential loop would multiply that by the number of machines.
        name, entry = item
        proc = ssh(entry, f"{WORKER_ROOT}/env/bin/python {WORKER_ROOT}/repo/scripts/dist/worker.py "
                          f"--root $HOME/{WORKER_ROOT} status --json", check=False)
        try:
            states = json.loads(proc.stdout.strip() or "[]")
        except json.JSONDecodeError:
            states = []
        return name, (None if proc.returncode != 0 and not states else states)

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, len(enabled))) as pool:
        results = list(pool.map(probe, enabled))
    for name, states in results:
        if states is None:
            rows.append({"machine": name, "label": "—", "status": "unreachable", "done": 0, "total": 0,
                         "games_per_hour": 0})
            continue
        if not states:
            rows.append({"machine": name, "label": "—", "status": "no state", "done": 0, "total": 0,
                         "games_per_hour": 0})
        for state in states:
            state["machine_name"] = name
            rows.append(state)
    width = max([len(str(r.get("label"))) for r in rows] + [5])
    print(f"{'machine':10} {'label':{width}} {'status':10} {'done':>9} {'/h':>7} {'eta':>8}")
    for r in rows:
        eta = r.get("eta_seconds")
        eta_text = f"{int(eta) // 3600}h{int(eta) % 3600 // 60:02d}m" if eta else "—"
        print(f"{r.get('machine_name', r.get('machine', '?')):10} {str(r.get('label')):{width}} "
              f"{str(r.get('status')):10} {str(r.get('done')) + '/' + str(r.get('total')):>9} "
              f"{str(r.get('games_per_hour')):>7} {eta_text:>8}")
    print()
    for seg in ledger["segments"]:
        print(f"ledger {seg['label']:16} {seg['machine']:8} offset {seg['seed_offset']:6} "
              f"{seg['seeds']:4} games  {seg['status']}")


# --------------------------------------------------------------------------- collect
def validate(label_dir: Path, segment: dict) -> dict:
    """Every finished game must be readable by the trainer's loader, and the seeds must be ours."""
    sys.path.insert(0, str(ROOT / "python"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from pvz_frame_store import frame_paths, read_frames

    label = segment["label"]
    out = label_dir / "out"
    games_path = out / "games.jsonl"
    if not games_path.is_file():
        raise SystemExit(f"{games_path} missing")
    rows = [json.loads(line) for line in games_path.read_text().splitlines() if line.strip()]
    seeds = [int(r["seed"]) for r in rows]
    if len(seeds) != len(set(seeds)):
        raise SystemExit(f"{label}: duplicate seeds in games.jsonl")
    lo, hi = segment["seed_offset"], segment["seed_offset"] + segment["seeds"]
    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    task = next(t for t in manifest["tasks"] if t["task_id"] == DEFAULT_TASK)
    expected = set(task["seeds"][lo:hi])
    stray = set(seeds) - expected
    if stray:
        raise SystemExit(f"{label}: {len(stray)} seeds outside the assigned segment, e.g. {sorted(stray)[:3]}")
    missing = expected - set(seeds)
    archives = frame_paths(label_dir / "frames")
    have = {int(p.parent.parent.name.split("_")[1]) for p in archives if p.parent.parent.name.startswith("seed_")}
    if missing - have:
        raise SystemExit(f"{label}: {len(missing - have)} games have no frame archive")
    unreadable = []
    for path in archives:
        try:
            frames = read_frames(path)
            if len(frames) < 2:
                unreadable.append((path, "fewer than 2 frames"))
        except Exception as exc:                      # noqa: BLE001 - report, do not crash
            unreadable.append((path, repr(exc)))
    if unreadable:
        raise SystemExit(f"{label}: {len(unreadable)} unreadable archives, e.g. {unreadable[:2]}")
    summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").is_file() else {}
    won = sum(1 for r in rows if r.get("won"))
    return {"label": label, "games": len(rows), "won": won,
            "win_rate": round(won / len(rows), 3) if rows else None,
            "mean_seconds": round(sum(r.get("seconds", 0) for r in rows) / len(rows), 1) if rows else None,
            "archives": len(archives), "summary": summary}


def collect_segment(entry: dict, segment: dict, args) -> dict:
    label = segment["label"]
    local = RESULTS / label
    (local / "out").mkdir(parents=True, exist_ok=True)
    (local / "frames").mkdir(parents=True, exist_ok=True)
    say(f"collecting {label} from {segment['machine']}")
    tries, delay = args.retries, args.retry_delay
    if not rsync_retry(entry, f"~/{WORKER_ROOT}/out/{label}/", str(local / "out") + "/",
                       tries=tries, delay=delay):
        raise SystemExit(f"{label}: could not fetch out/")
    # Frames are the bulk of the payload. One seed per rsync keeps a dropped tunnel cheap: what
    # arrived stays, and the next attempt only moves the missing archives.
    listing = ssh(entry, f"ls -1 {WORKER_ROOT}/frames/{label} 2>/dev/null", check=False).stdout
    seeds = [line.strip() for line in listing.splitlines() if line.strip().startswith("seed_")]
    if seeds:
        say(f"{len(seeds)} seed dirs on {segment['machine']}")
        for name in sorted(seeds):
            if not rsync_retry(entry, f"~/{WORKER_ROOT}/frames/{label}/{name}/",
                               str(local / "frames" / name) + "/", tries=tries, delay=delay):
                raise SystemExit(f"{label}: could not fetch frames/{name}")
    elif not rsync_retry(entry, f"~/{WORKER_ROOT}/frames/{label}/", str(local / "frames") + "/",
                         tries=tries, delay=delay):
        raise SystemExit(f"{label}: could not fetch frames/")
    return validate(local, segment)


def cmd_collect(args) -> None:
    ledger = load_json(LEDGER, {"segments": []})
    targets = [s for s in ledger["segments"]
               if (not args.label or s["label"] == args.label) and s["status"] in ("assigned", "running", "done", "collected")]
    if not targets:
        raise SystemExit("nothing to collect")
    deadline = time.time() + args.wait * 60 if args.wait else None
    for segment in targets:
        try:
            entry = machine(segment["machine"])
        except SystemExit:
            say(f"skip {segment['label']}: machine {segment['machine']!r} is not in machines.json")
            continue
        while True:
            try:
                report = collect_segment(entry, segment, args)
                break
            except SystemExit as exc:
                # --wait keeps the job alive across tunnel outages; collected files stay on disk.
                if deadline is None or time.time() >= deadline:
                    raise
                say(f"{exc} — retrying whole collect in {args.retry_delay}s")
                time.sleep(args.retry_delay)
        label = segment["label"]
        local = RESULTS / label
        say(f"validated: {report['games']} games, {report['archives']} archives, "
            f"won {report['won']} ({report['win_rate']}), {report['mean_seconds']}s/game")
        save_json(local / "validation.json", report)
        if args.no_upload:
            continue
        package = local / f"{label}.tgz"
        say(f"packing {package}")
        with tarfile.open(package, "w:gz") as tar:
            for seed_dir in sorted((local / "frames").glob("seed_*")):
                for archive in seed_dir.iterdir():
                    tar.add(archive, arcname=f"{seed_dir.name}/{archive.name}")
        upload(package)
        segment["status"] = "collected"
        segment["collected_at"] = now()
        segment["games"] = report["games"]
        segment["won"] = report["won"]
        segment["win_rate"] = report["win_rate"]
        segment["mean_seconds"] = report["mean_seconds"]
        segment["archives"] = report["archives"]
        segment["package"] = f"transfer/{package.name}"
        if report.get("summary"):
            segment["summary"] = report["summary"]
        save_json(LEDGER, ledger)


def upload(package: Path) -> None:
    repo = os.environ.get("PVZ_HF_REPO")
    token = os.environ.get("HF_TOKEN")
    if not repo or not token:
        raise SystemExit("PVZ_HF_REPO / HF_TOKEN not set (see ~/.zshrc) — package kept at " + str(package))
    code = (
        "import os, sys\n"
        "from huggingface_hub import HfApi\n"
        f"HfApi(token=os.environ['HF_TOKEN']).upload_file(path_or_fileobj={str(package)!r}, "
        f"path_in_repo='transfer/{package.name}', repo_id=os.environ['PVZ_HF_REPO'])\n"
        "print('uploaded transfer/" + package.name + "')\n"
    )
    for attempt in range(1, 6):
        proc = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
        if proc.returncode == 0:
            say(proc.stdout.strip())
            return
        say(f"upload failed ({attempt}/5): {proc.stderr.strip().splitlines()[-1] if proc.stderr else ''}")
        time.sleep(30)
    raise SystemExit("upload failed after 5 attempts; package kept at " + str(package))


# --------------------------------------------------------------------------- web
def cmd_web(args) -> None:
    entry = machine(args.machine)
    say(f"tunnel: http://127.0.0.1:{args.port}/ -> {args.machine}:8765")
    proc = subprocess.Popen(["ssh", "-N", "-o", "BatchMode=yes", "-L",
                             f"{args.port}:127.0.0.1:8765", entry["host"]])
    try:
        time.sleep(1.5)
        if args.open:
            subprocess.run(["open", f"http://127.0.0.1:{args.port}/"], check=False)
        say("ctrl-c to close the tunnel")
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


# --------------------------------------------------------------------------- main
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("status", help="every machine's progress + the ledger")
    q.set_defaults(func=cmd_status)

    q = sub.add_parser("bootstrap", help="install the whole toolchain on a machine")
    q.add_argument("machine")
    q.set_defaults(func=cmd_bootstrap)

    q = sub.add_parser("push", help="send code / models / resources / manifest")
    q.add_argument("machine")
    q.add_argument("--code", action="store_true")
    q.add_argument("--models", action="store_true")
    q.add_argument("--resources", action="store_true")
    q.add_argument("--manifest", action="store_true")
    q.add_argument("--all", action="store_true")
    q.add_argument("--build", action="store_true", help="rebuild after pushing code")
    q.add_argument("--commit", help="git ref to push (default HEAD)")
    q.add_argument("--student", help="student checkpoint to send (also used as selfcheck)")
    q.set_defaults(func=cmd_push)

    q = sub.add_parser("assign", help="create a segment and push its task.json")
    q.add_argument("machine")
    q.add_argument("student")
    q.add_argument("games", type=int)
    q.add_argument("--label")
    q.add_argument("--offset", type=int)
    q.add_argument("--task-id", default=DEFAULT_TASK)
    q.add_argument("--workers", type=int)
    q.add_argument("--push", action="store_true", help="also push models, manifest and code")
    q.add_argument("--start", action="store_true", help="start the worker right away")
    q.add_argument("--allow-reserved", action="store_true", help="allow offsets outside the distributed range")
    q.add_argument("--gpu", action="store_true",
                   help="CUDA rollout backend (1 worker, 1 game per wave; binary built on the worker)")
    q.set_defaults(func=cmd_assign)

    q = sub.add_parser("start", help="start the worker on a machine")
    q.add_argument("machine")
    q.add_argument("--label")
    q.set_defaults(func=cmd_start)

    q = sub.add_parser("stop", help="stop-now on a machine")
    q.add_argument("machine")
    q.set_defaults(func=cmd_stop)

    q = sub.add_parser("collect", help="pull results back, validate, package, upload to HF")
    q.add_argument("--label")
    q.add_argument("--no-upload", action="store_true")
    q.add_argument("--retries", type=int, default=3, help="attempts per transfer (default 3)")
    q.add_argument("--retry-delay", type=int, default=10, help="seconds between attempts (default 10)")
    q.add_argument("--wait", type=int, default=0,
                   help="keep retrying the whole collect for this many minutes while the tunnel flaps")
    q.set_defaults(func=cmd_collect)

    q = sub.add_parser("web", help="ssh tunnel to a machine's local page")
    q.add_argument("machine")
    q.add_argument("--port", type=int, default=8765)
    q.add_argument("--no-open", dest="open", action="store_false")
    q.set_defaults(func=cmd_web, open=True)

    args = p.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
