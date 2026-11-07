"""Task246 selection/label driver; replay, search, rollouts and labels use existing paths."""
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "python")]


def run_one(job):
    import numpy as np
    import search_play as play
    from human_demo import replay, reset_archive
    from pvz_episode_archive import read_meta
    from pvz_frame_store import frame_path, read_frames
    from pvz_event_env import EventWaitEnv
    from value_teacher_feasibility import sample_root, write_json
    base = Path(job["base"])
    target = base / "branches" / f"game_{job['game']:04d}"
    if (target / "complete.json").exists():
        return json.loads((target / "complete.json").read_text())
    if (target / "failed.json").exists():      # deterministic failure: do not replay it on every resume
        raise RuntimeError(json.loads((target / "failed.json").read_text())["error"])
    target.mkdir(parents=True, exist_ok=True)
    archive = base / "corpus" / job["archive"] if job.get("archive") else None
    if archive is not None:
        meta = read_meta(archive)
    else:
        from dataclasses import asdict
        from pvz_env import TaskSpec
        from pvz_tasks import profile_for_deck
        from render_episode import training_task_kwargs
        definition = job["task"]
        kw = training_task_kwargs(definition, job["probe_seed"])
        task_spec = TaskSpec(level=kw["level"], seed=kw["seed"], playthrough=2,
            profile=profile_for_deck(tuple(kw["deck"]),kw["first_aid_owned"]),
            **{k:v for k,v in kw["task_extra"].items() if v is not None})
        meta = dict(environment_task=asdict(task_spec),
                    task=dict(seed=kw["seed"],id=definition["task_id"],deck=kw["deck"]))
    task = dict(meta["environment_task"], task_id=meta["task"]["id"], deck=meta["task"]["deck"])
    models = base / "models"
    policy = play._terrain_policy(str(models / "student.pt"), str(models / "gptbc.pt"), task)
    value = play._leaf_value(str(models / "production.bin"), str(models / "roof.bin"), task)
    play._load_nets(value, policy, str(models / "static.bin"))
    args = play.parser().parse_args(["--rollout-policy", policy, "--value", value,
        "--manifest", str(base / "plan.json"), "--task-id", task["task_id"],
        "--task-json", json.dumps(task), "--out", str(target), "--student-candidates",
        str(job.get("student_candidates", 2)),
        "--verify-samples", "9", "--executable", job["executable"]])
    env = play._W["env"] = EventWaitEnv(args.resource_dir, executable=args.executable, headless=True)
    env.load_rollout_net(play._W["rollout_file"])
    env.load_value_net(value)
    env.load_value_net(str(models / "static.bin"), static=True)
    env._teacher_value_loaded = True
    rows, reports = [], []
    started = time.monotonic()
    try:
        frames = read_frames(frame_path(archive)) if archive is not None else []
        # Freeze roots from recorded actions, before reading any branch outcomes.
        plants = [i for i in range(1, len(frames)) if (frames[i].get("action") or {}).get("type") == "plant"]
        packet_types = {p[0]: p[1] for f in frames[:2] for p in f.get("packets", [])}
        economy = [i for i in plants if frames[i-1]["tick"] <= 6000 and
                   packet_types.get(frames[i]["action"]["packet"]) in (1, 16)]
        early = (economy or plants)[:2]
        middle = [min(plants, key=lambda i: abs(i-len(frames)//2))] if plants else []
        points = sorted(set(early + middle)) if archive is not None else [0]
        write_json(target / "selection.json", dict(**job, decisions=points,
            roots=[dict(decision=i, tick=frames[i-1]["tick"] if frames else 0,
                        recorded=frames[i]["action"] if frames else None) for i in points]))
        for point in points:
            if archive is not None:
                obs, _, prefix = replay(env, archive, decision=point)
            else:
                obs, prefix = reset_archive(env,meta), []
            state = dict(frame=prefix[-2] if len(prefix)>1 else None,
                         action=prefix[-1].get("action") if prefix else None)
            from pvz_student_observation import StudentHistory
            from pvz_value_net import observation_context
            state["history"] = StudentHistory(play._W["fast"].obs)
            for frame in prefix[:-1]:
                state["history"].extra(frame)
            play._W["context"] = observation_context(obs)
            play._fast_action(obs, state, env)
            play._W["student_history"] = state["history"]
            stats = dict(searched=0, overrides=0, override_kinds=Counter())
            labels, features = sample_root(env, obs, state, args,
                int(meta["task"]["seed"])*10000+point, stats, target / f"root_{point}", job["samples"])
            reports.append(dict(decision=point, tick=obs["tick"], labels=len(labels),
                                wins=sum(r["won"] for r in labels)))
            for index, feature in features:
                row = labels[index]
                same = [r for r in labels if r["candidate"] == row["candidate"]]
                rows.append((feature, row["won"], point, row["candidate"], row["future"],
                    sum(r["won"] for r in same)/len(same),
                    sum(not r["leaf_terminal"] for r in same)/len(same),
                    sum(r["won"] and r["leaf_terminal"] for r in same)/len(same)))
            print(f"game {job['game']} {job['terrain']} root {point}: {len(labels)} labels", flush=True)
        if rows:
            x = np.stack([r[0] for r in rows])
            np.savez_compressed(target / "rows.npz", board=x[:, :92*60].reshape(-1,92,60).astype(np.float16),
                scalars=x[:,92*60:92*60+119], context=x[:,92*60+119:],
                y=np.asarray([r[1] for r in rows],np.int8), op=np.asarray([r[2] for r in rows],np.int32),
                candidate=np.asarray([r[3] for r in rows],np.int16), future=np.asarray([r[4] for r in rows],np.int16),
                truth=np.asarray([r[5] for r in rows],np.float32),
                live_fraction=np.asarray([r[6] for r in rows],np.float32),
                terminal_win_fraction=np.asarray([r[7] for r in rows],np.float32),
                source=np.ones(len(rows),np.int8), seed=np.full(len(rows),int(meta['task']['seed']),np.int64),
                game=np.full(len(rows),job['game'],np.int32), level=np.full(len(rows),task['level'],np.int16))
        report = dict(game=job["game"], terrain=job["terrain"], seed=meta["task"]["seed"],
            roots=reports, rows=len(rows), wins=sum(r[1] for r in rows), seconds=time.monotonic()-started)
        write_json(target / "complete.json", report)
        return report
    finally:
        play._close_worker_env()
        for key in ("rollout_tmp",):
            obj=play._W.pop(key,None)
            if obj: obj.cleanup()
        play._W.clear()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--limit",type=int,default=0)
    p.add_argument("--plan",type=Path)
    p.add_argument("--external-pids",type=int,nargs='*',default=[],
        help="already running single-game workers; their slots open when they exit")
    args=p.parse_args()
    base=args.data.resolve()
    os.environ.update(PVZ_NO_POPUP="1",TMPDIR=str(base/"work"),OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1")
    import tempfile
    tempfile.tempdir=str(base/"work")
    jobs=json.loads((args.plan or base/"plan.json").read_text())["jobs"]
    if args.limit: jobs=jobs[:args.limit]
    started=time.monotonic()
    results=[]
    failed=[]
    owner={}
    external=set(args.external_pids)
    with ProcessPoolExecutor(max_workers=min(4,args.workers),mp_context=mp.get_context("spawn")) as pool:
        futures=set()
        remaining=iter(jobs)
        exhausted=False
        while futures or not exhausted:
            for pid in tuple(external):
                try:os.kill(pid,0)
                except ProcessLookupError:external.remove(pid)
            capacity=min(4,args.workers)-len(external)
            while not exhausted and len(futures)<capacity:
                job=next(remaining,None)
                if job is None:exhausted=True
                else:
                    future=pool.submit(run_one,job)
                    owner[future]=job
                    futures.add(future)
            if not futures:
                if not exhausted:time.sleep(10)
                continue
            done,futures=wait(futures,timeout=30,return_when=FIRST_COMPLETED)
            for f in done:
                try:
                    result=f.result()
                except Exception as error:
                    # One game whose teacher continuation hits the action cap must not discard
                    # the rest of a two-day generation; record it and keep going.
                    job=owner[f]
                    failed.append(dict(game=job["game"],terrain=job.get("terrain"),error=f"{type(error).__name__}: {error}"))
                    marker=base/"branches"/f"game_{job['game']:04d}"/"failed.json"
                    marker.parent.mkdir(parents=True,exist_ok=True)
                    marker.write_text(json.dumps(dict(error=str(error)))+"\n")
                    print(json.dumps(dict(failed=failed[-1])),flush=True)
                    continue
                results.append(result)
                print(json.dumps(result),flush=True)
    from value_teacher_feasibility import write_json
    name = args.plan.stem if args.plan else "collection"
    write_json(base/(f"{name}-report.json"),dict(
        games=results,failed=failed,seconds=time.monotonic()-started))


if __name__=="__main__":
    main()
