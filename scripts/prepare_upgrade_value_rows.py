"""Turn human upgrade decisions into rows the leaf value net can train on (task214c).

One human archive is replayed to each decision where the human planted an upgrade plant
(40-47). Every such state is branched into two arms -- the human's upgrade and a wait -- and
each arm is continued with the student to ``--horizon`` ticks. The leaf frame is encoded with
the value net's own encoder (``pvz_value_net.encode_frame``), then the same leaf is continued to
the end of the game for the win/loss label. Output is one torch shard (board/scalars/label/meta)
plus a manifest; nothing is trained here.

  python scripts/prepare_upgrade_value_rows.py \
    --archive artifacts/human_demo/raw/seed_3420196500/user_l29_c08db8116248 \
    --student /tmp/task214-models/student/model.pt \
    --value /tmp/task214-models/v5/model.pt --encoder-version 5 \
    --out artifacts/task214c/upgrade_value --limit 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]

UPGRADE_TYPES = tuple(range(40, 48))


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archive", type=Path, action="append", required=True,
                   help="human demo archive directory (frames.npz + meta.json); repeatable")
    p.add_argument("--student", type=Path, required=True, help="continuation policy checkpoint")
    p.add_argument("--value", type=Path, required=True, help="leaf value checkpoint (for the reference score)")
    p.add_argument("--encoder-version", type=int, default=5, choices=range(1, 6))
    p.add_argument("--horizon", type=int, default=2500, help="ticks after the action that produce the leaf")
    p.add_argument("--label-decisions", type=int, default=3000, help="action cap of the label continuation")
    p.add_argument("--wait", type=int, default=600, help="ticks of the negative 'wait' arm")
    p.add_argument("--limit", type=int, default=0, help="newest N upgrade decisions per archive (0 = all)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--executable", type=Path, default=Path.home() / "pvz-mac-collect/build/pvz-portable")
    p.add_argument("--resource-dir", type=Path, default=Path.home() / ".cache/pvz-research-resources")
    return p


def _upgrade_decisions(frames):
    """Decision indices whose recorded action planted an upgrade plant."""
    packet_type = {}
    found = []
    for index, frame in enumerate(frames):
        for card in frame.get("packets") or ():
            packet_type[card[0]] = card[1]
        action = frame.get("action") or {}
        # frames[i]["action"] produced frame i, so replay(i) stops on the state before it.
        if action.get("type") == "plant" and packet_type.get(action.get("packet")) in UPGRADE_TYPES:
            found.append(index)
    return found


def _leaf_frame(obs):
    from render_episode import _frame
    return _frame(obs)


def _arm_rows(env, root, obs, actions, args, seed):
    """Evaluate every arm from the same root; returns (encoded rows, metadata)."""
    from pvz_value_net import encode_frame
    rows, meta = [], []
    for name, action in actions:
        env.restore_fast(root)
        current, _, done, _, info = env.step(action)
        if not info.get("ok"):
            meta.append({"arm": name, "rejected": True})
            continue
        if not done:
            current = env.rollout(None, current["tick"] + args.horizon, seed=seed)
        board, scalars = encode_frame(_leaf_frame(current), args.encoder_version)
        label, outcome_tick = 0.0, current["tick"]
        if current.get("terminal"):
            label = 1.0 if current.get("result") == 1 else 0.0
        else:
            snapshot = env.snapshot_fast()
            try:
                end = env.rollout(None, 0, max_actions=args.label_decisions,
                                  snapshot_id=snapshot, seed=seed)
            finally:
                env.release_snapshot(snapshot)
            outcome_tick = end["tick"]
            label = 1.0 if end.get("result") == 1 else 0.0
        rows.append((board, scalars, label))
        meta.append({"arm": name, "action": action, "tick": current["tick"],
                     "outcome_tick": outcome_tick, "won": bool(label)})
    return rows, meta


def main():
    args = parser().parse_args()
    out = args.out
    if (out / "rows.pt").exists():
        raise SystemExit(f"{out/'rows.pt'} exists; choose a fresh --out")
    from pvz_env import PvZEnv
    from pvz_episode_archive import read_meta
    from pvz_frame_store import frame_path, read_frames
    import search_play

    keep, exported = search_play._export_values(str(args.value), None)
    search_play._load_nets(str(args.value), str(args.student))
    boards, scalars, labels, meta_rows = [], [], [], []
    started = time.time()
    with PvZEnv(args.resource_dir, executable=args.executable, headless=True) as env:
        env.load_rollout_net(search_play._W["rollout_file"])
        env.load_value_net(exported[0])
        from human_demo import replay
        for archive in args.archive:
            archive = archive.resolve()
            frames = read_frames(frame_path(archive))
            decisions = _upgrade_decisions(frames)
            if args.limit:
                decisions = decisions[-args.limit:]
            for decision in decisions:
                obs, _, _ = replay(env, archive, decision=decision)
                root = env.snapshot_fast()
                try:
                    legal = (obs.get("legal_actions") or {}).get("plants") or []
                    recorded = frames[decision]["action"]
                    action = next((a for a in legal
                                   if a["col"] == recorded["col"] and a["row"] == recorded["row"]
                                   and a["packet"] == recorded["packet"]), None)
                    if action is None:
                        continue
                    arms = [("upgrade", {"type": "plant", "packet": action["packet"],
                                         "col": action["col"], "row": action["row"]}),
                            ("wait", {"type": "wait", "ticks": args.wait})]
                    seed = int(hashlib.sha256(f"{archive}:{decision}".encode()).hexdigest()[:8], 16)
                    rows, meta = _arm_rows(env, root, obs, arms, args, seed)
                finally:
                    env.release_snapshot(root)
                for (board, scalar, label), row in zip(rows, meta):
                    boards.append(board)
                    scalars.append(scalar)
                    labels.append(label)
                    meta_rows.append(dict(archive=str(archive), decision=decision,
                                          level=read_meta(archive)["task"]["level"], **row))
                print(f"{archive.name} d{decision}: " + ", ".join(
                    f"{row['arm']}={'win' if row['won'] else 'loss'}" for row in meta), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    shard = {"board": torch.stack(boards).to(torch.float32),
             "scalars": torch.stack(scalars).to(torch.float32),
             "label": torch.tensor(labels, dtype=torch.float32), "meta": meta_rows}
    torch.save(shard, out / "rows.pt")
    (out / "manifest.json").write_text(json.dumps({
        "rows": len(labels), "wins": int(sum(labels)), "encoder_version": args.encoder_version,
        "horizon": args.horizon, "wait": args.wait, "label_decisions": args.label_decisions,
        "archives": [str(path.resolve()) for path in args.archive], "seconds": round(time.time() - started, 1),
        "shape": list(shard["board"].shape), "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
    keep.cleanup()
    print(f"{len(labels)} rows ({int(sum(labels))} win) -> {out/'rows.pt'}")


if __name__ == "__main__":
    main()
