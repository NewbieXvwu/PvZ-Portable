"""Mine "must-drill" rescue positions from episode archives (task234 v0).

A drill is a (archive, decision) pair plus a mechanical, policy-free local success
criterion.  Nothing here chooses an action or encodes a tactic: the miner only reads
engine states already stored in ``frames.npz`` and looks for counter zombies that are
about to cross the front line or the house.

The output bank is intentionally small (KB) and references archives by logical source
root, so the same bank can be replayed on a machine that keeps the archives elsewhere::

  python3 scripts/drill_mine.py --archive-root main=/path/to/PvZAgent \
      --archive artifacts/human_demo/raw/.../demo_fog_39... --out artifacts/task234/drill_bank.json

Format of one drill::

  {"id": "...", "source_root": "main", "archive": "artifacts/...", "decision": 812,
   "root_frame": 811, "root_tick": 20000, "terrain": "fog", "level": 39,
   "threats": [{"id": 123, "row": 2, "type": 18, "x": 330.0}],
   "horizon": 900, "criterion": {...}, "mined_from": {...}}

``decision`` is the existing demo_start convention: 1 = the archive's initial state,
N = the state immediately before the archive's N-th action.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]

from pvz_frame_store import frame_path, read_frames  # noqa: E402

# ZombieType, src/ConstEnums.h.  The "counter" set is the roster that asks for a
# specific answer (jump/balloon/digger/pole/dolphin/football/bungee/jack/...).  It is a
# roster fact, not a tactic: the miner never says which card to use.
COUNTER_TYPES = {
    3: "polevaulter", 7: "football", 10: "duckytube", 11: "snorkel", 12: "zamboni",
    13: "bobsled", 14: "dolphin", 15: "jack", 16: "balloon", 17: "digger",
    18: "pogo", 19: "yeti", 21: "ladder", 23: "gargantuar",
    32: "redeye_gargantuar", 25: "boss",
}
# Bungee and catapult are counter zombies too, but "it left the board" is not "it was
# killed": the bungee leaves after stealing, the catapult stays behind the line.  Their
# local success needs a different criterion, so v0 counts them separately instead of
# pretending the ground-zombie test covers them.
SPECIAL_TYPES = {20: "bungee", 22: "catapult"}
# These reach the house without walking through the front line, so a simple
# "x <= 4 columns" trigger is already the rescue window.
BYPASS_TYPES = {10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 23, 32}
TYPE_NAME = {**COUNTER_TYPES, 0: "normal", 1: "flag", 2: "cone", 4: "pail", 5: "newspaper",
             6: "door", 8: "dancer", 9: "backup", 24: "imp"}
TERRAIN_BY_LEVEL = (
    ("day", 1, 10), ("night", 11, 20), ("pool", 21, 30), ("fog", 31, 40),
    ("roof", 41, 49), ("boss", 50, 50),
)
PLANT_POT = 33


def opening_drills(panel, levels, seeds):
    """Reset existing panel tasks, preserving their decks and environment fields."""
    return [dict(id=f"opening:{task['task_id']}:{seed}", family='opening',
                 terrain=terrain_of(int(task['level']), {}), level=int(task['level']),
                 task_seed=int(seed), task_definition=dict(task, seeds=[int(seed)]),
                 horizon=6000, scoring='reference_terminal', targets=[],
                 criterion=dict(scoring='reference_terminal'))
            for task in panel['tasks'] if int(task['level']) in levels for seed in seeds]


def mine_families(item):
    """Real preparation/upgrade windows; inclusion uses engine facts only.

    Preparation precedes the first on-board occurrence of a counter type by 1800
    ticks. Upgrade windows start at tick 3000 when both cards are in the seed bank
    and that upgrade is absent. Neither inclusion nor scoring prescribes an action.
    """
    from pvz_constants import upgrade_bases
    archive = Path(item['path'])
    meta = json.loads((archive / 'meta.json').read_text())
    if meta.get('demo_archive'):
        return []
    frames = read_frames(frame_path(archive))
    if not frames:
        return []
    level = int(frames[0]['scene']['level'])
    terrain = terrain_of(level, frames[0]['scene'])
    result = []

    def add(family, index, horizon, trigger):
        frame = frames[index]
        if frame.get('terminal'):
            return
        result.append(dict(
            id=f"{item['source_root']}:{item['archive']}:{family}:{trigger['name']}:r{index}",
            source_root=item['source_root'], archive=item['archive'],
            decision=index + 1, root_frame=index, root_tick=int(frame['tick']),
            level=level, terrain=terrain, family=family, horizon=horizon,
            scoring='reference_terminal', criterion=dict(scoring='reference_terminal'),
            targets=[], trigger=trigger))

    seen = set()
    for frame in frames:
        for z in frame.get('zombies') or []:
            kind = int(z[1])
            if not z[5] or kind not in {7, 12, 14, 16, 18, 20, 21, 22} or kind in seen:
                continue
            seen.add(kind)
            tick = int(frame['tick'])
            if tick < 3000:
                continue
            index = frame_at_or_before(frames, tick - 1800)
            if any(q[5] and int(q[1]) == kind for q in frames[index]['zombies']):
                continue
            add('preparation', index, 3000,
                dict(name=TYPE_NAME.get(kind, SPECIAL_TYPES.get(kind)), type=kind,
                     arrival_tick=tick, lead_ticks=tick - int(frames[index]['tick'])))
    deck = {int(p[1]) for p in frames[0].get('packets') or []}
    index = frame_at_or_after(frames, 3000)
    frame = frames[index]
    if int(frame['tick']) >= 3000:
        chains = []
        for upgrade, base in upgrade_bases().items():
            if not {upgrade, base} <= deck or any(int(p[2]) == upgrade for p in frame['plants']):
                continue
            chains.append(dict(upgrade=upgrade, base=base,
                bases_at_root=sum(int(p[2]) == base for p in frame['plants'])))
        if chains:
            add('upgrade', index, 6000, dict(name='upgrade_chain', chains=chains))
    return result


def terrain_of(level: int, scene: dict) -> str:
    """Adventure level range is authoritative; the scene flags are only a fallback.

    L40 is a fog level but its engine scene reports ``fog: false, pool: true``
    (storm/hurricane presentation), so trusting the flags there would file the last
    fog level under pool and break the per-level bank coverage.
    """
    if level == 50:
        return "boss"
    for name, low, high in TERRAIN_BY_LEVEL:
        if low <= level <= high:
            return name
    if scene.get("roof"):
        return "roof"
    if scene.get("fog"):
        return "fog"
    if scene.get("pool"):
        return "pool"
    if scene.get("night"):
        return "night"
    return "other"


def front_x(plants, row: int):
    """Rightmost plant centre in a lane: the front line the zombie has to cross."""
    xs = [40 + 80 * p[1] + 40 for p in plants if p[0] == row]
    return max(xs) if xs else None


def lane_plants(plants, row: int):
    return [p for p in plants if p[0] == row]


def threat_payload(frame, horizon=900):
    """Counter zombies that are on board now and matter for this drill."""
    out = []
    for z in frame["zombies"]:
        if not z[5] or z[1] not in COUNTER_TYPES:
            continue
        out.append(dict(id=int(z[9]), row=int(z[0]), type=int(z[1]), name=COUNTER_TYPES[z[1]],
                        x=round(float(z[2]), 1), health=int(z[4])))
    return out


def mower_states(frame):
    return {int(row): int(state) for row, state in frame.get("mowers") or []}


def frame_at_or_after(frames, tick):
    for index, frame in enumerate(frames):
        if frame["tick"] >= tick:
            return index
    return len(frames) - 1


def frame_at_or_before(frames, tick):
    best = 0
    for index, frame in enumerate(frames):
        if frame["tick"] <= tick:
            best = index
        else:
            break
    return best


def breach_in_window(frames, start_index, row, ticks):
    """Did this lane lose a mower / did the house fall inside the window?

    Roof levels have no lawn mowers in the archive (roof cleaners are a separate engine
    object), so the terminal result is the only house-breach signal there.
    """
    before = mower_states(frames[start_index])
    end_tick = frames[start_index]["tick"] + ticks
    for frame in frames[start_index + 1:]:
        if frame["tick"] > end_tick:
            break
        after = mower_states(frame)
        if before.get(row, 1) in (0, 1) and after.get(row) in (2, 3):
            return True
        if frame.get("result") == 2 and (not frame.get("mowers") or after.get(row) in (2, 3)):
            return True
        before = after or before
    return False


def plants_lost_between(frames, start_index, end_index):
    """Archive-only plant-loss count from the per-frame totals (best effort)."""
    before = (frames[start_index].get("totals") or {})
    after = (frames[end_index].get("totals") or {})
    return max(0, int(before.get("plants_lost") or 0) - int(after.get("plants_lost") or 0))


def mine_archive(item, max_per_archive, horizon, lookback, min_gap, include_human):
    archive = Path(item["path"])
    frames = read_frames(frame_path(archive))
    if len(frames) < 12:
        return [], [], Counter()
    meta_path = archive / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta.get("demo_archive"):
        # A start-cell continuation begins mid-game: its frame 0 is the restored demo
        # state, not tick 0, so it cannot be replayed from reset as a drill root.
        return [], [], Counter()
    level = int(frames[0]["scene"].get("level") or (meta.get("task") or {}).get("level") or 0)
    terrain = terrain_of(level, frames[0]["scene"])
    lanes = list(frames[0].get("lanes") or range(len(frames[0].get("row_terrain") or [])))
    seen_ids = set()
    special = Counter()
    for frame in frames:
        for z in frame["zombies"]:
            if z[5] and z[1] in SPECIAL_TYPES:
                special[SPECIAL_TYPES[z[1]]] += 1
    events = []
    for index, frame in enumerate(frames):
        if frame.get("terminal"):
            break
        tick = int(frame["tick"])
        front = {row: front_x(frame["plants"], row) for row in lanes}
        for z in frame["zombies"]:
            if not z[5] or z[1] not in COUNTER_TYPES:
                continue
            zombie_id, row, ztype, x = int(z[9]), int(z[0]), int(z[1]), float(z[2])
            if zombie_id in seen_ids:
                continue
            front_line = front.get(row)
            crossed = front_line is not None and x < front_line
            near = x <= 280.0                 # <= 3 columns from the house
            approach = x <= 400.0             # inside a 900-tick rescue window for most speeds
            bypass = ztype in BYPASS_TYPES
            # "About to cross": within one and a half cells of the rightmost plant.  A
            # human demo often answers exactly here, before the zombie is on top of the
            # house; waiting for x < front would only mine the failures.
            close = front_line is not None and x <= front_line + 120.0
            if not (near or crossed or close or (approach and bypass)):
                continue
            seen_ids.add(zombie_id)
            start_index = frame_at_or_before(frames, tick - lookback)
            # Never start before the zombie exists, and never start on a terminal state.
            while start_index < index and not any(
                    int(q[9]) == zombie_id and q[5] for q in frames[start_index]["zombies"]):
                start_index += 1
            while start_index > 0 and frames[start_index].get("terminal"):
                start_index -= 1
            if frames[start_index].get("terminal"):
                continue
            threat = next((dict(id=zombie_id, row=row, type=ztype,
                                name=COUNTER_TYPES[ztype], x=round(x, 1),
                                health=int(z[4])) for z in frames[start_index]["zombies"]
                           if int(z[9]) == zombie_id and z[5]), None)
            if threat is None:
                continue
            end_index = frame_at_or_after(frames, frames[start_index]["tick"] + horizon)
            breach = breach_in_window(frames, start_index, row, horizon)
            danger = min([q[2] for q in (z for f in frames[start_index:end_index + 1]
                                          for z in f["zombies"])
                          if q[0] == row and q[1] in COUNTER_TYPES and q[5]] or [9999])
            # Original-archive evidence only: this is not the drill result.
            original_loss = sum(
                int((frames[i + 1].get("totals") or {}).get("plants_eaten", 0))
                - int((frames[i].get("totals") or {}).get("plants_eaten", 0))
                for i in range(start_index, min(end_index, len(frames) - 1)))
            events.append(dict(
                index=start_index, decision=start_index + 1, tick=frames[start_index]["tick"],
                trigger_index=index, trigger_tick=tick, row=row, type=ztype,
                name=COUNTER_TYPES[ztype], x=round(x, 1), threat=threat,
                front_x=round(float(front_line), 1) if front_line is not None else None,
                crossed=bool(crossed), near=bool(near), approach=bool(approach),
                breach_ahead=breach, min_x_ahead=round(float(danger), 1),
                original_plant_loss=max(0, original_loss),
                priority=(2 if breach else 0) + (1 if near else 0) + (1 if crossed else 0)
                         + (1 if ztype in BYPASS_TYPES else 0),
            ))
    # Pick a diverse, non-overlapping set per archive.  Breach-backed points first,
    # then near/crossed/bypass points; same (lane, type) not twice.
    events.sort(key=lambda e: (-e["priority"], e["tick"]))
    chosen, used = [], set()
    for event in events:
        if len(chosen) >= max_per_archive:
            break
        key = (event["row"], event["type"])
        if key in used:
            continue
        if any(abs(event["tick"] - other["tick"]) < min_gap for other in chosen):
            continue
        used.add(key)
        chosen.append(event)
    drills = []
    for event in sorted(chosen, key=lambda e: e["tick"]):
        root_frame = event["index"]
        root = frames[root_frame]
        front_line = front_x(root["plants"], event["row"])
        # The drill is about the zombie that triggered the window, plus the same-wave
        # neighbours within two cells.  A far counter zombie from a later wave is not a
        # target: requiring it to die inside 900 ticks would make every drill fail.
        trigger_x = float(event["threat"]["x"])
        targets = [dict(id=int(z[9]), row=int(z[0]), type=int(z[1]),
                        name=COUNTER_TYPES[z[1]], x=round(float(z[2]), 1))
                   for z in root["zombies"]
                   if z[5] and z[1] in COUNTER_TYPES and int(z[0]) == event["row"]
                   and trigger_x - 120.0 <= float(z[2]) <= trigger_x + 120.0
                   and float(z[2]) > 80.0]
        if not targets or trigger_x <= 80.0:
            continue
        drills.append(dict(
            id=f"{item['source_root']}:{archive.name}:r{root_frame}",
            source_root=item["source_root"], archive=item["archive"],
            decision=root_frame + 1, root_frame=root_frame, root_tick=int(root["tick"]),
            level=level, terrain=terrain, lanes=lanes,
            threats=threat_payload(root), targets=targets,
            trigger=dict(row=event["row"], type=event["type"], name=event["name"],
                         x=event["x"], front_x=event["front_x"], tick=event["trigger_tick"],
                         crossed=event["crossed"], near=event["near"],
                         breach_ahead=event["breach_ahead"],
                         min_x_ahead=event["min_x_ahead"],
                         original_plant_loss=event["original_plant_loss"]),
            horizon=horizon,
            criterion=dict(threats_neutralized=True, mower_unchanged=True,
                           house_unchanged=True, secondary="plants_lost"),
        ))
    human = []
    if include_human and (meta.get("label_source") == "human"):
        human = mine_human_sequences(archive, item, frames, terrain, level)
        for sequence in human:
            if not (sequence.get("complete") and sequence.get("removed_type") is not None
                    and sequence.get("replanted")
                    and sequence["replanted"].get("type") == sequence["removed_type"]):
                continue
            root = frames[sequence["root_frame"]]
            front = front_x(root["plants"], sequence["row"])
            near = [dict(id=int(z[9]), row=int(z[0]), type=int(z[1]),
                         name=COUNTER_TYPES[z[1]], x=round(float(z[2]), 1))
                    for z in root["zombies"]
                    if z[5] and z[1] in COUNTER_TYPES and int(z[0]) == sequence["row"]
                    and float(z[2]) <= (front + 120.0 if front is not None else 600.0)]
            if not near:
                continue
            sequence["targets"] = near
            drills.append(dict(
                id=f"{item['source_root']}:{archive.name}:human_r{sequence['root_frame']}",
                kind="human_rescue", source_root=item["source_root"], archive=item["archive"],
                decision=sequence["decision"], root_frame=sequence["root_frame"],
                root_tick=int(root["tick"]), level=level, terrain=terrain, lanes=lanes,
                threats=threat_payload(root), targets=near,
                trigger=dict(row=sequence["row"], type=near[0]["type"], name=near[0]["name"],
                             x=near[0]["x"], front_x=round(float(front), 1) if front is not None else None,
                             tick=int(root["tick"]), crossed=False, near=False,
                             breach_ahead=False, min_x_ahead=near[0]["x"],
                             original_plant_loss=0),
                horizon=horizon,
                criterion=dict(threats_neutralized=True, mower_unchanged=True,
                               house_unchanged=True, secondary="plants_lost",
                               human_sequence=sequence),
            ))
    return drills, human, special


def mine_human_sequences(archive, item, frames, terrain, level):
    """Find shovel -> plant -> replant-at-the-same-cell sequences in a human recording."""
    out = []
    for index, frame in enumerate(frames):
        action = frame.get("input_action") or frame.get("action")
        if not action or action.get("type") != "shovel":
            continue
        row, col = int(action["row"]), int(action["col"])
        # The action in a frame produced that frame's state (human_demo recorder), so the
        # pre-shovel state is the previous frame and decision = index.
        root = max(0, index - 1)
        before = frames[root]
        plant = next((p for p in before["plants"] if p[0] == row and p[1] == col
                      and p[2] != PLANT_POT), None)
        removed = int(plant[2]) if plant else None
        planted, replanted = None, None
        for later in range(index + 1, min(index + 30, len(frames))):
            nxt = frames[later].get("input_action") or frames[later].get("action")
            if not nxt or nxt.get("type") != "plant" or nxt.get("row") != row or nxt.get("col") != col:
                continue
            packet = int(nxt.get("packet", -1))
            ptype = next((int(p[1]) for p in frames[later]["packets"] if int(p[0]) == packet), None)
            if planted is None:
                planted = dict(type=ptype, tick=int(frames[later]["tick"]), packet=packet)
                continue
            replanted = dict(type=ptype, tick=int(frames[later]["tick"]), packet=packet)
            break
        out.append(dict(
            kind="human_shovel_sequence", source_root=item["source_root"],
            archive=item["archive"], decision=index, root_frame=root,
            root_tick=int(before["tick"]), level=level, terrain=terrain,
            row=row, col=col, removed_type=removed,
            planted=planted, replanted=replanted,
            complete=bool(planted and replanted),
        ))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive-root", action="append", default=[], metavar="NAME=PATH",
                        help="logical source root (repeatable)")
    parser.add_argument("--archive", action="append", default=[], metavar="NAME:REL",
                        help="archive relative to a named root (repeatable)")
    parser.add_argument("--archive-list", type=Path,
                        help="JSON list of {source_root, archive, path} records")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-per-archive", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=900)
    parser.add_argument("--lookback", type=int, default=300)
    parser.add_argument("--min-gap", type=int, default=900)
    parser.add_argument("--no-human-sequences", action="store_true")
    parser.add_argument('--families', action='store_true',
                        help='mine preparation and upgrade windows with reference-terminal scoring')
    parser.add_argument('--opening-panel', type=Path,
                        help='existing task panel for reset-state economy questions (no archives)')
    parser.add_argument('--opening-levels', default='1-50')
    parser.add_argument('--opening-seeds', default='89000100,89000101')
    args = parser.parse_args()

    roots = {}
    for entry in args.archive_root:
        name, sep, path = entry.partition("=")
        if not sep:
            parser.error("--archive-root needs NAME=PATH")
        roots[name] = Path(path).expanduser().resolve()
    items = []
    if args.archive_list:
        for record in json.loads(args.archive_list.read_text()):
            items.append(dict(source_root=record["source_root"], archive=record["archive"],
                              path=record.get("path") or roots[record["source_root"]] / record["archive"]))
    for entry in args.archive:
        name, sep, rel = entry.partition(":")
        if not sep or name not in roots:
            parser.error(f"--archive needs a known root: {entry}")
        items.append(dict(source_root=name, archive=rel, path=roots[name] / rel))
    if not items and not args.opening_panel:
        parser.error("no archives")

    drills, human = [], []
    if args.opening_panel:
        from drill_scan import parse_levels
        drills.extend(opening_drills(json.loads(args.opening_panel.read_text()),
                                      parse_levels(args.opening_levels),
                                      [int(s) for s in args.opening_seeds.split(',')]))
    stats, special_stats = Counter(), Counter()
    sources = []
    for item in items:
        if args.families:
            got = mine_families(item)
            drills.extend(got)
            sources.append(dict(item, drills=len(got)))
            continue
        got, seq, special = mine_archive(item, args.max_per_archive, args.horizon, args.lookback,
                                          args.min_gap, not args.no_human_sequences)
        drills.extend(got)
        human.extend(seq)
        sources.append(dict(source_root=item["source_root"], archive=item["archive"],
                            drills=len(got), human_sequences=len(seq),
                            level=(json.loads((Path(item["path"]) / "meta.json").read_text()).get("task") or {}).get("level"),
                            label_source=json.loads((Path(item["path"]) / "meta.json").read_text()).get("label_source")))
        for drill in got:
            stats[f"{drill['terrain']}|{drill['trigger']['name']}"] += 1
        for name, count in special.items():
            special_stats[f"{got[0]['terrain'] if got else 'mixed'}|{name}"] += count
    by_terrain = Counter(drill["terrain"] for drill in drills)
    by_type = Counter(drill.get('trigger', {}).get('name', 'opening') for drill in drills)
    document = dict(
        schema="pvz-drill-bank-v1", task="task240" if args.families or args.opening_panel else "task234",
        criterion=(dict(scoring='reference_terminal', reference='frozen student',
                        success='terminal win', secondary='mowers used')
                   if args.families or args.opening_panel else dict(horizon=args.horizon, lookback=args.lookback,
                       success=["all root threats gone or off board by horizon",
                                "no lawn mower fired", "no house breach"],
                       secondary="plants_eaten during the horizon")),
        counts=dict(drills=len(drills), human_sequences=len(human),
                    by_family=dict(Counter(d.get('family', 'rescue') for d in drills)),
                    by_level=dict(Counter(d['level'] for d in drills)),
                    by_terrain=dict(sorted(by_terrain.items())),
                    by_type=dict(sorted(by_type.items())),
                    by_terrain_type=dict(sorted(stats.items())),
                    special_type_frames=dict(sorted(special_stats.items()))),
        sources=sources, drills=drills, human_sequences=human,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(document["counts"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
