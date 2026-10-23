"""Write a hand-picked goal set for the organiser's physics (lockstep env).

Goals are named with the repo's target specs (``branch@mm``, ``branch@-mm``, ``xyz:x,y,z`` in VMR mm,
node ids; see ``stroke_nav.data.anatomy``), converted to their world frame (metres) with the
fitted ``OrganiserFrame``. When the entry-roll table exists (``scripts/calibrate_rolls.py``), each
goal gets the roll whose recorded tip paths pass closest, and the distance by which they miss it.
The tip rides 2-5 mm off the centerline, so ``--snap`` moves a goal onto the nearest point of that
roll's path when it misses by more than ``--reach`` mm.

    python scripts/pick_goals.py --target left_carotid@30 --target right_subclavian@40 --target right_carotid@60
    python scripts/pick_goals.py --target left_subclavian@50 --split test --append      # add to the same file
    python scripts/pick_goals.py --branches                                            # list branch names

Writes results/lockstep/manual_goals.json in the make_goals.py format, which teleop_lockstep.py,
record_demos.py, eval_lockstep.py and demo_lockstep.py read with --goals.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stroke_nav.agents.roll_table import RollTable  # noqa: E402
from stroke_nav.data.anatomy import Anatomy  # noqa: E402
from stroke_nav.organiser.frame import default_frame  # noqa: E402


def nearest_on_path(table: RollTable, roll: float, goal: np.ndarray) -> np.ndarray:
    """Nearest recorded tip position (world m) of ``roll``'s first seed to ``goal``."""
    path = next(p for p in table.paths if p.roll_deg == roll).tip_m
    return path[int(np.argmin(np.linalg.norm(path - goal, axis=1)))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", action="append", default=[], help="target spec; repeat for several goals")
    ap.add_argument("--split", default="train", help="split label for these goals (train or test)")
    ap.add_argument("--out", default=str(ROOT / "results" / "lockstep" / "manual_goals.json"))
    ap.add_argument("--append", action="store_true", help="add to an existing --out file")
    ap.add_argument("--table", default=str(ROOT / "results" / "lockstep" / "roll_sweep.json"))
    ap.add_argument("--snap", action="store_true", help="move goals that the paths miss onto the nearest path point")
    ap.add_argument("--reach", type=float, default=3.0, help="mm; a miss above this counts as off the paths")
    ap.add_argument("--branches", action="store_true", help="list branch names and lengths, then exit")
    args = ap.parse_args()

    anat, frame = Anatomy.load("0248_H_AOCERE_CAS"), default_frame()
    if args.branches:
        for name in anat.branch_names:
            print(f"  {name}  ({len(anat.branch_nodes(name))} centerline nodes)")
        return
    if not args.target:
        ap.error("give at least one --target")
    table = RollTable.load(args.table) if Path(args.table).exists() else None
    if table is None:
        print(f"[goals] no roll table at {args.table}: goals get no entry roll (run scripts/calibrate_rolls.py)")

    out = Path(args.out)
    doc = json.loads(out.read_text(encoding="utf-8")) if args.append and out.exists() else {"goals": []}
    next_id = max((g["id"] for g in doc["goals"]), default=-1) + 1
    for spec in args.target:
        node = anat.resolve(spec)
        goal = np.asarray(frame.vmr_mm_to_world_m(anat.nodes[node]), dtype=float)
        entry = {"id": next_id, "spec": spec, "branch": anat.branch_of(node), "split": args.split, "source": "manual"}
        if table is not None:
            roll, miss = table.reach(goal)
            entry.update(roll_deg=roll, miss_mm=round(miss, 2))
            if args.snap and miss > args.reach:
                goal = nearest_on_path(table, roll, goal)
                entry.update(snapped_from_m=frame.vmr_mm_to_world_m(anat.nodes[node]).tolist(),
                             miss_mm=round(table.worst_mm(roll, goal), 2))
        else:
            entry.update(roll_deg=None, miss_mm=None)
        entry["goal_m"] = goal.tolist()
        doc["goals"].append(entry)
        note = ""
        if entry["miss_mm"] is not None:
            note = f"roll {entry['roll_deg']:g} deg, recorded paths miss it by {entry['miss_mm']:.1f} mm"
            if entry["miss_mm"] > 5.0:
                note += "  <- probably out of reach (success needs 5 mm); try --snap or another target"
        print(f"[goals] {next_id:3d} {spec:28s} {entry['branch']:24s} {note}")
        next_id += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"[goals] {len(doc['goals'])} goals in {out}")


if __name__ == "__main__":
    main()
