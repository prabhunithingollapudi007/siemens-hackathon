"""Build a set of reachable goals on the organiser's physics from the entry-roll table.

Goals are points about ``--spacing`` mm apart along the recorded tip paths
(``scripts/calibrate_rolls.py``) that every seed of their roll passes within ``--reach`` mm of, so
the scripted expert can reach each one. They are labelled with the vessel branch, capped per branch
so the trunk of the aorta does not dominate, and split at random into train and test. The
cas-reference goal is added on its own (split ``cas``).

    python scripts/make_goals.py                                     # results/lockstep/goals.json
    python scripts/make_goals.py --spacing 4 --per-branch 30 --test-fraction 0.25
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stroke_nav.agents.roll_table import RollTable  # noqa: E402
from stroke_nav.data.anatomy import Anatomy  # noqa: E402
from stroke_nav.organiser.frame import default_frame  # noqa: E402
from stroke_nav.organiser.spec import CAS_REFERENCE_GOALS_CM  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default=str(ROOT / "results" / "lockstep" / "roll_sweep.json"))
    ap.add_argument("--spacing", type=float, default=5.0, help="mm between goals along a path")
    ap.add_argument("--reach", type=float, default=3.0, help="mm every seed's path must pass within")
    ap.add_argument("--min-ins", type=float, default=40.0, help="mm of insertion before the first goal")
    ap.add_argument("--per-branch", type=int, default=40)
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "lockstep" / "goals.json"))
    args = ap.parse_args()

    table = RollTable.load(args.table)
    anat, frame = Anatomy.load("0248_H_AOCERE_CAS"), default_frame()
    rng = np.random.default_rng(args.seed)
    goals = table.candidate_goals(args.spacing, args.reach, args.min_ins)
    by_branch = collections.defaultdict(list)
    for g in goals:
        g["branch"] = anat.branch_of(anat.nearest_node(frame.world_m_to_vmr_mm(np.asarray(g["goal_m"]))))
        by_branch[g["branch"]].append(g)
    chosen = []
    for branch, gs in sorted(by_branch.items()):
        if len(gs) > args.per_branch:  # evenly along the branch, not just the first ones
            gs = [gs[i] for i in np.linspace(0, len(gs) - 1, args.per_branch).round().astype(int)]
        chosen += gs
    test = set(rng.choice(len(chosen), size=int(round(args.test_fraction * len(chosen))), replace=False).tolist())
    for i, g in enumerate(chosen):
        g["split"] = "test" if i in test else "train"
    cas = np.asarray(CAS_REFERENCE_GOALS_CM[0]) / 100.0  # mesh cm -> world m (meters_per_unit 0.01)
    roll, miss = table.reach(cas)
    chosen.append({"goal_m": cas.tolist(), "roll_deg": roll, "miss_mm": round(miss, 2), "ins_mm": None,
                   "branch": anat.branch_of(anat.nearest_node(frame.world_m_to_vmr_mm(cas))), "split": "cas"})
    for i, g in enumerate(chosen):
        g["id"] = i
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"table": str(args.table), "reach_mm": args.reach, "goals": chosen}, indent=1), encoding="utf-8")

    print(f"{len(goals)} reachable candidates -> {len(chosen)} goals ({len(test)} test) -> {out}")
    print(f"{'branch':26s} {'train':>5} {'test':>5}  rolls")
    for branch in sorted({g["branch"] for g in chosen}):
        gs = [g for g in chosen if g["branch"] == branch]
        print(f"{branch:26s} {sum(g['split'] == 'train' for g in gs):5d} {sum(g['split'] == 'test' for g in gs):5d}  "
              f"{sorted({g['roll_deg'] for g in gs})}")
    print(f"cas-reference goal: roll {roll}, worst-seed miss {miss:.2f} mm")


if __name__ == "__main__":
    main()
