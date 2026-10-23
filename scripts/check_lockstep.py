"""Check the lockstep env against recordings of the organiser's live stack.

Replays the roll sweep of ``scripts/measure_gap.py --sweep`` (turn the guidewire to a roll at the
entry, then push it 260 mm) in ``stroke_nav.organiser.live.lockstep`` and compares the final tip with
``results/gap/roll_<deg>_theirs.csv``. Needs the organiser sources (HCMR_EV_ROOT); their Docker stack
is not used and is best stopped (tools/organiser_stack.ps1 stop) so the pages get the CPU.

    python scripts/check_lockstep.py                       # rolls 0, 60, 90, 120, 150, 180, 270
    python scripts/check_lockstep.py --rolls 120 --workers 1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stroke_nav.data.anatomy import Anatomy  # noqa: E402
from stroke_nav.organiser.frame import default_frame  # noqa: E402
from stroke_nav.organiser.live.lockstep import make_env  # noqa: E402

GAP = ROOT / "results" / "gap"
OUT = ROOT / "results" / "lockstep"
PUSH_MM = 260.0


def live_final(roll: int, frame) -> dict | None:
    import pandas as pd

    f = GAP / f"roll_{roll:03d}_theirs.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    tip = d[["gw_tip_x", "gw_tip_y", "gw_tip_z"]].to_numpy()
    return {"tip_world_m": tip[-1].tolist(), "ins_mm": float(d.gw_insertion_m.iloc[-1] * 1000),
            "roll_deg": float(np.degrees(d.gw_roll_rad.iloc[-1])), "path_vmr": [frame.world_m_to_vmr_mm(p) for p in tip[::200]]}


def run_roll(roll: int, worker: int) -> dict:
    env = make_env(worker=worker, seed=worker)
    try:
        obs, info = env.reset(seed=worker)
        goal = np.asarray(info["goal"])
        rot_steps = int(round(math.radians(roll) / 1.0 / 0.2))  # 1 rad/s at |a| = 1
        rows, t0 = [], time.monotonic()
        plan = [np.array([0.0, 1.0, 0.0, 0.0])] * rot_steps + [np.array([1.0, 0.0, 0.0, 0.0])] * int(PUSH_MM / 2.0 + 10)
        plan += [np.zeros(4)] * 5
        for a in plan:
            obs, r, term, trunc, info = env.step(a.astype(np.float32))
            tip = np.asarray(env.bridge.latest_true().tip_position)
            rows.append({"tip": tip.tolist(), "ins_mm": float(obs["state"][17] * 1000), "dist_mm": float(np.linalg.norm(tip - goal) * 1000)})
            if term or trunc:
                break
        return {"roll": roll, "steps": len(rows), "wall_s": round(time.monotonic() - t0, 1), "rows": rows,
                "success": bool(info.get("success")), "min_goal_mm": min(r["dist_mm"] for r in rows)}
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rolls", default="0,60,90,120,150,180,270")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    rolls = [int(r) for r in args.rolls.split(",")]
    anat, frame = Anatomy.load("0248_H_AOCERE_CAS"), default_frame()
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(run_roll, rolls, [i % args.workers for i in range(len(rolls))]))
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    print(f"{'roll':>4}  {'lockstep branch':24s} {'live branch':24s} {'final gap mm':>12} {'min goal mm':>11}  {'live min goal':>13}")
    for res in results:
        tip = np.asarray(res["rows"][-1]["tip"])
        branch = anat.branch_of(anat.nearest_node(frame.world_m_to_vmr_mm(tip)))
        live = live_final(res["roll"], frame)
        row = {"roll": res["roll"], "lockstep_branch": branch, "lockstep_final_tip_m": tip.tolist(),
               "lockstep_min_goal_mm": round(res["min_goal_mm"], 2), "success": res["success"],
               "lockstep_ins_mm": round(res["rows"][-1]["ins_mm"], 1), "wall_s": res["wall_s"], "steps": res["steps"]}
        if live:
            ltip = np.asarray(live["tip_world_m"])
            row.update(live_branch=anat.branch_of(anat.nearest_node(frame.world_m_to_vmr_mm(ltip))),
                       final_gap_mm=round(float(np.linalg.norm(tip - ltip) * 1000), 1), live_ins_mm=round(live["ins_mm"], 1))
            import pandas as pd
            d = pd.read_csv(GAP / f"roll_{res['roll']:03d}_theirs.csv")
            goal = np.asarray(results[0]["rows"][0]["tip"]) * 0 + np.array([0.07001, 0.09756, 0.1535])
            row["live_min_goal_mm"] = round(float(np.min(np.linalg.norm(d[["gw_tip_x", "gw_tip_y", "gw_tip_z"]].to_numpy() - goal, axis=1)) * 1000), 2)
        summary.append(row)
        print(f"{row['roll']:4d}  {branch:24s} {row.get('live_branch', '-'):24s} {row.get('final_gap_mm', float('nan')):12.1f} "
              f"{row['lockstep_min_goal_mm']:11.1f}  {row.get('live_min_goal_mm', float('nan')):13.1f}   ({row['steps']} steps, {row['wall_s']} s)")
    (OUT / "roll_check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {OUT / 'roll_check.json'}")


if __name__ == "__main__":
    main()
