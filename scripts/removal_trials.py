"""Compare guidewire withdrawal strategies on the organiser's physics, from the same inserted state.

    python scripts/removal_trials.py --model 0028_H_ABAO_H --target Left_Common_Carotid@30 --roll 120 --insert-to 500
    python scripts/removal_trials.py --target scenario --roll 120 --insert-to 120 --strategies naive,unwind,guarded
    python scripts/removal_trials.py --variant 10Right70Left --target left_carotid@140 --roll -15 --insert-to 150

Each strategy is one episode: the A* + entry-roll controller inserts the guidewire to ``--insert-to`` mm
(choose a depth before its first wall crossing, see docs/SAFETY.md §7), holds 1 s, then a
``stroke_nav.agents.removal.WithdrawalController`` pulls it out. Strategies:

    naive      pull at full speed (10 mm/s)
    slow       pull at 30 % (3 mm/s)
    unwind     turn the shaft back to its starting roll, then pull at full speed
    profiled   speed by depth: the monitor's lumen-radius envelope along the A* route (50 % below 5 mm radius, 25 % below 2 mm)
    guarded    full speed; on a wall crossing (wall guard on the noisy wire): stop, re-advance 3 mm, re-roll, pull at half speed
    unwind_guarded  unwind + guarded (the recommended removal)
    safe       unwind + profiled + guarded

Every step the true wire is checked against the vessel surface (``stroke_nav.safety.wall``). Writes
``results/removal/<case>/trials.json`` and prints a table: withdrawal time, wall crossings during insertion
and withdrawal, deepest point beyond the wall, peak contact and pull force, fastest tip motion (whip).
Robot profile: ``STROKE_NAV_ROBOT`` (e.g. ``long-reach`` for femoral entries). One lockstep job at a time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stroke_nav.agents.entry_roll import EntryRollExpert  # noqa: E402
from stroke_nav.agents.removal import InsertThenWithdraw, WithdrawalController  # noqa: E402
from stroke_nav.data.anatomy import Anatomy  # noqa: E402
from stroke_nav.organiser.frame import entry_node, frame_for  # noqa: E402
from stroke_nav.organiser.live.lockstep import kill_stray_pages, make_env, other_lockstep_jobs  # noqa: E402
from stroke_nav.organiser.live.rollout import wall_guard_for  # noqa: E402
from stroke_nav.organiser.spec import CAS_REFERENCE_GOALS_CM, DEFAULT_SPEC  # noqa: E402
from stroke_nav.safety.wall import guidewire_bodies  # noqa: E402

FAR_GOAL_M = (1.0, 1.0, 1.0)  # never reached: their success rule must not end the episode
STRATEGIES = ("naive", "slow", "unwind", "profiled", "guarded", "unwind_guarded", "safe")


def speed_envelope(anat, frame, target_node: int):
    """Insertion mm -> pull speed fraction: the safety monitor's lumen-radius envelope (R1: 100 % in lumens of
    5 mm radius or more, 50 % from 2 mm, 25 % below) at the point of the A* route the tip is at, taking the
    insertion depth as the arc length along the route (the narrowest lumen within +-10 mm)."""
    from stroke_nav.planning.candidates import _allowed_speed
    from stroke_nav.planning.constraints import GUIDEWIRE
    from stroke_nav.planning.planner import PathPlanner
    from stroke_nav.safety.monitor import SafetyConfig

    plan = PathPlanner(anat, GUIDEWIRE).plan(entry_node(anat, frame), target_node)
    pts = anat.nodes[plan.nodes]
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    radius, cfg = anat.radius[plan.nodes], SafetyConfig()
    return lambda ins_mm: float(_allowed_speed(float(radius[(s >= ins_mm - 10.0) & (s <= ins_mm + 10.0)].min()
                                                     if np.any((s >= ins_mm - 10.0) & (s <= ins_mm + 10.0)) else radius[-1]), cfg))


def make_controller(name: str, roll: float, insert_to: float, guard, envelope) -> InsertThenWithdraw:
    kw = {"naive": {}, "slow": {"speed": 0.3}, "unwind": {"unwind_first": True}, "profiled": {"speed_by_depth": envelope},
          "guarded": {"guard": guard},
          "unwind_guarded": {"unwind_first": True, "guard": guard}, "safe": {"unwind_first": True, "speed_by_depth": envelope, "guard": guard}}[name]
    return InsertThenWithdraw(EntryRollExpert(roll_deg=roll, slow_radius_mm=8.0), WithdrawalController(**kw), insert_to)


def trial(worker: int, jobs: list[dict], args, scen: dict, envelope) -> list[dict]:
    guard_true = wall_guard_for(scen.get("scenario_name", "cas-reference"), scen.get("scenario_root"))
    guard_noisy = wall_guard_for(scen.get("scenario_name", "cas-reference"), scen.get("scenario_root"))
    env = make_env(worker=worker, seed=worker, goals=[FAR_GOAL_M], max_steps=args.max_steps, visible=args.visible, **scen)
    out = []
    try:
        for job in jobs:
            ctl = make_controller(job["strategy"], args.roll, args.insert_to, guard_noisy, envelope)
            ctl.bind(env)
            obs, _ = env.reset(seed=args.seed)
            ctl.reset()
            rows, last_tip, t0 = [], None, time.monotonic()
            for step in range(args.max_steps):
                obs, _, term, trunc, _ = env.step(ctl.act(obs))
                s = env.bridge.true["guidewire"][-1]
                tip = np.asarray(s["tip"], float) * 1000.0
                c = guard_true.check(guidewire_bodies(s), float(s["ins"]) * 1000.0)
                rows.append({"phase": ctl.phase, "ins_mm": float(s["ins"]) * 1000.0, "rot_deg": float(np.degrees(s["rot"])),
                             "cross": bool(c.crossing), "depth_mm": c.depth_mm, "peak_n": float(s["peak"]),
                             "proximal_n": abs(float(s["proximal"])),
                             "tip_speed_mm_s": 0.0 if last_tip is None else float(np.linalg.norm(tip - last_tip)) / DEFAULT_SPEC.dt})
                last_tip = tip
                if ctl.done or term or trunc:
                    break
            ins_rows = [r for r in rows if r["phase"] != "withdraw"]
            w = [r for r in rows if r["phase"] == "withdraw"]
            res = {"strategy": job["strategy"], "outcome": ctl.out.mode if ctl.phase == "withdraw" else f"stopped in {ctl.phase}",
                   "inserted_mm": round(max((r["ins_mm"] for r in rows), default=0.0), 1),
                   "insert_cross_steps": sum(r["cross"] for r in ins_rows),
                   "withdraw_s": round(len(w) * DEFAULT_SPEC.dt, 1), "final_ins_mm": round(rows[-1]["ins_mm"], 1) if rows else None,
                   "withdraw_cross_steps": sum(r["cross"] for r in w), "withdraw_depth_mm": round(max((r["depth_mm"] for r in w), default=0.0), 2),
                   "withdraw_peak_mN": round(1000 * max((r["peak_n"] for r in w), default=0.0), 1),
                   "withdraw_pull_mN": round(1000 * max((r["proximal_n"] for r in w), default=0.0), 1),
                   "withdraw_tip_speed_max_mm_s": round(max((r["tip_speed_mm_s"] for r in w), default=0.0), 1),
                   "roll_at_start_deg": round(w[0]["rot_deg"], 1) if w else None, "events": ctl.out.events,
                   "wall_s": round(time.monotonic() - t0, 1), "rows": rows}
            out.append(res)
            print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "events")}), flush=True)
    finally:
        env.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="0248_H_AOCERE_CAS")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--target", required=True, help="target spec of the route (speed envelope), or 'scenario' (cas-reference goal)")
    ap.add_argument("--roll", type=float, required=True, help="entry roll of the inserting controller (deg)")
    ap.add_argument("--insert-to", type=float, required=True, help="insertion depth (mm) to withdraw from")
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--visible", action="store_true")
    ap.add_argument("--case-id", default=None)
    args = ap.parse_args()
    others = other_lockstep_jobs()
    if others:
        raise SystemExit("another lockstep job is running:\n  " + "\n  ".join(others))
    kill_stray_pages()

    from stroke_nav.organiser.scenario import DEFAULT_SCENARIO_ROOT, scenario_name

    anat, frame = Anatomy.load(args.model, args.variant), frame_for(args.model)
    if args.model == "0248_H_AOCERE_CAS" and not args.variant:
        scen = {}
    else:
        scen = {"scenario_name": scenario_name(args.model, args.variant), "scenario_root": Path(DEFAULT_SCENARIO_ROOT).resolve()}
    if args.target == "scenario":
        node = anat.nearest_node(frame.obj_cm_to_vmr_mm(np.asarray(CAS_REFERENCE_GOALS_CM[0])))
    else:
        node = anat.resolve(args.target)
    envelope = speed_envelope(anat, frame, node)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    jobs = [{"strategy": s} for s in strategies]
    n = 1 if args.visible else max(1, min(args.workers, len(jobs)))
    with ThreadPoolExecutor(n) as ex:
        parts = list(ex.map(lambda w: trial(w, jobs[w::n], args, scen, envelope), range(n)))
    results = sorted((r for p in parts for r in p), key=lambda r: strategies.index(r["strategy"]))
    case = args.case_id or f"{scen.get('scenario_name', 'cas-reference')}-from-{args.insert_to:g}mm"
    out = ROOT / "results" / "removal" / case
    out.mkdir(parents=True, exist_ok=True)
    (out / "trials.json").write_text(json.dumps({"args": vars(args), "robot": DEFAULT_SPEC.robot.name, "trials": results}), encoding="utf-8")
    print(f"\n{case} (robot {DEFAULT_SPEC.robot.name}, roll {args.roll:g} deg, inserted to {args.insert_to:g} mm)")
    print("| Strategy | Outcome | Withdrawal | Wall crossed (insert / withdraw) | Deepest | Peak force | Pull force | Fastest tip |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['strategy']} | {r['outcome']} (left {r['final_ins_mm']} mm) | {r['withdraw_s']} s | {r['insert_cross_steps']} / "
              f"{r['withdraw_cross_steps']} steps | {r['withdraw_depth_mm']} mm | {r['withdraw_peak_mN']} mN | {r['withdraw_pull_mN']} mN | "
              f"{r['withdraw_tip_speed_max_mm_s']} mm/s |")
    print(f"-> {out / 'trials.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
