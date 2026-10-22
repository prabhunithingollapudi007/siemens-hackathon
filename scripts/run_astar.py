"""Navigate to a target with A* + the closed-loop route follower, on our simulator or theirs.

    python scripts/run_astar.py --target right_carotid@40 --sim ours
    python scripts/run_astar.py --target right_carotid@40 --sim theirs          # organiser stack running
    python scripts/run_astar.py --target scenario --sim theirs                   # the cas-reference goal
    python scripts/run_astar.py --target left_subclavian@40 --sim theirs --noisy # their noisy tip topics

Devices come from device-parameters.json, the repo's source of truth (STROKE_NAV_DEVICE_PARAMS
selects another file).

Success is the organiser's: guidewire tip within 5 mm of the goal for 3 consecutive 5 Hz steps,
600 steps at most. Writes results/astar/<sim>_<target>.json (+ per-step CSV).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stroke_nav.agents.route_follower import ClosedLoopRouteFollower, FollowerParams, TipObs  # noqa: E402
from stroke_nav.data.anatomy import Anatomy  # noqa: E402
from stroke_nav.organiser.frame import default_frame  # noqa: E402
from stroke_nav.organiser.spec import CAS_REFERENCE_GOALS_CM, DEFAULT_SPEC  # noqa: E402
from stroke_nav.planning.constraints import CATHETER, GUIDEWIRE, get_device  # noqa: E402
from stroke_nav.planning.planner import PathPlanner  # noqa: E402

OUT = ROOT / "results" / "astar"


def build(anat: Anatomy, frame, target: str):
    if target == "scenario":
        goal = frame.obj_cm_to_vmr_mm(np.asarray(CAS_REFERENCE_GOALS_CM[0]))
        node = anat.nearest_node(goal)
    else:
        node = anat.resolve(target)
        goal = anat.nodes[node].copy()
    plan = PathPlanner(anat, GUIDEWIRE).plan(anat.default_start(), node)
    if not plan.feasible:
        raise SystemExit(f"no feasible route: {plan.message}")
    cath_r = get_device(CATHETER).min_lumen_radius_mm
    narrow = np.where(plan.radius < cath_r + 0.5)[0]
    cath_limit = float(plan.s[narrow[0]] - 10.0) if len(narrow) else float(plan.s[-1] - 20.0)
    return goal, node, plan, cath_limit


def quat_z(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])


class OursAdapter:
    def __init__(self, anat, frame, goal_vmr, spec=DEFAULT_SPEC):
        from stroke_nav.organiser.env import CompatConfig, OrganiserCompatEnv
        self.env = OrganiserCompatEnv(CompatConfig(image_mode="blank", noise=False, spec=spec), anatomy=anat, frame=frame)
        self.goal_world = frame.vmr_mm_to_world_m(goal_vmr)

    def reset(self):
        self.env.reset(seed=0, options={"goal_world_m": self.goal_world})
        return self.read()

    def step(self, a):
        self.env.step(a)
        return self.read()

    def read(self):
        from stroke_nav.env.devices import CATH, WIRE
        sim = self.env.inner.sim
        return {"wire_tip": sim.tracking_points(WIRE)[0], "wire_dir": sim.tip_direction(WIRE),
                "wire_ins": float(sim.L[WIRE]), "cath_ins": float(sim.L[CATH]), "force": 0.0}

    def close(self):
        pass


class TheirsAdapter:
    def __init__(self, frame, noisy: bool):
        from stroke_nav.organiser.live.bridge import LiveBridge
        from stroke_nav.organiser.live.renderer import PhysicsPage
        self.page = PhysicsPage().start()
        self.frame, self.noisy = frame, noisy
        self.bridge = LiveBridge().open()
        self.last_true = None

    def reset(self):
        # the bridge sends commands only to the newest page connection: reload ours so it is the newest
        self.page.reload()
        return self.read(self.bridge.reset())

    def step(self, a):
        o = self.read(self.bridge.step(a))
        if o["robot_ins"] - o["wire_ins"] > 40.0:
            raise RuntimeError(f"physics page stopped following the robot (robot {o['robot_ins']:.0f} mm, page "
                               f"{o['wire_ins']:.0f} mm): another browser tab with catheter_guidewire_ros2.html took "
                               "the command channel; close it and rerun")
        return o

    def read(self, obs):
        src = obs["noisy"] if self.noisy and obs["noisy"]["guidewire"] else obs["true"]
        gw, ca = src["guidewire"], src["catheter"]
        self.last_true = obs["true"]["guidewire"]
        tip = self.frame.world_m_to_vmr_mm(np.asarray(gw["tip"]))
        d_world = quat_z(gw["quat"])
        d_vmr = d_world @ self.frame.M.T
        return {"wire_tip": tip, "wire_dir": d_vmr / np.linalg.norm(d_vmr), "wire_ins": gw["insertion_m"] * 1000.0,
                "cath_ins": ca["insertion_m"] * 1000.0, "force": max(gw["peak_force_n"], ca["peak_force_n"]),
                "true_tip": self.frame.world_m_to_vmr_mm(np.asarray(obs["true"]["guidewire"]["tip"])),
                "robot_ins": (obs["joints"][0] - 1.2) * 1000.0 if obs.get("joints") else 0.0}

    def close(self):
        self.bridge.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="right_carotid@40", help="target spec (branch@mm, node id) or 'scenario'")
    ap.add_argument("--sim", choices=("ours", "theirs"), default="ours")
    ap.add_argument("--noisy", action="store_true", help="theirs: steer on their noisy tip topics")
    ap.add_argument("--max-steps", type=int, default=DEFAULT_SPEC.environment.max_steps)
    args = ap.parse_args()

    from stroke_nav.organiser.spec import describe_devices
    spec = DEFAULT_SPEC
    print(f"[astar] {describe_devices(spec)}")

    anat, frame = Anatomy.load("0248_H_AOCERE_CAS"), default_frame()
    goal, node, plan, cath_limit = build(anat, frame, args.target)
    print(f"[astar] {args.target}: {plan.summary()['branches']}, {plan.summary()['length_mm']} mm, "
          f"{len(plan.junctions)} bifurcations; catheter stops at {cath_limit:.0f} mm")
    follower = ClosedLoopRouteFollower(anat, plan, FollowerParams(cath_limit_mm=cath_limit,
                                                                  travel_limit_mm=spec.robot.guidewire_travel_mm - 0.5))
    sim = OursAdapter(anat, frame, goal, spec) if args.sim == "ours" else TheirsAdapter(frame, args.noisy)
    rows, in_goal, success, t0 = [], 0, False, time.monotonic()
    path_len, prev_tip = 0.0, None
    try:
        o = sim.reset()
        for k in range(args.max_steps):
            obs = TipObs(o["wire_tip"], o["wire_dir"], o["wire_ins"], o["cath_ins"], goal)
            a = follower.act(obs)
            o = sim.step(a)
            true_tip = o.get("true_tip", o["wire_tip"])
            dist = float(np.linalg.norm(true_tip - goal))
            if prev_tip is not None:
                path_len += float(np.linalg.norm(true_tip - prev_tip))
            prev_tip = true_tip
            in_goal = in_goal + 1 if dist <= DEFAULT_SPEC.environment.success_radius_m * 1000 else 0
            st = follower.state
            rows.append({"step": k, "gw_trans": round(a[0], 3), "gw_rot": round(a[1], 3), "cath_trans": round(a[2], 3),
                         "mode": st.mode, "junction": st.junction, "s_mm": round(st.s, 1), "dist_mm": round(dist, 2),
                         "wire_ins_mm": round(o["wire_ins"], 1), "cath_ins_mm": round(o["cath_ins"], 1),
                         "branch": anat.branch_of(anat.nearest_node(true_tip)), "force_n": round(o["force"], 4)})
            if k % 10 == 0:
                print(f"  step {k:3d}  {st.mode:9s} junction {st.junction}/{len(follower.junctions)}  s {st.s:6.1f}  "
                      f"wire {o['wire_ins']:6.1f} mm  cath {o['cath_ins']:6.1f}  {rows[-1]['branch']:24s} goal {dist:6.1f} mm", flush=True)
            if in_goal >= DEFAULT_SPEC.environment.success_steps:
                success = True
                break
            if st.mode == "travel_limit":
                for _ in range(3):  # let the rod settle, then stop
                    o = sim.step(np.zeros(4))
                break
    finally:
        sim.close()
    final = rows[-1]["dist_mm"] if rows else float("nan")
    best = min(r["dist_mm"] for r in rows) if rows else float("nan")
    result = {"sim": args.sim, "target": args.target, "route": plan.summary()["branches"], "route_mm": plan.summary()["length_mm"],
              "success": success, "steps": len(rows), "procedure_time_s": round(len(rows) * DEFAULT_SPEC.dt, 1),
              "wall_time_s": round(time.monotonic() - t0, 1), "final_distance_mm": final, "best_distance_mm": best,
              "end_mode": follower.state.mode, "wrong_vessel_events": sum(1 for e in follower.state.log if e["event"] == "wrong_vessel"),
              "tip_path_mm": round(path_len, 1), "max_force_n": max((r["force_n"] for r in rows), default=0.0),
              "rotation_sign_evidence": round(follower.state.rot_evidence, 3), "events": follower.state.log}
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{args.sim}{'_noisy' if args.noisy else ''}_{args.target.replace('@', '-at-')}"
    (OUT / f"{tag}.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    if rows:
        with open(OUT / f"{tag}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"[astar] {'SUCCESS' if success else 'no success'} in {len(rows)} steps; final {final} mm, best {best} mm; "
          f"end {follower.state.mode}; wrong-vessel events {result['wrong_vessel_events']}")


if __name__ == "__main__":
    main()
