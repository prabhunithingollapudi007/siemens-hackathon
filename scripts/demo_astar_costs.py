"""Demo: the enhanced constrained A* on the 0248 stenosis variants, judged on four evaluations,
executed on the organiser's physics.

For each variant and lesion target:

1. **Plan** with four weight profiles (``shortest``: inserted length; ``fastest``: estimated time;
   ``gentle``: damage terms; ``balanced``), the robot's guidewire from device-parameters.json, and the
   hard constraints (lumen clearance, forbidden vessels). A lesion that is too narrow for the device
   moves the target to the nearest feasible point proximal to it. A hard forbidden-vessel check
   forbids the route's first vessel change and reports the reroute, or that none exists.
2. **Evaluate before anything moves**: inserted length against the robot's travel (robot size),
   estimated time from the speed envelope (reach time), curvature integral, bending effort, length in
   narrow lumen and the force proxy (damage to arteries), bifurcations and vessel changes (wrong-artery
   exposure).
3. **Execute** each distinct route closed-loop on the organiser's physics: their pages run the
   variant's mesh as a scenario (``stroke_nav.organiser.scenario``) in the lockstep env, and the A*
   route follower steers on the noisy tip state. Actual time, final distance, peak force, the
   follower's wrong-vessel retries and the trajectory safety checks (``planning.candidates.assess``)
   complete the four evaluations. ``--sim ours`` executes in the kinematic simulator instead.
4. **Replan**: from the measured tip at the end of the executed episode, and from a point 15 mm into
   the wrong vessel at the route's first vessel change, warm-started with the previous route; against
   a cold plan from the same spot (expanded nodes, time, retraction).

    python scripts/demo_astar_costs.py                                           # both variants, lesions on both sides, their physics
    python scripts/demo_astar_costs.py --variants 10Right70Left --targets left_carotid@140 --episodes 1
    python scripts/demo_astar_costs.py --sim ours                                # kinematic simulator only (seconds)
    python scripts/demo_astar_costs.py --forbid left_vertebral,right_vertebral  # extra hard forbidden vessels

Writes results/astar_demo/<variant>/<target>/routes.png, replan.png, summary.md, summary.json and
executed_<profile>.json (the episodes in the common candidate format), plus results/astar_demo/summary.md.
Their physics runs in the lockstep env: stop their Docker stack first; one lockstep job at a time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_astar import OursAdapter  # noqa: E402

from stroke_nav.agents.entry_roll import EntryRollExpert  # noqa: E402
from stroke_nav.agents.roll_table import RollTable  # noqa: E402
from stroke_nav.agents.route_follower import ClosedLoopRouteFollower, FollowerParams, TipObs  # noqa: E402
from stroke_nav.data.anatomy import Anatomy  # noqa: E402
from stroke_nav.organiser.frame import default_frame  # noqa: E402
from stroke_nav.organiser.spec import DEFAULT_SPEC  # noqa: E402
from stroke_nav.planning.candidates import Candidate, EpisodeTrace, assess  # noqa: E402
from stroke_nav.planning.constraints import CATHETER, GUIDEWIRE, VesselGeometry, get_device  # noqa: E402
from stroke_nav.planning.planner import PathPlanner, Plan, PlannerWeights, evaluate_plan, nearest_feasible_target  # noqa: E402

PROFILES = ("shortest", "fastest", "gentle", "balanced")
COLORS = {"shortest": "#2563eb", "fastest": "#d97706", "gentle": "#059669", "balanced": "#7c3aed"}
SUCCESS_MM = DEFAULT_SPEC.environment.success_radius_m * 1000.0
SUCCESS_STEPS = DEFAULT_SPEC.environment.success_steps
TRAVEL = DEFAULT_SPEC.robot.guidewire_travel_mm


def log(msg: str) -> None:
    print(f"[astar] {msg}", flush=True)


def follower_params(plan: Plan) -> FollowerParams:
    cath_r = get_device(CATHETER).min_lumen_radius_mm
    narrow = np.where(plan.radius < cath_r + 0.5)[0]
    cath_limit = float(plan.s[narrow[0]] - 10.0) if len(narrow) else float(plan.s[-1] - 20.0)
    return FollowerParams(cath_limit_mm=cath_limit, travel_limit_mm=TRAVEL - 0.5)


# ---------------------------------------------------------------------- execution
def execute_ours(anat, frame, plan: Plan, goal_vmr, max_steps: int) -> tuple[list[EpisodeTrace], dict]:
    """Closed-loop execution in the kinematic simulator (one deterministic episode)."""
    follower = ClosedLoopRouteFollower(anat, plan, follower_params(plan))
    sim = OursAdapter(anat, frame, goal_vmr, DEFAULT_SPEC)
    o = sim.reset()
    in_goal, success, tips, ins, acts, dists = 0, False, [], [], [], []
    for _ in range(max_steps):
        a = np.asarray(follower.act(TipObs(o["wire_tip"], o["wire_dir"], o["wire_ins"], o["cath_ins"], goal_vmr)), np.float32)
        o = sim.step(a)
        dist = float(np.linalg.norm(o["wire_tip"] - goal_vmr))
        tips.append(frame.vmr_mm_to_world_m(o["wire_tip"]))
        ins.append(float(o["wire_ins"]))
        acts.append(a)
        dists.append(dist)
        in_goal = in_goal + 1 if dist <= SUCCESS_MM else 0
        if in_goal >= SUCCESS_STEPS:
            success = True
            break
        if follower.state.mode == "travel_limit":
            break
    trace = EpisodeTrace(seed=0, success=success, tip_m=np.array(tips), insertion_mm=np.array(ins), actions=np.array(acts),
                         dist_mm=np.array(dists), peak_force_n=0.0, reason="success" if success else follower.state.mode)
    events = {0: sum(1 for e in follower.state.log if e["event"] == "wrong_vessel")}
    return [trace], events


def execute_theirs(anat, frame, plan: Plan, goal_vmr, model: str, variant: str, seeds: list[int], workers: int,
                   max_steps: int, tag: str, visible: bool = False, roll_deg: float | None = None) -> tuple[list[EpisodeTrace], dict]:
    """Closed-loop execution on the organiser's physics (lockstep), the variant's mesh as their scenario.
    ``roll_deg``: execute with the entry-roll controller at that roll (their rod picks the branch at the
    entry); None: the A* route follower steering on the noisy tip state."""
    from stroke_nav.organiser.live.rollout import RouteFollowerOnObservation, lockstep_pool
    from stroke_nav.organiser.scenario import DEFAULT_SCENARIO_ROOT, scenario_name, write_scenario

    name = scenario_name(model, variant)
    if not (Path(DEFAULT_SCENARIO_ROOT) / name / "scenario.json").is_file():
        write_scenario(model, variant, [goal_vmr], DEFAULT_SCENARIO_ROOT, frame)
        log(f"wrote scenario {name} for their physics")
    goal_m = np.asarray(frame.vmr_mm_to_world_m(goal_vmr), float)
    params = follower_params(plan)
    events: dict[int, int] = {}

    def on_episode(job, trace, policy):
        events[job["seed"]] = getattr(policy, "wrong_vessel_events", 0)

    if roll_deg is not None:
        make_policy = lambda w: EntryRollExpert(roll_deg=roll_deg, slow_radius_mm=8.0)  # noqa: E731
    else:
        make_policy = lambda w: RouteFollowerOnObservation(anat, plan, params, frame, goal_vmr)  # noqa: E731
    res = lockstep_pool([{"seed": s} for s in seeds], make_policy,
                        1 if visible else workers, max_steps, goal_m, tag=tag, on_episode=on_episode, scenario_name=name,
                        scenario_root=Path(DEFAULT_SCENARIO_ROOT), visible=visible)
    return [r[1] for r in res], events


def executed_summary(traces: list[EpisodeTrace], events: dict, report: dict) -> dict:
    ok = [t for t in traces if t.success]
    return {"episodes": len(traces), "success": f"{len(ok)}/{len(traces)}",
            "time_to_goal_s": round(float(np.mean([t.steps for t in ok])) * DEFAULT_SPEC.dt, 1) if ok else None,
            "time_s": round(float(np.mean([t.steps for t in traces])) * DEFAULT_SPEC.dt, 1) if traces else None,
            "median_final_dist_mm": round(float(np.median([t.final_dist_mm for t in traces])), 1) if traces else None,
            "median_min_dist_mm": round(float(np.median([t.min_dist_mm for t in traces])), 1) if traces else None,
            "max_insertion_mm": round(max(float(np.max(t.insertion_mm)) for t in traces), 1) if traces else None,
            "peak_force_n": round(max(t.peak_force_n for t in traces), 4) if traces else None,
            "wrong_vessel_retries": sum(events.values()), "end_reasons": sorted({t.reason for t in traces}),
            "safety_verdict": report.get("verdict"), "safety_flags": report.get("flags", []),
            "worst_off_route_mm": report.get("worst", {}).get("off_route_mm"), "stall_mm": report.get("worst", {}).get("stall_mm"),
            "wrong_vessel_depth_mm": report.get("worst", {}).get("wrong_vessel_depth_mm")}


# ---------------------------------------------------------------------- replanning
def wrong_vessel_point(anat, plan: Plan, depth_mm: float = 15.0):
    """A point ``depth_mm`` into a sibling vessel at the first vessel change of the route (where a
    wrong entry would happen), or None when the route changes vessel nowhere."""
    for j in plan.junctions:
        if not j["changes_vessel"]:
            continue
        v, i = j["node"], j["index"]
        u, w = plan.nodes[i - 1], plan.nodes[i + 1]
        for nb in anat.neighbors[v]:
            if nb in (u, w):
                continue
            far = anat.walk_from(v, nb, int(depth_mm))
            return anat.nodes[far], far, j
    return None


def replan_report(planner: PathPlanner, anat, base: Plan, point_vmr, node: int, label: str) -> dict:
    t0 = time.perf_counter()
    warm = planner.replan(point_vmr, node, previous=base)
    warm_ms = (time.perf_counter() - t0) * 1000
    tip_node = anat.nearest_node(point_vmr)
    t0 = time.perf_counter()
    cold = planner.plan(tip_node, node)
    cold_ms = (time.perf_counter() - t0) * 1000
    out = {"from": label, "tip_branch": anat.branch_of(tip_node), "on_previous_route": bool(tip_node in base.nodes),
           "warm": {"method": warm.method, "feasible": warm.feasible, "ms": round(warm_ms, 1), "expanded": warm.expanded,
                    "rejoin_mm": round(warm.costs.get("rejoin_mm", 0.0), 1) if warm.feasible else None,
                    "length_mm": round(warm.length_mm, 1), "branches": warm.branches() if warm.feasible else [],
                    "cost": round(warm.cost, 1) if warm.feasible else None},
           "cold": {"method": cold.method, "feasible": cold.feasible, "ms": round(cold_ms, 1), "expanded": cold.expanded,
                    "length_mm": round(cold.length_mm, 1), "cost": round(cold.cost, 1) if cold.feasible else None}}
    log(f"  replan {label} (tip in {out['tip_branch']}): {warm.method} {warm.expanded} nodes / {warm_ms:.0f} ms"
        + (f", retract {out['warm']['rejoin_mm']} mm, {warm.length_mm:.0f} mm to go" if warm.feasible else f" infeasible: {warm.message}")
        + f"; cold plan {cold.expanded} nodes / {cold_ms:.0f} ms")
    return out, warm


# ---------------------------------------------------------------------- plots
def _views(ax_ap, ax_lat, anat):
    for name in anat.branch_names:
        p = anat.nodes[anat.branch_nodes(name)]
        ax_ap.plot(-p[:, 0], p[:, 2], color="#9ca3af", lw=1)
        ax_lat.plot(p[:, 1], p[:, 2], color="#9ca3af", lw=1)


def plot_routes(path: Path, anat, frame, plans: dict[str, Plan], executed: dict[str, list[EpisodeTrace]], goal, title: str, sim: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    _views(axes[0], axes[1], anat)
    for k, (prof, plan) in enumerate(plans.items()):
        if not plan.feasible:
            continue
        p = plan.points
        lw = 3.0 - 0.5 * k
        label = f"{prof}: {plan.length_mm:.0f} mm, ~{plan.costs['est_time_s']:.0f} s, bends {plan.costs['bend_effort']:.1f}"
        axes[0].plot(-p[:, 0], p[:, 2], color=COLORS[prof], lw=lw, alpha=0.8, label=label)
        axes[1].plot(p[:, 1], p[:, 2], color=COLORS[prof], lw=lw, alpha=0.8)
        k_travel = int(np.searchsorted(plan.s, TRAVEL))
        if k_travel < len(p):
            axes[0].plot(-p[k_travel, 0], p[k_travel, 2], "x", color=COLORS[prof], ms=9, mew=2)
            axes[1].plot(p[k_travel, 1], p[k_travel, 2], "x", color=COLORS[prof], ms=9, mew=2)
    for prof, traces in executed.items():
        if not traces:
            continue
        rep = min(traces, key=lambda t: (not t.success, t.min_dist_mm))
        q = frame.world_m_to_vmr_mm(rep.tip_m)
        axes[0].plot(-q[:, 0], q[:, 2], "--", color="#dc2626", lw=1.5,
                     label=f"executed on {'their physics' if sim == 'theirs' else 'our simulator'} ({prof} route, seed {rep.seed}: "
                           f"{'success' if rep.success else f'{rep.reason}, min {rep.min_dist_mm:.0f} mm'})")
        axes[1].plot(q[:, 1], q[:, 2], "--", color="#dc2626", lw=1.5)
        axes[0].plot(-q[-1, 0], q[-1, 2], "o", color="#dc2626", ms=5)
        axes[1].plot(q[-1, 1], q[-1, 2], "o", color="#dc2626", ms=5)
    for ax, (i, j, sx) in zip(axes, ((0, 2, -1), (1, 2, 1))):
        ax.plot(sx * goal[i], goal[j], "*", color="#16a34a", ms=16, label="target" if ax is axes[0] else None)
        ax.add_patch(plt.Circle((sx * goal[i], goal[j]), SUCCESS_MM, fill=False, color="#16a34a", lw=1))
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    axes[0].plot([], [], "kx", ms=9, mew=2, label=f"robot travel limit ({TRAVEL:g} mm)")
    axes[0].set_title("AP view (patient left on right)")
    axes[0].set_xlabel("-x (mm)")
    axes[0].set_ylabel("z (mm)")
    axes[1].set_title("Lateral view")
    axes[1].set_xlabel("y (mm, anterior)")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_replan(path: Path, anat, previous: Plan, replans: list[tuple[np.ndarray, Plan, str]], goal, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 9))
    for name in anat.branch_names:
        p = anat.nodes[anat.branch_nodes(name)]
        ax.plot(-p[:, 0], p[:, 2], color="#9ca3af", lw=1)
    ax.plot(-previous.points[:, 0], previous.points[:, 2], color="#7c3aed", lw=4, alpha=0.3, label="previous route")
    cols = ("#dc2626", "#0891b2")
    zs = [goal[2]]
    for (tip, plan, label), col in zip(replans, cols):
        if plan.feasible:
            ax.plot(-plan.points[:, 0], plan.points[:, 2], color=col, lw=2,
                    label=f"{label}: {plan.method}, {plan.costs.get('rejoin_mm', 0):.0f} mm back, {plan.expanded} nodes")
        ax.plot(-tip[0], tip[2], "o", color=col, ms=9)
        zs.append(tip[2])
    ax.plot(-goal[0], goal[2], "*", color="#16a34a", ms=16, label="target")
    ax.set_ylim(min(zs) - 60, max(zs) + 30)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("-x (mm)")
    ax.set_ylabel("z (mm)")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------- one case
def run_case(model: str, variant: str, target: str, out: Path, forbid: tuple[str, ...], sim: str, seeds: list[int],
             workers: int, max_steps: int, reuse: bool = False, visible: bool = False, table: RollTable | None = None) -> dict:
    anat, frame = Anatomy.load(model, variant), default_frame()
    geo = VesselGeometry(anat)
    start = anat.default_start()
    lesion = anat.resolve(target)
    planner0 = PathPlanner(anat, GUIDEWIRE, PlannerWeights.profile("balanced", forbidden_branches=forbid), geo)
    node, back = nearest_feasible_target(planner0, lesion)
    goal = anat.nodes[node].copy()
    case = {"model": model, "variant": variant, "target": target, "simulator": sim, "lesion_node": int(lesion),
            "lesion_lumen_radius_mm": round(float(anat.radius[lesion]), 2), "planned_node": int(node),
            "moved_proximally_mm": round(back, 1), "forbidden": list(forbid), "travel_mm": TRAVEL, "device": planner0.device.name,
            "profiles": {}}
    log(f"{variant} {target}: lesion lumen radius {anat.radius[lesion]:.2f} mm" + (f", planning to {back:.0f} mm proximal of it" if back else ""))
    roll_deg, roll_info = None, None
    if table is not None and sim == "theirs":
        goal_m = frame.vmr_mm_to_world_m(goal)
        roll_deg, miss = table.reach(goal_m, robust_within_mm=SUCCESS_MM)
        path = next(p for p in table.paths if p.roll_deg == roll_deg)
        end_branch = anat.branch_of(anat.nearest_node(frame.world_m_to_vmr_mm(path.tip_m[-1])))
        roll_info = {"roll_deg": roll_deg, "recorded_paths_miss_mm": round(miss, 2), "reachable_on_their_physics": bool(miss <= SUCCESS_MM),
                     "roll_path_end_branch": end_branch}
        log(f"  entry roll {roll_deg:g} deg from the calibrated table: recorded paths miss the target by {miss:.1f} mm, end in {end_branch}")
    case["entry_roll"] = roll_info

    # 1 + 2: plan with every profile and evaluate
    plans: dict[str, Plan] = {}
    for prof in PROFILES:
        planner = PathPlanner(anat, GUIDEWIRE, PlannerWeights.profile(prof, forbidden_branches=forbid), geo)
        t0 = time.perf_counter()
        plan = planner.plan(start, node)
        ms = (time.perf_counter() - t0) * 1000
        plans[prof] = plan
        case["profiles"][prof] = {"plan_ms": round(ms, 1), "expanded": plan.expanded, "evaluation": evaluate_plan(plan, TRAVEL),
                                  "branches": plan.branches() if plan.feasible else [],
                                  "costs": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in plan.costs.items()}}
        if plan.feasible:
            ev = case["profiles"][prof]["evaluation"]
            log(f"  {prof:9s} {plan.length_mm:6.1f} mm ({'within' if ev['robot_size']['within_travel'] else 'BEYOND'} travel)  "
                f"~{plan.costs['est_time_s']:5.1f} s  bend {plan.costs['bend_effort']:5.2f}  turn {plan.costs['curvature_integral_rad']:4.2f} rad  "
                f"narrow {plan.costs['narrow_mm']:5.1f} mm  force {plan.costs['force_proxy']:5.2f}  changes {plan.costs['vessel_changes']}  "
                f"{' > '.join(plan.branches())}  [{ms:.0f} ms, {plan.expanded} nodes]")
        else:
            log(f"  {prof:9s} infeasible: {plan.message}")

    # hard forbidden-vessel check: forbid the route's first vessel change
    base = plans["balanced"]
    forbidden_check = None
    if base.feasible:
        first = next((j for j in base.junctions if j["changes_vessel"]), None)
        if first is not None:
            fb = forbid + (first["to_branch"],)
            alt = PathPlanner(anat, GUIDEWIRE, PlannerWeights.profile("balanced", forbidden_branches=fb), geo).plan(start, node)
            forbidden_check = {"forbidden": first["to_branch"], "feasible": alt.feasible,
                               "route": alt.branches() if alt.feasible else None, "length_mm": round(alt.length_mm, 1) if alt.feasible else None,
                               "message": alt.message}
            log(f"  forbidding {first['to_branch']}: " + (f"reroute {' > '.join(alt.branches())}, {alt.length_mm:.0f} mm" if alt.feasible
                                                          else f"no route ({alt.message})"))
    case["forbidden_check"] = forbidden_check

    # 3: execute each distinct route
    executed: dict[str, list[EpisodeTrace]] = {}
    by_route: dict[tuple, str] = {}
    for prof, plan in plans.items():
        if not plan.feasible:
            continue
        key = tuple(plan.nodes)
        if key in by_route:
            case["profiles"][prof]["executed"] = dict(case["profiles"][by_route[key]]["executed"], same_route_as=by_route[key])
            continue
        by_route[key] = prof
        saved = out / f"executed_{prof}.json"
        if reuse and saved.is_file():
            prev = Candidate.load(saved)
            traces, events = prev.episodes, {0: int(prev.notes[0].rsplit(" ", 1)[-1]) if prev.notes else 0}
            log(f"  reusing {saved.name} ({len(traces)} episodes)")
        elif sim == "theirs":
            traces, events = execute_theirs(anat, frame, plan, goal, model, variant, seeds, workers, max_steps, f"{variant}/{target}/{prof}", visible, roll_deg)
        else:
            traces, events = execute_ours(anat, frame, plan, goal, max_steps)
        cand = Candidate(f"astar_{prof}", "theirs-lockstep" if sim == "theirs" else "ours-kinematic",
                         f"A* ({prof} profile) + " + (f"entry-roll controller at {roll_deg:g} deg" if roll_deg is not None else "closed-loop route follower"), plan_wall_s=sum(t.wall_s for t in traces), episodes=traces,
                         notes=[f"route {' > '.join(plan.branches())}, {plan.length_mm:.0f} mm; follower wrong-vessel retries {sum(events.values())}"])
        cand.save(out / f"executed_{prof}.json")
        report = assess(cand, anat, frame, plan.points, set(plan.nodes), TRAVEL)
        executed[prof] = traces
        case["profiles"][prof]["executed"] = executed_summary(traces, events, report)
        ex = case["profiles"][prof]["executed"]
        log(f"  executed {prof}: {ex['success']}, {ex['time_s']} s, final {ex['median_final_dist_mm']} mm, max insertion {ex['max_insertion_mm']} mm, "
            f"peak force {ex['peak_force_n']} N, wrong-vessel retries {ex['wrong_vessel_retries']}, safety {ex['safety_verdict']} {ex['safety_flags']}")

    # 4: replan from the executed tip and from a synthetic wrong-vessel point
    replans, drawn = [], []
    if base.feasible:
        planner = PathPlanner(anat, GUIDEWIRE, PlannerWeights.profile("balanced", forbidden_branches=forbid), geo)
        traces = executed.get("balanced") or next(iter(executed.values()), [])
        if traces:
            rep = min(traces, key=lambda t: (not t.success, t.min_dist_mm))
            tip = frame.world_m_to_vmr_mm(rep.tip_m[-1])
            info, plan_r = replan_report(planner, anat, base, tip, node, f"the executed tip (seed {rep.seed}, {rep.reason})")
            replans.append(info)
            drawn.append((tip, plan_r, "from the executed tip"))
        wv = wrong_vessel_point(anat, base)
        if wv is not None:
            tip, tip_node, junction = wv
            info, plan_r = replan_report(planner, anat, base, tip, node, f"15 mm into the wrong vessel at {junction['from_branch']} > {junction['to_branch']}")
            replans.append(info)
            drawn.append((tip, plan_r, "from 15 mm into the wrong vessel"))
        t0 = time.perf_counter()
        resumed = planner.replan(base.nodes[len(base.nodes) // 2], node, previous=base)
        replans.append({"from": "a point on the route", "warm": {"method": resumed.method, "ms": round((time.perf_counter() - t0) * 1000, 2),
                                                                 "expanded": resumed.expanded}})
        if drawn:
            plot_replan(out / "replan.png", anat, base, drawn, goal, f"{variant} {target}: local replans (warm start on the previous route)")
    case["replans"] = replans
    plot_routes(out / "routes.png", anat, frame, plans, executed, goal, f"{variant} {target}: routes by cost profile (x = robot travel limit)", sim)

    # reports
    lines = [f"# {variant} / {target} ({'organiser physics, lockstep' if sim == 'theirs' else 'kinematic simulator'})", "",
             f"- Lesion node {lesion} in `{anat.branch_of(lesion)}`, lumen radius {anat.radius[lesion]:.2f} mm; device {planner0.device.name} "
             f"needs {planner0.device.min_lumen_radius_mm:.2f} mm" + (f"; planning to {back:.0f} mm proximal of the lesion." if back else "."),
             f"- Robot guidewire travel {TRAVEL:g} mm." + (f" Forbidden vessels: {', '.join(forbid)}." if forbid else "")]
    if roll_info:
        lines.append(f"- Executed with the entry-roll controller at {roll_info['roll_deg']:g}° (calibrated on their physics; recorded paths miss the "
                     f"target by {roll_info['recorded_paths_miss_mm']} mm and end in `{roll_info['roll_path_end_branch']}`; the A* route ends in "
                     f"`{plans['balanced'].branches()[-1] if plans['balanced'].feasible else '?'}`).")
    if forbidden_check:
        lines.append(f"- Hard constraint check, `{forbidden_check['forbidden']}` forbidden: "
                     + (f"reroute {' > '.join(forbidden_check['route'])}, {forbidden_check['length_mm']:.0f} mm." if forbidden_check["feasible"]
                        else f"no route ({forbidden_check['message']})."))
    lines += ["", "## Routes and their evaluation", "",
              "| Profile | Route | Length (mm) | Within travel | Est. time (s) | Bend effort | Turning (rad) | Narrow (mm) | Force proxy | Bifurcations | Vessel changes | Plan (ms) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for prof, row in case["profiles"].items():
        plan = plans[prof]
        if not plan.feasible:
            lines.append(f"| {prof} | infeasible: {plan.message} | | | | | | | | | | |")
            continue
        ev, c = row["evaluation"], plan.costs
        lines.append(f"| {prof} | {' > '.join(plan.branches())} | {plan.length_mm:.0f} | {'yes' if ev['robot_size']['within_travel'] else '**no**'} | "
                     f"{c['est_time_s']:.0f} | {c['bend_effort']:.2f} | {c['curvature_integral_rad']:.2f} | {c['narrow_mm']:.0f} | {c['force_proxy']:.2f} | "
                     f"{c['bifurcations']} | {c['vessel_changes']} | {row['plan_ms']:.0f} |")
    lines += ["", f"## Executed on {'their physics' if sim == 'theirs' else 'our simulator'}", "",
              "| Route of | Episodes | Success | Time (s) | Final dist. (mm) | Min dist. (mm) | Max insertion (mm) | Peak force (N) | Wrong-vessel retries | Safety | Flags |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for prof, row in case["profiles"].items():
        ex = row.get("executed")
        if not ex or "same_route_as" in ex:
            continue
        lines.append(f"| {prof} | {ex['episodes']} | {ex['success']} | {ex['time_s']} | {ex['median_final_dist_mm']} | {ex['median_min_dist_mm']} | "
                     f"{ex['max_insertion_mm']} | {ex['peak_force_n']} | {ex['wrong_vessel_retries']} | {ex['safety_verdict']} | {', '.join(ex['safety_flags']) or '-'} |")
    same = [f"{p} (same route as {r['executed']['same_route_as']})" for p, r in case["profiles"].items() if "same_route_as" in r.get("executed", {})]
    if same:
        lines.append("")
        lines.append("Profiles sharing a route were executed once: " + ", ".join(same) + ".")
    if replans:
        lines += ["", "## Replanning", ""]
        for r in replans:
            w = r["warm"]
            if "cold" in r:
                c = r["cold"]
                lines.append(f"- From {r['from']} (tip in `{r['tip_branch']}`): {w['method']}, {w['expanded']} nodes in {w['ms']:.0f} ms"
                             + (f", {w['rejoin_mm']} mm back to the route, {w['length_mm']:.0f} mm to go" if w["feasible"] else ", infeasible")
                             + f"; cold plan {c['expanded']} nodes in {c['ms']:.0f} ms.")
            else:
                lines.append(f"- From {r['from']}: {w['method']} in {w['ms']:.2f} ms, {w['expanded']} nodes expanded.")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(case, indent=1, default=float), encoding="utf-8")
    return case


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="0248_H_AOCERE_CAS")
    ap.add_argument("--variants", default="10Right70Left,25Right50Left")
    ap.add_argument("--targets", default="left_carotid@140,right_carotid@118", help="the ICA lesions of the variants")
    ap.add_argument("--forbid", default="", help="comma-separated vessels the route may never use (hard constraint)")
    ap.add_argument("--sim", choices=("theirs", "ours"), default="theirs", help="execute on the organiser's physics (lockstep) or our simulator")
    ap.add_argument("--episodes", type=int, default=1, help="their physics: episodes (noise seeds) per distinct route")
    ap.add_argument("--visible", action="store_true", help="their physics: show the physics page and the fluoro viewer in Chrome windows (one worker)")
    ap.add_argument("--controller", choices=("roll", "follower"), default="roll",
                    help="their physics: execute the route with the calibrated entry-roll controller (their rod picks the branch at the entry) or the A* route follower")
    ap.add_argument("--table", default=f"{ROOT / 'results' / 'lockstep' / 'roll_sweep.json'},{ROOT / 'results' / 'lockstep' / 'roll_sweep_neg.json'}",
                    help="entry-roll sweep file(s) from calibrate_rolls.py")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=300, help="episode limit (5 Hz steps)")
    ap.add_argument("--out", default=str(ROOT / "results" / "astar_demo"))
    ap.add_argument("--reuse-executed", action="store_true", help="reuse executed_<profile>.json already in the case folder instead of running")
    args = ap.parse_args()
    forbid = tuple(f.strip() for f in args.forbid.split(",") if f.strip())
    if args.sim == "theirs" and not args.reuse_executed:
        from stroke_nav.organiser import ensure_policy_importable
        from stroke_nav.organiser.live.lockstep import kill_stray_pages, other_lockstep_jobs

        ensure_policy_importable()
        others = other_lockstep_jobs()
        if others:
            raise SystemExit("[astar] another lockstep job is running; wait for it or stop it first:\n  " + "\n  ".join(others))
        n = kill_stray_pages()
        if n:
            log(f"killed {n} stray lockstep Chrome processes")
    table = None
    if args.sim == "theirs" and args.controller == "roll":
        files = [f for f in args.table.split(",") if f and Path(f).exists()]
        if files:
            table = RollTable.load(",".join(files))
        else:
            log("no entry-roll table (calibrate_rolls.py): executing with the A* route follower instead")
    out_root = Path(args.out)
    seeds = [args.seed + i for i in range(args.episodes)]
    cases = []
    t_all = time.monotonic()
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        for target in [t.strip() for t in args.targets.split(",") if t.strip()]:
            out = out_root / variant / target.replace("@", "-at-")
            out.mkdir(parents=True, exist_ok=True)
            cases.append(run_case(args.model, variant, target, out, forbid, args.sim, seeds, args.workers, args.max_steps, args.reuse_executed, args.visible, table))
    lines = [f"# Constrained A* with length, time and damage costs: demo summary ({'organiser physics' if args.sim == 'theirs' else 'kinematic simulator'})", "",
             "| Variant | Target | Profile | Route | Length (mm) | Within travel | Est. time (s) | Executed | Time (s) | Final (mm) | Bend effort | Narrow (mm) | Vessel changes | Wrong-vessel retries | Safety |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cases:
        for prof, row in c["profiles"].items():
            ev = row["evaluation"]
            if not ev["feasible"]:
                lines.append(f"| {c['variant']} | {c['target']} | {prof} | infeasible | | | | | | | | | | | |")
                continue
            ex, cs = row.get("executed", {}), row["costs"]
            lines.append(f"| {c['variant']} | {c['target']} | {prof} | {' > '.join(row['branches'])} | {cs['length_mm']:.0f} | "
                         f"{'yes' if ev['robot_size']['within_travel'] else '**no**'} | {cs['est_time_s']:.0f} | {ex.get('success', '-')} | {ex.get('time_s', '-')} | "
                         f"{ex.get('median_final_dist_mm', '-')} | {cs['bend_effort']:.2f} | {cs['narrow_mm']:.0f} | {cs['vessel_changes']} | "
                         f"{ex.get('wrong_vessel_retries', '-')} | {ex.get('safety_verdict', '-')} |")
    lines += ["", f"Whole demo: {(time.monotonic() - t_all) / 60:.1f} min."]
    (out_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print()
    print("\n".join(lines))
    log(f"wrote {out_root}")


if __name__ == "__main__":
    main()
