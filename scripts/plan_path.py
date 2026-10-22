"""Plan a device-feasible path to a target and turn it into a robot trajectory.

    python scripts/plan_path.py --target right_carotid@-5
    python scripts/plan_path.py --variant 10Left100Right --target right_carotid@-5 --follow 20 --record
    python scripts/plan_path.py --target left_anterior@20 --device guidewire --hard-curvature

Outputs in results/plans/<name>/:
    report.md        device-aware A* vs plain Dijkstra (and the other devices), bifurcation table,
                     trajectory summary, and (with --follow) START metrics in the simulator
    profiles.png     lumen radius, curvature and speed along the path; joint references over time
    path3d.png       3D view: planned path coloured by speed, Dijkstra path in grey if different
    trajectory.csv   joint-space reference at the control rate (t, L and roll of both devices)
    plan.json        path nodes with geometry, speed reference and roll events
    follow.gif       (with --record) the trajectory follower executing the plan in the simulator
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from stroke_nav.agents.follower import TrajectoryFollower
from stroke_nav.agents.scripted import ScriptedOracle, run_episode
from stroke_nav.data.anatomy import Anatomy
from stroke_nav.env.vessel_env import EnvConfig, VesselNavEnv
from stroke_nav.eval.start_metrics import episode_start_metrics, markdown_table, summarize
from stroke_nav.planning.constraints import DEVICES, GUIDEWIRE
from stroke_nav.planning.planner import PathPlanner, PlannerWeights, compare_plans
from stroke_nav.planning.trajectory import TrajectoryParams, build_trajectory


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="0248_H_AOCERE_CAS")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--target", default="right_carotid@40")
    ap.add_argument("--start", default=None)
    ap.add_argument("--device", default=GUIDEWIRE, choices=sorted(DEVICES),
                    help="device the path must be feasible for (default: the robot's guidewire from device-parameters.json)")
    ap.add_argument("--hard-curvature", action="store_true", help="forbid bends tighter than the device's minimum bend radius")
    ap.add_argument("--w-curvature", type=float, default=PlannerWeights.curvature)
    ap.add_argument("--w-narrow", type=float, default=PlannerWeights.narrow)
    ap.add_argument("--w-branch", type=float, default=PlannerWeights.branch)
    ap.add_argument("--forbid", default="", help="comma-separated branches the path may not use (e.g. anterior_connector)")
    ap.add_argument("--follow", type=int, default=0, help="run the trajectory follower N times in the simulator")
    ap.add_argument("--record", action="store_true", help="save follow.gif of one follower episode")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    anat = Anatomy.load(args.model, args.variant)
    start = anat.default_start() if args.start is None else anat.resolve(args.start)
    target = anat.resolve(args.target)
    forbidden = tuple(b.strip() for b in args.forbid.split(",") if b.strip())
    weights = PlannerWeights(curvature=args.w_curvature, narrow=args.w_narrow, branch=args.w_branch,
                             hard_curvature=args.hard_curvature, forbidden_branches=forbidden)
    name = args.name or _slug(f"{anat.name}_{args.target}_{args.device}")
    out = Path("results/plans") / name
    out.mkdir(parents=True, exist_ok=True)

    planner = PathPlanner(anat, args.device, weights)
    plan = planner.plan(start, target)
    baseline = planner.shortest(start, target)
    others = [PathPlanner(anat, d, weights).plan(start, target) for d in sorted(DEVICES) if d != args.device]

    lines = [f"# Path plan: {anat.name}", "",
             f"Start: node {start} ({anat.branch_of(start)}). Target: `{args.target}` = node {target} "
             f"({anat.branch_of(target)}). Planned for: **{plan.device.name}** "
             f"(lumen radius >= {plan.device.min_lumen_radius_mm:.2f} mm, preferred bend radius >= "
             f"{plan.device.min_bend_radius_mm:.1f} mm{', enforced' if args.hard_curvature else ''})."
             + (f" Forbidden vessels: {', '.join(forbidden)}." if forbidden else ""), ""]
    occl = anat.meta.get("occlusions") or []
    if occl:
        lines += ["Occlusions in this anatomy: " + ", ".join(f"{o['branch']} at {o['from_mm']} mm ({o['length_mm']} mm long)" for o in occl), ""]
    lines += ["## Route comparison", "", compare_plans([plan, baseline] + others), ""]

    if not plan.feasible:
        lines += [f"**No feasible path:** {plan.message}"]
        (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        return

    env_cfg = EnvConfig(model=args.model, variant=args.variant, target=target, start=start, max_steps=900)
    traj = build_trajectory(plan, env_cfg.sim, TrajectoryParams())
    traj.to_csv(out / "trajectory.csv")
    traj.to_json(out / "plan.json")

    lines += ["## Bifurcations and guidewire roll", "",
              "| s (mm) | from | into | take-off angle (deg) | roll target (deg) | steerable window (deg) |",
              "|---|---|---|---|---|---|"]
    events = {e["node"]: e for e in traj.roll_events}
    for j in plan.junctions:
        e = events.get(j["node"])
        roll = f"{e['phi_deg']:.0f}" if e else "any"
        win = f"{e['window_deg']:.0f}" if e else "360"
        lines.append(f"| {j['s_mm']:.1f} | {j['from_branch']} | {j['to_branch']} | {j['turn_deg']:.0f} | {roll} | {win} |")
    ts = traj.summary()
    lines += ["", "## Trajectory", "",
              f"- Duration {ts['duration_s']} s over {ts['steps']} control steps of {env_cfg.sim.dt:.3f} s; "
              f"mean speed {ts['mean_speed_mm_s']} mm/s (limit {env_cfg.sim.guidewire.v_max_mm_s:.0f} mm/s).",
              f"- {ts['rotation_events']} rotation targets; total guidewire rotation {ts['total_wire_rotation_deg']:.0f} deg; "
              f"{ts['wire_dwell_steps']} steps waiting to finish a rotation.",
              f"- The microcatheter trails {traj.params.catheter_gap_mm:.0f} mm behind and stops at the target; the "
              f"guidewire leads {traj.params.overshoot_mm:.0f} mm past it.", ""]

    plot_profiles(traj, baseline, out / "profiles.png")
    render_3d(env_cfg, traj, baseline, out / "path3d.png")

    if args.follow:
        env = VesselNavEnv(env_cfg)
        rows = []
        for label, agent in (("planned follower", TrajectoryFollower(trajectory=traj)),
                             ("shortest-path follower", TrajectoryFollower(use_shortest_path=True)),
                             ("oracle", ScriptedOracle())):
            eps = [episode_start_metrics(run_episode(env, agent, seed=1000 + i)) for i in range(args.follow)]
            rows.append(summarize(label, eps))
            print(f"[follow] {label}: success {np.mean([e['success'] for e in eps]):.0%}")
        lines += ["## Execution in the simulator (START metrics)", "", markdown_table(rows), ""]
    if args.record:
        record_follow(env_cfg, traj, baseline, out / "follow.gif")
        lines += ["![follower](follow.gif)", ""]
    lines += ["![profiles](profiles.png)", "", "![3D path](path3d.png)", ""]

    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")


# ---------------------------------------------------------------------- figures
def plot_profiles(traj, baseline, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plan, dev = traj.plan, traj.plan.device
    fig, ax = plt.subplots(4, 1, figsize=(11, 12), sharex=False)
    s = traj.s
    junction_s = [j["s_mm"] for j in plan.junctions]
    for a in ax[:3]:
        for sj in junction_s:
            a.axvline(sj, color="#c4b5fd", lw=0.8, zorder=0)
    ax[0].plot(s, traj.radius, color="#b91c1c", lw=1.5, label="planned path")
    if baseline.nodes and baseline.nodes != plan.nodes:
        ax[0].plot(baseline.s, baseline.radius, color="#6b7280", lw=1, ls="--", label="shortest path (Dijkstra)")
    ax[0].axhline(dev.min_lumen_radius_mm, color="k", ls=":", label=f"{dev.name} minimum ({dev.min_lumen_radius_mm:.2f} mm)")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("lumen radius (mm)")
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"{plan.anatomy.name}: {' > '.join(plan.branches())}", fontsize=10)
    bend = 1.0 / np.maximum(traj.curvature, 1e-6)
    ax[1].plot(s, np.minimum(bend, 200), color="#1d4ed8", lw=1.2)
    ax[1].axhline(dev.min_bend_radius_mm, color="k", ls=":", label=f"device min bend radius ({dev.min_bend_radius_mm} mm)")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("bend radius (mm)")
    ax[1].legend(fontsize=8)
    ax[2].plot(s, traj.v_limit, color="#9ca3af", lw=1, label="speed limit (curvature, lumen, bifurcations)")
    ax[2].plot(s, traj.v_ref, color="#047857", lw=1.8, label="reference speed (with acceleration limit)")
    ax[2].axvline(traj.s_target, color="#16a34a", ls="--", label="target")
    ax[2].set_ylabel("speed (mm/s)")
    ax[2].set_xlabel("distance along path (mm); purple lines = bifurcations")
    ax[2].legend(fontsize=8)
    a3 = ax[3]
    a3.plot(traj.t, traj.L_wire, color="#111827", label="guidewire inserted length")
    a3.plot(traj.t, traj.L_cath, color="#d97706", label="microcatheter inserted length")
    a3.set_ylabel("inserted length (mm)")
    a3.set_xlabel("time (s)")
    a3b = a3.twinx()
    a3b.plot(traj.t, np.degrees(np.unwrap(traj.phi_wire)), color="#7c3aed", lw=1, alpha=0.8, label="guidewire roll")
    a3b.set_ylabel("guidewire roll, cumulative (deg)", color="#7c3aed")
    a3.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def render_3d(env_cfg, traj, baseline, path: Path) -> None:
    import pyvista as pv

    from stroke_nav.viz.scene import Scene

    env = VesselNavEnv(env_cfg)
    env.reset(seed=0)
    pl = pv.Plotter(off_screen=True, window_size=(900, 1000))
    pl.set_background("white")
    scene = Scene(env, pl, show_centerlines=False)
    scene.draw_trajectory(traj)
    if baseline.nodes and baseline.nodes != traj.plan.nodes:
        scene.draw_path(baseline.points, name="dijkstra")
    scene.update(show_hud=False)
    title = f"{traj.plan.anatomy.name}  |  {traj.plan.device.name}"
    pl.add_text(title + "\n" + "planned path coloured by reference speed; grey = plain shortest path",
                position="upper_left", font_size=9, color="black")
    scene.set_front_view()
    pl.screenshot(str(path))
    pl.close()


def record_follow(env_cfg, traj, baseline, path: Path) -> None:
    import pyvista as pv

    from stroke_nav.viz.scene import Scene

    env = VesselNavEnv(env_cfg)
    env.reset(seed=0)
    agent = TrajectoryFollower(trajectory=traj)
    agent.reset(env)
    pl = pv.Plotter(off_screen=True, window_size=(760, 900))
    pl.set_background("white")
    scene = Scene(env, pl, show_centerlines=False)
    scene.draw_trajectory(traj, label_junctions=False)
    if baseline.nodes and baseline.nodes != traj.plan.nodes:
        scene.draw_path(baseline.points, name="dijkstra")
    scene.update("following planned trajectory")
    scene.set_front_view()
    pl.open_gif(str(path), fps=12)
    done, info = False, {}
    while not done:
        _, _, term, trunc, info = env.step(agent.act(env))
        done = term or trunc
        if env.steps % 2 == 0 or done:
            scene.update("following planned trajectory")
            pl.write_frame()
    m = info["episode_metrics"]
    scene.update(f"{'SUCCESS' if m['success'] else 'FAILED'} in {m['steps'] * m['dt']:.1f} s")
    for _ in range(12):
        pl.write_frame()
    pl.close()


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:90]


if __name__ == "__main__":
    main()
