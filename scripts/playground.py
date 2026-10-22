"""Interactive playground: pick a start and a target on a VMR vessel model, then drive the
guidewire and microcatheter yourself or let the scripted oracle do it. START metrics are printed
when an episode ends.

    python scripts/playground.py --model 0248_H_AOCERE_CAS
    python scripts/playground.py --model 0248_H_AOCERE_CAS --target left_carotid@-5
    python scripts/playground.py --variant 10Left100Right --target right_carotid@-5 --autoplay plan
    python scripts/playground.py --model 0248_H_AOCERE_CAS --target left_carotid@-5 --record demo.gif

Keys (each press = one control step of 1/7.5 s at full speed):
    Up / Down       guidewire advance / retract
    Left / Right    guidewire rotate
    i / k           microcatheter advance / retract
    n / m           microcatheter rotate
    b               advance both devices
    o               one oracle step;  space: start/stop oracle autopilot
    l               plan a device-feasible path to the target and show it (colour = planned speed)
    h               start/stop following the planned trajectory (robot-style autopilot)
    p               (hover over the vessel) pick a point -> becomes target (or start, see g)
    g               toggle what 'p' sets: target / start
    x               reset the episode (new random target if the target spec is random)
    c               show / hide centerlines
    z               print START metrics of the current episode
    q               quit
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from stroke_nav.agents.follower import TrajectoryFollower
from stroke_nav.agents.scripted import ScriptedOracle
from stroke_nav.env.vessel_env import EnvConfig, VesselNavEnv
from stroke_nav.eval.start_metrics import episode_start_metrics, markdown_table, summarize

KEYS_HELP = (
    "wire: Up/Down move, Left/Right rotate | catheter: i/k move, n/m rotate | b both\n"
    "o oracle step, space autopilot | l plan path, h follow plan | p pick (g: target/start) | x reset | c centerlines | z metrics | q quit"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="0248_H_AOCERE_CAS")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--target", default="left_carotid@-5", help="target spec (see stroke_nav.data.anatomy)")
    ap.add_argument("--start", default=None)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--record", default=None, help="run the oracle off-screen and save a .gif (or .png)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0, help="autopilot speed relative to real time")
    ap.add_argument("--autoplay", choices=["plan", "oracle"], default=None,
                    help="start driving as soon as the window opens (plan = follow the planned trajectory)")
    args = ap.parse_args()

    import pyvista as pv

    cfg = EnvConfig(model=args.model, variant=args.variant, target=args.target, start=args.start,
                    max_steps=args.max_steps)
    env = VesselNavEnv(cfg)
    env.reset(seed=args.seed)

    if args.record:
        record(env, args.record)
        return

    from stroke_nav.viz.scene import Scene

    pl = pv.Plotter(title=f"stroke-nav playground: {env.anat.name}", window_size=(1400, 950))
    pl.set_background("white")
    scene = Scene(env, pl)
    oracle = ScriptedOracle()
    oracle.reset(env)
    state = {"auto": False, "pick": "target", "done": False, "status": "", "follower": None, "follow": False}

    def finish(info: dict) -> None:
        state["done"] = True
        state["auto"] = False
        m = episode_start_metrics(info["episode_metrics"])
        verdict = "SUCCESS" if m["success"] else "time limit reached"
        state["status"] = f"episode finished: {verdict} in {m['procedure_time_s']:.1f} s (press x to reset)"
        print(markdown_table([summarize("this episode", [m])]), flush=True)

    def do_step(action, render: bool = True) -> None:
        if state["done"]:
            return
        _, _, term, trunc, info = env.step(np.asarray(action, dtype=float))
        if term or trunc:
            finish(info)
        if render:
            scene.update(state["status"], KEYS_HELP)

    def reset(options=None) -> None:
        env.reset(options=options)
        oracle.reset(env)
        state.update(done=False, auto=False, status="", follower=None, follow=False, followed=False)
        scene.clear("plan", "plan_labels", "dijkstra")
        scene.update("", KEYS_HELP)

    key_actions = {
        "Up": [1, 0, 0, 0], "Down": [-1, 0, 0, 0], "Left": [0, -1, 0, 0], "Right": [0, 1, 0, 0],
        "i": [0, 0, 1, 0], "k": [0, 0, -1, 0], "n": [0, 0, 0, -1], "m": [0, 0, 0, 1], "b": [1, 0, 1, 0],
    }
    for key, act in key_actions.items():
        pl.add_key_event(key, lambda act=act: do_step(act))
    pl.add_key_event("o", lambda: do_step(oracle.act(env)))

    def toggle_auto():
        state["auto"] = not state["auto"] and not state["done"]
        state["status"] = "oracle autopilot ON" if state["auto"] else ""
        scene.update(state["status"], KEYS_HELP)

    pl.add_key_event("space", toggle_auto)
    pl.add_key_event("x", lambda: reset())

    def toggle_pick():
        state["pick"] = "start" if state["pick"] == "target" else "target"
        state["status"] = f"'p' now sets the {state['pick']}"
        scene.update(state["status"], KEYS_HELP)

    pl.add_key_event("g", toggle_pick)

    def toggle_cl():
        scene.cl_actor.SetVisibility(not scene.cl_actor.GetVisibility())
        pl.render()

    pl.add_key_event("c", toggle_cl)

    def print_metrics():
        m = episode_start_metrics(env.episode_metrics())
        print(markdown_table([summarize("current episode", [m])]))

    pl.add_key_event("z", print_metrics)

    def show_plan():
        follower = TrajectoryFollower()
        try:
            follower.reset(env)
        except ValueError as e:
            state["status"] = f"planning failed: {e}"
            scene.update(state["status"], KEYS_HELP)
            return
        traj = follower.traj
        scene.draw_trajectory(traj)
        from stroke_nav.planning.planner import PathPlanner

        base = PathPlanner(env.anat, traj.plan.device).shortest(env.start_node, env.target_node)
        if base.nodes and base.nodes != traj.plan.nodes:
            scene.draw_path(base.points, name="dijkstra")
        sm, ts = traj.plan.summary(), traj.summary()
        print(f"[plan] {sm['device']}: {sm['branches']}  length {sm['length_mm']} mm, min lumen r "
              f"{sm['min_lumen_radius_mm']} mm (needs {sm['required_lumen_radius_mm']}), "
              f"{ts['rotation_events']} rotations, planned duration {ts['duration_s']} s")
        if base.nodes and not base.feasible:
            print(f"[plan] shortest path (grey) is not feasible: {base.message}")
        state["follower"] = follower
        state["status"] = (f"plan: {' > '.join(traj.plan.branches())} | {sm['length_mm']} mm, "
                           f"{ts['duration_s']} s planned (press h to follow)")
        scene.update(state["status"], KEYS_HELP)

    def toggle_follow():
        starting = not state["follow"]
        if starting and (state["follower"] is None or (env.steps > 0 and not state.get("followed"))):
            # the trajectory starts at the insertion point: restart the episode toward the same target
            if env.steps > 0 or state["done"]:
                reset({"target": env.target_node})
            show_plan()
            if state["follower"] is None:
                return
        state["followed"] = True
        state["follow"] = not state["follow"] and not state["done"]
        state["auto"] = False
        state["status"] = "following planned trajectory" if state["follow"] else "paused"
        scene.update(state["status"], KEYS_HELP)

    pl.add_key_event("l", show_plan)
    pl.add_key_event("h", toggle_follow)

    def on_pick(point, *_):
        if point is None:
            return
        node = env.anat.nearest_node(point)
        name = env.anat.branch_of(node)
        saved = save_selection(env, args.model, **({"target_node": node} if state["pick"] == "target" else {"start_node": node}))
        if state["pick"] == "target":
            reset({"target": node})
        else:
            reset({"start": node})
        state["status"] = f"{state['pick']} set to node {node} in {name} (saved to {saved})"
        scene.update(state["status"], KEYS_HELP)

    pl.enable_point_picking(callback=on_pick, show_message=False, show_point=False, left_clicking=False)

    def tick(*_):
        """Autopilot at real-time pace: run the simulation steps that are due since the autopilot
        started (scaled by --speed), then draw one frame. Slow frames drop, the robot keeps time."""
        follow = state["follow"] and state["follower"] is not None
        if state["done"] or not (state["auto"] or follow):
            state["clock"] = None
            return
        now = time.monotonic()
        if state.get("clock") is None:
            state["clock"] = (now, env.steps)
        t0, s0 = state["clock"]
        due = s0 + int((now - t0) * args.speed / env.cfg.sim.dt) + 1
        for _ in range(max(0, min(due - env.steps, 12))):
            if state["done"]:
                break
            do_step(oracle.act(env) if state["auto"] else state["follower"].act(env), render=False)
        scene.update(state["status"], KEYS_HELP)

    scene.update("", KEYS_HELP)
    scene.set_front_view()
    print(__doc__, flush=True)
    # The window must exist before the timer is created: on Windows a VTK timer created earlier
    # is tied to no window and never fires (the autopilot keys then do nothing).
    pl.show(interactive_update=True, auto_close=False)
    pl.add_timer_event(max_steps=10_000_000, duration=60, callback=tick)
    if args.autoplay == "plan":
        toggle_follow()
    elif args.autoplay == "oracle":
        toggle_auto()
    pl.iren.start()
    pl.close()


def save_selection(env: VesselNavEnv, model: str, **nodes) -> Path:
    path = Path("data/targets") / f"{model}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    for k, n in nodes.items():
        data[k] = int(n)
        data[k.replace("_node", "_xyz")] = [round(float(x), 2) for x in env.anat.nodes[n]]
        data[k.replace("_node", "_branch")] = env.anat.branch_of(n)
    path.write_text(json.dumps(data, indent=2))
    return path


def record(env: VesselNavEnv, out: str) -> None:
    """Run the oracle off-screen and save frames to a GIF (or the final frame to a PNG)."""
    import pyvista as pv

    from stroke_nav.viz.scene import Scene

    pl = pv.Plotter(off_screen=True, window_size=(760, 900))
    pl.set_background("white")
    scene = Scene(env, pl, show_centerlines=False)
    oracle = ScriptedOracle()
    oracle.reset(env)
    scene.update("oracle")
    scene.set_front_view()
    gif = out.lower().endswith(".gif")
    if gif:
        pl.open_gif(out, fps=12)
        pl.write_frame()
    done, info = False, {}
    while not done:
        _, _, term, trunc, info = env.step(oracle.act(env))
        done = term or trunc
        if gif and (env.steps % 2 == 0 or done):
            scene.update("oracle autopilot")
            pl.write_frame()
    m = episode_start_metrics(info["episode_metrics"])
    scene.update(f"{'SUCCESS' if m['success'] else 'FAILED'} in {m['procedure_time_s']:.1f} s")
    if gif:
        for _ in range(12):
            pl.write_frame()
        pl.close()
    else:
        pl.screenshot(out)
        pl.close()
    print(markdown_table([summarize("oracle", [m])]))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
