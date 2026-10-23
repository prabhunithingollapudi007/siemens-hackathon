"""Demo: closed-loop A* navigation with the organiser's devices, shown in 3D.

For each target the planner finds a device-feasible route (purple), and the closed-loop follower
drives the guidewire (blue) and catheter (orange) along it, steering only from the measured tip
pose. The devices come from device-parameters.json, the repo's source of truth
(STROKE_NAV_DEVICE_PARAMS selects another file), and the robot's travel limits are enforced,
exactly as in scripts/run_astar.py --sim ours.

    python scripts/demo_astar.py                                   # live window, several targets in a row
    python scripts/demo_astar.py --targets right_carotid@40 --speed 3
    python scripts/demo_astar.py --record results/demo_astar.mp4   # smooth off-screen video (or .gif) for sharing

Keys in the window: space pause / resume, n skip to the next target, q quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_astar import OursAdapter, build  # noqa: E402

from stroke_nav.agents.route_follower import ClosedLoopRouteFollower, FollowerParams, TipObs  # noqa: E402
from stroke_nav.data.anatomy import Anatomy  # noqa: E402
from stroke_nav.organiser.frame import default_frame  # noqa: E402
from stroke_nav.organiser.spec import DEFAULT_SPEC, describe_devices  # noqa: E402

DEFAULT_TARGETS = ["right_carotid@40", "left_vertebral@20", "right_subclavian@60", "right_vertebral@10"]
ROUTE_COLOR = "#7c3aed"


class Episode:
    """One target: plan, environment, follower and the organiser's success rule."""

    def __init__(self, anat, frame, spec, target: str):
        self.target = target
        self.goal, node, self.plan, cath_limit = build(anat, frame, target)
        self.sim = OursAdapter(anat, frame, self.goal, spec)
        self.sim.env.inner.cfg.success_threshold_mm = spec.environment.success_radius_m * 1000  # target marker size
        self.follower = ClosedLoopRouteFollower(anat, self.plan, FollowerParams(
            cath_limit_mm=cath_limit, travel_limit_mm=spec.robot.guidewire_travel_mm - 0.5))
        self.spec = spec
        self.obs = self.sim.reset()
        self.steps, self.in_goal, self.result = 0, 0, None
        self.dist = float(np.linalg.norm(self.obs["wire_tip"] - self.goal))

    @property
    def env(self):
        return self.sim.env.inner

    def step(self) -> None:
        o = self.obs
        a = self.follower.act(TipObs(o["wire_tip"], o["wire_dir"], o["wire_ins"], o["cath_ins"], self.goal))
        self.obs = self.sim.step(a)
        self.steps += 1
        self.dist = float(np.linalg.norm(self.obs["wire_tip"] - self.goal))
        self.in_goal = self.in_goal + 1 if self.dist <= self.spec.environment.success_radius_m * 1000 else 0
        if self.in_goal >= self.spec.environment.success_steps:
            self.result = "SUCCESS"
        elif self.follower.state.mode == "travel_limit":
            self.result = "robot travel limit reached"
        elif self.steps >= self.spec.environment.max_steps:
            self.result = "step limit reached"

    def status(self) -> str:
        st = self.follower.state
        head = f"A* closed loop -> {self.target}   route: {self.plan.summary()['branches']}"
        body = (f"step {self.steps}  ({self.steps * self.spec.dt:5.1f} s)   wire tip to goal {self.dist:5.1f} mm   "
                f"bifurcation {min(st.junction + 1, len(self.follower.junctions))}/{len(self.follower.junctions)}   "
                f"mode {st.mode}   wrong-vessel retries {sum(e['event'] == 'wrong_vessel' for e in st.log)}")
        tail = f"\n{self.result} in {self.steps * self.spec.dt:.1f} s" if self.result else ""
        return f"{head}\n{body}{tail}"


def device_line(spec) -> str:
    gw, ca = spec.guidewire_geometry(), spec.catheter_geometry()
    r = spec.robot
    return (f"devices (device-parameters.json): {gw['preset']} wire {gw['outer_diameter_mm']} mm, tip {gw['tip_length_mm']:.0f} mm "
            f"@ {gw['tip_angle_deg']:.0f} deg | {ca['preset']} catheter {ca['outer_diameter_mm']} mm, tip "
            f"{ca['tip_length_mm']:.0f} mm @ {ca['tip_angle_deg']:.0f} deg | robot travel {r.guidewire_travel_mm} mm")


def make_scene(ep: Episode, pl, spec):
    from stroke_nav.viz.scene import Scene

    scene = Scene(ep.env, pl, show_centerlines=False)
    draw_route(scene, ep)
    return scene


def draw_route(scene, ep: Episode) -> None:
    scene.draw_path(ep.plan.points, color=ROUTE_COLOR, name="route", radius=0.25, opacity=0.35)
    scene.focus(ep.plan.points)


def switch(scene, ep: Episode) -> None:
    scene.env = ep.env
    scene._prev_state = None
    scene.clear("route")
    draw_route(scene, ep)


def run_window(anat, frame, spec, targets: list[str], speed: float) -> None:
    import pyvista as pv

    pl = pv.Plotter(title="stroke-nav: closed-loop A* with the organiser's devices", window_size=(1400, 950))
    pl.set_background("white")
    st = {"i": 0, "ep": Episode(anat, frame, spec, targets[0]), "paused": False, "clock": None, "done_at": None,
          "step_t": time.monotonic()}
    scene = make_scene(st["ep"], pl, spec)
    results = []
    period = spec.dt / speed  # wall-clock time per control step

    def draw(alpha: float = 1.0):
        scene.update(st["ep"].status(), device_line(spec) + "\nspace pause   n next target   q quit", alpha=alpha)

    def next_target():
        ep = st["ep"]
        results.append((ep.target, ep.result or "skipped", round(ep.steps * spec.dt, 1)))
        st["i"] = (st["i"] + 1) % len(targets)
        st["ep"] = Episode(anat, frame, spec, targets[st["i"]])
        switch(scene, st["ep"])
        st["clock"], st["done_at"] = None, None
        draw()

    def toggle_pause():
        st["paused"] = not st["paused"]
        st["clock"] = None

    def tick(*_):
        """~60 frames per second: simulate the control steps that are due, and draw the devices
        part-way between the last two simulated states so the motion is smooth."""
        ep = st["ep"]
        now = time.monotonic()
        alpha = min((now - st["step_t"]) / period, 1.0)
        if ep.result:
            if st["done_at"] is None:
                st["done_at"] = now
                print(f"[demo] {ep.target}: {ep.result} in {ep.steps * spec.dt:.1f} s", flush=True)
            if now - st["done_at"] <= period:
                draw(alpha)  # finish the last move
            elif now - st["done_at"] > 3.0:
                next_target()
            return
        if st["paused"]:
            st["clock"] = None
            return
        if st["clock"] is None:
            st["clock"] = (now, ep.steps)
        t0, s0 = st["clock"]
        due = s0 + int((now - t0) / period) + 1
        for _ in range(max(0, min(due - ep.steps, 10))):
            scene.begin_step()
            ep.step()
            st["step_t"], alpha = now, 0.0
            if ep.result:
                break
        draw(alpha)

    pl.add_key_event("space", toggle_pause)
    pl.add_key_event("n", next_target)
    draw()
    print(__doc__, flush=True)
    print(device_line(spec), flush=True)
    # window first, then the timer (a VTK timer created before the window never fires on Windows)
    pl.show(interactive_update=True, auto_close=False)
    pl.add_timer_event(max_steps=10_000_000, duration=15, callback=tick)
    pl.iren.start()
    pl.close()


def record(anat, frame, spec, targets: list[str], out: Path, speed: float = 3.0, fps: int = 30) -> None:
    """Off-screen recording with interpolated in-between frames (.mp4, or .gif at <= 15 fps)."""
    import pyvista as pv

    from stroke_nav.viz.scene import frames_per_step, open_recording

    out.parent.mkdir(parents=True, exist_ok=True)
    gif = out.suffix.lower() == ".gif"
    fps = min(fps, 15) if gif else fps
    n = frames_per_step(fps, spec.dt, speed)
    pl = pv.Plotter(off_screen=True, window_size=(720, 800) if gif else (900, 1000))
    pl.set_background("white")
    ep = Episode(anat, frame, spec, targets[0])
    scene = make_scene(ep, pl, spec)
    open_recording(pl, str(out), fps)
    for i, target in enumerate(targets):
        if i:
            ep = Episode(anat, frame, spec, target)
            switch(scene, ep)
        scene.update(ep.status(), device_line(spec))
        pl.write_frame()
        while not ep.result:
            scene.begin_step()
            ep.step()
            scene.write_step_frames(n, ep.status(), device_line(spec))
        for _ in range(2 * fps):  # hold the final frame for two seconds
            pl.write_frame()
        print(f"[demo] {target}: {ep.result} in {ep.steps * spec.dt:.1f} s", flush=True)
    pl.close()
    print(f"[demo] saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS, help="target specs (branch@mm)")
    ap.add_argument("--speed", type=float, default=3.0, help="playback speed relative to real time")
    ap.add_argument("--record", type=Path, default=None, help="write an off-screen GIF instead of opening a window")
    args = ap.parse_args()

    spec = DEFAULT_SPEC
    print(describe_devices(spec), flush=True)
    anat, frame = Anatomy.load("0248_H_AOCERE_CAS"), default_frame()
    if args.record:
        record(anat, frame, spec, args.targets, args.record, args.speed)
    else:
        run_window(anat, frame, spec, args.targets, args.speed)


if __name__ == "__main__":
    main()
