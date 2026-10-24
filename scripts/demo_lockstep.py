"""Demo: the behaviour-cloned policy driving the organiser's own physics page, with a recording.

Opens their physics page (``catheter_guidewire_ros2.html``) and their noisy fluoro viewer in two
visible Chrome windows (``stroke_nav.organiser.live.lockstep``), runs one episode of a SAC model (or
the scripted expert) in their ``OpenNG3Env`` on the cas-reference scenario, and records a video:
their physics view on the left, the policy's fluoro input on the right, with step, simulated time,
distance to the goal and the action.

    python scripts/demo_lockstep.py                                   # runs/stroke-nav-bc-v1, visible
    python scripts/demo_lockstep.py --record results/demo_bc_policy.mp4
    python scripts/demo_lockstep.py --expert --headless --record results/demo_expert.mp4
    python scripts/demo_lockstep.py --model runs/stroke-nav-bc-multi/final-model.zip         --goals results/lockstep/goals.json --goal-id 92        # one goal of a make_goals.py set
    python scripts/demo_lockstep.py --variant 10Right70Left --target left_carotid@140 --expert --slow-radius 8 \n        --record results/astar_demo/10Right70Left_left_carotid_at_140.mp4   # a stenosis variant as their scenario; A* route + calibrated entry roll

With ``--goals`` the physics view shows the episode's goal (loaded through the page's own goal-file
input); their fluoro viewer, the policy's input, still shows the scenario's goal marker, as in their
stack, so the policy takes the goal from the state.

Stop their Docker stack first (tools/organiser_stack.ps1 stop); it is not used.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stroke_nav.agents.entry_roll import EntryRollExpert  # noqa: E402
from stroke_nav.agents.roll_table import load_goal_set  # noqa: E402
from stroke_nav.organiser import ensure_policy_importable  # noqa: E402
from stroke_nav.organiser.live.lockstep import make_env, show_goal  # noqa: E402

HEIGHT = 720
LEFT_WIDTH = 880  # their 1100 x 900 physics window at 720 px high


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def compose(physics_jpeg: bytes, fluoro_jpeg: bytes | None, lines: list[str], banner: str | None) -> np.ndarray:
    """One video frame: physics page | fluoro image over a text panel (even width and height for H.264)."""
    shot = Image.open(io.BytesIO(physics_jpeg)).convert("RGB")
    shot.thumbnail((LEFT_WIDTH, HEIGHT))  # the visible window may be resized mid-run: fit it into a fixed box
    left = Image.new("RGB", (LEFT_WIDTH, HEIGHT), (255, 255, 255))
    left.paste(shot, ((LEFT_WIDTH - shot.width) // 2, (HEIGHT - shot.height) // 2))
    side = 460
    right = Image.new("RGB", (side, HEIGHT), (18, 18, 22))
    if fluoro_jpeg:
        right.paste(Image.open(io.BytesIO(fluoro_jpeg)).convert("RGB").resize((side, side)), (0, 0))
    d = ImageDraw.Draw(right)
    d.text((12, side + 10), "policy input: noisy fluoro (their viewer)", fill=(150, 150, 160), font=_font(16))
    step, size = (30, 20) if len(lines) <= 7 else (22, 16)  # the panel under the fluoro image is 260 px high
    for i, line in enumerate(lines):
        d.text((12, side + 40 + step * i), line, fill=(235, 235, 240), font=_font(size))
    frame = Image.new("RGB", (left.width + side + (left.width + side) % 2, HEIGHT))
    frame.paste(left, (0, 0))
    frame.paste(right, (left.width, 0))
    if banner:
        dd = ImageDraw.Draw(frame)
        dd.rectangle((0, 0, left.width, 52), fill=(170, 95, 10) if banner.startswith("SAFE STOP") else (20, 110, 60))
        dd.text((16, 12), banner, fill=(255, 255, 255), font=_font(26))
    return np.asarray(frame)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "runs" / "stroke-nav-bc-v1" / "final-model.zip"))
    ap.add_argument("--expert", action="store_true", help="run EntryRollExpert instead of the model")
    ap.add_argument("--headless", action="store_true", help="do not show the Chrome windows")
    ap.add_argument("--record", default=str(ROOT / "results" / "demo_bc_policy.mp4"), help=".mp4 or .gif ('' for none)")
    ap.add_argument("--fps", type=float, default=10.0, help="video frames per second (5 = real time)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--goals", default=None, help="goal set from scripts/make_goals.py")
    ap.add_argument("--goal-id", type=int, default=None, help="goal id in --goals (default: the cas-reference goal)")
    ap.add_argument("--slow-radius", type=float, default=25.0, help="expert slow-down distance (mm)")
    ap.add_argument("--variant", default=None, help="0248 stenosis variant: run on its mesh as their scenario (stroke_nav.organiser.scenario)")
    ap.add_argument("--target", default=None, help="target spec (branch@mm, xyz:...) on the variant; the entry roll comes from the calibrated table")
    ap.add_argument("--roll", type=float, default=None, help="entry roll for the expert (deg); default: the table's robust roll for --target")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--vmr-model", default="0248_H_AOCERE_CAS", help="VMR model of --target (other models: their scenario from make_scenario.py)")
    ap.add_argument("--table", default=None, help="entry-roll sweep file(s) (default: the model's)")
    ap.add_argument("--wall-safety", action="store_true",
                    help="wrap the controller in the wall safety layer (stroke_nav.safety.wall): back off, re-roll, stop safely")
    args = ap.parse_args()

    env_kwargs, route_line = {}, None
    if args.variant or args.target:
        from stroke_nav.agents.roll_table import RollTable
        from stroke_nav.data.anatomy import Anatomy
        from stroke_nav.organiser.frame import entry_node, frame_for
        from stroke_nav.organiser.scenario import DEFAULT_SCENARIO_ROOT, scenario_name, write_scenario
        from stroke_nav.planning.constraints import GUIDEWIRE
        from stroke_nav.planning.planner import PathPlanner, PlannerWeights

        model_name = args.vmr_model
        anat, frame = Anatomy.load(model_name, args.variant), frame_for(model_name)
        scenario_goal = args.target in (None, "scenario")
        if scenario_goal:  # the organiser's own goal (cas-reference goals.txt), exactly as their env samples it
            from stroke_nav.organiser.spec import CAS_REFERENCE_GOALS_CM

            goal_vmr = frame.obj_cm_to_vmr_mm(np.asarray(CAS_REFERENCE_GOALS_CM[0]))
            node = anat.nearest_node(goal_vmr)
        else:
            node = anat.resolve(args.target)
            goal_vmr = anat.nodes[node]
        plan = PathPlanner(anat, GUIDEWIRE, PlannerWeights.profile("balanced")).plan(entry_node(anat, frame), node)
        route_line = f"A* route: {' > '.join(plan.branches())} ({plan.length_mm:.0f} mm, ~{plan.costs['est_time_s']:.0f} s)" if plan.feasible else "A*: no route"
        roll = args.roll
        if roll is None:
            default_tables = ((ROOT / "results" / "lockstep" / "roll_sweep.json", ROOT / "results" / "lockstep" / "roll_sweep_neg.json")
                              if model_name == "0248_H_AOCERE_CAS" and not args.variant else
                              (ROOT / "results" / "lockstep" / f"roll_sweep_{scenario_name(model_name, args.variant)}.json",))
            files = [Path(f) for f in (args.table.split(",") if args.table else default_tables) if Path(f).exists()]
            if not files:
                ap.error(f"no entry-roll table for {model_name}: run calibrate_rolls.py --model {model_name} or pass --roll")
            roll, miss = RollTable.load(",".join(map(str, files))).reach(frame.vmr_mm_to_world_m(goal_vmr), robust_within_mm=5.0)
            print(f"entry roll {roll:g} deg from the calibrated table (recorded paths miss the target by {miss:.1f} mm)")
        goal = {"id": 0, "split": args.variant or "base", "branch": anat.branch_of(node), "roll_deg": roll,
                "goal_m": frame.vmr_mm_to_world_m(goal_vmr).tolist(), "scenario_goal": scenario_goal}
        if args.variant or model_name != "0248_H_AOCERE_CAS":
            name = scenario_name(model_name, args.variant)
            if not (Path(DEFAULT_SCENARIO_ROOT) / name / "scenario.json").is_file():
                write_scenario(model_name, args.variant, [goal_vmr], DEFAULT_SCENARIO_ROOT, frame)
            env_kwargs = {"scenario_name": name, "scenario_root": Path(DEFAULT_SCENARIO_ROOT)}

    if not (args.variant or args.target):
        goal = None
    if args.goals:
        goals = load_goal_set(args.goals, "train,test,cas")
        goal = next((g for g in goals if g["id"] == args.goal_id), None) if args.goal_id is not None else             next(g for g in goals if g["split"] == "cas")
        if goal is None:
            ap.error(f"no goal with id {args.goal_id} in {args.goals}")
    if args.expert:
        policy = EntryRollExpert(roll_deg=goal["roll_deg"] if goal else 120.0, slow_radius_mm=args.slow_radius)
        name = "A* route + calibrated entry-roll controller" if (args.variant or args.target) else "scripted expert (EntryRollExpert)"
    else:
        ensure_policy_importable()
        from stable_baselines3 import SAC

        model, name = SAC.load(args.model, device="cpu"), f"BC policy {Path(args.model).parent.name}"
    controller = policy if args.expert else None
    guard = None
    if args.wall_safety:
        from stroke_nav.organiser.live.rollout import wall_guard_for
        from stroke_nav.safety.wall import WallSafeController

        class _Model:  # the SB3 model as a controller
            def reset(self):
                pass

            def act(self, o):
                return model.predict(o, deterministic=True)[0]

        sc = (env_kwargs.get("scenario_name", "cas-reference"), env_kwargs.get("scenario_root"))
        controller = WallSafeController(controller if args.expert else _Model(), wall_guard_for(*sc))
        guard = wall_guard_for(*sc)  # the true wire, for the overlay
        name += " + wall safety layer"
    their_goal = bool(goal and goal.get("scenario_goal"))
    env = make_env(worker=0, seed=args.seed, visible=not args.headless, goals=None if (goal is None or their_goal) else [goal["goal_m"]],
                   max_steps=args.max_steps, **env_kwargs)
    bridge = env.bridge
    frames = []
    try:
        if goal and not their_goal:
            show_goal(bridge.physics, goal["goal_m"])
        elif goal:
            print(f"goal: the organiser's cas-reference goal ({goal['branch']}), entry roll {goal['roll_deg']:g} deg")
            print(f"goal {goal['id']} ({goal['split']}): {goal['branch']}, entry roll {goal['roll_deg']:g} deg")
        if controller is not None and hasattr(controller, "bind"):
            controller.bind(env)
        obs, info = env.reset(seed=args.seed)
        if controller is not None:
            controller.reset()
        done, step, banner, held, worst = False, 0, None, 0, 0.0
        while not done:
            act = controller.act(obs) if controller is not None else model.predict(obs, deterministic=True)[0]
            obs, _, term, trunc, info = env.step(np.asarray(act, np.float32))
            step += 1
            done = term or trunc
            safety_line = None
            if args.wall_safety:
                from stroke_nav.safety.wall import guidewire_bodies

                checks = [guard.check(guidewire_bodies(bridge.true[d][-1]), bridge.true[d][-1]["ins"] * 1000.0)
                          for d in ("guidewire", "catheter") if bridge.true[d] and bridge.true[d][-1]["ins"] * 1000.0 > 2.0]
                depth = max([c.depth_mm for c in checks if c.crossing], default=0.0)
                worst = max(worst, depth)
                safety_line = [f"wall safety: {controller.mode}, retry {controller.retry}/{len(controller.reroll_deg)}",
                               f"true wire beyond the wall {depth:.1f} mm (worst {worst:.1f})"]
                held = held + 1 if controller.mode == "abort" and not np.any(act) else 0
                if held >= 15:  # backed off and held for 3 s: the operator takes over
                    done = True
            dist = info["eval/true_distance_m"] * 1000
            ins, rot = obs["state"][17] * 1000, np.degrees(np.arctan2(obs["state"][18], obs["state"][19]))
            if done:
                banner = (f"GOAL REACHED in {step / 5:.1f} s  ({dist:.1f} mm, within 5 mm for 3 steps)" if info["success"]
                          else f"SAFE STOP at {ins:.0f} mm: wall crossing ahead, backed off"
                          if args.wall_safety and controller.mode == "abort"
                          else f"episode ended: {info.get('reset_reason')}  ({dist:.1f} mm)")
            target = (f"{args.variant or args.vmr_model.split('_')[0]}: {'organiser cas-reference goal' if their_goal else args.target} ({goal['branch']}), entry roll {goal['roll_deg']:g} deg" if args.target or args.variant
                      else f"goal {goal['id']} ({goal['split']}): {goal['branch']}" if goal else "goal: cas-reference")
            lines = [name, target, *([route_line] if route_line else []), f"step {step:3d}   simulated {step / 5:5.1f} s", f"distance to goal {dist:6.1f} mm",
                     f"guidewire  in {ins:6.1f} mm  roll {rot:6.1f} deg",
                     f"action  v {act[0]:+.2f}  w {act[1]:+.2f}  (catheter {act[2]:+.2f})",
                     f"peak force {bridge.latest_true().peak_force:.3f} N", *(safety_line or [])]
            if args.record:
                frames.append(compose(bridge.physics.screenshot(), bridge.last_viewer_jpeg, lines, banner))
            if step % 10 == 0 or done:
                print(f"step {step:3d}  t={step / 5:5.1f}s  dist={dist:6.1f} mm  ins={ins:6.1f} mm  roll={rot:6.1f}", flush=True)
        print(banner)
    finally:
        env.close()

    if args.record and frames:
        import imageio.v2 as imageio

        out = Path(args.record)
        out.parent.mkdir(parents=True, exist_ok=True)
        frames += [frames[-1]] * int(2 * args.fps)  # hold the final frame for 2 s
        if out.suffix.lower() == ".gif":
            small = [np.asarray(Image.fromarray(f).resize((f.shape[1] // 2, f.shape[0] // 2))) for f in frames]
            imageio.mimsave(out, small, duration=1.0 / min(args.fps, 15), loop=0)
        else:
            imageio.mimsave(out, frames, fps=args.fps, quality=8, macro_block_size=1)
        print(f"wrote {out} ({len(frames)} frames, {args.fps:g} fps = {args.fps / 5:g}x real time)")


if __name__ == "__main__":
    main()
