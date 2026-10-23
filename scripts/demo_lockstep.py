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
    for i, line in enumerate(lines):
        d.text((12, side + 44 + 30 * i), line, fill=(235, 235, 240), font=_font(20))
    frame = Image.new("RGB", (left.width + side + (left.width + side) % 2, HEIGHT))
    frame.paste(left, (0, 0))
    frame.paste(right, (left.width, 0))
    if banner:
        dd = ImageDraw.Draw(frame)
        dd.rectangle((0, 0, left.width, 52), fill=(20, 110, 60))
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
    args = ap.parse_args()

    goal = None
    if args.goals:
        goals = load_goal_set(args.goals, "train,test,cas")
        goal = next((g for g in goals if g["id"] == args.goal_id), None) if args.goal_id is not None else             next(g for g in goals if g["split"] == "cas")
        if goal is None:
            ap.error(f"no goal with id {args.goal_id} in {args.goals}")
    if args.expert:
        policy = EntryRollExpert(roll_deg=goal["roll_deg"] if goal else 120.0, slow_radius_mm=args.slow_radius)
        name = "scripted expert (EntryRollExpert)"
    else:
        ensure_policy_importable()
        from stable_baselines3 import SAC

        model, name = SAC.load(args.model, device="cpu"), f"BC policy {Path(args.model).parent.name}"
    env = make_env(worker=0, seed=args.seed, visible=not args.headless, goals=[goal["goal_m"]] if goal else None)
    bridge = env.bridge
    frames = []
    try:
        if goal:
            show_goal(bridge.physics, goal["goal_m"])
            print(f"goal {goal['id']} ({goal['split']}): {goal['branch']}, entry roll {goal['roll_deg']:g} deg")
        obs, info = env.reset(seed=args.seed)
        if args.expert:
            policy.reset()
        done, step, banner = False, 0, None
        while not done:
            act = policy.act(obs) if args.expert else model.predict(obs, deterministic=True)[0]
            obs, _, term, trunc, info = env.step(np.asarray(act, np.float32))
            step += 1
            done = term or trunc
            dist = info["eval/true_distance_m"] * 1000
            ins, rot = obs["state"][17] * 1000, np.degrees(np.arctan2(obs["state"][18], obs["state"][19]))
            if done:
                banner = (f"GOAL REACHED in {step / 5:.1f} s  ({dist:.1f} mm, within 5 mm for 3 steps)" if info["success"]
                          else f"episode ended: {info.get('reset_reason')}  ({dist:.1f} mm)")
            target = f"goal {goal['id']} ({goal['split']}): {goal['branch']}" if goal else "goal: cas-reference"
            lines = [name, target, f"step {step:3d}   simulated {step / 5:5.1f} s", f"distance to goal {dist:6.1f} mm",
                     f"guidewire  in {ins:6.1f} mm  roll {rot:6.1f} deg",
                     f"action  v {act[0]:+.2f}  w {act[1]:+.2f}  (catheter {act[2]:+.2f})",
                     f"peak force {bridge.latest_true().peak_force:.3f} N"]
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
