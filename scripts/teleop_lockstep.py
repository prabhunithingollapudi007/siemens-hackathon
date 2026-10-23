"""Record demonstrations by keyboard teleoperation on the organiser's physics (lockstep env).

Their physics page and noisy fluoro viewer open in Chrome windows, as in demo_lockstep.py, with
the episode's goal shown in the physics view. You drive from a small "teleop" window that shows
the policy's fluoro input, the distance to the goal and the robot state. Every step is saved in
the format of record_demos.py (observation before the step, your action as the label), so
train_bc.py trains on manual demos, scripted ones, or both (--demos a,b).

    python scripts/pick_goals.py --target left_carotid@30 --target right_subclavian@40
    python scripts/teleop_lockstep.py --goals results/lockstep/manual_goals.json --episodes 3
    python scripts/teleop_lockstep.py --goals results/lockstep/manual_goals.json --goal-id 1 --out runs/demos/manual
    python scripts/teleop_lockstep.py --goals ... --headless --keys "d*10,w,_*140"     # scripted keys (testing)

Keys, in the teleop window (it must have the focus):

    W / S (or Up / Down)       hold to push / pull the guidewire
    A / D (or Left / Right)    hold to turn it (11.5 deg per 0.2 s step at full speed)
    1 / 2 / 3                  speed 25 / 50 / 100 %
    Enter                      end the episode and save it
    Backspace                  end it and discard it
    Esc                        quit (the current episode is discarded)

The wire moves only while a key is held: on Windows the window reads the key state itself
(GetAsyncKeyState). Elsewhere, or with --keys, keys are events: W / S latch a push or pull until
Space, and each A / D press turns one step.

The episode also ends, and is saved, on their success rule (within 5 mm of the goal for 3 steps)
or at --max-steps. On their rod the entry roll picks the branch: turn first, then push (the goal
file lists the calibrated roll for each goal). The simulator runs at most at real time (5 steps/s).
Stop their Docker stack first so the pages get the CPU.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stroke_nav.agents.entry_roll import GOAL_DELTA, GW_COS, GW_INSERTION, GW_SIN  # noqa: E402
from stroke_nav.agents.roll_table import load_goal_set  # noqa: E402

STEP_S = 0.2  # their 5 Hz control period
ENTER, BACKSPACE, ESC = (13, 10), (8,), (27,)


class Teleop:
    """Keyboard -> organiser action ``[v_guidewire, w_guidewire, 0, 0]`` in [-1, 1].

    With ``held`` keys (hold mode) the action follows the keys held during the step. Without them
    (event mode) W / S latch a push or pull and each A / D press queues one turning step; the queue
    is short so a held, auto-repeating key does not keep the wire turning long after release.
    """

    max_turn_queue = 5

    def __init__(self):
        self.push = 0  # +1 push, -1 pull, 0 hold
        self.speed = 1.0
        self.turn = 0  # queued rotation steps, signed

    def handle(self, key: int, hold_mode: bool = False) -> str | None:
        """Apply one key code; returns "save", "discard" or "quit" when the episode should end."""
        if key in ENTER:
            return "save"
        if key in BACKSPACE:
            return "discard"
        if key in ESC:
            return "quit"
        c = chr(key).lower() if 0 <= key < 256 else ""
        if hold_mode and c in ("w", "s", " ", "a", "d"):
            return None  # motion comes from the held keys
        if c == "w":
            self.push = 1
        elif c == "s":
            self.push = -1
        elif c == " ":
            self.push = 0
        elif c in ("a", "d"):
            self.turn = int(np.clip(self.turn + (1 if c == "d" else -1), -self.max_turn_queue, self.max_turn_queue))
        elif c in ("1", "2", "3"):
            self.speed = {"1": 0.25, "2": 0.5, "3": 1.0}[c]
        return None

    def action(self, held: set[str] | None = None) -> np.ndarray:
        if held is not None:
            self.push = int("w" in held) - int("s" in held)
            w = float(int("d" in held) - int("a" in held))
        else:
            w = float(np.sign(self.turn))
            self.turn -= int(w)
        self.last_turn = w
        return np.array([self.push * self.speed, w * self.speed, 0.0, 0.0], dtype=np.float32)

    def describe(self) -> str:
        move = {1: "push", -1: "pull", 0: "hold"}[self.push]
        turn = {1.0: "turn +", -1.0: "turn -", 0.0: "no turn"}[getattr(self, "last_turn", 0.0)]
        return f"{move}  {turn}  speed {self.speed * 100:.0f}%"


class ScriptedKeys:
    """Key presses per control step from a script like ``"d*10,w,_*140"`` (``_`` = no key); Enter at the end."""

    def __init__(self, script: str):
        self.steps: list[list[int]] = []
        for part in filter(None, (p.strip() for p in script.split(","))):
            key, _, n = part.partition("*")
            code = {"space": ord(" "), "enter": 13, "_": None}.get(key.lower(), ord(key[0]) if key else None)
            self.steps += [[code] if code is not None else [] for _ in range(int(n) if n else 1)]

    def poll(self, deadline: float) -> tuple[list[int], None]:
        del deadline
        return (self.steps.pop(0) if self.steps else [13]), None

    def show(self, image: np.ndarray) -> None:
        pass

    def close(self) -> None:
        pass


HOLD_KEYS = {"w": (0x57, 0x26), "s": (0x53, 0x28), "a": (0x41, 0x25), "d": (0x44, 0x27)}  # letter, arrow


class WindowKeys:
    """An OpenCV window that shows the operator view and collects key presses; on Windows it also
    reads which motion keys are held, while the window has the focus."""

    def __init__(self, title: str = "teleop"):
        import cv2

        self.cv2, self.title = cv2, title
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        self.user32 = None
        if sys.platform == "win32":
            import ctypes

            self.user32 = ctypes.windll.user32

    @property
    def hold_mode(self) -> bool:
        return self.user32 is not None

    def show(self, image: np.ndarray) -> None:
        self.cv2.imshow(self.title, image)

    def _held(self) -> set[str]:
        u = self.user32
        if u.GetForegroundWindow() != u.FindWindowW(None, self.title):
            return set()  # keys typed into another window do not drive the robot
        return {k for k, vks in HOLD_KEYS.items() if any(u.GetAsyncKeyState(vk) & 0x8000 for vk in vks)}

    def poll(self, deadline: float) -> tuple[list[int], set[str] | None]:
        """Key events until ``deadline``, and (hold mode) every motion key held at some point then."""
        keys, held = [], set() if self.hold_mode else None
        while True:
            if held is not None:
                held |= self._held()
            left_ms = int((deadline - time.monotonic()) * 1000)
            k = self.cv2.waitKeyEx(max(1, min(left_ms, 15)))
            if k != -1:
                keys.append(k & 0xFFFF if k > 0xFFFF else k)
            elif left_ms <= 1:
                if held is not None:
                    held |= self._held()
                return keys, held

    def close(self) -> None:
        self.cv2.destroyWindow(self.title)


def operator_view(fluoro_jpeg: bytes | None, lines: list[str]) -> np.ndarray:
    import cv2

    side = 480
    img = np.zeros((side + 30 * len(lines) + 20, side, 3), np.uint8)
    if fluoro_jpeg:
        f = cv2.imdecode(np.frombuffer(fluoro_jpeg, np.uint8), cv2.IMREAD_COLOR)
        if f is not None:
            img[:side] = cv2.resize(f, (side, side))
    for i, line in enumerate(lines):
        cv2.putText(img, line, (10, side + 28 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 240), 1, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--goals", required=True, help="goal set (pick_goals.py or make_goals.py)")
    ap.add_argument("--goal-id", default=None, help="comma-separated goal ids (default: every goal of --split)")
    ap.add_argument("--split", default="train,test", help="splits to take goals from when --goal-id is not given")
    ap.add_argument("--episodes", type=int, default=1, help="episodes per goal")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--out", default=str(ROOT / "runs" / "demos" / "manual"))
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--headless", action="store_true", help="do not show their Chrome windows")
    ap.add_argument("--keys", default=None, help="scripted keys instead of the keyboard, e.g. 'd*10,w,_*140'")
    args = ap.parse_args()

    from stroke_nav.organiser.live.lockstep import make_env, show_goal

    goals = load_goal_set(args.goals, "train,test,cas")
    if args.goal_id is not None:
        ids = [int(i) for i in args.goal_id.split(",")]
        goals = [g for g in goals if g["id"] in ids]
    else:
        goals = [g for g in goals if g["split"] in set(args.split.split(","))]
    if not goals:
        ap.error("no goals selected")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    episode = len(list(out.glob("ep_*.npz")))
    keys = ScriptedKeys(args.keys) if args.keys else WindowKeys()
    env = make_env(worker=0, seed=args.seed, goals=[g["goal_m"] for g in goals], max_steps=args.max_steps,
                   visible=not args.headless)
    saved, quit_ = 0, False
    try:
        for goal in goals:
            for rep in range(args.episodes):
                show_goal(env.bridge.physics, goal["goal_m"])
                env.goal_sampler.set_next(goal["goal_m"])
                seed = args.seed + episode
                obs, info = env.reset(seed=seed)
                ctl = Teleop()
                images, states, actions, dists, rewards = [], [], [], [], []
                rot, last_raw, end, dist = 0.0, None, None, float("nan")
                label = f"goal {goal['id']} ({goal['split']}): {goal['branch']}"
                if goal.get("roll_deg") is not None:
                    label += f", entry roll {goal['roll_deg']:g} deg"
                while end is None:
                    t_step = time.monotonic()
                    raw = math.atan2(float(obs["state"][GW_SIN]), float(obs["state"][GW_COS]))
                    rot = raw if last_raw is None else rot + (raw - last_raw + math.pi) % (2 * math.pi) - math.pi
                    last_raw = raw
                    seen = float(np.linalg.norm(obs["state"][GOAL_DELTA]) * 1000.0)
                    if not args.keys:
                        keys.show(operator_view(env.bridge.last_viewer_jpeg, [
                            label, f"episode {rep + 1}/{args.episodes}   step {len(actions):3d}   {len(actions) * STEP_S:5.1f} s",
                            f"distance to goal {seen:6.1f} mm (observed)",
                            f"guidewire in {float(obs['state'][GW_INSERTION]) * 1000:6.1f} mm   roll {math.degrees(rot):7.1f} deg",
                            f"command: {ctl.describe()}",
                            ("hold W/S push/pull, hold A/D turn, 1/2/3 speed" if getattr(keys, "hold_mode", False)
                             else "W/S push/pull  Space stop  A/D turn  1/2/3 speed"),
                            "Enter save   Backspace discard   Esc quit"]))
                    events, held = keys.poll(t_step + STEP_S)
                    for k in events:
                        end = ctl.handle(k, hold_mode=held is not None) or end
                    if end is not None:
                        break
                    act = ctl.action(held)
                    images.append(obs["image"])
                    states.append(obs["state"])
                    actions.append(act)
                    obs, r, term, trunc, info = env.step(act)
                    dist = info["eval/true_distance_m"] * 1000.0
                    dists.append(info["eval/true_distance_m"])
                    rewards.append(r)
                    if term or trunc:
                        end = "save"
                quit_ = end == "quit"
                if end == "save" and actions:
                    acts = np.stack(actions)
                    np.savez_compressed(
                        out / f"ep_{episode:04d}.npz", images=np.stack(images), states=np.stack(states), labels=acts,
                        actions=acts, true_dist_m=np.asarray(dists, np.float32), rewards=np.asarray(rewards, np.float32),
                        success=bool(info.get("success", False)), steps=len(acts), goal_m=np.asarray(goal["goal_m"], np.float32),
                        goal_id=goal["id"], branch=goal["branch"], roll_deg=np.nan, sigma=0.0, seed=seed, source="manual")
                    row = {"episode": episode, "goal_id": goal["id"], "branch": goal["branch"], "source": "manual",
                           "success": bool(info.get("success", False)), "steps": len(acts),
                           "min_dist_mm": round(float(np.min(dists)) * 1000, 2), "final_dist_mm": round(dist, 2)}
                    with open(out / "episodes.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")
                    print(json.dumps(row), flush=True)
                    episode += 1
                    saved += 1
                else:
                    print(f"[teleop] goal {goal['id']}: episode {'discarded' if end != 'quit' else 'discarded, quitting'}")
                if quit_:
                    break
            if quit_:
                break
    finally:
        keys.close()
        env.close()
    print(f"[teleop] saved {saved} episodes to {out}")


if __name__ == "__main__":
    main()
