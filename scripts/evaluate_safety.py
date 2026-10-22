"""Safety evaluation: policies x tracking-fault levels x safety monitor off/on.

Examples:
    python scripts/evaluate_safety.py                                   # VMR 0248 (after fetch_data.py)
    python scripts/evaluate_safety.py --synthetic --episodes 30          # offline, synthetic arch tree
    python scripts/evaluate_safety.py --policies follower,random,forward --faults clean,stress

Policies: oracle, follower (A* plan + trajectory follower), sloppy (follower at full speed with roll
errors, a stand-in for an imperfect learned policy), random, forward, ppo:<path>.
Fault levels: clean, organiser (their noise layer: 0.1 mm, 7 % dropout, 1-step latency),
stress (1 mm, bursty dropouts that rise with tip speed, 1-step latency).
Every cell uses the same episode seeds, so the comparison is paired.
Writes results/<name>/safety_summary.md, safety_episodes.csv and safety_summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import make_agent as _make_base_agent  # noqa: E402

from stroke_nav.agents.follower import TrajectoryFollower  # noqa: E402
from stroke_nav.agents.scripted import run_episode  # noqa: E402
from stroke_nav.env.vessel_env import EnvConfig, VesselNavEnv  # noqa: E402
from stroke_nav.safety import FaultModel, SafetyShield  # noqa: E402
from stroke_nav.safety.report import markdown, summarize_safety  # noqa: E402


class SloppyFollower(TrajectoryFollower):
    """Stand-in for an imperfect learned controller: follows the plan, but always pushes at full speed
    and rotates imprecisely (Gaussian error on roll commands), so it sometimes enters wrong branches."""

    def __init__(self, seed: int, roll_noise: float = 0.8):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.roll_noise = roll_noise

    def act(self, env, obs=None):
        a = super().act(env, obs).copy()
        for i in (0, 2):
            if a[i] > 0.05:
                a[i] = 1.0
        a[[1, 3]] = np.clip(a[[1, 3]] + self.roll_noise * self.rng.standard_normal(2), -1, 1)
        return a


def make_agent(spec: str, seed: int):
    if spec == "follower":
        return TrajectoryFollower()
    if spec == "sloppy":
        return SloppyFollower(seed)
    return _make_base_agent(spec, seed)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="0248_H_AOCERE_CAS")
    p.add_argument("--variant", default=None)
    p.add_argument("--synthetic", action="store_true", help="use the built-in synthetic arch tree (no download)")
    p.add_argument("--target", default=None, help="default: random point in the distal L/R carotid path")
    p.add_argument("--policies", default="follower,sloppy,forward")
    p.add_argument("--faults", default="clean,organiser,stress")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=600, help="the organiser's episode length (10 mm/s devices)")
    p.add_argument("--no-robot-limits", action="store_true",
                   help="ignore the robot's travel (guidewire 264.5 mm); distal carotid targets lie beyond it")
    p.add_argument("--success-device", default="microcatheter", choices=["microcatheter", "guidewire"])
    p.add_argument("--name", default=None)
    args = p.parse_args()

    anat = None
    if args.synthetic:
        from stroke_nav.safety.demo_anatomy import demo_anatomy

        anat = demo_anatomy()
    target = args.target or ("random:left_carotid@-40--3|right_carotid@-40--3" if args.synthetic
                             else "random:left_carotid@-15--1|right_carotid@-15--1")
    from stroke_nav.env.devices import device_sim_params
    from stroke_nav.organiser.spec import describe_devices

    print(describe_devices(), flush=True)
    cfg = EnvConfig(model=args.model, variant=args.variant, target=target, max_steps=args.max_steps,
                    success_device=args.success_device, sim=device_sim_params(robot_limits=not args.no_robot_limits))
    base = VesselNavEnv(cfg, anatomy=anat)
    out_dir = Path("results") / (args.name or f"{base.anat.name}_safety")
    out_dir.mkdir(parents=True, exist_ok=True)

    columns, rows, summary = [], [], {}
    for spec in [s.strip() for s in args.policies.split(",") if s.strip()]:
        for fault in [f.strip() for f in args.faults.split(",") if f.strip()]:
            for mon in (False, True):
                env = SafetyShield(VesselNavEnv(cfg, anatomy=base.anat), faults=getattr(FaultModel, fault)(), monitor=mon)
                eps = []
                for i in range(args.episodes):
                    m = run_episode(env, make_agent(spec, seed=i), seed=2000 + i)
                    m.pop("return_terms", None)
                    m.update(policy=spec, faults=fault, monitor=mon, episode=i)
                    eps.append(m)
                rows += eps
                label = f"{spec.split('/')[-1]} · {fault} · {'monitor' if mon else 'no monitor'}"
                s = summarize_safety(eps)
                columns.append((label, s))
                summary[label] = s
                print(f"[safety] {label:40s} success {s['success_pct']:5.1f}%  wrong-depth worst {s['wrong_depth_worst_mm']:5.1f} mm  "
                      f"blocked-push eps {s['wall_push_eps_pct']:5.1f}%  overshoot worst {s['overshoot_worst_mm']:5.1f} mm  "
                      f"interventions {s['steps_intervened']:5.1f}", flush=True)

    title = f"Safety evaluation: {base.anat.name}, target {target}, {args.episodes} episodes per column"
    md = markdown(columns, title)
    (out_dir / "safety_summary.md").write_text(md, encoding="utf-8")
    (out_dir / "safety_summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    keys = sorted({k for r in rows for k in r})
    with open(out_dir / "safety_episodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[safety] wrote {out_dir / 'safety_summary.md'}, safety_episodes.csv, safety_summary.json")


if __name__ == "__main__":
    main()
