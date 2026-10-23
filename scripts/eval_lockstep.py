"""Evaluate a SAC model (or the scripted expert) on the organiser's physics in the lockstep env.

Their env, success rule (within 5 mm of the goal for 3 steps), observation and reward; episodes
up to 600 steps. Writes a JSON summary next to the model (or under results/lockstep/).

    python scripts/eval_lockstep.py --model runs/<run_id>/final-model.zip --episodes 10 --workers 2
    python scripts/eval_lockstep.py --expert --roll 120 --episodes 10
    python scripts/eval_lockstep.py --model runs/<run_id>/final-model.zip --goals results/lockstep/goals.json         --split test --episodes 40 --workers 4       # held-out goals; one goal per episode, per-branch results
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
from stroke_nav.agents.roll_table import load_goal_set  # noqa: E402
from stroke_nav.organiser import ensure_policy_importable  # noqa: E402
from stroke_nav.organiser.live.lockstep import make_env  # noqa: E402


def evaluate(worker: int, jobs: list[dict], args, goals: list | None) -> list[dict]:
    if args.expert:
        policy = EntryRollExpert(roll_deg=args.roll, slow_radius_mm=args.slow_radius)
    else:
        from stable_baselines3 import SAC

        model = SAC.load(args.model, device="cpu")
    env = make_env(worker=worker, seed=worker, max_steps=args.max_steps, goals=goals)
    rows = []
    try:
        for job in jobs:
            seed, goal = job["seed"], job.get("goal")
            t0 = time.monotonic()
            if goal is not None:
                env.goal_sampler.set_next(goal["goal_m"])
            obs, _ = env.reset(seed=seed)
            if args.expert:
                policy = EntryRollExpert(roll_deg=goal["roll_deg"] if goal is not None else args.roll,
                                         slow_radius_mm=args.slow_radius)
                policy.reset()
            dists, forces, done = [], [], False
            while not done:
                act = policy.act(obs) if args.expert else model.predict(obs, deterministic=not args.stochastic)[0]
                obs, _, term, trunc, info = env.step(np.asarray(act, np.float32))
                dists.append(info["eval/true_distance_m"] * 1000)
                forces.append(float(env.bridge.latest_true().peak_force))
                done = term or trunc
            row = {"seed": seed, "goal_id": goal["id"] if goal else None, "branch": goal["branch"] if goal else None,
                   "success": bool(info["success"]), "unsafe": bool(info["unsafe"]), "steps": len(dists),
                   "time_s": round(len(dists) / 5.0, 1), "min_dist_mm": round(min(dists), 2),
                   "final_dist_mm": round(dists[-1], 2), "peak_force_n": round(max(forces), 4),
                   "wall_s": round(time.monotonic() - t0, 1)}
            print(json.dumps(row), flush=True)
            rows.append(row)
    finally:
        env.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="SB3 SAC zip (e.g. runs/<run_id>/final-model.zip)")
    ap.add_argument("--expert", action="store_true", help="evaluate EntryRollExpert instead of a model")
    ap.add_argument("--roll", type=float, default=120.0)
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of the deterministic mean")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1000, help="first evaluation seed (demos use 0..)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--slow-radius", type=float, default=25.0,
                    help="expert slows down inside this distance (mm); 8 suits goals along the whole path")
    ap.add_argument("--goals", default=None, help="goal set from scripts/make_goals.py: one goal per episode")
    ap.add_argument("--split", default="test", help="comma-separated goal splits to evaluate (train, test, cas)")
    args = ap.parse_args()
    if not args.expert and not args.model:
        ap.error("pass --model or --expert")
    if not args.expert:
        ensure_policy_importable()

    goal_set = load_goal_set(args.goals, args.split) if args.goals else None
    jobs = [{"seed": args.seed + i, **({"goal": goal_set[i % len(goal_set)]} if goal_set else {})}
            for i in range(args.episodes)]
    goal_points = [g["goal_m"] for g in goal_set] if goal_set else None
    with ThreadPoolExecutor(args.workers) as ex:
        parts = list(ex.map(evaluate, range(args.workers), [jobs[w::args.workers] for w in range(args.workers)],
                            [args] * args.workers, [goal_points] * args.workers))
    rows = sorted((r for p in parts for r in p), key=lambda r: r["seed"])
    ok = [r for r in rows if r["success"]]
    summary = {
        "policy": f"EntryRollExpert(roll_deg={args.roll})" if args.expert else str(args.model),
        "deterministic": not args.stochastic, "episodes": len(rows), "success_rate": len(ok) / len(rows),
        "unsafe_rate": sum(r["unsafe"] for r in rows) / len(rows),
        "median_min_dist_mm": float(np.median([r["min_dist_mm"] for r in rows])),
        "median_final_dist_mm": float(np.median([r["final_dist_mm"] for r in rows])),
        "mean_time_to_success_s": float(np.mean([r["time_s"] for r in ok])) if ok else None,
        "max_peak_force_n": max(r["peak_force_n"] for r in rows),
        "simulator": "organiser XPBD physics page in lockstep (stroke_nav.organiser.live.lockstep)",
    }
    suffix = ""
    if goal_set:
        summary["goals"] = f"{args.goals} [{args.split}]"
        summary["per_branch"] = {b: {"episodes": sum(r["branch"] == b for r in rows),
                                     "success": sum(r["success"] for r in rows if r["branch"] == b)}
                                 for b in sorted({r["branch"] for r in rows})}
        suffix = f"-{args.split.replace(',', '-')}"
    out = Path(args.out) if args.out else (Path(args.model).parent / f"eval-lockstep{suffix}.json" if args.model
                                           else ROOT / "results" / "lockstep" / f"expert{suffix or f'_roll{args.roll:g}'}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "episodes": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
