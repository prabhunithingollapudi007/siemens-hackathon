"""Record expert demonstrations on the organiser's physics (lockstep env) for behaviour cloning.

Runs ``EntryRollExpert`` in ``stroke_nav.organiser.live.lockstep`` (their pages, their env, their
observation, faster than real time) and writes one compressed ``.npz`` per episode:

    images (T,1,128,128) uint8, states (T,43) float32   observation before each step
    labels (T,4) float32                                 expert action for that observation
    actions (T,4) float32                                action executed (label + DART noise)
    true_dist_m (T,) float32, rewards (T,) float32        after each step
    success, steps, roll_deg, sigma, seed                 episode metadata

DART noise (``--sigmas``): the executed action is the expert's plus Gaussian noise, but the label
stays the expert's, so the clone also sees how to recover from small mistakes.

    python scripts/record_demos.py --episodes 4 --rolls 120 --no-save        # try the expert
    python scripts/record_demos.py --episodes 120 --workers 4 --out runs/demos/cas-ref

Multiple targets (``--goals``, from ``scripts/make_goals.py``): each episode gets one goal of the
chosen split as its goal and the expert uses that goal's entry roll; ``goal_m``, ``goal_id`` and
``branch`` are saved with the episode.

    python scripts/record_demos.py --goals results/lockstep/goals.json --split train --episodes 240
        --sigmas 0,0.2,0.4 --slow-radius 8 --workers 4 --max-steps 300 --out runs/demos/multi-target

DAgger (``--policy``): a trained model drives, except that with probability ``--beta`` per step the
expert's action is executed instead; the label is always the expert's. Use ``--expert recovering``
(``RecoveringExpert``) so the labels also say how to recover from a wrong roll.

    python scripts/record_demos.py --goals results/lockstep/goals_signed.json --expert recovering
        --policy runs/<bc_run>/final-model.zip --beta 0.3 --episodes 80 --workers 4 --out runs/demos/dagger-1

Needs the organiser sources (HCMR_EV_ROOT); stop their Docker stack first so the pages get the CPU.
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

from stroke_nav.agents.entry_roll import EntryRollExpert, RecoveringExpert  # noqa: E402
from stroke_nav.agents.roll_table import load_goal_set  # noqa: E402
from stroke_nav.organiser.live.lockstep import make_env  # noqa: E402

WIRE_ONLY = np.array([1.0, 1.0, 0.0, 0.0])


def run_episode(env, expert: EntryRollExpert, seed: int, sigma: float, rng: np.random.Generator,
                policy=None, beta: float = 1.0) -> dict:
    obs, info = env.reset(seed=seed)
    expert.reset()
    images, states, labels, actions, dists, rewards = [], [], [], [], [], []
    done = False
    while not done:
        label = expert.act(obs)  # the expert sees every observation, also when the policy drives
        base = label if policy is None or rng.random() < beta else policy.predict(obs, deterministic=True)[0]
        noise = rng.normal(0.0, sigma, 4) * WIRE_ONLY if sigma else 0.0  # the expert never moves the catheter
        act = np.clip(base + noise, -1.0, 1.0).astype(np.float32)
        images.append(obs["image"])
        states.append(obs["state"])
        labels.append(label)
        actions.append(act)
        obs, r, term, trunc, info = env.step(act)
        dists.append(info["eval/true_distance_m"])
        rewards.append(r)
        done = term or trunc
    return {"images": np.stack(images), "states": np.stack(states), "labels": np.stack(labels).astype(np.float32),
            "actions": np.stack(actions), "true_dist_m": np.asarray(dists, np.float32),
            "rewards": np.asarray(rewards, np.float32), "success": bool(info["success"]), "steps": len(labels),
            "reason": info.get("reset_reason")}


def worker_job(worker: int, jobs: list[dict], out: Path | None, max_steps: int | None, goals: list | None,
               slow_radius: float, expert_kind: str = "entry", policy_path: str | None = None,
               beta: float = 1.0) -> list[dict]:
    policy = None
    if policy_path:
        from stable_baselines3 import SAC

        policy = SAC.load(policy_path, device="cpu")
    env = make_env(worker=worker, seed=worker, max_steps=max_steps, goals=goals)
    rows = []
    try:
        for job in jobs:
            t0 = time.monotonic()
            cls = RecoveringExpert if expert_kind == "recovering" else EntryRollExpert
            expert = cls(roll_deg=job["roll"], slow_radius_mm=slow_radius)
            if "goal" in job:
                env.goal_sampler.set_next(job["goal"]["goal_m"])
            ep = run_episode(env, expert, job["seed"], job["sigma"], np.random.default_rng(job["seed"]), policy, beta)
            if "goal" in job:
                ep.update(goal_m=np.asarray(job["goal"]["goal_m"], np.float32), goal_id=job["goal"]["id"],
                          branch=job["goal"]["branch"])
            row = {"episode": job["episode"], "roll_deg": job["roll"], "sigma": job["sigma"], "seed": job["seed"],
                   "goal_id": job.get("goal", {}).get("id"), "branch": job.get("goal", {}).get("branch"),
                   "success": ep["success"], "steps": ep["steps"], "reason": ep["reason"],
                   "min_dist_mm": round(float(ep["true_dist_m"].min()) * 1000, 2),
                   "final_dist_mm": round(float(ep["true_dist_m"][-1]) * 1000, 2), "wall_s": round(time.monotonic() - t0, 1)}
            if out is not None:
                np.savez_compressed(out / f"ep_{job['episode']:04d}.npz", **{k: v for k, v in ep.items() if k != "reason"},
                                    roll_deg=job["roll"], sigma=job["sigma"], seed=job["seed"], slow_radius_mm=slow_radius,
                                    expert=expert_kind, policy=policy_path or "", beta=beta if policy_path else 1.0)
            print(json.dumps(row), flush=True)
            rows.append(row)
    finally:
        env.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--rolls", default="120", help="comma-separated entry rolls (deg), cycled over episodes")
    ap.add_argument("--sigmas", default="0", help="comma-separated DART noise levels, cycled over episodes")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=None, help="episode limit (organiser default 600)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/demos/cas-reference")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--slow-radius", type=float, default=25.0,
                    help="expert slows down inside this distance (mm); 8 suits goals along the whole path")
    ap.add_argument("--goals", default=None, help="goal set from scripts/make_goals.py (overrides --rolls)")
    ap.add_argument("--split", default="train", help="comma-separated goal splits to use (train, test, cas)")
    ap.add_argument("--expert", choices=("entry", "recovering"), default="entry")
    ap.add_argument("--policy", default=None, help="DAgger: SAC model that drives (labels stay the expert's)")
    ap.add_argument("--beta", type=float, default=0.0, help="DAgger: probability per step of executing the expert")
    args = ap.parse_args()
    if args.policy:
        from stroke_nav.organiser import ensure_policy_importable

        ensure_policy_importable()

    rolls = [float(r) for r in args.rolls.split(",")]
    sigmas = [float(s) for s in args.sigmas.split(",")]
    goal_set = None
    if args.goals:
        goal_set = load_goal_set(args.goals, args.split)
        order = np.random.default_rng(args.seed).permutation(len(goal_set))
        jobs = [{"episode": i, "goal": goal_set[order[i % len(goal_set)]], "sigma": sigmas[i % len(sigmas)],
                 "seed": args.seed + i} for i in range(args.episodes)]
        for j in jobs:
            j["roll"] = float(j["goal"]["roll_deg"])
        rolls = sorted({j["roll"] for j in jobs})
    else:
        jobs = [{"episode": i, "roll": rolls[i % len(rolls)], "sigma": sigmas[(i // len(rolls)) % len(sigmas)],
                 "seed": args.seed + i} for i in range(args.episodes)]
    out = None
    if not args.no_save:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        start = len(list(out.glob("ep_*.npz")))  # append to an existing folder
        for j in jobs:
            j["episode"] += start
    t0 = time.monotonic()
    with ThreadPoolExecutor(args.workers) as ex:
        parts = list(ex.map(worker_job, range(args.workers), [jobs[w::args.workers] for w in range(args.workers)],
                            [out] * args.workers, [args.max_steps] * args.workers,
                            [[g["goal_m"] for g in goal_set] if goal_set else None] * args.workers,
                            [args.slow_radius] * args.workers, [args.expert] * args.workers,
                            [args.policy] * args.workers, [args.beta] * args.workers))
    rows = sorted((r for p in parts for r in p), key=lambda r: r["episode"])
    ok = [r for r in rows if r["success"]]
    print(f"\n{len(ok)}/{len(rows)} successful in {(time.monotonic() - t0) / 60:.1f} min")
    key = "branch" if goal_set else "roll_deg"
    for value in sorted({r[key] for r in rows}):
        rr = [r for r in rows if r[key] == value]
        print(f"  {key} {value}: {sum(r['success'] for r in rr)}/{len(rr)} success, "
              f"median min distance {np.median([r['min_dist_mm'] for r in rr]):.1f} mm")
    if out is not None:
        with open(out / "episodes.jsonl", "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
