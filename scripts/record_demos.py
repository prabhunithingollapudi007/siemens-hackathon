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

from stroke_nav.agents.entry_roll import EntryRollExpert  # noqa: E402
from stroke_nav.organiser.live.lockstep import make_env  # noqa: E402

WIRE_ONLY = np.array([1.0, 1.0, 0.0, 0.0])


def run_episode(env, expert: EntryRollExpert, seed: int, sigma: float, rng: np.random.Generator) -> dict:
    obs, info = env.reset(seed=seed)
    expert.reset()
    images, states, labels, actions, dists, rewards = [], [], [], [], [], []
    done = False
    while not done:
        label = expert.act(obs)
        noise = rng.normal(0.0, sigma, 4) * WIRE_ONLY if sigma else 0.0  # the expert never moves the catheter
        act = np.clip(label + noise, -1.0, 1.0).astype(np.float32)
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


def worker_job(worker: int, jobs: list[dict], out: Path | None, max_steps: int | None) -> list[dict]:
    env = make_env(worker=worker, seed=worker, max_steps=max_steps)
    rows = []
    try:
        for job in jobs:
            t0 = time.monotonic()
            expert = EntryRollExpert(roll_deg=job["roll"])
            ep = run_episode(env, expert, job["seed"], job["sigma"], np.random.default_rng(job["seed"]))
            row = {"episode": job["episode"], "roll_deg": job["roll"], "sigma": job["sigma"], "seed": job["seed"],
                   "success": ep["success"], "steps": ep["steps"], "reason": ep["reason"],
                   "min_dist_mm": round(float(ep["true_dist_m"].min()) * 1000, 2),
                   "final_dist_mm": round(float(ep["true_dist_m"][-1]) * 1000, 2), "wall_s": round(time.monotonic() - t0, 1)}
            if out is not None:
                np.savez_compressed(out / f"ep_{job['episode']:04d}.npz", **{k: v for k, v in ep.items() if k != "reason"},
                                    roll_deg=job["roll"], sigma=job["sigma"], seed=job["seed"])
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
    args = ap.parse_args()

    rolls = [float(r) for r in args.rolls.split(",")]
    sigmas = [float(s) for s in args.sigmas.split(",")]
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
                            [out] * args.workers, [args.max_steps] * args.workers))
    rows = sorted((r for p in parts for r in p), key=lambda r: r["episode"])
    ok = [r for r in rows if r["success"]]
    print(f"\n{len(ok)}/{len(rows)} successful in {(time.monotonic() - t0) / 60:.1f} min")
    for roll in rolls:
        rr = [r for r in rows if r["roll_deg"] == roll]
        print(f"  roll {roll:5.1f}: {sum(r['success'] for r in rr)}/{len(rr)} success, "
              f"median min distance {np.median([r['min_dist_mm'] for r in rr]):.1f} mm")
    if out is not None:
        with open(out / "episodes.jsonl", "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
