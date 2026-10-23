"""DAgger on the organiser's physics (lockstep env) with the roll-table expert as the teacher.

Each round trains a behaviour-cloned SAC model (train_bc.py) on every demonstration so far, lets that
model drive on the train goals while ``RecoveringExpert`` labels every step (record_demos.py
--policy, executing the expert instead with probability beta), and adds those episodes to the data.
Unsuccessful rollouts are kept: their labels are how to recover. After the last round the final
model is trained on everything and evaluated on the held-out goals; the round-0 model (plain
behaviour cloning on the expert demos) is evaluated too, for comparison.

    python scripts/run_dagger.py --goals results/lockstep/goals_signed.json --init runs/demos/signed-expert \\
        --betas 0.3,0 --episodes 60 --workers 4 --run-id stroke-nav-dagger

Writes runs/<run-id>-r<k>/ (models), runs/demos/<run-id>-r<k>/ (rollouts), and
runs/<run-id>/eval-lockstep-test.json for the final model.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(args: list[str], log: Path) -> None:
    print(f"[dagger] {' '.join(args)}", flush=True)
    t0 = time.monotonic()
    with open(log, "a", encoding="utf-8") as f:
        r = subprocess.run([PY, *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit(f"[dagger] failed ({r.returncode}); see {log}")
    print(f"[dagger]   done in {(time.monotonic() - t0) / 60:.1f} min", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--goals", required=True)
    ap.add_argument("--init", required=True, help="comma-separated folders of expert demos to start from")
    ap.add_argument("--betas", default="0.3,0", help="one DAgger round per value: probability of executing the expert")
    ap.add_argument("--episodes", type=int, default=60, help="rollouts per round")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--slow-radius", type=float, default=8.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--eval-episodes", type=int, default=33)
    ap.add_argument("--no-eval-baseline", dest="eval_baseline", action="store_false",
                    help="skip evaluating the round-0 (plain behaviour cloning) model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--run-id", default="stroke-nav-dagger")
    args = ap.parse_args()

    log = ROOT / "runs" / f"{args.run_id}.log"
    demos = [d for d in args.init.split(",") if d]
    model = None
    for k, beta in enumerate(float(b) for b in args.betas.split(",")):
        run_id = f"{args.run_id}-r{k}"
        run(["scripts/train_bc.py", "--demos", ",".join(demos), "--epochs", str(args.epochs),
             "--device", args.device, "--run-id", run_id], log)
        model = f"runs/{run_id}/final-model.zip"
        if k == 0 and args.eval_baseline:  # plain behaviour cloning, for comparison
            run(["scripts/eval_lockstep.py", "--model", model, "--goals", args.goals, "--split", "test",
                 "--episodes", str(args.eval_episodes), "--workers", str(args.workers),
                 "--max-steps", str(args.max_steps)], log)
        out = f"runs/demos/{run_id}"
        run(["scripts/record_demos.py", "--goals", args.goals, "--split", "train", "--expert", "recovering",
             "--policy", model, "--beta", str(beta), "--episodes", str(args.episodes), "--sigmas", "0",
             "--slow-radius", str(args.slow_radius), "--workers", str(args.workers),
             "--max-steps", str(args.max_steps), "--seed", str(10_000 * (k + 1)), "--out", out], log)
        demos.append(out)
    run(["scripts/train_bc.py", "--demos", ",".join(demos), "--epochs", str(args.epochs),
         "--device", args.device, "--run-id", args.run_id], log)
    run(["scripts/eval_lockstep.py", "--model", f"runs/{args.run_id}/final-model.zip", "--goals", args.goals,
         "--split", "test", "--episodes", str(args.eval_episodes), "--workers", str(args.workers),
         "--max-steps", str(args.max_steps)], log)
    print(f"[dagger] final model runs/{args.run_id}/final-model.zip; results runs/{args.run_id}/eval-lockstep-test.json")


if __name__ == "__main__":
    main()
