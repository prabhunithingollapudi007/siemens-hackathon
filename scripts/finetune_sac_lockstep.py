"""Fine-tune a behaviour-cloned SAC model with SAC on the organiser's physics (lockstep env).

Loads a model from ``train_bc.py`` (their ``SAC("MultiInputPolicy")`` with ``FluoroStateExtractor``)
and keeps training it with SAC in ``stroke_nav.organiser.live.lockstep``, one env per worker process
(their pages, their env and reward), on goals of a ``make_goals.py`` set:

* the replay buffer is new (the BC model is saved with ``buffer_size=1``); the cloned policy fills
  it for ``--learning-starts`` transitions before any update
* the entropy coefficient starts at ``--init-ent-coef`` (SB3's automatic one starts at 1.0, which
  would pull the cloned policy towards random actions) and is then tuned automatically
* the critic starts from train_bc.py's Monte-Carlo warm start

    python scripts/finetune_sac_lockstep.py --init runs/stroke-nav-bc-multi/final-model.zip \\
        --goals results/lockstep/goals.json --split train --timesteps 20000 --workers 4

Writes runs/<run_id>/ (final-model.zip, checkpoints/, organiser run-folder files, finetune.json).
Lockstep runs at about 3-4 steps/s per worker; stop their Docker stack first.

Status (2026-10-23): written but not yet run.
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

from stroke_nav.organiser import ensure_policy_importable  # noqa: E402


def env_factory(worker: int, goals: list, max_steps: int | None):
    def make():
        sys.path.insert(0, str(ROOT / "src"))
        from stable_baselines3.common.monitor import Monitor

        from stroke_nav.organiser import ensure_policy_importable as ensure
        from stroke_nav.organiser.live.lockstep import make_env

        ensure()
        return Monitor(make_env(worker=worker, seed=worker, goals=goals, max_steps=max_steps),
                       info_keywords=("success",))
    return make


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True, help="SAC zip to start from (e.g. a train_bc.py final-model.zip)")
    ap.add_argument("--goals", required=True, help="goal set from scripts/make_goals.py")
    ap.add_argument("--split", default="train")
    ap.add_argument("--timesteps", type=int, default=20_000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=300, help="episode limit (organiser default 600)")
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--buffer-size", type=int, default=50_000)
    ap.add_argument("--learning-starts", type=int, default=1_000)
    ap.add_argument("--init-ent-coef", type=float, default=0.01)
    ap.add_argument("--checkpoint-every", type=int, default=2_000)
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="stop learning after this much wall time (a training budget; --timesteps is then an upper bound)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--run-root", default="runs")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from stroke_nav.agents.roll_table import load_goal_set
    from stroke_nav.organiser.export import check_run_dir, new_run_id, write_run_dir
    from stroke_nav.organiser.frame import default_frame
    from stroke_nav.organiser.spec import DEFAULT_SPEC

    ensure_policy_importable()
    goals = [g["goal_m"] for g in load_goal_set(args.goals, args.split)]
    run_dir = Path(args.run_root) / (args.run_id or new_run_id("stroke-nav-sacft"))
    run_dir.mkdir(parents=True, exist_ok=True)
    venv = SubprocVecEnv([env_factory(w, goals, args.max_steps) for w in range(args.workers)], start_method="spawn")
    model = SAC.load(args.init, env=venv, device=args.device, custom_objects={
        "learning_rate": args.learning_rate, "lr_schedule": lambda _: args.learning_rate,
        "buffer_size": args.buffer_size, "learning_starts": args.learning_starts,
        "gradient_steps": 1, "train_freq": (1, "step")})
    if getattr(model, "log_ent_coef", None) is not None:
        with torch.no_grad():
            model.log_ent_coef.fill_(math.log(args.init_ent_coef))
    model.tensorboard_log = str(run_dir / "tensorboard")
    t0 = time.time()

    class TimeBudget(BaseCallback):
        """Ends ``learn`` once the wall-time budget is spent (SB3 stops when a callback returns False)."""

        def _on_step(self) -> bool:
            return args.max_minutes is None or time.time() - t0 < 60.0 * args.max_minutes

    try:
        model.learn(total_timesteps=args.timesteps, reset_num_timesteps=True, log_interval=4,
                    callback=[CheckpointCallback(max(1, args.checkpoint_every // args.workers), str(run_dir / "checkpoints"),
                                                 name_prefix="sac"), TimeBudget()])
    finally:
        model.save(run_dir / "final-model.zip")
        venv.close()
    episodes = [e for e in model.ep_info_buffer]
    summary = {"init": args.init, "goals": f"{args.goals} [{args.split}]", "timesteps": args.timesteps,
               "workers": args.workers, "learning_rate": args.learning_rate, "init_ent_coef": args.init_ent_coef,
               "train_seconds": round(time.time() - t0, 1),
               "last_episodes": len(episodes),
               "last_success_rate": float(np.mean([e.get("success", 0.0) for e in episodes])) if episodes else None,
               "last_mean_return": float(np.mean([e["r"] for e in episodes])) if episodes else None}
    (run_dir / "finetune.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_run_dir(run_dir, DEFAULT_SPEC, frame=default_frame(), extra={
        "trained_in": "SAC fine-tuning of a behaviour-cloned policy on the organiser's physics "
                      "(stroke_nav.organiser.live.lockstep)", "finetune": summary})
    problems = check_run_dir(run_dir)
    print(json.dumps(summary, indent=2))
    print(f"[sacft] wrote {run_dir}; organiser run-folder check:", "OK" if not problems else problems)


if __name__ == "__main__":
    main()
