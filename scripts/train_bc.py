"""Behaviour-clone expert demonstrations into the organiser's SAC policy and write a hand-over run.

The model is an ordinary SB3 ``SAC("MultiInputPolicy")`` with ``openng3_rl.policy.FluoroStateExtractor``
(same spaces and hyperparameters as ``train_sac.py`` / their ``openng3-rl-train``), so their
evaluation loads it unchanged. Training touches the actor only, plus an optional critic warm start:

* actor: deterministic action ``tanh(mean)`` regressed onto the expert labels (MSE); ``log_std``
  pulled towards ``--log-std`` so stochastic rollouts stay close to the expert
* critic (``--critic-epochs``): both Q heads regressed onto the discounted Monte-Carlo return of the
  executed actions, so SAC fine-tuning from this model does not start from a random critic

    python scripts/train_bc.py --demos runs/demos/cas-reference --epochs 30
    python scripts/train_bc.py --demos runs/demos/a,runs/demos/b --successful-only --device cuda

Writes runs/<run_id>/final-model.zip plus the organiser run-folder files and bc-training.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stroke_nav.organiser import ensure_policy_importable  # noqa: E402


class SpacesOnly(gym.Env):
    """The organiser's observation and action spaces, for building the model without a simulator."""

    def __init__(self):
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, (1, 128, 128), dtype=np.uint8),
            "state": gym.spaces.Box(-10.0, 10.0, (43,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, True, False, {}


def load_demos(dirs: list[Path], successful_only: bool, gamma: float) -> list[dict]:
    episodes = []
    for d in dirs:
        for f in sorted(d.glob("ep_*.npz")):
            z = np.load(f)
            if successful_only and not bool(z["success"]):
                continue
            r = z["rewards"].astype(np.float64)
            ret = np.zeros_like(r)
            acc = 0.0
            for t in range(len(r) - 1, -1, -1):  # episodes end in success, unsafe or truncation: treat all as terminal
                acc = r[t] + gamma * acc
                ret[t] = acc
            episodes.append({"file": str(f), "images": z["images"], "states": z["states"], "labels": z["labels"],
                             "actions": z["actions"], "returns": ret.astype(np.float32), "success": bool(z["success"])})
    return episodes


def stack(episodes: list[dict]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([e[k] for e in episodes]) for k in ("images", "states", "labels", "actions", "returns")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demos", required=True, help="comma-separated folders of ep_*.npz (record_demos.py)")
    ap.add_argument("--successful-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--critic-epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log-std", type=float, default=-2.0, help="target log std of the policy (~0.14 at -2)")
    ap.add_argument("--state-noise", type=float, default=0.002, help="Gaussian augmentation on the state (obs units)")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--run-root", default="runs")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    ensure_policy_importable()
    import torch
    import torch.nn.functional as F
    from openng3_rl.policy import FluoroStateExtractor
    from stable_baselines3 import SAC

    from stroke_nav.organiser.export import check_run_dir, new_run_id, write_run_dir
    from stroke_nav.organiser.frame import default_frame
    from stroke_nav.organiser.spec import DEFAULT_SPEC

    tr = DEFAULT_SPEC.training
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    episodes = load_demos([Path(p) for p in args.demos.split(",")], args.successful_only, tr.gamma)
    if len(episodes) < 2:
        raise SystemExit(f"need at least 2 demonstration episodes, found {len(episodes)}")
    order = rng.permutation(len(episodes))
    n_val = max(1, int(round(args.val_fraction * len(episodes))))
    val, train = stack([episodes[i] for i in order[:n_val]]), stack([episodes[i] for i in order[n_val:]])
    print(f"[bc] {len(episodes)} episodes ({sum(e['success'] for e in episodes)} successful): "
          f"{len(train['labels'])} train / {len(val['labels'])} val samples")

    run_dir = Path(args.run_root) / (args.run_id or new_run_id("stroke-nav-bc"))
    run_dir.mkdir(parents=True, exist_ok=True)
    model = SAC("MultiInputPolicy", SpacesOnly(), policy_kwargs={"features_extractor_class": FluoroStateExtractor},
                learning_rate=tr.learning_rate, buffer_size=1, learning_starts=tr.learning_starts,
                batch_size=tr.batch_size, gamma=tr.gamma, tau=tr.tau, train_freq=(1, "step"), gradient_steps=1,
                seed=args.seed, device=args.device, verbose=0)
    dev = model.device
    actor, critic = model.actor, model.critic

    def batch(data: dict, idx: np.ndarray, augment: bool):
        state = data["states"][idx]
        if augment and args.state_noise > 0:
            state = state + rng.normal(0.0, args.state_noise, state.shape).astype(np.float32)
        obs = {"image": torch.as_tensor(data["images"][idx], dtype=torch.float32, device=dev),
               "state": torch.as_tensor(np.clip(state, -10.0, 10.0), device=dev)}
        return obs, {k: torch.as_tensor(data[k][idx], device=dev) for k in ("labels", "actions", "returns")}

    def actor_loss(obs, target):
        mean, log_std, _ = actor.get_action_dist_params(obs)
        mse = F.mse_loss(torch.tanh(mean), target["labels"])
        return mse, mse + 0.01 * F.mse_loss(log_std, torch.full_like(log_std, args.log_std))

    history = []
    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    n = len(train["labels"])
    t0 = time.time()
    for epoch in range(args.epochs):
        actor.train()
        perm, losses = rng.permutation(n), []
        for start in range(0, n, args.batch_size):
            obs, target = batch(train, perm[start:start + args.batch_size], augment=True)
            _, loss = actor_loss(obs, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        actor.eval()
        with torch.no_grad():
            val_mse = np.mean([float(actor_loss(*batch(val, np.arange(s, min(len(val["labels"]), s + 512)), False))[0])
                               for s in range(0, len(val["labels"]), 512)])
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_action_mse": float(val_mse)})
        print(f"[bc] epoch {epoch + 1:3d}  train {np.mean(losses):.5f}  val action MSE {val_mse:.5f}", flush=True)

    if args.critic_epochs:
        copt = torch.optim.Adam(critic.parameters(), lr=args.lr)
        for epoch in range(args.critic_epochs):
            perm, losses = rng.permutation(n), []
            for start in range(0, n, args.batch_size):
                obs, target = batch(train, perm[start:start + args.batch_size], augment=True)
                qs = critic(obs, target["actions"])
                loss = sum(F.mse_loss(q.squeeze(-1), target["returns"]) for q in qs)
                copt.zero_grad()
                loss.backward()
                copt.step()
                losses.append(loss.item())
            print(f"[bc] critic epoch {epoch + 1:3d}  loss {np.mean(losses):.4f}", flush=True)
        model.critic_target.load_state_dict(critic.state_dict())
    train_s = time.time() - t0

    model.save(run_dir / "final-model.zip")
    summary = {"episodes": len(episodes), "train_samples": n, "val_samples": len(val["labels"]),
               "final_val_action_mse": history[-1]["val_action_mse"], "train_seconds": round(train_s, 1),
               "demos": args.demos, "successful_only": args.successful_only, "history": history}
    (run_dir / "bc-training.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_run_dir(run_dir, DEFAULT_SPEC, frame=default_frame(), extra={
        "trained_in": "behaviour cloning of EntryRollExpert demonstrations recorded on the organiser's physics "
                      "(stroke_nav.organiser.live.lockstep)",
        "bc": {k: v for k, v in summary.items() if k != "history"},
    })
    problems = check_run_dir(run_dir)
    print(f"[bc] wrote {run_dir} in {train_s / 60:.1f} min; organiser run-folder check:", "OK" if not problems else problems)


if __name__ == "__main__":
    main()
