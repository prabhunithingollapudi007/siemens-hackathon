"""Evaluate navigation policies with START metrics.

Examples:
    python scripts/evaluate.py --model 0248_H_AOCERE_CAS --target "random:left_carotid@-15--1|right_carotid@-15--1"
    python scripts/evaluate.py --policies oracle,random,forward --episodes 100
    python scripts/evaluate.py --policies oracle,ppo:runs/ppo/best_model.zip

Policies: oracle (scripted, privileged), random, forward (push both devices, no rotation),
ppo:<path to a stable-baselines3 PPO .zip>.
Every policy sees the same episode seeds (same targets), so results are paired.
Writes results/<name>/episodes.csv and results/<name>/start_summary.md.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from stroke_nav.agents.scripted import RandomAgent, ScriptedOracle, run_episode
from stroke_nav.env.vessel_env import EnvConfig, VesselNavEnv
from stroke_nav.eval.start_metrics import episode_start_metrics, markdown_table, summarize


class ForwardAgent:
    def reset(self, env):
        pass

    def act(self, env, obs=None):
        return np.array([1.0, 0.0, 1.0, 0.0])


class SB3Agent:
    def __init__(self, path: str):
        from stable_baselines3 import PPO

        self.model = PPO.load(path, device="cpu")

    def reset(self, env):
        pass

    def act(self, env, obs=None):
        action, _ = self.model.predict(obs, deterministic=True)
        return action


def make_agent(spec: str, seed: int):
    if spec == "oracle":
        return ScriptedOracle()
    if spec == "random":
        return RandomAgent(seed)
    if spec == "forward":
        return ForwardAgent()
    if spec.startswith("ppo:"):
        return SB3Agent(spec[4:])
    raise ValueError(f"unknown policy {spec!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="0248_H_AOCERE_CAS")
    p.add_argument("--variant", default=None)
    p.add_argument("--target", default="random:left_carotid@20-70|right_carotid@20-70",  # within the robot's travel
                   help="target spec (node id, branch@mm, xyz:x,y,z, random:..., file:targets.json)")
    p.add_argument("--start", default=None, help="start spec (default: anatomy default)")
    p.add_argument("--policies", default="oracle,random,forward")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=600)  # the organiser's episode length
    p.add_argument("--success-device", default="microcatheter", choices=["microcatheter", "guidewire"])
    p.add_argument("--name", default=None, help="results subfolder name")
    args = p.parse_args()

    cfg = EnvConfig(model=args.model, variant=args.variant, target=args.target, start=args.start,
                    max_steps=args.max_steps, success_device=args.success_device)
    env = VesselNavEnv(cfg)
    out_dir = Path("results") / (args.name or f"{args.model}_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries, rows = [], []
    for spec in [s.strip() for s in args.policies.split(",") if s.strip()]:
        agent = make_agent(spec, seed=0)
        eps = []
        for i in range(args.episodes):
            m = episode_start_metrics(run_episode(env, agent, seed=1000 + i))
            m["policy"], m["episode"] = spec, i
            eps.append(m)
        rows += eps
        summaries.append(summarize(spec.split("/")[-1], eps))
        print(f"[eval] {spec}: success {np.mean([e['success'] for e in eps]):.0%}")

    title = f"START metrics: {env.anat.name}, target {args.target}, success device {args.success_device}"
    md = markdown_table(summaries, title)
    (out_dir / "start_summary.md").write_text(md, encoding="utf-8")
    with open(out_dir / "episodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print()
    print(md)
    print(f"[eval] wrote {out_dir / 'start_summary.md'} and {out_dir / 'episodes.csv'}")


if __name__ == "__main__":
    main()
