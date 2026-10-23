"""Evaluate the stroke-nav entry-roll controller with the organiser's own evaluator.

Runs inside their ``hcmr/openng3-rl`` image, from a run folder under ``HCMR-EV-RL/runs/``:

    python3 /workspace/openng3_rl/runs/<run_id>/evaluate_controller.py --run-dir /workspace/openng3_rl/runs/<run_id> \
        --model controller.json --episodes 5 --scenario-root /workspace/openng3_rl/scenarios \
        --scenario-runtime /workspace/openng3_rl/runs/scenario-runtime.json [--capture-rollouts --rollout-id ID]

The arguments are those of their ``openng3-rl-evaluate``, and everything except loading the model is
their code (``openng3_rl.evaluation.main``): run-folder, scenario, device-parameter and renderer
checks, ``OpenNG3Env``, deterministic rollouts, LeRobot capture and ``evaluations/<id>/episodes.json``.
Only ``SAC.load`` is replaced: ``controller.json`` names an ``EntryRollExpert`` (``entry_roll.py``,
copied next to this file from ``stroke_nav.agents.entry_roll``) with its entry roll from the A* route
and the roll calibration. It reads the same 43-value observation a SAC policy gets and returns the
same normalised 4-vector action ``[v_guidewire, w_guidewire, v_catheter, w_catheter]``.

Self-contained on purpose: their container has NumPy and ``openng3_rl`` but not ``stroke_nav``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = "stroke-nav/entry-roll-controller/1"


class ControllerPolicy:
    """Duck-typed stand-in for an SB3 model: ``predict(observation, deterministic=True)``."""

    def __init__(self, controller) -> None:
        self.controller = controller

    def reset(self) -> None:
        self.controller.reset()

    def predict(self, observation, state=None, episode_start=None, deterministic: bool = True):
        del state, episode_start, deterministic  # the controller is deterministic
        return self.controller.act(observation), None


def load_controller(path: str | Path, env=None, **_ignored) -> ControllerPolicy:
    """Build the controller named in ``controller.json``. With ``env``, the controller is reset at the
    start of every episode (their ``run_evaluation`` calls ``env.reset`` and then ``predict`` only)."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from entry_roll import EntryRollExpert, RecoveringExpert

    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{path}: expected schema {SCHEMA!r}, got {doc.get('schema')!r}")
    kinds = {"EntryRollExpert": EntryRollExpert, "RecoveringExpert": RecoveringExpert}
    if doc["controller"] not in kinds:
        raise ValueError(f"{path}: unknown controller {doc['controller']!r}")
    policy = ControllerPolicy(kinds[doc["controller"]](**doc["params"]))
    policy.reset()
    if env is not None:
        env_reset = env.reset

        def reset(*args, **kwargs):
            policy.reset()
            return env_reset(*args, **kwargs)

        env.reset = reset
    print(f"[stroke-nav] {doc['controller']}({doc['params']}) from {path}", flush=True)
    return policy


class _ControllerLoader:
    """Replaces ``stable_baselines3.SAC`` inside their evaluation module (only ``load`` is used)."""

    @staticmethod
    def load(path, env=None, device=None, **kwargs):
        del device, kwargs
        return load_controller(path, env=env)


def main() -> None:
    from openng3_rl import evaluation

    evaluation.SAC = _ControllerLoader
    evaluation.main()


if __name__ == "__main__":
    main()
