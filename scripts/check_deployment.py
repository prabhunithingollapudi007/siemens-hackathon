"""Check a deployment bundle from ``export_deployment.py`` the way the organiser's stack will run it.

    python scripts/check_deployment.py --bundle handover/stroke-nav-v70-left
    python scripts/check_deployment.py --bundle handover/stroke-nav-v70-left --episodes 1 --visible
    python scripts/check_deployment.py --bundle handover/stroke-nav-v70-left --static

1. static: each run folder against the bundled scenario with their own code (``load_scenario``: the run's
   ``scenario.json`` must equal the scenario manifest, as ``evaluation._make_environment`` requires;
   device parameters validated and hashed by their ``device_parameters`` module, equal to the hash in
   ``resolved-config.json`` and to this repo's ``device-parameters.json``)
2. rollouts (skipped with ``--static``): both runs loaded from the bundle files as their evaluator loads
   them (``SAC.load`` for the policy, the bundled ``evaluate_controller.load_controller`` for the
   controller) and run by their ``evaluation.run_evaluation`` on their physics in lockstep, on the
   bundled scenario. Stop their Docker stack first; one lockstep job at a time.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def static_checks(bundle: Path) -> tuple[str, list[Path], bool]:
    from stroke_nav.organiser.live.lockstep import organiser_rl

    sc, dp = organiser_rl("scenario"), organiser_rl("device_parameters")
    root = (bundle / "HCMR-EV-RL" / "scenarios").resolve()
    repo_hash = dp.device_parameters_hash(dp.load_device_parameters(ROOT / "device-parameters.json"))
    runs = sorted(p for p in (bundle / "HCMR-EV-RL" / "runs").iterdir() if p.is_dir())
    ok, scenario = True, None
    for run in runs:
        cfg = json.loads((run / "resolved-config.json").read_text(encoding="utf-8"))
        scenario = sc.load_scenario(cfg["scenario"]["name"], root)
        problems = []
        if json.loads((run / "scenario.json").read_text(encoding="utf-8")) != scenario.manifest:
            problems.append("scenario.json differs from the scenario manifest")
        if cfg["scenario"]["mesh_sha256"] != scenario.manifest["mesh_sha256"]:
            problems.append("mesh hash differs")
        h = dp.device_parameters_hash(dp.load_device_parameters(run / "device-parameters.json"))
        if h != cfg["device_parameters"]["sha256"] or h != repo_hash:
            problems.append(f"device-parameter hash {h[:8]} vs config {cfg['device_parameters']['sha256'][:8]}, repo {repo_hash[:8]}")
        model = "controller.json" if (run / "controller.json").is_file() else "final-model.zip"
        if not (run / model).is_file():
            problems.append(f"missing {model}")
        print(f"[static] {run.name}: scenario {scenario.name}, {len(scenario.goals)} goal(s), device parameters "
              f"{h[:8]}..., {model}: " + ("OK" if not problems else "FAIL " + "; ".join(problems)))
        ok &= not problems
    return scenario.name, runs, ok


def load_bundled_controller(run: Path, env):
    spec = importlib.util.spec_from_file_location("bundled_evaluate_controller", run / "evaluate_controller.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_controller(run / "controller.json", env=env)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="their evaluation uses the run's training seed (0)")
    ap.add_argument("--only", choices=("astar-roll", "bc"), default=None)
    ap.add_argument("--visible", action="store_true", help="show their physics page and noisy viewer")
    ap.add_argument("--static", action="store_true", help="only the file checks, no rollouts")
    args = ap.parse_args()

    scen_name, runs, ok = static_checks(args.bundle)
    if args.static or not ok:
        print("RESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    from stable_baselines3 import SAC

    from stroke_nav.organiser import ensure_policy_importable
    from stroke_nav.organiser.live import lockstep
    from stroke_nav.organiser.spec import DEFAULT_SPEC

    others = lockstep.other_lockstep_jobs()
    if others:
        raise SystemExit("another lockstep job is running (one at a time):\n  " + "\n  ".join(others))
    lockstep.kill_stray_pages()
    ensure_policy_importable()
    try:
        evaluation = lockstep.organiser_rl("evaluation")
    except ModuleNotFoundError as e:  # their ros_io needs rclpy (ROS 2); run_evaluation itself does not use it
        if e.name != "rclpy":
            raise
        import types

        stub = types.ModuleType(f"{lockstep.ALIAS}.ros_io")
        stub.OpenNG3Bridge = None
        sys.modules[stub.__name__] = stub
        sys.modules.pop(f"{lockstep.ALIAS}.evaluation", None)
        evaluation = lockstep.organiser_rl("evaluation")
    env = lockstep.make_env(visible=args.visible, scenario_name=scen_name,
                            scenario_root=(args.bundle / "HCMR-EV-RL" / "scenarios").resolve())
    try:
        for run in runs:
            kind = "astar-roll" if (run / "controller.json").is_file() else "bc"
            if args.only and kind != args.only:
                continue
            model = load_bundled_controller(run, env) if kind == "astar-roll" else SAC.load(str(run / "final-model.zip"), env=env, device="cpu")
            results = evaluation.run_evaluation(model, env, episodes=args.episodes, seed=args.seed,
                                                max_steps=DEFAULT_SPEC.environment.max_steps)
            for r in results:
                print(f"[{kind}] {run.name} episode {r['episode_index']}: {r['outcome']}, {r['steps']} steps "
                      f"({r['steps'] / DEFAULT_SPEC.environment.control_hz:.1f} s), final distance "
                      f"{r['final_true_distance_m'] * 1000:.1f} mm, return {r['return']:.1f}", flush=True)
                ok &= r["success"]
    finally:
        env.close()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
