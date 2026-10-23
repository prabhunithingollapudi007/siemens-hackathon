"""Package a planned case for deployment on the organiser's stack: the A* + entry-roll controller and the
behaviour-cloned SAC policy, on the case's own scenario, runnable with their ``./sys``.

    python scripts/export_deployment.py --case results/cases/v70-left
    python scripts/export_deployment.py --case results/cases/v70-left --name stroke-nav-v70-left --slow-radius 8

Reads ``results/cases/<id>/`` (from ``plan_case.py`` with ``--methods astar,roll,bc``) and the case's
scenario under ``data/scenarios/``. Writes ``handover/<name>/`` and ``handover/<name>.zip``:

    HCMR-EV-RL/scenarios/<scenario>/      the case mesh with only the case goal (their scenarios draw a
                                          goal uniformly per episode)
    HCMR-EV-RL/runs/stroke-nav-bc-<id>/   SAC run folder: their showcase evaluates it unchanged
    HCMR-EV-RL/runs/stroke-nav-astar-roll-<id>/
                                          run folder with controller.json (entry roll from the A* route
                                          and the roll calibration), entry_roll.py, evaluate_controller.py
                                          (their evaluator with the controller in place of SAC.load)
    deploy.sh                             install | bc | astar-roll | results | shutdown
    README.md, MANIFEST.json              steps, our results, sha256 of every file

Every run folder is checked as their evaluation checks it, the scenario is loaded with their loader,
and the SAC model is checked for their feature extractor and NumPy 1.x pickles. Check the bundle on
their physics with ``scripts/check_deployment.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEPLOY_SRC = ROOT / "src" / "stroke_nav" / "organiser" / "deploy"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", type=Path, required=True, help="results/cases/<id> from plan_case.py")
    ap.add_argument("--name", default=None, help="bundle name (default stroke-nav-<case id>)")
    ap.add_argument("--scenario-name", default=None, help="deployed scenario name (default <case scenario>-<target>)")
    ap.add_argument("--scenario-root", type=Path, default=ROOT / "data" / "scenarios")
    ap.add_argument("--slow-radius", type=float, default=8.0, help="controller slows inside this distance (mm), as plan_case.py")
    ap.add_argument("--out", type=Path, default=ROOT / "handover")
    args = ap.parse_args()

    from check_handover import inspect_model_zip

    from stroke_nav.organiser import ensure_policy_importable
    from stroke_nav.organiser.export import check_run_dir, write_run_dir
    from stroke_nav.organiser.frame import default_frame
    from stroke_nav.organiser.scenario import read_goals_cm, scenario_with_goals

    case = json.loads((args.case / "case.json").read_text(encoding="utf-8"))
    cid = case["case_id"]
    name = args.name or f"stroke-nav-{cid}"
    out = args.out / name
    if out.exists():
        shutil.rmtree(out)
    rl = out / "HCMR-EV-RL"
    frame = default_frame()

    # scenario: the case mesh with only the case goal
    src = args.scenario_root / case["scenario"]
    goals_cm = read_goals_cm(src)
    goal_cm = np.asarray(case["goal_world_m"]) / 0.01  # bake "authored": world = mesh * meters_per_unit
    idx = int(np.argmin(np.linalg.norm(goals_cm - goal_cm, axis=1)))
    if np.linalg.norm(goals_cm[idx] - goal_cm) * 10 > 1.0:
        raise SystemExit(f"case goal {goal_cm} cm is not among the goals of {src}")
    scen_name = args.scenario_name or f"{case['scenario']}-{case['target'].replace('@', '-')}"
    scen_dir = scenario_with_goals(src, rl / "scenarios", scen_name, [idx])
    manifest = json.loads((scen_dir / "scenario.json").read_text(encoding="utf-8"))
    print(f"[deploy] scenario {scen_name}: mesh {manifest['mesh_sha256'][:8]}..., goal {goals_cm[idx]} cm "
          f"({case['target']}), validated with their loader")

    common = {"case": {k: case[k] for k in ("case_id", "model", "variant", "target", "branch", "goal_world_m")},
              "deployed_scenario": scen_name}
    cands = {m: json.loads((args.case / "candidates" / f"{m}.json").read_text(encoding="utf-8"))
             for m in ("roll", "bc") if (args.case / "candidates" / f"{m}.json").is_file()}

    # behaviour-cloned SAC policy
    ensure_policy_importable()
    bc_src = args.case / "runs" / "bc"
    bc_run = rl / "runs" / f"stroke-nav-bc-{cid}"
    bc_prov = json.loads((bc_src / "stroke-nav.json").read_text(encoding="utf-8"))
    write_run_dir(bc_run, model_path=bc_src / "final-model.zip", frame=frame, scenario_manifest=manifest,
                  extra={**common, "trained_in": bc_prov.get("trained_in"), "bc": bc_prov.get("bc")})
    if (bc_src / "bc-training.json").is_file():
        shutil.copyfile(bc_src / "bc-training.json", bc_run / "bc-training.json")
    problems = check_run_dir(bc_run, manifest) + inspect_model_zip(bc_run / "final-model.zip")
    if problems:
        raise SystemExit(f"BC run folder would be rejected: {problems}")
    print(f"[deploy] {bc_run.name}: run folder and model OK")

    # A* route + calibrated entry roll
    roll_run = rl / "runs" / f"stroke-nav-astar-roll-{cid}"
    controller = {
        "schema": "stroke-nav/entry-roll-controller/1",
        "controller": "EntryRollExpert",
        "params": {"roll_deg": float(case["entry_roll_deg"]), "slow_radius_mm": float(args.slow_radius)},
        "description": "roll the guidewire to roll_deg at the entry, then push; slow inside slow_radius_mm of "
                       "the goal, hold inside stop_radius_mm; the catheter stays home",
        "route": case["route"], "reach": case["reach"], **common,
    }
    write_run_dir(roll_run, frame=frame, scenario_manifest=manifest,
                  extra={**common, "trained_in": "none: scripted controller; entry roll from the A* route and the "
                                                 "roll calibration on their physics (calibrate_rolls.py)"})
    (roll_run / "controller.json").write_text(json.dumps(controller, indent=2), encoding="utf-8")
    shutil.copyfile(ROOT / "src" / "stroke_nav" / "agents" / "entry_roll.py", roll_run / "entry_roll.py")
    shutil.copyfile(DEPLOY_SRC / "evaluate_controller.py", roll_run / "evaluate_controller.py")
    problems = check_run_dir(roll_run, manifest, model="controller.json")
    if problems:
        raise SystemExit(f"controller run folder would be rejected: {problems}")
    print(f"[deploy] {roll_run.name}: run folder OK, entry roll {controller['params']['roll_deg']:g} deg")

    script = (DEPLOY_SRC / "deploy.sh").read_text(encoding="utf-8")
    for key, value in (("@SCENARIO@", scen_name), ("@BC_RUN@", bc_run.name), ("@ROLL_RUN@", roll_run.name)):
        script = script.replace(key, value)
    (out / "deploy.sh").write_bytes(script.replace("\r\n", "\n").encode("utf-8"))
    (out / "README.md").write_text(readme(name, case, scen_name, bc_run.name, roll_run.name, controller, cands,
                                          (bc_prov.get("bc") or {}).get("episodes")),
                                   encoding="utf-8", newline="\n")

    files = {str(p.relative_to(out)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(out.rglob("*")) if p.is_file()}
    (out / "MANIFEST.json").write_text(json.dumps(files, indent=2), encoding="utf-8")
    archive = shutil.make_archive(str(out), "zip", root_dir=out)
    print(f"[deploy] wrote {out} ({len(files)} files) and {archive}")


def _result_line(cand: dict | None) -> str:
    if not cand:
        return "not evaluated"
    eps = cand["episodes"]
    ok = [e for e in eps if e["success"]]
    line = f"{len(ok)}/{len(eps)} successful"
    if ok:
        line += (f", {np.mean([e['steps'] for e in ok]) / 5:.1f} s to the goal, final distance "
                 f"{np.mean([e['final_dist_mm'] for e in ok]):.1f} mm, peak force {max(e['peak_force_n'] for e in ok) * 1000:.0f} mN")
    return line


def readme(name, case, scen, bc_run, roll_run, controller, cands, n_demos) -> str:
    r = case["route"]
    return f"""# {name}: deployment on the organiser's stack

Case `{case['case_id']}`: VMR `{case['model']}` stenosis variant `{case['variant']}`, target `{case['target']}`
(the ICA lesion in `{case['branch']}`), goal {case['goal_world_m']} m in their world frame.

| Run | What it is | Our result on their physics (lockstep) |
|---|---|---|
| `{roll_run}` | A* route + calibrated entry roll: roll the guidewire to {controller['params']['roll_deg']:g}° at the entry, then push | {_result_line(cands.get('roll'))} |
| `{bc_run}` | SAC policy (their `FluoroStateExtractor`) behaviour-cloned from {n_demos} demonstrations of that controller | {_result_line(cands.get('bc'))} |

A* route: {r['branches']}, {r['length_mm']} mm, {r['bifurcations']} bifurcations, minimum lumen radius
{r['min_lumen_radius_mm']} mm. The route is longer than the 264.5 mm guidewire travel, but their rod cuts
corners: the calibrated entry roll ends within {case['reach']['roll_table']['recorded_paths_miss_mm']} mm of the goal.

## Contents

| Path | Use |
|---|---|
| `HCMR-EV-RL/scenarios/{scen}/` | their scenario format (`vessel.obj` in cm, `goals.txt`, `scenario.json`): the variant mesh with only this goal |
| `HCMR-EV-RL/runs/{bc_run}/` | run folder for their showcase: `final-model.zip` (SB3 2.9.0 SAC, NumPy 1.x pickles), `resolved-config.json`, `device-parameters.json` (hash `38b1ea53…`, as their simulator), `scenario.json` (= the scenario manifest) |
| `HCMR-EV-RL/runs/{roll_run}/` | the same run files with `controller.json`, `entry_roll.py` and `evaluate_controller.py` in place of a SAC model |
| `deploy.sh` | install and run both on their stack |
| `MANIFEST.json` | sha256 of every file |

## Deploy (Linux with Docker, from their workspace)

```bash
unzip {name}.zip -d {name}                          # next to HCMR-EV-OpenNG3
cd HCMR-EV-OpenNG3                                  # the folder with ./sys
bash ../{name}/deploy.sh install                    # copies the scenario and both runs into HCMR-EV-RL
bash ../{name}/deploy.sh astar-roll 5 --visible     # A* + entry roll, 5 episodes, physics page visible
bash ../{name}/deploy.sh results                    # when `docker logs -f hcmr-online-rl` shows it finished
bash ../{name}/deploy.sh bc 5 --visible             # the behaviour-cloned policy, same scenario
bash ../{name}/deploy.sh results
bash ../{name}/deploy.sh shutdown                   # ./sys shutdown, finalises the rollout dataset
```

Set `HCMR_OPENNG3_PATH` / `HCMR_RL_REPO_PATH` if the repositories are elsewhere (defaults as in `./sys`).

- **`bc`** is their showcase, unchanged: `./sys startup --mode showcase --scenario {scen} --run-id {bc_run} --model final-model.zip --episodes N [--visible]`.
- **`astar-roll`** runs the same showcase startup for `{roll_run}` (renderer with this scenario and the run's
  device parameters, robot, bridge, reset coordinator, rollout recorder). Their `openng3-rl-evaluate` can only
  load SAC models, so `deploy.sh` stops it while it waits for stream readiness, before it acts, and starts
  `evaluate_controller.py` in the same `hcmr/openng3-rl` image. That is their `openng3_rl.evaluation.main`
  with `SAC.load` replaced by the controller, so the run-folder, scenario and renderer checks, `OpenNG3Env`,
  success rule, rollout capture and `evaluations/<id>/episodes.json` are theirs. The stopped SAC evaluator
  leaves one evaluation folder with state `stopped`.
- Results: `HCMR-EV-RL/runs/<run>/evaluations/<id>/{{status,episodes}}.json`; rollouts in their dataset viewer
  (http://localhost:8090).

What to expect: the guidewire turns to about {controller['params']['roll_deg']:g}° at the entry, then pushes up the
left common carotid into the ICA to the lesion in about 27 s of simulated time; the catheter stays home.

**Not yet run on their wall-clock stack.** Our results come from their physics page, observation, reward and
success rule in lockstep (the robot's 50 ms delay and their noise node emulated from their sources). On a
machine below their 4.5 Hz image rate, episodes end as `aborted` (stale packets): that is the machine, not the
policy. Failures without aborts would be a real lockstep-to-wall-clock gap.
"""


if __name__ == "__main__":
    main()
