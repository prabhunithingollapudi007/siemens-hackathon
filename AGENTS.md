# AGENTS.md

Guidance for coding agents and new contributors working in this repository. For the project overview
and user-facing commands, see [README.md](README.md). For the organiser's system, see
[docs/ORGANISER_SETUP.md](docs/ORGANISER_SETUP.md); for handing work over to them, see
[docs/HANDOVER.md](docs/HANDOVER.md).

## What this repo is

`stroke-nav` is a playground for guidewire and microcatheter navigation toward a stroke target, on
patient vessel models from the Vascular Model Repository (VMR). It contains:

- a fast kinematic simulator with a Gymnasium environment
- a device-aware path planner and trajectory generator
- START evaluation metrics
- a layer that reproduces the hackathon organiser's RL interface, so that what we train loads in their stack

## Layout

| Path | Contents |
|---|---|
| `src/stroke_nav/data/` | VMR download (`fetch.py`, HTTP range requests), centerline graph and lumen radius (`centerline.py`), queries and target specs (`anatomy.py`) |
| `src/stroke_nav/env/` | kinematic two-device simulator (`devices.py`) and `VesselNavEnv` (`vessel_env.py`) |
| `src/stroke_nav/planning/` | device limits and curvature (`constraints.py`), constrained A\* (`planner.py`), speed profile, roll schedule and joint trajectory (`trajectory.py`) |
| `src/stroke_nav/agents/` | `ScriptedOracle`, `RandomAgent`, `TrajectoryFollower` |
| `src/stroke_nav/eval/` | START metrics and tables |
| `src/stroke_nav/viz/` | pyvista scene for the playground and renders |
| `src/stroke_nav/organiser/` | organiser compatibility: spec, frame transform, fluoro image, `OrganiserCompatEnv`, run-folder export, ROS 2 templates |
| `compat/openng3_rl/` | stand-in for the organiser's `openng3_rl.policy.FluoroStateExtractor` (see invariants) |
| `scripts/` | command-line entry points (fetch, plot, playground, plan, evaluate, compare, train, check, export) |
| `tests/` | pytest suite, mostly on synthetic vessel trees |
| `docs/` | organiser reference and hand-over guide |

Git ignores `data/`, `runs/`, `results/`, `handover/`, `Hackathon.zip` and the organiser artefact
folders. Never commit downloaded data, trained models, recordings or renders.

## Setup (Windows, Python 3.12)

```bash
py -3.12 -m venv --system-site-packages .venv
.venv\Scripts\pip install -e .[viz,dev]
.venv\Scripts\pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130   # GPU training (optional)
.venv\Scripts\python.exe scripts/fetch_data.py --model 0248_H_AOCERE_CAS                     # ~14 MB, builds data/processed/
```

- **Use the venv interpreter** (`.venv\Scripts\python.exe`). The system `python` is 3.14 and has no ML stack.
- **Versions are pinned to the organiser's:** NumPy 1.26.4, Gymnasium 1.4.0, Stable-Baselines3 2.9.0, PyTorch 2.14.1. Do not upgrade NumPy to 2.x (see invariants).
- **Disk space is tight on this machine.** Pass `--no-cache-dir` for large wheels.

## Commands

```bash
.venv\Scripts\python.exe -m pytest -q                                   # full suite (46 tests), about 20 s
.venv\Scripts\python.exe scripts/playground.py --model 0248_H_AOCERE_CAS   # interactive 3D window
.venv\Scripts\python.exe scripts/plan_path.py --target right_carotid@-5 --follow 20 --record
.venv\Scripts\python.exe scripts/evaluate.py --policies oracle,random,forward --episodes 50
.venv\Scripts\python.exe scripts/train_sac.py --total-timesteps 50000 --device cuda
.venv\Scripts\python.exe scripts/check_handover.py --run-dir runs/<run_id>
.venv\Scripts\python.exe scripts/export_handover.py --run-dir runs/<run_id> --plan-target right_carotid_external@21
```

- **Off-screen rendering** works with pyvista (`--record`). Interactive windows need a desktop session.
- **Long jobs:** run training in the background and log to `runs/<run_id>.log`.

## Conventions

- **Units:** our anatomy, simulator and planner use **mm** and **radians** internally. Degrees appear only in parameters, displays and exports.
- **Frames:** VMR models are in RAS-like axes; `fetch`/`build` multiply SimVascular cm by 10. The organiser world frame is in **metres**. Convert only through `stroke_nav.organiser.frame.OrganiserFrame`; never hand-roll the transform.
- **Device indices:** `WIRE = 0`, `CATH = 1`. The action order everywhere is `[v_guidewire, w_guidewire, v_catheter, w_catheter]` in [−1, 1].
- **Target specs** (`branch@mm`, `branch@-mm`, `random:...`, `xyz:...`, node ids) are parsed in `data/anatomy.py`. Reuse them rather than adding new formats.
- **Downloads are untrusted data.** They are written only under `data/raw/` and are never executed or imported.
- **Style:** small modules, type hints, docstrings that state units. No new heavy dependencies without a reason.
- **Tests:** add tests on the synthetic trees in `tests/conftest.py` so they run without downloads. Tests that need real data or organiser files must `skipif` when those are absent.

## Invariants: do not break the organiser hand-over

`tests/test_organiser.py` guards these. Keep it passing.

1. **Observation and action spaces of `OrganiserCompatEnv` are exactly the organiser's:** `Dict(image=(1,128,128) uint8, state=(43,) float32 in ±10)` and `Box(-1,1,(4,))`.
   - The state layout and units must follow `openng3_rl.observation.ObservationBuilder`.
   - That includes their quirk: the previous action in an observation is from step t−1.
2. **SAC models must reference `openng3_rl.policy.FluoroStateExtractor`.** Call `stroke_nav.organiser.ensure_policy_importable()` before building or loading models.
   - Do not rename `compat/openng3_rl` or change the network layers.
3. **Models must be pickled under NumPy 1.x.** NumPy 2 pickles contain `numpy._core`, which the organiser's container (NumPy 1.26.4) cannot load. `check_handover.py` detects this.
4. **The device parameters in `organiser/spec.py` must hash to the organiser run's value** (`38b1ea53…`). Change them only by loading the organiser's newer files with `spec.load_from_repo(...)`.
5. **`scenario.json` written to run folders must equal their `cas-reference` manifest** (mesh sha256 `04fbd581…`).
6. **The frame transform is fitted, not assumed.** Their goal file's comment claims an axis flip that the geometry does not have. Refit with `organiser.frame.fit_frame` if the mesh changes.

## Known issues

- **Playground timers on Windows:** a pyvista/VTK timer created before the window exists never fires. `scripts/playground.py` therefore opens the window first (`pl.show(interactive_update=True, auto_close=False)`), then adds the timer, then calls `pl.iren.start()`. Keep that order.
- **Interactive rendering is slow** (around 1 frame/s while GPU training runs). The autopilot steps the simulation at real-time pace and draws once per frame, and `Scene.update` updates actors in place. Don't go back to recreating actors every frame.
- **The simulator is kinematic.** Devices ride the centerlines, and roll picks the branch at bifurcations. There are no wall forces, so contact force reports n/a and path-following error is 0. Policies trained here need fine-tuning in the organiser's rod-physics simulator.
- **Junction gaps:** some VMR models (e.g. `0024_H_AO_H`) attach branches through long straight junction edges, because a branch centerline starts inside its parent. Code that measures distance to a planned path must densify it first (see `agents/follower.py`).

## Before you finish a change

- `pytest -q` passes.
- If you touched the organiser layer, `scripts/check_handover.py` passes on a fresh `train_sac.py --total-timesteps 600 --learning-starts 200 --buffer-size 2000` run.
- README, docs and this file are updated when commands, invariants or known issues change.
- Commit messages are imperative and explain why. Commit only source and docs, never data or runs.
