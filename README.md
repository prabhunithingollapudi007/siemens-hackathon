# stroke-nav

A playground for training controllers that steer a **guidewire + microcatheter** through
patient-specific vessel models toward a stroke target, along the mechanical-thrombectomy access route
(aorta, aortic arch, carotid, ICA, then M1 or A1). It contains:

- a fast kinematic two-device simulator with a Gymnasium environment, on any
  [Vascular Model Repository](https://www.vascularmodel.com/dataset.html) (VMR) model with centerlines
- a device-aware path planner and robot trajectory generator
- START consensus metrics (Robertshaw et al., JAHA 2026) and a runtime safety monitor
- a layer that reproduces the hackathon organiser's RL interface, plus a faster-than-real-time
  ("lockstep") env on the organiser's own physics, so what we train loads and runs in their stack

The robot has 4 action parameters, `[v_guidewire, w_guidewire, v_catheter, w_catheter]` in [−1, 1].
The default anatomy is `0248_H_AOCERE_CAS`, a CT model from the aortic root to the circle of Willis
in 30 carotid-stenosis variants.

## Results

| What | Where | Result |
|---|---|---|
| Behaviour-cloned SAC policy (organiser format) | organiser's XPBD physics, lockstep | **8/8** to their cas-reference goal, 30.4 s, 4.0 mm, peak force 0.009 N |
| Scripted expert `EntryRollExpert` (roll 120° at the entry, then push) | organiser's physics, lockstep | 60/60 demos, including with action noise |
| Constrained A\* + trajectory follower, catheter to right M1 past an 85 % ICA stenosis | our simulator, 0248 | 20/20 in 26.5 s (plain shortest path: 0/20) |
| Safety monitor on an imperfect policy | synthetic arch, stressed tracking | success 60 → 100 %, target overshoot 40 → 0 % |

A video of the cloned policy on the organiser's physics page is written to `results/demo_bc_policy.mp4`
by `scripts/demo_lockstep.py` (`results/` is not in git).

## Setup (Windows, Python 3.12)

```bash
py -3.12 -m venv --system-site-packages .venv
.venv\Scripts\activate
pip install -e .[viz,dev]
python scripts/fetch_data.py --model 0248_H_AOCERE_CAS      # ~14 MB, builds data/processed/
```

Versions are pinned to the organiser's toolchain so trained models load on their side: NumPy 1.26.4,
Gymnasium 1.4.0, Stable-Baselines3 2.9.0, PyTorch 2.14.1. For GPU training install a CUDA build of
PyTorch 2.14.1 (the organiser uses cu130). All commands below run in the activated venv.

## 1. Anatomy

```bash
python scripts/fetch_data.py --list --anatomy cerebral          # models that have centerlines
python scripts/fetch_data.py --model 0248_H_AOCERE_CAS --list-variants
python scripts/fetch_data.py --model 0248_H_AOCERE_CAS --variant 10Left100Right
python scripts/plot_anatomy.py --model 0248_H_AOCERE_CAS        # results/<model>_anatomy.png
```

Only the centerlines, the surface model and the license are pulled, with HTTP range requests (the
717 MB zip is never downloaded). `data/processed/<model>_<variant>.npz` then holds the centerline graph
(1 mm nodes) and a lumen radius per node, measured from wall-to-wall rays so it stays accurate where
the centerline runs off-centre. Tails outside the surface are trimmed, and gaps of 3 mm or more in a
large vessel are marked as occlusions (the 100 % stenosis variants).

## 2. Drive and pick targets

```bash
python scripts/playground.py --model 0248_H_AOCERE_CAS
python scripts/playground.py --model 0248_H_AOCERE_CAS --target right_anterior@20 --record demo.gif
```

| Key | Action |
|---|---|
| Up / Down, Left / Right | guidewire advance / retract, rotate |
| i / k, n / m | microcatheter advance / retract, rotate |
| b | advance both |
| p | pick the point under the mouse as the target (g: set the start instead) |
| o / space | one oracle step / oracle autopilot |
| l / h | plan a device-feasible path / follow it (robot autopilot) |
| x, c, z | reset, toggle centerlines, print START metrics |

**Target specs** work everywhere (`--target`, `EnvConfig.target`, `reset(options={"target": ...})`):

| Spec | Meaning |
|---|---|
| `left_carotid@120`, `left_carotid@-5` | 120 mm along a branch; 5 mm from its distal end |
| `random:left_carotid@-15--1\|right_carotid@-15--1` | a new random node per episode (here M1) |
| `xyz:-120,110,-200`, `1234` | nearest node to a point (mm); a node id |
| `file:data/targets/0248_H_AOCERE_CAS.json` | a point picked in the playground |

For 0248 the carotid paths run CCA, ICA, then M1; `*_anterior` is A1 and `*_carotid_external` the ECA.

## 3. Plan a path and a robot trajectory

```bash
python scripts/plan_path.py --target right_carotid@40 --follow 20 --record
python scripts/plan_path.py --target right_carotid@40 --forbid anterior_connector,left_communicating,right_communicating
python scripts/plan_path.py --target left_anterior@20 --device guidewire --hard-curvature
python scripts/compare_routes.py          # GIF: shortest path vs device-aware plan around an occluded ICA
```

`src/stroke_nav/planning/` has four parts:

1. **Device limits** (`constraints.py`): outer diameter, minimum bend radius and clearance; a device
   enters only lumen at least 1.2× its diameter. The organiser's devices (`radifocus_035`,
   `judkins_5f`) come from `device-parameters.json`; generic presets are 0.014" wire, 1.7F
   microcatheter and 6F guide.
2. **Constrained A\*** (`planner.py`) over directed edges. Hard limits: clearance, optionally bend
   radius and forbidden vessels. Soft costs: length, narrow lumen, bending, vessel changes, sharp
   take-off angles. Plain Dijkstra is reported as the baseline.
3. **Trajectory** (`trajectory.py`): a speed profile limited by curvature, lumen and bifurcations; the
   guidewire roll for each bifurcation from the simulator's tip model; a 5 Hz joint-space reference
   with the catheter trailing the wire.
4. **Follower** (`agents/follower.py`): tracks the reference from encoders and the observed tip, and
   backs up and retries if the tip leaves the path.

Each run writes `results/plans/<name>/`: `report.md`, `profiles.png`, `path3d.png`, `trajectory.csv`
(`t_s, L_guidewire_mm, roll_guidewire_deg, L_microcatheter_mm, roll_microcatheter_deg`), `plan.json`
and, with `--record`, `follow.gif`.

```python
from stroke_nav.data.anatomy import Anatomy
from stroke_nav.planning.planner import PathPlanner
from stroke_nav.planning.trajectory import build_trajectory

anat = Anatomy.load("0248_H_AOCERE_CAS")
plan = PathPlanner(anat, "radifocus_035").plan(anat.default_start(), anat.resolve("right_carotid@40"))
build_trajectory(plan).to_csv("trajectory.csv")
```

On 0248 with an 85 % right ICA stenosis, the microcatheter cannot pass the stenosis (0/20 on the
shortest path); constrained A\* routes it up the left carotid and across the anterior communicating
connector to the right M1 (44 mm longer, 20/20). That detour is feasible but not a standard clinical
approach; `--forbid` excludes collaterals.

## 4. Evaluate

```bash
python scripts/evaluate.py --policies oracle,random,forward --episodes 100 --target "random:left_carotid@20-70|right_carotid@20-70"
python scripts/evaluate_safety.py --synthetic --episodes 20 --policies follower,sloppy,forward
```

`evaluate.py` writes `results/<name>/start_summary.md` and `episodes.csv` with the START metrics:
success (Wilson 95 % interval), procedure and fluoroscopy time, phases, wrong-branch failures,
handling errors, path length per device, tip speed, path-following error, contact force and path
ratio. Policies get the same episode seeds, so comparisons are paired; a trained SB3 policy plugs in
as `ppo:path/to/model.zip`. The safety monitor and fault injection are described in
[docs/SAFETY.md](docs/SAFETY.md).

## 5. Train for the organiser and hand over

The organiser's stack (OpenNG3, the EndoNav/CatForce browser physics and `openng3_rl` SAC) is described in
[docs/ORGANISER_SETUP.md](docs/ORGANISER_SETUP.md). There are two ways to train a policy in their format:

**On their physics (recommended).** `stroke_nav.organiser.live.lockstep` runs their physics page, noisy
viewer and `OpenNG3Env` headless and faster than real time, without Docker or ROS (their live stack
reaches only 0.18× real time on this laptop). A scripted expert is recorded there and cloned into
their SAC policy:

```bash
python scripts/record_demos.py --episodes 60 --sigmas 0,0.2,0.4 --workers 4 --max-steps 300   # expert demos
python scripts/train_bc.py --demos runs/demos/cas-reference --device cuda --run-id stroke-nav-bc-v1
python scripts/eval_lockstep.py --model runs/stroke-nav-bc-v1/final-model.zip --episodes 8 --workers 4
python scripts/demo_lockstep.py --record results/demo_bc_policy.mp4                          # watch and record
```

Where the policy's behaviour comes from:

1. **A measurement on their physics.** On their rod, the roll set at the entry decides which vessel
   the guidewire ends up in; 120° and a 260 mm push reach the right external carotid, where the
   cas-reference goal is (docs/ORGANISER_SETUP.md §12).
2. **A scripted expert.** `EntryRollExpert` (`src/stroke_nav/agents/entry_roll.py`) turns that into a
   controller: roll to 120°, push, slow down within 25 mm of the goal, stop there; the catheter stays
   at home. It reads only the policy's own 43-value noisy state, with no privileged information.
3. **Demonstrations.** `record_demos.py` runs the expert in the lockstep env and saves, per step, the
   fluoro image, the state and the expert's action (`runs/demos/cas-reference`, 60 episodes, all
   successful). In 40 of them Gaussian noise (σ 0.2 or 0.4) is added to the executed action while
   the label stays the expert's (DART), so the clone also sees how to recover from small mistakes.
4. **Behaviour cloning.** `train_bc.py` fits the actor of an ordinary SB3 SAC model with their
   `FluoroStateExtractor` to the expert labels (MSE; 40 epochs on 8,587 samples, about 30 s). It
   warm-starts the critic from the Monte-Carlo returns so SAC fine-tuning can follow; the model loads
   in their stack unchanged.

There is no reward-based learning and no A\* in this pipeline. Their scenario has a single goal, so
the policy has only learned the cas-reference target.

To watch the policy, stop their Docker stack first; it is not used, and the pages need the CPU. Two
Chrome windows open: their physics page and their noisy fluoro viewer (the policy's camera input).

```bash
powershell -ExecutionPolicy Bypass -File tools\organiser_stack.ps1 stop
python scripts/demo_lockstep.py --record results/demo_bc_policy.mp4
powershell -ExecutionPolicy Bypass -File tools\organiser_stack.ps1 start -Visible   # their stack again afterwards
```

`demo_lockstep.py` options:
- `--headless` records without showing the windows.
- `--expert` runs the scripted teacher instead of the policy.
- `--seed N` changes the noise seed.
- `--fps 5` records at real time (the default of 10 is 2×).
- `--record ""` skips the video.

**In our simulator.** `src/stroke_nav/organiser/` reproduces their action and observation spaces,
reward, success rule, noise, device parameters, scenario and SAC settings on our kinematic simulator,
with a fitted transform to their world frame (median 0.45 mm). Policies trained this way are a warm
start that needs fine-tuning on their physics. `python scripts/train_sac.py --total-timesteps 50000`
writes an organiser-format run folder; for long runs see [docs/HPC.md](docs/HPC.md) (FAU TinyGPU).

**Hand over.** Check and package a run, then run their own evaluation on their machine:

```bash
python scripts/check_handover.py --run-dir runs/stroke-nav-bc-v1
python scripts/export_handover.py --run-dir runs/stroke-nav-bc-v1 --plan-target right_carotid_external@21
# on their Linux machine, after copying runs/stroke-nav-bc-v1 into HCMR-EV-RL/runs/, from HCMR-EV-OpenNG3:
./sys startup --mode showcase --scenario cas-reference --run-id stroke-nav-bc-v1 --model final-model.zip --episodes 10 --visible
```

[docs/HANDOVER.md](docs/HANDOVER.md) lists what matches their stack and what is approximate, what the
bundle contains, and how to read their results (including what `aborted` episodes mean). Their own
stack also runs on this laptop with `powershell -ExecutionPolicy Bypass -File tools\organiser_stack.ps1 start`
(docs/ORGANISER_SETUP.md §11), but not fast enough for their RL env.

## Environment reference (`stroke_nav.env.vessel_env.VesselNavEnv`)

- **Devices:** `device-parameters.json` is the source of truth for both devices (tip angle and length,
  diameters) and the organiser's robot (5 Hz, 10 mm/s, 1 rad/s, guidewire travel 264.5 mm, catheter at
  most 229.2 mm ahead of the wire). `STROKE_NAV_DEVICE_PARAMS=<file>` selects another file. Targets past
  the travel cannot be reached (for 0248 the distal ICA and M1 lie 450-530 mm along the vessels).
- **Action:** `Box(-1, 1, (4,))`, scaled by the speed limits above.
- **Observation** (46 floats): 3 tracking points per device tip, now and at the previous step, the
  target, target minus tip, and the previous action (as in Robertshaw et al. 2024, 2025).
- **Reward:** `-0.005 - 0.01 * Δ(geodesic distance, mm) + 1.0 on success` (the papers use 0.001 for
  progress; set `progress_coef`).
- **Success:** the microcatheter tip (or guidewire, via `success_device`) within 5 mm of the target
  along the centerlines.
- **Device model:** both devices share one track through the centerline graph; the leading device's
  pre-curved tip picks the vessel at each bifurcation from its roll, so pushing alone never reaches the
  carotids. It is kinematic: no wall forces, so blocked pushes stand in for contact force. A device
  cannot enter lumen narrower than its clearance, so the catheter stops at a stenosis the wire crosses.

## Baselines and tests

- `ScriptedOracle` (privileged route and branch rule) reaches 100 % of targets in every tested region
  of 0248 and 0050; `RandomAgent` is the lower bound.
- `TrajectoryFollower` reaches 100 % of random CCA, ICA, M1, A1 and vertebral targets on the stenosis
  and occlusion variants, at about 1.5× the oracle's time.
- The pipeline also runs unchanged on a second patient, `0024_H_AO_H` (shared carotid origin): the
  follower and oracle reach all four branch ends, against 0-5 % for random actions
  (`python scripts/plan_path.py --model 0024_H_AO_H --target LCCA@-2 --follow 20 --record`).
- `python -m pytest -q` runs 79 tests, mostly on synthetic vessel trees, so no downloads are needed:
  env API, oracle and planner behaviour, trajectory limits, START metrics, safety monitor, the
  organiser interface (spaces, state layout, reward, run folders, model pickling) and the expert.

Repository layout, conventions and the hand-over invariants that tests guard are in
[AGENTS.md](AGENTS.md).

## Data license and acknowledgement

VMR data is free for research and development. Keep each model's `LICENSE` file with any copy of the data.
Publications should acknowledge:

> The data used herein was provided in whole or in part with Federal funds from the National Library of
> Medicine under Grant No. R01LM013120, and the National Heart, Lung, and Blood Institute, National Institutes
> of Health, Department of Health and Human Services, under Contract No. HHSN268201100035C.

Also cite N.M. Wilson, A.K. Ortiz, A.B. Johnson, "The Vascular Model Repository: A Public Resource of Medical
Imaging Data and Blood Flow Simulation Results," J. Med. Devices 7(4), 040923 (2013), doi:10.1115/1.4025983.

## References

- Robertshaw H. et al. Autonomous navigation of catheters and guidewires in mechanical thrombectomy using inverse reinforcement learning. IJCARS 19, 1569–1578 (2024).
- Robertshaw H. et al. Reinforcement learning for safe autonomous two-device navigation of cerebral vessels in mechanical thrombectomy. IJCARS 20, 1077–1086 (2025).
- Robertshaw H. et al. (START). A position statement on endovascular models and effectiveness metrics for mechanical thrombectomy navigation. JAHA 15(7), e044931 (2026).
