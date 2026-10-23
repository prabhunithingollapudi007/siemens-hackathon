# stroke-nav

A playground for training controllers that steer a **guidewire + microcatheter** through
patient-specific vessel models toward a stroke target. The setting is the mechanical-thrombectomy
access route: aorta, aortic arch, carotid, ICA, then M1 or A1.

- **Anatomy:** any model from the [Vascular Model Repository](https://www.vascularmodel.com/dataset.html) (VMR) that has centerline paths.
  The default is `0248_H_AOCERE_CAS`, a CT model from the aortic root to the circle of Willis that
  comes in 30 carotid-stenosis variants.
- **Robot:** 4 learnable action parameters, `[v_guidewire, w_guidewire, v_microcatheter, w_microcatheter]`.
- **Evaluation:** START consensus metrics (Robertshaw et al., JAHA 2026).

## Setup

```bash
py -3.12 -m venv --system-site-packages .venv
.venv\Scripts\pip install -e .[viz,dev]
```

The dependencies are pinned to the organiser's toolchain so trained models load on their side:
NumPy 1.26.4, Gymnasium 1.4.0, Stable-Baselines3 2.9.0 and PyTorch 2.14.1 on Python 3.12.
For GPU training, install a CUDA build of PyTorch 2.14.1 (the organiser uses cu130).

## 1. Download a model

```bash
python scripts/fetch_data.py --list --anatomy cerebral        # browse models that have centerlines
python scripts/fetch_data.py --model 0248_H_AOCERE_CAS         # ~14 MB of a 717 MB zip
python scripts/fetch_data.py --model 0248_H_AOCERE_CAS --list-variants
python scripts/fetch_data.py --model 0248_H_AOCERE_CAS --variant 10Left100Right
python scripts/fetch_data.py --model 0050_H_CERE_H
```

Only the centerlines (`Paths/*.pth`), the surface model (`Models/*.vtp`) and the license are pulled.
The script uses HTTP range requests, so the full zip is never downloaded. The script then builds
`data/processed/<model>_<variant>.npz`, which holds the centerline graph (1 mm nodes, junctions,
loops) and the lumen radius per node:

- The radius is half the median wall-to-wall chord across the vessel, from rays cast perpendicular to the centerline.
  This stays accurate where the centerline runs off-centre, such as the carotid siphon.
- Centerline tails that run outside the surface model are trimmed.
- A gap of 3 mm or more inside a large vessel (median radius 1.5 mm or more) is marked as an occlusion, with radius 0.
  The 100% stenosis variants of 0248 show up this way.

```bash
python scripts/plot_anatomy.py --model 0248_H_AOCERE_CAS      # results/<model>_anatomy.png
```

## 2. Pick a target and drive

```bash
python scripts/playground.py --model 0248_H_AOCERE_CAS
python scripts/playground.py --model 0248_H_AOCERE_CAS --target right_anterior@20 --record demo.gif
```

| Key | Action |
|---|---|
| Up / Down | guidewire advance / retract |
| Left / Right | guidewire rotate |
| i / k | microcatheter advance / retract |
| n / m | microcatheter rotate |
| b | advance both |
| p | pick the point under the mouse as the target (press g to switch to setting the start) |
| o / space | one oracle step / oracle autopilot on-off |
| l | plan a device-feasible path to the target and show it, coloured by planned speed |
| h | follow the planned trajectory (robot autopilot) on-off |
| x | reset episode |
| c | toggle centerlines |
| z | print START metrics |

Picked points snap to the nearest centerline node and are saved to `data/targets/<model>.json`.

**Target specs.** The same format works everywhere (`--target`, `EnvConfig.target`, `reset(options={"target": ...})`):

| Spec | Meaning |
|---|---|
| `left_carotid@120` | 120 mm along a branch |
| `left_carotid@-5` | 5 mm from the branch's distal end |
| `random:left_carotid@-15--1\|right_carotid@-15--1` | a new random node per episode (here M1) |
| `xyz:-120,110,-200` | nearest node to a point (mm) |
| `1234` | a node id |
| `file:data/targets/0248_H_AOCERE_CAS.json` | a saved pick |

For 0248, the carotid paths run CCA, ICA, then M1; `*_anterior` is A1; `*_carotid_external` is the ECA.

## 3. Evaluate with START metrics

```bash
python scripts/evaluate.py --policies oracle,random,forward --episodes 100 \
    --target "random:left_carotid@20-70|right_carotid@20-70"
```

The script writes `results/<name>/start_summary.md` and `episodes.csv`. The reported metrics are:

- success rate with a Wilson 95% interval
- procedure time and fluoroscopy time
- number of steps
- phases (route bifurcations passed)
- wrong-branch failures
- handling errors (retractions plus wall pushes)
- path length per device
- tip speed
- path-following error
- contact force
- path ratio

Policies get identical episode seeds, so the comparison is paired. A trained Stable-Baselines3
policy plugs in as `ppo:path/to/model.zip`.

## 4. Plan a feasible path and a robot trajectory

```bash
python scripts/plan_path.py --target right_carotid@40 --follow 20 --record
python scripts/plan_path.py --variant 10Left100Right --target right_carotid@40 --follow 20
python scripts/plan_path.py --target right_carotid@40 --forbid anterior_connector,left_communicating,right_communicating,right_posterior
python scripts/plan_path.py --target left_anterior@20 --device guidewire --hard-curvature
python scripts/compare_routes.py        # side-by-side GIF: shortest path vs device-aware plan (occluded right ICA)
```

The pipeline has four parts (`src/stroke_nav/planning/`):

1. **Device limits** (`constraints.py`). Each device has an outer diameter, a minimum bend radius and a clearance factor.
   Presets: 0.014" guidewire (0.36 mm), 1.7F microcatheter (0.57 mm) and 6F guide catheter (2.0 mm).
   A device may only enter lumen with diameter at least 1.2 times its outer diameter.
2. **Constrained A\*** (`planner.py`). The search runs over directed edges, so each step knows the incoming direction.
   - Hard limits: lumen clearance, and optionally bend radius (`--hard-curvature`) and forbidden vessels (`--forbid`).
   - Soft costs: length, narrow lumen, bending effort (curvature relative to the device limit), each vessel change, and sharp bifurcation take-off angles.
   - Plain Dijkstra is reported next to it as the baseline.
3. **Trajectory** (`trajectory.py`).
   - The speed profile is limited by curvature, narrow lumen and a slow zone before each bifurcation, then by acceleration.
   - For each bifurcation, the guidewire roll that makes the pre-curved tip enter the planned vessel is computed with the simulator's own tip model.
   - The output is a joint-space reference at the control rate (5 Hz): inserted length and roll for the guidewire and the catheter.
     The wire pauses before a junction until it has rotated, and the catheter trails 10 mm behind, then advances to the target.
4. **Follower** (`agents/follower.py`). The follower tracks the reference using only joint encoders and the observed tip position.
   If the tip leaves the planned path, it backs up past the last bifurcation and retries.

Each run writes `results/plans/<name>/` with these files:

| File | Contents |
|---|---|
| `report.md` | route comparison, bifurcation and roll table, trajectory summary, START metrics |
| `profiles.png` | radius, bend radius and speed along the path; joint references over time |
| `path3d.png` | 3D view: planned path coloured by speed, plain shortest path in grey |
| `trajectory.csv` | joint-space reference: `t_s, L_guidewire_mm, roll_guidewire_deg, L_microcatheter_mm, roll_microcatheter_deg` |
| `plan.json` | path nodes with geometry and speed, plus roll events |
| `follow.gif` | the follower executing the plan (with `--record`) |

In code:

```python
from stroke_nav.data.anatomy import Anatomy
from stroke_nav.planning.planner import PathPlanner
from stroke_nav.planning.trajectory import build_trajectory

anat = Anatomy.load("0248_H_AOCERE_CAS")
plan = PathPlanner(anat, "radifocus_035").plan(anat.default_start(), anat.resolve("right_carotid@40"))
traj = build_trajectory(plan)          # traj.t, traj.L_wire, traj.phi_wire, traj.L_cath, traj.phi_cath
traj.to_csv("trajectory.csv")
```

Example results on 0248, reference variant with 85% right ICA stenosis, target right M1. Each simulator row is 20 episodes.

| Route | Result |
|---|---|
| Guidewire plan | Straight up the right ICA. The 0.26 mm narrowest lumen clears the 0.22 mm the wire needs. |
| Microcatheter, plain Dijkstra | Infeasible, because the stenosis is narrower than the 0.34 mm the catheter needs. In the simulator it gets stuck at the stenosis: 0/20. |
| Microcatheter, constrained A\* | Up the left carotid, across the anterior communicating connector, then to the right M1. 44 mm longer. Executed 20/20 in 26.5 s. |
| Same plan, collaterals forbidden | No feasible path. In that case only the wire can cross; the catheter cannot. |

The detour through the circle of Willis is geometrically feasible, but it is not a standard clinical approach.
Use `--forbid` to exclude collateral vessels.

## 5. Hand over to the organiser

The organiser's stack (OpenNG3 plus the EndoNav/CatForce simulator and `openng3_rl` SAC) is documented
in [docs/ORGANISER_SETUP.md](docs/ORGANISER_SETUP.md). `src/stroke_nav/organiser/` reproduces their RL
interface on top of our simulator. That covers their action and observation spaces, reward, success
rule, noise, device parameters, scenario and SAC settings. It also contains the fitted transform
between our VMR anatomy and their world frame (median 0.45 mm). What we train can therefore be
dropped into their `runs/` folder:

```bash
.venv\Scripts\python.exe scripts/train_sac.py --total-timesteps 50000              # organiser-format run folder
.venv\Scripts\python.exe scripts/check_handover.py --run-dir runs/<run_id>          # validates like their evaluation
.venv\Scripts\python.exe scripts/export_handover.py --run-dir runs/<run_id> --plan-target right_carotid_external@21
```

The organiser's own stack also runs on this laptop: `powershell -ExecutionPolicy Bypass -File tools\organiser_stack.ps1 start`.
See [docs/ORGANISER_SETUP.md](docs/ORGANISER_SETUP.md), section 11.

**Policy trained on their physics.** `stroke_nav.organiser.live.lockstep` runs their physics page and
their env faster than real time, without Docker or ROS. A scripted expert (`EntryRollExpert`: roll
the wire to 120° at the entry, then push) reaches their goal in every episode. A SAC policy
behaviour-cloned from it reaches it too: 8 of 8 on unseen seeds, in 30 s. See section 14 of
[docs/ORGANISER_SETUP.md](docs/ORGANISER_SETUP.md).

```bash
.venv\Scripts\python.exe scripts/record_demos.py --episodes 60 --sigmas 0,0.2,0.4 --workers 4 --max-steps 300
.venv\Scripts\python.exe scripts/train_bc.py --demos runs/demos/cas-reference --device cuda
.venv\Scripts\python.exe scripts/eval_lockstep.py --model runs/<run_id>/final-model.zip --episodes 8 --workers 4
```

[docs/HANDOVER.md](docs/HANDOVER.md) lists what is identical, what is approximate and what is missing,
and gives the exact commands to run on their side. The organiser's own 210k-step checkpoint loads in
our compatible environment with identical spaces.

## 6. Train on the FAU TinyGPU cluster

[TinyGPU](https://doc.nhr.fau.de/clusters/tinygpu/) is FAU's Tier3 GPU cluster (RTX 2080 Ti, RTX 3080, V100 and
A100 nodes, batch jobs up to 24 h). Measured on a 2080 Ti node, SAC runs at about 26 steps/s once learning
starts, against 14-20 on a laptop GPU, so the organiser's 1M-step run takes roughly 11-20 h and the job script
defaults to the 24 h limit. The bigger win is running several seeds or settings in parallel, one GPU each. Scripts are in `hpc/`; they follow the NHR@FAU "HPC in a Nutshell" and "AI on HPC in a
Nutshell" guidance (Oct 2026).

**Access.** Login is by SSH key only (upload it in the [HPC portal](https://portal.hpc.fau.de/)). The frontend
`tinyx.nhr.fau.de` is reachable directly only from the FAU network or VPN; from elsewhere go through the dialog
server `csnhr.nhr.fau.de` by adding `ProxyJump csnhr.nhr.fau.de` to the host entry. A minimal `~/.ssh/config`:

```
Host tinyx.nhr.fau.de
    User <hpc-account>
    IdentityFile ~/.ssh/<key-file>
    # ProxyJump csnhr.nhr.fau.de    # only when outside the FAU network and VPN
```

Run the copy commands below **on your own machine**, not in a shell already logged in on the cluster; the
cluster has no private key, so an `scp`/`rsync` from there back to itself is refused.

**Storage.** `$HOME` is 50 GB with backups, `$WORK` is 1 TB without. The repo, the venv and the finished run folders
live in `$WORK`. Inside a job, TensorBoard logs and checkpoints are written to the node-local SSD `$TMPDIR` and
synced back to `runs/` every 15 min, at the end and on the time-limit signal, so the NFS servers are not hit with
small frequent writes.

```bash
# 1. on your machine: copy only code and data (~10 MB plus anatomy; the full checkout is GBs).
#    scp cannot expand $WORK, so look it up once and use the real path:
ssh tinyx.nhr.fau.de 'mkdir -p $WORK/stroke-nav && echo $WORK'          # e.g. /home/woody/iwi5/<account>
scp -r src scripts compat hpc tests data pyproject.toml README.md .gitattributes tinyx.nhr.fau.de:<WORK>/stroke-nav/

# 2. on the frontend (ssh tinyx.nhr.fau.de): venv under $WORK, CUDA PyTorch 2.14.1, package, anatomy
cd $WORK/stroke-nav && bash hpc/setup_tinygpu.sh

# 3. submit; everything after the script name goes to scripts/train_sac.py
sbatch.tinygpu hpc/train_sac.sbatch --total-timesteps 1000000                  # ~11-20 h
sbatch.tinygpu --array=0-3 hpc/train_sac.sbatch --total-timesteps 200000      # 4 seeds, 4 GPUs
sbatch.tinygpu -p a100 --gres=gpu:a100:1 hpc/train_sac.sbatch                   # other GPU type
squeue.tinygpu && tail -f slurm-stroke-nav-sac-*.out

# 4. back on your machine: fetch the run folders
scp -r tinyx.nhr.fau.de:<WORK>/stroke-nav/runs/<run_id> runs/
```

For a quick test use an interactive job (limit 4 h): `salloc.tinygpu --gres=gpu:1 --time=01:00:00`, then
`source $WORK/venvs/stroke-nav/bin/activate` and run `scripts/train_sac.py` directly. Compute nodes have no
direct internet (the scripts export the NHR proxy for pip), so fetch anatomy on the frontend or copy
`data/processed/` from your machine. All Slurm commands on this cluster use the `.tinygpu` wrappers.
Check GPU use of a running job with `srun.tinygpu --jobid=<id> --pty nvidia-smi` or in
[ClusterCockpit](https://doc.nhr.fau.de/job-monitoring-with-clustercockpit/).

## Environment (`stroke_nav.env.vessel_env.VesselNavEnv`)

- **Devices: `device-parameters.json` is the source of truth.** Every simulator, planner and robot path reads it
  through `stroke_nav.organiser.spec.DEFAULT_SPEC`: tip angle and length of both devices, and the planner's device
  diameters (`radifocus_035`, `judkins_5f`). With the organiser's robot come the control rate (5 Hz), speed limits
  (10 mm/s, 1 rad/s) and travel (guidewire 264.5 mm, catheter at most 229.2 mm ahead of the wire).
  `STROKE_NAV_DEVICE_PARAMS=<file>` selects another file. Stiffness, friction and damping only matter to the
  organiser's rod physics, because our simulator is kinematic. Targets past the robot's travel cannot be reached
  (for 0248, the distal ICA and M1 lie 450-530 mm along the vessels).
- **Action** `Box(-1, 1, (4,))`: `[v_guidewire, w_guidewire, v_catheter, w_catheter]`, scaled by the speed limits above.
  `device_sim_params()` builds the defaults; pass an explicit `SimParams` only for experiments outside the organiser's setup.
- **Observation** (46 floats): 3 tracking points per device tip, now and at the previous step, plus the target, the target minus the tip, and the previous action.
  These are the inputs used by Robertshaw et al. 2024 and 2025.
- **Reward:** `-0.005 - 0.01 * Δ(geodesic distance to target, mm) + 1.0 on success`.
  The papers use a progress coefficient of 0.001; set `progress_coef` to match.
- **Success:** the microcatheter tip (or the guidewire tip, via `success_device`) is within 5 mm of the target along the centerlines.
- **Device model** (`env/devices.py`): both devices share one track through the centerline graph. The leading device's pre-curved tip chooses the vessel at each bifurcation, based on its roll angle.
  Rotation is therefore required: pushing forward alone never reaches the carotids. A catheter close behind the wire reduces steering noise.
  The model is kinematic, with no wall forces yet, so contact force reports n/a and wall pushes stand in as the proxy.
- **Lumen limits:** a device cannot enter lumen narrower than its clearance requires (`SimParams.enforce_lumen`).
  For example, the microcatheter stops at a tight stenosis that the guidewire can cross.
  Episode routes, used by the oracle and for wrong-branch counting, are planned by the constrained A\* for the success device (`EnvConfig.plan_routes`).

## Baselines and tests

- `agents/scripted.py`: `ScriptedOracle` uses privileged knowledge of the route and the branch rule. It reaches 100% of targets in every tested region of 0248 (CCA, ICA, M1, A1, ECA, subclavian, vertebral) and of 0050.
  `RandomAgent` is the lower bound.
- **Second patient.** The whole pipeline also runs unchanged on `0024_H_AO_H`, a healthy aortic arch from another patient.
  Its left common carotid shares an origin with the right one. The follower and the oracle reach all four branch ends 100% of the time, against 0–5% for random actions and 0% for pushing only.
  Try it with `python scripts/plan_path.py --model 0024_H_AO_H --target LCCA@-2 --follow 20 --record`.
- `agents/follower.py`: `TrajectoryFollower` executes planned trajectories.
  It reaches 100% of random CCA, ICA, M1, A1 and vertebral targets on both the stenosis and the occlusion variant, at about 1.5 times the oracle's time, because it respects the speed profile.
- `pytest -q`: 45 tests, mostly on synthetic vessel trees. They cover:
  - env API conformance, determinism and the parser
  - oracle success on every branch and the need for rotation
  - START metrics
  - planner rerouting around a narrowing for the catheter but not the wire, and A\* matching Dijkstra without penalties
  - curvature-aware rerouting
  - trajectory speed, acceleration and ordering limits
  - the follower succeeding where shortest-path execution gets stuck
  - occlusion detection
  - the organiser interface: spaces, state layout, reward, success rule, run-folder validation, model pickling and class path, frame round trips

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
