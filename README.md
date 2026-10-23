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
- a pre-operative case planner: for one patient and one target, candidate trajectories from A\*, the
  calibrated entry-roll controller, behaviour cloning, action chunking and SAC, each under a time
  budget, in one format with safety checks ([§6](#6-plan-a-patient-case-pre-operative-planning))

The robot has 4 action parameters, `[v_guidewire, w_guidewire, v_catheter, w_catheter]` in [−1, 1].
The default anatomy is `0248_H_AOCERE_CAS`, a CT model from the aortic root to the circle of Willis
in 30 carotid-stenosis variants.

## Results

> **Wall crossings (2026-10-23).** The organiser's physics lets a pushed wire slip through a vessel wall
> between its collision bodies. Re-checked with `stroke_nav.safety.wall`, every entry-roll "reached" below on
> their physics crossed a wall on the way (cas-reference 3.6 mm, right ICA occlusion 3.1 mm, 70 % left lesion
> 1.2 mm, femoral to left CCA 2.2 mm). With the wall safety layer these runs back off and stop instead; none
> reaches its target without a crossing yet. See [docs/SAFETY.md](docs/SAFETY.md) §7.

One episode per row on the organiser's physics unless a count is given; videos are 2× real time with
their physics view on the left and the policy's fluoro input on the right (`results/` is not in git).

| Demo | Runs on | Result |
|---|---|---|
| New mesh `0024_H_AO_H` (aortic arch), target `RSA@10`: A\* route + calibrated entry roll, 20 min from the VMR model to the result | organiser's physics on the new mesh | **reached in 11.6 s**, 1.1 mm |
| Same mesh and target, behaviour-cloned policy from 12 demonstrations (2 min of training) | organiser's physics on the new mesh | **reached in 12.8 s**, 1.6 mm; passes the hand-over check |
| A\* route + calibrated entry roll on the organiser's own cas-reference goal | organiser's physics | **reached in 28.8 s**, 4.0 mm |
| Behaviour-cloned policy on that goal (24 demonstrations of the route, 10 min of training) | organiser's physics | **reached in 29.4 s**, 4.1 mm; passes the hand-over check |
| A\* route + calibrated entry roll, 70 % left ICA lesion (`10Right70Left`, `left_carotid@140`) | organiser's physics | **reached in 27 s**, 1.8 mm from the lesion, peak force 5 mN |
| Same, right ICA lesion (`right_carotid@118`) | organiser's physics | 9 mm short: the 301 mm route exceeds the 264.5 mm guidewire travel |
| Behaviour-cloned policy from 12 of those demonstrations (3.6 min of training) | organiser's physics | **reached in 26.4 s**, 1.9 mm; passes the hand-over check |
| Case planner on their cas-reference goal: entry roll, BC, ACT | organiser's physics | each **6/6** at 29 s; BC and ACT trained in about 10 min |
| Behaviour-cloned SAC policy (hand-over bundle) | organiser's physics | **8/8**, 30.4 s, 4.0 mm, peak force 9 mN |
| A\* planning on the centerline graph | CPU | 10-40 ms per route; local replan 143-350 edge states against 620-1040 for a cold plan |
| A\* + trajectory follower, catheter to right M1 past an 85 % ICA stenosis | our simulator | 20/20 in 26.5 s (plain shortest path 0/20) |
| Safety monitor on an imperfect policy | synthetic arch, stressed tracking | success 60 → 100 %, target overshoot 40 → 0 % |

### Demo commands

```bash
powershell -ExecutionPolicy Bypass -File tools\organiser_stack.ps1 stop        # their Docker stack would take the CPU
python scripts/demo_astar_costs.py --visible                                   # A*: plan, evaluate, execute on their physics; both variants, Chrome windows
python scripts/demo_lockstep.py --variant 10Right70Left --target left_carotid@140 --expert --slow-radius 8 --record results/astar_demo/astar_success.mp4        # video: A* success case
python scripts/demo_lockstep.py --variant 10Right70Left --target right_carotid@118 --expert --slow-radius 8 --record results/astar_demo/astar_travel_limit.mp4  # video: travel-limit case
python scripts/plan_case.py --variant 10Right70Left --target left_carotid@140 --case-id v70-left --methods astar,roll,bc --bc-demos 12 --sigmas 0,0.2 --eval-episodes 1   # 12 demos + behaviour cloning, ~6 min
python scripts/demo_lockstep.py --model results/cases/v70-left/runs/bc/final-model.zip --variant 10Right70Left --target left_carotid@140 --record results/cases/v70-left/bc_success.mp4   # video: the clone
python scripts/plan_case.py --target scenario --case-id cas-reference          # every method on their cas-reference goal, ~20 min
python scripts/demo_lockstep.py --target scenario --expert --slow-radius 8 --record results/cases/cas-reference/astar_roll.mp4   # video: A* route + entry roll on their goal
python scripts/demo_lockstep.py --target scenario --model results/cases/cas-reference/runs/bc/final-model.zip --record results/cases/cas-reference/bc.mp4   # video: the clone on their goal
```

A new mesh and target (the recipe in [docs/NEW_MESH.md](docs/NEW_MESH.md), about 20 minutes end to end):

```bash
python scripts/fetch_data.py --model 0024_H_AO_H                                               # VMR model -> centerlines
python scripts/make_scenario.py --model 0024_H_AO_H --target RSA@10                           # frame + their scenario + route length
python scripts/calibrate_rolls.py --model 0024_H_AO_H --rolls 0:360:30 --seeds 1 --workers 4    # which entry roll reaches which vessel, ~5 min
python scripts/plan_case.py --model 0024_H_AO_H --target RSA@10 --case-id 0024-rsa10 --table results/lockstep/roll_sweep_0024_H_AO_H.json --methods astar,roll,bc --bc-demos 12 --sigmas 0,0.2 --eval-episodes 1
python scripts/demo_lockstep.py --vmr-model 0024_H_AO_H --target RSA@10 --expert --slow-radius 8 --record results/cases/0024-rsa10/astar_roll.mp4
python scripts/demo_lockstep.py --vmr-model 0024_H_AO_H --target RSA@10 --model results/cases/0024-rsa10/runs/bc/final-model.zip --record results/cases/0024-rsa10/bc.mp4
```

Details and the full tables: [docs/PLANNER.md](docs/PLANNER.md), [docs/CASE_PLANNING.md](docs/CASE_PLANNING.md) and [docs/NEW_MESH.md](docs/NEW_MESH.md).

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
python scripts/fetch_data.py --model 0248_H_AOCERE_CAS --variant 40Right85Left
python scripts/fetch_data.py --model 0248_H_AOCERE_CAS --variant 10Left100Right
python scripts/plot_anatomy.py --model 0248_H_AOCERE_CAS        # results/<model>_anatomy.png
```

Only the centerlines, the surface model and the license are pulled, with HTTP range requests (the
717 MB zip is never downloaded). `data/processed/<model>_<variant>.npz` then holds the centerline graph
(1 mm nodes) and a lumen radius per node, measured from wall-to-wall rays so it stays accurate where
the centerline runs off-centre. Tails outside the surface are trimmed, and gaps of 3 mm or more in a
large vessel are marked as occlusions (the 100 % stenosis variants).

### `0248_H_AOCERE_CAS` Stenosis Variants (30 Configurations)

`0248_H_AOCERE_CAS` includes 30 paired bilateral carotid artery stenosis (CAS) surface meshes combining
10%, 25%, 40%, 50%, 70%, 85%, and 100% (complete occlusion) blockages on the Left and Right internal
carotid arteries (`left_carotid@140` and `right_carotid@118`):

| Dominant Side | Blockage Severity | Variant Configurations (`--variant <NAME>`) |
|---|---|---|
| **Left > Right** *(Left ≥ 50%)* | **100% Left** (occluded) | `10Right100Left`, `25Right100Left`, `40Right100Left`, `50Right100Left`, `70Right100Left`, `85Right100Left` |
| **Left > Right** *(Left ≥ 50%)* | **85% Left** | `10Right85Left`, `25Right85Left`, `40Right85Left` *(default symmetry pair to `40Left85Right`)* |
| **Left > Right** *(Left ≥ 50%)* | **70% Left** | `10Right70Left`, `25Right70Left`, `40Right70Left` |
| **Left > Right** *(Left ≥ 50%)* | **50% Left** | `10Right50Left`, `25Right50Left`, `40Right50Left` |
| **Right > Left** *(Right ≥ 50%)* | **100% Right** (occluded) | `10Left100Right`, `25Left100Right`, `40Left100Right`, `50Left100Right`, `70Left100Right`, `85Left100Right` |
| **Right > Left** *(Right ≥ 50%)* | **85% Right** | `10Left85Right`, `25Left85Right`, `40Left85Right` *(default variant)* |
| **Right > Left** *(Right ≥ 50%)* | **70% Right** | `10Left70Right`, `25Left70Right`, `40Left70Right` |
| **Right > Left** *(Right ≥ 50%)* | **50% Right** | `10Left50Right`, `25Left50Right`, `40Left50Right` |

*(Note: `70Left100Right` and `85Left100Right` also have >50% Left blockage combined with 100% Right occlusion.)*

Other cerebral/aortic VMR models tested with this pipeline include `0024_H_AO_H` (healthy aortic arch,
5 branches) and `0050_H_CERE_H`, `0051_H_CERE_H`, `0052_H_CERE_H`, `0053_H_CERE_H` (healthy cerebral
arteries).

## 2. Drive and pick targets

```bash
python scripts/playground.py --model 0248_H_AOCERE_CAS
python scripts/playground.py --model 0248_H_AOCERE_CAS --variant 40Right85Left
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
| `occlusion` (or `stenosis`) | proximal entrance of the primary occlusion / stenosis (default in `playground.py`) |
| `left_carotid@120`, `left_carotid@-5` | 120 mm along a branch; 5 mm from its distal end |
| `random:left_carotid@-15--1\|right_carotid@-15--1` | a new random node per episode (here M1) |
| `xyz:-120,110,-200`, `1234` | nearest node to a point (mm); a node id |
| `file:data/targets/0248_H_AOCERE_CAS.json` | a point picked in the playground |

For 0248 the carotid paths run CCA, ICA, then M1; `*_anterior` is A1 and `*_carotid_external` the ECA.

## 3. Plan a path and a robot trajectory

The constrained A\* costs inserted length, estimated time and damage (bending effort, curvature
integral, narrow lumen, a contact-force proxy) with named profiles (`shortest`, `fastest`, `gentle`,
`balanced`), keeps forbidden vessels and lumen clearance as hard constraints, reports the four
evaluations of a route before execution, and replans locally from the current tip
([docs/PLANNER.md](docs/PLANNER.md)). `python scripts/demo_astar_costs.py --visible` plans the ICA
lesions of two stenosis variants and executes the routes on the organiser's physics.

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

**Multiple targets (experimental).** The same pipeline for goals anywhere the entry roll can reach:

```bash
python scripts/calibrate_rolls.py --workers 4                    # roll sweep, 24 rolls x 2 seeds, ~15 min
python scripts/make_goals.py --spacing 3 --per-branch 60          # 168 goals: 134 train, 33 test, + cas-reference
python scripts/record_demos.py --goals results/lockstep/goals.json --split train --episodes 268 \
    --sigmas 0,0.2,0.4 --slow-radius 8 --workers 4 --max-steps 300 --out runs/demos/multi-target
python scripts/train_bc.py --demos runs/demos/multi-target --successful-only --epochs 40 --device cuda --run-id stroke-nav-bc-multi
python scripts/eval_lockstep.py --model runs/stroke-nav-bc-multi/final-model.zip --goals results/lockstep/goals.json \
    --split test --episodes 33 --workers 4 --max-steps 300
python scripts/demo_lockstep.py --model runs/stroke-nav-bc-multi/final-model.zip --goals results/lockstep/goals.json --goal-id 153
```

How it works:
- `calibrate_rolls.py` records the tip path of each entry roll.
- `RollTable` (`src/stroke_nav/agents/roll_table.py`) counts a point as reachable when every seed's
  path for some roll passes within 3 mm of it.
- `make_goals.py` samples such points by branch and splits them into train and test.
- The expert uses each goal's roll. It slows only in the last 8 mm: with the 25 mm default it missed
  one goal by 15 mm, because a slower push can change which branch the rod takes.
- `make_env(goals=...)` gives their env a `GoalSet`, whose `set_next` picks each episode's goal.

The fluoro image still shows the scenario's goal marker, as in their stack, so the policy reads the
goal from the state.

Results so far (2026-10-23):
- **Expert:** 144 of 167 recorded demos succeeded (86%). The failures are near misses that do not
  hold for 3 steps, and deep right-carotid goals where the stop sends the rod into another branch.
- **BC policy** (`stroke-nav-bc-multi`, on the 144 successful demos): validation action MSE 0.020,
  against 0.005 for the single-goal model. **6 of 16** held-out goals reached before the evaluation
  was stopped; the cas-reference split was not run.
  - The 6 successes: 5 aorta goals and left-carotid goal 70.
  - Shallow aorta goals (40-72 mm in): it passed within about 5 mm, then pushed on to the end of
    travel.
  - Left carotid and left subclavian goals: it chose the wrong entry roll. Goal 82 needs 345° and got
    about -168°.

It learned the push, not the choice of roll or when to stop. Hand over `stroke-nav-bc-v1` (8/8 on
the cas-reference goal). `scripts/finetune_sac_lockstep.py` (SAC fine-tuning from a BC model) is
written but has not been run.

**Manual goals and demonstrations.** A person picks the goals and drives the robot by keyboard on
their physics. The recordings use the same format as `record_demos.py`, so `train_bc.py` trains on
manual demos, scripted ones, or both.

```bash
python scripts/pick_goals.py --target left_subclavian@50 --target right_subclavian@40 --target right_carotid@60
python scripts/pick_goals.py --target left_carotid@30 --split test --append       # add held-out goals
python scripts/teleop_lockstep.py --goals results/lockstep/manual_goals.json --episodes 3 --out runs/demos/manual
python scripts/train_bc.py --demos runs/demos/manual,runs/demos/multi-target --successful-only --device cuda --run-id stroke-nav-bc-manual
python scripts/eval_lockstep.py --model runs/stroke-nav-bc-manual/final-model.zip --goals results/lockstep/manual_goals.json --split test
```

- **Picking goals:** `pick_goals.py` takes the repo's target specs (`branch@mm`, `xyz:...`;
  `--branches` lists the branch names). It converts them to their world frame and prints the entry
  roll whose recorded paths pass closest, with the miss in mm. A miss above 5 mm is probably out of
  reach, and `--snap` moves such a goal onto the rod's path.
- **Driving:** `teleop_lockstep.py` opens their two pages in Chrome, with the goal in the physics
  view, plus a "teleop" window that shows the policy's fluoro input, the distance, the insertion and
  the roll. Keys, with the teleop window focused:
  - Hold W / S (or Up / Down) to push or pull.
  - Hold A / D (or Left / Right) to turn: 11.5° per 0.2 s step at full speed.
  - 1 / 2 / 3 set the speed to 25 / 50 / 100 %.
  - Enter saves the episode, Backspace discards it, Esc quits.

  The wire moves only while a key is held (Windows reads the key state directly). With `--keys`,
  or on other systems, W / S latch until Space and each A / D press turns one step.

  An episode also ends, and is saved, on their success rule or at `--max-steps`. On their rod: turn
  to the listed entry roll first, then push. It runs at most at real time.
- `--keys "d*10,w,_*150"` replays scripted key presses instead of the keyboard, for testing.

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

**Deploy a planned case** (A\* + entry roll and the behaviour-cloned policy, on the case's variant mesh):

```bash
python scripts/export_deployment.py --case results/cases/v70-left                          # handover/stroke-nav-v70-left.zip
python scripts/check_deployment.py --bundle handover/stroke-nav-v70-left --episodes 1 --visible   # both, from the bundle, through their evaluator on their physics
# on their Linux machine, bundle unzipped next to HCMR-EV-OpenNG3, from HCMR-EV-OpenNG3:
bash ../stroke-nav-v70-left/deploy.sh install
bash ../stroke-nav-v70-left/deploy.sh astar-roll 5 --visible       # or: bc 5 --visible; then: results
```

[docs/HANDOVER.md](docs/HANDOVER.md) lists what matches their stack and what is approximate, what the
bundle contains, and how to read their results (including what `aborted` episodes mean). Their own
stack also runs on this laptop with `powershell -ExecutionPolicy Bypass -File tools\organiser_stack.ps1 start`
(docs/ORGANISER_SETUP.md §11), but not fast enough for their RL env.

## 6. Plan a patient case (pre-operative planning)

The revised requirement is per-patient, per-target planning rather than a generic policy: given the
patient's vessel model and one target, propose the path the robot will take, from several methods,
each within a known training budget, so that a safety layer or a clinician can choose
([docs/CASE_PLANNING.md](docs/CASE_PLANNING.md)).

```bash
powershell -ExecutionPolicy Bypass -File tools\organiser_stack.ps1 stop          # the lockstep pages need the CPU
python scripts/plan_case.py --target right_carotid@40 --case-id rca40            # all default methods, ~25 min
python scripts/plan_case.py --target xyz:-78.0,118.5,-274.0 --methods astar,roll,bc --bc-minutes 5
```

- **Methods** (`--methods`): `astar` (A\* + closed-loop follower, our simulator, seconds), `astar_theirs`
  (the same follower on their physics), `roll` (the calibrated entry-roll controller), `bc`
  (demonstrations at this target + behaviour cloning into the hand-over SAC model), `act` (action
  chunking on the same demonstrations), `sac` (SAC fine-tuning from the BC model). Budgets:
  `--bc-minutes`, `--act-minutes`, `--sac-minutes`, `--eval-episodes`.
- **Output** `results/cases/<case-id>/`: `summary.md` (one row per method: train time, plan time, success,
  time to goal, safety verdict and flags), `candidates/<method>.json` (every rollout in one format: tip
  path, insertion, actions, distance, force), `trajectory_<method>.csv` (the representative episode),
  `candidates.png`, `case.json` (route and reachability) and `goals.json` (the target for
  `eval_lockstep.py` / `demo_lockstep.py --goals ... --split train`).
- **Safety checks** on every trajectory, with the thresholds of docs/SAFETY.md: lumen corridor,
  wrong-vessel entry, overshoot, contact force, speed envelope, robot travel.
- Learned candidates are trained on the organiser's physics in the lockstep env; the BC and SAC models
  pass `check_handover.py` and can be handed over for this one target.

## Environment reference (`stroke_nav.env.vessel_env.VesselNavEnv`)

- **Devices:** `device-parameters.json` is the source of truth for both devices (tip angle and length,
  diameters) and the organiser's robot (5 Hz, 10 mm/s, 1 rad/s, guidewire travel 264.5 mm, catheter at
  most 229.2 mm ahead of the wire). `STROKE_NAV_DEVICE_PARAMS=<file>` selects another file. Targets past
  the travel cannot be reached (for 0248 the distal ICA and M1 lie 450-530 mm along the vessels).
- **Robot profiles** (`robots/<name>/`: an organiser-format `device-parameters.json` and a `robot.json`
  with the travel limits), selected with `STROKE_NAV_ROBOT=<name>` for every simulator, planner, lockstep
  page and run folder. `long-reach` is a hypothetical robot for femoral access: guidewire travel 750 mm
  (their page accepts insertion up to 800 mm) and a 1.2 m catheter (their page's maximum; the wire is
  already 1.2 m). On `0028_H_ABAO_H` (aortic root to both femorals), from the right femoral end the left
  CCA@30 is 604 mm and the right CCA@-5 665 mm along the route. Its device hash differs from theirs, so
  it is for planning ahead of a robot change, not for a hand-over:

  ```bash
  set STROKE_NAV_ROBOT=long-reach
  python scripts/make_scenario.py --model 0028_H_ABAO_H --start 1054 --target Left_Common_Carotid@30 --target Right_Common_Carotid@-5
  ```
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
