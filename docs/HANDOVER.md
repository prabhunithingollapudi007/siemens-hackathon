# Handing work over to the organiser

There are two kinds of deliverables:

- a **trained SAC policy** that loads in their `openng3_rl` evaluation
- a **planned trajectory** they can replay on their ROS 2 stack

Their system is described in [ORGANISER_SETUP.md](ORGANISER_SETUP.md). The code for this
compatibility layer is in `src/stroke_nav/organiser/`.

## What matches, what is approximate, what is missing

| | Status | Notes |
|---|---|---|
| Action space and scaling | **identical** | `Box(-1,1,(4,))`; ±10 mm/s, ±1 rad/s; 5 Hz control |
| Observation space | **identical** | `Dict(image (1,128,128) uint8, state (43,) float32 in ±10)`, same layout and units (world metres) |
| Reward, success, episode length | **identical** | their `compute_reward`; 5 mm for 3 steps; 600 steps |
| Noise | **same model** | tip σ 0.1 mm, tip dropout 7 % with hold; no latency |
| Policy class and hyperparameters | **identical** | SB3 2.9.0 SAC `MultiInputPolicy` + `openng3_rl.policy.FluoroStateExtractor`; their `online_rl.json` |
| Device parameters | **identical file** | `device-parameters.json` hash `38b1ea53…` equals their run's |
| Scenario | **identical manifest** | `cas-reference`, mesh sha256 `04fbd581…`, the same goal |
| Anatomy and frame | **same patient, fitted** | our VMR 0248 vs their `vessel.obj`: rigid, median 0.45 mm, max 1.6 mm; our start sits within 2 mm of their device entry |
| Toolchain | **pinned to theirs** | Python 3.12, NumPy 1.26.4, Gymnasium 1.4.0, SB3 2.9.0, PyTorch 2.14.1 |
| Device physics | **approximate** | ours is kinematic: devices ride the centerlines and roll picks the branch. Theirs is XPBD rods with contact. |
| Contact forces | **missing** | all force inputs are 0 unless `--force-proxy-n` reports blocked pushes as force |
| Fluoro image | **approximate** | same size, grey levels and roughly the same camera, but no grid or lighting; expect a visual domain gap |

Because of the last three rows, a policy trained here is a **warm start**. Plan to fine-tune it in their
simulator.

**Policy trained on their physics instead.** `stroke_nav.organiser.live.lockstep` runs their own
physics page, observation, reward and success rule faster than real time. A behaviour-cloned SAC
policy trained there (`runs/stroke-nav-bc-v1`, `scripts/record_demos.py` + `scripts/train_bc.py`)
reaches the cas-reference goal in 8 of 8 lockstep episodes. Its physics, images and forces come from
their own pages (only the robot and noise node are emulated), so it is the run to hand over first. Steps 2-4 below apply to it unchanged. See
[ORGANISER_SETUP.md](ORGANISER_SETUP.md) §14. Their `./sys startup --mode rl` continues training from a checkpoint if their training code
is pointed at it. That isn't automated by them; it's a small change on their side.

## Steps

### 1. Train in our simulator (organiser interface)

```bash
.venv\Scripts\python.exe scripts/train_sac.py --total-timesteps 50000
.venv\Scripts\python.exe scripts/train_sac.py --goal-mode curriculum --curriculum-group near --total-timesteps 20000
.venv\Scripts\python.exe scripts/train_sac.py --goal-mode target --target "random:right_carotid@20-60"
```

- **Goal modes:**
  - `scenario` uses their cas-reference goal.
  - `curriculum` uses their near, medium and far offsets along +Z from the reset tip.
  - `target` takes any of our target specs.
- **Output:** `runs/<run_id>/`, already in their run-folder layout.
- **Memory:** their 50,000-transition image replay buffer needs about 1.7 GB of RAM. Lower it with `--buffer-size`.
- **Speed:** CPU training is slow, about 3 steps/s with their CNN at batch 256. For real runs, install a
  CUDA build of PyTorch 2.14.1 in the venv. The organiser uses cu130.

### 2. Check it

```bash
.venv\Scripts\python.exe scripts/check_handover.py --run-dir runs/<run_id> --episodes 3
```

This validates the run folder the way their evaluation does: schema, device-parameter hash and scenario
manifest. It also confirms three things about the model:

- it references `openng3_rl.policy.FluoroStateExtractor`
- it was pickled with NumPy 1.x
- it loads with identical spaces and runs

Add `--organiser-model "existing code base/sac_210000_steps.zip"` to load their own checkpoint the same way.
It loads with identical spaces.

### 3. Package it

```bash
.venv\Scripts\python.exe scripts/export_handover.py --run-dir runs/<run_id> --plan-target right_carotid_external@21
```

This writes `handover/<name>/` and `handover/<name>.zip`:

| Path | Use on their side |
|---|---|
| `runs/<run_id>/` | copy into `HCMR-EV-RL/runs/` |
| `trajectory/device_commands.csv` | `ros2/ros2_device_command_player.py`: absolute setpoints (mm, deg) at 20 Hz on `/endovascular/<dev>/command` |
| `trajectory/twists.csv` | `ros2/ros2_twist_player.py`: velocities (mm/s, rad/s) at 5 Hz through `AlgorithmControllerBase` |
| `trajectory/policy_actions.csv` | the same trajectory as normalised RL actions |
| `trajectory/goals.txt` | the target in their mesh frame (cm), for a scenario goals file |
| `organiser_frame.json` | our anatomy and their world frame |
| `compat/openng3_rl/policy.py` | only needed where their `openng3_rl` package is not installed |
| `MANIFEST.json` | sha256 of every file |

The planned path uses their devices: the 0.035" wire is the hard limit, and the 5F catheter stops where
the lumen gets too narrow for it.

- **Insertion setpoints** transfer directly.
- **Rotation setpoints** come from our kinematic tip model, so treat them as a starting point that needs closed-loop correction.
- **The ROS 2 players** have not been run outside their stack.

### 4. Run it on their side (from `HCMR-EV-OpenNG3`)

```bash
cp -r runs/<run_id> ../HCMR-EV-RL/runs/
./sys startup --mode showcase --scenario cas-reference --run-id <run_id> \
    --model final-model.zip --episodes 10 --visible
```

Their evaluation writes `HCMR-EV-RL/runs/<run_id>/evaluations/<id>/episodes.json`, and recorded rollouts
appear in the dataset viewer at http://localhost:8090.

## Frame transform

```
vmr_cm  = M @ obj_cm + t_cm        (our VMR model frame; mm = 10 * cm)
world_m = 0.01 * obj_cm            (their simulator world frame)
```

| Constant | Value |
|---|---|
| `M` | rotation of about 19.5°, no axis flip |
| `t_cm` | (−10.0697, 3.9739, −38.7242) |

Load it with `stroke_nav.organiser.frame.default_frame()`, which reads `data/processed/organiser_frame.json`
if present. Refit it with `fit_frame(...)`.

Validation:

| Check | Result |
|---|---|
| Recorded wire body positions (3,016) from a teleoperation dataset | 100 % inside our lumen, at least 1.5 mm from the wall |
| Recorded tip positions | 100 % inside our lumen |
| Their goal, round-tripped through the transform | within 0.5 mm |

## Keeping in sync with the organiser

`stroke_nav.organiser.spec.load_from_repo("<path>/HCMR-EV-RL", run_id=...)` re-reads their
`configs/online_rl.json` and a run's `device-parameters.json`. If they change speeds, rates or devices,
rebuild with that spec instead of the built-in defaults.
