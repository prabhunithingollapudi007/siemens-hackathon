# Organiser setup (HCMR-EV) — reference

A reference for the organiser's code base, as delivered in `Hackathon.zip`. File paths are relative to its
`Hackathon/` folder. A source-only copy is extracted at `C:\Users\prabh\Documents\hackathon\Hackathon_extract\Hackathon`.
How our work plugs into this stack is described in [HANDOVER.md](HANDOVER.md).

The values below were read from the code and the run snapshots on 2026-10-22. Items marked *(doc)* come
from the organiser's own documentation and were not checked against code.

## 1. What it is

A two-module endovascular robot:

- **Module 1** advances and rotates the **guidewire**.
- **Module 2** advances and rotates the **catheter**.

That gives 4 degrees of freedom in total. The same control stack drives either a simulator or real
hardware.

| Sub-project | Role |
|---|---|
| `HCMR-EV-OpenNG3` | ROS 2 Jazzy orchestrator (`./sys`). Arbitrates Twist commands from teleoperation, Xbox, RQT and autonomous sources, and runs the UDP bridge to the robot model or the hardware. |
| `HCMR-EndoNavSimDocker` | "CatForce" physics simulator running in the browser, with its ROS bridge, a sensor-noise layer, a LeRobot recorder and a dataset viewer. |
| `HCMR-EV-RL` | `openng3_rl`, a Gymnasium environment plus SAC training and evaluation. |
| `HCMR-EV-Integration` | Integration contract, hand-off notes and the episode reset coordinator. |
| `HCMR-EV-LeRobot_Training` | Offline imitation learning (ACT, SmolVLA) on recorded datasets. |
| Monadnock *(doc)* | Generated robot model and simulation HAL. Delivered only as a Docker image (`monadnock-sim`). |

## 2. Data flow

```
policy / RQT / Xbox / PIM ── Twist ──> /control/<source>/module_<n>/cmd_vel
   └─ arbiter (first non-zero source owns a module; 0.35 s source timeout) ──> /module_<n>/cmd_vel
      └─ adapter ──> /ng3/controller/joint_trajectory (velocities only)
         └─ ros_to_udp ── UDP :47100 ──> monadnock-sim | Speedgoat 192.168.7.2 | remote VNG33
            └─ UDP :47101 ── udp_to_ros ──> /ng3/joint_states (m, rad)
               └─ joint_state_follower ──> /endovascular/{guidewire,catheter}/command (DeviceCommand, mm/deg)
                  └─ WebSocket :18765-18768 ──> browser physics page (catheter_guidewire_ros2.html, 60 Hz)
                     ├─ /endovascular/<dev>/state, contact forces, /imaging/simulation/compressed (ground truth)
                     └─ sensor_sim noise ──> /endovascular/<dev>/state/noisy + noisy viewer (:18769)
                          └─ /endovascular/imaging/noisy/compressed ──> RL env and LeRobot recorder
reset: Trigger service /hcmr/episode/reset (reset_coordinator)     TensorBoard :6006   dataset viewer :8090
```

Everything runs on ROS domain 42 with Fast DDS, over the Docker network `hcmr-ev-integration`.

Sources:
- `HCMR-EV-Integration/INTEGRATION_CONTRACT.md`
- `HCMR-EV-OpenNG3/src/bridge/config/bridge.yaml`
- `HCMR-EV-RL/compose.yaml`

The simulator pages are:

- Two-device page: http://localhost:8000/catheter_guidewire_ros2.html
- Single-wire page: http://localhost:8000/guidewire_simulation_ros2.html
- Noisy viewer: http://localhost:8001/noisy_visualization_ros2.html

## 3. Simulator (EndoNav / CatForce)

- **Physics.** JavaScript in the browser, not SOFA. Two 3D Cosserat rods are solved with direct XPBD,
  with the guidewire running coaxially inside the catheter lumen. It runs at 60 Hz, with 5 substeps and
  2 iterations by default.
  - Source: `HCMR-EndoNavSimDocker/src/catforce_ros_bridge/web/catheter_guidewire_ros2.html`.
  - SOFA/stEVE appears only as an optional legacy visualiser in OpenNG3.
- **Devices.** The run snapshot that SAC was trained with
  (`HCMR-EV-RL/runs/online-rl-pool-20261021-145046-707530Z/device-parameters.json`, schema 1, 42 controls) uses:

  | | Preset | Diameter | Length | Tip length | Tip angle | Bodies |
  |---|---|---|---|---|---|---|
  | Guidewire | `radifocus` (0.035") | 0.89 mm | 1.2 m | 30 mm | 45° | 160 |
  | Catheter | `judkins` (5F) | 1.67 mm | 0.9 m | 28 mm | 70° | 56 |

  - A 0.014" preset (`synchro2`) exists.
  - The catheter diameter slider starts at 0.60 mm, so a 1.7F (0.57 mm) microcatheter is just out of range.
  - UI limits: insertion −50 to 800 mm, roll ±1080°.
  - Default feed rate 45 mm/s, spin rate 180°/s.
- **"Fluoro" image.** Not an X-ray simulation.
  - It is a 480×480 JPEG perspective WebGL render of the vessel (dark red), a cyan grid, the devices and
    an orange goal marker, from a fixed oblique camera.
  - It is published at up to 20 Hz on `/endovascular/imaging/noisy/compressed`.
  - The RL bridge converts it to greyscale and resizes it to 128×128 (`openng3_rl/ros_io.py`).
- **Noise** (`HCMR-EndoNavSimDocker/src/sensor_sim/config/noise_params.yaml`):
  - tip position σ 0.1 mm
  - tip dropout probability 0.07
  - 50–150 ms latency
- **Units and frames.** SI inside the simulator.
  - World frame = mesh × `meters_per_unit` (0.01, so cm become m).
  - The device entry is at the origin, and insertion runs along +Z (bake mode "authored").
  - `DeviceState` reports insertion in m, rotation in rad, and poses in m in the world frame.
- **Scenario `cas-reference`** (`HCMR-EV-RL/scenarios/cas-reference/`):
  - `vessel.obj` is identical to `HCMR-EndoNavSimDocker/meshes/Mesh_1/0248_H_AOCERE_CAS_reference_10k.obj`:
    sha256 `04fbd581…`, 4,995 vertices, 10,000 faces, in cm.
  - It is VMR model 0248 (reference variant 40Left85Right), decimated, rotated about 19.5° and translated.
    We fitted the transform to our VMR anatomy ourselves: median residual 0.45 mm. See HANDOVER.md.
  - `goals.txt` holds one goal, `7.001,9.756,15.350` (mesh cm), which is the world goal [0.07001, 0.09756, 0.1535] m.
    It lies 21 mm into the **right external carotid**, just past the right carotid bifurcation.
  - The header claims "OBJ LPS (R and A negated)", but the geometry shows no axis flip relative to VMR.
  - Goal file format: one `x,y,z` row per goal, `#` comments allowed, and a uniform pick per episode.
  - `configs/goals.json` is an alternative curriculum: offsets of 15–180 mm along +Z from the
    post-reset guidewire tip, in groups near, medium and far.

## 4. RL interface (`HCMR-EV-RL/src/openng3_rl`)

| Item | Value | Source |
|---|---|---|
| Environment | `OpenNG3Env(gym.Env)`, frame-synchronised, real time | `env.py` |
| Action | `Box(-1, 1, (4,), float32)` = [gw_trans, gw_rot, cath_trans, cath_rot]; scaled to ±0.01 m/s and ±1 rad/s; sent as Twist on `/control/autonomous/module_{1,2}/cmd_vel` with linear.x in **mm/s** and angular.z in rad/s | `action.py`, `ros_io.py` |
| Control rate | 5 Hz on a fixed wall-clock grid; each step waits for a synchronised image + state packet (skew ≤ 0.1 s, timeout 2 s gives `stale_abort`) | `env.py`, `configs/online_rl.json` |
| Observation | `Dict(image=Box(0,255,(1,128,128),uint8), state=Box(-10,10,(43,),float32))`, built only from **noisy** topics | `observation.py` |
| Reward | `clip(Δdistance / 0.01 m, ±1) − force·dt / 0.01 − 0.02·mean(Δaction²) − 0.001 + 10·success − 10·unsafe`, using the **ground-truth Euclidean** distance from the guidewire tip to the goal | `reward.py`, `env.py` |
| Success / unsafe | Guidewire tip within 5 mm for 3 consecutive steps / peak force > 3 N | `configs/online_rl.json` |
| Episode | 600 steps (120 s); the `EnvConfig` default and README say 300 | `configs/online_rl.json` |
| Reset | Trigger `/hcmr/episode/reset`; the goal is sampled after reset (curriculum goals are relative to the reset tip) | `env.py`, `goal.py` |

State vector layout (43 values, clipped to ±10; positions in world metres):

| Index | Content |
|---|---|
| 0–6 | guidewire tip position (3) + orientation quaternion x, y, z, w (4) |
| 7–13 | catheter tip position (3) + quaternion (4) |
| 14–24 | guidewire aux: tip velocity (3, m/s), insertion depth (m), sin(roll), cos(roll), log1p(tip force), log1p(peak force), log1p(contact count), tip dropout flag, force dropout flag |
| 25–35 | catheter aux (same layout) |
| 36–38 | goal − guidewire tip (m) |
| 39–42 | previous action |

Further details:
- **Quirk:** the "previous action" (39–42) in the observation returned by `step(a_t)` is `a_{t-1}`,
  not `a_t`. The observation is built before `_prev_action` is updated (`env.py`, around line 186).
- During a dropout, the last good sample is held.
- The tip quaternion's local +Z points along the tip. We inferred this from recorded data.
- Segment poses are ordered from proximal (index 0) to the tip.

**Policy and training** (`training.py`, `policy.py`, `configs/online_rl.json`):
- Stable-Baselines3 `SAC("MultiInputPolicy")` with `features_extractor_class=openng3_rl.policy.FluoroStateExtractor`:
  - a CNN on the image (32-64-64 conv layers, then 256 features)
  - an MLP on the state (128-128)
- Hyperparameters:
  - learning rate 3e-4, buffer 50,000, learning starts 2,000, batch 256
  - γ 0.99, τ 0.005, train_freq 1 step
  - gradient steps 1 for a single worker, −1 for pools
  - 1,000,000 total steps, checkpoint every 10,000 steps (`checkpoints/sac_<N>_steps.zip`), seed 0
- Versions recorded in their model: Python 3.12.3, SB3 2.9.0, Gymnasium 1.4.0, PyTorch 2.14.1+cu130, NumPy 1.26.4.

**Run folder** (`HCMR-EV-RL/runs/<run_id>/`):
- `resolved-config.json` (schema 1, with scenario name and mesh hash, plus the device-parameters file and sha256)
- `device-parameters.json`
- `scenario.json` (must equal the scenario manifest)
- `status.json`, `checkpoints/`, replay buffers, `tensorboard/`, `final-model.zip`
- `evaluations/<id>/{status,episodes}.json`
- `runs/latest-run.txt` names the newest run.

The evaluation refuses a run whose device parameters, scenario manifest or mesh hash differ from the live
simulator (`evaluation.py`, `device_parameters.py`).

**Commands**, run from `HCMR-EV-OpenNG3` (`./sys`):

```bash
./sys startup --mode rl --scenario cas-reference [--visible]          # train SAC
./sys startup --mode showcase --scenario cas-reference --run-id RUN_ID \
    --model checkpoints/sac_10000_steps.zip --episodes 10 [--visible]  # evaluate + record rollouts
./sys pool start fast --workers 2 --scenario cas-reference --gpu GPU-UUID   # experimental parallel pool
./sys startup --mode vla                                              # teleoperation recording
```

## 5. Control interfaces and units (`HCMR-EV-Integration/INTEGRATION_CONTRACT.md`)

- **Autonomous control path.**
  - Publish Twist on `/control/autonomous/module_<n>/cmd_vel` at 5 Hz or more (20 Hz recommended):
    linear.x in mm/s, angular.z in rad/s.
  - Template: `HCMR-EV-OpenNG3/src/bridge/scripts/algorithm_control_template.py`
    (`AlgorithmControllerBase.compute_command(module_id, feedback)`).
  - The bridge clamps to ±20 mm/s and ±5 rad/s.
  - Commands are **velocity-only**.
- **Direct simulator path.** Publish `catforce_interfaces/msg/DeviceCommand` on `/endovascular/{guidewire,catheter}/command`.
  - Fields: absolute `insertion_setpoint` (mm), `rotation_setpoint` (deg), `enabled: true`.
  - The two-device page latches each setpoint and tracks it at the feed and spin rates.
  - `frames` matters only on the single-wire page (3 frames = 20 Hz).
  - Source: `HCMR-EndoNavSimDocker/src/catforce_ros_bridge/ROS2_INTERFACE.md`.
- **Feedback.**
  - `/ng3/joint_states` in m and rad, relative to the first sample.
  - `/endovascular/<dev>/state` (`DeviceState`) carries insertion depth (m), shaft rotation (rad), tip pose and twist,
    segment poses, and forces (proximal, contact, tip, axial, peak, wall friction, viscous, elastic…).
  - `NoisyDeviceState` is the same with noise and dropout flags.
- **Reset coordinator** (`HCMR-EV-Integration/reset_coordinator/`): a fail-closed transaction of 15 s or less:
  1. recorder prepare
  2. reset inhibit
  3. follower off
  4. restart the plant
  5. wait until stationary
  6. simulator reset
  7. noise reset
  8. follower on, then release

## 6. Datasets (LeRobot v3.0)

The recorder writes `data/chunk-000/file-000.parquet`, `videos/observation.images.fluoro/...mp4` and
`meta/{info.json, episode_outcomes.json, tasks.parquet, stats.json}`. Features:

- `observation.images.fluoro`: 480×480×3 video
- `observation.state` [30]: per device, tip position (m), quaternion, linear and angular velocity, proximal force (N), tip force (N)
- `observation.guidewire_segment_poses` [160, 7] and `observation.catheter_segment_poses` [56, 7], in world m
- `action` [4]: guidewire mm/s, rad/s, catheter mm/s, rad/s
- `episode_outcome`, `timestamp`, `frame_index`, `episode_index`, `task_index`

Teleoperation is recorded at 20 fps and RL rollouts at 5 fps. `HCMR-EV-LeRobot_Training` trains ACT
(100k steps) or SmolVLA (20k steps, from `lerobot/smolvla_base`) with `lerobot==0.6.0`.

## 7. Requirements

- Linux or WSL2 with Docker and Compose v2. Ubuntu 24.04 is recommended. Native Linux Chromium is needed for pools.
- ROS 2 Jazzy, an NVIDIA GPU for training (their image uses CUDA 13.0 PyTorch), and Chrome or Chromium for the physics page.
- Python 3.12. `openng3_rl` pins `gymnasium==1.4.0` and `stable-baselines3==2.9.0`; the container has NumPy 1.26.4.

## 8. Results so far (in the delivered snapshot)

- **Deterministic evaluations: 0% success everywhere.**
  - Run `online-rl-20261021-073747` at 10k steps: 2×10 episodes.
  - Pool run `114214` at 10k steps: 10 episodes, final distance about 82 mm.
- **4-worker pool `145046`:** 95 successes in 394 stochastic training episodes. That was at 193,820 steps,
  10.4 transitions/s and about 5.2 h, and it was still training when the snapshot was taken.
- A later checkpoint, `sac_210000_steps.zip`, was provided separately.
- `docs/pool-acceptance-2026-10-21.md` reports two-worker isolation passing and a 1.55× throughput gain
  (target 1.6×). Policy quality was not evaluated there.

## 9. Open items they list

These come from the sub-project `ROADMAP.md` and `HANDOFF.md` files:

- unit doc mismatch between `DeviceCommand` and `ROS2_INTERFACE.md`
- only two modules supported
- no faster-than-real-time lockstep simulation (we built one on their pages: see §14)
- a headless or CLI simulator instead of the browser
- OpenNG3 owning the joint trajectory
- recorder migration to structured topics
- remote Monadnock acceptance
- a 4-worker pool benchmark and a policy-quality comparison

No task statement, judging criteria or deadlines are included in the archive.

## 10. Getting more out of `Hackathon.zip`

The zip is 2.2 GB compressed and 41.9 GB extracted, and 39.8 GB of that is SAC replay buffers. Read single
files without extracting:

```python
import zipfile
z = zipfile.ZipFile("Hackathon.zip")
print([n for n in z.namelist() if n.endswith(".md")])             # list
print(z.read("Hackathon/HCMR-EV-RL/configs/online_rl.json").decode())  # read one file
z.extract("Hackathon/HCMR-EV-RL/runs/<run>/final-model.zip", "Hackathon_extract")  # extract one file
```

## 11. Running their stack on this laptop (Windows 11 + Docker Desktop)

Status on 2026-10-22: the simulator, the control path and the browser physics run locally and were
verified end to end. A 5 mm/s Twist on module 1 moved the Monadnock joint and the XPBD guidewire in
the browser by the same distance (+34.6 mm robot, +33 mm browser). Their RL training and showcase modes
go through the bash `./sys` script and still need Linux or a WSL Ubuntu; they are not set up yet.

**Layout.** The repositories are extracted as siblings in `C:\Users\prabh\Documents\hackathon\HCMR-EV\`
(source, Docker files and the Monadnock image archive; no build output, datasets or runs).

**One command** (from this repo):

```bash
powershell -ExecutionPolicy Bypass -File tools\organiser_stack.ps1 start
```

`tools/organiser_stack.ps1` follows `HCMR-EV-Integration/HANDOFF.md`:

1. It loads the Monadnock image and builds `hcmr/openng3-bridge:dev` (5.7 GB) and the two EndoNav images (1.3 GB each) when they are missing.
2. It starts `ros2-bridge`, `noise-injection`, `openng3-sim` and `monadnock-sim`.
3. It compiles `bridge` with colcon and launches `full_system.launch.py` headless.
4. It waits until `/ng3/joint_states` is published, then re-zeroes the EndoNav follower (a zero latched before feedback pins the page at absolute positions), serves the pages on ports 8000, 8001 and 8002 (scenario files), and opens a fresh physics page (a page left from an earlier run is closed, never reused).
5. It checks the page end to end: all four bridge sockets (18765-18768) held by our Chrome only, a checked coordinated reset that leaves the page guidewire at 0 mm, a 6 s push at 5 mm/s that the page must follow, and a reset back home. On failure it re-zeroes the follower and reloads the page, then starts a new Chrome, and exits 1 if the page still does not follow. A page in another browser (for example an Edge tab) is reported by name and not repaired: close that tab.

Other commands: `status`, `test` (5 mm/s on module 1), `rebaseline`, `stop` (also closes the physics-page Chrome).

`start` now also starts their reset coordinator (`/hcmr/episode/reset`, from `HCMR-EV-Integration`)
and runs the physics page in headless Chrome with the flags of their RL renderer
(`stroke_nav.organiser.live.renderer.PhysicsPage`). Without a `scenario_url` the page loads the
cas-reference scenario by itself (`meshes/Mesh_1`). `reset` calls the coordinated reset.

**Only one physics page may be open.** The bridge's command node sends commands to the newest
browser connection only (`simulation_command_node.py` keeps a single websocket). A second tab with
`catheter_guidewire_ros2.html`, for example one opened earlier in Edge, takes over the commands
whenever it reconnects, and it also publishes its own device states. `status` lists how many pages
are connected; close any extra tab. The probe and the relay lock onto the page that restarted at the
last reset, and `scripts/run_astar.py` reloads the headless page before each episode and stops if the
page no longer follows the robot.

**Problems found and how they were fixed**

| Symptom | Cause | Fix |
|---|---|---|
| Docker Desktop would not start (`sailor-ingest.sock … cannot be accessed`) | stale socket stubs from crashes in Docker's runtime folder | moved `%LOCALAPPDATA%\Docker\run` aside (`run.stale-20261022`); Docker recreates it |
| First image build aborted | Docker Desktop auto-updated (29.7.2 → 29.9.2) and restarted mid-build | rebuilt; avoid updating Docker during builds |
| `udp_to_ros_node: received 126 bytes, expected 66` flood, no joint feedback | the Monadnock tar in the zip is a legacy 4-module build; the current OpenNG3 bridge only accepts 2 modules | run it with `SIM_MODULE_COUNT=2` (`HCMR-EV\docker-compose.local-2module.yml`, created by the script). The proper fix is the organiser's current `monadnock-model-linux-amd64.tar`, which was not included |
| Browser devices pinned at ~1140/840 mm, not following commands | the follower latched its zero before valid joint feedback arrived, so it sent absolute positions | re-zero it via `/endovascular/follower/enabled` (false, then true), then reload the page. Their reset coordinator normally does this |
| `ros2 topic echo` shows "A message was lost" or nothing | `/ng3/joint_states` runs at about 2 kHz, and cross-container matching takes a few seconds | use `--qos-reliability best_effort` and windows of at least 10 s |

| Physics page could not load its mesh (404) | `web/meshes` in the extract are git symlink placeholders (text files) on Windows | `start` replaces them with directory junctions to `HCMR-EndoNavSimDocker\meshes` |
| Physics stalls when the page is not in front | the XPBD loop runs on `requestAnimationFrame`, which browsers pause in hidden tabs | run it headless with throttling disabled (`PhysicsPage`) |
| Devices stop following mid-run although the robot moves | a second page took the bridge's single command connection | close the other tab; see above |

**Not done yet:** `./sys startup --mode rl|showcase|vla` (needs a Linux shell with Docker, i.e. a WSL Ubuntu
with Docker Desktop's WSL integration, plus their RL image with CUDA PyTorch) and the LeRobot recorder.

## 12. Our simulator vs theirs (measured 2026-10-22)

`scripts/measure_gap.py` plays the same actions in both simulators (their side through
`stroke_nav.organiser.live.probe`, which runs a ROS 2 script in the bridge container) and writes
`results/gap/report.md` with plots. `--sweep` adds the roll-to-branch sweep.

| Quantity | Theirs (XPBD rods) | Ours (kinematic) |
|---|---|---|
| Insertion vs command | follows, guidewire 5-8 mm behind, catheter 1-2 mm | exact |
| Guidewire travel | stops at 264.5 mm (module 1 from 1.200 to 1.4645 m) | now limited the same way (`RobotSpec`) |
| Catheter travel | module 2 stays 70.8 mm behind module 1, so catheter insertion <= wire + 229.2 mm | now limited the same way |
| Tip position | 2-5 mm off the centerline, against the wall | tip end kept within 0.7 x radius (was a rigid 30 mm segment) |
| Branch at the arch with roll 0 | left subclavian, then left vertebral | left carotid |
| Roll response | follows the command, with sudden whips of up to ~120 deg as stored twist releases | exact |
| Contact force | below 0.01 N in all runs | none |
| Simulated time | about 0.4 x real time in headless Chrome | n/a |

Roll sweep on their side (wire turned at the entry, then pushed 260 mm): 0-30 deg reaches the left
subclavian and vertebral; 60-120 deg the right carotid side (90-120 deg into the right external
carotid); 150 deg the right subclavian; 180-330 deg the left carotid or vertebral.

**Reachability.** The scenario goal (right external carotid, 21 mm in) is 288 mm along the
centerlines from the entry, more than the 264.5 mm of guidewire travel, so it is out of reach in our
kinematic simulator. On their rod physics it is reachable: the rod cuts corners (0.92 mm of route per
mm inserted), and in the roll sweep the wire turned to 120° at the entry and pushed 260 mm came within
2.8 mm of the goal and ended 4.0 mm from it (90°: 3.5 mm). Their SAC pool's 95 training successes agree.

**Entry roll decides the branch.** Pushing after a single roll at the entry already passes within
5 mm of right carotid @10-80 and right external carotid @10-20 (60-120°), left carotid @10-40
(180-330°), left subclavian @10-50 and left vertebral @10-30 (0-30°, 270°). Re-steering at each
bifurcation from the measured tip direction (the A* follower below) does not work there.

**Roll sweep in lockstep (2026-10-23, `scripts/calibrate_rolls.py`).** The same sweep on their
physics in the lockstep env (0-345° every 15°, roll then push to the end of travel, two noise seeds).
The seeds agree within 0.3 mm almost everywhere, so the outcome is repeatable. Final branch:

| Entry roll | Ends in |
|---|---|
| 0° | left subclavian |
| 15-30° | left vertebral (45°: one seed left vertebral, one left carotid) |
| 60-75°, 300-345° | left carotid |
| 90-150° | right carotid, passing the right external carotid |
| 165-285° | aorta |

180-240° stays in the aorta here, but entered the left carotid in the live sweep above. There, the
rod lagged the wall-clock robot (0.18× real time), so lockstep is the reference. The
cas-reference goal lies about 3.1 mm off every 90-150° path, against the wall.

**Real-time speed on this laptop (2026-10-23).** With their whole stack running, the physics solve
takes about 65 ms per frame (11 frames/s, 0.18× real time) and states and images arrive at 3.3 Hz.
Their env's readiness check needs at least 4.5 Hz of images, so `OpenNG3Env` refuses to start. Their
ROS nodes each spin near one core (the joint-state follower, both command nodes, the reset
coordinator), driven by the ~2 kHz `/ng3/joint_states`. With the containers paused the solve takes
28 ms (28 frames/s). `RLRenderer` (`organiser.live.renderer`) starts their RL renderer the way
`headless-renderer` does: asset server, then the physics page and noisy viewer in two headless Chromes.

## 13. A* on both simulators

`scripts/run_astar.py` plans a route with `PathPlanner` and drives it with
`stroke_nav.agents.route_follower.ClosedLoopRouteFollower`, which uses only the measured tip
position and tip direction: it rotates the wire until its bend points into the planned child vessel,
retracts and retries with another roll offset when the tip enters the wrong vessel, keeps the catheter
40 mm behind and stops at the goal. `--sim ours` runs our simulator, `--sim theirs` runs their live
stack through `stroke_nav.organiser.live.bridge.LiveBridge` (a JSON-lines relay into the bridge
container). Device geometry comes from `device-parameters.json`.

Status: on our simulator with the organiser's device parameters, 6 of 8 targets succeed (left
carotid targets fail on a false wrong-vessel alarm in the crowded arch). On their simulator a first run
to the right subclavian passed two of three bifurcations before the 600-step limit, slowed by the
second browser tab taking the command channel. Open-loop replay of our planned trajectory does not
work there: their rod picks different branches than our tip model.

## 14. Lockstep physics, scripted expert and behaviour cloning (2026-10-23)

**Lockstep env** (`stroke_nav.organiser.live.lockstep.make_env`). Their physics page and noisy viewer
run in headless Chrome with `lockstep_page.js` injected: we run their animation frames ourselves
(1/60 s of simulated time each) and speak their ROS-bridge WebSocket protocol in-page, so no Docker,
ROS or wall clock is involved. Their `OpenNG3Env`, observation, reward, success rule and goal come
from their `openng3_rl` sources. The Monadnock robot (50 ms delay, deadband, travel limits) and their
noise node (noise, 7 % tip dropout, 50-150 ms latency) are emulated. One control step takes about
0.25 s on this laptop (0.8x real time per worker; 4 workers in parallel). Stop their Docker stack
first so the pages get the CPU. Each worker gets its own page and asset-server ports
(8100 + 10·worker, 8102 + 10·worker): their asset server keeps a single authoritative page session,
and workers sharing one got each other's mesh-to-world transform, i.e. wrong goals.

**Scripted expert** (`stroke_nav.agents.entry_roll.EntryRollExpert`). It uses only the organiser
observation: it turns the guidewire to 120° at the entry, pushes it at full speed, slows inside 25 mm
of the goal and holds inside 2.5 mm. The catheter stays at home. In lockstep it reaches the
cas-reference goal (right external carotid) in 151-165 steps (median 153, about 31 s), every time.

**Behaviour cloning.**

```bash
.venv\Scripts\python.exe scripts/record_demos.py --episodes 60 --rolls 120 --sigmas 0,0.2,0.4 --workers 4 --max-steps 300
.venv\Scripts\python.exe scripts/train_bc.py --demos runs/demos/cas-reference --epochs 40 --device cuda --run-id stroke-nav-bc-v1
.venv\Scripts\python.exe scripts/eval_lockstep.py --model runs/stroke-nav-bc-v1/final-model.zip --episodes 8 --workers 4
```

- `record_demos.py` adds DART noise to the executed action (σ 0, 0.2, 0.4 on the guidewire) but keeps
  the expert's action as the label. 60 of 60 demos succeeded (18.6 min, 10,099 samples, 5.4 MB).
- `train_bc.py` builds the ordinary SB3 SAC model with `FluoroStateExtractor` and regresses the
  actor's deterministic action onto the labels. It also warm-starts both critics on the Monte-Carlo
  return, so SAC fine-tuning can start from it. 40 epochs take 30 s on the RTX 3060.
- `eval_lockstep.py` runs a model deterministically, as their `evaluation.py` does.
- `demo_lockstep.py` shows one episode in two visible Chrome windows (their physics page and their
  noisy viewer) and records a video: their physics view next to the policy's fluoro input.

| Policy (lockstep, their physics) | Episodes | Success | Time to goal | Final distance | Peak force |
|---|---|---|---|---|---|
| `EntryRollExpert` (σ = 0 demos) | 20 | 100 % | 30.6 s | 4.0 mm | < 0.01 N |
| BC policy `stroke-nav-bc-v1` (unseen seeds) | 8 | 100 % | 30.4 s | 4.0 mm | 0.009 N |

`check_handover.py` passes on the run folder. In our kinematic simulator the same policy fails: the
goal is beyond our guidewire travel (§12), and our simulator picks a different branch at the arch.

**Limits.** One goal (their scenario has one). The robot and noise are emulated from their sources,
not their containers. The policy has not yet been run on their wall-clock stack, which is too slow on
this laptop (§12).
