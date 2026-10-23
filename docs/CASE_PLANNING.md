# Pre-operative case planning

The revised requirement (2026-10-23): the system is not a generic policy for a generic target. For
**one patient** (a CTA segmentation as a vessel mesh) and **one target** (the lesion), it should plan the
path the robot will take from the entry to the target, offer **several proposals from different
methods** so that safety constraints or a clinician can choose between them, and say how long each
method needs. The robot, the device parameters and the organiser's physics are fixed and known.
Overfitting to the case is the point, and training time is a budget, not a cost to minimise blindly.

`scripts/plan_case.py` is that system. Code: `stroke_nav.planning.candidates` (common trajectory
format, safety checks, reports), `stroke_nav.agents.act` (action chunking), the existing planner,
expert, behaviour cloning and lockstep env.

## 1. What a case run does

```
vessel model + target ──▶ A* route, reachability ──▶ one controller per method ──▶ rollouts on their physics
                                                                                  │
                        summary.md / summary.json / candidates.png  ◀── safety checks ◀── candidates/<method>.json
```

1. **Case.** The target spec (`branch@mm`, `xyz:x,y,z` in VMR mm, a node id, or `scenario` for the
   organiser's goal) is resolved on the centerline graph and converted to the organiser's world frame
   with the fitted transform. The constrained A\* route gives the geometric reference, its length is
   checked against the robot's guidewire travel (264.5 mm), and the entry-roll table says whether the
   calibrated rod paths pass within 5 mm of the target on their physics. `case.json` records all of it;
   `goals.json` holds the target as a one-goal set for `record_demos.py`, `eval_lockstep.py` and
   `demo_lockstep.py` (`--goals <case>/goals.json --split train`).
2. **Methods**, each under its own budget (`--methods`, default `astar,astar_theirs,roll,bc,act`):

   | Method | What it is | Train time | Plan time | Runs on | Hand-over format |
   |---|---|---|---|---|---|
   | `astar` | constrained A\* on the centerline graph + closed-loop route follower | none | seconds | our kinematic simulator | route + trajectory CSV |
   | `astar_theirs` | the same follower steering on the organiser's noisy tip state | none | minutes (rollouts) | their physics (lockstep) | trajectory CSV |
   | `roll` | entry-roll controller: the table's roll at the entry, then push | one roll sweep per anatomy (`calibrate_rolls.py`, ~15 min, done once) | minutes | their physics | trajectory CSV; the controller is 10 lines |
   | `bc` | expert demonstrations at this target, cloned into the organiser's SAC model | `--bc-minutes` of demonstrations + ~30 s fitting | minutes | their physics | SB3 SAC zip, loads in their stack |
   | `act` | action-chunking transformer on the same demonstrations | `--act-minutes` | minutes | their physics | torch checkpoint (trajectory only) |
   | `sac` | SAC fine-tuning from the BC model on this target | `--sac-minutes` | minutes | their physics | SB3 SAC zip |

   Demonstrations are recorded with the entry-roll expert and DART noise (σ 0, 0.2, 0.4 on the
   guidewire) until the budget or `--bc-demos` is reached; `bc`, `act` and `sac` share them.
3. **Common format.** Every method's result is a `Candidate` (`candidates/<method>.json`): the method's
   train and plan wall time and one `EpisodeTrace` per rollout, as the organiser's stack would log it:
   true guidewire tip after each 5 Hz step (world metres), observed insertion, the executed action,
   true distance to the goal, peak force, outcome. `trajectory_<method>.csv` is the representative
   episode (fastest success, else closest miss) in both frames, for a safety layer or a reviewer.
4. **Safety assessment** (`candidates.assess`), deterministic, on the patient's anatomy, with the
   thresholds of `stroke_nav.safety.SafetyConfig` (placeholders, see [SAFETY.md](SAFETY.md)):

   | Flag | Measured | Fires when |
   |---|---|---|
   | `lumen` | tip distance outside the nearest vessel's lumen | > 4 mm |
   | `wrong_vessel` | depth along a vessel off the A\* route while the tip is outside the route vessel's lumen by > 4 mm for ≥ 2 consecutive steps | > 12 mm (the junction zone) |
   | `overshoot` | insertion beyond the closest approach to the goal | > 10 mm |
   | `force` | peak contact force (their physics) | > 0.8 N |
   | `travel` | max insertion | > 264.5 mm |
   | `stall` | insertion gained while the tip's net displacement stays below 20 % of it (the wire buckles or pushes against a wall) | > 30 mm (successful runs on their rod show 16-23 mm of slack in the arch) |
   | `speed` (advisory) | push command above the lumen-radius envelope (100 % ≥ 5 mm, 50 % ≥ 2 mm, 25 % below) | any step |

   Verdict per candidate: `reject` (never reached the target), `review` (a structural flag fired), `pass`.
   `speed` is reported but does not change the verdict: the expert pushes at full speed everywhere, so it
   fires on every candidate recorded on their physics; the envelope is a placeholder to calibrate.
   Branch centerlines start inside their parent vessel, so the depth test is what separates a tip hugging
   the arch wall near a side branch's first nodes from a real entry into that branch.
5. **Reports.** `summary.md` (the table below), `summary.json`, `candidates.png` (AP and lateral views of
   every representative path over the centerlines, with the route and the target).

## 2. Running it

```bash
powershell -ExecutionPolicy Bypass -File tools\organiser_stack.ps1 stop      # their Docker stack would take the CPU
.venv\Scripts\python.exe scripts/plan_case.py --target right_carotid@40 --case-id rca40
.venv\Scripts\python.exe scripts/plan_case.py --target xyz:-78.0,118.5,-274.0 --methods astar,roll,bc --bc-minutes 5 --eval-episodes 4
.venv\Scripts\python.exe scripts/plan_case.py --target scenario --methods astar,astar_theirs,roll,bc,act,sac --sac-minutes 20
```

- Budgets: `--bc-demos 24 --bc-minutes 10` (demonstrations), `--act-minutes 3`, `--sac-minutes 15`;
  `--eval-episodes 6` rollouts per method on their physics, `--max-steps 300`, `--workers 4`.
- `--roll` overrides the table's entry roll; `--table` names other sweep files.
- Stray headless Chromes from a crashed lockstep run are killed at the start (`--keep-stray` to skip).
  The run refuses to start while another lockstep script is running anywhere on the machine: all
  jobs share the worker ports and Chrome debug ports, and cleaning up one job's pages kills the other's.
- `--resume` reuses the candidates and demonstrations already in the case folder, so a crashed or
  interrupted run continues where it stopped and a method can be added later without rerunning the rest.
- A page that dies mid-episode (WinError 10054) is restarted and the episode retried; a worker whose
  pages do not come up is dropped and the others take its episodes.
- Watch a candidate: `demo_lockstep.py --model <case>/runs/bc/final-model.zip --goals <case>/goals.json --split train`.
- Hand over the BC or SAC model: `check_handover.py --run-dir <case>/runs/bc`, then `export_handover.py`.

The case folder lives under `results/cases/` (ignored by git, like all results).

## 3. Results

### Case `cas-reference` (the organiser's goal, right external carotid), 2026-10-23

`python scripts/plan_case.py --target scenario --case-id cas-reference` with the default budgets
(24 demonstrations within 10 min, ACT 3 min, 6 evaluation episodes per method, 4 workers). The whole
case took 19.7 min of wall time; the route is 288 mm against a guidewire travel of 264.5 mm.

| Method | Simulator | Train time | Plan time | Success | Time to goal | Final dist. | Peak force | Safety | Flags |
|---|---|---|---|---|---|---|---|---|---|
| astar | ours-kinematic | <1 s | 1 s | 0/1 | - | 23.2 mm | 0.0 N | reject | speed |
| astar_theirs | theirs-lockstep | <1 s | 5.1 min | 0/6 | - | 91.1 mm | 0.013 N | reject | speed, wrong_vessel |
| roll | theirs-lockstep | <1 s | 2.5 min | 6/6 | 29.0 s | 4.0 mm | 0.009 N | pass | speed |
| bc | theirs-lockstep | 9.8 min | 3.5 min | 6/6 | 29.4 s | 4.1 mm | 0.009 N | pass | speed |
| act | theirs-lockstep | 10.4 min | 2.8 min | 6/6 | 29.1 s | 4.1 mm | 0.009 N | pass | speed |

What the case says:

- **The classical planner is right about geometry and wrong about the rod.** A\* finds the route in
  under a second and correctly reports that the target lies beyond the robot's insertion travel on the
  centerlines. On their physics the same follower never gets past the arch (0/6, 40 mm into a wrong
  vessel), because their rod does not steer at bifurcations the way our tip model does.
- **The calibrated controller is the cheapest working proposal.** The entry-roll table costs one
  15-minute sweep per anatomy and nothing per target; 6/6 at 29 s.
- **Both learned candidates reproduce it within a 10-minute budget.** Behaviour cloning (24
  demonstrations, 23 s of fitting) and action chunking (80 epochs in 62 s) each reach 6/6 unseen
  seeds, with the same time to goal and the same safety profile. The BC model passes
  `check_handover.py` and can be handed over for this target. The budget was dominated by recording
  demonstrations (9.4 min for 24 episodes on 4 workers); fitting is seconds.
- **The robust roll matters more than the closest roll.** The table's roll with the smallest recorded
  miss was 150°, the edge of the 90–150° arc that reaches the external carotid; with it the controller
  failed 3 of 6 episodes (the rod took the descending aorta after 150 mm of insertion). The arc centre,
  120°, gives 6/6. `RollTable.reach(robust_within_mm=5)` now picks the centre; `case.json` records both.
- **Every physics candidate trips the speed envelope**, because the expert pushes at full speed in
  vessels under 5 mm radius. The flag is advisory; it is the first thing a safety layer would change
  (a slower push also changes which branch the rod takes, README §5).

Files: `results/cases/cas-reference/` (not in git).

### Case `rca40` (`right_carotid@40`), stopped before completion

Started as a harder target and stopped on request after the roll and BC stages. A\* reaches it on our
simulator (route 223 mm, 1/1) and misses by 41 mm on their rod (0/6). The table's roll (135°, arc of
120–135°) passes the target at 5–10 mm and pushes on to the end of travel (0/6 near misses); of 20
demonstrations in 12 min only 2 succeeded, and the clone fitted on them behaved like its teacher
(0/4 evaluated before the stop). The planner's verdict for this target is therefore `reject` for
every method so far: the entry-roll controller is not precise enough where the rod rides a few mm off
the goal, and a target-specific stop (slower push, or DAgger with the recovering expert) is the next
thing to try. Saved stages are in `results/cases/rca40/candidates/` and resume with `--resume`.

## 4. Replanning during the procedure

What exists: the A\* follower re-steers at every bifurcation from the measured tip and retracts and
retries when the tip enters a wrong vessel; the `RecoveringExpert` (DAgger teacher) pulls back and
re-rolls when the wire went in at the wrong roll; DAgger (`run_dagger.py`) teaches the cloned policy
those recoveries. All of this is closed-loop replanning of the *controller*.

What is missing: re-running the case planner from a mid-procedure state. The lockstep env always
resets to the home position, and their physics page has no way to inject a rod state, so a candidate
cannot be evaluated "from here". The practical route is to plan the remaining path geometrically
(A\* from the node nearest the current tip, which `PathPlanner.plan(start, target)` already accepts)
and to execute it with the follower or the expert, while learned candidates stay pre-operative.

## 5. Methods considered and not used

- **DAgger** is implemented (`run_dagger.py`) and lifted the multi-target clone; per case it adds rounds
  of rollouts and is a natural `--methods` extension when the BC candidate fails on a target.
- **ACT with the CVAE prior**: one target per case leaves no multimodality to model; the deterministic
  chunking policy is kept small so it fits a minutes-long budget on a 6 GB laptop GPU.
- **VLA / SmolVLA**: 450 M+ parameters, language conditioning and multi-camera inputs; fine-tuning takes
  hours on this hardware and the lockstep env delivers about 4 steps/s per worker, so a VLA could not
  be trained within a pre-operative budget here. The candidate interface (`reset()`, `act(obs)` on the
  organiser observation) is what such a policy would plug into; its output would be assessed like any
  other candidate.
