# Safety layer

A runtime safety monitor, tracking fault injection and safety metrics for any navigation policy in
this repo (planner follower, scripted agents, trained SB3 policies). Code: `src/stroke_nav/safety/`,
tests: `tests/test_safety.py`, evaluation: `scripts/evaluate_safety.py`.

## 1. Idea: doer and checker

A learned navigation policy cannot be verified the way classic safety software is (IEC 62304 class C,
ISO 26262 ASIL D). The standard answer, used in automotive (ISO/PAS 8800 "safety cage") and robotics
("simplex architecture"), is to keep the learned part as the **doer** and put a small, deterministic
**checker** between it and the actuators:

```
policy (RL / VLA / follower / assist)  ──action──▶  SafetyMonitor  ──filtered action──▶  robot / simulator
                                                        ▲
        encoders (insertion, roll) · tracked tip + dropout flag · blocked push / force · approved plan
```

The monitor may only make a command **less** aggressive: clamp a speed, replace an advance by a hold
or a bounded back-off, or hold all axes. It never adds motion the policy did not ask for (property
test `test_monitor_never_adds_motion`). It reads only what a real system measures, never simulator
ground truth, so the same logic can run as a ROS 2 node or be ported to embedded C.

## 2. Hazards and controls

| Hazard | Clinical harm | Cause in an autonomous system | Control | Verified by |
|---|---|---|---|---|
| Deep entry into a wrong vessel | dissection, embolism into the wrong territory | wrong roll at a bifurcation, seen late because of tracking latency | **R3 corridor** (no deeper while > 4 mm off the approved path) + **R1 junction slow zone** (bounds depth before the error becomes visible) | `test_r3_*`, `test_shield_reduces_hazards_*`, evaluation |
| Pushing against a wall or dead end | perforation, prolapse (wire buckles and snaps back) | policy keeps pushing when the device stalls | **R2 contact back-off** (blocked push or tip force > 0.8 N: back off for 2 steps) | `test_r2_*` |
| Too fast in small vessels | vessel injury | policy optimised for time | **R1 speed envelope** by lumen radius: 100 % ≥ 5 mm, 50 % ≥ 2 mm, 25 % below | `test_r1_*` |
| Acting on lost or wrong tracking | any of the above, blind | motion blur, dropout bursts, implausible jumps | **R4** hold after 2 dropouts; reject frames that moved further than the encoders allow | `test_r4_*` incl. regression |
| Overshooting the target | distal vessel injury | no stopping criterion | **R5 insertion fence** (target arc length + 10 mm) | `test_r5_*` |
| Catheter advanced without a leading wire | vessel injury by the stiffer catheter tip | uncoordinated two-device control | **R6** catheter may not overtake the wire tip | `test_r6_*` |
| Operator cannot take over | delayed reaction to a complication | autonomy holds the actuators | **Override**: a non-zero operator command always replaces the policy; only R1 still applies | `test_operator_override_*` |

Thresholds are engineering placeholders for the demo. None is clinically validated. The START position
statement says that a validated force threshold does not exist yet (Robertshaw et al., JAHA 2026).

## 3. Fault injection

`FaultModel` turns true tracking points into delivered measurements:

| Level | Noise | Dropouts | Latency | Purpose |
|---|---|---|---|---|
| `clean` | 0 | 0 | 0 | baseline |
| `organiser` | 0.1 mm | 7 %, single frames | 1 step | the organiser's `noise_params.yaml` |
| `stress` | 1 mm | 8 % bursts of 1–5 frames, +25 % when the tip moves > 25 mm/s | 1 step | motion blur as in Eyberg et al. 2022 |

The policy sees the corrupted observation (held samples during dropouts, as in the organiser's stack).
The monitor sees the same corrupted measurement. Evaluation statistics use ground truth.

## 4. Results (synthetic arch tree, 20 paired episodes per column)

Run offline with `python scripts/evaluate_safety.py --synthetic --episodes 20 --policies follower,sloppy,forward`.
`sloppy` is the A* follower at full speed with roll errors, a stand-in for an imperfect learned policy.

| Policy · faults | Monitor | Success | Time (s) | Deepest wrong-branch entry (mm) | Blocked pushes (eps.) | Overshoot > 10 mm (eps.) |
|---|---|---|---|---|---|---|
| follower · clean | off / on | 100 / 100 % | 14.1 / 19.9 | 0 / 0 | 0 / 0 % | 0 / 0 % |
| follower · stress | off / on | 100 / 100 % | 14.1 / 26.0 | 0 / 4.5 | 0 / 0 % | 0 / 0 % |
| sloppy · clean | off / on | 60 / 100 % | 16.1 / 21.0 | 4.8 / 4.3 | 0 / 0 % | 40 / 0 % |
| sloppy · stress | off / on | 60 / 100 % | 16.1 / 27.2 | 4.8 / 4.5 | 0 / 0 % | 40 / 0 % |
| forward (push only) · clean | off / on | 0 / 0 % | – | 93.1 / 5.9 | 100 / 0 % | 0 / 0 % |
| forward · stress | off / on | 0 / 0 % | – | 93.1 / 10.7 | 100 / 0 % | 0 / 0 % |

What this shows:

1. **No false harm on a good controller.** The follower keeps 100 % success under every fault level. The cost is time (about +40 % clean, +85 % under stress), mostly from the speed envelope and from holding during tracking loss.
2. **An imperfect controller gets safer and more successful.** The sloppy policy overshoots the target in 40 % of episodes without the monitor and never with it.
3. **A blind controller is contained.** Pushing straight ahead goes 93 mm into a wrong vessel and pushes against its end in every episode. With the monitor it stops after 6–11 mm and never pushes against a wall.

## 5. Two defects found by testing (and fixed)

- **The safety function itself deadlocked the robot.** The first plausibility check compared each new
  tip position with the last accepted one using a one-step limit. After a dropout burst during motion,
  every new frame looked like a jump, was rejected, and the monitor held forever (follower under stress:
  0 % success). Fix: bound the allowed tip travel by the encoder travel since the last accepted frame.
  Regression tests: `test_r4_recovers_after_a_dropout_burst_while_moving`,
  `test_shield_does_not_deadlock_follower_under_stress`. Lesson: availability failures of a safety
  function are hazards too (a stuck robot during a stroke intervention costs brain tissue).
- **The corridor alone does not bound wrong-branch depth.** With one step of latency at 40 mm/s, the
  wire travels about 5 mm per step before an error is visible, so blind pushing still reached 15–21 mm.
  The junction slow zone in R1 (30 % of v_max from 8 mm before to 12 mm after each planned bifurcation)
  halved that to 9–11 mm. The depth bound is roughly `corridor + (latency + 1) · v_junction · dt`. This
  is the quantitative argument for the "selection mode" on the navigation sketch.

## 6. How it maps onto the organiser's stack

See ORGANISER_SETUP.md for their data flow. The monitor belongs **between the policy and the arbiter**:

```
openng3_rl policy ─▶ /control/autonomous/module_n/cmd_vel ─▶ [safety node] ─▶ arbiter ─▶ robot
```

| Monitor input | Organiser topic |
|---|---|
| encoders | `/ng3/joint_states` (m, rad) |
| tracked tip + dropout flag | `/endovascular/<dev>/state/noisy` (`NoisyDeviceState`, dropout flags) |
| tip force (R2) | `DeviceState` tip / peak force (their "unsafe" episode limit is 3 N; the monitor stops at 0.8 N) |
| approved plan | `plan.json` / `trajectory.csv` from `scripts/plan_path.py` |

Open points to check in their code before relying on this:

- **Operator priority.** Their documentation says the arbiter gives a module to "the first non-zero
  source" (0.35 s timeout). If that holds literally, a physician on teleoperation could not take over
  while the policy is sending commands. A safety-critical arbiter needs a fixed priority: operator >
  monitor stop > policy.
- **Three speed limits.** RL action ±10 mm/s, bridge clamp ±20 mm/s, simulator default 45 mm/s. The
  effective limit depends on who drives. One envelope, enforced in one place, is easier to argue.
- **Force units and gain.** The simulator UI has a force scale (seen at 80×) that multiplies every
  force published to ROS. Recorded forces are only usable if that gain is stored with the data.

## 7. Wall crossings on the organiser's physics (2026-10-23)

**What happened.** In the femoral-to-carotid video (`0028_H_ABAO_H`, long-reach robot) the guidewire
entered the left common carotid *through the aortic arch wall* in the last 2 s, and the episode counted
as a success. Peak force stayed at 1-2 mN, so neither the force readings nor their success rule noticed.

**Cause.** Their page (`projectContainment` in `catheter_guidewire_ros2.html`) keeps only the wire's
*bodies* inside the vessel, 7.5 mm apart (1.2 m / 160 bodies, the page's maximum), by pushing each body
towards its *nearest* wall, capped at half a body spacing per pass. Where two lumens run a few mm apart
(arch and left CCA, femoral and profunda), a pushed body slips across the wall, finds the other lumen's
wall nearest and counts as inside; the wire between bodies is never checked. Pushing open-loop after the
entry roll, as the A* + roll controller and its clones do, drives the wire into exactly these places.
Slowing the push (3 mm/s) did not help; more collision passes (`collIters` 6) changed the path and missed.

**How often.** Re-checked with the guard below, one episode each, A* + roll without the safety layer:

| Case | Their success rule | Wall crossed | Deepest |
|---|---|---|---|
| 0028 femoral to left CCA@30 (long-reach) | reached | 6 steps at the end (arch into the CCA) | 2.2 mm |
| 0248 cas-reference (behind the hand-over BC policy) | reached | 35 steps, in the arch at the left CCA origin while crossing to the right | 3.6 mm |
| 0248 `10Left100Right` right ICA occlusion | reached | 19 steps | 3.1 mm |
| 0248 `10Right70Left` left ICA lesion (deployment bundle) | reached | 2 steps | 1.2 mm |

So every "success" of the entry-roll technique on their physics so far passed through a wall at some point.

**Controls** (`src/stroke_nav/safety/wall.py`, tests `tests/test_wall.py`):

- **`WallGuard`** checks what their page draws, a Catmull-Rom curve through the inserted bodies (every
  0.5 mm), against the scenario's vessel surface by signed distance (5 ms per check). A stretch beyond the
  wall counts as a crossing when it joins two lumens far apart along the vessel tree (*shortcut*: tree
  distance > 2 x wire length + 10 mm, at any depth, since a thin wall allows only ~0.5 mm) or goes deeper
  than 1.5 mm (*deep*). Same-vessel bows of the curve in a bend (up to ~1.1 mm in the femoral) are not crossings.
- **Evaluation**: `run_lockstep_episode` checks the true wire every step; `assess` adds a rejecting
  `wall` flag and counts only `safe_success` (reached without crossing). Case reports reject the runs above.
- **Calibration**: `calibrate_rolls.py` records each path's first crossing; `RollTable` uses a path only
  up to it, so a roll is never chosen because its path reached the target through a wall.
- **Runtime safety layer** `WallSafeController` around any controller (on by default in `plan_case.py`,
  `--no-wall-safety` to compare): on a crossing it stops pushing and retracts 15 mm past the crossing,
  re-rolls (+20, -20, +40, -40 deg), creeps forward at 30 %, and aborts and holds after the last retry. It
  reads the wire's noisy segment poses (their `/endovascular/guidewire/state/noisy`) and the pre-operative
  vessel surface. It reacts one control step after the crossing appears (2-3 steps per incident).

**Result with the layer** (one episode each, A* + roll at the table's roll): the wire no longer pushes
on through a wall, but at the arch every retry crossed again, so each case ends safely without reaching its
target instead of reaching it through the wall:

| Case | Outcome with the safety layer |
|---|---|
| 0028 femoral to left CCA@30 | 3 retries at the arch crossed again; 600-step limit, 14.7 mm short |
| 0248 cas-reference | crossing at 134 mm in the arch, 4 retries crossed again, abort at 109 mm |
| 0248 `10Right70Left` left ICA lesion | crossing at 166 mm, 4 retries crossed again, abort at 128 mm |
 Reaching these targets safely needs a different technique, e.g. advancing
the catheter behind the wire through the arch for support, or selective cannulation with the shaped
catheter tip, not only the entry roll.

## 8. Device removal (2026-10-23)

A procedure ends with the devices coming out. `stroke_nav.agents.removal.WithdrawalController` withdraws
the guidewire from their observation alone; `scripts/removal_trials.py` inserts with the A* + roll
controller to a depth before its first wall crossing, holds 1 s and compares strategies from that state,
with the wall guard on the true wire every step (one episode each).

| Hazard on the way out | Strategy tried |
|---|---|
| tensioned wire cuts the inner curve of a bend | slow pull; wall-guarded pull (stop, re-advance 3 mm, re-roll, half speed) |
| stored twist from the entry roll whips the tip | turn the shaft back to its starting roll first (`unwind`) |
| bent tip snags in a branch opening | speed by lumen radius (the monitor's R1 envelope along the A* route) |
| catheter and wire dragging each other | catheter out first over the wire (not exercised: the controllers keep the catheter at home) |

`0028_H_ABAO_H` from the right femoral end, long-reach robot, withdrawing 500 mm:

| Strategy | Time | Wall crossed while withdrawing | Deepest | Peak force | Fastest tip |
|---|---|---|---|---|---|
| naive (10 mm/s, roll kept at 120 deg) | 50.6 s | 33 steps | 5.1 mm | 6.0 mN | 40 mm/s |
| slow (3 mm/s) | 167.4 s | 109 steps | 5.1 mm | 6.2 mN | 42 mm/s |
| profiled (lumen-radius envelope) | 75.0 s | 64 steps | 5.0 mm | 6.1 mN | 41 mm/s |
| guarded | 65.0 s | 5 steps, handled | 1.6 mm | 2.2 mN | 22 mm/s |
| **unwind** (shaft back to 0 deg, then 10 mm/s) | **53.4 s** | **0** | 0.2 mm | 2.2 mN | 17 mm/s |
| unwind + profiled + guarded | 78.0 s | 0 | 0.2 mm | 2.4 mN | 16 mm/s |

Every crossing happened in the last 80 mm (femoral, beside the profunda): with the shaft still turned
120 deg the pre-bent tip points sideways, hooks into the profunda opening and is dragged through the wall,
snapping free at 40 mm/s. Turned back to its starting roll, the tip lines up with its way in and slides out.
Slowing down only keeps it hooked longer, as when pushing. On cas-reference (0248, organiser robot, 120 mm
of straight descending aorta) no strategy crossed; unwinding still lowered the fastest tip motion from 35 to
22 mm/s for 3.6 s more.

**Recommended removal:** `unwind_guarded`: turn the shaft back to its starting roll, pull at full speed,
with the wall guard as the backstop.

```bash
set STROKE_NAV_ROBOT=long-reach
python scripts/removal_trials.py --model 0028_H_ABAO_H --target Left_Common_Carotid@30 --roll 120 --insert-to 500
python scripts/removal_trials.py --target scenario --roll 120 --insert-to 120 --strategies naive,unwind,unwind_guarded
```

## 9. Limitations and next steps

- The simulator is kinematic: R2 reacts to blocked pushes only. With the organiser's XPBD forces, R2 can
  use real tip force; thresholds then need calibration (and units checked, see above).
- Thresholds (speeds, corridor, overshoot) are placeholders. A real safety case needs them derived from
  bench and in-vitro data (START: "correlate in vitro measurements to in vivo complications").
- Python is fine for the demo. A product monitor would be a separate process or MCU with bounded
  runtime, freedom from interference from the policy, and its own watchdog (ISO 26262-6 / IEC 62304 practice).
- Next: train a policy **with the monitor in the loop** (wrap the env in `SafetyShield` during SB3
  training) so the policy learns not to rely on interventions, and compare interventions per episode.
