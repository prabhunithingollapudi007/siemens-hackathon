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

## 7. Limitations and next steps

- The simulator is kinematic: R2 reacts to blocked pushes only. With the organiser's XPBD forces, R2 can
  use real tip force; thresholds then need calibration (and units checked, see above).
- Thresholds (speeds, corridor, overshoot) are placeholders. A real safety case needs them derived from
  bench and in-vitro data (START: "correlate in vitro measurements to in vivo complications").
- Python is fine for the demo. A product monitor would be a separate process or MCU with bounded
  runtime, freedom from interference from the policy, and its own watchdog (ISO 26262-6 / IEC 62304 practice).
- Next: train a policy **with the monitor in the loop** (wrap the env in `SafetyShield` during SB3
  training) so the policy learns not to rely on interventions, and compare interventions per episode.
