# Constrained A\* with length, time and damage costs

The planner (`stroke_nav.planning.planner`) finds a device-feasible route on the patient's centerline
graph and judges it on the four things the clinicians asked for, before anything moves:

| Evaluation | Cost term in A\* | Reported as (`Plan.costs`, `evaluate_plan`) |
|---|---|---|
| shortest robot size | `length` per mm inserted | `length_mm`, within the robot's 264.5 mm travel or not |
| shortest reach time | `time` per second: length / (v_max · speed fraction), the safety speed envelope by lumen radius (100 % ≥ 5 mm, 50 % ≥ 2 mm, 25 % below) and a 30 % zone within 12 mm of every bifurcation | `est_time_s` |
| less damage to arteries | `curvature` (bending effort, (κ/κ_max)² per mm), `curvature_integral` (radians of turning), `narrow` (length in lumen tighter than comfortable), `force` (predicted contact force per mm: a fitted model, or the proxy bending × narrowness) | `bend_effort`, `curvature_integral_rad`, `narrow_mm`, `force_proxy`, tight bends, min lumen |
| avoid wrong arteries | `branch` (every vessel change) and `takeoff` (sharp take-off angles) make routes with fewer, gentler bifurcations cheaper; the route itself never enters a vessel it does not need | `bifurcations`, `vessel_changes`, `max_takeoff_deg`; after execution: the follower's wrong-vessel retries and the trajectory checks |

Hard constraints remove edges: lumen radius below the device's clearance, **forbidden vessels**
(`PlannerWeights.forbidden_branches`, e.g. collateral connectors), and optionally curvature above the
device limit. They hold under every weight profile.

`PlannerWeights.profile(name)` gives named trade-offs: `shortest` (length only), `fastest` (time),
`gentle` (damage terms dominate), `balanced`. The heuristic stays admissible (straight-line distance
scaled by the length and time weights), so A\* is optimal for the chosen weights; a plan on 0248 takes
10 to 40 ms and expands 500 to 1000 edge states.

## Warm-start / local replan

`PathPlanner.replan(current_tip, target, previous=plan)`:

- tip on the previous route: the remainder is returned without a search (`astar-resume`, about 1 ms);
- tip off it (a wrong vessel, a retraction): a **local** A\* that only expands nodes within 40 mm of the
  tip and closes as soon as it reaches a node of the previous route, whose remaining cost-to-go is
  known (`astar-warm`); `costs["rejoin_mm"]` is the distance travelled before rejoining, i.e. the way
  back out of the wrong vessel;
- nothing within the radius: a full plan from the tip.

On 0248, replans from 15 mm into the wrong arch vessel expanded 143 to 206 edge states in 6 to 10 ms,
against 620 to 670 for a cold plan from the same spot. From a tip left 22 mm inside the left
subclavian after a failed execution: 350 states against 1039.

## Lesion targets and the robot

`nearest_feasible_target` moves a target the device cannot fit through (a stenosis) to the nearest
feasible point proximal to it. The ICA lesions of the 0248 stenosis variants (`left_carotid@140`,
`right_carotid@118`) lie 287 and 301 mm along the centerlines from the aortic entrance, beyond the
264.5 mm guidewire travel; `evaluate_plan` reports this. On the organiser's rod the path is shorter
than the centerline (it cuts the arch), so the left lesion is reached at 262 mm of insertion while the
right one ends 9 mm short at full travel.

## Running the demo on the organiser's physics

`scripts/demo_astar_costs.py` plans each lesion target of the variants `10Right70Left` and
`25Right50Left` with the four profiles, evaluates the routes, executes them on the organiser's physics
and replans from where the rod ended.

```bash
powershell -ExecutionPolicy Bypass -File tools\organiser_stack.ps1 stop
python scripts/demo_astar_costs.py --visible                       # both variants, both lesions, their physics, Chrome windows
python scripts/demo_astar_costs.py --controller follower           # the A* route follower steering at bifurcations (fails on their rod)
python scripts/demo_astar_costs.py --sim ours                      # kinematic simulator, seconds
python scripts/demo_astar_costs.py --reuse-executed                # regenerate the reports from recorded executions
python scripts/demo_lockstep.py --variant 10Right70Left --target left_carotid@140 --expert --slow-radius 8 \
    --record results/astar_demo/video.mp4                          # a video: physics view | fluoro input
```

- **Their physics on another mesh.** `stroke_nav.organiser.scenario.write_scenario` decimates a VMR
  variant surface to their triangle count, converts it to their mesh frame with the fitted
  `OrganiserFrame`, writes `vessel.obj`, `goals.txt` and `scenario.json` under `data/scenarios/`, and
  validates the folder with their own loader. With the `authored` bake the mesh-to-world transform is
  a pure scale, so goals and the frame are unchanged; the variant bounds agree with their mesh to
  about a millimetre. `make_env(scenario_name=..., scenario_root=...)` serves it to the pages.
- **Execution.** Their rod chooses the branch by the roll set at the entry, and re-steering at a
  bifurcation releases stored twist (docs/ORGANISER_SETUP.md §12). So the route is executed by the
  calibrated entry-roll controller (`--controller roll`, the default): A\* gives the route, its
  evaluation and the stop; `RollTable.reach(goal, robust_within_mm=5)` gives the roll whose recorded
  paths pass the target and end in the route's final vessel. `--controller follower` runs the A\* route
  follower instead, which steers at each bifurcation from the measured tip; it worked on our kinematic
  simulator and fails on their rod.
- The start is theirs: the mesh's bake entrance, the open end of the descending aorta (femoral
  access), the same for every target. Their page lists other tube ends as start points, but the
  selector is disabled for `authored` scenarios.

## Results (2026-10-23, one episode per case, their physics)

Entry-roll execution of the A\* route (`results/astar_demo/`):

| Variant | Lesion | Route (A\*) | Length | Est. time | Roll | Outcome on their rod | Time | Final dist. | Peak force | Safety |
|---|---|---|---|---|---|---|---|---|---|---|
| 10Right70Left | left_carotid@140 (70 %, lumen 0.81 mm) | aorta > left_subclavian > left_carotid | 287 mm | 51 s | -15° | **reached**, 262 mm inserted | 27.0 s | 1.9 mm | 5 mN | pass (speed advisory) |
| 10Right70Left | right_carotid@118 (10 %) | aorta > right_subclavian > right_carotid | 301 mm | 62 s | 120° | full travel, 9 mm short | - | 9.4 mm | 10 mN | reject (travel) |
| 25Right50Left | left_carotid@140 (50 %) | aorta > left_subclavian > left_carotid | 287 mm | 51 s | -15° | wire stalled in the arch at 154 mm of tip progress while 110 mm more were inserted | - | 97 mm | 5 mN | reject (stall) |
| 25Right50Left | right_carotid@118 (25 %) | aorta > right_subclavian > right_carotid | 301 mm | 62 s | 120° | **reached**, 264 mm inserted | 29.8 s | 3.2 mm | 7 mN | pass (speed advisory) |

Videos: `results/astar_demo/video_10Right70Left_left_carotid_at_140_success.mp4` and
`..._right_carotid_at_118_travel_limit.mp4` (their physics view next to the policy's fluoro input,
2× real time).

**From the happy path to a learned policy** (`results/cases/v70-left/`, `plan_case.py --variant
10Right70Left --target left_carotid@140 --methods astar,roll,bc --bc-demos 12 --sigmas 0,0.2`): the A\*
route executed by the entry-roll controller recorded 12 demonstrations on their physics in 3.4 min,
all successful (6 with action noise 0.2, labels the controller's); behaviour cloning into the hand-over
SAC model took 20 s; the clone reached the lesion in 2 of 2 unseen episodes, 26.4 s, within 2 mm,
peak force 5 mN, and passes `check_handover.py`. Video:
`results/cases/v70-left/video_bc_policy_success.mp4`. DAgger (`run_dagger.py` with the recovering
expert) is the next step if a clone fails on a target.

A\* route follower execution (`results/astar_demo_follower/`): 0 of 4 on their rod, 76 to 125 mm short,
stuck in the wrong arch branch or at full travel. The pre-execution evaluation is the same for both
executions; what differs is whether the controller matches the physics.

What the numbers say about the four evaluations:

- **Robot size**: the planner's inserted length is an upper bound; the rod reaches a 287 mm route with
  262 mm of insertion, and a 301 mm route with 264.5 mm leaves the right lesion 9 mm short.
- **Reach time**: the estimate from the speed envelope (51 to 62 s) is pessimistic by half against the
  27 to 30 s the full-speed controller needs; the envelope is the safety layer's placeholder, not a
  measurement of their rod.
- **Damage**: peak contact forces on their physics stayed at 5 to 10 mN, two orders below the 0.8 N
  limit; bending effort and narrow length rank routes, and the stall check catches buckling.
- **Wrong arteries**: no wrong-vessel excursion in any successful run; the route follower's
  bifurcation steering produced them on the rod, the entry-roll execution did not.

Tests: `tests/test_planning.py` (profiles, time model, hard constraints under every profile, resume and
local replan, lesion target), `tests/test_scenario.py` (a variant scenario loads with their loader and
keeps the world frame).
