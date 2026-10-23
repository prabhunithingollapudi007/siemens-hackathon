# A new mesh and a new target: the recipe

What to do when a new 3D vessel mesh and a target site arrive (the hackathon's final test). The goal
is the same chain as on 0248: **A\* plans the route**, the **calibrated entry roll executes it** on the
organiser's physics, **demonstrations are recorded and behaviour-cloned** into their SAC model, and the
clone is checked on the same target. Every step below ran on a second patient model, `0024_H_AO_H`,
in about 20 minutes of wall time (section 3).

## 1. What the mesh must give us

| Need | From a VMR model | From a plain mesh (OBJ / STL) |
|---|---|---|
| centerlines with lumen radius (A\*, the follower, the safety checks) | `fetch_data.py` builds them from the model's `Paths` | not shipped; their physics page skeletonises the lumen on load (`mesh.centerlineCanon`, a TEASAR trace), which can be read out through the page's debugging socket and turned into our centerline graph. Not built yet; see section 4. |
| the surface for their physics | `data/raw/<model>/Models/*.vtp`, decimated to 10k triangles | the mesh itself |
| placement in their frame | `make_scenario.py` puts the entrance at the origin, vessel axis +Z | same, once centerlines exist |
| the target | a target spec on the centerlines, or `xyz:` in mm | `xyz:` in the mesh's units |

Their robot enters at the mesh origin along +Z (an *authored* scenario); the entrance must be an open
end of the vessel. Everything beyond the 264.5 mm guidewire travel is out of reach from there.

## 2. Commands

```bash
python scripts/fetch_data.py --model 0024_H_AO_H                                  # VMR model: download, centerlines, radius
python scripts/make_scenario.py --model 0024_H_AO_H --target LCCA@20 --target RSA@15   # frame + their scenario + route lengths
powershell -ExecutionPolicy Bypass -File tools\organiser_stack.ps1 stop            # their Docker stack would take the CPU
python scripts/calibrate_rolls.py --model 0024_H_AO_H --rolls 0:360:30 --seeds 1 --workers 4   # which entry roll ends in which vessel, ~5 min
python scripts/plan_case.py --model 0024_H_AO_H --target LCCA@20 --case-id 0024-lcca20 \
    --table results/lockstep/roll_sweep_0024_H_AO_H.json --methods astar,roll,bc --bc-demos 12 --sigmas 0,0.2 --eval-episodes 1
python scripts/demo_lockstep.py --vmr-model 0024_H_AO_H --target LCCA@20 --expert --slow-radius 8 --record results/cases/0024-lcca20/astar_roll.mp4
python scripts/demo_lockstep.py --vmr-model 0024_H_AO_H --target LCCA@20 --model results/cases/0024-lcca20/runs/bc/final-model.zip --record results/cases/0024-lcca20/bc.mp4
```

- `make_scenario.py` saves `data/processed/organiser_frame_<model>.json` (`frame_for(model)` loads it) and
  `data/scenarios/<model>/` (`vessel.obj`, `goals.txt`, `scenario.json`), validated with their loader,
  and prints each target's route length against the travel.
- `calibrate_rolls.py` records the rod's path for each entry roll on the new mesh; `RollTable.reach`
  then picks the roll whose paths pass the target (centre of the arc of rolls that do).
- `plan_case.py` writes `results/cases/<id>/` with the A\* evaluation, the demonstrations, the clone,
  its evaluation and `summary.md`; `check_handover.py --run-dir results/cases/<id>/runs/bc` checks the
  model for their stack.
- **Deploying to their stack**: copy `data/scenarios/<model>/` to `HCMR-EV-RL/scenarios/<model>/` and the
  run folder to `HCMR-EV-RL/runs/`, then from `HCMR-EV-OpenNG3`:
  `./sys startup --mode showcase --scenario <model> --run-id <run> --model final-model.zip --episodes 1 --visible`.
  The entry-roll controller is 10 lines on their observation (`agents/entry_roll.py`) and runs on their
  pages through `demo_lockstep.py`; packaging both into one bundle is the deployment export being
  built alongside (`export_deployment.py`).

## 3. Dry run on `0024_H_AO_H` (2026-10-23)

`0024_H_AO_H` is a healthy aortic arch with four branches (`LCCA`, `LSA`, `RCCA`, `RSA`), aorta 148 mm,
branches 19-27 mm, all within the robot's travel. One episode per row on their physics.

| Step | Wall time | Result |
|---|---|---|
| `make_scenario.py`: entry frame, decimated mesh (5002 vertices), their loader validates | 10 s | entrance band centroid at the origin, transform a pure 0.01 scale |
| straight push smoke test | 1 min | wire runs up the aorta into the right subclavian, tip inside the lumen, 0.2-0.5 mN |
| `calibrate_rolls.py`, 12 rolls × 1 seed | 4.4 min | 0-60° and 240-330° end in `LSA`, 90-120° in `RCCA`, 150° in `LCCA`, 180-210° run on in the aorta |
| A\* to `RSA@10` (balanced profile) | <1 s | route Aorta > RSA, 94 mm, ~15 s estimated; the table picks roll 120° (paths pass within 0.1 mm) |
| entry-roll controller, roll 120° | 27 s | **reached in 12.2 s**, 0.8 mm, peak force 0.2 mN |
| 12 demonstrations (σ 0 and 0.2) | 1.8 min | 12/12 successful |
| behaviour cloning + 1 evaluation episode | 20 s + 31 s | **reached in 12.8 s**, 1.3 mm; `check_handover.py` passes |
| videos (`results/cases/0024-rsa10/video_*.mp4`) | 2 min | A\* + roll 11.6 s at 1.1 mm; clone 12.8 s at 1.6 mm |

Two things went wrong first and are fixed in the code:

- **The expert overshot the first target** (`LCCA@20`, 20 mm into a 27 mm vessel). At full speed it went
  from 14 mm to 4.4 mm in one step, overshot to 10 mm, and its back-off oscillated between 8.5 and
  11 mm because the rod's slack makes the tip lag the robot. `EntryRollExpert` now backs off while the
  distance still shrinks and then creeps in at low speed (`backoff` / `creep` phases); with that it
  reached `LCCA@20` in 21.8 s at 4.3 mm.
- **A roll of 180° cannot be cloned.** The policy sees sin/cos of the shaft rotation, and 180° is where
  that representation wraps: the teacher hunted around it for 20 steps and the clone kept correcting
  the roll during the push and ended in the wrong vessel (0/1). Targets whose calibrated roll lies
  near ±180° need another roll in the arc or another target; `RollTable.reach(robust_within_mm=5)`
  now picks, inside the arc of rolls that reach the target, the one whose paths pass closest (ties to
  the arc centre), which gave 120° instead of 30° for the subclavian target. The ±180° case itself is
  still open: a signed-roll goal set (`make_goals.py --signed`) or a target further from the wrap.

Total: about 20 minutes from a fresh VMR model to a cloned policy with videos, excluding the two
detours.

## 4. If the mesh comes without centerlines

Their page's skeleton is the fastest source: after the scenario loads, `mesh.centerlineCanon` holds
one polyline per branch in the baked frame (metres), and the voxel distance field used for the trace
gives the lumen radius. Reading those through the debugging socket (`LockstepPage.cdp.eval`) and
feeding them to `data/centerline.build_graph` would give the anatomy our planner needs, in their
frame directly (identity `OrganiserFrame` with the 0.01 scale). Budget about an hour; not started.
