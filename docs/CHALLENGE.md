# Challenge day: from a new mesh and target to candidate trajectories

The task: one patient (a vessel mesh) and one target (the clot); propose robot trajectories from several
methods, each with its training time, checked for safety, so a clinician or a safety layer can choose
(docs/CASE_PLANNING.md). Below is the fast path with the tools in this repo. Details: docs/NEW_MESH.md
(recipe), docs/PLANNER.md (A* costs), docs/SAFETY.md §7-8 (wall crossings, removal).

## 0. Before the task arrives (5 min)

```bash
powershell -ExecutionPolicy Bypass -File tools\organiser_stack.ps1 stop     # their Docker stack takes the CPU
.venv\Scripts\python.exe -c "from stroke_nav.organiser.live.lockstep import kill_stray_pages, other_lockstep_jobs; print(kill_stray_pages(), other_lockstep_jobs())"
```

One lockstep job at a time, across terminals and sessions. Runs: one episode, Chrome visible (`--visible`).

## 1. Look before running (10 min) - show the user, get an OK

| Question | How |
|---|---|
| What is the anatomy, where is the open entry end, which vessel is the target in? | `fetch_data.py --model <id>` (VMR) or the given mesh; `Anatomy.load(...).branch_names`, leaves, junction distances |
| Is the target in reach? | `make_scenario.py --model <id> --start <entry node> --target <spec>` prints the route length against the robot's travel |
| Does their page draw it right? | `make_scenario.py` prints the page's `lumenR` (devices drawn >= 20 % of it thick); render the page at rest |
| Robot changed? | new `robots/<name>/` profile (device file within their page's sliders + `robot.json` travel <= 800 mm), `STROKE_NAV_ROBOT=<name>` |

Plain meshes without centerlines are not supported yet (docs/NEW_MESH.md §4): ask for the VMR id or centerlines.

## 2. Plan and execute (20-40 min)

```bash
.venv\Scripts\python.exe scripts/calibrate_rolls.py --model <id> --rolls 0:360:30 --seeds 1 --workers 4 --out results/lockstep/roll_sweep_<id>.json   # once per mesh+entry+robot, 5-15 min
.venv\Scripts\python.exe scripts/plan_case.py --model <id> --target <spec> --case-id <case> --table results/lockstep/roll_sweep_<id>.json \
    --methods astar,roll,roll_cath,bc --cath-park 1000 --bc-demos 12 --sigmas 0,0.1 --eval-episodes 1 --max-steps 600
```

- `astar`: route, length, time, damage terms (seconds). `roll`: A* + calibrated entry roll on their physics.
  `roll_cath`: the same with the catheter 25 mm behind the wire tip (the method that worked at the arch).
  `bc`: demonstrations of the roll controller cloned into their SAC model (~5 min).
- The wall safety layer is on by default; `--no-wall-safety` shows raw behaviour.
- Read `results/cases/<case>/summary.md`: success counts only runs that **did not cross a wall**
  (`wall` flag = reject). Never report their success rule alone.

## 3. If it does not reach

| Symptom | Try |
|---|---|
| wire crosses the wall at a branch opening | `roll_cath`; finer rods (`fine-rod`, or a `long-reach-fine`-style profile); neighbouring rolls |
| wrong vessel at a junction | another roll from the table (`RollTable.reach`), or `--roll` |
| out of travel | a robot profile with more travel (`long-reach`), or a mid-route entry to test the hard segment (`scripts/loop_hook.py` pattern) |
| long approach works, last segment does not | report the checkpoint: autonomous up to there, operator (teleop) beyond |

## 4. Deliver

- `summary.md` table per method: train time, plan time, safe success, time to goal, final distance, peak force, flags
- one video per working method: `demo_lockstep.py --vmr-model <id> --target <spec> --expert --roll <deg> --record <mp4>` (`--wall-safety` to show the stops)
- removal: unwind the entry roll, then pull (`scripts/removal_trials.py ... --strategies unwind_guarded`)
- hand-over bundle only with their device file (hash `38b1ea53...`): `export_deployment.py --case results/cases/<case>`

## What we know going in (honest)

- On their physics, pushed rods slip through walls between collision bodies; our wall check catches it.
  Every earlier entry-roll "success" crossed somewhere; with catheter support from below the arch the left CCA
  of 0028 was reached with a brief 3-step crossing.
- Straight segments (aorta, iliacs, CCAs) are easy; branch openings next to another lumen are the hard
  checkpoints. That is the pitch: fast A* + roll where the anatomy is simple, checkpoints where it is not.
