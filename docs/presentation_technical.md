# stroke-nav: technical slides (hackathon, 2026-10-24)

Four slides, technical only. Each `---` is one slide. The ASCII diagrams are block diagrams: redraw
each box as a shape and each arrow as a connector. Numbers come from `README.md`, `docs/PLANNER.md`
and `docs/CASE_PLANNING.md` (runs of 2026-10-23, on the organiser's physics unless stated).

---

## Slide 1: System architecture

**From a patient's CT segmentation to a robot trajectory the organiser's stack can execute.**

```
 PATIENT DATA                 PLANNING                       CONTROL                      ORGANISER STACK
┌──────────────────┐   ┌──────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────────┐
│ VMR vessel model │   │ Centerline graph     │   │ Controllers (candidates) │   │ Their physics (EndoNav / │
│ CT, 30 stenosis  │──▶│ 1 mm nodes, lumen    │──▶│  • A* route follower     │──▶│ CatForce rod simulator)  │
│ variants         │   │ radius per node      │   │  • entry-roll controller │   │ run "lockstep": headless,│
└──────────────────┘   └──────────┬───────────┘   │  • behaviour cloning (BC)│   │ no Docker/ROS, 4 workers │
                                  │               │  • action chunking (ACT) │   └────────────┬─────────────┘
┌──────────────────┐   ┌──────────▼───────────┐   │  • SAC fine-tuning       │                │
│ Device params    │   │ Constrained A*       │   └──────────────────────────┘   ┌────────────▼─────────────┐
│ (organiser JSON: │──▶│ length · time ·      │              ▲                   │ Safety checks + report   │
│ wire, catheter,  │   │ damage · wrong vessel│──────────────┘                   │ lumen, wrong vessel,     │
│ robot limits)    │   │ hard: clearance,     │   route, roll, stop point        │ overshoot, force, travel │
└──────────────────┘   │ forbidden vessels    │                                  └────────────┬─────────────┘
                       └──────────────────────┘                                               │
                                                                                 ┌────────────▼─────────────┐
                                                                                 │ Hand-over bundle         │
                                                                                 │ SB3 SAC zip + scenario + │
                                                                                 │ trajectory CSV; runs in  │
                                                                                 │ their ./sys evaluator    │
                                                                                 └──────────────────────────┘
```

- One source of truth for devices and robot: the organiser's `device-parameters.json` (5 Hz, 10 mm/s, guidewire travel 264.5 mm).
- Everything in the organiser's formats: observation `image 128x128 + 43-value state`, action `[v_wire, w_wire, v_cath, w_cath]`, their SAC policy class, their scenario manifest.
- Our own fast kinematic simulator (Gymnasium) is used for planning and unit tests; results are always confirmed on their physics.

---

## Slide 2: Pre-operative case planner (`plan_case.py`)

**One patient, one target, several proposals, each with a known time budget and a safety verdict.**

```
 vessel model + target
        │
        ▼
┌──────────────────────┐     ┌────────────────────────────────┐     ┌────────────────────────────┐
│ 1. Case              │     │ 2. One controller per method    │     │ 3. Rollouts on their       │
│ resolve target on    │────▶│    under its own budget         │────▶│    physics (lockstep)      │
│ graph, A* route,     │     │    astar │ roll │ bc │ act │ sac│     │    6 unseen seeds / method │
│ reachability vs      │     │    demos recorded once, shared  │     │    true tip path, force,   │
│ 264.5 mm travel      │     │    by bc / act / sac            │     │    insertion, actions      │
└──────────────────────┘     └────────────────────────────────┘     └─────────────┬──────────────┘
                                                                                  │
        ┌─────────────────────────────┐     ┌─────────────────────────────────────▼──────────────┐
        │ 5. summary.md, candidates   │◀────│ 4. Common trajectory format + deterministic safety │
        │    .png, trajectory CSV,    │     │    checks: lumen, wrong_vessel, overshoot, force,  │
        │    hand-over model          │     │    travel, stall, speed  → pass / review / reject  │
        └─────────────────────────────┘     └────────────────────────────────────────────────────┘
```

| Method | What it is | Train time | Plan time |
|---|---|---|---|
| A* | constrained A* on the centerline graph + closed-loop follower | none | 10-40 ms per route |
| Entry roll | calibrated roll at the entry, then push (table from one 15 min sweep per anatomy) | none per target | minutes (rollouts) |
| BC | expert demonstrations at this target, cloned into the organiser's SAC model | ~10 min (demos) + 30 s fit | minutes |
| ACT | action-chunking transformer on the same demonstrations | ~3 min | minutes |
| SAC | fine-tuning from the BC model on this target | budgeted | minutes |

Result on the organiser's own goal: entry roll, BC and ACT each **6/6** at about 29 s; the whole case took 19.7 min of wall time.

---

## Slide 3: Planning and learning details

**Constrained A\*** (what the clinicians asked a route to be judged on, before anything moves):

```
 cost = w_len · length + w_time · time(speed envelope by lumen, slow near bifurcations)
      + w_damage · (bending effort + curvature integral + narrow lumen + contact-force proxy)
      + w_branch · vessel changes + w_takeoff · sharp take-off angles
 hard constraints: lumen ≥ 1.2 × device diameter · forbidden vessels · optional bend-radius limit
 profiles: shortest | fastest | gentle | balanced
 local replan from the current tip: 143-350 states, against 620-1040 for a cold plan
```

**Why execution is "roll at the entry, then push":** on their rod the roll set at the entry decides which
vessel the wire takes, and re-steering at a bifurcation releases stored twist. A\* gives the route, its
evaluation and the stop point; a calibrated roll table gives the roll whose recorded paths pass the target.

**Learning pipeline (behaviour cloning into their format):**

```
 expert (entry roll)  ──▶  demos on their physics  ──▶  BC: fit actor of SB3 SAC  ──▶  check_handover.py
 reads only the 43-state     image + state + action        (their FluoroStateExtractor,      NumPy 1.x pickle,
 + fluoro, no privileged     DART noise σ 0 / 0.2 / 0.4     MSE, 40 epochs ≈ 30 s)            spaces, scenario sha
 information                 labels stay the expert's       critics warm-started
```

- Lockstep env: their physics page and noisy viewer run in headless Chrome; we drive the animation frames ourselves and speak their ROS-bridge protocol in-page. Their robot delay, deadband, travel limits and sensor noise are emulated. 0.8× real time per worker with 4 workers, against 0.18× for their full Docker stack.
- 70 % left ICA lesion (`10Right70Left`): A\* + roll reached it in 27 s, 1.9 mm, 5 mN; 12 demos + BC (3.6 min of training) reached it in 26.4 s.

---

## Slide 4: Safety layer and hand-over

**Doer / checker (simplex) architecture:** the learned part never talks to the actuators directly.

```
 policy (BC / ACT / SAC / follower) ──action──▶ ┌───────────────────┐ ──filtered action──▶ robot / simulator
                                                │  SafetyMonitor    │
   encoders (insertion, roll) · tracked tip     │  deterministic,   │   may only make a command LESS
   + dropout flag · blocked push / force        │  measured inputs  │   aggressive: clamp, hold, bounded
   · approved plan                       ──────▶│  only             │   back-off; never adds motion
                                                └───────────────────┘
```

- Runtime: on an imperfect policy (synthetic arch, stressed tracking) success 60 → 100 %, target overshoot 40 → 0 %.
- Offline: every candidate trajectory gets the same flags (lumen, wrong vessel, overshoot, force, travel, stall, speed) and a verdict before a clinician sees it.

**Hand-over, verified end to end on their evaluator:**

```
 results/cases/<case>/ ──▶ export_deployment.py ──▶ bundle: SAC zip + scenario (mesh, goal) + deploy.sh
                                                         │
                                                         ▼
                      their ./sys / evaluate_controller.py on their physics  ──▶  results (check_deployment.py)
```

- Invariants guarded by tests: exact organiser observation and action spaces, their policy class, NumPy 1.x pickles, device-parameter hash, scenario manifest hash, fitted frame transform (median 0.45 mm).
- 79 pytest tests on synthetic vessel trees, no downloads needed.
