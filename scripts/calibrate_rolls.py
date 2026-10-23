"""Calibrate the entry-roll table on the organiser's physics (lockstep env).

For each roll, turns the guidewire to it at the entry and pushes it to the end of the robot's
travel (``EntryRollExpert`` with stopping disabled), recording the true tip after every control
step, for several noise seeds. ``stroke_nav.agents.roll_table.RollTable`` reads the result.

    python scripts/calibrate_rolls.py                                     # 0-345 deg every 15, 2 seeds
    python scripts/calibrate_rolls.py --rolls 90,120 --seeds 1 --workers 2 --out results/lockstep/try.json

Writes results/lockstep/roll_sweep.json (paths) and prints which branch each roll ends in. Stop
their Docker stack first (tools/organiser_stack.ps1 stop) so the pages get the CPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stroke_nav.agents.entry_roll import GW_INSERTION, EntryRollExpert  # noqa: E402
from stroke_nav.organiser.live.lockstep import make_env  # noqa: E402
from stroke_nav.organiser.spec import DEFAULT_SPEC  # noqa: E402

FAR_GOAL_M = (1.0, 1.0, 1.0)  # never reached, so their success rule does not end the push early
MAX_STEPS = 240


ENV_KWARGS: dict = {}


def sweep(worker: int, jobs: list[dict]) -> list[dict]:
    env = make_env(worker=worker, seed=worker, goals=[FAR_GOAL_M], max_steps=MAX_STEPS, **ENV_KWARGS)
    travel = DEFAULT_SPEC.robot.guidewire_travel_mm
    out = []
    try:
        for job in jobs:
            t0 = time.monotonic()
            obs, _ = env.reset(seed=job["seed"])
            expert = EntryRollExpert(roll_deg=job["roll"], slow_radius_mm=1e-3, stop_radius_mm=-1.0)
            expert.reset()
            tips, ins, stalled = [], [], 0
            for _ in range(MAX_STEPS):
                obs, _, term, trunc, _ = env.step(expert.act(obs))
                tips.append(list(env.bridge.latest_true().tip_position))
                ins.append(float(obs["state"][GW_INSERTION]) * 1000.0)
                stalled = stalled + 1 if len(ins) > 1 and ins[-1] - ins[-2] < 0.2 and ins[-1] > travel - 15.0 else 0
                if stalled >= 5 or term or trunc:
                    break
            out.append({"roll_deg": job["roll"], "seed": job["seed"], "tip_m": tips, "ins_mm": [round(v, 2) for v in ins]})
            print(json.dumps({"roll_deg": job["roll"], "seed": job["seed"], "steps": len(tips),
                              "final_ins_mm": round(ins[-1], 1), "wall_s": round(time.monotonic() - t0, 1)}), flush=True)
    finally:
        env.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rolls", default="0:360:15", help="start:stop:step or a comma-separated list (deg)")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None, help="default results/lockstep/roll_sweep.json (0248) or roll_sweep_<model>.json")
    ap.add_argument("--model", default="0248_H_AOCERE_CAS", help="VMR model; other models run their scenario from scripts/make_scenario.py")
    ap.add_argument("--variant", default=None)
    args = ap.parse_args()
    global ENV_KWARGS
    if args.model != "0248_H_AOCERE_CAS" or args.variant:
        from stroke_nav.organiser.scenario import DEFAULT_SCENARIO_ROOT, scenario_name

        name = scenario_name(args.model, args.variant)
        if not (Path(DEFAULT_SCENARIO_ROOT) / name / "scenario.json").is_file():
            raise SystemExit(f"scenario {name} not found: run scripts/make_scenario.py --model {args.model} first")
        ENV_KWARGS = {"scenario_name": name, "scenario_root": Path(DEFAULT_SCENARIO_ROOT).resolve()}
    if args.out is None:
        tag = "" if args.model == "0248_H_AOCERE_CAS" and not args.variant else f"_{name}"
        args.out = str(ROOT / "results" / "lockstep" / f"roll_sweep{tag}.json")
    if ":" in args.rolls:
        a, b, c = (float(v) for v in args.rolls.split(":"))
        rolls = list(np.arange(a, b, c))
    else:
        rolls = [float(r) for r in args.rolls.split(",")]
    jobs = [{"roll": float(r), "seed": 100 + s} for s in range(args.seeds) for r in rolls]
    t0 = time.monotonic()
    with ThreadPoolExecutor(args.workers) as ex:
        parts = list(ex.map(sweep, range(args.workers), [jobs[w::args.workers] for w in range(args.workers)]))
    paths = sorted((p for part in parts for p in part), key=lambda p: (p["roll_deg"], p["seed"]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device_parameters": "device-parameters.json", "paths": paths}), encoding="utf-8")
    print(f"\n{len(paths)} paths in {(time.monotonic() - t0) / 60:.1f} min -> {out}")

    from stroke_nav.data.anatomy import Anatomy
    from stroke_nav.organiser.frame import frame_for

    anat, frame = Anatomy.load(args.model, args.variant), frame_for(args.model)
    for p in paths:
        tip = frame.world_m_to_vmr_mm(np.asarray(p["tip_m"][-1]))
        print(f"  roll {p['roll_deg']:5.1f} seed {p['seed']}: ends in {anat.branch_of(anat.nearest_node(tip))}")


if __name__ == "__main__":
    main()
