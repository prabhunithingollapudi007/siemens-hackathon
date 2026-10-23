"""Prepare a new vessel mesh for the organiser's physics: frame, scenario, and a reachability check.

Given a VMR model (built by ``fetch_data.py``) and one or more targets, this

1. places the model the way their physics page expects an *authored* mesh: the insertion entrance
   (the model's default start node) at the origin and the vessel axis there along +Z
   (``organiser.frame.entry_frame``); the fitted frame is kept for 0248, whose mesh is theirs;
2. writes ``data/scenarios/<model>/{vessel.obj, goals.txt, scenario.json}`` with the surface decimated
   to their triangle count and the targets as goals, and validates it with their own loader;
3. saves the frame to ``data/processed/organiser_frame_<model>.json`` (``frame_for(model)`` loads it)
   and reports the route length of each target against the robot's travel.

    python scripts/make_scenario.py --model 0024_H_AO_H --target LCCA@20 --target RSA@15
    python scripts/make_scenario.py --model 0248_H_AOCERE_CAS --variant 25Right50Left --target left_carotid@140

Then: ``calibrate_rolls.py --model <model>`` (which entry roll reaches which vessel on their rod),
``plan_case.py --model <model> --target ... --table results/lockstep/roll_sweep_<model>.json`` and
``demo_lockstep.py --vmr-model <model> --target ...``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stroke_nav.data.anatomy import Anatomy  # noqa: E402
from stroke_nav.organiser.frame import default_frame, entry_frame, frame_path, read_obj_vertices  # noqa: E402
from stroke_nav.organiser.scenario import (DEFAULT_SCENARIO_ROOT, page_lumen_radius_cm, scenario_name,  # noqa: E402
                                           validate_scenario, write_scenario)
from stroke_nav.organiser.spec import DEFAULT_SPEC  # noqa: E402
from stroke_nav.planning.constraints import GUIDEWIRE  # noqa: E402
from stroke_nav.planning.planner import PathPlanner, PlannerWeights, nearest_feasible_target  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--target", action="append", default=[], help="target spec(s); become the scenario's goals")
    ap.add_argument("--start", default=None, help="entrance node id or target spec (default: the model's default start)")
    ap.add_argument("--triangles", type=int, default=10_000)
    ap.add_argument("--root", default=str(DEFAULT_SCENARIO_ROOT))
    args = ap.parse_args()
    if not args.target:
        ap.error("give at least one --target")

    anat = Anatomy.load(args.model, args.variant)
    start = anat.default_start() if args.start is None else anat.resolve(args.start)
    if args.model == "0248_H_AOCERE_CAS":
        frame = default_frame()
        print("[scenario] 0248: keeping the frame fitted to the organiser's cas-reference mesh")
    else:
        frame = entry_frame(anat, start)
        frame.save(frame_path(args.model))
        print(f"[scenario] entry frame: start node {start} ({anat.branch_of(start)}, lumen radius {anat.radius[start]:.1f} mm) at the origin, "
              f"vessel axis +Z -> {frame_path(args.model)}")

    planner = PathPlanner(anat, GUIDEWIRE, PlannerWeights.profile("balanced"))
    goals, travel = [], DEFAULT_SPEC.robot.guidewire_travel_mm
    for spec in args.target:
        lesion = anat.resolve(spec)
        node, back = nearest_feasible_target(planner, lesion)
        plan = planner.plan(start, node)
        goals.append(anat.nodes[node])
        if plan.feasible:
            print(f"[scenario] {spec:20s} -> node {node} ({anat.branch_of(node)}){f', {back:.0f} mm proximal of the lesion' if back else ''}: "
                  f"route {' > '.join(plan.branches())}, {plan.length_mm:.0f} mm "
                  f"({'within' if plan.length_mm <= travel else 'BEYOND'} the {travel:g} mm travel), ~{plan.costs['est_time_s']:.0f} s")
        else:
            print(f"[scenario] {spec:20s} -> node {node}: no feasible route ({plan.message})")

    out = write_scenario(args.model, args.variant, goals, args.root, frame, target_triangles=args.triangles,
                         entry_radius_mm=None if args.model == "0248_H_AOCERE_CAS" else float(anat.radius[start]))
    summary = validate_scenario(scenario_name(args.model, args.variant), args.root)
    verts = read_obj_vertices(out / "vessel.obj")
    band = verts[np.abs(verts[:, 2]) <= 0.02 * verts[:, 2].max()]
    print(f"[scenario] wrote {out}: {len(verts)} vertices, bounds {np.round(verts.min(0), 1).tolist()}..{np.round(verts.max(0), 1).tolist()} cm; "
          f"entrance band: {len(band)} vertices, centroid {np.round(band.mean(0), 2).tolist() if len(band) else None} cm "
          f"(their page re-centres x, y); page lumenR {page_lumen_radius_cm(verts) * 10:.1f} mm (devices drawn >= "
          f"{0.2 * page_lumen_radius_cm(verts) * 10:.1f} mm thick); mesh_to_world diag {np.round(np.asarray(summary['mesh_to_world']).reshape(4, 4).diagonal(), 3).tolist()}; "
          f"{summary.get('goal_count')} goals")


if __name__ == "__main__":
    main()
