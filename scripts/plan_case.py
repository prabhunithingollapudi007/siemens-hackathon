"""Plan one patient case: candidate trajectories to one target, from several methods, each under a time budget.

The input is a vessel model (a VMR id built by ``fetch_data.py``; the patient's CTA segmentation in
production) and one target: ``branch@mm``, ``xyz:x,y,z`` (VMR mm, e.g. a lesion from the segmentation),
a node id, or ``scenario`` (the organiser's cas-reference goal). Each method is run on this one target
and its result is written in the common candidate format, so a reviewer or a safety layer can compare
proposals (``stroke_nav.planning.candidates``). Overfitting to the case is intended.

Methods (``--methods``):

* ``astar``: constrained A* on the centerline graph + the closed-loop route follower, on our kinematic
  simulator. Seconds. The geometric route is also the reference for the safety checks.
* ``astar_theirs``: the same follower on the organiser's physics (lockstep), steered by the noisy state.
* ``roll``: the entry-roll controller with the roll the calibrated table gives for this target, on their
  physics. No training beyond the table (``calibrate_rolls.py``, done once per anatomy).
* ``bc``: records expert demonstrations at this target until ``--bc-minutes`` or ``--bc-demos`` is
  reached, clones them into the hand-over SAC model (``train_bc.py``), evaluates it.
* ``act``: fits the action-chunking policy (``stroke_nav.agents.act``) to the same demonstrations within
  ``--act-minutes``, evaluates it.
* ``sac``: SAC fine-tuning from the BC model on this target for ``--sac-minutes`` (``finetune_sac_lockstep.py``).

Output: ``results/cases/<case-id>/`` with ``case.json`` (target, route, reachability), ``goals.json`` (the
target as a one-goal set for record_demos / eval_lockstep / demo_lockstep ``--goals ... --split train``),
``candidates/<method>.json``, ``trajectory_<method>.csv``, ``summary.json``, ``summary.md`` and
``candidates.png``.

    python scripts/plan_case.py --target right_carotid@40 --case-id rca40
    python scripts/plan_case.py --target scenario --methods astar,roll --eval-episodes 4
    python scripts/plan_case.py --target xyz:-10.5,120.2,-80.0 --methods astar,astar_theirs,roll,bc,act --bc-minutes 10 --act-minutes 3

Their physics runs in the lockstep env (organiser sources at HCMR_EV_ROOT); stop their Docker stack
first and run one case at a time. Stray lockstep Chromes from a crashed run are killed at the start.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_astar import OursAdapter, build  # noqa: E402

from stroke_nav.agents.entry_roll import EntryRollExpert  # noqa: E402
from stroke_nav.agents.roll_table import RollTable  # noqa: E402
from stroke_nav.agents.route_follower import ClosedLoopRouteFollower, FollowerParams, TipObs  # noqa: E402
from stroke_nav.data.anatomy import Anatomy  # noqa: E402
from stroke_nav.organiser.frame import default_frame  # noqa: E402
from stroke_nav.organiser.spec import DEFAULT_SPEC  # noqa: E402
from stroke_nav.planning.candidates import (Candidate, EpisodeTrace, assess, plot_candidates, summary_table,  # noqa: E402
                                            write_trajectory_csv)
from stroke_nav.safety.monitor import SafetyConfig  # noqa: E402

PY = sys.executable
SUCCESS_MM = DEFAULT_SPEC.environment.success_radius_m * 1000.0
SUCCESS_STEPS = DEFAULT_SPEC.environment.success_steps


def log(msg: str) -> None:
    print(f"[case] {msg}", flush=True)


# ---------------------------------------------------------------------- rollouts on their physics
from stroke_nav.organiser.live.rollout import RouteFollowerOnObservation as FollowerOnTheirs  # noqa: E402
from stroke_nav.organiser.live.rollout import SACPolicy, lockstep_pool  # noqa: E402


def astar_on_ours(anat, frame, goal_vmr, plan, params: FollowerParams, max_steps: int) -> EpisodeTrace:
    follower = ClosedLoopRouteFollower(anat, plan, params)
    sim = OursAdapter(anat, frame, goal_vmr, DEFAULT_SPEC)
    t0 = time.monotonic()
    tips, ins, acts, dists = [], [], [], []
    in_goal, success = 0, False
    o = sim.reset()
    for _ in range(max_steps):
        a = np.asarray(follower.act(TipObs(o["wire_tip"], o["wire_dir"], o["wire_ins"], o["cath_ins"], goal_vmr)), np.float32)
        o = sim.step(a)
        dist = float(np.linalg.norm(o["wire_tip"] - goal_vmr))
        tips.append(frame.vmr_mm_to_world_m(o["wire_tip"]))
        ins.append(float(o["wire_ins"]))
        acts.append(a)
        dists.append(dist)
        in_goal = in_goal + 1 if dist <= SUCCESS_MM else 0
        if in_goal >= SUCCESS_STEPS:
            success = True
            break
        if follower.state.mode == "travel_limit":
            break
    return EpisodeTrace(seed=0, success=success, tip_m=np.array(tips), insertion_mm=np.array(ins), actions=np.array(acts),
                        dist_mm=np.array(dists), peak_force_n=0.0, reason="success" if success else follower.state.mode,
                        wall_s=time.monotonic() - t0)


def run_script(args: list[str], logfile: Path) -> float:
    """Run another repo script with this interpreter, appending its output to ``logfile``; returns seconds."""
    log(" ".join(["python", *args]))
    t0 = time.monotonic()
    with open(logfile, "a", encoding="utf-8") as f:
        r = subprocess.run([PY, *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit(f"[case] {args[0]} failed ({r.returncode}); see {logfile}")
    return time.monotonic() - t0


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="0248_H_AOCERE_CAS", help="VMR model (built by fetch_data.py)")
    ap.add_argument("--variant", default=None, help="0248 stenosis variant; their physics then runs its mesh as a scenario (stroke_nav.organiser.scenario)")
    ap.add_argument("--visible", action="store_true", help="their physics: show the pages in Chrome windows (one worker)")
    ap.add_argument("--target", required=True, help="branch@mm, xyz:x,y,z (VMR mm), node id, or 'scenario'")
    ap.add_argument("--case-id", default=None, help="folder name under --out-root (default: from the target)")
    ap.add_argument("--out-root", default=str(ROOT / "results" / "cases"))
    ap.add_argument("--methods", default="astar,astar_theirs,roll,bc,act",
                    help="comma-separated subset of astar, astar_theirs, roll, bc, act, sac")
    ap.add_argument("--workers", type=int, default=4, help="lockstep envs in parallel")
    ap.add_argument("--eval-episodes", type=int, default=6, help="rollouts per method on their physics")
    ap.add_argument("--max-steps", type=int, default=300, help="episode limit (5 Hz steps; organiser default 600)")
    ap.add_argument("--seed", type=int, default=1000, help="first evaluation seed; demonstrations use 100..")
    ap.add_argument("--table", default=f"{ROOT / 'results' / 'lockstep' / 'roll_sweep.json'},{ROOT / 'results' / 'lockstep' / 'roll_sweep_neg.json'}",
                    help="entry-roll sweep file(s) from calibrate_rolls.py, comma-separated (missing ones are skipped)")
    ap.add_argument("--roll", type=float, default=None, help="entry roll (deg) to use instead of the table's")
    ap.add_argument("--slow-radius", type=float, default=8.0, help="the expert slows inside this distance (mm)")
    ap.add_argument("--bc-demos", type=int, default=24, help="demonstrations to record at most")
    ap.add_argument("--bc-minutes", type=float, default=10.0, help="wall-time budget for recording demonstrations")
    ap.add_argument("--bc-epochs", type=int, default=40)
    ap.add_argument("--sigmas", default="0,0.2,0.4", help="DART noise levels cycled over the demonstrations")
    ap.add_argument("--act-minutes", type=float, default=3.0, help="wall-time budget for fitting the chunk policy")
    ap.add_argument("--act-epochs", type=int, default=80, help="upper bound on epochs within --act-minutes")
    ap.add_argument("--act-chunk", type=int, default=10)
    ap.add_argument("--sac-minutes", type=float, default=15.0, help="wall-time budget for SAC fine-tuning (method sac)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--keep-stray", action="store_true", help="do not kill stray lockstep Chromes at the start")
    ap.add_argument("--resume", action="store_true",
                    help="reuse candidates/<method>.json and demos/ already in the case folder instead of rerunning them")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = set(methods) - {"astar", "astar_theirs", "roll", "bc", "act", "sac"}
    if unknown:
        ap.error(f"unknown methods {sorted(unknown)}")
    needs_theirs = bool(set(methods) & {"astar_theirs", "roll", "bc", "act", "sac"})
    case_id = args.case_id or args.target.replace("@", "-at-").replace(":", "-").replace(",", "_")
    case_dir = Path(args.out_root) / case_id
    (case_dir / "candidates").mkdir(parents=True, exist_ok=True)
    logfile = case_dir / "case.log"
    t_case = time.monotonic()

    if needs_theirs:
        from stroke_nav.organiser import ensure_policy_importable
        from stroke_nav.organiser.live.lockstep import kill_stray_pages, other_lockstep_jobs

        ensure_policy_importable()
        others = other_lockstep_jobs()
        if others:
            # they share the worker ports and Chrome debug ports; cleaning up our pages would kill theirs
            raise SystemExit("[case] another lockstep job is running; wait for it or stop it first:\n  " + "\n  ".join(others))
        if not args.keep_stray:
            n = kill_stray_pages()
            if n:
                log(f"killed {n} stray lockstep Chrome processes")

    # ------------------------------------------------------------------ the case: target, route, reachability
    anat, frame = Anatomy.load(args.model, args.variant), default_frame()
    t0 = time.monotonic()
    goal_vmr, node, plan, cath_limit = build(anat, frame, args.target)
    env_kwargs: dict = {"visible": True} if args.visible else {}
    if args.visible:
        args.workers = 1
    if args.variant and needs_theirs:
        from stroke_nav.organiser.scenario import DEFAULT_SCENARIO_ROOT, scenario_name, write_scenario

        if "sac" in methods:
            ap.error("--variant does not support the sac method yet (finetune_sac_lockstep.py runs the cas-reference scenario)")
        sc_name = scenario_name(args.model, args.variant)
        if not (Path(DEFAULT_SCENARIO_ROOT) / sc_name / "scenario.json").is_file():
            write_scenario(args.model, args.variant, [goal_vmr], DEFAULT_SCENARIO_ROOT, frame)
            log(f"wrote scenario {sc_name} for their physics")
        env_kwargs.update(scenario_name=sc_name, scenario_root=Path(DEFAULT_SCENARIO_ROOT))
    plan_s = time.monotonic() - t0
    goal_m = np.asarray(frame.vmr_mm_to_world_m(goal_vmr), float)
    travel = DEFAULT_SPEC.robot.guidewire_travel_mm
    route_mm = float(plan.length_mm)
    reach = {"kinematic_route_mm": round(route_mm, 1), "guidewire_travel_mm": travel,
             "reachable_by_travel": bool(route_mm <= travel)}
    roll, miss = args.roll, None
    tables = [t for t in args.table.split(",") if t and Path(t).exists()]
    if tables:
        table = RollTable.load(",".join(tables))
        best_roll, miss = table.reach(goal_m, robust_within_mm=SUCCESS_MM)  # centre of the arc of rolls that reach
        closest_roll, closest_miss = table.reach(goal_m)
        reach["roll_table"] = {"files": tables, "roll_deg": best_roll, "recorded_paths_miss_mm": round(miss, 2),
                               "reachable": bool(miss <= SUCCESS_MM), "closest_roll_deg": closest_roll,
                               "closest_miss_mm": round(closest_miss, 2)}
        if roll is None:
            roll = best_roll
    elif roll is None:
        reach["roll_table"] = None
    case = {"case_id": case_id, "model": args.model, "variant": args.variant, "scenario": env_kwargs.get("scenario_name", "cas-reference"),
            "target": args.target, "node": int(node), "branch": anat.branch_of(node),
            "goal_vmr_mm": np.round(goal_vmr, 3).tolist(), "goal_world_m": np.round(goal_m, 6).tolist(),
            "route": plan.summary(), "plan_seconds": round(plan_s, 2), "catheter_limit_mm": round(cath_limit, 1),
            "entry_roll_deg": roll, "reach": reach, "methods": methods,
            "budgets": {"bc_demos": args.bc_demos, "bc_minutes": args.bc_minutes, "act_minutes": args.act_minutes,
                        "sac_minutes": args.sac_minutes if "sac" in methods else 0.0, "eval_episodes": args.eval_episodes,
                        "max_steps": args.max_steps, "workers": args.workers}}
    (case_dir / "case.json").write_text(json.dumps(case, indent=2, default=float), encoding="utf-8")
    (case_dir / "goals.json").write_text(json.dumps({"case": case_id, "reach_mm": SUCCESS_MM, "goals": [
        {"id": 0, "spec": args.target, "branch": case["branch"], "split": "train", "source": "case", "roll_deg": roll,
         "miss_mm": None if miss is None else round(miss, 2), "goal_m": goal_m.tolist()}]}, indent=1), encoding="utf-8")
    log(f"{case_id}: {args.target} -> node {node} ({case['branch']}); route {plan.summary()['branches']} "
        f"{route_mm:.0f} mm, {len(plan.junctions)} bifurcations; travel {travel} mm -> "
        f"{'reachable' if reach['reachable_by_travel'] else 'OUT OF REACH'} kinematically")
    if miss is not None:
        log(f"entry-roll table: {roll:g} deg, recorded paths miss the target by {miss:.1f} mm -> "
            f"{'reachable' if miss <= SUCCESS_MM else 'probably out of reach'} on their physics")
    elif roll is not None:
        log(f"entry roll {roll:g} deg from --roll (no table)")

    params = FollowerParams(cath_limit_mm=cath_limit, travel_limit_mm=travel - 0.5)
    seeds = [args.seed + i for i in range(args.eval_episodes)]
    cands: list[Candidate] = []

    def finish(c: Candidate) -> None:
        c.save(case_dir / "candidates" / f"{c.method}.json")
        cands.append(c)
        log(f"{c.method}: {sum(e.success for e in c.episodes)}/{len(c.episodes)} successful; train {c.train_wall_s:.0f} s, "
            f"plan {c.plan_wall_s:.0f} s")

    reused: set[str] = set()

    def done(method: str) -> bool:
        """With --resume: a candidate already in the case folder is loaded instead of rerun."""
        f = case_dir / "candidates" / f"{method}.json"
        if method in reused:
            return True
        if args.resume and f.exists():
            cands.append(Candidate.load(f))
            reused.add(method)
            log(f"{method}: reusing {f.relative_to(case_dir)}")
            return True
        return False

    # ------------------------------------------------------------------ astar on our simulator
    if "astar" in methods and not done("astar"):
        t0 = time.monotonic()
        trace = astar_on_ours(anat, frame, goal_vmr, plan, params, args.max_steps)
        c = Candidate("astar", "ours-kinematic", "constrained A* on the centerline graph + closed-loop route follower",
                      plan_wall_s=plan_s + (time.monotonic() - t0), episodes=[trace],
                      notes=["kinematic simulator: no wall forces, the tip rides the centerline, roll picks the branch"])
        if not reach["reachable_by_travel"]:
            c.notes.append(f"route {route_mm:.0f} mm exceeds the guidewire travel {travel} mm")
        finish(c)

    # ------------------------------------------------------------------ astar follower on their physics
    if "astar_theirs" in methods and not done("astar_theirs"):
        t0 = time.monotonic()
        res = lockstep_pool([{"seed": s} for s in seeds], lambda w: FollowerOnTheirs(anat, plan, params, frame, goal_vmr),
                            args.workers, args.max_steps, goal_m, tag="astar_theirs", **env_kwargs)
        finish(Candidate("astar_theirs", "theirs-lockstep", "the A* route follower steering on the noisy tip state, on their rod physics",
                         plan_wall_s=plan_s + (time.monotonic() - t0), episodes=[r[1] for r in res],
                         notes=["their rod picks branches differently from our tip model; re-steering at bifurcations is unreliable there"]))

    # ------------------------------------------------------------------ entry-roll controller
    if roll is None and set(methods) & {"roll", "bc", "act", "sac"}:
        log("no entry roll (no table and no --roll): skipping roll, bc, act and sac")
        methods = [m for m in methods if m not in {"roll", "bc", "act", "sac"}]
    if "roll" in methods and not done("roll"):
        t0 = time.monotonic()
        res = lockstep_pool([{"seed": s} for s in seeds], lambda w: EntryRollExpert(roll_deg=roll, slow_radius_mm=args.slow_radius),
                            args.workers, args.max_steps, goal_m, tag="roll", **env_kwargs)
        c = Candidate("roll", "theirs-lockstep", f"entry-roll controller: roll to {roll:g} deg at the entry, then push",
                      plan_wall_s=time.monotonic() - t0, episodes=[r[1] for r in res],
                      notes=["training cost is the one-off roll sweep per anatomy (calibrate_rolls.py, ~15 min)"])
        if miss is not None:
            c.notes.append(f"calibrated paths miss the target by {miss:.1f} mm")
        finish(c)

    # ------------------------------------------------------------------ demonstrations, BC, ACT, SAC
    demos_dir, demo_meta, demo_s = case_dir / "demos", case_dir / "demos" / "meta.json", 0.0
    bc_model = case_dir / "runs" / "bc" / "final-model.zip"
    learned = [m for m in ("bc", "act", "sac") if m in methods and not done(m)]
    if learned:
        demos_dir.mkdir(exist_ok=True)
        if args.resume and demo_meta.exists() and len(list(demos_dir.glob("ep_*.npz"))) >= 2:
            demo_s = float(json.loads(demo_meta.read_text(encoding="utf-8"))["wall_s"])
            log(f"reusing {len(list(demos_dir.glob('ep_*.npz')))} demonstrations in {demos_dir.relative_to(case_dir)}")
        else:
            sigmas = [float(s) for s in args.sigmas.split(",")]
            jobs = [{"seed": 100 + i, "sigma": sigmas[i % len(sigmas)], "record": True} for i in range(args.bc_demos)]
            t0 = time.monotonic()
            res = lockstep_pool(jobs, lambda w: EntryRollExpert(roll_deg=roll, slow_radius_mm=args.slow_radius), args.workers,
                                args.max_steps, goal_m, deadline=t0 + 60.0 * args.bc_minutes, tag="demo", **env_kwargs)
            demo_s = time.monotonic() - t0
            for f in demos_dir.glob("ep_*.npz"):
                f.unlink()
            for k, (job, trace, raw) in enumerate(res):
                np.savez_compressed(demos_dir / f"ep_{k:04d}.npz", **raw, goal_m=goal_m.astype(np.float32), goal_id=0,
                                    branch=case["branch"], roll_deg=roll, sigma=job["sigma"], seed=job["seed"],
                                    slow_radius_mm=args.slow_radius, expert="entry", policy="", beta=1.0)
            n_ok = sum(r[1].success for r in res)
            demo_meta.write_text(json.dumps({"episodes": len(res), "successful": n_ok, "wall_s": round(demo_s, 1),
                                             "budget_minutes": args.bc_minutes, "max_episodes": args.bc_demos,
                                             "sigmas": sigmas, "roll_deg": roll}, indent=1), encoding="utf-8")
            log(f"{len(res)} demonstrations ({n_ok} successful) in {demo_s / 60:.1f} min (budget {args.bc_minutes:g} min, "
                f"{args.bc_demos} at most)")
            if len(res) < 2:
                log("fewer than 2 demonstrations: skipping bc, act and sac")
                learned = []

    train_s = 0.0
    if set(learned) & {"bc", "sac"}:
        if args.resume and bc_model.exists() and (bc_model.parent / "bc-training.json").exists():
            train_s = float(json.loads((bc_model.parent / "bc-training.json").read_text(encoding="utf-8"))["train_seconds"])
            log(f"reusing the BC model {bc_model.relative_to(case_dir)}")
        else:
            n_ok = sum(bool(np.load(f)["success"]) for f in demos_dir.glob("ep_*.npz"))
            train_s = run_script(["scripts/train_bc.py", "--demos", str(demos_dir), "--epochs", str(args.bc_epochs), "--device", args.device,
                                  "--run-root", str(case_dir / "runs"), "--run-id", "bc", *(["--successful-only"] if n_ok >= 2 else [])], logfile)
    if "bc" in learned:
        t0 = time.monotonic()
        res = lockstep_pool([{"seed": s} for s in seeds], lambda w: SACPolicy(str(bc_model)), args.workers, args.max_steps, goal_m, tag="bc", **env_kwargs)
        finish(Candidate("bc", "theirs-lockstep", "behaviour cloning of the entry-roll demonstrations into the hand-over SAC model",
                         train_wall_s=demo_s + train_s, plan_wall_s=time.monotonic() - t0, episodes=[r[1] for r in res],
                         artifacts={"model": str(bc_model), "demos": str(demos_dir), "run_dir": str(bc_model.parent)},
                         notes=[f"{len(list(demos_dir.glob('ep_*.npz')))} demonstrations ({demo_s / 60:.1f} min) + {train_s:.0f} s of fitting",
                                "loads in the organiser's evaluation (check_handover.py)"]))

    if "act" in learned:
        from train_bc import load_demos

        from stroke_nav.agents.act import ACTConfig, ChunkExecutor, load_act, save_act, train_act

        episodes = load_demos([demos_dir], successful_only=False, gamma=DEFAULT_SPEC.training.gamma)
        cfg = ACTConfig(chunk=args.act_chunk)
        t0 = time.monotonic()
        model, history = train_act(episodes, cfg, epochs=args.act_epochs, max_seconds=60.0 * args.act_minutes, device=args.device, log=log)
        act_path = save_act(model, cfg, case_dir / "runs" / "act" / "act.pt")
        (case_dir / "runs" / "act" / "act-training.json").write_text(json.dumps({"config": cfg.__dict__, "history": history}, indent=1),
                                                                      encoding="utf-8")
        train_s = time.monotonic() - t0
        t0 = time.monotonic()
        res = lockstep_pool([{"seed": s} for s in seeds], lambda w: ChunkExecutor(*load_act(act_path)), args.workers, args.max_steps,
                            goal_m, tag="act", **env_kwargs)
        finish(Candidate("act", "theirs-lockstep", f"action chunking (ACT-style transformer, chunk {cfg.chunk}) on the same demonstrations",
                         train_wall_s=demo_s + train_s, plan_wall_s=time.monotonic() - t0, episodes=[r[1] for r in res],
                         artifacts={"model": str(act_path), "demos": str(demos_dir)},
                         notes=[f"{len(history)} epochs in {train_s:.0f} s (budget {args.act_minutes:g} min); final train L1 {history[-1]['train_l1']:.4f}",
                                "plain torch checkpoint, not an SB3 model: its product is the trajectory, not a hand-over policy"]))

    if "sac" in learned:
        train_s = run_script(["scripts/finetune_sac_lockstep.py", "--init", str(bc_model), "--goals", str(case_dir / "goals.json"),
                              "--split", "train", "--timesteps", "1000000", "--max-minutes", str(args.sac_minutes),
                              "--workers", str(args.workers), "--max-steps", str(args.max_steps), "--device", args.device,
                              "--run-root", str(case_dir / "runs"), "--run-id", "sac"], logfile)
        sac_model = case_dir / "runs" / "sac" / "final-model.zip"
        t0 = time.monotonic()
        res = lockstep_pool([{"seed": s} for s in seeds], lambda w: SACPolicy(str(sac_model)), args.workers, args.max_steps, goal_m, tag="sac", **env_kwargs)
        finish(Candidate("sac", "theirs-lockstep", "SAC fine-tuning of the BC model on this target, on their physics",
                         train_wall_s=demo_s + train_s, plan_wall_s=time.monotonic() - t0, episodes=[r[1] for r in res],
                         artifacts={"model": str(sac_model), "init": str(bc_model)},
                         notes=[f"{train_s / 60:.1f} min of SAC after the BC warm start (budget {args.sac_minutes:g} min)"]))

    # ------------------------------------------------------------------ assessment and reports
    cfg_s = SafetyConfig()
    assessments = {c.method: assess(c, anat, frame, plan.points, set(int(n) for n in plan.nodes), travel, cfg_s, DEFAULT_SPEC.dt) for c in cands}
    for c in cands:
        rep = c.representative()
        if rep is not None:
            write_trajectory_csv(case_dir / f"trajectory_{c.method}.csv", rep, frame, DEFAULT_SPEC.dt)
    table = summary_table(cands, assessments)
    total_s = time.monotonic() - t_case
    summary = {"case": case, "total_wall_s": round(total_s, 1),
               "candidates": [{k: v for k, v in c.to_dict().items() if k != "episodes"} for c in cands],
               "assessments": assessments}
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    lines = [f"# Case `{case_id}`: {args.target} on {args.model}", "",
             f"- Target node {node} in `{case['branch']}`; goal {np.round(goal_vmr, 1).tolist()} mm (VMR), {np.round(goal_m, 4).tolist()} m (organiser world).",
             f"- A* route: {' > '.join(plan.branches())}, {route_mm:.0f} mm, {len(plan.junctions)} bifurcations, planned in {plan_s:.1f} s.",
             f"- Guidewire travel {travel} mm: {'reachable' if reach['reachable_by_travel'] else '**out of reach**'} by insertion length."]
    if miss is not None:
        lines.append(f"- Entry-roll table: {roll:g}° reaches within {miss:.1f} mm of the target on their physics "
                     f"({'reachable' if miss <= SUCCESS_MM else '**probably out of reach**'}; success needs {SUCCESS_MM:g} mm).")
    lines += [f"- Budgets: demonstrations {args.bc_demos} / {args.bc_minutes:g} min, ACT {args.act_minutes:g} min"
              + (f", SAC {args.sac_minutes:g} min" if "sac" in methods else "")
              + f"; {args.eval_episodes} evaluation episodes per method, {args.max_steps} steps at most, {args.workers} workers. "
              f"Whole case: {total_s / 60:.1f} min.", "", "## Candidates", "", table, "",
              "Train time: demonstrations and model fitting for this target. Plan time: the rollouts that produced the trajectories "
              "(A*: planning + one kinematic rollout). Safety: `pass` = no structural rule fired, `review` = look at the flags, "
              "`reject` = never reached the target. Flags (thresholds from docs/SAFETY.md, placeholders): lumen = tip outside the "
              "vessel corridor; wrong_vessel = tip more than 12 mm into a vessel off the A* route; overshoot = insertion beyond the "
              "closest approach by more than 10 mm; force = peak force above 0.8 N; travel = beyond the robot's travel; speed = push "
              "above the lumen-radius envelope (advisory: reported, does not change the verdict).", "", "## Notes", ""]
    for c in cands:
        a = assessments[c.method]
        lines.append(f"- **{c.method}** ({c.description}): {a['success']} successful; worst off-route {a['worst'].get('off_route_mm', 0)} mm, "
                     f"overshoot {a['worst'].get('overshoot_mm', 0)} mm, max insertion {a['worst'].get('max_insertion_mm', 0)} mm.")
        lines += [f"  - {n}" for n in c.notes]
        for k, v in c.artifacts.items():
            lines.append(f"  - {k}: `{Path(v).relative_to(ROOT) if Path(v).is_absolute() and str(v).startswith(str(ROOT)) else v}`")
    lines += ["", "Files: `candidates/<method>.json` (all episodes, common format), `trajectory_<method>.csv` (the representative "
              "episode), `candidates.png`, `goals.json` (use with `--goals ... --split train` in eval_lockstep.py / demo_lockstep.py)."]
    (case_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not args.no_plot and cands:
        plot_candidates(case_dir / "candidates.png", anat, frame, cands, goal_vmr, plan.points,
                        title=f"{case_id}: {args.target} ({case['branch']}), representative trajectory per method")
    print()
    print(table)
    log(f"wrote {case_dir} in {total_s / 60:.1f} min")


if __name__ == "__main__":
    main()
