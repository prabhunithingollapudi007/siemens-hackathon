"""Measure the gap between our kinematic simulator and the organiser's XPBD rod physics.

Plays the same action sequences in both and compares what the devices do:

    python scripts/measure_gap.py                      # all sequences, needs the organiser stack running
    python scripts/measure_gap.py --only wire_straight --skip-theirs   # ours only (re-uses saved recordings)

Their side: the stack from tools/organiser_stack.ps1 plus their physics page in headless Chrome
(started here if needed). Each run resets their page and follower first, and homes the robot after.
Our side: OrganiserCompatEnv with sensor noise off and a blank image, on the same cas-reference anatomy.

Writes results/gap/: <sequence>_theirs.csv (raw, ~1 kHz), <sequence>_steps.csv (both, per control
step), <sequence>.png, gap.json and report.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stroke_nav.organiser.spec import CAS_REFERENCE_GOALS_CM, DEFAULT_SPEC  # noqa: E402

DT = DEFAULT_SPEC.dt
OUT = ROOT / "results" / "gap"


# ---------------------------------------------------------------------- sequences
def _seg(n, gw_t=0.0, gw_r=0.0, ca_t=0.0, ca_r=0.0):
    return np.tile([gw_t, gw_r, ca_t, ca_r], (int(n), 1))


def planned_actions(target_spec: str = "right_carotid_external@21") -> np.ndarray:
    from stroke_nav.data.anatomy import Anatomy
    from stroke_nav.organiser.env import organiser_sim_params
    from stroke_nav.organiser.export import trajectory_policy_actions
    from stroke_nav.planning.planner import PathPlanner
    from stroke_nav.planning.trajectory import build_trajectory

    anat = Anatomy.load("0248_H_AOCERE_CAS")
    from stroke_nav.planning.constraints import GUIDEWIRE

    plan = PathPlanner(anat, GUIDEWIRE).plan(anat.default_start(), anat.resolve(target_spec))
    return trajectory_policy_actions(build_trajectory(plan, organiser_sim_params()))


def sequences() -> dict[str, tuple[str, np.ndarray]]:
    return {
        "wire_straight": ("guidewire alone, full speed (10 mm/s) for 30 s", _seg(150, 1.0)),
        "both_straight": ("guidewire and catheter together at full speed for 20 s", _seg(100, 1.0, 0, 1.0)),
        "catheter_follow": ("wire 120 mm ahead, then catheter follows over it, then wire pulled back 40 mm",
                            np.vstack([_seg(60, 1.0), _seg(50, 0, 0, 1.0), _seg(20, -1.0)])),
        "rotation_sweep": ("wire in 120 mm, full turn of the wire in place, 40 mm in, half turn back, 40 mm in",
                           np.vstack([_seg(60, 1.0), _seg(32, 0, 1.0), _seg(20, 1.0), _seg(16, 0, -1.0), _seg(20, 1.0)])),
        "planned_route": ("our planner + trajectory to right_carotid_external@21 (the cas-reference goal)", planned_actions()),
    }


def sweep_sequences(step_deg: int = 30, insert_mm: float = 260.0) -> dict[str, tuple[str, np.ndarray]]:
    """Roll -> branch map: rotate the wire to a fixed roll at the entry, then push it straight in."""
    out = {}
    for deg in range(0, 360, step_deg):
        n_rot = int(round(np.radians(deg) / DT))  # full-speed rotation (1 rad/s), rounded to whole steps
        out[f"roll_{deg:03d}"] = (f"rotate wire to {deg} deg at the entry, then push it {insert_mm:.0f} mm",
                                  np.vstack([_seg(n_rot, 0, 1.0), _seg(insert_mm / 10.0 / DT, 1.0)]))
    out["cath_only"] = ("catheter alone (no wire) pushed for 35 s, to find module 2's travel limit", _seg(175, 0, 0, 1.0))
    return out


# ---------------------------------------------------------------------- our simulator
def run_ours(actions: np.ndarray, seed: int = 0, sim_params=None) -> dict[str, np.ndarray]:
    from stroke_nav.env.devices import CATH, WIRE
    from stroke_nav.organiser.env import CompatConfig, OrganiserCompatEnv

    env = OrganiserCompatEnv(CompatConfig(image_mode="blank", noise=False))
    if sim_params is not None:
        env.inner.sim.p = sim_params
    goal = np.asarray(CAS_REFERENCE_GOALS_CM[0]) * env.frame.meters_per_unit
    env.reset(seed=seed, options={"goal_world_m": goal})
    sim = env.inner.sim
    rows = {k: [] for k in ("gw_L_mm", "cath_L_mm", "gw_roll_rad", "cath_roll_rad",
                            "gw_tip_vmr", "cath_tip_vmr", "gw_blocked", "cath_blocked")}
    for a in actions:
        env.step(a)
        rows["gw_L_mm"].append(sim.L[WIRE])
        rows["cath_L_mm"].append(sim.L[CATH])
        rows["gw_roll_rad"].append(env._roll[WIRE])
        rows["cath_roll_rad"].append(env._roll[CATH])
        rows["gw_tip_vmr"].append(sim.tracking_points(WIRE)[0])
        rows["cath_tip_vmr"].append(sim.tracking_points(CATH)[0])
        rows["gw_blocked"].append(bool(env._blocked[WIRE]))
        rows["cath_blocked"].append(bool(env._blocked[CATH]))
    out = {k: np.asarray(v) for k, v in rows.items()}
    out["start_vmr"] = env.anat.nodes[env.anat.default_start()]
    return out


# ---------------------------------------------------------------------- comparison
def compare(name: str, actions: np.ndarray, theirs_raw: dict, ours: dict, anat, frame) -> tuple[dict, list[dict]]:
    from stroke_nav.organiser.live.probe import sample_at_steps

    n = len(actions)
    th = sample_at_steps(theirs_raw, n, DT)
    cmd = {"gw": np.cumsum(actions[:, 0]) * DT * 10.0, "cath": np.cumsum(actions[:, 2]) * DT * 10.0}
    cmd_roll = {"gw": np.cumsum(actions[:, 1]) * DT, "cath": np.cumsum(actions[:, 3]) * DT}
    dist_from_start, _ = anat.distances_from(anat.default_start())
    steps = [{"step": k, "t_s": round((k + 1) * DT, 2),
              **dict(zip(("gw_trans", "gw_rot", "cath_trans", "cath_rot"), actions[k].round(4).tolist()))} for k in range(n)]
    summary = {"sequence": name, "steps": n}
    for dev in ("gw", "cath"):
        tip_world = np.stack([th[f"{dev}_tip_x"], th[f"{dev}_tip_y"], th[f"{dev}_tip_z"]], axis=1)
        tip_vmr = frame.world_m_to_vmr_mm(tip_world)
        node = np.array([anat.nearest_node(p) for p in tip_vmr])
        off_cl = np.linalg.norm(tip_vmr - anat.nodes[node], axis=1)
        radius = anat.radius[node]
        ins_theirs = th[f"{dev}_insertion_m"] * 1000.0
        arc_theirs = dist_from_start[node]
        ours_tip = ours[f"{dev}_tip_vmr"]
        ours_node = np.array([anat.nearest_node(p) for p in ours_tip])
        err = np.linalg.norm(tip_vmr - ours_tip, axis=1)
        br_theirs = [anat.branch_of(i) for i in node]
        br_ours = [anat.branch_of(i) for i in ours_node]
        moved = ins_theirs > 20.0
        slack = np.where(moved, arc_theirs / np.maximum(ins_theirs, 1e-6), np.nan)
        roll_theirs = np.unwrap(th[f"{dev}_roll_rad"])
        summary[dev] = {
            "commanded_insertion_mm": round(float(cmd[dev][-1]), 1),
            "their_insertion_mm": round(float(ins_theirs[-1]), 1),
            "our_insertion_mm": round(float(ours[f"{dev}_L_mm"][-1]), 1),
            "insertion_ratio_theirs_vs_command": _ratio(ins_theirs[-1], cmd[dev][-1]),
            "their_lag_mm_mean": round(float(np.mean(cmd[dev] - ins_theirs)), 2),
            "commanded_roll_deg": round(float(np.degrees(cmd_roll[dev][-1])), 1),
            "their_roll_deg": round(float(np.degrees(roll_theirs[-1])), 1),
            "our_roll_deg": round(float(np.degrees(ours[f"{dev}_roll_rad"][-1])), 1),
            "tip_error_mm_mean": round(float(np.mean(err)), 1),
            "tip_error_mm_final": round(float(err[-1]), 1),
            "tip_error_mm_max": round(float(err.max()), 1),
            "their_tip_off_centerline_mm_median": round(float(np.median(off_cl)), 2),
            "their_tip_outside_lumen_fraction": round(float(np.mean(off_cl > radius)), 3),
            "their_arc_per_insertion_median": round(float(np.nanmedian(slack)), 3) if np.isfinite(slack).any() else None,
            "their_branch_sequence": _runs(br_theirs),
            "our_branch_sequence": _runs(br_ours),
            "same_branch_fraction": round(float(np.mean([a == b for a, b in zip(br_theirs, br_ours)])), 3),
            "their_peak_force_n_max": round(float(np.nanmax(th[f"{dev}_peak_force_n"])), 4),
            "their_contacts_max": int(np.nanmax(th[f"{dev}_contacts"])),
            "our_blocked_steps": int(ours[f"{dev}_blocked"].sum()),
        }
        for k in range(n):
            steps[k].update({
                f"{dev}_cmd_mm": round(float(cmd[dev][k]), 2), f"{dev}_their_ins_mm": round(float(ins_theirs[k]), 2),
                f"{dev}_our_ins_mm": round(float(ours[f"{dev}_L_mm"][k]), 2),
                f"{dev}_cmd_roll_deg": round(float(np.degrees(cmd_roll[dev][k])), 1),
                f"{dev}_their_roll_deg": round(float(np.degrees(roll_theirs[k])), 1),
                f"{dev}_our_roll_deg": round(float(np.degrees(ours[f"{dev}_roll_rad"][k])), 1),
                f"{dev}_tip_err_mm": round(float(err[k]), 2), f"{dev}_their_off_cl_mm": round(float(off_cl[k]), 2),
                f"{dev}_their_branch": br_theirs[k], f"{dev}_our_branch": br_ours[k],
                f"{dev}_their_arc_mm": round(float(arc_theirs[k]), 1),
                f"{dev}_their_force_n": round(float(th[f"{dev}_peak_force_n"][k]), 4),
                **{f"{dev}_their_tip_vmr_{a}": round(float(tip_vmr[k, i]), 2) for i, a in enumerate("xyz")},
                **{f"{dev}_our_tip_vmr_{a}": round(float(ours_tip[k, i]), 2) for i, a in enumerate("xyz")}})
    return summary, steps


def _ratio(a, b):
    return round(float(a / b), 3) if abs(b) > 1e-6 else None


def _runs(labels: list[str]) -> list[str]:
    out = []
    for lab in labels:
        if not out or out[-1] != lab:
            out.append(lab)
    return out


def plot(name: str, title: str, steps: list[dict], anat, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {k: np.array([r[k] for r in steps]) for k in steps[0] if not isinstance(steps[0][k], str)}
    t = col["t_s"]
    fig, ax = plt.subplots(1, 4, figsize=(22, 5))
    for dev, c in (("gw", "tab:blue"), ("cath", "tab:orange")):
        ax[0].plot(t, col[f"{dev}_cmd_mm"], color=c, ls=":", lw=1, label=f"{dev} commanded")
        ax[0].plot(t, col[f"{dev}_their_ins_mm"], color=c, lw=2, label=f"{dev} theirs")
        ax[0].plot(t, col[f"{dev}_our_ins_mm"], color=c, ls="--", lw=1.5, label=f"{dev} ours")
        ax[1].plot(t, col[f"{dev}_cmd_roll_deg"], color=c, ls=":", lw=1, label=f"{dev} commanded")
        ax[1].plot(t, col[f"{dev}_their_roll_deg"], color=c, lw=2, label=f"{dev} theirs")
        ax[1].plot(t, col[f"{dev}_our_roll_deg"], color=c, ls="--", lw=1.5, label=f"{dev} ours")
        ax[2].plot(t, col[f"{dev}_tip_err_mm"], color=c, label=f"{dev} tip distance theirs-ours")
        ax[2].plot(t, col[f"{dev}_their_off_cl_mm"], color=c, ls=":", label=f"{dev} theirs: off centerline")
    ax[0].set(xlabel="time (s)", ylabel="insertion (mm)", title="insertion")
    ax[1].set(xlabel="time (s)", ylabel="roll (deg)", title="shaft rotation")
    ax[2].set(xlabel="time (s)", ylabel="mm", title="tip difference")
    for a in ax[:3]:
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    ax[3].scatter(anat.nodes[:, 0], anat.nodes[:, 2], s=0.3, color="0.8")
    for dev, c in (("gw", "tab:blue"), ("cath", "tab:orange")):
        ax[3].plot(col[f"{dev}_their_tip_vmr_x"], col[f"{dev}_their_tip_vmr_z"], color=c, lw=2, label=f"{dev} theirs")
        ax[3].plot(col[f"{dev}_our_tip_vmr_x"], col[f"{dev}_our_tip_vmr_z"], color=c, ls="--", lw=1.5, label=f"{dev} ours")
    ax[3].set(xlabel="x (mm)", ylabel="z (mm)", title="tip paths (frontal)", aspect="equal")
    ax[3].legend(fontsize=7)
    fig.suptitle(f"{name}: {title}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_report(summaries: list[dict], descriptions: dict, path: Path) -> None:
    lines = ["# Simulator gap: ours (kinematic) vs organiser (XPBD rod physics)", "",
             "Same normalised actions in both, 5 Hz, 10 mm/s and 1 rad/s at |a| = 1. Positions in our VMR frame (mm).", ""]
    for s in summaries:
        lines += [f"## {s['sequence']}", "", descriptions[s["sequence"]], "",
                  "| | guidewire | catheter |", "|---|---|---|"]
        keys = [k for k in s["gw"] if not isinstance(s["gw"][k], list)]
        for k in keys:
            lines.append(f"| {k} | {s['gw'][k]} | {s['cath'][k]} |")
        for dev, label in (("gw", "guidewire"), ("cath", "catheter")):
            lines.append(f"\n{label} branches, theirs: {' > '.join(s[dev]['their_branch_sequence'])}  ")
            lines.append(f"{label} branches, ours: {' > '.join(s[dev]['our_branch_sequence'])}")
        lines += ["", f"![{s['sequence']}]({s['sequence']}.png)", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None, help="sequence names to run")
    ap.add_argument("--skip-theirs", action="store_true", help="re-use results/gap/<seq>_theirs.csv instead of running their stack")
    ap.add_argument("--sweep", action="store_true", help="also run the roll -> branch sweep and the catheter travel test")
    ap.add_argument("--visible", action="store_true", help="show their physics page instead of running it headless")
    args = ap.parse_args()

    from stroke_nav.data.anatomy import Anatomy
    from stroke_nav.organiser.frame import default_frame
    from stroke_nav.organiser.live.probe import read_recording, run_probe
    from stroke_nav.organiser.live.renderer import PhysicsPage

    OUT.mkdir(parents=True, exist_ok=True)
    anat, frame = Anatomy.load("0248_H_AOCERE_CAS"), default_frame()
    seqs = sequences()
    if args.sweep or (args.only and any(n.startswith(("roll_", "cath_only")) for n in args.only)):
        seqs.update(sweep_sequences())
    names = args.only or [n for n in seqs if args.sweep or not n.startswith(("roll_", "cath_only"))]
    if not args.skip_theirs:
        info = PhysicsPage().start(headless=not args.visible).state()
        print(f"[gap] physics page ready: {info['mesh_info'][:70]}")
    summaries = []
    for name in names:
        desc, actions = seqs[name]
        raw_path = OUT / f"{name}_theirs.csv"
        print(f"[gap] {name}: {len(actions)} steps ({len(actions) * DT:.0f} s)", flush=True)
        if args.skip_theirs:
            theirs = read_recording(raw_path)
        else:
            theirs = run_probe(actions, DT)
            with open(raw_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(theirs.keys())
                w.writerows(np.column_stack(list(theirs.values())).tolist())
        ours = run_ours(actions)
        summary, steps = compare(name, actions, theirs, ours, anat, frame)
        with open(OUT / f"{name}_steps.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(steps[0]))
            w.writeheader()
            w.writerows(steps)
        plot(name, desc, steps, anat, OUT / f"{name}.png")
        summaries.append(summary)
        g = summary["gw"]
        print(f"      wire: cmd {g['commanded_insertion_mm']} mm, theirs {g['their_insertion_mm']}, ours {g['our_insertion_mm']}; "
              f"tip error mean {g['tip_error_mm_mean']} mm, final {g['tip_error_mm_final']}; "
              f"theirs {' > '.join(g['their_branch_sequence'])} | ours {' > '.join(g['our_branch_sequence'])}", flush=True)
    prev = {}
    gp = OUT / "gap.json"
    if gp.exists() and args.only:
        prev = {s["sequence"]: s for s in json.loads(gp.read_text(encoding="utf-8"))}
    prev.update({s["sequence"]: s for s in summaries})
    allsum = [prev[k] for k in seqs if k in prev]
    gp.write_text(json.dumps(allsum, indent=2, default=float), encoding="utf-8")
    write_report(allsum, {k: v[0] for k, v in seqs.items()}, OUT / "report.md")
    print(f"[gap] wrote {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
