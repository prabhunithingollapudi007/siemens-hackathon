"""Convert a planned path into a time-parameterised device trajectory for the simulator.

Steps:
1. Extend the path a few mm beyond the target (the guidewire leads past the clot).
2. Speed profile v(s) along the path: limited by the device's speed limit, curvature
   (v <= v_max / (1 + R_slow * kappa)), narrow lumen, and a slow zone before every bifurcation;
   then acceleration limits by forward/backward passes, ending at rest.
3. Roll schedule: for each bifurcation, the guidewire roll angle that makes its pre-curved tip pick
   the planned vessel. Computed with the simulator's own frame transport and branch rule, so the
   reference is consistent with what the simulator will do (minus its steering noise).
4. Joint-space reference sampled at the control rate: inserted length and roll of the guidewire and
   microcatheter over time. The wire pauses before a bifurcation if it has not finished rotating;
   the microcatheter follows ``catheter_gap_mm`` behind and finally advances to the target.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from stroke_nav.env.devices import WIRE, SimParams, TrackSim, device_sim_params
from stroke_nav.planning.constraints import VesselGeometry
from stroke_nav.planning.planner import Plan


CLUSTER_MM = 0.5  # junctions closer than this are crossed in the same step with one roll


@dataclass
class TrajectoryParams:
    a_max_mm_s2: float = 60.0
    v_min_mm_s: float = 3.0
    curve_slow_radius_mm: float = 8.0  # higher = slower in bends
    junction_speed_mm_s: float = 12.0
    junction_zone_mm: float = 6.0
    narrow_min_fraction: float = 0.25  # speed fraction at the narrowest passable lumen
    catheter_gap_mm: float = 10.0
    overshoot_mm: float = 6.0
    roll_grid: int = 72
    max_duration_s: float = 300.0
    catheter_limit_mm: float | None = None  # stop the catheter here (e.g. where lumen gets too narrow for it)


@dataclass
class Trajectory:
    plan: Plan
    params: TrajectoryParams
    dt: float
    # path profile (planned path + overshoot), per node
    nodes: list[int]
    s: np.ndarray
    points: np.ndarray
    curvature: np.ndarray
    radius: np.ndarray
    v_limit: np.ndarray  # speed limit before acceleration limits
    v_ref: np.ndarray  # final speed profile along s
    s_target: float
    roll_events: list[dict]
    # joint-space reference, per control step
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    L_wire: np.ndarray = field(default_factory=lambda: np.zeros(0))
    phi_wire: np.ndarray = field(default_factory=lambda: np.zeros(0))
    L_cath: np.ndarray = field(default_factory=lambda: np.zeros(0))
    phi_cath: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def duration_s(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0

    def summary(self) -> dict:
        dwell = int(np.sum(np.abs(np.diff(self.L_wire)) < 1e-6)) if len(self.L_wire) > 1 else 0
        return {
            "duration_s": round(self.duration_s, 1),
            "steps": int(len(self.t)),
            "path_length_mm": round(float(self.s[-1]), 1),
            "target_at_mm": round(self.s_target, 1),
            "mean_speed_mm_s": round(float(self.s_target / max(self.duration_s, 1e-9)), 1),
            "rotation_events": len(self.roll_events),
            "unreachable_junctions": sum(1 for e in self.roll_events if not e["reachable"]),
            "total_wire_rotation_deg": round(float(np.sum(np.abs(np.diff(np.unwrap(self.phi_wire))))) * 180 / np.pi, 0),
            "wire_dwell_steps": dwell,
        }

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "L_guidewire_mm", "roll_guidewire_deg", "L_microcatheter_mm", "roll_microcatheter_deg"])
            for row in zip(self.t, self.L_wire, np.degrees(self.phi_wire), self.L_cath, np.degrees(self.phi_cath)):
                w.writerow([f"{x:.4f}" for x in row])
        return path

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "anatomy": self.plan.anatomy.name,
            "device": self.plan.device.name,
            "plan": self.plan.summary(),
            "trajectory": self.summary(),
            "params": asdict(self.params),
            "dt_s": self.dt,
            "target_node": self.plan.nodes[-1],
            "path": [
                {"node": int(n), "s_mm": round(float(s), 2), "xyz_mm": [round(float(x), 2) for x in p],
                 "radius_mm": round(float(r), 3), "curvature_per_mm": round(float(k), 4), "v_ref_mm_s": round(float(v), 2)}
                for n, s, p, r, k, v in zip(self.nodes, self.s, self.points, self.radius, self.curvature, self.v_ref)
            ],
            "roll_events": self.roll_events,
        }
        path.write_text(json.dumps(data, indent=1), encoding="utf-8")
        return path


def build_trajectory(plan: Plan, sim_params: SimParams | None = None,
                     params: TrajectoryParams | None = None) -> Trajectory:
    if not plan.feasible or not plan.nodes:
        raise ValueError(f"cannot build a trajectory from an infeasible plan: {plan.message}")
    sp = sim_params or device_sim_params()
    tp = params or TrajectoryParams()
    a = plan.anatomy
    dev = plan.device

    nodes = list(plan.nodes) + _continuation(a, plan.nodes, tp.overshoot_mm)
    pts = a.nodes[nodes]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    s_target = float(s[len(plan.nodes) - 1])
    kappa = VesselGeometry(a).path_curvature(nodes)
    radius = a.radius[nodes]

    # ---- speed limits along s
    v_max = min(sp.guidewire.v_max_mm_s, sp.microcatheter.v_max_mm_s)
    v_curve = v_max / (1.0 + tp.curve_slow_radius_mm * kappa)
    span = max(dev.comfort_lumen_radius_mm - dev.min_lumen_radius_mm, 1e-6)
    frac = np.clip((radius - dev.min_lumen_radius_mm) / span, 0.0, 1.0)
    v_narrow = v_max * (tp.narrow_min_fraction + (1 - tp.narrow_min_fraction) * frac)
    v_lim = np.minimum(v_curve, v_narrow)
    junction_s = [s[i] for i in range(1, len(nodes) - 1) if a.degree[nodes[i]] >= 3]
    for sj in junction_s:
        zone = (s >= sj - tp.junction_zone_mm) & (s <= sj + 1.0)
        v_lim[zone] = np.minimum(v_lim[zone], tp.junction_speed_mm_s)
    v_lim = np.maximum(v_lim, tp.v_min_mm_s)
    wire_goal = float(s[-1])
    v_ref = _accel_limited(s, v_lim, tp.a_max_mm_s2, s_start=sp.initial_insertion_mm[0], s_end=wire_goal)

    events = roll_schedule(plan, nodes, sp, tp.roll_grid)
    traj = Trajectory(plan, tp, sp.dt, nodes, s, pts, kappa, radius, v_lim, v_ref, s_target, events)
    _sample_joint_reference(traj, sp, wire_goal)
    return traj


# ---------------------------------------------------------------------- roll schedule
def roll_schedule(plan: Plan, nodes: list[int], sp: SimParams, grid: int = 72) -> list[dict]:
    """Guidewire roll needed at each bifurcation, using the simulator's frame transport and rule."""
    a = plan.anatomy
    sim = TrackSim(a, sp)
    sim.reset(nodes[0], rng=np.random.default_rng(0))
    if sim.track != nodes[: len(sim.track)]:
        raise ValueError("initial insertion already deviates from the planned path")
    phis = np.linspace(0.0, 2 * np.pi, grid, endpoint=False)
    events: list[dict] = []
    phi_prev = 0.0
    for i in range(len(sim.track), len(nodes)):
        u = sim.track[-1]
        prev = sim.track[-2] if len(sim.track) > 1 else None
        cands = [c for c in a.neighbors[u] if c != prev]
        desired = nodes[i]
        if len(cands) > 1:
            t, n1 = sim._end_frame()
            ok = np.array([sim.choose(u, prev, cands, t, n1, ph, WIRE, noise=0.0) == desired for ph in phis])
            if not ok.all():
                s_here = float(sim.track_s[-1])
                cluster = events and abs(s_here - events[-1]["s_mm"]) < CLUSTER_MM
                if cluster:
                    # crossed in the same control step as the previous junction: one roll for both
                    joint = ok & events[-1]["_ok"]
                    if joint.any():
                        ok = joint
                reachable = bool(ok.any())
                phi = _nearest_arc_center(ok, phis, phi_prev) if reachable else phi_prev
                width = _arc_width_deg(ok, phis, phi)
                if cluster and reachable and events[-1]["_ok"][int(round(phi / (phis[1] - phis[0]))) % len(phis)]:
                    for e in events[::-1]:  # re-target the earlier junctions of this cluster
                        if abs(e["s_mm"] - s_here) >= CLUSTER_MM:
                            break
                        e["phi_deg"] = float(np.degrees(phi) % 360.0)
                        e["window_deg"] = width
                events.append({
                    "node": int(u),
                    "s_mm": float(sim.track_s[-1]),
                    "to_branch": a.branch_of(desired),
                    "from_branch": a.branch_of(prev) if prev is not None else a.branch_of(u),
                    "phi_deg": float(np.degrees(phi) % 360.0),
                    "window_deg": width,
                    "reachable": reachable,
                    "_ok": ok,
                })
                phi_prev = phi
        sim._append(desired)
    for e in events:
        e.pop("_ok", None)
    return events


def _nearest_arc_center(ok: np.ndarray, phis: np.ndarray, phi_now: float) -> float:
    from stroke_nav.agents.scripted import _nearest_arc_center as f

    return f(ok, phis, phi_now)


def _arc_width_deg(ok: np.ndarray, phis: np.ndarray, phi: float) -> float:
    step = phis[1] - phis[0]
    k = int(round((phi % (2 * np.pi)) / step)) % len(ok)
    if not ok[k]:
        return 0.0
    n, width = len(ok), 1
    for d in (1, -1):
        j = k
        while True:
            j = (j + d) % n
            if not ok[j] or width >= n:
                break
            width += 1
    return float(np.degrees(width * step))


# ---------------------------------------------------------------------- speed profile
def _accel_limited(s: np.ndarray, v_lim: np.ndarray, a_max: float, s_start: float, s_end: float) -> np.ndarray:
    v = v_lim.copy()
    i0 = int(np.searchsorted(s, s_start))
    v[: i0 + 1] = np.minimum(v[: i0 + 1], 0.0 if i0 < len(v) else v[: i0 + 1])
    for i in range(i0 + 1, len(s)):  # forward pass: accelerate from rest
        ds = s[i] - s[i - 1]
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * a_max * ds))
    v[-1] = 0.0  # come to rest at the end
    for i in range(len(s) - 2, -1, -1):  # backward pass: decelerate in time
        ds = s[i + 1] - s[i]
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a_max * ds))
    v[: i0 + 1] = np.maximum(v[: i0 + 1], 0.0)
    return v


# ---------------------------------------------------------------------- joint reference
def _sample_joint_reference(traj: Trajectory, sp: SimParams, wire_goal: float) -> None:
    tp, dt = traj.params, sp.dt
    s, v_ref = traj.s, traj.v_ref
    w_step = np.deg2rad(sp.guidewire.w_max_deg_s) * dt
    v_c_max = sp.microcatheter.v_max_mm_s
    L_w, L_c = float(sp.initial_insertion_mm[0]), float(sp.initial_insertion_mm[1])
    phi_w, phi_c = 0.0, 0.0
    v_prev = 0.0
    events = [e for e in traj.roll_events if e["reachable"]]
    j = 0
    out_t, out_Lw, out_pw, out_Lc, out_pc = [0.0], [L_w], [phi_w], [L_c], [phi_c]
    v_floor = tp.v_min_mm_s
    max_steps = int(tp.max_duration_s / dt)
    for k in range(1, max_steps):
        while j < len(events) and L_w > events[j]["s_mm"] + 1e-6:
            j += 1  # junction already passed
        # rotate toward the next junction's roll target
        aligned = True
        if j < len(events):
            err = _wrap(np.deg2rad(events[j]["phi_deg"]) - phi_w)
            phi_w += float(np.clip(err, -w_step, w_step))
            aligned = abs(_wrap(np.deg2rad(events[j]["phi_deg"]) - phi_w)) < np.deg2rad(2.0)
        # translate along the speed profile (with acceleration limit)
        v = max(float(np.interp(L_w, s, v_ref)), v_floor if L_w < wire_goal - 0.5 else 0.0)
        v = min(v, v_prev + tp.a_max_mm_s2 * dt)
        ds = min(v * dt, wire_goal - L_w)
        if j < len(events):
            s_j = events[j]["s_mm"]
            if not aligned:
                ds = min(ds, max(s_j - 0.5 - L_w, 0.0))  # wait before the junction until rotated
            nxt = next((e for e in events[j + 1:] if e["s_mm"] > s_j + CLUSTER_MM), None)
            if nxt is not None:
                # never pass two separate junctions in one step: stop just short of the next one
                ds = min(ds, max(nxt["s_mm"] - 1e-3 - L_w, 0.0))
        ds = max(ds, 0.0)
        L_w += ds
        v_prev = ds / dt
        # catheter: follow at a gap, then finish at the target
        wire_done = L_w >= wire_goal - 0.5
        goal_c = traj.s_target if wire_done else min(traj.s_target, L_w - tp.catheter_gap_mm)
        goal_c = min(goal_c, L_w)
        if tp.catheter_limit_mm is not None:
            goal_c = min(goal_c, tp.catheter_limit_mm)
        v_c = min(v_c_max, max(float(np.interp(L_c, s, v_ref)), v_floor))
        L_c += float(np.clip(goal_c - L_c, 0.0, v_c * dt))
        out_t.append(k * dt)
        out_Lw.append(L_w)
        out_pw.append(phi_w)
        out_Lc.append(L_c)
        out_pc.append(phi_c)
        cath_goal = traj.s_target if tp.catheter_limit_mm is None else min(traj.s_target, tp.catheter_limit_mm)
        if wire_done and L_c >= cath_goal - 0.25:
            break
    traj.t = np.array(out_t)
    traj.L_wire = np.array(out_Lw)
    traj.phi_wire = np.mod(np.array(out_pw), 2 * np.pi)
    traj.L_cath = np.array(out_Lc)
    traj.phi_cath = np.array(out_pc)


def _wrap(x: float) -> float:
    return float((x + np.pi) % (2 * np.pi) - np.pi)


def _continuation(a, nodes: list[int], length: float) -> list[int]:
    if len(nodes) < 2 or length <= 0:
        return []
    prev, cur, acc, out = nodes[-2], nodes[-1], 0.0, []
    while acc < length:
        nxt = [w for w in a.neighbors[cur] if w != prev]
        if not nxt:
            break
        same = [w for w in nxt if a.branch_id[w] == a.branch_id[cur]]
        d0 = a.nodes[cur] - a.nodes[prev]
        w = max(same or nxt, key=lambda w: float(np.dot(a.nodes[w] - a.nodes[cur], d0)))
        acc += float(np.linalg.norm(a.nodes[w] - a.nodes[cur]))
        out.append(int(w))
        prev, cur = cur, w
    return out
