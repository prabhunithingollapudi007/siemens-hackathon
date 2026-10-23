"""Candidate trajectories for one patient case, from different methods, in one format.

Pre-operative planning produces several proposals for the same target (A* route following, the
calibrated entry-roll controller, a behaviour-cloned policy, an action-chunking policy, SAC ...).
A clinician or a safety layer compares them, so they must share one representation:

* ``EpisodeTrace``: one rollout, as the organiser's stack would log it: true guidewire tip after
  each 5 Hz control step (world **metres**), observed insertion (mm), the executed action, the true
  tip-to-goal distance (mm), peak contact force (N) and the episode's outcome.
* ``Candidate``: a method's episodes plus how long the method took to **train** (demonstrations and
  model fitting) and to **plan** (rollouts), both wall-clock seconds, and where its artefacts are.
* ``assess``: deterministic checks of every episode against the patient's anatomy, with the
  thresholds of ``stroke_nav.safety.SafetyConfig`` (lumen corridor, wrong-vessel depth, overshoot,
  contact force, speed envelope, robot travel). Thresholds are engineering placeholders
  (docs/SAFETY.md); the result says what a reviewer should look at, it is not a clinical judgement.
* ``summary_table`` / ``plot_candidates`` / ``write_trajectory_csv``: what ``scripts/plan_case.py``
  writes into the case folder.

Positions in traces are world metres (organiser frame); the checks convert to VMR mm through the
fitted ``OrganiserFrame``, never by hand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from stroke_nav.agents.roll_table import path_distance_m
from stroke_nav.safety.monitor import SafetyConfig

METHOD_ORDER = ("astar", "astar_theirs", "roll", "bc", "act", "sac")
# flags that are reported but do not turn a candidate's verdict into "review": the expert pushes at full
# speed in every vessel, so the speed envelope (a placeholder threshold) fires on every physics candidate
ADVISORY_FLAGS = {"speed"}


@dataclass
class EpisodeTrace:
    seed: int
    success: bool
    tip_m: np.ndarray  # (N, 3) true guidewire tip after each control step, world metres
    insertion_mm: np.ndarray  # (N,) observed guidewire insertion after each step
    actions: np.ndarray  # (N, 4) executed actions in [-1, 1]
    dist_mm: np.ndarray  # (N,) true tip-to-goal distance after each step
    peak_force_n: float = 0.0
    reason: str = ""
    wall_s: float = 0.0

    @property
    def steps(self) -> int:
        return int(len(self.actions))

    @property
    def min_dist_mm(self) -> float:
        return float(np.min(self.dist_mm)) if len(self.dist_mm) else float("nan")

    @property
    def final_dist_mm(self) -> float:
        return float(self.dist_mm[-1]) if len(self.dist_mm) else float("nan")

    def to_dict(self) -> dict:
        return {"seed": int(self.seed), "success": bool(self.success), "steps": self.steps,
                "peak_force_n": float(self.peak_force_n), "reason": self.reason, "wall_s": round(float(self.wall_s), 1),
                "min_dist_mm": round(self.min_dist_mm, 2), "final_dist_mm": round(self.final_dist_mm, 2),
                "tip_m": np.round(np.asarray(self.tip_m, float), 6).tolist(),
                "insertion_mm": np.round(np.asarray(self.insertion_mm, float), 2).tolist(),
                "actions": np.round(np.asarray(self.actions, float), 4).tolist(),
                "dist_mm": np.round(np.asarray(self.dist_mm, float), 2).tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodeTrace":
        return cls(seed=int(d["seed"]), success=bool(d["success"]), tip_m=np.asarray(d["tip_m"], float).reshape(-1, 3),
                   insertion_mm=np.asarray(d["insertion_mm"], float), actions=np.asarray(d["actions"], float).reshape(-1, 4),
                   dist_mm=np.asarray(d["dist_mm"], float), peak_force_n=float(d.get("peak_force_n", 0.0)),
                   reason=d.get("reason", ""), wall_s=float(d.get("wall_s", 0.0)))


@dataclass
class Candidate:
    method: str  # astar, astar_theirs, roll, bc, act, sac
    simulator: str  # "ours-kinematic" or "theirs-lockstep"
    description: str
    train_wall_s: float = 0.0  # demonstrations + model fitting
    plan_wall_s: float = 0.0  # rollouts that produced the trajectories
    episodes: list[EpisodeTrace] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return float(np.mean([e.success for e in self.episodes])) if self.episodes else 0.0

    def representative(self) -> EpisodeTrace | None:
        """The episode a reviewer should look at first: the fastest success, else the closest miss."""
        if not self.episodes:
            return None
        ok = [e for e in self.episodes if e.success]
        return min(ok, key=lambda e: e.steps) if ok else min(self.episodes, key=lambda e: e.min_dist_mm)

    def to_dict(self) -> dict:
        return {"method": self.method, "simulator": self.simulator, "description": self.description,
                "train_wall_s": round(float(self.train_wall_s), 1), "plan_wall_s": round(float(self.plan_wall_s), 1),
                "success_rate": self.success_rate, "artifacts": self.artifacts, "notes": self.notes,
                "episodes": [e.to_dict() for e in self.episodes]}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Candidate":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(method=d["method"], simulator=d["simulator"], description=d.get("description", ""),
                   train_wall_s=float(d.get("train_wall_s", 0.0)), plan_wall_s=float(d.get("plan_wall_s", 0.0)),
                   episodes=[EpisodeTrace.from_dict(e) for e in d.get("episodes", [])],
                   artifacts=dict(d.get("artifacts", {})), notes=list(d.get("notes", [])))


# ---------------------------------------------------------------------- safety assessment
def _allowed_speed(radius_mm: float, cfg: SafetyConfig) -> float:
    for min_radius, fraction in cfg.speed_zones:
        if radius_mm >= min_radius:
            return fraction
    return cfg.speed_zones[-1][1]


def assess_episode(trace: EpisodeTrace, anat, frame, route_points_mm: np.ndarray | None, route_nodes: set[int] | None,
                   travel_mm: float, cfg: SafetyConfig = SafetyConfig(), dt: float = 0.2) -> dict:
    """Checks of one trace against the anatomy (VMR mm). Returns the measured values and the rules they violate.

    * ``lumen``: tip farther outside the nearest vessel's lumen than the corridor allows (mm beyond the radius)
    * ``wrong_vessel``: tip in a vessel off the planned route for consecutive steps, farther along that vessel than
      the junction zone (``cfg.junction_after_mm``); the distance to the route polyline is reported too
    * ``overshoot``: insertion beyond the point of closest approach to the goal, after that point
    * ``force``: peak contact force above the monitor's limit
    * ``speed``: push commands above the lumen-radius speed envelope (steps)
    * ``travel``: insertion beyond the robot's guidewire travel
    """
    tip = frame.world_m_to_vmr_mm(np.asarray(trace.tip_m, float)) if len(trace.tip_m) else np.zeros((0, 3))
    nearest = np.array([anat.nearest_node(p) for p in tip], dtype=int)
    radius = anat.radius[nearest] if len(nearest) else np.zeros(0)
    d_center = np.linalg.norm(tip - anat.nodes[nearest], axis=1) if len(nearest) else np.zeros(0)
    off_lumen = float(np.max(np.maximum(d_center - radius, 0.0))) if len(tip) else 0.0

    off_route_mm, off_route_steps, wrong_depth = 0.0, 0, 0.0
    if route_points_mm is not None and len(route_points_mm) > 1 and len(tip):
        on_route = np.array([n in route_nodes for n in nearest]) if route_nodes else np.zeros(len(tip), bool)
        d_route = np.array([path_distance_m(route_points_mm, p) for p in tip])  # unit-agnostic polyline distance
        # the lumen of the route vessel at the point nearest to the tip: a tip hugging the aortic arch wall is
        # still inside the aorta, even where a side branch's first centerline nodes are nearer to it
        route_radius = anat.radius[[anat.nearest_node(p) for p in route_points_mm]]
        # smooth the lumen along the route (+-20 mm): at a junction the radius drops from the parent's to
        # the branch's within one node, while branch centerlines start inside the parent (junction gaps)
        # and the device cutting the corner is still in the parent's lumen for a couple of centimetres
        seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(route_points_mm, axis=0), axis=1))]
        smoothed = np.array([route_radius[(seg >= s - 20.0) & (seg <= s + 20.0)].max() for s in seg])
        nearest_route = np.argmin(np.linalg.norm(tip[:, None, :] - route_points_mm[None, :, :], axis=2), axis=1)
        off = (~on_route) & (d_route > smoothed[nearest_route] + cfg.corridor_mm)
        # one isolated sample is a junction artefact (the bent tip swings past a sibling's first nodes);
        # an entry into a wrong vessel shows as consecutive samples, as in the follower and the monitor
        runs = np.flatnonzero(np.diff(np.r_[0, off.astype(int), 0]))
        consecutive = np.any(runs[1::2] - runs[::2] >= 2) if len(runs) else False
        off_route_steps = int(np.sum(off)) if consecutive else 0
        off_route_mm = float(np.max(d_route[off])) if off_route_steps else 0.0
        if off_route_steps:
            # how far along the wrong vessel the tip went (arc from that branch's root). Branch centerlines
            # start inside their parent (junction gaps), so a tip hugging the arch wall is "nearest" to a
            # side branch's first nodes without having entered it; depth separates that from a real entry.
            roots = {}
            for n in nearest[off]:
                b = anat.branch_of(int(n))
                if b not in roots:
                    roots[b] = float(np.min(anat.arc[anat.branch_nodes(b)]))
                wrong_depth = max(wrong_depth, float(anat.arc[n] - roots[b]))

    overshoot = 0.0
    if len(trace.dist_mm):
        k = int(np.argmin(trace.dist_mm))
        if trace.dist_mm[k] <= 3 * 5.0:  # only meaningful once the tip came near the goal
            overshoot = float(np.max(trace.insertion_mm[k:]) - trace.insertion_mm[k])

    # stall: the robot keeps inserting while the tip does not move (the wire buckles or pushes against a wall)
    # measured from an anchor: a run of steps in which the tip's net displacement stays below 20 % of the
    # insertion gained since the anchor (in a bend the tip still moves about half of what is inserted)
    stall, anchor = 0.0, 0
    ins = np.asarray(trace.insertion_mm, float)
    for k in range(1, len(tip)):
        gain = ins[k] - ins[anchor]
        moved = float(np.linalg.norm(tip[k] - tip[anchor]))
        if gain > 2.0 and moved < 0.2 * gain:
            stall = max(stall, gain)
        elif ins[k] <= ins[anchor] or moved >= 0.2 * max(gain, 2.0):
            anchor = k

    allowed = np.array([_allowed_speed(r, cfg) for r in radius]) if len(radius) else np.zeros(0)
    push = np.abs(np.asarray(trace.actions, float)[:, 0]) if len(trace.actions) else np.zeros(0)
    n = min(len(push), len(allowed))
    speed_steps = int(np.sum(push[:n] > allowed[:n] + 1e-6))
    max_ins = float(np.max(trace.insertion_mm)) if len(trace.insertion_mm) else 0.0

    flags = []
    if off_lumen > cfg.corridor_mm:
        flags.append("lumen")
    if off_route_steps and wrong_depth > cfg.junction_after_mm:  # past the junction zone of the wrong vessel
        flags.append("wrong_vessel")
    if overshoot > cfg.overshoot_mm:
        flags.append("overshoot")
    if trace.peak_force_n > cfg.force_limit_N:
        flags.append("force")
    if speed_steps:
        flags.append("speed")
    if max_ins > travel_mm + 0.5:
        flags.append("travel")
    if stall > 30.0:  # successful runs on their rod show 16-23 mm of slack in the arch; a buckled wire shows 100+
        flags.append("stall")
    return {"seed": int(trace.seed), "success": bool(trace.success), "steps": trace.steps, "time_s": round(trace.steps * dt, 1),
            "min_dist_mm": round(trace.min_dist_mm, 2), "final_dist_mm": round(trace.final_dist_mm, 2),
            "off_lumen_mm": round(off_lumen, 2), "off_route_mm": round(off_route_mm, 2), "off_route_steps": off_route_steps,
            "wrong_vessel_depth_mm": round(wrong_depth, 1),
            "overshoot_mm": round(overshoot, 2), "peak_force_n": round(float(trace.peak_force_n), 4),
            "speed_violation_steps": speed_steps, "max_insertion_mm": round(max_ins, 1), "stall_mm": round(stall, 1), "flags": flags}


def assess(cand: Candidate, anat, frame, route_points_mm: np.ndarray | None, route_nodes: set[int] | None,
           travel_mm: float, cfg: SafetyConfig = SafetyConfig(), dt: float = 0.2) -> dict:
    """``assess_episode`` for every episode plus the candidate's verdict: ``reject`` when no episode
    reached the goal, ``review`` when any rule fired, else ``pass``."""
    eps = [assess_episode(e, anat, frame, route_points_mm, route_nodes, travel_mm, cfg, dt) for e in cand.episodes]
    flags = sorted({f for e in eps for f in e["flags"]})
    if not eps or not any(e["success"] for e in eps):
        verdict = "reject"
    elif set(flags) - ADVISORY_FLAGS:
        verdict = "review"
    else:
        verdict = "pass"
    ok = [e for e in eps if e["success"]]
    return {"method": cand.method, "verdict": verdict, "flags": flags, "episodes": eps,
            "success": f"{len(ok)}/{len(eps)}",
            "time_to_goal_s": round(float(np.mean([e["time_s"] for e in ok])), 1) if ok else None,
            "median_final_dist_mm": round(float(np.median([e["final_dist_mm"] for e in eps])), 2) if eps else None,
            "worst": {k: max(e[k] for e in eps) for k in ("off_lumen_mm", "off_route_mm", "wrong_vessel_depth_mm", "overshoot_mm",
                                                           "peak_force_n", "speed_violation_steps", "max_insertion_mm", "stall_mm")} if eps else {}}


# ---------------------------------------------------------------------- reports
def _fmt_s(seconds: float) -> str:
    if seconds < 1.0:
        return "<1 s"
    if seconds < 90.0:
        return f"{seconds:.0f} s"
    return f"{seconds / 60.0:.1f} min"


def summary_table(cands: list[Candidate], assessments: dict[str, dict]) -> str:
    """Markdown table: one row per candidate, train and plan time, outcome and safety verdict."""
    head = ("| Method | Simulator | Train time | Plan time | Success | Time to goal | Final dist. | Peak force | Safety | Flags |\n"
            "|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for c in sorted(cands, key=lambda c: METHOD_ORDER.index(c.method) if c.method in METHOD_ORDER else 99):
        a = assessments.get(c.method, {})
        ttg = a.get("time_to_goal_s")
        rows.append(f"| {c.method} | {c.simulator} | {_fmt_s(c.train_wall_s)} | {_fmt_s(c.plan_wall_s)} | {a.get('success', '-')} | "
                    f"{f'{ttg:.1f} s' if ttg is not None else '-'} | "
                    f"{a.get('median_final_dist_mm', '-')} mm | {a.get('worst', {}).get('peak_force_n', '-')} N | "
                    f"{a.get('verdict', '-')} | {', '.join(a.get('flags', [])) or '-'} |")
    return "\n".join([head, *rows])


def write_trajectory_csv(path: str | Path, trace: EpisodeTrace, frame, dt: float = 0.2) -> Path:
    """The representative trajectory as a table: step, time, the four actions, the true tip (world m and
    VMR mm), observed insertion and distance to the goal. For a safety layer or a reviewer."""
    import csv

    path = Path(path)
    tip_vmr = frame.world_m_to_vmr_mm(np.asarray(trace.tip_m, float)) if len(trace.tip_m) else np.zeros((0, 3))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "t_s", "v_guidewire", "w_guidewire", "v_catheter", "w_catheter",
                    "tip_x_m", "tip_y_m", "tip_z_m", "tip_x_mm_vmr", "tip_y_mm_vmr", "tip_z_mm_vmr",
                    "insertion_mm", "goal_dist_mm"])
        for k in range(trace.steps):
            a, t, v = trace.actions[k], trace.tip_m[k], tip_vmr[k]
            w.writerow([k, round(k * dt, 2), *[round(float(x), 4) for x in a], *[round(float(x), 6) for x in t],
                        *[round(float(x), 3) for x in v], round(float(trace.insertion_mm[k]), 2), round(float(trace.dist_mm[k]), 2)])
    return path


def plot_candidates(path: str | Path, anat, frame, cands: list[Candidate], goal_vmr_mm, route_points_mm=None,
                    title: str = "") -> Path:
    """AP and lateral views of the centerlines with every candidate's representative tip path and the goal."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"astar": "#2563eb", "astar_theirs": "#0891b2", "roll": "#d97706", "bc": "#dc2626", "act": "#7c3aed", "sac": "#059669"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    for name in anat.branch_names:
        p = anat.nodes[anat.branch_nodes(name)]
        axes[0].plot(-p[:, 0], p[:, 2], color="#9ca3af", lw=1)
        axes[1].plot(p[:, 1], p[:, 2], color="#9ca3af", lw=1)
    if route_points_mm is not None and len(route_points_mm):
        r = np.asarray(route_points_mm)
        axes[0].plot(-r[:, 0], r[:, 2], color="#111827", lw=3, alpha=0.25, label="A* route")
        axes[1].plot(r[:, 1], r[:, 2], color="#111827", lw=3, alpha=0.25)
    for c in cands:
        e = c.representative()
        if e is None or not len(e.tip_m):
            continue
        p = frame.world_m_to_vmr_mm(np.asarray(e.tip_m, float))
        col = colors.get(c.method, "#374151")
        label = f"{c.method} ({'success' if e.success else f'miss {e.min_dist_mm:.0f} mm'}, {c.success_rate:.0%})"
        axes[0].plot(-p[:, 0], p[:, 2], color=col, lw=1.8, label=label)
        axes[1].plot(p[:, 1], p[:, 2], color=col, lw=1.8)
        axes[0].plot(-p[-1, 0], p[-1, 2], "o", color=col, ms=5)
        axes[1].plot(p[-1, 1], p[-1, 2], "o", color=col, ms=5)
    g = np.asarray(goal_vmr_mm, float)
    for ax, (i, j, sx) in zip(axes, ((0, 2, -1), (1, 2, 1))):
        ax.plot(sx * g[i], g[j], "*", color="#16a34a", ms=16, label="target" if ax is axes[0] else None)
        ax.add_patch(plt.Circle((sx * g[i], g[j]), 5.0, fill=False, color="#16a34a", lw=1))
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    axes[0].set_title("AP view (patient left on right)")
    axes[0].set_xlabel("-x (mm)")
    axes[0].set_ylabel("z (mm)")
    axes[1].set_title("Lateral view")
    axes[1].set_xlabel("y (mm, anterior)")
    axes[0].legend(loc="lower left", fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
