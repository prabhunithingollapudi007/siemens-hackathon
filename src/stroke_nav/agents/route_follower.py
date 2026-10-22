"""Closed-loop follower of an A* route, driven only by measured tip poses.

Works on any simulator that reports where the guidewire tip is and where its bent tip points
(our kinematic model or the organiser's XPBD rod physics). The route comes from ``PathPlanner``;
the follower does not assume a tip model:

* localise the wire tip on the route (nearest route point near the previous estimate);
* before each bifurcation on the route, rotate the shaft until the tip's sideways bend points into
  the planned child vessel (and away from the other children), then advance slowly through it;
* if the tip ends up in another vessel, pull back past the bifurcation and try again with a
  different roll offset;
* keep the catheter a fixed distance behind the wire tip, and stop it where the vessel gets too
  narrow for it;
* stop at the goal, or when the robot's insertion travel is used up.

Rotation sign is learnt online (does a positive rotation command turn the tip forwards or
backwards about the vessel?), so the same controller runs on both simulators.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from stroke_nav.data.anatomy import Anatomy
from stroke_nav.planning.planner import Plan

ROLL_OFFSETS_DEG = (0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0)


@dataclass
class FollowerParams:
    dt: float = 0.2
    max_trans_mm_s: float = 10.0
    max_rot_rad_s: float = 1.0
    steer_window_mm: float = 30.0  # start aligning the tip this far before a bifurcation
    hold_window_mm: float = 12.0  # stop advancing within this distance until the tip is aligned
    commit_mm: float = 12.0  # a bifurcation counts as passed once the tip is this far into the child
    align_tol_deg: float = 35.0
    child_lookahead_mm: float = 10.0
    retract_mm: float = 30.0  # wrong vessel: pull the wire back this much insertion
    slow: float = 0.4  # advance speed (fraction) while passing a bifurcation
    cath_gap_mm: float = 40.0  # catheter tip stays this far behind the wire (in insertion)
    cath_limit_mm: float | None = None  # catheter stops here (arc on the route), e.g. where it no longer fits
    travel_limit_mm: float = 264.0  # organiser robot: module 1 travel from home is 264.5 mm
    goal_tol_mm: float = 4.0
    off_route_margin_mm: float = 2.5
    max_hold_steps: int = 40  # rotate in place at most this long (8 s) before creeping on
    rot_gain: float = 0.6  # fraction of the angle error corrected per control step
    arc_ratio_min: float = 0.8  # lower bound on route progress per mm of insertion (localisation window)


@dataclass
class TipObs:
    """Measured state in our VMR frame (mm)."""
    wire_tip: np.ndarray
    wire_dir: np.ndarray
    wire_ins_mm: float
    cath_ins_mm: float
    goal: np.ndarray


@dataclass
class FollowerState:
    s: float = 0.0
    junction: int = 0
    mode: str = "advance"  # advance | retract | done | travel_limit
    retract_until_ins: float = 0.0
    attempts: dict = field(default_factory=dict)
    rot_evidence: float = 0.0  # > 0: positive command turns the tip positively about the tangent
    rot_sign: float = 1.0  # flipped only on clear evidence (hysteresis)
    last_k: int = -1
    last_ins: float = 0.0
    hold_steps: int = 0
    prev_goal_dist: float = float("inf")
    off_count: int = 0
    last_lat: np.ndarray | None = None
    last_rot_cmd: float = 0.0
    log: list = field(default_factory=list)


class ClosedLoopRouteFollower:
    def __init__(self, anatomy: Anatomy, plan: Plan, params: FollowerParams | None = None):
        if not plan.feasible:
            raise ValueError(f"plan is not feasible: {plan.message}")
        self.anat, self.plan = anatomy, plan
        self.p = params or FollowerParams()
        self.P = np.asarray(plan.points, dtype=float)
        self.S = np.asarray(plan.s, dtype=float)
        self.R = np.asarray(plan.radius, dtype=float)
        self.route_nodes = set(int(n) for n in plan.nodes)
        self.junctions = [self._junction_info(j) for j in plan.junctions]
        self.state = FollowerState()

    # ------------------------------------------------------------------ setup
    def _junction_info(self, j: dict) -> dict:
        i, node = int(j["index"]), int(j["node"])
        k = self._index_at(self.S[i] + self.p.child_lookahead_mm)
        u_route = _unit(self.P[k] - self.P[i])
        others = []
        nxt = int(self.plan.nodes[min(i + 1, len(self.plan.nodes) - 1)])
        prv = int(self.plan.nodes[max(i - 1, 0)])
        for nb in self.anat.neighbors[node]:
            if nb in (nxt, prv):
                continue
            far = self.anat.walk_from(node, nb, int(self.p.child_lookahead_mm))
            others.append(_unit(self.anat.nodes[far] - self.anat.nodes[node]))
        return {"index": i, "s": float(self.S[i]), "node": node, "u_route": u_route, "u_others": others,
                "to_branch": j.get("to_branch", "")}

    def reset(self) -> None:
        self.state = FollowerState()

    # ------------------------------------------------------------------ control
    def act(self, o: TipObs) -> np.ndarray:
        st, p = self.state, self.p
        k = self._localise(o.wire_tip, o.wire_ins_mm)
        st.last_ins = o.wire_ins_mm
        s_tip = float(self.S[k])
        d_route = float(np.linalg.norm(o.wire_tip - self.P[k]))
        n_any = self.anat.nearest_node(o.wire_tip)
        d_any = float(np.linalg.norm(o.wire_tip - self.anat.nodes[n_any]))
        off_now = (n_any not in self.route_nodes and d_route > max(self.R[k], 2.0) + p.off_route_margin_mm
                   and d_route - d_any > p.off_route_margin_mm)
        st.off_count = st.off_count + 1 if off_now else 0
        off_route = st.off_count >= 2
        goal_dist = float(np.linalg.norm(o.wire_tip - o.goal))
        t = self._tangent(k)
        lat = _perp(o.wire_dir, t)
        self._learn_rotation_sign(lat, t, k)

        a = np.zeros(4)
        if goal_dist <= p.goal_tol_mm:
            st.mode = "done"
        elif st.mode == "retract":
            a[0] = -1.0
            if o.wire_ins_mm <= st.retract_until_ins:
                st.mode = "advance"
        elif off_route:
            j = min(st.junction, len(self.junctions) - 1)
            st.attempts[j] = st.attempts.get(j, 0) + 1
            st.mode = "retract"
            st.retract_until_ins = o.wire_ins_mm - p.retract_mm
            st.hold_steps = 0
            st.off_count = 0
            if j >= 0:
                st.s = max(0.0, min(st.s, self.junctions[j]["s"]) - p.retract_mm)
            st.log.append({"event": "wrong_vessel", "junction": j, "branch": self.anat.branch_of(n_any),
                           "attempt": st.attempts[j], "s": round(s_tip, 1)})
            a[0] = -1.0
        elif o.wire_ins_mm >= p.travel_limit_mm:
            st.mode = "travel_limit"
        else:
            st.mode = "advance"
            if not off_route:
                st.s = s_tip
            while st.junction < len(self.junctions) and s_tip > self.junctions[st.junction]["s"] + p.commit_mm and not off_route:
                st.hold_steps = 0
                st.log.append({"event": "passed", "junction": st.junction, "to": self.junctions[st.junction]["to_branch"]})
                st.junction += 1
            a[0] = 1.0
            if st.junction < len(self.junctions):
                J = self.junctions[st.junction]
                ahead = J["s"] - s_tip
                if ahead < p.steer_window_mm:
                    theta = self._steer_angle(J, lat, t, st.attempts.get(st.junction, 0))
                    if abs(theta) > np.radians(120) and abs(st.last_rot_cmd) > 0.2:
                        rot = float(np.sign(st.last_rot_cmd))  # far off: keep turning the same way (no dithering at +-180)
                    else:
                        rot = float(np.clip(p.rot_gain * theta / (p.max_rot_rad_s * p.dt), -1.0, 1.0)) * st.rot_sign
                    a[1] = rot
                    aligned = abs(np.degrees(theta)) <= p.align_tol_deg
                    if not aligned and ahead < p.hold_window_mm and st.hold_steps < p.max_hold_steps:
                        a[0] = 0.0
                        st.hold_steps += 1
                    elif not aligned and ahead < p.hold_window_mm:
                        a[0] = 0.2  # cannot align in place: creep forward, wrong-vessel check will catch a miss
                    elif ahead < p.hold_window_mm or ahead < 0:
                        a[0] = p.slow
                    else:
                        a[0] = 0.6
            # slow down for the goal; back off if the tip has gone past it
            if goal_dist < 15.0:
                a[0] = min(a[0], max(0.15, goal_dist / 15.0))
            near_end = s_tip >= self.S[-1] - 8.0
            if near_end and goal_dist < 15.0 and goal_dist > st.prev_goal_dist + 0.2:
                a[0] = -0.2
            elif s_tip >= self.S[-1] - 0.5 and goal_dist < 15.0:
                a[0] = min(a[0], 0.0)
        if st.mode in ("done", "travel_limit"):
            a[:] = 0.0
        # catheter: follow the wire at a gap, never past its limit
        cath_target = o.wire_ins_mm - p.cath_gap_mm
        if p.cath_limit_mm is not None:
            cath_target = min(cath_target, p.cath_limit_mm)
        cath_target = max(cath_target, 0.0)
        err = cath_target - o.cath_ins_mm
        a[2] = float(np.clip(err / (p.max_trans_mm_s * p.dt), -0.5, 1.0)) if st.mode not in ("done", "travel_limit") else 0.0
        st.last_lat, st.last_rot_cmd = lat, float(a[1])
        st.prev_goal_dist = goal_dist
        return a

    # ------------------------------------------------------------------ pieces
    def _steer_angle(self, J: dict, lat: np.ndarray, t: np.ndarray, attempt: int) -> float:
        want = _perp(J["u_route"], t)
        if np.linalg.norm(want) < 0.25 and J["u_others"]:  # route goes straight on: point away from the side branches
            want = -np.sum([_unit_or_zero(_perp(u, t)) for u in J["u_others"]], axis=0)
        else:
            want = _unit_or_zero(want) - 0.5 * np.sum([_unit_or_zero(_perp(u, t)) for u in J["u_others"]], axis=0) \
                if J["u_others"] else want
        want = _unit_or_zero(_perp(want, t))
        if np.linalg.norm(want) < 1e-6 or np.linalg.norm(lat) < 1e-6:
            return 0.0
        off = np.radians(ROLL_OFFSETS_DEG[attempt % len(ROLL_OFFSETS_DEG)])
        want = _rotate_about(want, t, off)
        lat = _unit(lat)
        return float(np.arctan2(np.dot(np.cross(lat, want), t), np.dot(lat, want)))

    def _learn_rotation_sign(self, lat: np.ndarray, t: np.ndarray, k: int) -> None:
        st = self.state
        same_place = st.last_k >= 0 and abs(int(k) - st.last_k) <= 3
        st.last_k = int(k)
        if (not same_place or st.last_lat is None or abs(st.last_rot_cmd) < 0.2
                or np.linalg.norm(lat) < 0.2 or np.linalg.norm(st.last_lat) < 0.2):
            return
        a, b = _unit(_perp(st.last_lat, t)), _unit(lat)
        turned = float(np.arctan2(np.dot(np.cross(a, b), t), np.dot(a, b)))
        if abs(turned) < np.radians(60):  # ignore whips
            # evidence in the frame of the command actually sent (which already includes rot_sign)
            st.rot_evidence = float(np.clip(0.9 * st.rot_evidence + np.sign(st.last_rot_cmd) * st.rot_sign * turned, -3, 3))
            if st.rot_evidence * st.rot_sign < -1.0:  # clear evidence the tip turns the other way
                st.rot_sign = -st.rot_sign

    def _localise(self, tip: np.ndarray, ins_mm: float) -> int:
        """Nearest route point, searched only where the tip can be given the insertion: the tip covers
        0.9-1.0 mm of route per mm inserted (measured on their rod), never more than the insertion."""
        p = self.p
        lo = self._index_at(p.arc_ratio_min * ins_mm - 15.0)
        hi = self._index_at(ins_mm + 8.0) + 1
        seg = self.P[lo:hi]
        return lo + int(np.argmin(np.linalg.norm(seg - tip, axis=1)))

    def _index_at(self, s: float) -> int:
        return int(np.clip(np.searchsorted(self.S, s), 0, len(self.S) - 1))

    def _tangent(self, k: int) -> np.ndarray:
        """Direction the shaft arrives with: chord over the last 10 mm of route (centerline joins at
        bifurcations kink, so a local tangent there is unreliable)."""
        a = self._index_at(self.S[k] - 10.0)
        if k - a < 2:
            return _unit(self.P[min(k + 5, len(self.P) - 1)] - self.P[k])
        return _unit(self.P[k] - self.P[a])


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _unit_or_zero(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(3)


def _perp(v, t):
    v = np.asarray(v, dtype=float)
    return v - np.dot(v, t) * t


def _rotate_about(v, axis, ang):
    axis = _unit(axis)
    return v * np.cos(ang) + np.cross(axis, v) * np.sin(ang) + axis * np.dot(axis, v) * (1 - np.cos(ang))
