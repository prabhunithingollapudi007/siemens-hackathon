"""Device-aware path planning on the centerline graph.

Search space: directed edges (u -> v) of the centerline graph, so the cost of entering node w can
depend on where the device came from (curvature and bifurcation take-off angle need the incoming
direction). Search: A* with the straight-line distance to the target as heuristic, scaled by the
length and time weights (admissible: every edge costs at least ``length * w_length`` plus
``length / v_max * w_time``).

Cost of moving v -> w after arriving from u (all terms in mm-equivalents; see ``PlannerWeights``):

    soft, per edge of length L (mm) into a node of lumen radius r:
        w_length * L                                           inserted length ("robot size")
      + w_time   * L / (v_max * speed_fraction(r, junction))   estimated travel time (the safety
                                                               speed envelope by lumen radius and
                                                               the slow zone around bifurcations)
      + w_narrow * L * narrowness(r)                           lumen tighter than comfortable (damage)
      + w_curve  * L * (kappa / kappa_max) ** 2                bending effort against the device limit
      + w_curve_int * kappa * L                                curvature integral (radians of turning:
                                                               wall contact along bends, damage)
      + w_force  * L * force(u, v, w)                          predicted contact force per mm: a
                                                               supplied model (N) or the built-in proxy
      + w_branch                                               if v is a bifurcation and w leaves the vessel
      + w_takeoff * excess angle                               turn sharper than takeoff_comfort_deg

    hard (edge removed):
        lumen radius at w < device.min_lumen_radius_mm         (always)
        w lies in a forbidden branch                            (always, when listed)
        kappa(u, v, w) > device.max_curvature_per_mm            (only with hard_curvature=True)

Every plan carries ``Plan.costs``: the unweighted totals of these terms, which are the evaluation
of a route before anything moves: inserted length (mm), estimated time (s), curvature integral (rad),
length in narrow lumen (mm), force proxy, bifurcations and vessel changes, and whether the route
stays within the robot's travel. ``PathPlanner.replan`` plans again from the current tip, warm-started
with the previous route. Dijkstra (plain shortest path, no device limits) is kept as the baseline.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from stroke_nav.data.anatomy import Anatomy
from stroke_nav.planning.constraints import DeviceSpec, VesselGeometry, get_device

# the organiser robot's speed limit at |v| = 1 (10 mm/s) and the runtime safety monitor's speed
# envelope (docs/SAFETY.md R1): (min lumen radius mm, fraction of v_max), first matching row wins
V_MAX_MM_S = 10.0
SPEED_ZONES = ((5.0, 1.0), (2.0, 0.5), (0.0, 0.25))


@dataclass
class PlannerWeights:
    # soft costs (mm-equivalents)
    length: float = 1.0  # per mm of inserted length
    time: float = 0.0  # per second of estimated travel time
    narrow: float = 4.0  # per mm in lumen narrower than comfortable, scaled by narrowness (0..1)
    curvature: float = 1.0  # bending effort: per mm times (kappa / kappa_max) ** 2
    curvature_integral: float = 0.0  # per radian of total turning along the route
    force: float = 0.0  # per (N or proxy unit) x mm of predicted contact force
    branch: float = 5.0  # every vessel change at a bifurcation
    takeoff: float = 0.2  # per degree above takeoff_comfort_deg
    takeoff_comfort_deg: float = 60.0
    # hard constraints
    hard_curvature: bool = False
    forbidden_branches: tuple[str, ...] = ()  # e.g. ("anterior_connector",) to exclude collateral routes
    # the time model
    v_max_mm_s: float = V_MAX_MM_S
    speed_zones: tuple[tuple[float, float], ...] = SPEED_ZONES
    junction_fraction: float = 0.3  # speed fraction within junction_zone_mm of a bifurcation
    junction_zone_mm: float = 12.0

    @classmethod
    def profile(cls, name: str, **overrides) -> "PlannerWeights":
        """Named trade-offs: ``shortest`` (inserted length only, device limits kept), ``fastest``
        (estimated time), ``gentle`` (damage terms dominate), ``balanced`` (the defaults plus time
        and damage at moderate weight)."""
        base = {
            "shortest": dict(length=1.0, time=0.0, narrow=0.0, curvature=0.0, curvature_integral=0.0, force=0.0,
                             branch=0.0, takeoff=0.0),
            "fastest": dict(length=0.0, time=10.0, narrow=0.0, curvature=0.0, curvature_integral=0.0, force=0.0,
                            branch=0.0, takeoff=0.0),
            "gentle": dict(length=1.0, time=0.0, narrow=8.0, curvature=4.0, curvature_integral=20.0, force=20.0,
                           branch=10.0, takeoff=0.5),
            "balanced": dict(length=1.0, time=2.0, narrow=4.0, curvature=1.0, curvature_integral=5.0, force=5.0,
                             branch=5.0, takeoff=0.2),
        }
        if name not in base:
            raise KeyError(f"unknown profile {name!r}; choose from {sorted(base)}")
        return cls(**{**base[name], **overrides})


@dataclass
class Plan:
    """A planned route plus everything needed to judge and execute it."""

    anatomy: Anatomy
    device: DeviceSpec
    nodes: list[int]
    method: str
    cost: float
    feasible: bool
    message: str = ""
    s: np.ndarray = field(default_factory=lambda: np.zeros(0))  # arc length (mm)
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    radius: np.ndarray = field(default_factory=lambda: np.zeros(0))
    curvature: np.ndarray = field(default_factory=lambda: np.zeros(0))
    junctions: list[dict] = field(default_factory=list)  # bifurcations passed on the route
    expanded: int = 0
    costs: dict = field(default_factory=dict)  # unweighted totals of the cost terms (the route's evaluation)

    @property
    def length_mm(self) -> float:
        return float(self.s[-1]) if len(self.s) else 0.0

    def branches(self) -> list[str]:
        out: list[str] = []
        for n in self.nodes:
            b = self.anatomy.branch_of(n)
            if not out or out[-1] != b:
                out.append(b)
        return out

    def summary(self) -> dict:
        d = self.device
        if not self.nodes:
            return {"method": self.method, "device": d.name, "feasible": False, "message": self.message}
        bend_r = 1.0 / np.maximum(self.curvature, 1e-9)
        c = self.costs
        return {
            "method": self.method,
            "device": d.name,
            "feasible": self.feasible,
            "message": self.message,
            "length_mm": round(self.length_mm, 1),
            "est_time_s": round(c.get("est_time_s", float("nan")), 1),
            "curvature_integral_rad": round(c.get("curvature_integral_rad", float("nan")), 2),
            "narrow_mm": round(c.get("narrow_mm", float("nan")), 1),
            "force_proxy": round(c.get("force_proxy", float("nan")), 2),
            "rejoin_mm": round(c.get("rejoin_mm", 0.0), 1),
            "cost": round(self.cost, 1),
            "branches": " > ".join(self.branches()),
            "bifurcations": len(self.junctions),
            "vessel_changes": sum(1 for j in self.junctions if j["changes_vessel"]),
            "min_lumen_radius_mm": round(float(self.radius.min()), 2),
            "required_lumen_radius_mm": round(d.min_lumen_radius_mm, 3),
            "min_bend_radius_mm": round(float(bend_r.min()), 2),
            "device_min_bend_radius_mm": d.min_bend_radius_mm,
            "tight_bends": int(np.sum(bend_r < d.min_bend_radius_mm)),
            "max_takeoff_deg": round(max((j["turn_deg"] for j in self.junctions), default=0.0), 0),
            "nodes_expanded": self.expanded,
        }


class PathPlanner:
    def __init__(self, anatomy: Anatomy, device: str | DeviceSpec = "microcatheter",
                 weights: PlannerWeights | None = None, geometry: VesselGeometry | None = None,
                 force_model: Callable[[int, int, int], float] | None = None):
        """``force_model(u, v, w)`` predicts the contact force (N) per mm of advance into w from u via
        v, e.g. fitted on the organiser's physics. Without it the force term uses a proxy: bending
        effort times narrowness, i.e. the device pressed against the wall of a tight bend."""
        self.anat = anatomy
        self.device = get_device(device)
        self.w = weights or PlannerWeights()
        self.geo = geometry or VesselGeometry(anatomy)
        self.force_model = force_model
        self._junction_dist: np.ndarray | None = None

    # ------------------------------------------------------------------ public API
    def plan(self, start: int, target: int, came_from: int | None = None) -> Plan:
        """Device-aware A* from start to target. ``came_from``: the node behind the start (the
        device's direction), so the first turn is costed and the device does not reverse into it."""
        start, target = int(start), int(target)
        fail = self._check_ends(start, target)
        if fail is not None:
            return fail
        path, cost, expanded = self._search(start, target, -1 if came_from is None else int(came_from), None)
        if path is None:
            p = self._fail(self._why_unreachable(start, target))
            p.expanded = expanded
            return p
        return self._make_plan(path, cost, "astar", expanded)

    def replan(self, current, target: int, previous: Plan | None = None, came_from: int | None = None,
               local_radius_mm: float | None = 40.0) -> Plan:
        """Plan again from the current tip (a node id or an xyz point in mm), warm-started with
        ``previous``.

        * On the previous route: its remainder is returned without a search (``astar-resume``).
        * Off it: a **local** search (``astar-warm``) that only expands nodes within ``local_radius_mm``
          of the previous route or of the tip, and closes as soon as no open state can beat the best
          splice found: reaching any node of the previous route offers a complete solution through
          its remaining cost-to-go. ``costs["rejoin_mm"]`` is the length of the new route before it
          rejoins the previous one (the way back out of a wrong vessel).
        * If the local search finds nothing (the vessel the tip is in leads elsewhere, the previous
          route is now forbidden ...), a full plan from the tip is returned (``astar``).
        """
        a = self.anat
        start = int(current) if np.isscalar(current) else a.nearest_node(np.asarray(current, dtype=float))
        target = int(target)
        if previous is None or not previous.feasible or not previous.nodes:
            return self.plan(start, target, came_from)
        if start in previous.nodes:
            i = previous.nodes.index(start)
            rest = previous.nodes[i:]
            plan = self._make_plan(rest, self.path_cost(rest), "astar-resume", 0)
            plan.costs["rejoin_mm"] = 0.0
            return plan
        fail = self._check_ends(start, target)
        if fail is not None:
            return fail
        old_index = {n: i for i, n in enumerate(previous.nodes)}
        ctg = self._cost_to_go(previous.nodes)
        came = -1 if came_from is None else int(came_from)
        best: tuple[list[int], float, int] | None = None
        if local_radius_mm is not None:
            # only the neighbourhood of the tip: the way back to the previous route is found there, and the
            # route's own cost-to-go completes the solution, so the search stays small however long the route.
            # Local means local: a cheaper way that rejoins outside the circle is not looked for.
            d_tip = np.linalg.norm(a.nodes - a.nodes[start], axis=1)
            path, cost, expanded = self._search(start, target, came, (old_index, ctg, previous.nodes), d_tip <= local_radius_mm)
            if path is not None:
                best = (path, cost, expanded)
        if best is None:
            path, cost, expanded = self._search(start, target, came, (old_index, ctg, previous.nodes), None)
            if path is None:
                return self.plan(start, target, came_from)
            best = (path, cost, expanded)
        plan = self._make_plan(best[0], best[1], "astar-warm", best[2])
        plan.costs["rejoin_mm"] = self._rejoin_mm(best[0], old_index)
        return plan

    def _rejoin_mm(self, path: list[int], old_index: dict[int, int]) -> float:
        """Length of ``path`` before its first node on the previous route: the way back out of a wrong
        vessel when the route rejoins behind the tip, or the way ahead to it (0 when it starts there)."""
        a = self.anat
        rejoin = next((k for k, n in enumerate(path) if n in old_index), None)
        if rejoin is None or rejoin == 0:
            return 0.0
        pts = a.nodes[path[: rejoin + 1]]
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    def shortest(self, start: int, target: int) -> Plan:
        """Plain Dijkstra shortest path that ignores device limits (baseline)."""
        try:
            path = self.anat.route(int(start), int(target))
        except ValueError as e:
            return self._fail(str(e), method="dijkstra")
        plan = self._make_plan(path, float("nan"), "dijkstra", 0)
        plan.cost = self.path_cost(path)
        violations = int(np.sum(plan.radius < self.device.min_lumen_radius_mm))
        if violations:
            plan.feasible = False
            plan.message = (f"{violations} nodes narrower than {self.device.min_lumen_radius_mm:.2f} mm "
                            f"(min {plan.radius.min():.2f} mm in {self.anat.branch_of(path[int(np.argmin(plan.radius))])})")
        return plan

    # ------------------------------------------------------------------ search
    def _check_ends(self, start: int, target: int) -> Plan | None:
        a, r_min = self.anat, self.device.min_lumen_radius_mm
        if a.radius[start] < r_min:
            return self._fail(f"start node lumen ({a.radius[start]:.2f} mm) is narrower than the device needs ({r_min:.2f} mm)")
        if a.radius[target] < r_min:
            return self._fail(f"target lumen ({a.radius[target]:.2f} mm) is narrower than the device needs ({r_min:.2f} mm)")
        return None

    def _search(self, start: int, target: int, came_from: int, warm, allowed: np.ndarray | None = None):
        """A* over directed edges. ``warm`` = (index of node on the previous route, cost-to-go per index,
        previous nodes): popping such a node offers a complete solution through the previous route; the
        search ends once no open state can beat the best one found. ``allowed``: boolean mask of the
        nodes the search may expand (a local replan)."""
        a, w = self.anat, self.w
        goal = a.nodes[target]
        scale = w.length + w.time / max(w.v_max_mm_s, 1e-9)

        def h(n: int) -> float:
            return scale * float(np.linalg.norm(a.nodes[n] - goal))

        g: dict[tuple[int, int], float] = {(came_from, start): 0.0}
        parent: dict[tuple[int, int], tuple[int, int] | None] = {(came_from, start): None}
        heap = [(h(start), 0.0, came_from, start)]
        expanded = 0
        best: tuple[float, list[int]] | None = None
        while heap:
            f, gc, prev, node = heapq.heappop(heap)
            state = (prev, node)
            if gc > g.get(state, np.inf) + 1e-9:
                continue
            if best is not None and f >= best[0] - 1e-9:
                break
            expanded += 1
            if node == target:
                path = self._reconstruct(parent, state)
                if best is None or gc < best[0]:
                    best = (gc, path)
                break  # the first target pop is optimal (consistent heuristic); warm splices were offered earlier
            if warm is not None and node in warm[0]:
                i = warm[0][node]
                total = gc + warm[1][i]
                if best is None or total < best[0]:
                    best = (total, self._reconstruct(parent, state) + list(warm[2][i + 1:]))
            for nxt in a.neighbors[node]:
                if nxt == prev or (allowed is not None and not allowed[nxt]):
                    continue
                step = self.transition_cost(prev, node, nxt)
                if not np.isfinite(step):
                    continue
                ns = (node, nxt)
                ng = gc + step
                if ng < g.get(ns, np.inf) - 1e-9:
                    g[ns] = ng
                    parent[ns] = state
                    heapq.heappush(heap, (ng + h(nxt), ng, node, nxt))
        if best is None:
            return None, float("inf"), expanded
        return best[1], best[0], expanded

    # ------------------------------------------------------------------ costs
    def junction_distance(self, node: int) -> float:
        """Arc distance (mm) from ``node`` to the nearest bifurcation (cached for the whole graph)."""
        if self._junction_dist is None:
            a = self.anat
            d = np.full(len(a.nodes), np.inf)
            for j in a.junctions():
                d = np.minimum(d, a.distances_from(int(j))[0])
            self._junction_dist = d
        return float(self._junction_dist[node])

    def speed_fraction(self, w: int) -> float:
        """Fraction of v_max allowed at node w: the lumen-radius envelope, slower near bifurcations."""
        r, wt = self.anat.radius[w], self.w
        frac = wt.speed_zones[-1][1]
        for min_r, f in wt.speed_zones:
            if r >= min_r:
                frac = f
                break
        if self.junction_distance(w) <= wt.junction_zone_mm:
            frac = min(frac, wt.junction_fraction)
        return frac

    def edge_terms(self, u: int, v: int, w: int) -> dict | None:
        """Unweighted quantities of moving v -> w after u, or None where a hard constraint removes the edge."""
        a, dev, wt = self.anat, self.device, self.w
        r = a.radius[w]
        if r < dev.min_lumen_radius_mm:
            return None
        if wt.forbidden_branches and a.branch_of(w) in wt.forbidden_branches:
            return None
        length = float(np.linalg.norm(a.nodes[w] - a.nodes[v]))
        r_ok = dev.comfort_lumen_radius_mm
        narrow = max(0.0, (r_ok - r) / (r_ok - dev.min_lumen_radius_mm)) if r < r_ok else 0.0
        kappa = self.geo.turn_curvature(u, v, w) if u >= 0 else 0.0
        if wt.hard_curvature and kappa > dev.max_curvature_per_mm:
            return None
        bend = (kappa / dev.max_curvature_per_mm) ** 2
        frac = self.speed_fraction(w)
        time_s = length / max(wt.v_max_mm_s * frac, 1e-9)
        force = self.force_model(u, v, w) if self.force_model is not None else bend * (1.0 + narrow)
        junction = u >= 0 and a.degree[v] >= 3
        changes = bool(junction and a.branch_id[w] != a.branch_id[u])
        turn = self.geo.turn_angle_deg(u, v, w) if junction else 0.0
        return {"length": length, "narrow": narrow, "kappa": kappa, "bend": bend, "time_s": time_s, "force": force,
                "junction": junction, "changes_vessel": changes, "turn_deg": turn}

    def transition_cost(self, u: int, v: int, w: int) -> float:
        t = self.edge_terms(u, v, w)
        if t is None:
            return np.inf
        wt, L = self.w, t["length"]
        cost = (wt.length * L + wt.time * t["time_s"] + L * (wt.narrow * t["narrow"] + wt.curvature * t["bend"]
                                                             + wt.curvature_integral * t["kappa"] + wt.force * t["force"]))
        if t["changes_vessel"]:
            cost += wt.branch
        if t["junction"]:
            cost += wt.takeoff * max(0.0, t["turn_deg"] - wt.takeoff_comfort_deg)
        return cost

    def path_cost(self, path: list[int]) -> float:
        c = 0.0
        for i in range(1, len(path)):
            u = path[i - 2] if i >= 2 else -1
            step = self.transition_cost(u, path[i - 1], path[i])
            c += step if np.isfinite(step) else float(np.linalg.norm(self.anat.nodes[path[i]] - self.anat.nodes[path[i - 1]])) * 100
        return c

    def path_costs(self, path: list[int]) -> dict:
        """Unweighted totals along a node path: the route's evaluation before execution."""
        tot = {"length_mm": 0.0, "est_time_s": 0.0, "curvature_integral_rad": 0.0, "narrow_mm": 0.0, "force_proxy": 0.0,
               "bend_effort": 0.0, "infeasible_edges": 0}
        for i in range(1, len(path)):
            u = path[i - 2] if i >= 2 else -1
            t = self.edge_terms(u, path[i - 1], path[i])
            if t is None:
                tot["infeasible_edges"] += 1
                tot["length_mm"] += float(np.linalg.norm(self.anat.nodes[path[i]] - self.anat.nodes[path[i - 1]]))
                continue
            L = t["length"]
            tot["length_mm"] += L
            tot["est_time_s"] += t["time_s"]
            tot["curvature_integral_rad"] += t["kappa"] * L
            tot["narrow_mm"] += L if t["narrow"] > 0 else 0.0
            tot["force_proxy"] += t["force"] * L
            tot["bend_effort"] += t["bend"] * L
        return tot

    def _cost_to_go(self, nodes: list[int]) -> list[float]:
        """Cost of the remainder of ``nodes`` from each index to the end (the splice at index i is
        costed with the previous route's own incoming direction, an approximation at the junction)."""
        n = len(nodes)
        ctg = [0.0] * n
        for i in range(n - 2, -1, -1):
            u = nodes[i - 1] if i > 0 else -1
            step = self.transition_cost(u, nodes[i], nodes[i + 1])
            ctg[i] = ctg[i + 1] + (step if np.isfinite(step) else 1e6)
        return ctg

    # ------------------------------------------------------------------ helpers
    def _reconstruct(self, parent, state) -> list[int]:
        out = []
        while state is not None:
            out.append(state[1])
            state = parent[state]
        return out[::-1]

    def _make_plan(self, path: list[int], cost: float, method: str, expanded: int) -> Plan:
        a = self.anat
        pts = a.nodes[path]
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        kappa = self.geo.path_curvature(path)
        junctions = []
        for i in range(1, len(path) - 1):
            v = path[i]
            if a.degree[v] < 3:
                continue
            u, w = path[i - 1], path[i + 1]
            junctions.append({
                "index": i,
                "node": int(v),
                "s_mm": float(s[i]),
                "from_branch": a.branch_of(u),
                "to_branch": a.branch_of(w),
                "changes_vessel": bool(a.branch_id[w] != a.branch_id[u]),
                "turn_deg": self.geo.turn_angle_deg(u, v, w),
                "options": [a.branch_of(c) for c in a.neighbors[v] if c != u],
            })
        costs = self.path_costs(path)
        costs.update({"bifurcations": len(junctions), "vessel_changes": sum(1 for j in junctions if j["changes_vessel"]),
                      "rejoin_mm": 0.0})
        return Plan(a, self.device, list(map(int, path)), method, float(cost), True, "", s, pts.copy(),
                    a.radius[path].copy(), kappa, junctions, expanded, costs)

    def _fail(self, message: str, method: str = "astar") -> Plan:
        return Plan(self.anat, self.device, [], method, float("inf"), False, message)

    def _why_unreachable(self, start: int, target: int) -> str:
        a = self.anat
        try:
            path = a.route(start, target)
        except ValueError:
            return "target is not connected to the start in the vessel graph"
        r = a.radius[path]
        i = int(np.argmin(r))
        forbidden = self.w.forbidden_branches
        extra = f" or uses a forbidden vessel ({', '.join(forbidden)})" if forbidden else ""
        return (f"no route satisfies the device limits: every route is narrower than "
                f"{self.device.min_lumen_radius_mm:.2f} mm somewhere{extra} (shortest route: {r[i]:.2f} mm in "
                f"{a.branch_of(path[i])})")


def nearest_feasible_target(planner: PathPlanner, target: int, max_back_mm: float = 30.0) -> tuple[int, float]:
    """The target itself if the device fits there, else the nearest node proximal to it (towards the
    default start) with enough lumen, within ``max_back_mm``. Returns (node, mm moved proximally).
    A lesion narrows the lumen at the target, and a thrombectomy approaches it from the proximal side."""
    a, r_min = planner.anat, planner.device.min_lumen_radius_mm
    if a.radius[target] >= r_min:
        return int(target), 0.0
    try:
        path = a.route(a.default_start(), int(target))
    except ValueError:
        return int(target), 0.0
    pts = a.nodes[path]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    for i in range(len(path) - 1, -1, -1):
        back = float(s[-1] - s[i])
        if back > max_back_mm:
            break
        if a.radius[path[i]] >= r_min:
            return int(path[i]), back
    return int(target), 0.0


def evaluate_plan(plan: Plan, travel_mm: float | None = None) -> dict:
    """The four evaluations of a route before execution: inserted length (robot size), estimated time,
    damage (curvature integral, narrow length, force proxy, tight bends) and wrong-artery exposure
    (vessel changes at bifurcations, where a wrong entry can happen; the route itself never enters a
    vessel it does not need). ``travel_mm``: the robot's guidewire travel, to say whether the target
    is within reach by insertion length."""
    if not plan.feasible or not plan.nodes:
        return {"feasible": False, "message": plan.message}
    c, s = plan.costs, plan.summary()
    out = {"feasible": True, "method": plan.method,
           "robot_size": {"inserted_length_mm": round(plan.length_mm, 1), "rejoin_mm": round(c.get("rejoin_mm", 0.0), 1),
                          "within_travel": None if travel_mm is None else bool(plan.length_mm <= travel_mm)},
           "reach_time": {"estimated_s": round(c["est_time_s"], 1)},
           "damage": {"curvature_integral_rad": round(c["curvature_integral_rad"], 2), "narrow_lumen_mm": round(c["narrow_mm"], 1),
                      "force_proxy": round(c["force_proxy"], 2), "tight_bends": s["tight_bends"],
                      "min_lumen_radius_mm": s["min_lumen_radius_mm"]},
           "wrong_arteries": {"bifurcations": c["bifurcations"], "vessel_changes": c["vessel_changes"],
                              "max_takeoff_deg": s["max_takeoff_deg"]}}
    return out


def compare_plans(plans: list[Plan]) -> str:
    """Markdown table comparing plans (e.g. constrained A* vs plain Dijkstra, or devices)."""
    rows = [p.summary() for p in plans]
    keys = ["method", "device", "feasible", "length_mm", "est_time_s", "curvature_integral_rad", "narrow_mm", "force_proxy",
            "branches", "bifurcations", "vessel_changes", "min_lumen_radius_mm", "required_lumen_radius_mm",
            "min_bend_radius_mm", "tight_bends", "max_takeoff_deg", "cost", "message"]
    head = "| " + " | ".join(keys) + " |"
    sep = "|" + "---|" * len(keys)
    lines = [head, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    return "\n".join(lines)

