"""Baseline controllers.

``ScriptedOracle`` uses privileged simulator access (the route and the noise-free branch-choice
rule) to drive the guidewire ahead and the microcatheter behind it. It is an upper bound and a
sanity check: if it cannot reach a target, the environment (not the learner) is broken.
"""

from __future__ import annotations

import numpy as np

from stroke_nav.env.devices import CATH, WIRE
from stroke_nav.env.vessel_env import VesselNavEnv


class RandomAgent:
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def reset(self, env: VesselNavEnv) -> None:
        pass

    def act(self, env: VesselNavEnv, obs=None) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, size=4)


class ScriptedOracle:
    def __init__(self, overshoot_mm: float = 6.0, catheter_gap_mm: float = 10.0, grid: int = 72):
        self.overshoot_mm = overshoot_mm
        self.catheter_gap_mm = catheter_gap_mm
        self.phis = np.linspace(0.0, 2 * np.pi, grid, endpoint=False)

    def reset(self, env: VesselNavEnv) -> None:
        a = env.anat
        self.full_route = list(env.route) + env._continuation(env.target_node, env.route, 30.0)
        self.idx = {n: i for i, n in enumerate(self.full_route)}
        pts = a.nodes[self.full_route]
        self.route_s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        self.s_target = float(self.route_s[len(env.route) - 1])
        self.s_end = float(self.route_s[-1])

    def act(self, env: VesselNavEnv, obs=None) -> np.ndarray:
        sim, p = env.sim, env.cfg.sim
        step_w = p.guidewire.v_max_mm_s * p.dt
        step_c = p.microcatheter.v_max_mm_s * p.dt
        action = np.zeros(4)

        # 1) Off the route? Pull back whichever device is past the wrong turn.
        bad = next((k for k, n in enumerate(sim.track) if n not in self.idx), None)
        if bad is not None:
            s_back = sim.track_s[max(bad - 1, 0)] - 0.5
            if sim.L[WIRE] > s_back:
                action[0] = -np.clip((sim.L[WIRE] - s_back) / step_w, 0.2, 1.0)
            if sim.L[CATH] > s_back:
                action[2] = -np.clip((sim.L[CATH] - s_back) / step_c, 0.2, 1.0)
            return action

        # 2) Guidewire: rotate before each junction so the tip picks the route's branch.
        wire_goal = min(self.s_target + self.overshoot_mm, self.s_end)
        v_w = np.clip((wire_goal - sim.L[WIRE]) / step_w, -1.0, 1.0)
        leader_s = max(sim.L)
        end_i = self.idx.get(sim.track[-1], 0)
        junctions = []
        for j in range(end_i, len(self.full_route) - 1):
            if self.route_s[j] - leader_s > step_w + 1.5:
                break
            n = self.full_route[j]
            if env.anat.degree[n] >= 3 and self.route_s[j] >= sim.track_s[-1] - 1e-6:
                junctions.append(j)
        junction = junctions[0] if junctions else None
        if len(junctions) > 1:
            # Several junctions within one step: stop at the second one so it is prepared next step.
            v_w = min(v_w, max((self.route_s[junctions[1]] - sim.L[WIRE]) / step_w, 0.0))
        w_w = 0.0
        if junction is not None:
            extra = self.full_route[end_i + 1 : junction + 1]
            desired = self.full_route[junction + 1]
            ok = np.array([sim.preview_choice(extra, ph, WIRE) == desired for ph in self.phis])
            if ok.any():
                target_phi = _nearest_arc_center(ok, self.phis, sim.phi[WIRE])
                err = _wrap(target_phi - sim.phi[WIRE])
                w_w = np.clip(err / np.deg2rad(p.guidewire.w_max_deg_s * p.dt), -1.0, 1.0)
                if abs(err) > np.deg2rad(12):
                    v_w = min(v_w, 0.0)  # hold position while turning
        action[0], action[1] = v_w, w_w

        # 3) Microcatheter: follow the wire at a fixed gap; finish at the target.
        wire_at_limit = p.wire_travel_mm is not None and sim.L[WIRE] >= p.wire_travel_mm - 0.5
        wire_done = (
            wire_at_limit
            or sim.L[WIRE] >= wire_goal - 0.5
            or (sim.L[WIRE] >= sim.track_s[-1] - 0.5 and v_w <= 0)
        )
        cath_goal = self.s_target if wire_done else min(self.s_target, sim.L[WIRE] - self.catheter_gap_mm)
        if not wire_at_limit:
            cath_goal = min(cath_goal, sim.L[WIRE])
        action[2] = np.clip((cath_goal - sim.L[CATH]) / step_c, -1.0, 1.0)
        if not wire_done and sim.L[CATH] <= sim.L[WIRE]:
            action[2] = max(action[2], 0.0)  # while following, hold instead of backing up
        return action


def _wrap(x: float) -> float:
    return float((x + np.pi) % (2 * np.pi) - np.pi)


def _nearest_arc_center(ok: np.ndarray, phis: np.ndarray, phi_now: float) -> float:
    """Centre of the contiguous feasible arc (circular) closest to the current roll."""
    n = len(ok)
    if ok.all():
        return phi_now
    start = int(np.argmin(ok))  # begin scanning at an infeasible index so arcs don't wrap
    arcs, cur = [], []
    for k in range(1, n + 1):
        i = (start + k) % n
        if ok[i]:
            cur.append(i)
        elif cur:
            arcs.append(cur)
            cur = []
    if cur:
        arcs.append(cur)
    step = phis[1] - phis[0]
    best, best_d = phi_now, np.inf
    for arc in arcs:
        center = phis[arc[0]] + step * (len(arc) - 1) / 2.0
        d = abs(_wrap(center - phi_now)) - 0.01 * len(arc)  # prefer wide arcs when tied
        if d < best_d:
            best, best_d = center, d
    return float(best)


def run_episode(env: VesselNavEnv, agent, seed: int | None = None, options: dict | None = None) -> dict:
    obs, _ = env.reset(seed=seed, options=options)
    agent.reset(env)
    done = False
    info: dict = {}
    while not done:
        obs, _, term, trunc, info = env.step(agent.act(env, obs))
        done = term or trunc
    return info["episode_metrics"]
