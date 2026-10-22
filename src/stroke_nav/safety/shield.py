"""Gymnasium wrapper: fault injection + safety monitor + safety metrics around ``VesselNavEnv``.

    env = SafetyShield(VesselNavEnv(cfg), faults=FaultModel.stress(), monitor=True)

* The policy sees observations corrupted by the fault model (noise, latency, held samples).
* The monitor sees the same delivered measurement plus encoders and the blocked-push signal, and
  filters every action before it reaches the simulator (``monitor=False`` only records metrics).
* ``info["safety"]`` reports the rules fired at each step; at episode end the safety metrics are
  merged into ``info["episode_metrics"]`` (keys prefixed ``safety_``), next to the START metrics.

Agents that read privileged simulator state (``ScriptedOracle``, ``TrajectoryFollower``) still do so;
the shield only filters what they command, which is how it would sit behind any controller.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from stroke_nav.env.devices import CATH, WIRE
from stroke_nav.safety.faults import FaultInjector, FaultModel
from stroke_nav.safety.monitor import RULES, Measurement, SafetyConfig, SafetyMonitor


class SafetyShield(gym.Wrapper):
    def __init__(self, env, faults: FaultModel | None = None, monitor: bool = True,
                 config: SafetyConfig | None = None, operator=None):
        """``operator``: optional callable(env) -> action or None, simulating a human override input."""
        super().__init__(env)
        self.faults = faults or FaultModel.clean()
        self.monitor_on = monitor
        p = env.unwrapped.cfg.sim
        tip_swing = 2 * max(p.guidewire.tip_length_mm, p.microcatheter.tip_length_mm)
        self.monitor = SafetyMonitor(config, (p.guidewire.v_max_mm_s, p.microcatheter.v_max_mm_s), p.dt, tip_swing)
        self.operator = operator
        self._fault_rng = np.random.default_rng()

    def __getattr__(self, name):
        # Agents in this repo read simulator state (env.sim, env.anat, env.route ...): pass those through.
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.env.unwrapped, name)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        e = self.env.unwrapped
        if seed is not None:
            self._fault_rng = np.random.default_rng([int(seed), 7_919])
        self.inj = FaultInjector(self.faults, self._fault_rng)
        pts, s, r, target_s, junction_s = _approved_path(e)
        self.monitor.reset(pts, s, r, target_s, junction_s)
        self.target_s = target_s
        self.blocked = np.zeros(2, dtype=bool)
        self.prev_true_tip = e.sim.position_at(e.sim.L[WIRE])
        self.prev_L = e.sim.L.copy()
        self.prev_delivered = None
        self.stats = {
            "fired": {k: 0 for k in RULES}, "steps_intervened": 0, "holds": 0, "overrides": 0,
            "max_dev_mm": 0.0, "steps_off_corridor": 0, "max_wrong_depth_mm": 0.0,
            "unsupported_catheter_advance_mm": 0.0, "max_overshoot_mm": -np.inf, "speed_jumps": 0, "max_tip_speed_mm_s": 0.0,
        }
        obs, self.meas = self._deliver(obs, tip_speed=0.0)
        return obs, info

    def step(self, action):
        e = self.env.unwrapped
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(4), -1.0, 1.0)
        op = self.operator(e) if self.operator is not None else None
        if self.monitor_on:
            d = self.monitor.filter(action, self.meas, op)
            cmd, fired = d.action, d.fired
            if d.override:
                self.stats["overrides"] += 1
        else:
            cmd, fired = (np.clip(op, -1, 1) if op is not None and np.any(np.abs(op) > 1e-6) else action), []
        if fired:
            self.stats["steps_intervened"] += 1
            for r in fired:
                self.stats["fired"][r] += 1
            if "R4" in fired:
                self.stats["holds"] += 1

        obs, reward, term, trunc, info = self.env.step(cmd.astype(np.float32))
        self.blocked = np.asarray(info["events"]["blocked"], dtype=bool)
        tip = e.sim.position_at(e.sim.L[WIRE])
        speed = float(np.linalg.norm(tip - self.prev_true_tip)) / e.cfg.sim.dt
        self.prev_true_tip = tip
        self._update_stats(e, speed)
        obs, self.meas = self._deliver(obs, tip_speed=speed)
        info["safety"] = {"fired": fired, "command": cmd, "dropped": self.meas.tip_mm is None}
        if term or trunc:
            info["episode_metrics"].update(self.episode_safety_metrics())
        return obs, reward, term, trunc, info

    # ------------------------------------------------------------------ measurement path
    def _deliver(self, obs, tip_speed):
        e = self.env.unwrapped
        true_pts = np.stack([e.sim.tracking_points(WIRE), e.sim.tracking_points(CATH)])
        pts, dropped = self.inj.deliver(true_pts, tip_speed)
        if self.faults.active:
            obs = self._corrupt_obs(e, obs, pts, dropped)
        tip = None if dropped else pts[WIRE, 0].copy()
        return obs, Measurement(L_mm=e.sim.L.copy(), tip_mm=tip, blocked=self.blocked.copy())

    def _corrupt_obs(self, e, obs, pts, dropped):
        n = 2 if e.cfg.include_catheter_obs else 1
        obs = obs.copy()
        cur = slice(0, 9 * n)
        prev = slice(9 * n, 18 * n)
        diff = slice(18 * n + 3, 18 * n + 6)
        if dropped and self.prev_delivered is not None:
            obs[cur] = self.prev_delivered[cur]
            obs[diff] = self.prev_delivered[diff]
        else:
            obs[cur] = ((pts[:n] - e.anat.center) / e.anat.half_extent).ravel()
            tip = pts[e.success_dev, 0]
            obs[diff] = (e.target_pos - tip) / e.anat.half_extent
        if self.prev_delivered is not None:
            obs[prev] = self.prev_delivered[cur]
        obs = np.clip(obs, -10.0, 10.0).astype(np.float32)
        self.prev_delivered = obs.copy()
        return obs

    # ------------------------------------------------------------------ evaluation (ground truth allowed here)
    def _update_stats(self, e, speed):
        s = self.stats
        L = e.sim.L
        dev, _ = self.monitor._tree.query(e.sim.position_at(L[WIRE]))
        s["max_dev_mm"] = max(s["max_dev_mm"], float(dev))
        s["steps_off_corridor"] += int(dev > self.monitor.cfg.corridor_mm)
        bad = next((k for k, nd in enumerate(e.sim.track) if nd not in e.route_set), None)
        if bad is not None:
            s["max_wrong_depth_mm"] = max(s["max_wrong_depth_mm"], float(max(L) - e.sim.track_s[max(bad - 1, 0)]))
        # catheter pushed forward while its tip is ahead of the wire tip (wire retracted inside is fine)
        adv = L[CATH] - self.prev_L[CATH]
        if adv > 0 and L[CATH] > L[WIRE] + 0.5:
            s["unsupported_catheter_advance_mm"] += float(adv)
        self.prev_L = L.copy()
        s["max_overshoot_mm"] = max(s["max_overshoot_mm"], float(max(L) - self.target_s))
        vmax = max(e.cfg.sim.guidewire.v_max_mm_s, e.cfg.sim.microcatheter.v_max_mm_s)
        s["speed_jumps"] += int(speed > 1.25 * vmax)
        s["max_tip_speed_mm_s"] = max(s["max_tip_speed_mm_s"], speed)

    def episode_safety_metrics(self) -> dict:
        s = self.stats
        out = {f"safety_{k}": v for k, v in s.items() if k != "fired"}
        out.update({f"safety_fired_{k}": v for k, v in s["fired"].items()})
        out["safety_dropouts"] = self.inj.dropouts
        out["safety_monitor"] = bool(self.monitor_on)
        return out


def _approved_path(e) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Approved path (route + short continuation past the target), densified to 0.5 mm."""
    a = e.anat
    nodes = list(e.route) + e._continuation(e.target_node, e.route, 20.0)
    pts = a.nodes[nodes]
    rad = a.radius[nodes]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    target_s = float(s[len(e.route) - 1])
    fine = np.arange(0.0, s[-1] + 1e-9, 0.5)
    P = np.column_stack([np.interp(fine, s, pts[:, k]) for k in range(3)])
    R = np.interp(fine, s, rad)
    pos = {n: i for i, n in enumerate(nodes)}
    junction_s = np.array([s[pos[n]] for n in e.route_junctions if n in pos])
    return P, fine, R, target_s, junction_s
