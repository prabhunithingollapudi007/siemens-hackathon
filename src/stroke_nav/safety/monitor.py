"""Runtime safety monitor (the "checker" in a doer/checker architecture).

The navigation policy (RL, VLA, planner follower, teleoperation assistant) is the *doer*: it may be
learned and is not verifiable. This monitor is the *checker*: a small, deterministic rule set that
sits between the policy and the robot and may only make a command *less* aggressive:

* clamp a speed,
* replace an advance by a hold or a short back-off,
* hold all axes.

It never adds motion the policy did not ask for, apart from a bounded back-off after a blocked push.
Every decision depends only on quantities a real system measures (joint encoders, tracked tip
position, tracking-dropout flag, contact/force signal) plus the approved plan. It never reads
simulator ground truth, so the same logic can later run as a ROS 2 node or in embedded C.

Rules (IDs are used in logs, metrics and docs/SAFETY.md):

R1 speed envelope     advance speed limited by the lumen size at the tip (navigation modes:
                      quick in the aorta, careful distally) and by a slow zone around every planned
                      bifurcation ("selection mode"): a wrong turn is only seen after latency, so the
                      depth reached in a wrong vessel is bounded by the speed at the junction
R2 contact back-off   after a blocked push (or tip force above the limit) the device may not push
                      again for a few steps; it backs off instead (prevents slack build-up/prolapse)
R3 corridor           tracked tip further than ``corridor_mm`` from the approved path: no further
                      advance of the leading device (early wrong-branch containment); retract allowed
R4 tracking loss      ``hold_after_dropouts`` consecutive dropouts, or an implausible tip jump: hold
R5 insertion fence    no advance beyond the target along the approved path (+ margin)
R6 wire leads         the microcatheter may not overtake the guidewire tip by more than
                      ``catheter_overtake_mm`` (clinical rule: a catheter is advanced over a wire)

Human override: if the operator command is non-zero it replaces the policy command and only R1
(speed envelope) still applies. The operator always outranks both the policy and the monitor's
tracking-based rules, because the operator still sees live imaging when automatic tracking fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

WIRE, CATH = 0, 1

RULES = {
    "R1": "speed envelope",
    "R2": "contact back-off",
    "R3": "corridor",
    "R4": "tracking loss",
    "R5": "insertion fence",
    "R6": "wire leads",
}


@dataclass
class SafetyConfig:
    # R1: (min lumen radius mm, max fraction of v_max). First matching row wins.
    speed_zones: tuple[tuple[float, float], ...] = ((5.0, 1.0), (2.0, 0.5), (0.0, 0.25))
    junction_fraction: float = 0.3  # max fraction of v_max within the junction slow zone
    junction_before_mm: float = 8.0
    junction_after_mm: float = 12.0
    # R2
    force_limit_N: float = 0.8  # used when a force signal is available
    backoff_steps: int = 2
    backoff_speed: float = 0.15  # fraction of v_max
    # R3
    corridor_mm: float = 4.0
    # R4
    hold_after_dropouts: int = 2
    jump_margin_mm: float = 3.0  # plausibility: tip may move v_max*dt + tip swing + margin per step
    # R5
    overshoot_mm: float = 10.0
    # R6
    catheter_overtake_mm: float = 0.0
    enabled: tuple[str, ...] = tuple(RULES)


@dataclass
class Measurement:
    """What the robot knows at this control step (no simulator ground truth)."""

    L_mm: np.ndarray  # inserted length per device (encoders)
    tip_mm: np.ndarray | None  # tracked guidewire tip position (None during a dropout)
    blocked: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=bool))  # last push stalled
    tip_force_N: float | None = None  # if the plant reports a force


@dataclass
class Decision:
    action: np.ndarray
    fired: list[str]
    override: bool = False
    deviation_mm: float = 0.0
    zone_fraction: float = 1.0


class SafetyMonitor:
    """Stateful rule checker. Call ``reset`` per episode with the approved plan, then ``filter`` per step."""

    def __init__(self, config: SafetyConfig | None = None, v_max_mm_s=(40.0, 40.0), dt: float = 1 / 7.5,
                 tip_swing_mm: float = 4.0):
        self.cfg = config or SafetyConfig()
        self.v_max = np.asarray(v_max_mm_s, dtype=float)
        self.dt = float(dt)
        self.tip_swing_mm = tip_swing_mm
        self._tree = None

    # ------------------------------------------------------------------ episode setup
    def reset(self, path_points: np.ndarray, path_s: np.ndarray, path_radius: np.ndarray,
              target_s_mm: float, junction_s_mm=()) -> None:
        """Approved plan: densified path points (N, 3), arc length (N,), lumen radius (N,), target arc
        length and the arc lengths of the bifurcations on the route."""
        self.junction_s = np.asarray(junction_s_mm, dtype=float)
        from scipy.spatial import cKDTree

        self.path_points = np.asarray(path_points, dtype=float)
        self.path_s = np.asarray(path_s, dtype=float)
        self.path_radius = np.asarray(path_radius, dtype=float)
        self.target_s = float(target_s_mm)
        self._tree = cKDTree(self.path_points)
        self.dropouts = 0
        self.last_tip: np.ndarray | None = None
        self.last_tip_L = 0.0
        self.backoff = np.zeros(2, dtype=int)
        self.counts = {r: 0 for r in RULES}
        self.overrides = 0

    # ------------------------------------------------------------------ main entry
    def filter(self, action, m: Measurement, operator_action=None) -> Decision:
        cfg, on = self.cfg, set(self.cfg.enabled)
        a = np.clip(np.asarray(action, dtype=float).reshape(4), -1.0, 1.0).copy()
        fired: list[str] = []
        override = operator_action is not None and np.any(np.abs(operator_action) > 1e-6)
        if override:
            a = np.clip(np.asarray(operator_action, dtype=float).reshape(4), -1.0, 1.0).copy()
            self.overrides += 1

        # ---- R4 tracking loss and plausibility (evaluated first: without a valid tip nothing else is trusted)
        tip_valid = m.tip_mm is not None
        if tip_valid and self.last_tip is not None:
            # Cross-check vision against the encoders: since the last accepted frame the tip can only
            # have moved as far as the wire was inserted or withdrawn (plus tip swing from rotation,
            # one step of latency and a margin). Bounding by encoder travel, not by one step, keeps
            # the check from rejecting every frame after a dropout burst (which would hold forever).
            travel = abs(float(m.L_mm[WIRE]) - self.last_tip_L)
            max_step = travel + self.v_max.max() * self.dt + self.tip_swing_mm + cfg.jump_margin_mm
            if np.linalg.norm(np.asarray(m.tip_mm) - self.last_tip) > max_step:
                tip_valid = False  # implausible jump: treat as a bad frame
        self.dropouts = 0 if tip_valid else self.dropouts + 1
        if tip_valid:
            self.last_tip = np.asarray(m.tip_mm, dtype=float)
            self.last_tip_L = float(m.L_mm[WIRE])
        if "R4" in on and not override and self.dropouts >= cfg.hold_after_dropouts:
            return self._done(np.zeros(4), ["R4"], override, 0.0, 0.0)

        # ---- geometry of the tracked tip relative to the approved path
        dev, zone = 0.0, 1.0
        if self.last_tip is not None and self._tree is not None:
            dev, k = self._tree.query(self.last_tip)
            r = float(self.path_radius[k])
            zone = next(f for rmin, f in cfg.speed_zones if r >= rmin)

        # ---- R1 speed envelope (also applies to the operator)
        if "R1" in on:
            for dvc, i in ((WIRE, 0), (CATH, 2)):
                lim = zone
                if len(self.junction_s):
                    rel = float(m.L_mm[dvc]) - self.junction_s
                    if np.any((rel > -cfg.junction_before_mm) & (rel < cfg.junction_after_mm)):
                        lim = min(lim, cfg.junction_fraction)
                if a[i] > lim + 1e-9:
                    a[i] = lim
                    fired.append("R1")

        if not override:
            leader = WIRE if m.L_mm[WIRE] >= m.L_mm[CATH] else CATH
            # ---- R2 contact back-off
            if "R2" in on:
                force_hit = m.tip_force_N is not None and m.tip_force_N > cfg.force_limit_N
                for dvc, idx in ((WIRE, 0), (CATH, 2)):
                    if m.blocked[dvc] or (force_hit and dvc == leader):
                        self.backoff[dvc] = cfg.backoff_steps
                    if self.backoff[dvc] > 0:
                        self.backoff[dvc] -= 1
                        if a[idx] > -cfg.backoff_speed:
                            a[idx] = -cfg.backoff_speed
                            fired.append("R2")
            # ---- R3 corridor: the leading device may not go deeper while off the approved path
            if "R3" in on and dev > cfg.corridor_mm:
                idx = 0 if leader == WIRE else 2
                if a[idx] > 0:
                    a[idx] = 0.0
                    fired.append("R3")
                other = 2 if idx == 0 else 0
                if a[other] > 0 and m.L_mm[CATH if other == 2 else WIRE] >= m.L_mm[leader] - 1e-6:
                    a[other] = 0.0
                    fired.append("R3")
            # ---- R5 insertion fence (encoder arc length vs approved target arc length)
            if "R5" in on:
                fence = self.target_s + cfg.overshoot_mm
                for dvc, idx in ((WIRE, 0), (CATH, 2)):
                    room = fence - m.L_mm[dvc]
                    vmax_step = self.v_max[dvc] * self.dt
                    if a[idx] > 0 and a[idx] * vmax_step > room:
                        a[idx] = max(room / vmax_step, 0.0)
                        fired.append("R5")
            # ---- R6 the guidewire leads
            if "R6" in on and a[2] > 0:
                room = m.L_mm[WIRE] + cfg.catheter_overtake_mm - m.L_mm[CATH]
                step_c = self.v_max[CATH] * self.dt
                wire_adv = max(a[0], 0.0) * self.v_max[WIRE] * self.dt
                if a[2] * step_c > room + wire_adv + 1e-9:
                    a[2] = max((room + wire_adv) / step_c, 0.0)
                    fired.append("R6")
        return self._done(a, fired, override, float(dev), float(zone))

    def _done(self, a, fired, override, dev, zone) -> Decision:
        for r in set(fired):
            self.counts[r] += 1
        return Decision(np.clip(a, -1.0, 1.0), sorted(set(fired)), override, dev, zone)
