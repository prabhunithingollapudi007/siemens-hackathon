"""Kinematic guidewire + microcatheter model on a centerline graph.

Both devices share one *track*: the sequence of centerline nodes the leading device has entered,
starting at the insertion point. Each device has an inserted length ``L`` (mm) along the track and
a roll angle ``phi`` (rad). The device with the larger ``L`` leads; when it advances past the end of
the track, the track is extended by one node, and at a bifurcation the next vessel is chosen by
the leader's pre-curved tip:

* the tip bends by ``tip_angle`` toward direction ``b = cos(phi) n1 + sin(phi) n2``, where (n1, n2)
  is a frame perpendicular to the vessel tangent, parallel-transported along the track;
* a side branch is entered when ``b`` points within ``branch_window_deg`` of that branch's off-axis
  direction (plus noise); otherwise the device continues along the straightest vessel.

So reaching a target requires rotating each device at the right moments. When the follower
(catheter) sits within ``support_distance_mm`` behind the leading wire tip, steering is less noisy
(the catheter supports the wire). Retracting pops track nodes the leader no longer occupies, so
re-advancing makes a new branch choice. Pushing against a dead end is recorded as a wall push.

This is deliberately simple (no elasticity, no wall forces); a physics-based wire can later
replace it behind the same interface.

Device parameters: ``device_sim_params()`` (the default everywhere) builds both devices from
device-parameters.json via ``organiser.spec.DEFAULT_SPEC``: tip angle and tip length per device,
planner device specs whose diameters come from the same file, the organiser's control rate and
speed limits, and the robot's travel limits. ``SimParams()`` itself keeps generic research
defaults for tests and experiments that set every value explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from stroke_nav.data.anatomy import Anatomy

WIRE, CATH = 0, 1
DEVICE_NAMES = ("guidewire", "microcatheter")


@dataclass
class DeviceParams:
    """Per-device limits. Actions in [-1, 1] are scaled by v_max and w_max."""

    name: str
    v_max_mm_s: float = 40.0  # translation speed limit
    w_max_deg_s: float = 180.0  # rotation speed limit
    tip_angle_deg: float = 40.0  # pre-curved tip bend
    tip_length_mm: float = 2.0  # length of the bent tip segment (for tracking points)
    spec: str = "guidewire"  # key into stroke_nav.planning.constraints.DEVICES (diameter, clearance)


@dataclass
class SimParams:
    dt: float = 1.0 / 7.5  # control period (s); paper: 7.5 Hz
    guidewire: DeviceParams = field(default_factory=lambda: DeviceParams("guidewire", 40.0, 180.0, 40.0))
    microcatheter: DeviceParams = field(default_factory=lambda: DeviceParams("microcatheter", 40.0, 180.0, 30.0, spec="microcatheter"))
    branch_window_deg: float = 50.0  # tip azimuth must be within this of a side branch to enter it
    straight_cone_deg: float = 35.0  # a candidate within this of the tangent counts as "straight on"
    choice_noise: float = 0.15  # noise on azimuth score (cosine units)
    supported_noise: float = 0.05  # noise when catheter supports the wire
    support_distance_mm: float = 30.0
    tangent_window_mm: float = 5.0
    lookahead_nodes: int = 5
    max_insertion_mm: float = 1500.0
    wire_travel_mm: float | None = None  # robot limit on guidewire insertion (None = unlimited)
    cath_lead_max_mm: float | None = None  # robot limit: catheter insertion <= wire insertion + this
    initial_insertion_mm: tuple[float, float] = (20.0, 15.0)  # (wire, catheter)
    tip_offset_frac: float = 0.7  # bent tip end stays within this fraction of the vessel radius from the centerline
    enforce_lumen: bool = True  # a device cannot enter lumen narrower than its clearance needs

    def device(self, i: int) -> DeviceParams:
        return self.guidewire if i == WIRE else self.microcatheter


def device_sim_params(spec=None, tracking_tip_mm: float | None = None, robot_limits: bool = True) -> SimParams:
    """Simulator settings for the robot's two devices, from device-parameters.json.

    * tip angle and tip length of each device: ``tipAngle``/``tipLength`` and ``cathTipAngle``/
      ``cathTipLength``; ``tracking_tip_mm`` overrides both tip lengths
    * planner device specs ``radifocus_035`` / ``judkins_5f``, whose diameters come from the same file
    * control period and speed limits: the organiser's action spec (5 Hz, 10 mm/s, 1 rad/s)
    * robot travel: ``spec.robot`` (guidewire 264.5 mm, catheter at most 229.2 mm ahead of the wire)
    """
    from stroke_nav.organiser.spec import DEFAULT_SPEC
    from stroke_nav.planning.constraints import CATHETER, GUIDEWIRE

    spec = spec or DEFAULT_SPEC
    v = spec.action.max_translation_m_s * 1000.0
    w = float(np.degrees(spec.action.max_rotation_rad_s))
    gw, ca = spec.guidewire_geometry(), spec.catheter_geometry()
    gw_tip = gw["tip_length_mm"] if tracking_tip_mm is None else tracking_tip_mm
    ca_tip = ca["tip_length_mm"] if tracking_tip_mm is None else tracking_tip_mm
    return SimParams(
        dt=spec.dt,
        guidewire=DeviceParams("guidewire", v, w, gw["tip_angle_deg"], gw_tip, spec=GUIDEWIRE),
        microcatheter=DeviceParams("catheter", v, w, ca["tip_angle_deg"], ca_tip, spec=CATHETER),
        max_insertion_mm=spec.insertion_limits_mm[1],
        wire_travel_mm=spec.robot.guidewire_travel_mm if robot_limits else None,
        cath_lead_max_mm=spec.robot.catheter_lead_max_mm if robot_limits else None,
        initial_insertion_mm=(3.0, 0.0),  # their reset leaves both tips at the entry
    )


class TrackSim:
    """Single-environment simulator (cheap enough for SB3 DummyVecEnv)."""

    def __init__(self, anatomy: Anatomy, params: SimParams | None = None):
        self.anat = anatomy
        self.p = params or device_sim_params()
        self.rng = np.random.default_rng()
        from stroke_nav.planning.constraints import get_device

        self.required_radius = np.array([
            get_device(self.p.guidewire.spec).min_lumen_radius_mm,
            get_device(self.p.microcatheter.spec).min_lumen_radius_mm,
        ])

    # ------------------------------------------------------------------ state
    def reset(self, start_node: int, rng: np.random.Generator | None = None) -> None:
        if rng is not None:
            self.rng = rng
        a = self.anat
        if a.degree[start_node] != 1:
            # Not a leaf: insert toward the neighbour farthest from the root-ish centroid direction.
            pass
        self.track: list[int] = [int(start_node)]
        self.track_s: list[float] = [0.0]
        first = a.neighbors[start_node][0] if a.neighbors[start_node] else start_node
        t0 = a.nodes[first] - a.nodes[start_node]
        t0 = t0 / max(np.linalg.norm(t0), 1e-9)
        self.insert_dir = t0
        self.n_ref: list[np.ndarray] = [_any_perpendicular(t0)]
        self.L = np.zeros(2)
        self.phi = np.zeros(2)
        self.wall_pushes = np.zeros(2, dtype=int)
        self.decisions: list[tuple[int, int, int]] = []  # (junction node, chosen node, leader)
        self.last_entered: list[int] = []
        L0 = self.p.initial_insertion_mm
        self._extend_to(max(L0), leader=WIRE)
        self.L[:] = [min(L0[0], self.track_s[-1]), min(L0[1], self.track_s[-1])]
        self._pop_unused()

    @property
    def leader(self) -> int:
        return WIRE if self.L[WIRE] >= self.L[CATH] else CATH

    # ------------------------------------------------------------------ dynamics
    def step(self, action: np.ndarray) -> dict:
        """Apply action [v_gw, w_gw, v_cath, w_cath] in [-1, 1]. Returns per-step event dict."""
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        dt = self.p.dt
        v = np.array([a[0] * self.p.guidewire.v_max_mm_s, a[2] * self.p.microcatheter.v_max_mm_s]) * dt
        w = np.deg2rad([a[1] * self.p.guidewire.w_max_deg_s, a[3] * self.p.microcatheter.w_max_deg_s]) * dt
        self.phi = np.mod(self.phi + w, 2 * np.pi)

        old_L = self.L.copy()
        new_L = np.clip(self.L + v, 0.0, self.p.max_insertion_mm)
        # robot travel: the modules stop at their limits (the device is not blocked by the vessel)
        if self.p.wire_travel_mm is not None:
            new_L[WIRE] = min(new_L[WIRE], max(self.p.wire_travel_mm, self.L[WIRE]))
        if self.p.cath_lead_max_mm is not None:
            new_L[CATH] = min(new_L[CATH], max(new_L[WIRE] + self.p.cath_lead_max_mm, self.L[CATH]))
        self.last_entered = []
        want = float(new_L.max())
        leader = WIRE if new_L[WIRE] >= new_L[CATH] else CATH
        if want > self.track_s[-1]:
            self._extend_to(want, leader)
        end = self.track_s[-1]
        blocked = np.zeros(2, dtype=bool)
        for i in (WIRE, CATH):
            limit = min(end, self.passable_limit(i))
            if new_L[i] > limit + 1e-6:
                new_L[i] = max(limit, min(old_L[i], new_L[i]))
                if v[i] > 0:
                    blocked[i] = True
        self.wall_pushes += blocked
        self.L = new_L
        self._pop_unused()
        return {"moved": np.abs(self.L - old_L), "blocked": blocked, "entered": list(self.last_entered)}

    def _extend_to(self, length: float, leader: int) -> None:
        a = self.anat
        while self.track_s[-1] < length:
            u = self.track[-1]
            prev = self.track[-2] if len(self.track) > 1 else None
            cands = [c for c in a.neighbors[u] if c != prev]
            if not cands:
                return  # dead end
            if len(cands) == 1:
                nxt = cands[0]
            else:
                t, n1 = self._end_frame()
                nxt = self.choose(u, prev, cands, t, n1, self.phi[leader], leader, noise=self._noise(leader))
                self.decisions.append((u, nxt, leader))
            if self.p.enforce_lumen and a.radius[nxt] < self.required_radius[leader]:
                return  # lumen too narrow for the leading device: it stops here
            self._append(nxt)

    def _append(self, nxt: int) -> None:
        a = self.anat
        u = self.track[-1]
        seg = a.nodes[nxt] - a.nodes[u]
        ln = float(np.linalg.norm(seg))
        t_new = seg / max(ln, 1e-9)
        n = self.n_ref[-1] - np.dot(self.n_ref[-1], t_new) * t_new
        nn = np.linalg.norm(n)
        n = n / nn if nn > 1e-6 else _any_perpendicular(t_new)
        self.track.append(int(nxt))
        self.track_s.append(self.track_s[-1] + max(ln, 1e-3))
        self.n_ref.append(n)
        self.last_entered.append(int(nxt))

    def passable_limit(self, device: int) -> float:
        """Furthest arc length along the track this device can reach (lumen clearance)."""
        if not self.p.enforce_lumen or len(self.track) < 2:
            return float("inf")
        r = self.anat.radius[self.track[1:]]
        bad = np.where(r < self.required_radius[device])[0]
        if len(bad) == 0:
            return float("inf")
        return float(self.track_s[int(bad[0])])  # stop at the last passable node

    def _pop_unused(self) -> None:
        m = float(self.L.max())
        while len(self.track) > 2 and self.track_s[-2] >= m:
            self.track.pop()
            self.track_s.pop()
            self.n_ref.pop()

    def _noise(self, leader: int) -> float:
        if leader == WIRE and 0.0 <= self.L[WIRE] - self.L[CATH] <= self.p.support_distance_mm:
            return self.p.supported_noise
        return self.p.choice_noise

    # ------------------------------------------------------------------ branch choice
    def choose(self, u, prev, cands, t, n1, phi, device, noise=0.0) -> int:
        """Pick the next node at junction ``u`` given tip frame (t, n1) and roll ``phi``."""
        a = self.anat
        n2 = np.cross(t, n1)
        b = np.cos(phi) * n1 + np.sin(phi) * n2
        dirs = np.array([a.forward_direction(u, c, self.p.lookahead_nodes) for c in cands])
        axial = dirs @ t
        perp = dirs - axial[:, None] * t
        pn = np.linalg.norm(perp, axis=1)
        az = (perp / np.maximum(pn, 1e-9)[:, None]) @ b
        score = az + (noise * self.rng.standard_normal(len(cands)) if noise > 0 else 0.0)
        straight = int(np.argmax(axial))
        if axial[straight] >= np.cos(np.deg2rad(self.p.straight_cone_deg)):
            side = [i for i in range(len(cands)) if i != straight]
            best = max(side, key=lambda i: score[i]) if side else straight
            if side and score[best] >= np.cos(np.deg2rad(self.p.branch_window_deg)):
                return int(cands[best])
            return int(cands[straight])
        return int(cands[int(np.argmax(score))])

    def preview_choice(self, extra_nodes: list[int], phi: float, device: int) -> int:
        """Noise-free choice the leader would make after following ``extra_nodes`` (ending at a junction).

        ``extra_nodes`` continue the current track (excluding its last node) and end at the junction.
        Used by the scripted oracle; does not modify state.
        """
        a = self.anat
        track = list(self.track) + list(extra_nodes)
        n = self.n_ref[-1]
        for k in range(len(self.track), len(track)):
            seg = a.nodes[track[k]] - a.nodes[track[k - 1]]
            tt = seg / max(np.linalg.norm(seg), 1e-9)
            n = n - np.dot(n, tt) * tt
            n = n / max(np.linalg.norm(n), 1e-9)
        u = track[-1]
        prev = track[-2] if len(track) > 1 else None
        cands = [c for c in a.neighbors[u] if c != prev]
        t = self._tangent_of(track)
        n1 = n - np.dot(n, t) * t
        n1 = n1 / max(np.linalg.norm(n1), 1e-9)
        if len(cands) == 1:
            return cands[0]
        return self.choose(u, prev, cands, t, n1, phi, device, noise=0.0)

    def _end_frame(self) -> tuple[np.ndarray, np.ndarray]:
        t = self._tangent_of(self.track)
        n1 = self.n_ref[-1] - np.dot(self.n_ref[-1], t) * t
        nn = np.linalg.norm(n1)
        n1 = n1 / nn if nn > 1e-6 else _any_perpendicular(t)
        return t, n1

    def _tangent_of(self, track: list[int]) -> np.ndarray:
        a = self.anat
        end = a.nodes[track[-1]]
        acc, k = 0.0, len(track) - 1
        while k > 0 and acc < self.p.tangent_window_mm:
            acc += float(np.linalg.norm(a.nodes[track[k]] - a.nodes[track[k - 1]]))
            k -= 1
        d = end - a.nodes[track[k]] if k < len(track) - 1 else self.insert_dir
        return d / max(np.linalg.norm(d), 1e-9)

    # ------------------------------------------------------------------ geometry queries
    def position_at(self, s: float) -> np.ndarray:
        """Point on the track at arc length s (mm); s < 0 extends back along the insertion direction."""
        a = self.anat
        if s <= 0:
            return a.nodes[self.track[0]] + s * self.insert_dir
        ts = self.track_s
        i = int(np.searchsorted(ts, s) - 1)
        i = min(max(i, 0), len(ts) - 2) if len(ts) > 1 else 0
        if len(ts) == 1:
            return a.nodes[self.track[0]].copy()
        f = (s - ts[i]) / max(ts[i + 1] - ts[i], 1e-9)
        f = min(max(f, 0.0), 1.0)
        return (1 - f) * a.nodes[self.track[i]] + f * a.nodes[self.track[i + 1]]

    def index_at(self, s: float) -> tuple[int, float]:
        """(segment index i, fraction f) such that s lies between track[i] and track[i+1]."""
        ts = self.track_s
        if len(ts) == 1 or s <= 0:
            return 0, 0.0
        i = int(np.searchsorted(ts, s) - 1)
        i = min(max(i, 0), len(ts) - 2)
        f = (s - ts[i]) / max(ts[i + 1] - ts[i], 1e-9)
        return i, float(min(max(f, 0.0), 1.0))

    def node_at(self, s: float) -> int:
        i, f = self.index_at(s)
        return self.track[min(i + (1 if f > 0.5 else 0), len(self.track) - 1)]

    def frame_at(self, s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tangent and transported normal pair at arc length s."""
        p1 = self.position_at(s)
        p0 = self.position_at(s - self.p.tangent_window_mm)
        t = p1 - p0
        t = t / max(np.linalg.norm(t), 1e-9)
        i, _ = self.index_at(s)
        n = self.n_ref[min(i, len(self.n_ref) - 1)]
        n1 = n - np.dot(n, t) * t
        nn = np.linalg.norm(n1)
        n1 = n1 / nn if nn > 1e-6 else _any_perpendicular(t)
        return t, n1, np.cross(t, n1)

    def tip_direction(self, device: int, Ls: np.ndarray | None = None, phis: np.ndarray | None = None) -> np.ndarray:
        """Direction the device's bent tip points to (straight for a wire inside the catheter).

        ``Ls``/``phis`` (both devices) override the current insertions and rolls, e.g. to draw an
        in-between frame; they must lie on the current track.
        """
        Ls = self.L if Ls is None else Ls
        phis = self.phi if phis is None else phis
        t, n1, n2 = self.frame_at(Ls[device])
        if device == WIRE and Ls[WIRE] < Ls[CATH]:
            return t
        th = np.deg2rad(self.p.device(device).tip_angle_deg)
        b = np.cos(phis[device]) * n1 + np.sin(phis[device]) * n2
        d = np.cos(th) * t + np.sin(th) * b
        return d / np.linalg.norm(d)

    def _tip_offset(self, device: int, Ls: np.ndarray, phis: np.ndarray) -> np.ndarray:
        """Sideways displacement of the bent tip's end from the centerline at the device's arc
        position: towards the bend, at most ``tip_offset_frac`` of the local vessel radius."""
        tl = min(self.p.device(device).tip_length_mm, max(float(Ls[device]), 0.0))
        d = self.tip_direction(device, Ls, phis)
        t, _, _ = self.frame_at(Ls[device])
        lat = d - np.dot(d, t) * t
        n = np.linalg.norm(lat)
        if n < 1e-9:
            return np.zeros(3)
        r = float(self.anat.radius[self.node_at(Ls[device])])
        return min(tl * n, self.p.tip_offset_frac * r) * lat / n

    def tracking_points(self, device: int, spacing: float = 2.0, Ls: np.ndarray | None = None,
                        phis: np.ndarray | None = None) -> np.ndarray:
        """Three points along the device tip (stEVE-style): bent tip end, bend start, one more back.

        The bent tip lies inside the lumen: its end sits at the device's arc position, displaced
        sideways towards the bend by at most ``tip_offset_frac`` of the local vessel radius (a long
        J-tip is pressed against the wall rather than sticking out of the vessel).
        """
        Ls = self.L if Ls is None else Ls
        phis = self.phi if phis is None else phis
        L = float(Ls[device])
        tip_end = self.position_at(L) + self._tip_offset(device, Ls, phis)
        s_b = L - self.p.device(device).tip_length_mm
        return np.stack([tip_end, self.position_at(s_b), self.position_at(s_b - spacing)])

    def device_polyline(self, device: int, step_mm: float = 2.0, Ls: np.ndarray | None = None,
                        phis: np.ndarray | None = None) -> np.ndarray:
        """Points along the inserted part of a device, for drawing.

        Only the inserted length is drawn. The shaft follows the centerline; over the last tip
        length it curves smoothly (zero slope at the bend start) to the bent tip's end, so the
        drawing ends exactly at ``tracking_points(device)[0]`` and stays inside the lumen.
        """
        Ls = self.L if Ls is None else Ls
        phis = self.phi if phis is None else phis
        L = float(Ls[device])
        if L <= 1e-6:
            return self.position_at(0.0)[None, :]
        tl = min(self.p.device(device).tip_length_mm, L)
        s_b = L - tl
        body_s = np.arange(0.0, s_b, step_mm)
        n_tip = max(int(np.ceil(tl / min(step_mm, 2.0))), 2)
        tip_s = np.linspace(s_b, L, n_tip + 1)
        off = self._tip_offset(device, Ls, phis)
        pts = [self.position_at(x) for x in body_s]
        pts += [self.position_at(x) + ((x - s_b) / tl) ** 2 * off for x in tip_s]
        return np.array(pts)


def _any_perpendicular(t: np.ndarray) -> np.ndarray:
    ref = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    n = np.cross(t, ref)
    return n / np.linalg.norm(n)
