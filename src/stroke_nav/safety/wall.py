"""Wall guard for the organiser's rod physics: does the guidewire cross the vessel wall?

Their physics page (``projectContainment``) only keeps the wire's bodies inside the lumen, 7.5 mm apart
(1.2 m / 160 bodies), and pushes each one back towards the *nearest* wall. Where two lumens run a few mm
apart (the aortic arch and the left common carotid of ``0028_H_ABAO_H``), a pushed body can slip
across the wall, find the other lumen's wall nearest, count as inside and stay there: the wire then
reaches its goal *through* the wall, with no force reading and their success rule satisfied.

``WallGuard`` checks what their page draws: a Catmull-Rom curve through the bodies of the inserted
part, sampled every ``1 / samples_per_mm`` mm, against the scenario's vessel surface (the mesh their
physics collides with), by signed distance. Positions in their world frame, metres in and mm out;
with the authored bake the world frame is the mesh frame times ``meters_per_unit`` (0.01).

In their stack the bodies come from ``/endovascular/guidewire/state/noisy`` (segment poses); in
lockstep from ``LockstepBridge.noisy``. The patient surface is pre-operative data (the segmented CTA).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def catmull_rom(points: np.ndarray, samples_per_mm: float = 2.0) -> np.ndarray:
    """A Catmull-Rom curve through ``points`` (N, 3), as their page draws the rod (end points clamped)."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 2:
        return p.copy()
    q = np.vstack([p[:1], p, p[-1:]])
    out = []
    for i in range(1, len(q) - 2):
        p0, p1, p2, p3 = q[i - 1], q[i], q[i + 1], q[i + 2]
        n = max(2, int(np.ceil(np.linalg.norm(p2 - p1) * samples_per_mm)))
        t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(p[-1:])
    return np.vstack(out)


def read_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (file units) and triangles of a Wavefront OBJ (``v`` and triangular ``f`` lines)."""
    verts, faces = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


@dataclass
class WallCheck:
    crossing: bool
    depth_mm: float  # deepest point of the wire curve beyond the wall (0 when inside)
    outside_mm: float  # length of wire curve beyond the wall (deeper than ``shallow_mm``)
    where_world_m: list | None  # deepest point
    inserted_bodies: int
    kind: str = ""  # "shortcut" (into another lumen through the wall) or "deep" (beyond ``depth_mm``)


class WallGuard:
    """Signed distance of the inserted wire curve to the vessel surface (VTK implicit distance).

    The curve between bodies may bow slightly past a wall of the same vessel (bodies 7.5 mm apart, a
    10k-triangle mesh). A stretch of curve beyond the wall by more than ``shallow_mm`` is a crossing when

    * it is a **shortcut**: the inside points just before and after it lie in lumens whose distance along
      the vessel tree is more than ``shortcut_ratio`` times the wire between them plus ``shortcut_slack_mm``
      (the arch into the left common carotid through the wall), at any depth; needs ``frame`` (world m to
      the model's mm) and ``anat`` (centerlines, ``stroke_nav.data.anatomy.Anatomy``)
    * or it goes deeper than ``depth_mm`` (a tip pressed through a wall, or any stretch without centerlines).
    """

    def __init__(self, verts_mm: np.ndarray, faces: np.ndarray, depth_mm: float = 1.5, samples_per_mm: float = 2.0,
                 inside_point_mm: np.ndarray | None = None, anat=None, frame=None, shallow_mm: float = 0.3,
                 shortcut_ratio: float = 2.0, shortcut_slack_mm: float = 10.0):
        import pyvista as pv
        import vtk

        self.depth_mm, self.samples_per_mm = float(depth_mm), float(samples_per_mm)
        self.anat, self.frame, self.shallow_mm = anat, frame, float(shallow_mm)
        self.shortcut_ratio, self.shortcut_slack_mm = float(shortcut_ratio), float(shortcut_slack_mm)
        if anat is not None:
            from scipy.spatial import cKDTree

            self._tree = cKDTree(anat.nodes)
        surface = pv.PolyData(np.asarray(verts_mm, float), np.hstack([np.full((len(faces), 1), 3), faces]).ravel())
        surface = surface.compute_normals(cell_normals=False, point_normals=True, auto_orient_normals=True, consistent_normals=True)
        self._f = vtk.vtkImplicitPolyDataDistance()
        self._f.SetInput(surface)
        self._sign = 1.0
        probe = np.array([0.0, 0.0, 5.0]) if inside_point_mm is None else np.asarray(inside_point_mm, float)
        if self._f.EvaluateFunction(*probe) > 0:  # orient: negative inside the lumen
            self._sign = -1.0

    @classmethod
    def from_scenario(cls, scenario_dir: str | Path, anat=None, frame=None, **kwargs) -> "WallGuard":
        """Guard for a scenario folder (``vessel.obj`` with the entrance at the origin, insertion along +Z).
        ``anat`` / ``frame``: the model's centerlines and organiser frame, for the shortcut test."""
        import json

        scenario_dir = Path(scenario_dir)
        manifest = json.loads((scenario_dir / "scenario.json").read_text(encoding="utf-8"))
        verts, faces = read_obj(scenario_dir / manifest.get("mesh_file", "vessel.obj"))
        mm_per_unit = float(manifest.get("meters_per_unit", 0.01)) * 1000.0
        return cls(verts * mm_per_unit, faces, anat=anat, frame=frame, **kwargs)

    def _lumen_node(self, point_world_mm: np.ndarray) -> int:
        """Centerline node of the lumen a point is in: the nearest node whose radius contains it."""
        p = self.frame.world_m_to_vmr_mm(np.asarray(point_world_mm, float) / 1000.0)
        d, idx = self._tree.query(p, k=12)
        for dist, i in zip(np.atleast_1d(d), np.atleast_1d(idx)):
            if dist <= self.anat.radius[i] + 0.5:
                return int(i)
        return int(np.atleast_1d(idx)[0])

    def _is_shortcut(self, curve: np.ndarray, d: np.ndarray, lo: int, hi: int) -> bool:
        """Does the outside stretch ``curve[lo:hi]`` join two lumens far apart along the vessel tree?"""
        if self.anat is None or self.frame is None or lo == 0 or hi >= len(curve):
            return False
        a, b = self._lumen_node(curve[lo - 1]), self._lumen_node(curve[hi])
        if a == b:
            return False
        along_tree = float(self.anat.distances_from(a)[0][b])
        along_wire = float(np.linalg.norm(np.diff(curve[lo - 1:hi + 1], axis=0), axis=1).sum())
        return along_tree > self.shortcut_ratio * along_wire + self.shortcut_slack_mm

    def signed_distance_mm(self, points_mm: np.ndarray) -> np.ndarray:
        """Signed distance to the vessel wall, mm: negative inside the lumen, positive beyond the wall."""
        f = self._f.EvaluateFunction
        return np.array([self._sign * f(*p) for p in np.asarray(points_mm, float)])

    def check(self, segments_world_m, inserted_mm: float) -> WallCheck:
        """Check the inserted part of the wire. ``segments_world_m``: body positions (N, 3) from proximal to
        tip (their segment poses, world m); ``inserted_mm``: insertion depth (their DeviceState)."""
        pts = np.asarray(segments_world_m, dtype=np.float64).reshape(-1, 3) * 1000.0
        if len(pts) == 0:
            return WallCheck(False, 0.0, 0.0, None, 0)
        arc_from_tip = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts[::-1], axis=0), axis=1))])[::-1]
        inside_body = pts[arc_from_tip <= max(float(inserted_mm) - 2.0, 0.0)]
        inside_body = inside_body[inside_body[:, 2] > 0.0]  # their containment skips bodies before the entrance (z < 0)
        if len(inside_body) < 2:
            return WallCheck(False, 0.0, 0.0, None, len(inside_body))
        curve = catmull_rom(inside_body, self.samples_per_mm)
        d = self.signed_distance_mm(curve)
        k = int(np.argmax(d))
        depth = max(float(d[k]), 0.0)
        out = d > self.shallow_mm
        kind = ""
        if depth > self.depth_mm:
            kind = "deep"
        else:  # each stretch beyond the wall: a shortcut into another lumen?
            edges = np.flatnonzero(np.diff(np.r_[0, out.astype(int), 0]))
            for lo, hi in zip(edges[::2], edges[1::2]):
                if self._is_shortcut(curve, d, int(lo), int(hi)):
                    kind = "shortcut"
                    break
        return WallCheck(bool(kind), round(depth, 2), round(float(out.sum()) / self.samples_per_mm, 1),
                         (curve[k] / 1000.0).round(5).tolist() if depth > 0 else None, len(inside_body), kind)


def guidewire_bodies(sample: dict) -> np.ndarray:
    """Body positions (N, 3), world m, of a device-state sample (``seg``: 7 values per body: position, quaternion)."""
    return np.asarray(sample.get("seg") or [], dtype=np.float64).reshape(-1, 7)[:, :3]


GW_INSERTION, GW_SIN, GW_COS = 17, 18, 19  # organiser state indices (openng3_rl.observation)
RAD_PER_STEP = 0.2  # guidewire rotation per 5 Hz step at |w| = 1


class WallSafeController:
    """Safety layer around any controller on their physics: never keep pushing while the wire crosses a wall.

    Wraps ``policy`` (``act(obs)`` / ``reset()``: the A* + entry-roll controller or a cloned SAC policy) and
    checks the wire's noisy segment poses with ``WallGuard`` before every command. It only makes motion
    less aggressive or backs off, never pushes harder than the policy:

    * ``pass``: the policy's action, unchanged, while the wire is inside the vessel
    * ``retract``: on a crossing, pull the guidewire back (full speed while it is still across the wall, then
      ``retract_speed``) until the wire is inside again
      and ``retract_mm`` behind where the crossing was seen
    * ``reroll``: turn the guidewire by the next offset in ``reroll_deg`` from the roll at the crossing
    * ``advance``: the policy's push, capped at ``creep``, with the roll held at the new angle; a new crossing
      starts the next retry
    * ``abort``: after the last retry, retract and hold (handed to the operator); ``events`` records why

    ``bind(env)`` gives it the lockstep env, whose ``bridge.noisy`` holds the device states their sensor
    node publishes (in their stack: subscribe to ``/endovascular/guidewire/state/noisy``).
    """

    def __init__(self, policy, guard: WallGuard, retract_mm: float = 15.0, retract_speed: float = 0.5,
                 reroll_deg: tuple[float, ...] = (20.0, -20.0, 40.0, -40.0), creep: float = 0.3, roll_tol_deg: float = 3.0):
        self.policy, self.guard = policy, guard
        self.retract_mm, self.retract_speed, self.creep = retract_mm, retract_speed, creep
        self.reroll_deg, self.roll_tol = tuple(reroll_deg), np.radians(roll_tol_deg)
        self.env = None

    def bind(self, env) -> None:
        self.env = env

    def reset(self) -> None:
        self.policy.reset()
        self.mode, self.retry, self.events = "pass", 0, []
        self.rot, self.last_raw = 0.0, None
        self.retract_to = self.roll_at_cross = self.target_rot = 0.0

    def _rotation(self, s: np.ndarray) -> float:
        raw = float(np.arctan2(s[GW_SIN], s[GW_COS]))
        self.rot = raw if self.last_raw is None else self.rot + (raw - self.last_raw + np.pi) % (2 * np.pi) - np.pi
        self.last_raw = raw
        return self.rot

    def _bodies(self) -> np.ndarray:
        noisy = self.env.bridge.noisy["guidewire"] if self.env is not None else []
        return guidewire_bodies(noisy[-1]) if noisy else np.zeros((0, 3))

    def act(self, obs: dict) -> np.ndarray:
        s = np.asarray(obs["state"], np.float64)
        ins_mm, rot = float(s[GW_INSERTION]) * 1000.0, self._rotation(s)
        proposed = np.asarray(self.policy.act(obs), np.float32)  # keep the policy's own state current
        check = self.guard.check(self._bodies(), ins_mm)
        if check.crossing and self.mode in ("pass", "advance"):
            self.events.append({"event": "wall_crossing", "mode": self.mode, "retry": self.retry, "ins_mm": round(ins_mm, 1),
                                "depth_mm": check.depth_mm, "where_world_m": check.where_world_m})
            self.mode, self.retract_to = "retract", ins_mm - self.retract_mm
            if self.retry == 0:
                self.roll_at_cross = rot
        if self.mode == "pass":
            return proposed
        a = np.zeros(4, np.float32)
        if self.mode in ("retract", "abort"):
            if check.crossing:
                a[0] = -1.0  # still across the wall: out at full speed
                return a
            if ins_mm > self.retract_to:
                a[0] = -self.retract_speed
                return a
            if self.mode == "abort":
                return a  # clear of the wall and backed off: hold for the operator
            if self.retry >= len(self.reroll_deg):
                self.mode = "abort"
                self.events.append({"event": "abort", "reason": "every retry crossed the wall", "ins_mm": round(ins_mm, 1)})
                return a
            self.target_rot = self.roll_at_cross + np.radians(self.reroll_deg[self.retry])
            self.retry += 1
            self.mode = "reroll"
            self.events.append({"event": "reroll", "retry": self.retry, "roll_deg": round(float(np.degrees(self.target_rot)), 1)})
        err = self.target_rot - rot
        if self.mode == "reroll":
            if abs(err) > self.roll_tol:
                a[1] = np.clip(err / RAD_PER_STEP, -1.0, 1.0)
                return a
            self.mode = "advance"
        # advance: the policy's push, slower, with the roll held at the new angle
        a[0] = min(float(proposed[0]), self.creep)
        a[1] = np.clip(err / RAD_PER_STEP, -0.5, 0.5)
        return a
