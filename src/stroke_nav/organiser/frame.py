"""Rigid transform between our VMR anatomy and the organiser's simulator world frame.

Frames
------
* VMR (ours): SimVascular model coordinates converted to mm (RAS-like axes), as stored in
  ``data/processed/*.npz``.
* Organiser mesh: ``vessel.obj`` / ``0248_H_AOCERE_CAS_reference_10k.obj`` in cm. It is our surface,
  decimated to 10k faces, rotated ~19.5 deg and translated so the device entry sits at the origin
  with insertion along +Z.
* Organiser world: mesh * meters_per_unit (0.01) -> metres. Tip poses, goals and ``segment_poses``
  on ROS topics are in this frame.

Relation (fitted with trimmed ICP, median residual 0.45 mm, max 1.6 mm = decimation error):

    vmr_cm = M @ obj_cm + t_cm          world_m = 0.01 * obj_cm

Note: their goals.txt header says "OBJ LPS (R and A negated)", but the fit shows no axis flip
between our VMR frame and their OBJ (a flipped fit is ~16x worse). Use this module, not the header.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Fitted on 0248_H_AOCERE_CAS / 40Left85Right (Models/40Left85Right.vtp) vs their vessel.obj
# (sha256 04fbd581...). Re-derive with scripts/fit_organiser_frame.py.
_DEFAULT_M = np.array([
    [0.94562, -0.00267, -0.32525],
    [-0.02330, 0.99684, -0.07591],
    [0.32443, 0.07936, 0.94258],
])
_DEFAULT_T_CM = np.array([-10.0697, 3.9739, -38.7242])


def _orthonormal(m: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


@dataclass
class OrganiserFrame:
    M: np.ndarray = field(default_factory=lambda: _orthonormal(_DEFAULT_M))  # vmr_cm = M @ obj_cm + t_cm
    t_cm: np.ndarray = field(default_factory=lambda: _DEFAULT_T_CM.copy())
    meters_per_unit: float = 0.01
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ points
    def obj_cm_to_vmr_mm(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        return 10.0 * (p @ self.M.T + self.t_cm)

    def vmr_mm_to_obj_cm(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        return (p / 10.0 - self.t_cm) @ self.M  # M orthonormal: inverse = transpose

    def vmr_mm_to_world_m(self, p) -> np.ndarray:
        return self.vmr_mm_to_obj_cm(p) * self.meters_per_unit

    def world_m_to_vmr_mm(self, p) -> np.ndarray:
        return self.obj_cm_to_vmr_mm(np.asarray(p, dtype=np.float64) / self.meters_per_unit)

    # ------------------------------------------------------------------ directions / rotations
    @property
    def R_world_from_vmr(self) -> np.ndarray:
        return self.M.T

    def vmr_dir_to_world(self, d) -> np.ndarray:
        return np.asarray(d, dtype=np.float64) @ self.M  # == (M.T @ d.T).T

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return {
            "schema": "stroke-nav/organiser-frame/1",
            "relation": "vmr_cm = M @ obj_cm + t_cm; world_m = meters_per_unit * obj_cm; vmr_mm = 10 * vmr_cm",
            "M": self.M.round(8).tolist(),
            "t_cm": self.t_cm.round(6).tolist(),
            "meters_per_unit": self.meters_per_unit,
            "meta": self.meta,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "OrganiserFrame":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(np.asarray(d["M"], float), np.asarray(d["t_cm"], float), float(d["meters_per_unit"]), d.get("meta", {}))


def default_frame(processed_dir: str | Path = "data/processed") -> OrganiserFrame:
    """Fitted frame from ``data/processed/organiser_frame.json`` if present, else the built-in fit."""
    p = Path(processed_dir) / "organiser_frame.json"
    if p.is_file():
        return OrganiserFrame.load(p)
    return OrganiserFrame(meta={"source": "built-in fit (0248 40Left85Right vs cas-reference vessel.obj)"})


# ---------------------------------------------------------------------- fitting
def read_obj_vertices(path: str | Path) -> np.ndarray:
    verts = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not verts:
        raise ValueError(f"no vertices in {path}")
    return np.asarray(verts)


def _kabsch(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ca, cb = a.mean(0), b.mean(0)
    h = (a - ca).T @ (b - cb)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, cb - r @ ca


def fit_frame(vmr_surface: str | Path, obj_path: str | Path, vmr_unit_scale: float = 1.0,
              iterations: int = 80, keep_percentile: float = 80.0) -> OrganiserFrame:
    """Rigidly register their OBJ (cm) onto our VMR surface (file units * vmr_unit_scale = cm).

    Their mesh is a decimated subset of ours, so OBJ points are matched to the dense VMR surface
    (trimmed ICP). Tries the identity orientation and three axis flips; keeps the best.
    """
    import pyvista as pv
    from scipy.spatial import cKDTree

    vmr = pv.read(str(vmr_surface)).extract_surface(algorithm="dataset_surface").points.astype(float) * vmr_unit_scale
    obj = read_obj_vertices(obj_path)
    tree = cKDTree(vmr)
    best = None
    for flip in (np.diag([1.0, 1.0, 1.0]), np.diag([-1.0, -1.0, 1.0]), np.diag([-1.0, 1.0, 1.0]), np.diag([1.0, -1.0, 1.0])):
        p = obj @ flip.T
        t = vmr.mean(0) - p.mean(0)
        cur = p + t
        R = np.eye(3)
        for _ in range(iterations):
            d, j = tree.query(cur)
            keep = d < max(np.percentile(d, keep_percentile), 0.05)
            r, tt = _kabsch(cur[keep], vmr[j[keep]])
            cur = cur @ r.T + tt
            R, t = r @ R, r @ t + tt
        d, _ = tree.query(cur)
        score = float(np.median(d))
        if best is None or score < best[0]:
            best = (score, flip, R, t, d)
    score, flip, R, t, d = best
    M = _orthonormal(R @ flip)
    meta = {
        "source": "fit_frame",
        "vmr_surface": str(vmr_surface),
        "obj": str(obj_path),
        "obj_sha256": hashlib.sha256(Path(obj_path).read_bytes()).hexdigest(),
        "median_residual_mm": round(10 * float(np.median(d)), 3),
        "p90_residual_mm": round(10 * float(np.percentile(d, 90)), 3),
        "max_residual_mm": round(10 * float(d.max()), 3),
        "rotation_deg": round(float(np.degrees(np.arccos(np.clip((np.trace(M) - 1) / 2, -1, 1)))), 3),
        "axis_flip": [int(x) for x in np.diag(flip)],
    }
    return OrganiserFrame(M, np.asarray(t, float), 0.01, meta)


# ---------------------------------------------------------------------- goals
def read_goals_txt(path: str | Path) -> np.ndarray:
    """Organiser goals.txt: one 'x,y,z' (or space separated) row per goal, '#' comments; mesh units."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        vals = [float(v) for v in line.replace(",", " ").split()]
        if len(vals) != 3:
            raise ValueError(f"bad goal row {line!r} in {path}")
        rows.append(vals)
    return np.asarray(rows, dtype=np.float64)


def write_goals_txt(path: str | Path, goals_obj_cm: np.ndarray, comment: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Stored coordinates: organiser mesh (OBJ) frame; native OBJ units (cm)."]
    if comment:
        lines.append(f"# {comment}")
    lines += [",".join(f"{v:.3f}" for v in g) for g in np.atleast_2d(goals_obj_cm)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------- frames for other meshes
def entry_frame(anat, start: int | None = None, meta: dict | None = None) -> "OrganiserFrame":
    """A frame that places a VMR model the way the organiser's physics page expects an *authored*
    mesh: the insertion entrance (the default start node) at the origin and the vessel axis there
    along +Z, since their wire is built on the -Z side and pushed in along +Z. Everything else about
    the placement is arbitrary (the roll about Z is chosen so the model's +Y stays as upright as it can).

    Relation as for the fitted 0248 frame: ``vmr_cm = M @ obj_cm + t_cm``; ``world_m = 0.01 * obj_cm``.
    """
    start = anat.default_start() if start is None else int(start)
    p0_cm = np.asarray(anat.nodes[start], dtype=np.float64) / 10.0
    nb = list(anat.neighbors[start])
    z = np.asarray(anat.forward_direction(start, nb[0]), dtype=np.float64)
    z /= np.linalg.norm(z)
    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(up, z)) > 0.9:
        up = np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])  # rows: obj axes expressed in VMR coordinates -> obj = R @ (vmr - p0)
    frame = OrganiserFrame(M=_orthonormal(R.T), t_cm=p0_cm, meta={"source": f"entry frame of {anat.name}: start node {start} at the origin, "
                                                                  "vessel axis +Z (authored bake)", "start_node": start, **(meta or {})})
    return frame


def entry_node(anat, frame: "OrganiserFrame") -> int:
    """The node where the robot enters: the entry frame's ``start_node`` (``make_scenario.py --start``),
    else the model's default start. Plan routes from here, so they match their physics' entrance."""
    node = (frame.meta or {}).get("start_node")
    return anat.default_start() if node is None else int(node)


def frame_path(model: str, processed_dir: str | Path = "data/processed") -> Path:
    return Path(processed_dir) / f"organiser_frame_{model}.json"


def frame_for(model: str, processed_dir: str | Path = "data/processed") -> "OrganiserFrame":
    """The organiser frame of a model: the fitted one for 0248 (``default_frame``), else the entry
    frame saved by ``scripts/make_scenario.py`` (``organiser_frame_<model>.json``)."""
    if model == "0248_H_AOCERE_CAS":
        return default_frame(processed_dir)
    p = frame_path(model, processed_dir)
    if not p.is_file():
        raise FileNotFoundError(f"no organiser frame for {model} at {p}: run scripts/make_scenario.py --model {model} first")
    return OrganiserFrame.load(p)
