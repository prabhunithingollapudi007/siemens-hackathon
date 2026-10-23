"""Write organiser scenarios for other vessel meshes (the 0248 stenosis variants, other patients).

Their RL stack loads one *scenario*: a folder with ``vessel.obj`` (authored units, here cm, OBJ axes
= their mesh frame), ``goals.txt`` (points in that frame) and ``scenario.json`` (manifest with the
files' sha256). With ``bake.axis = "authored"`` the mesh-to-world transform is a pure scale
(``meters_per_unit``), so a mesh written in the same object frame as their ``cas-reference`` mesh
lands in the same world coordinates, and goals converted with the fitted ``OrganiserFrame`` stay
valid. ``write_scenario`` does that for a VMR surface: decimate to about their triangle count,
convert VMR mm to their object frame (cm), write the three files, and validate the folder with
their own loader (``openng3_rl.scenario.load_scenario``).

The hand-over scenario stays ``cas-reference`` (AGENTS.md invariant 5); these scenarios are for
planning and evaluation on other anatomies, under ``data/scenarios/`` (ignored by git).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from stroke_nav.organiser.frame import OrganiserFrame, default_frame
from stroke_nav.organiser.spec import CAS_REFERENCE_MANIFEST

DEFAULT_SCENARIO_ROOT = Path("data") / "scenarios"


def scenario_name(model: str, variant: str | None) -> str:
    """Their ``validate_name`` allows letters, digits, ``-`` and ``_``: ``0248_H_AOCERE_CAS-10Right70Left``."""
    return f"{model}-{variant}" if variant else model


def variant_surface(model: str, variant: str | None, raw_dir: str | Path = Path("data") / "raw") -> Path:
    """The VMR surface file of a model variant (``data/raw/<model>/Models/<variant>.vtp``)."""
    models = Path(raw_dir) / model / "Models"
    if variant:
        p = models / f"{variant}.vtp"
        if p.is_file():
            return p
    candidates = sorted(models.glob("*.vtp")) if models.is_dir() else []
    if not variant and candidates:
        return candidates[0]
    raise FileNotFoundError(f"no surface for {model} {variant or ''} under {models} (run fetch_data.py --variant)")


def decimated_triangles(surface: str | Path, target_triangles: int = 10_000, surface_unit_scale_mm: float = 10.0
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (VMR mm) and triangles of the surface, decimated to about ``target_triangles``."""
    import pyvista as pv

    mesh = pv.read(str(surface)).extract_surface(algorithm="dataset_surface").triangulate().clean()
    n = mesh.n_cells
    if n > target_triangles:
        mesh = mesh.decimate_pro(1.0 - target_triangles / n, preserve_topology=True).clean()
    faces = mesh.faces.reshape(-1, 4)
    assert np.all(faces[:, 0] == 3), "surface must be triangles"
    tri = faces[:, 1:].astype(np.int64)
    verts = np.asarray(mesh.points, dtype=np.float64) * surface_unit_scale_mm
    # drop degenerate triangles (their loader rejects them) and unused vertices
    a, b, c = verts[tri[:, 0]], verts[tri[:, 1]], verts[tri[:, 2]]
    area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    tri = tri[area2 > 1e-9]
    used = np.unique(tri)
    remap = -np.ones(len(verts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[tri]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_scenario(model: str, variant: str | None, goals_vmr_mm, root: str | Path = DEFAULT_SCENARIO_ROOT,
                   frame: OrganiserFrame | None = None, target_triangles: int = 10_000, name: str | None = None,
                   raw_dir: str | Path = Path("data") / "raw", validate: bool = True) -> Path:
    """Write ``<root>/<name>/{vessel.obj, goals.txt, scenario.json}`` for a VMR model variant, with the
    goals (VMR mm) converted to their mesh frame. Returns the scenario folder."""
    frame = frame or default_frame()
    name = name or scenario_name(model, variant)
    out = Path(root) / name
    out.mkdir(parents=True, exist_ok=True)
    verts_mm, tri = decimated_triangles(variant_surface(model, variant, raw_dir), target_triangles)
    verts_cm = frame.vmr_mm_to_obj_cm(verts_mm)
    goals_cm = np.atleast_2d(frame.vmr_mm_to_obj_cm(np.asarray(goals_vmr_mm, dtype=float)))
    lo, hi = verts_cm.min(0), verts_cm.max(0)
    for g in goals_cm:
        if np.any(g < lo) or np.any(g > hi):
            raise ValueError(f"goal {g} lies outside the mesh bounds {lo}..{hi} (their loader rejects it)")
    obj = out / "vessel.obj"
    with open(obj, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# {model} {variant or ''}: VMR surface decimated to {len(tri)} triangles, in the organiser's mesh frame (cm).\n")
        f.write("# Converted with the fitted OrganiserFrame (stroke_nav.organiser.frame); same frame as cas-reference/vessel.obj.\n")
        for x, y, z in verts_cm:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in tri + 1:
            f.write(f"f {a} {b} {c}\n")
    goals = out / "goals.txt"
    with open(goals, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# Goals in the mesh frame (cm) for {name}; written by stroke_nav.organiser.scenario.\n")
        for g in goals_cm:
            f.write(f"{g[0]:.3f},{g[1]:.3f},{g[2]:.3f}\n")
    manifest = {
        "bake": dict(CAS_REFERENCE_MANIFEST.get("bake", {"axis": "authored", "flip": False, "roll_180": False})),
        "goal_frame": "mesh", "goals_file": "goals.txt", "goals_sha256": _sha256(goals),
        "mesh_file": "vessel.obj", "mesh_format": "obj", "mesh_sha256": _sha256(obj),
        "meters_per_unit": CAS_REFERENCE_MANIFEST.get("meters_per_unit", 0.01), "name": name, "schema_version": 1,
        "selection": "uniform", "units": "cm",
    }
    (out / "scenario.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if validate:
        validate_scenario(name, root)
    return out


def validate_scenario(name: str, root: str | Path = DEFAULT_SCENARIO_ROOT) -> dict:
    """Load the scenario with the organiser's own loader; returns their summary (raises if invalid)."""
    from stroke_nav.organiser.live.lockstep import organiser_rl

    sc = organiser_rl("scenario")
    scenario = sc.load_scenario(name, Path(root).resolve())
    return sc.scenario_summary(scenario)
