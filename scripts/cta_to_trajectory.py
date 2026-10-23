"""CTA/MRA volume -> vessel centerline graph -> route -> timed reference trajectory.

Classical pipeline (no learning):
  1. load volume (NIfTI via nibabel, DICOM folder / NRRD / MHA via SimpleITK)
  2. vessel mask: your own segmentation (--mask, recommended) or a simple threshold
  3. skeleton (skimage) + radius from Euclidean distance transform
  4. voxel graph (26-neighbourhood) -> Dijkstra / device-aware Dijkstra
  5. spline smoothing, arc-length resampling, curvature, speed profile, timestamps
  6. export route.json, trajectory.csv, metrics.json (+ optional route.vtp)

All coordinates are WORLD millimetres (the volume's affine). Start/goal must be given in the
same world space as the image header (check in 3D Slicer: hover = RAS/LPS coordinates).
NOTE: nibabel gives RAS, SimpleITK gives LPS. Keep your start/goal in the convention of the loader.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import splev, splprep
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


# ----------------------------------------------------------------------------- loading
@dataclass
class Volume:
    data: np.ndarray        # (i, j, k)
    affine: np.ndarray      # 4x4 voxel -> world mm
    spacing: np.ndarray     # mm per voxel along i, j, k

    def to_vox(self, xyz) -> np.ndarray:
        return (np.linalg.inv(self.affine) @ np.append(xyz, 1.0))[:3]

    def to_world(self, ijk) -> np.ndarray:
        ijk = np.atleast_2d(ijk).astype(float)
        return (self.affine @ np.c_[ijk, np.ones(len(ijk))].T).T[:, :3]


def load_volume(path: str) -> Volume:
    p = Path(path)
    if p.suffix in (".nii", ".gz"):
        import nibabel as nib  # lazy: only needed for NIfTI
        img = nib.load(str(p))
        aff = img.affine
        return Volume(np.asarray(img.dataobj, dtype=np.float32), aff, np.linalg.norm(aff[:3, :3], axis=0))
    import SimpleITK as sitk  # DICOM folder, .nrrd, .mha ...
    if p.is_dir():
        r = sitk.ImageSeriesReader()
        r.SetFileNames(r.GetGDCMSeriesFileNames(str(p)))
        img = r.Execute()
    else:
        img = sitk.ReadImage(str(p))
    arr = sitk.GetArrayFromImage(img).transpose(2, 1, 0).astype(np.float32)  # (x,y,z)
    sp = np.array(img.GetSpacing())
    D = np.array(img.GetDirection()).reshape(3, 3)
    aff = np.eye(4)
    aff[:3, :3] = D * sp
    aff[:3, 3] = img.GetOrigin()
    return Volume(arr, aff, sp)


# ----------------------------------------------------------------------------- segmentation
def vessel_mask(vol: Volume, mask_path: str | None, modality: str, threshold: float | None) -> np.ndarray:
    """Prefer a real segmentation (nnU-Net/TopCoW/TotalSegmentator/Slicer). Threshold is a crude fallback."""
    if mask_path:
        m = load_volume(mask_path).data > 0
    else:
        d = vol.data
        if threshold is None:
            # CTA: contrast lumen ~ >200 HU (bone also passes -> mask/crop first!). MRA: top intensities.
            threshold = 200.0 if modality == "cta" else float(np.percentile(d[d > 0], 97))
        m = d > threshold
    m = ndi.binary_closing(m, iterations=1)
    return m


def keep_component_containing(mask: np.ndarray, seed_ijk: np.ndarray) -> np.ndarray:
    lab, _ = ndi.label(mask, structure=np.ones((3, 3, 3)))
    l = lab[tuple(np.clip(np.round(seed_ijk).astype(int), 0, np.array(mask.shape) - 1))]
    if l == 0:  # seed not in mask -> snap to nearest mask voxel
        idx = ndi.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
        s = np.clip(np.round(seed_ijk).astype(int), 0, np.array(mask.shape) - 1)
        l = lab[tuple(idx[:, s[0], s[1], s[2]])]
    return lab == l


# ----------------------------------------------------------------------------- graph
class VascularGraph:
    """Skeleton-voxel graph. Node = skeleton voxel, edge weight = physical length (mm)."""

    def __init__(self, mask: np.ndarray, vol: Volume):
        self.vol = vol
        self.edt = ndi.distance_transform_edt(mask, sampling=vol.spacing)  # mm to nearest wall
        sk = skeletonize(mask)
        self.vox = np.argwhere(sk)
        if len(self.vox) < 2:
            raise ValueError("Skeleton is empty - mask/threshold did not capture a vessel.")
        self.xyz = vol.to_world(self.vox)
        self.radius = self.edt[tuple(self.vox.T)]  # local lumen radius, mm
        self.tree = cKDTree(self.xyz)
        index = {tuple(v): n for n, v in enumerate(self.vox)}
        G = nx.Graph()
        G.add_nodes_from(range(len(self.vox)))
        offs = [o for o in np.ndindex(3, 3, 3) if o != (1, 1, 1)]
        for n, v in enumerate(self.vox):
            for o in offs:
                m = index.get((v[0] + o[0] - 1, v[1] + o[1] - 1, v[2] + o[2] - 1))
                if m is not None and m > n:
                    G.add_edge(n, m, length=float(np.linalg.norm(self.xyz[n] - self.xyz[m])))
        self.G = G
        self.deg = dict(G.degree())

    def snap(self, xyz, max_dist_mm=10.0) -> int:
        d, n = self.tree.query(xyz)
        if d > max_dist_mm:
            raise ValueError(f"Point {np.round(xyz, 1)} is {d:.1f} mm from the centerline (> {max_dist_mm}). "
                             "Wrong coordinate space/units or point outside the vessel.")
        return int(n)

    def components(self):
        return list(nx.connected_components(self.G))

    def bifurcations(self):
        return [n for n, d in self.deg.items() if d >= 3]

    # --- planners
    def dijkstra(self, s: int, g: int) -> list[int]:
        if not nx.has_path(self.G, s, g):
            raise ValueError("Start and goal are in disconnected vessel components.")
        return nx.dijkstra_path(self.G, s, g, weight="length")

    def device_aware(self, s: int, g: int, device_d_mm: float, clearance_mm=0.1,
                     w_len=1.0, w_rad=5.0, w_curv=20.0) -> list[int]:
        """J = w_L*L + w_R*R + w_C*C evaluated per edge.
        - hard constraint: lumen radius >= device_radius + clearance
        - R: penalises narrow vessels  ((r_ref / r)^2 - 1, r_ref = median radius)
        - C: penalises local direction change between consecutive skeleton steps (needs edge-state
             search, so done with a small custom Dijkstra on (node, previous_node) states)
        Bifurcation turn penalty is covered by the same direction-change term (a sharp branch turn
        is a large direction change)."""
        import heapq
        rmin = device_d_mm / 2 + clearance_mm
        ok = self.radius >= rmin
        if not (ok[s] and ok[g]):
            raise ValueError("Start or goal violates the clearance constraint for this device.")
        rref = float(np.median(self.radius))
        cost = lambda a, b: w_len * self.G[a][b]["length"] * (1 + w_rad * max((rref / self.radius[b]) ** 2 - 1, 0))
        pq, best, prev = [(0.0, s, -1)], {(s, -1): 0.0}, {}
        while pq:
            c, u, p = heapq.heappop(pq)
            if c > best.get((u, p), np.inf):
                continue
            if u == g:
                path, st = [], (u, p)
                while st in prev or st == (s, -1):
                    path.append(st[0])
                    if st == (s, -1):
                        break
                    st = prev[st]
                return path[::-1]
            for v in self.G[u]:
                if not ok[v] or v == p:
                    continue
                turn = 0.0
                if p >= 0:
                    a, b = self.xyz[u] - self.xyz[p], self.xyz[v] - self.xyz[u]
                    turn = np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9), -1, 1))
                nc = c + cost(u, v) + w_curv * turn
                if nc < best.get((v, u), np.inf):
                    best[(v, u)] = nc
                    prev[(v, u)] = (u, p)
                    heapq.heappush(pq, (nc, v, u))
        raise ValueError("No route satisfies the device clearance constraint (vessel too narrow).")


# ----------------------------------------------------------------------------- trajectory
def smooth_resample(pts: np.ndarray, ds=0.5, smooth_mm=0.5):
    """Spline through the centerline polyline, resampled uniformly in arc length."""
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    keep = np.r_[True, np.diff(d) > 1e-6]
    pts, d = pts[keep], d[keep]
    k = min(3, len(pts) - 1)
    tck, u = splprep(pts.T, u=d / d[-1], s=smooth_mm ** 2 * len(pts), k=k)
    uu = np.linspace(0, 1, max(int(d[-1] / (ds / 5)), 200))
    P = np.array(splev(uu, tck)).T
    s_dense = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    s = np.arange(0, s_dense[-1], ds)
    s = np.r_[s, s_dense[-1]]
    ur = np.interp(s, s_dense, uu)
    r = np.array(splev(ur, tck)).T
    d1 = np.array(splev(ur, tck, der=1)).T
    d2 = np.array(splev(ur, tck, der=2)).T
    cross = np.linalg.norm(np.cross(d1, d2), axis=1)
    kappa = cross / np.maximum(np.linalg.norm(d1, axis=1) ** 3, 1e-9)
    tan = d1 / np.maximum(np.linalg.norm(d1, axis=1, keepdims=True), 1e-9)
    return r, tan, kappa, s


def speed_profile(s, kappa, v_max=10.0, a_max=20.0, d_max=20.0, a_lat=15.0, eps=1e-3):
    """v(s) = min(v_max, sqrt(a_lat/|k|)), then forward (accel) / backward (decel) passes; v=0 at both ends."""
    v = np.minimum(v_max, np.sqrt(a_lat / np.maximum(np.abs(kappa), eps)))
    v[0] = v[-1] = 0.0
    for i in range(len(s) - 1):
        v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2 * a_max * (s[i + 1] - s[i])))
    for i in range(len(s) - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * d_max * (s[i + 1] - s[i])))
    ds = np.diff(s)
    dt = 2 * ds / np.maximum(v[:-1] + v[1:], 1e-6)  # consistent with trapezoidal speed
    t = np.r_[0, np.cumsum(dt)]
    acc = np.gradient(v, t)
    return t, v, acc


def lumen_check(vol: Volume, edt: np.ndarray, pts: np.ndarray, device_d_mm: float):
    ijk = np.array([vol.to_vox(p) for p in pts]).T
    clearance = ndi.map_coordinates(edt, ijk, order=1, mode="nearest")  # mm to wall
    return clearance, bool(np.all(clearance > device_d_mm / 2))


# ----------------------------------------------------------------------------- main
def run(a):
    t0 = time.time()
    vol = load_volume(a.volume)
    start, goal = np.array(a.start, float), np.array(a.goal, float)
    mask = vessel_mask(vol, a.mask, a.modality, a.threshold)
    mask = keep_component_containing(mask, vol.to_vox(start))
    VG = VascularGraph(mask, vol)
    s, g = VG.snap(start, a.snap_mm), VG.snap(goal, a.snap_mm)
    path_ids = VG.dijkstra(s, g) if a.algorithm == "dijkstra" else VG.device_aware(s, g, a.device_diameter_mm)
    pts = VG.xyz[path_ids]
    r, tan, kappa, sarc = smooth_resample(pts, a.ds, a.smooth_mm)
    t, v, acc = speed_profile(sarc, kappa, a.v_max, a.a_max, a.a_max, a.a_lat)
    clr, lumen_ok = lumen_check(vol, VG.edt, r, a.device_diameter_mm)

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "trajectory.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow("time_s,x_mm,y_mm,z_mm,tangent_x,tangent_y,tangent_z,speed_mm_s,"
                   "acceleration_mm_s2,curvature_mm_inv,distance_mm".split(","))
        for i in range(len(t)):
            w.writerow([t[i], *r[i], *tan[i], v[i], acc[i], kappa[i], sarc[i]])
    n_bif = int(sum(VG.deg[n] >= 3 for n in path_ids))
    (out / "route.json").write_text(json.dumps({
        "algorithm": a.algorithm, "start_xyz_mm": start.tolist(), "goal_xyz_mm": goal.tolist(),
        "snapped_start_mm": VG.xyz[s].tolist(), "snapped_goal_mm": VG.xyz[g].tolist(),
        "centerline_xyz_mm": pts.tolist(), "path_length_mm": float(sarc[-1]),
        "min_vessel_radius_mm": float(VG.radius[path_ids].min()),
        "geometry_validation": {"smoothed_path_inside_lumen_for_device": lumen_ok},
        "note": "Desired tip reference path; NOT an actuator command."}, indent=1))
    (out / "metrics.json").write_text(json.dumps({
        "planning_runtime_s": time.time() - t0, "route_found": True,
        "total_path_length_mm": float(sarc[-1]), "max_curvature_mm_inv": float(kappa.max()),
        "mean_curvature_mm_inv": float(kappa.mean()),
        "min_clearance_mm": float(clr.min() - a.device_diameter_mm / 2),
        "bifurcation_like_nodes_on_path": n_bif, "duration_s": float(t[-1]),
        "geometry_validation_passed": lumen_ok}, indent=1))
    try:  # optional VTP for ParaView/PyVista
        import pyvista as pv
        poly = pv.lines_from_points(r)
        poly["speed_mm_s"] = v
        poly.save(out / "route.vtp")
    except ImportError:
        pass
    print(f"OK  {a.algorithm}: {sarc[-1]:.1f} mm, {t[-1]:.1f} s, lumen_ok={lumen_ok}, output -> {out}")


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--volume", required=True, help=".nii/.nii.gz, DICOM folder, .nrrd, .mha")
    p.add_argument("--mask", help="vessel segmentation volume (recommended)")
    p.add_argument("--modality", choices=["cta", "mra"], default="cta")
    p.add_argument("--threshold", type=float)
    p.add_argument("--start", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    p.add_argument("--goal", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    p.add_argument("--algorithm", choices=["dijkstra", "device-aware"], default="device-aware")
    p.add_argument("--device-diameter-mm", type=float, default=0.89)  # 0.035" guidewire
    p.add_argument("--snap-mm", type=float, default=10.0)
    p.add_argument("--ds", type=float, default=0.5)
    p.add_argument("--smooth-mm", type=float, default=0.5)
    p.add_argument("--v-max", type=float, default=10.0)
    p.add_argument("--a-max", type=float, default=20.0)
    p.add_argument("--a-lat", type=float, default=15.0)
    p.add_argument("--output", default="navigation_output")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
