"""Interactive 3D check: vessel surface (transparent) + planned path(s) + start/goal.
Also exports vessel.obj and path.obj for Blender/MeshLab/Slicer."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, pyvista as pv
from skimage.measure import marching_cubes
import cta_to_trajectory as m

p = argparse.ArgumentParser()
p.add_argument("--image", required=True)
p.add_argument("--mask", required=True)
p.add_argument("--routes", nargs="+", required=True, help="output folders, e.g. nav_dijkstra nav_device_aware")
p.add_argument("--export-dir", default="viewer_export")
a = p.parse_args()

vol = m.load_volume(a.image)
mask = m.load_volume(a.mask).data > 0
idx = np.argwhere(mask); lo, hi = idx.min(0), idx.max(0) + 1
sub = np.pad(mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], 1)
v, f, _, _ = marching_cubes(sub.astype(np.float32), 0.5)
v = vol.to_world(v - 1 + lo)                       # voxel -> world mm (same space as the route)
surf = pv.PolyData(v, np.c_[np.full(len(f), 3), f].ravel()).smooth(30)

out = Path(a.export_dir); out.mkdir(exist_ok=True)
surf.save(out / "vessel.obj")

pl = pv.Plotter()
pl.add_mesh(surf, color="tomato", opacity=0.25, label="vessel lumen")
colors = ["dodgerblue", "limegreen", "gold", "magenta"]
for k, rd in enumerate(a.routes):
    tr = pd.read_csv(Path(rd) / "trajectory.csv")
    pts = tr[["x_mm", "y_mm", "z_mm"]].to_numpy()
    line = pv.lines_from_points(pts)
    tube = line.tube(radius=0.25)
    pl.add_mesh(tube, color=colors[k % 4], label=Path(rd).name)
    tube.save(out / f"path_{Path(rd).name}.obj" if False else out / f"path_{Path(rd).name}.ply")
    # arrows along the direction of travel
    step = max(len(pts) // 15, 1)
    pl.add_arrows(pts[::step], tr[["tangent_x", "tangent_y", "tangent_z"]].to_numpy()[::step], mag=3, color=colors[k % 4])
    r = json.loads((Path(rd) / "route.json").read_text())
    if k == 0:
        pl.add_mesh(pv.Sphere(radius=1.2, center=r["snapped_start_mm"]), color="green", label="start")
        pl.add_mesh(pv.Sphere(radius=1.2, center=r["snapped_goal_mm"]), color="red", label="goal")
pl.add_legend(); pl.add_axes()
pl.show(screenshot=str(out / "screenshot.png"))
print("exported to", out)