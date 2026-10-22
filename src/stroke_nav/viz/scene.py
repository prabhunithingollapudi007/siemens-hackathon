"""3D scene for the playground: vessel surface, centerlines, devices, start/target markers, HUD."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from stroke_nav.env.devices import CATH, WIRE
from stroke_nav.env.vessel_env import VesselNavEnv

WIRE_COLOR = "#1f2937"
CATH_COLOR = "#f59e0b"
TARGET_COLOR = "#16a34a"
START_COLOR = "#2563eb"
VESSEL_COLOR = "#e8a0a0"


def load_surface(env: VesselNavEnv, target_cells: int = 80_000):
    """Load the anatomy's surface model in mm, decimated and cached next to the processed graph."""
    import pyvista as pv

    meta = env.anat.meta
    src = meta.get("surface_file")
    if not src or not Path(src).exists():
        return None
    cache = Path("data/processed") / f"{Path(src).stem}_{meta.get('model', 'model')}_display.vtp"
    if cache.exists():
        return pv.read(str(cache))
    surf = pv.read(src).extract_surface(algorithm="dataset_surface").triangulate()
    surf.points = surf.points * float(meta.get("unit_scale", 1.0))
    if surf.n_cells > target_cells:
        surf = surf.decimate(1.0 - target_cells / surf.n_cells)
    surf = surf.compute_normals(auto_orient_normals=True)
    surf.save(str(cache))
    return surf


class Scene:
    def __init__(self, env: VesselNavEnv, plotter, show_centerlines: bool = True):
        import pyvista as pv

        self.pv = pv
        self.env = env
        self.pl = plotter
        self.surface = load_surface(env)
        if self.surface is not None:
            self.pl.add_mesh(self.surface, color=VESSEL_COLOR, opacity=0.22, smooth_shading=True,
                             pickable=True, name="vessel")
        a = env.anat
        lines = []
        for u, v in a.graph.edges:
            lines += [2, int(u), int(v)]
        self.centerlines = pv.PolyData(a.nodes.copy(), lines=np.array(lines))
        self.centerlines["branch"] = a.branch_id.astype(float)
        self.cl_actor = self.pl.add_mesh(self.centerlines, scalars="branch", cmap="tab20", line_width=1.5,
                                         opacity=0.55, show_scalar_bar=False, pickable=True, name="centerlines")
        self.cl_actor.SetVisibility(show_centerlines)
        self.hud = None

    # ------------------------------------------------------------------ planned paths
    def draw_trajectory(self, traj, name: str = "plan", label_junctions: bool = True) -> None:
        """Planned path as a tube coloured by the reference speed, with bifurcation labels."""
        pv = self.pv
        line = pv.lines_from_points(traj.points)
        line["speed (mm/s)"] = traj.v_ref
        tube = line.tube(radius=0.9, n_sides=12)
        self.pl.add_mesh(tube, scalars="speed (mm/s)", cmap="turbo_r", clim=(0, float(max(traj.v_ref.max(), 1.0))),
                         opacity=0.55, name=name, pickable=False, reset_camera=False,
                         scalar_bar_args={"title": "mm/s", "vertical": True, "position_x": 0.88,
                                          "position_y": 0.25, "height": 0.5, "color": "black"})
        if label_junctions and traj.roll_events:
            pts, labels = [], []
            for e in traj.roll_events:
                pts.append(self.env.anat.nodes[e["node"]])
                labels.append(f"{e['to_branch']}  roll {e['phi_deg']:.0f}")
            self.pl.add_point_labels(np.array(pts), labels, font_size=9, point_size=6, point_color="#7c3aed",
                                     text_color="#4c1d95", shape_opacity=0.6, name=name + "_labels",
                                     always_visible=True, reset_camera=False)

    def draw_path(self, points: np.ndarray, color: str = "#6b7280", name: str = "baseline", radius: float = 0.5) -> None:
        tube = self.pv.lines_from_points(points).tube(radius=radius, n_sides=8)
        self.pl.add_mesh(tube, color=color, opacity=0.6, name=name, pickable=False, reset_camera=False)

    def clear(self, *names: str) -> None:
        for n in names:
            self.pl.remove_actor(n, reset_camera=False)

    # ------------------------------------------------------------------ updates
    def update(self, status: str = "", keys_help: str | None = None, show_hud: bool = True) -> None:
        """Redraw devices, markers and text. Actors are created once and then updated in place,
        so this stays cheap enough to call every frame of an interactive session."""
        env, sim, pv = self.env, self.env.sim, self.pv
        cache = self.__dict__.setdefault("_cache", {})
        for dev, color, radius, name in ((WIRE, WIRE_COLOR, 0.35, "wire"), (CATH, CATH_COLOR, 0.8, "cath")):
            pts = sim.device_polyline(dev)
            if len(pts) >= 2:
                tube = pv.lines_from_points(pts).tube(radius=radius, n_sides=8)
                if name in cache:
                    cache[name].copy_from(tube)
                else:
                    cache[name] = tube
                    self.pl.add_mesh(tube, color=color, name=name, pickable=False, reset_camera=False)
            self._marker(cache, name + "_tip", sim.tracking_points(dev)[0], radius * 1.6, color, 1.0)
        self._marker(cache, "target", env.target_pos, env.cfg.success_threshold_mm / 2, TARGET_COLOR, 0.8)
        self._marker(cache, "start", env.anat.nodes[env.start_node], 3.0, START_COLOR, 1.0)
        if show_hud:
            self._text(cache, "hud", self.hud_text(status), "upper_left", 9, "black")
        if keys_help:
            self._text(cache, "keys", keys_help, "lower_left", 8, "#374151")

    def _marker(self, cache: dict, name: str, center, radius: float, color: str, opacity: float) -> None:
        actor = cache.get(name)
        if actor is None:
            actor = self.pl.add_mesh(self.pv.Sphere(radius=radius, center=(0.0, 0.0, 0.0)), color=color,
                                     opacity=opacity, name=name, pickable=False, reset_camera=False)
            cache[name] = actor
        actor.SetPosition(*[float(c) for c in center])

    def _text(self, cache: dict, name: str, text: str, position: str, size: int, color: str) -> None:
        key = name + "_text"
        if cache.get(key) == text:
            return
        actor = cache.get(name)
        corners = {"lower_left": 0, "lower_right": 1, "upper_left": 2, "upper_right": 3}
        if actor is not None and hasattr(actor, "SetText") and position in corners:
            actor.SetText(corners[position], text)  # vtkCornerAnnotation
        elif actor is not None and hasattr(actor, "SetInput"):
            actor.SetInput(text)
        else:
            cache[name] = self.pl.add_text(text, position=position, font_size=size, name=name, color=color)
        cache[key] = text

    def hud_text(self, status: str) -> str:
        env, sim = self.env, self.env.sim
        a = env.anat
        lines = [
            f"{a.name}",
            f"target: {a.branch_of(env.target_node)} (node {env.target_node})   start: {a.branch_of(env.start_node)}",
            f"step {env.steps}   time {env.steps * env.cfg.sim.dt:5.1f} s   "
            f"distance to target {env.D:6.1f} mm (geodesic, {env.cfg.success_device} tip)",
            f"guidewire     L {sim.L[WIRE]:6.1f} mm  roll {np.degrees(sim.phi[WIRE]):5.0f} deg  "
            f"in {a.branch_of(sim.node_at(sim.L[WIRE]))}",
            f"microcatheter L {sim.L[CATH]:6.1f} mm  roll {np.degrees(sim.phi[CATH]):5.0f} deg  "
            f"in {a.branch_of(sim.node_at(sim.L[CATH]))}",
            f"wrong-branch entries {env._ep['wrong_branch_events']}   last action "
            + "[" + ", ".join(f"{x:+.1f}" for x in env.prev_action) + "]",
        ]
        if status:
            lines.append(status)
        return "\n".join(lines)

    def set_front_view(self, zoom: float | None = None) -> None:
        """Head-up view along the anatomy's thinnest horizontal axis (VMR models use RAS axes).

        For full aorta-to-brain models this is the anterior (AP) view with the patient's left on the
        screen's right. For an isolated aortic arch, which lies in the front-to-back plane, it is a
        lateral view so the arch and its branches do not overlap.
        """
        a = self.env.anat
        c = a.center
        d = 3.0 * a.half_extent
        ext = a.bbox_max - a.bbox_min
        if ext[1] <= ext[0]:
            pos, default_zoom = (c[0], c[1] + d, c[2]), 1.35  # anterior view (camera in front of the patient)
        else:
            pos, default_zoom = (c[0] - d, c[1], c[2]), 0.9  # lateral view from the patient's left
        self.pl.camera_position = [pos, tuple(c), (0, 0, 1)]
        self.pl.reset_camera(bounds=(*_interleave(a.bbox_min, a.bbox_max),))
        self.pl.camera.zoom(zoom if zoom is not None else default_zoom)


def _interleave(lo: np.ndarray, hi: np.ndarray) -> list[float]:
    return [float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), float(lo[2]), float(hi[2])]
