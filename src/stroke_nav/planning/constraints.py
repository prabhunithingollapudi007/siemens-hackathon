"""Device physical limits and vessel geometry measures used by the planner.

The robot's guidewire and catheter (``radifocus_035``, ``judkins_5f``) take their diameters from
device-parameters.json. The other entries are representative catalogue-level values, not a
specific product's IFU:
    guidewire 0.014"      OD 0.36 mm (e.g. Synchro-14 class), very flexible soft tip
    microcatheter 1.7F    distal OD 0.57 mm (e.g. Echelon-10 class)
    guide catheter 6F     OD 2.0 mm, stiff; only for the extracranial route
The minimum bend radius is the tightest radius of curvature the device can follow without
kinking or losing pushability; values are assumptions to be tuned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stroke_nav.data.anatomy import Anatomy


@dataclass(frozen=True)
class DeviceSpec:
    name: str
    outer_diameter_mm: float
    min_bend_radius_mm: float
    clearance_factor: float = 1.2  # hard limit: lumen diameter >= clearance_factor * OD
    comfort_factor: float = 3.0  # lumen diameter below comfort_factor * OD is penalised

    @property
    def min_lumen_radius_mm(self) -> float:
        """Smallest lumen radius the device can pass (hard constraint)."""
        return 0.5 * self.clearance_factor * self.outer_diameter_mm

    @property
    def comfort_lumen_radius_mm(self) -> float:
        return 0.5 * self.comfort_factor * self.outer_diameter_mm

    @property
    def max_curvature_per_mm(self) -> float:
        return 1.0 / self.min_bend_radius_mm


DEVICES: dict[str, DeviceSpec] = {
    "guidewire": DeviceSpec("guidewire 0.014in", outer_diameter_mm=0.36, min_bend_radius_mm=1.0),
    "microcatheter": DeviceSpec("microcatheter 1.7F", outer_diameter_mm=0.57, min_bend_radius_mm=2.0),
    "guide6f": DeviceSpec("guide catheter 6F", outer_diameter_mm=2.0, min_bend_radius_mm=10.0),
}

# The two devices the robot drives. Their geometry comes from device-parameters.json (the source of
# truth, via organiser.spec.DEFAULT_SPEC); the bend limits are not in that file and stay assumptions.
GUIDEWIRE = "radifocus_035"
CATHETER = "judkins_5f"
MIN_BEND_RADIUS_MM = {GUIDEWIRE: 3.0, CATHETER: 8.0}


def robot_devices(spec=None) -> dict[str, DeviceSpec]:
    """Planner specs of the robot's guidewire and catheter, from the device parameters."""
    from stroke_nav.organiser.spec import DEFAULT_SPEC

    spec = spec or DEFAULT_SPEC
    gw, ca = spec.guidewire_geometry(), spec.catheter_geometry()
    return {
        GUIDEWIRE: DeviceSpec(f"guidewire {gw['outer_diameter_mm']} mm ({gw['preset']})", outer_diameter_mm=gw["outer_diameter_mm"],
                              min_bend_radius_mm=MIN_BEND_RADIUS_MM[GUIDEWIRE]),
        CATHETER: DeviceSpec(f"catheter {ca['outer_diameter_mm']} mm ({ca['preset']})", outer_diameter_mm=ca["outer_diameter_mm"],
                             min_bend_radius_mm=MIN_BEND_RADIUS_MM[CATHETER]),
    }


DEVICES.update(robot_devices())


def get_device(name: str | DeviceSpec) -> DeviceSpec:
    if isinstance(name, DeviceSpec):
        return name
    if name not in DEVICES:
        raise KeyError(f"unknown device {name!r}; choose from {sorted(DEVICES)}")
    return DEVICES[name]


def menger_curvature(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Curvature (1/mm) of the circle through three points; 0 for collinear points."""
    ab, bc, ca = np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c)
    denom = ab * bc * ca
    if denom < 1e-12:
        return 0.0
    area2 = np.linalg.norm(np.cross(b - a, c - a))  # twice the triangle area
    return float(2.0 * area2 / denom)


class VesselGeometry:
    """Curvature measures on an anatomy's centerline graph (cached).

    ``turn_curvature(u, v, w)`` is the curvature a device experiences at node v when it arrives
    from u and leaves toward w, measured with points ``window`` nodes behind and ahead. Along a
    branch it is the vessel's own curvature; at a bifurcation it captures the take-off angle.
    """

    def __init__(self, anatomy: Anatomy, window: int = 3):
        self.anat = anatomy
        self.window = window
        self._cache: dict[tuple[int, int, int], float] = {}

    def turn_curvature(self, u: int, v: int, w: int) -> float:
        key = (u, v, w)
        k = self._cache.get(key)
        if k is None:
            a = self.anat
            back = a.walk_from(v, u, self.window)
            ahead = a.walk_from(v, w, self.window)
            k = menger_curvature(a.nodes[back], a.nodes[v], a.nodes[ahead])
            self._cache[key] = k
        return k

    def turn_angle_deg(self, u: int, v: int, w: int) -> float:
        a = self.anat
        d_in = a.nodes[v] - a.nodes[a.walk_from(v, u, self.window)]
        d_out = a.nodes[a.walk_from(v, w, self.window)] - a.nodes[v]
        cos = np.dot(d_in, d_out) / max(np.linalg.norm(d_in) * np.linalg.norm(d_out), 1e-12)
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    def path_curvature(self, path: list[int]) -> np.ndarray:
        """Curvature at every node of a node path (0 at the two ends)."""
        k = np.zeros(len(path))
        for i in range(1, len(path) - 1):
            k[i] = self.turn_curvature(path[i - 1], path[i], path[i + 1])
        return k
