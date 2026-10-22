"""Small synthetic arch-to-M1 tree with realistic lumen radii, for running the safety evaluation offline
(no VMR download) and in tests. Not patient anatomy."""

from __future__ import annotations

import numpy as np

from stroke_nav.data.anatomy import Anatomy
from stroke_nav.data.centerline import build_graph


def _poly(*pts, n_per=40):
    pts = [np.asarray(p, float) for p in pts]
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        t = np.linspace(0, 1, n_per)[1:, None]
        out.extend((1 - t) * a + t * b)
    return np.array(out)


def demo_anatomy() -> Anatomy:
    th = np.linspace(0, np.pi, 60)
    arch = np.column_stack([30 * np.cos(th), np.zeros_like(th), 150 + 30 * np.sin(th)])
    paths = {
        "aorta": np.vstack([_poly([30, 0, 0], [30, 0, 150])[:-1], arch, _poly([-30, 0, 150], [-30, 0, 110])[1:]]),
        "left_subclavian": _poly(arch[18], [45, 0, 205], [80, 0, 215]),
        "left_carotid": _poly(arch[26], [10, 0, 260], [15, 0, 320], [5, 0, 350], [-5, 0, 365]),
        "left_carotid_external": _poly([15, 0, 320], [35, 0, 345]),
        "brachiocephalic": _poly(arch[38], [-20, 0, 205]),
        "right_subclavian": _poly([-20, 0, 205], [-60, 0, 215]),
        "right_carotid": _poly([-20, 0, 205], [-25, 0, 260], [-25, 0, 320], [-15, 0, 350], [-5, 0, 368]),
        "right_carotid_external": _poly([-25, 0, 320], [-45, 0, 345]),
    }
    g = build_graph(paths, root="aorta", spacing=1.0)
    nodes = g.nodes
    r = np.full(len(nodes), 3.5)
    z = nodes[:, 2]
    r[z < 190] = 11.0  # aorta and arch
    r[(z > 300)] = 2.5  # ICA / ECA
    r[(z > 345)] = 1.4  # M1-like distal end
    g.radius[:] = r
    g.meta.update({"model": "demo_arch"})
    return Anatomy(g)
