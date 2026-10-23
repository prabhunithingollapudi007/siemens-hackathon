"""Wall guard and the wall safety layer, on a synthetic tube and a stub guard (no organiser pages)."""

import numpy as np
import pytest

from stroke_nav.safety.wall import GW_COS, GW_INSERTION, WallCheck, WallGuard, WallSafeController, catmull_rom


@pytest.fixture(scope="module")
def tube_guard():
    pv = pytest.importorskip("pyvista")
    tube = pv.Cylinder(center=(0, 0, 100), direction=(0, 0, 1), radius=5.0, height=200.0, resolution=48, capping=True)
    tube = tube.triangulate().subdivide(2).clean()
    faces = tube.faces.reshape(-1, 4)[:, 1:]
    return WallGuard(np.asarray(tube.points), faces, inside_point_mm=np.array([0.0, 0.0, 100.0]))


def wire(x_of_z, z_tip=120.0, spacing=7.5):
    z = np.arange(-60.0, z_tip + 1e-9, spacing)
    return np.stack([x_of_z(z), np.zeros_like(z), z], axis=1) / 1000.0  # world m, proximal -> tip


def test_catmull_rom_passes_through_the_bodies():
    p = np.array([[0, 0, 0], [0, 0, 7.5], [1, 0, 15], [3, 0, 22.5]], float)
    c = catmull_rom(p, samples_per_mm=2.0)
    for q in p:
        assert np.min(np.linalg.norm(c - q, axis=1)) < 1e-9


def test_wire_inside_the_tube_is_fine(tube_guard):
    c = tube_guard.check(wire(lambda z: np.zeros_like(z)), inserted_mm=120.0)
    assert not c.crossing and c.depth_mm == 0.0 and c.inserted_bodies > 10


def test_wire_bowing_through_the_wall_is_a_crossing(tube_guard):
    bow = lambda z: np.where(z > 0, 8.0 * np.sin(np.clip(z, 0, 120) / 120.0 * np.pi), 0.0)  # 3 mm beyond a 5 mm wall
    c = tube_guard.check(wire(bow), inserted_mm=120.0)
    assert c.crossing and c.kind == "deep" and 2.0 < c.depth_mm < 4.0
    # the part not yet inserted (z < 0, outside the patient) is never checked
    assert not tube_guard.check(wire(lambda z: np.where(z < 0, 50.0, 0.0)), inserted_mm=120.0).crossing


class StubGuard:
    def __init__(self):
        self.crossing = False

    def check(self, bodies, inserted_mm):
        return WallCheck(self.crossing, 2.0 if self.crossing else 0.0, 0.0, None, 10, "deep" if self.crossing else "")


class Push:
    def reset(self):
        pass

    def act(self, obs):
        return np.array([1.0, 0.0, 0.0, 0.0], np.float32)


def obs(ins_mm, rot_rad=0.0):
    s = np.zeros(43, np.float32)
    s[GW_INSERTION], s[GW_COS - 1], s[GW_COS] = ins_mm / 1000.0, np.sin(rot_rad), np.cos(rot_rad)
    return {"state": s}


def test_safety_layer_backs_off_rerolls_and_aborts():
    guard = StubGuard()
    ctl = WallSafeController(Push(), guard, retract_mm=10.0, reroll_deg=(20.0,))
    ctl.reset()
    assert ctl.act(obs(100))[0] == 1.0  # inside: the policy's push unchanged
    guard.crossing = True
    a = ctl.act(obs(102))
    assert a[0] < 0 and ctl.mode == "retract"  # crossing: back off at once
    guard.crossing = False
    assert ctl.act(obs(95))[0] < 0  # still within retract_mm of the crossing
    a = ctl.act(obs(91))  # clear and backed off: turn by the first offset
    assert ctl.mode == "reroll" and a[0] == 0 and a[1] > 0
    a = ctl.act(obs(91, np.radians(20)))  # turned: creep forward, slower than the policy
    assert ctl.mode == "advance" and 0 < a[0] <= ctl.creep + 1e-6
    guard.crossing = True
    ctl.act(obs(95, np.radians(20)))  # crossing again: retract; no offsets left -> abort once clear
    guard.crossing = False
    ctl.act(obs(80, np.radians(20)))
    assert ctl.mode == "abort" and not np.any(ctl.act(obs(80, np.radians(20))))
    assert [e["event"] for e in ctl.events] == ["wall_crossing", "reroll", "wall_crossing", "abort"]


def test_safety_layer_never_pushes_harder_than_the_policy():
    guard, rng = StubGuard(), np.random.default_rng(0)

    class Random(Push):
        def act(self, obs):
            self.last = rng.uniform(-1, 1, 4).astype(np.float32)
            return self.last

    pol = Random()
    ctl = WallSafeController(pol, guard)
    ctl.reset()
    ins = 50.0
    for k in range(300):
        guard.crossing = bool(rng.random() < 0.1)
        a = ctl.act(obs(ins, 0.0))
        assert a[0] <= max(pol.last[0], 0.0) + 1e-6 or a[0] < 0  # only the policy's push or less, or a back-off
        ins = max(0.0, ins + 2.0 * float(a[0]))
