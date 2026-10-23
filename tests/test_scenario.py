"""Organiser scenarios for other meshes: written in their mesh frame, validated with their own loader."""

from pathlib import Path

import numpy as np
import pytest

from stroke_nav.organiser.scenario import scenario_name, variant_surface, write_scenario

RAW = Path("data/raw/0248_H_AOCERE_CAS/Models")
ORGANISER = Path(__file__).resolve().parents[2] / "HCMR-EV" / "HCMR-EV-RL" / "src" / "openng3_rl" / "scenario.py"

needs_data = pytest.mark.skipif(not (RAW / "10Right70Left.vtp").is_file() or not ORGANISER.is_file(),
                                reason="needs the 0248 10Right70Left surface and the organiser sources")


def test_scenario_name_is_valid_for_their_loader():
    assert scenario_name("0248_H_AOCERE_CAS", "10Right70Left") == "0248_H_AOCERE_CAS-10Right70Left"
    assert scenario_name("0248_H_AOCERE_CAS", None) == "0248_H_AOCERE_CAS"


@needs_data
def test_variant_scenario_loads_with_their_loader_and_keeps_the_world_frame(tmp_path):
    pytest.importorskip("pyvista")
    from stroke_nav.data.anatomy import Anatomy
    from stroke_nav.organiser.frame import default_frame, read_obj_vertices
    from stroke_nav.organiser.scenario import validate_scenario

    assert variant_surface("0248_H_AOCERE_CAS", "10Right70Left").name == "10Right70Left.vtp"
    anat, frame = Anatomy.load("0248_H_AOCERE_CAS", "10Right70Left"), default_frame()
    goal = anat.nodes[anat.resolve("left_carotid@100")]
    out = write_scenario("0248_H_AOCERE_CAS", "10Right70Left", [goal], tmp_path, frame, target_triangles=4000)
    summary = validate_scenario(out.name, tmp_path)
    transform = np.asarray(summary["mesh_to_world"]).reshape(4, 4)
    assert np.allclose(transform, np.diag([0.01, 0.01, 0.01, 1.0]))  # pure scale: same world frame as cas-reference
    verts = read_obj_vertices(out / "vessel.obj")
    assert 1000 < len(verts) < 4000
    theirs = Path(ORGANISER).parents[2] / "scenarios" / "cas-reference" / "vessel.obj"
    if theirs.is_file():  # the variant mesh is the same patient: bounds agree to a few mm
        ref = read_obj_vertices(theirs)
        assert np.all(np.abs(verts.min(0) - ref.min(0)) < 0.5) and np.all(np.abs(verts.max(0) - ref.max(0)) < 0.5)
    goal_cm = np.loadtxt(out / "goals.txt", delimiter=",", comments="#").reshape(-1, 3)
    assert np.allclose(frame.obj_cm_to_vmr_mm(goal_cm[0]), goal, atol=0.05)  # 3 decimals in cm = 0.01 mm


def _tube(x0_cm: float, z0_cm: float, z1_cm: float, r_cm: float, n: int = 12, rings: int = 6):
    """Open cylinder along +Z at (x0, 0): vertices (cm) and triangles."""
    zs, a = np.linspace(z0_cm, z1_cm, rings), np.linspace(0, 2 * np.pi, n, endpoint=False)
    v = np.array([[x0_cm + r_cm * np.cos(t), r_cm * np.sin(t), z] for z in zs for t in a])
    tri = [[i * n + j, i * n + (j + 1) % n, (i + 1) * n + j] for i in range(rings - 1) for j in range(n)]
    return v, np.asarray(tri)


def test_entrance_band_holds_only_the_entry_vessel():
    from stroke_nav.organiser.scenario import clear_entrance_band, page_lumen_radius_cm

    entry_v, entry_t = _tube(0.0, 0.0, 50.0, 0.25)  # right femoral at the origin, up to the aorta
    other_v, other_t = _tube(-9.0, 0.3, 20.0, 0.25)  # other femoral ending at the same height
    verts = np.vstack([entry_v, other_v])
    tri = np.vstack([entry_t, other_t + len(entry_v)])
    assert page_lumen_radius_cm(verts) > 4.0  # their page: devices drawn > 8 mm thick
    v2, t2, removed = clear_entrance_band(verts, tri, keep_radius_cm=1.0)
    assert removed > 0 and page_lumen_radius_cm(v2) == pytest.approx(0.25, abs=0.01)
    assert v2[:, 2].max() == pytest.approx(50.0) and (v2[:, 0] < -8).any()  # the other vessel stays above the band
    assert clear_entrance_band(entry_v, entry_t, 1.0)[2] == 0  # nothing else in the band: unchanged
