import numpy as np
import pytest

from stroke_nav.data.centerline import infer_unit_scale, parse_pth_text, resample_polyline
from stroke_nav.data.fetch import model_url, relative_entry

PTH = """<?xml version="1.0" encoding="UTF-8" ?>
<format version="1.0" />
<path id="1" method="0" calculation_number="100" spacing="0" version="1.0">
  <timestep id="0">
    <path_element method="0" calculation_number="100" spacing="0">
      <control_points><point id="0" x="9" y="9" z="9" /></control_points>
      <path_points>
        <path_point id="0"><pos x="0.0" y="0.0" z="0.0" /><tangent x="0" y="0" z="1" /></path_point>
        <path_point id="1"><pos x="0.0" y="0.0" z="1.5" /><tangent x="0" y="0" z="1" /></path_point>
        <path_point id="2"><pos x="0.0" y="0.0" z="3.0" /><tangent x="0" y="0" z="1" /></path_point>
      </path_points>
    </path_element>
  </timestep>
</path>
"""


def test_parse_pth_uses_path_points():
    pts = parse_pth_text(PTH)
    assert pts.shape == (3, 3)
    np.testing.assert_allclose(pts[:, 2], [0.0, 1.5, 3.0])


def test_unit_scale_cm_vs_mm():
    cm = np.array([[0, 0, 0], [20, 10, 33]], dtype=float)  # ~33 cm extent
    mm = cm * 10
    assert infer_unit_scale(cm) == 10.0
    assert infer_unit_scale(mm) == 1.0


def test_resample_spacing():
    p = np.array([[0, 0, 0], [0, 0, 10.0]])
    r = resample_polyline(p, 1.0)
    assert len(r) == 11
    np.testing.assert_allclose(np.diff(r[:, 2]), 1.0)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("0050_H_CERE_H/Paths/LVA.pth", "Paths/LVA.pth"),
        ("0050_H_CERE_H/Models/0078_0001.vtp", "Models/0078_0001.vtp"),
        ("0050_H_CERE_H/LICENSE", "LICENSE"),
        ("0050_H_CERE_H/Meshes/0078_0001.vtu", None),
        ("0050_H_CERE_H/Paths/", None),
    ],
)
def test_relative_entry(name, expected):
    assert relative_entry(name) == expected


def test_model_urls():
    assert model_url("0050_H_CERE_H").endswith("svprojects/0050_H_CERE_H.zip")
    assert model_url("0248_H_AOCERE_CAS").endswith("_reference.zip")
    assert model_url("0248_H_AOCERE_CAS", "10Left100Right").endswith("_all.zip")


def test_toy_graph_is_connected(toy_anatomy):
    a = toy_anatomy
    d, _ = a.distances_from(a.default_start())
    assert np.isfinite(d).all()
    assert len(a.junctions()) >= 3
    assert a.branch_of(a.default_start()) == "aorta"


def test_target_specs(toy_anatomy):
    a = toy_anatomy
    end = a.resolve("left_branch@-1")
    assert a.branch_of(end) == "left_branch" and a.degree[end] == 1
    n = a.resolve("left_branch@10")
    assert abs(a.arc[n] - a.arc[a.branch_nodes("left_branch")].min() - 10) <= 1.0
    rng = np.random.default_rng(0)
    pool = a.sample_pool("random:left_branch@-10--1|right_branch@5-10")
    assert {a.branch_of(int(x)) for x in pool} == {"left_branch", "right_branch"}
    assert a.branch_of(a.resolve("random:right_branch@5-10", rng)) == "right_branch"
    assert a.resolve("xyz:0,0,0") == a.default_start()
    occ = a.resolve("occlusion")
    assert a.radius[occ] >= 1.0

