"""Entry-roll table and per-episode goals for the lockstep env (synthetic paths, no browser)."""

import json

import numpy as np
import pytest

from stroke_nav.agents.roll_table import RollPath, RollTable, load_goal_set, path_distance_m
from stroke_nav.organiser.live.lockstep import GoalSet


def _path(roll, seed, direction, offset=0.0):
    s = np.linspace(0.0, 0.2, 101)[:, None]  # 200 mm in 2 mm steps
    tip = s * np.asarray(direction, float) + np.array([offset, 0.0, 0.0])
    return RollPath(roll, seed, tip, np.linspace(0.0, 200.0, 101))


@pytest.fixture
def table():
    # roll 0 goes along +z, roll 90 along +y; roll 90's two seeds differ by 2 mm sideways
    return RollTable([_path(0.0, 1, (0, 0, 1)), _path(0.0, 2, (0, 0, 1)),
                      _path(90.0, 1, (0, 1, 0)), _path(90.0, 2, (0, 1, 0), offset=0.002)])


def test_path_distance_is_to_segments():
    path = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]])
    assert path_distance_m(path, np.array([0.003, 0.0, 0.05])) == pytest.approx(0.003)
    assert path_distance_m(path, np.array([0.0, 0.0, 0.104])) == pytest.approx(0.004)


def test_reach_uses_the_worst_seed(table):
    roll, miss = table.reach([0.0, 0.0, 0.1])
    assert roll == 0.0 and miss == pytest.approx(0.0, abs=1e-9)
    roll, miss = table.reach([0.0, 0.1, 0.0])
    assert roll == 90.0 and miss == pytest.approx(2.0, abs=1e-6)  # seed 2 passes 2 mm away
    assert table.roll_for([0.0, 0.1, 0.0], reach_mm=1.5) is None
    assert table.roll_for([0.05, 0.05, 0.05]) is None  # between the branches


def test_candidate_goals_are_reachable_and_spaced(table):
    goals = table.candidate_goals(spacing_mm=10.0, reach_mm=2.5, min_ins_mm=40.0)
    points = np.array([g["goal_m"] for g in goals])
    assert len(goals) > 20 and {g["roll_deg"] for g in goals} == {0.0, 90.0}
    assert all(g["ins_mm"] >= 40.0 and g["miss_mm"] <= 2.5 for g in goals)
    d = np.linalg.norm(points[:, None] - points[None], axis=2) + np.eye(len(points))
    assert d.min() * 1000.0 >= 10.0 - 1e-6


def test_goal_set_next_then_uniform(tmp_path):
    gs = GoalSet([[0.0, 0.0, 0.1], [0.0, 0.1, 0.0]])
    rng = np.random.default_rng(0)
    gs.set_next([1.0, 2.0, 3.0])
    assert gs.sample(rng).tolist() == [1.0, 2.0, 3.0]
    drawn = {tuple(gs.sample(rng)) for _ in range(50)}
    assert drawn == {(0.0, 0.0, 0.1), (0.0, 0.1, 0.0)}
    f = tmp_path / "goals.json"
    f.write_text(json.dumps({"goals": [{"id": 0, "goal_m": [0, 0, 1], "roll_deg": 0.0, "branch": "a", "split": "train"},
                                       {"id": 1, "goal_m": [0, 1, 0], "roll_deg": 90.0, "branch": "b", "split": "test"}]}))
    assert [g["id"] for g in load_goal_set(f, "test")] == [1]
    assert len(load_goal_set(f, "train,test")) == 2
    with pytest.raises(ValueError):
        load_goal_set(f, "cas")


def test_signed_rolls_prefer_the_smallest_turn(tmp_path):
    exact, near = _path(345.0, 1, (0, 0, 1)), _path(-15.0, 1, (0, 0, 1), offset=0.001)
    t = RollTable([exact, near])
    assert t.reach([0.0, 0.0, 0.1])[0] == 345.0  # closest
    roll, miss = t.reach([0.0, 0.0, 0.1], prefer_small_within_mm=2.5)
    assert roll == -15.0 and miss == pytest.approx(1.0, abs=1e-6)
    f = tmp_path / "sweep.json"
    f.write_text(json.dumps({"paths": [{"roll_deg": p.roll_deg, "seed": p.seed, "tip_m": p.tip_m.tolist(),
                                        "ins_mm": p.ins_mm.tolist()} for p in (exact, near)]}))
    assert RollTable.load(f"{f},{f}", roll_range=(-180.0, 180.0)).rolls == [-15.0]


def _straight(roll: float, offset_m: float) -> RollPath:
    """A path along +z that passes ``offset_m`` beside the point (0, 0, 0.1)."""
    z = np.linspace(0.0, 0.2, 41)
    return RollPath(roll, 1, np.c_[np.full_like(z, offset_m), np.zeros_like(z), z], z * 1000.0)


def test_robust_roll_is_the_centre_of_the_reaching_arc():
    good = {90.0, 105.0, 120.0, 135.0, 150.0}
    # 150 passes closest, but sits at the edge of the arc
    t = RollTable([_straight(r, 0.0005 if r == 150.0 else 0.001 if r in good else 0.05) for r in np.arange(0.0, 360.0, 15.0)])
    assert t.reach([0.0, 0.0, 0.1])[0] == 150.0
    roll, miss = t.reach([0.0, 0.0, 0.1], robust_within_mm=5.0)
    assert roll == 120.0 and miss == pytest.approx(1.0, abs=1e-6)
    far = t.reach([0.0, 0.0, 0.1], robust_within_mm=0.1)  # nothing within: falls back to the closest
    assert far[0] == 150.0


def test_robust_roll_wraps_around_the_circle():
    good = {330.0, 345.0, 0.0, 15.0}
    t = RollTable([_straight(r, 0.001 if r in good else 0.05) for r in np.arange(0.0, 360.0, 15.0)])
    assert t.reach([0.0, 0.0, 0.1], robust_within_mm=5.0)[0] == 0.0
