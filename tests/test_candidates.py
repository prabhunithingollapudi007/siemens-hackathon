"""Candidate trajectories: common format, safety assessment and the case reports (synthetic tree, no downloads)."""

import json

import numpy as np
import pytest

from stroke_nav.planning.candidates import (Candidate, EpisodeTrace, assess, assess_episode, plot_candidates,
                                            summary_table, write_trajectory_csv)
from stroke_nav.planning.planner import PathPlanner


class MetresFrame:
    """Stand-in for OrganiserFrame: world metres are VMR mm / 1000, same axes."""

    M = np.eye(3)

    def world_m_to_vmr_mm(self, p):
        return np.asarray(p, float) * 1000.0

    def vmr_mm_to_world_m(self, p):
        return np.asarray(p, float) / 1000.0


FRAME = MetresFrame()


def _route(anat):
    start = anat.default_start()
    leaf = max(anat.leaves(), key=lambda n: anat.distances_from(start)[0][n])
    plan = PathPlanner(anat, "guidewire").plan(start, leaf)
    assert plan.feasible
    return plan


def _trace(points_mm, goal_mm, insertion=None, actions=None, success=True, force=0.0, seed=0):
    points_mm = np.asarray(points_mm, float)
    n = len(points_mm)
    dist = np.linalg.norm(points_mm - np.asarray(goal_mm, float), axis=1)
    ins = np.asarray(insertion, float) if insertion is not None else np.cumsum(np.r_[0.0, np.linalg.norm(np.diff(points_mm, axis=0), axis=1)])
    acts = np.asarray(actions, float) if actions is not None else np.zeros((n, 4))
    return EpisodeTrace(seed=seed, success=success, tip_m=FRAME.vmr_mm_to_world_m(points_mm), insertion_mm=ins,
                        actions=acts, dist_mm=dist, peak_force_n=force, reason="success" if success else "step_limit")


def test_on_route_trace_passes(toy_anatomy):
    plan = _route(toy_anatomy)
    goal = plan.points[-1]
    cand = Candidate("astar", "ours-kinematic", "test", episodes=[_trace(plan.points, goal)])
    report = assess(cand, toy_anatomy, FRAME, plan.points, set(plan.nodes), travel_mm=1000.0)
    assert report["verdict"] == "pass", report
    assert report["flags"] == []
    assert report["success"] == "1/1"


def test_wrong_vessel_and_lumen_flags(toy_anatomy):
    plan = _route(toy_anatomy)
    goal = plan.points[-1]
    on_route = set(plan.nodes)
    other = [n for n in range(len(toy_anatomy.nodes)) if n not in on_route and toy_anatomy.branch_of(n) != toy_anatomy.branch_of(plan.nodes[0])]
    other_pts = toy_anatomy.nodes[sorted(other, key=lambda n: toy_anatomy.arc[n])[-25:]]  # the far end of a side branch
    rep = assess_episode(_trace(np.vstack([plan.points[:10], other_pts]), goal, success=False), toy_anatomy, FRAME,
                         plan.points, on_route, travel_mm=1000.0)
    assert "wrong_vessel" in rep["flags"] and rep["off_route_mm"] > 4.0 and rep["wrong_vessel_depth_mm"] > 12.0

    one_sample = np.vstack([plan.points[:10], other_pts[-1:], plan.points[10:]])  # a single junction artefact
    rep = assess_episode(_trace(one_sample, goal), toy_anatomy, FRAME, plan.points, on_route, travel_mm=1000.0)
    assert "wrong_vessel" not in rep["flags"] and rep["off_route_steps"] == 0

    off = plan.points + np.array([25.0, 0.0, 0.0])  # 25 mm beside every vessel
    rep = assess_episode(_trace(off, goal), toy_anatomy, FRAME, plan.points, on_route, travel_mm=1000.0)
    assert "lumen" in rep["flags"] and rep["off_lumen_mm"] > 10.0


def test_overshoot_force_speed_and_travel_flags(toy_anatomy):
    plan = _route(toy_anatomy)
    goal = plan.points[-1]
    pts = plan.points
    n = len(pts)
    # the tip reaches the goal, then the insertion grows by 30 mm while the tip stays there (the rod buckles)
    ins = np.r_[np.linspace(0.0, 100.0, n), 115.0, 130.0]
    rep = assess_episode(_trace(np.vstack([pts, pts[-1:], pts[-1:]]), goal, insertion=ins), toy_anatomy, FRAME,
                         pts, set(plan.nodes), travel_mm=1000.0)
    assert "overshoot" in rep["flags"] and rep["overshoot_mm"] == pytest.approx(30.0), rep

    rep = assess_episode(_trace(pts, goal, force=1.5), toy_anatomy, FRAME, pts, set(plan.nodes), travel_mm=1000.0)
    assert "force" in rep["flags"]

    full = np.zeros((n, 4))
    full[:, 0] = 1.0
    rep = assess_episode(_trace(pts, goal, actions=full), toy_anatomy, FRAME, pts, set(plan.nodes), travel_mm=1000.0)
    small = np.any(toy_anatomy.radius[[toy_anatomy.nearest_node(p) for p in pts]] < 5.0)
    assert ("speed" in rep["flags"]) == bool(small)

    rep = assess_episode(_trace(pts, goal), toy_anatomy, FRAME, pts, set(plan.nodes), travel_mm=10.0)
    assert "travel" in rep["flags"]


def test_verdicts_and_representative(toy_anatomy):
    plan = _route(toy_anatomy)
    goal = plan.points[-1]
    miss = _trace(plan.points[: len(plan.points) // 2], goal, success=False, seed=1)
    slow = _trace(plan.points, goal, seed=2)
    fast = _trace(plan.points[::2], goal, seed=3)
    cand = Candidate("roll", "theirs-lockstep", "test", episodes=[miss, slow, fast])
    assert cand.representative() is fast
    assert cand.success_rate == pytest.approx(2 / 3)
    assert assess(cand, toy_anatomy, FRAME, plan.points, set(plan.nodes), 1000.0)["verdict"] == "pass"
    only_miss = Candidate("bc", "theirs-lockstep", "test", episodes=[miss])
    assert only_miss.representative() is miss
    assert assess(only_miss, toy_anatomy, FRAME, plan.points, set(plan.nodes), 1000.0)["verdict"] == "reject"
    forced = Candidate("act", "theirs-lockstep", "test", episodes=[_trace(plan.points, goal, force=2.0)])
    assert assess(forced, toy_anatomy, FRAME, plan.points, set(plan.nodes), 1000.0)["verdict"] == "review"


def test_save_load_round_trip_and_reports(toy_anatomy, tmp_path):
    plan = _route(toy_anatomy)
    goal = plan.points[-1]
    cand = Candidate("astar", "ours-kinematic", "A* route", train_wall_s=0.0, plan_wall_s=2.5,
                     episodes=[_trace(plan.points, goal, seed=7)], artifacts={"plan": "plan.json"}, notes=["n"])
    f = cand.save(tmp_path / "astar.json")
    back = Candidate.load(f)
    assert back.method == "astar" and back.episodes[0].seed == 7 and back.episodes[0].steps == cand.episodes[0].steps
    assert np.allclose(back.episodes[0].tip_m, cand.episodes[0].tip_m, atol=1e-6)
    assert json.loads(f.read_text())["success_rate"] == 1.0

    report = assess(cand, toy_anatomy, FRAME, plan.points, set(plan.nodes), 1000.0)
    table = summary_table([cand], {"astar": report})
    assert "| astar |" in table and "pass" in table and "1/1" in table

    csv_file = write_trajectory_csv(tmp_path / "t.csv", cand.episodes[0], FRAME)
    assert len(csv_file.read_text().splitlines()) == cand.episodes[0].steps + 1

    pytest.importorskip("matplotlib")
    png = plot_candidates(tmp_path / "c.png", toy_anatomy, FRAME, [cand], goal, plan.points, title="toy")
    assert png.stat().st_size > 1000
