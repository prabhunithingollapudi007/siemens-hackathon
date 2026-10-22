"""Host-side pieces of the live-stack tooling (no Docker needed) and the closed-loop A* follower."""

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

from stroke_nav.agents.route_follower import ClosedLoopRouteFollower, FollowerParams, TipObs
from stroke_nav.env.devices import CATH, WIRE
from stroke_nav.organiser.env import CompatConfig, OrganiserCompatEnv
from stroke_nav.organiser.frame import OrganiserFrame
from stroke_nav.organiser.live.probe import read_recording, sample_at_steps
from stroke_nav.planning.planner import PathPlanner

ROOT = Path(__file__).resolve().parents[1]


def test_sample_at_steps_takes_the_end_of_each_control_period():
    t = np.arange(0.0, 1.0, 0.01)
    rec = {"t_s": t, "x": t * 10}
    s = sample_at_steps(rec, 4, dt=0.2)
    assert np.allclose(s["t_s"], [0.2, 0.4, 0.6, 0.8], atol=0.011)


def test_read_recording_keeps_missing_values_as_nan(tmp_path):
    p = tmp_path / "rec.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "gw_insertion_m"])
        w.writerows([[0.0, ""], [0.1, 0.002]])
    r = read_recording(p)
    assert np.isnan(r["gw_insertion_m"][0]) and r["gw_insertion_m"][1] == pytest.approx(0.002)


def test_gap_sequences_are_valid_actions():
    sys.path.insert(0, str(ROOT / "scripts"))
    import measure_gap

    seqs = measure_gap.sweep_sequences()
    assert sum(k.startswith("roll_") for k in seqs) == 12
    for _, (_, a) in seqs.items():
        assert a.ndim == 2 and a.shape[1] == 4 and np.all(np.abs(a) <= 1.0)
    # the 90 deg roll sequence rotates by ~90 deg before pushing
    a = seqs["roll_090"][1]
    assert np.degrees(a[:, 1].sum() * 0.2) == pytest.approx(90, abs=6)


@pytest.mark.parametrize("target", ["left_branch@-3", "right_branch@-3", "back_branch@-3"])
def test_route_follower_reaches_side_branches_in_our_sim(toy_anatomy, target):
    anat = toy_anatomy
    node = anat.resolve(target)
    env = OrganiserCompatEnv(CompatConfig(goal_mode="target", target=node, image_mode="blank", noise=False),
                             anatomy=anat, frame=OrganiserFrame())
    env.reset(seed=1)
    plan = PathPlanner(anat, "radifocus_035").plan(anat.default_start(), node)
    fol = ClosedLoopRouteFollower(anat, plan, FollowerParams(cath_limit_mm=plan.s[-1] - 30.0))
    sim, goal = env.inner.sim, anat.nodes[node]
    for _ in range(400):
        obs = TipObs(sim.tracking_points(WIRE)[0], sim.tip_direction(WIRE), float(sim.L[WIRE]), float(sim.L[CATH]), goal)
        a = fol.act(obs)
        if fol.state.mode == "done":
            break
        env.step(a)
    assert fol.state.mode == "done", fol.state.log
    assert anat.branch_of(anat.nearest_node(sim.tracking_points(WIRE)[0])) == target.split("@")[0]
