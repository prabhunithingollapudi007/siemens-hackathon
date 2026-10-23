import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from stroke_nav.agents.scripted import RandomAgent, ScriptedOracle, run_episode
from stroke_nav.env.devices import CATH, WIRE
from stroke_nav.env.vessel_env import EnvConfig, VesselNavEnv
from stroke_nav.eval.start_metrics import episode_start_metrics, markdown_table, summarize, wilson_ci

TARGETS = ["left_branch@-3", "right_branch@-3", "back_branch@-3", "aorta@-3"]


def make_env(anat, **kw):
    cfg = EnvConfig(target=kw.pop("target", "random:left_branch@20-40|right_branch@20-40"), max_steps=300, **kw)
    return VesselNavEnv(cfg, anatomy=anat)


def test_env_api(toy_anatomy):
    env = make_env(toy_anatomy)
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (46,) and env.action_space.shape == (4,)
    for _ in range(50):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(obs).all() and np.isfinite(r)
        if term or trunc:
            break


def test_determinism(toy_anatomy):
    def rollout(seed):
        env = make_env(toy_anatomy)
        env.reset(seed=seed)
        rng = np.random.default_rng(123)
        out = []
        for _ in range(60):
            obs, r, term, trunc, _ = env.step(rng.uniform(-1, 1, 4))
            out.append(np.concatenate([obs, [r]]))
            if term or trunc:
                break
        return np.array(out)

    np.testing.assert_array_equal(rollout(5), rollout(5))


@pytest.mark.parametrize("target", TARGETS)
def test_oracle_reaches_every_branch(toy_anatomy, target):
    env = make_env(toy_anatomy, target=target)
    results = [run_episode(env, ScriptedOracle(), seed=s) for s in range(5)]
    assert all(r["success"] for r in results), [r["final_distance_mm"] for r in results]


def test_rotation_is_needed(toy_anatomy):
    """Pushing without rotating must not reach every side branch, otherwise the task is trivial."""
    reached = []
    for target in TARGETS[:3]:
        env = make_env(toy_anatomy, target=target)
        env.reset(seed=0)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(np.array([1.0, 0.0, 1.0, 0.0]))
            done = term or trunc
        reached.append(info["episode_metrics"]["success"])
    assert not all(reached)


def test_catheter_follows_wire_track(toy_anatomy):
    env = make_env(toy_anatomy, target="left_branch@-3")
    env.reset(seed=0)
    oracle = ScriptedOracle()
    oracle.reset(env)
    for _ in range(40):
        env.step(oracle.act(env))
    sim = env.sim
    # both devices lie on the same track; the catheter tip is a track point
    assert sim.L[CATH] <= sim.track_s[-1] + 1e-6 and sim.L[WIRE] <= sim.track_s[-1] + 1e-6
    p = sim.position_at(sim.L[CATH])
    assert np.min(np.linalg.norm(toy_anatomy.nodes - p, axis=1)) < 1.0


def test_retraction_pops_track_and_counts(toy_anatomy):
    env = make_env(toy_anatomy, target="left_branch@-3")
    env.reset(seed=0)
    for _ in range(10):
        env.step(np.array([1.0, 0.0, 1.0, 0.0]))
    n_before = len(env.sim.track)
    for _ in range(5):
        env.step(np.array([-1.0, 0.0, -1.0, 0.0]))
    assert len(env.sim.track) < n_before
    assert env.episode_metrics()["retractions_guidewire"] == 1


def test_start_metrics_table(toy_anatomy):
    env = make_env(toy_anatomy, target="left_branch@-3")
    eps = [episode_start_metrics(run_episode(env, a, seed=s)) for a in (ScriptedOracle(), RandomAgent(0)) for s in range(3)]
    for m in eps:
        assert 0.0 <= m["path_ratio"] <= 100.0
        assert m["fluoroscopy_time_s"] == m["procedure_time_s"]
    md = markdown_table([summarize("oracle", eps[:3]), summarize("random", eps[3:])])
    assert "Success rate" in md and "oracle (n=3)" in md


def test_wilson_ci():
    lo, hi = wilson_ci(10, 10)
    assert hi == 1.0 and 0.65 < lo < 0.75
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.35


def test_drawn_device_ends_at_its_tip_and_only_shows_the_inserted_part(toy_anatomy):
    from stroke_nav.env.devices import CATH, WIRE, TrackSim, device_sim_params

    sim = TrackSim(toy_anatomy, device_sim_params())
    sim.reset(toy_anatomy.default_start(), np.random.default_rng(0))
    sim.L[:] = (3.0, 0.0)
    assert len(sim.device_polyline(CATH)) == 1  # not inserted: nothing to draw
    wire = sim.device_polyline(WIRE)
    assert np.linalg.norm(np.diff(wire, axis=0), axis=1).sum() < 6.0  # 3 mm in, not a 30 mm stick
    for _ in range(30):
        sim.step(np.array([1.0, 0.4, 1.0, 0.0]))
    for dev in (WIRE, CATH):
        pts = sim.device_polyline(dev)
        assert np.allclose(pts[-1], sim.tracking_points(dev)[0])
        r = toy_anatomy.radius[sim.node_at(sim.L[dev])]
        assert np.linalg.norm(pts[-1] - sim.position_at(sim.L[dev])) <= 0.7 * r + 1e-9  # inside the lumen
    # in-between frames: explicit insertions/rolls equal to the state reproduce it
    assert np.allclose(sim.device_polyline(WIRE, Ls=sim.L.copy(), phis=sim.phi.copy()), sim.device_polyline(WIRE))
