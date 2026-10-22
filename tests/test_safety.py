import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from stroke_nav.agents.scripted import RandomAgent, ScriptedOracle, run_episode
from stroke_nav.env.vessel_env import EnvConfig, VesselNavEnv
from stroke_nav.safety import FaultModel, Measurement, SafetyConfig, SafetyMonitor, SafetyShield

TARGETS = ["left_branch@-3", "right_branch@-3", "back_branch@-3"]


# ---------------------------------------------------------------- monitor unit tests (no simulator)
def straight_monitor(radius=10.0, length=200.0, target_s=150.0, **cfg):
    s = np.arange(0.0, length + 1e-9, 0.5)
    pts = np.column_stack([np.zeros_like(s), np.zeros_like(s), s])
    mon = SafetyMonitor(SafetyConfig(**cfg), v_max_mm_s=(40.0, 40.0), dt=1 / 7.5)
    mon.reset(pts, s, np.full_like(s, radius), target_s)
    return mon


def meas(L=(50.0, 40.0), tip=(0.0, 0.0, 50.0), blocked=(False, False), force=None):
    return Measurement(np.array(L, float), None if tip is None else np.array(tip, float), np.array(blocked), force)


def test_r1_speed_envelope_by_lumen():
    assert np.allclose(straight_monitor(radius=10.0).filter([1, 0, 1, 0], meas()).action, [1, 0, 1, 0])
    d = straight_monitor(radius=1.0).filter([1, 0, 1, 0], meas(L=(50, 30)))
    assert d.action[0] == pytest.approx(0.25) and "R1" in d.fired


def test_r2_backoff_after_blocked_push():
    mon = straight_monitor()
    d = mon.filter([1, 0, 0, 0], meas(blocked=(True, False)))
    assert d.action[0] < 0 and "R2" in d.fired
    assert mon.filter([1, 0, 0, 0], meas(tip=(0, 0, 49.5))).action[0] < 0
    assert mon.filter([1, 0, 0, 0], meas(tip=(0, 0, 49.0))).action[0] == pytest.approx(1.0)


def test_r2_force_limit_on_leader():
    d = straight_monitor().filter([1, 0, 1, 0], meas(L=(50, 30), force=1.2))
    assert d.action[0] < 0 and d.action[2] == pytest.approx(1.0)


def test_r3_corridor_blocks_advance_allows_retract():
    # corridor = 4 mm + 0.7 x lumen radius (10 mm here) = 11 mm from the approved path
    mon = straight_monitor()
    d = mon.filter([1, 0.5, 0, 0], meas(tip=(12.0, 0.0, 50.0)))
    assert d.action[0] == 0 and d.action[1] == pytest.approx(0.5) and "R3" in d.fired
    assert mon.filter([-0.5, 0, 0, 0], meas(tip=(12.0, 0.0, 49.0))).action[0] == pytest.approx(-0.5)


def test_r3_allows_a_tip_resting_against_the_wall():
    """A pre-curved tip lies against the vessel wall, off the centerline but inside the lumen."""
    d = straight_monitor().filter([1, 0, 0, 0], meas(tip=(7.0, 0.0, 50.0)))
    assert d.action[0] == pytest.approx(1.0) and "R3" not in d.fired
    narrow = straight_monitor(radius=2.0).filter([1, 0, 0, 0], meas(L=(50, 30), tip=(7.0, 0.0, 50.0)))
    assert "R3" in narrow.fired  # the same offset in a 2 mm vessel is outside the lumen


def test_r4_hold_after_consecutive_dropouts_and_on_jumps():
    mon = straight_monitor()
    assert mon.filter([1, 0, 0, 0], meas(tip=None)).action[0] == pytest.approx(1.0)  # one dropout tolerated
    assert np.all(mon.filter([1, 1, 1, 1], meas(tip=None)).action == 0)  # second one: hold
    mon2 = straight_monitor()
    mon2.filter([0, 0, 0, 0], meas())
    mon2.filter([0, 0, 0, 0], meas(tip=(0, 0, 90.0)))  # 40 mm jump while the encoder did not move: rejected
    assert np.all(mon2.filter([1, 0, 0, 0], meas(tip=(0, 0, 95.0))).action == 0)


def test_r4_recovers_after_a_dropout_burst_while_moving():
    """Regression: a tip that moved during a burst must be accepted again (encoder cross-check),
    otherwise the plausibility check rejects every later frame and the robot holds forever."""
    mon = straight_monitor()
    mon.filter([1, 0, 0, 0], meas(L=(50, 40), tip=(0, 0, 50)))
    for k in range(1, 6):  # wire advances 5 mm per step while tracking is lost
        mon.filter([1, 0, 0, 0], meas(L=(50 + 5 * k, 40), tip=None))
    d = mon.filter([1, 0, 0, 0], meas(L=(80, 40), tip=(0, 0, 80)))
    assert "R4" not in d.fired and d.action[0] > 0


def test_r5_insertion_fence():
    d = straight_monitor(target_s=150.0).filter([1, 0, 0, 0], meas(L=(158.0, 140.0), tip=(0, 0, 158.0)))
    assert d.action[0] == pytest.approx(2.0 / (40 / 7.5)) and "R5" in d.fired


def test_r6_catheter_does_not_overtake_wire():
    mon = straight_monitor()
    assert mon.filter([0, 0, 1, 0], meas(L=(50, 50))).action[2] == 0
    d = mon.filter([1, 0, 1, 0], meas(L=(50, 50)))
    assert d.action[2] == pytest.approx(1.0)  # wire advances by the same step, so the catheter may follow


def test_operator_override_outranks_monitor_except_speed():
    mon = straight_monitor(radius=1.0)
    d = mon.filter([0, 0, 0, 0], meas(tip=(6.0, 0, 50.0)), operator_action=[1, 0, 0, 0])
    assert d.override and d.action[0] == pytest.approx(0.25)
    mon.filter([0, 0, 0, 0], meas(tip=None))
    d = mon.filter([0, 0, 0, 0], meas(tip=None), operator_action=[-1, 0, 0, 0])
    assert d.action[0] == pytest.approx(-1.0)


def test_monitor_never_adds_motion():
    rng = np.random.default_rng(0)
    for _ in range(500):
        mon = straight_monitor(radius=rng.uniform(0.5, 12))
        a = rng.uniform(-1, 1, 4)
        m = meas(L=rng.uniform(0, 160, 2), tip=None if rng.random() < 0.2 else rng.normal(0, 3, 3) + [0, 0, 50],
                 blocked=rng.random(2) < 0.2, force=rng.uniform(0, 2))
        f = mon.filter(a, m).action
        assert f[0] <= a[0] + 1e-9 and f[2] <= a[2] + 1e-9  # never pushes more than asked
        assert abs(f[1]) <= abs(a[1]) + 1e-9 and abs(f[3]) <= abs(a[3]) + 1e-9  # never rotates more
        assert f[0] >= min(a[0], -0.15) - 1e-9 and f[2] >= min(a[2], -0.15) - 1e-9  # back-off is bounded


# ---------------------------------------------------------------- shield on the simulator
def shielded(anat, target, faults=None, monitor=True):
    cfg = EnvConfig(target=target, max_steps=300)
    return SafetyShield(VesselNavEnv(cfg, anatomy=anat), faults=faults or FaultModel.clean(), monitor=monitor)


def test_shield_api(toy_anatomy):
    env = shielded(toy_anatomy, "random:left_branch@20-40|right_branch@20-40", FaultModel.stress())
    check_env(env, skip_render_check=True)


@pytest.mark.parametrize("target", TARGETS)
def test_shield_keeps_oracle_successful_under_faults(toy_anatomy, target):
    env = shielded(toy_anatomy, target, FaultModel.stress())
    res = [run_episode(env, ScriptedOracle(), seed=s) for s in range(5)]
    assert all(r["success"] for r in res)
    assert all(r["safety_max_wrong_depth_mm"] == 0 for r in res)


def test_shield_does_not_deadlock_follower_under_stress():
    from stroke_nav.agents.follower import TrajectoryFollower
    from stroke_nav.safety.demo_anatomy import demo_anatomy

    from stroke_nav.env.devices import device_sim_params

    anat = demo_anatomy()
    # Devices from device-parameters.json. The demo routes (~390 mm) are longer than the robot's
    # guidewire travel (264.5 mm), so travel limits are off here; 10 mm/s needs ~500 steps.
    env = SafetyShield(VesselNavEnv(EnvConfig(target="random:left_carotid@-40--3|right_carotid@-40--3", max_steps=800,
                                              sim=device_sim_params(robot_limits=False)),
                                    anatomy=anat), faults=FaultModel.stress(), monitor=True)
    res = [run_episode(env, TrajectoryFollower(), seed=s) for s in range(6)]
    assert all(r["success"] for r in res)


def test_shield_reduces_hazards_of_random_policy(toy_anatomy):
    def worst(monitor):
        env = shielded(toy_anatomy, "random:left_branch@20-40|right_branch@20-40", FaultModel.stress(), monitor)
        res = [run_episode(env, RandomAgent(seed=s), seed=s) for s in range(15)]
        return (max(r["safety_max_wrong_depth_mm"] for r in res),
                sum(r["wall_pushes_guidewire"] + r["wall_pushes_microcatheter"] for r in res),
                max(r["safety_max_overshoot_mm"] for r in res),
                sum(r["safety_unsupported_catheter_advance_mm"] for r in res))
    off, on = worst(False), worst(True)
    assert on[0] < off[0] / 2  # wrong-branch depth contained
    assert on[1] <= off[1]  # no more blocked pushes than without the shield
    assert on[2] <= 10.0 + 1e-6  # insertion fence holds
    assert on[3] < off[3]


def test_shield_is_deterministic(toy_anatomy):
    def run(seed):
        env = shielded(toy_anatomy, "random:left_branch@20-40|right_branch@20-40", FaultModel.stress())
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(1)
        out = []
        for _ in range(80):
            obs, r, te, tr, info = env.step(rng.uniform(-1, 1, 4))
            out.append(np.concatenate([obs, info["safety"]["command"], [r]]))
            if te or tr:
                break
        return np.array(out)
    np.testing.assert_array_equal(run(3), run(3))
