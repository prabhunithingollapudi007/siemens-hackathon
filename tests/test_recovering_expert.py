"""RecoveringExpert: the DAgger teacher retracts and re-rolls when the guidewire went in at the wrong roll."""

import math

import numpy as np

from stroke_nav.agents.entry_roll import GOAL_DELTA, GW_COS, GW_INSERTION, GW_SIN, RecoveringExpert


def obs(ins_mm: float, roll_deg: float, goal_mm: float = 100.0) -> dict:
    s = np.zeros(43, np.float32)
    s[GW_INSERTION] = ins_mm / 1000.0
    s[GW_SIN], s[GW_COS] = math.sin(math.radians(roll_deg)), math.cos(math.radians(roll_deg))
    s[GOAL_DELTA] = [0.0, 0.0, goal_mm / 1000.0]
    return {"state": s, "image": np.zeros((1, 128, 128), np.uint8)}


def test_rolls_first_at_the_entry():
    e = RecoveringExpert(roll_deg=-60.0)
    e.reset()
    a = e.act(obs(0.0, 0.0))
    assert a[0] == 0.0 and a[1] < 0.0  # signed roll: turn the negative way


def test_retracts_when_in_at_the_wrong_roll_then_rolls_again():
    e = RecoveringExpert(roll_deg=90.0)
    e.reset()
    assert e.act(obs(60.0, 0.0)).tolist() == [-1.0, 0.0, 0.0, 0.0]  # 90 deg off at 60 mm: pull back
    assert e.act(obs(30.0, 0.0)).tolist() == [-1.0, 0.0, 0.0, 0.0]  # keeps pulling back
    a = e.act(obs(2.0, 0.0))
    assert a[0] == 0.0 and a[1] > 0.0  # back at the entry: roll


def test_pushes_when_the_roll_is_right():
    e = RecoveringExpert(roll_deg=90.0)
    e.reset()
    e.act(obs(0.0, 90.0))  # aligned: switches to push
    assert e.act(obs(40.0, 91.0))[0] > 0.0
