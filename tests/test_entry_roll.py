import math

import numpy as np

from stroke_nav.agents.entry_roll import GOAL_DELTA, GW_COS, GW_SIN, EntryRollExpert


def _obs(rot_rad: float, goal_delta_mm) -> dict:
    state = np.zeros(43, dtype=np.float32)
    state[GW_SIN], state[GW_COS] = math.sin(rot_rad), math.cos(rot_rad)
    state[GOAL_DELTA] = np.asarray(goal_delta_mm, dtype=np.float32) / 1000.0
    return {"image": np.zeros((1, 128, 128), np.uint8), "state": state}


def test_rolls_first_then_pushes_and_holds_in_goal():
    ex = EntryRollExpert(roll_deg=120.0)
    ex.reset()
    rot = 0.0
    for _ in range(30):  # integrate the commanded roll: 0.2 rad per step at |w| = 1
        a = ex.act(_obs(rot, [0.0, 0.0, 200.0]))
        assert a[0] == 0.0 and a[2] == 0.0 and a[3] == 0.0
        if ex.phase == "push":
            break
        rot += 0.2 * float(a[1])
    assert ex.phase == "push"
    assert abs(math.degrees(rot) - 120.0) < 2.0
    far = ex.act(_obs(rot, [0.0, 0.0, 200.0]))
    near = ex.act(_obs(rot, [0.0, 0.0, 10.0]))
    hold = ex.act(_obs(rot, [0.0, 0.0, 1.0]))
    assert far[0] == 1.0 and 0.0 < near[0] < 1.0 and hold[0] == 0.0


def test_unwraps_rotation_past_180_degrees():
    ex = EntryRollExpert(roll_deg=270.0)
    ex.reset()
    rot = 0.0
    for _ in range(40):
        a = ex.act(_obs(rot, [0.0, 0.0, 200.0]))
        if ex.phase == "push":
            break
        rot += 0.2 * float(a[1])
    assert abs(math.degrees(rot) - 270.0) < 2.0


def test_backs_off_after_passing_the_goal():
    ex = EntryRollExpert(roll_deg=0.0)
    ex.reset()
    ex.act(_obs(0.0, [0.0, 0.0, 50.0]))  # roll reached at once
    ex.act(_obs(0.0, [0.0, 0.0, 6.0]))  # closest approach
    a = ex.act(_obs(0.0, [0.0, 0.0, 12.0]))  # 6 mm past it
    assert a[0] < 0.0
