"""Withdrawal controller and insert-then-withdraw sequencing, on synthetic observations (no organiser pages)."""

import numpy as np

from stroke_nav.agents.entry_roll import GW_COS, GW_INSERTION, GW_SIN
from stroke_nav.agents.removal import InsertThenWithdraw, WithdrawalController
from stroke_nav.safety.wall import WallCheck


def obs(ins_mm, rot_rad=0.0):
    s = np.zeros(43, np.float32)
    s[GW_INSERTION], s[GW_SIN], s[GW_COS] = ins_mm / 1000.0, np.sin(rot_rad), np.cos(rot_rad)
    return {"state": s}


def test_pull_unwind_and_out():
    c = WithdrawalController(speed=0.5, unwind_first=True)
    c.reset()
    a = c.act(obs(200, np.radians(120)))  # 120 deg of entry roll: turn back first, no pull yet
    assert a[0] == 0 and a[1] < 0 and c.mode == "unwind"
    a = c.act(obs(200, 0.0))
    assert c.mode == "pull" and abs(a[0] + 0.5) < 1e-6
    assert not np.any(c.act(obs(0.5))) and c.done and c.mode == "out"


def test_speed_by_depth_caps_the_pull():
    c = WithdrawalController(speed=1.0, speed_by_depth=lambda ins: 0.25 if ins < 100 else 1.0)
    c.reset()
    assert abs(c.act(obs(300))[0] + 1.0) < 1e-6 and abs(c.act(obs(50))[0] + 0.25) < 1e-6


class StubEnv:
    class bridge:
        noisy = {"guidewire": [{"seg": [0.0] * 14}]}


class StubGuard:
    crossing = False

    def check(self, bodies, ins):
        return WallCheck(self.crossing, 1.0 if self.crossing else 0.0, 0.0, None, 2, "shortcut" if self.crossing else "")


def test_guarded_pull_readvances_rerolls_then_gives_up():
    g = StubGuard()
    c = WithdrawalController(guard=g, readvance_mm=3.0, reroll_deg=(15.0,))
    c.bind(StubEnv())
    c.reset()
    assert c.act(obs(100))[0] < 0
    g.crossing = True
    a = c.act(obs(98))
    assert c.mode == "readvance" and a[0] > 0  # free the wire: a little forward, never a pull while crossed
    g.crossing = False
    c.act(obs(101))  # re-advanced 3 mm: now turn by the offset
    assert c.mode == "reroll"
    a = c.act(obs(101, np.radians(15)))
    assert c.mode == "pull" and -1.0 < a[0] < 0  # pulls on, at half speed
    g.crossing = True
    assert not np.any(c.act(obs(97, np.radians(15)))) and c.mode == "stuck" and c.done


def test_insert_then_withdraw_sequence():
    class Push:
        def reset(self):
            pass

        def act(self, o):
            return np.array([1.0, 0.2, 0.0, 0.0], np.float32)

    seq = InsertThenWithdraw(Push(), WithdrawalController(), insert_to_mm=10.0, hold_steps=2)
    seq.reset()
    assert seq.act(obs(0))[0] == 1.0 and seq.phase == "insert"
    assert abs(seq.act(obs(9))[0] - 0.5) < 1e-5  # slows to stop at the depth (2 mm per step at |v| = 1)
    assert not np.any(seq.act(obs(10))) and seq.phase == "hold"
    assert seq.act(obs(10))[0] < 0 and seq.phase == "withdraw"
    seq.act(obs(0.2))
    assert seq.done
