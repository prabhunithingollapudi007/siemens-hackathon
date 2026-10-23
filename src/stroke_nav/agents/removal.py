"""Withdrawing the guidewire from the patient on the organiser's physics, from their observation alone.

Strategies (``WithdrawalController``), compared by ``scripts/removal_trials.py``:

* ``speed``: the pull command, a fraction of their 10 mm/s; ``speed_by_depth`` (insertion mm -> fraction)
  slows it where the tip passes bends and junctions (the A* speed envelope along the route, reversed)
* ``unwind_first``: turn the shaft back to its starting roll before pulling, to release the twist stored
  by the entry roll instead of letting it snap the tip round on the way out
* ``guard`` (``stroke_nav.safety.wall.WallGuard``, with ``bind(env)``): on a wall crossing while pulling, stop,
  advance ``readvance_mm`` to free the wire, turn by the next ``reroll_deg`` offset and pull on at half speed;
  gives up (holds) after the last offset

``InsertThenWithdraw`` drives a navigation controller to ``insert_to_mm``, holds ``hold_steps`` and then
withdraws, so every strategy starts from the same state. Units: insertion mm, roll rad, actions in [-1, 1]
(``[v_guidewire, w_guidewire, v_catheter, w_catheter]``); the catheter stays at home.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from stroke_nav.agents.entry_roll import GW_COS, GW_INSERTION, GW_SIN, RAD_PER_STEP


class _Roll:
    """Unwrapped guidewire shaft rotation from the observation's sin/cos (rad)."""

    def __init__(self) -> None:
        self.rot, self.last = 0.0, None

    def update(self, s: np.ndarray) -> float:
        raw = math.atan2(float(s[GW_SIN]), float(s[GW_COS]))
        self.rot = raw if self.last is None else self.rot + (raw - self.last + math.pi) % (2 * math.pi) - math.pi
        self.last = raw
        return self.rot


class WithdrawalController:
    def __init__(self, speed: float = 1.0, unwind_first: bool = False, start_roll_rad: float = 0.0,
                 speed_by_depth: Callable[[float], float] | None = None, guard=None, readvance_mm: float = 3.0,
                 reroll_deg: tuple[float, ...] = (15.0, -15.0, 30.0, -30.0), out_mm: float = 1.0):
        self.speed, self.unwind_first, self.start_roll = float(speed), unwind_first, float(start_roll_rad)
        self.speed_by_depth, self.guard, self.readvance_mm = speed_by_depth, guard, float(readvance_mm)
        self.reroll_deg, self.out_mm = tuple(reroll_deg), float(out_mm)
        self.env = None

    def bind(self, env) -> None:
        self.env = env

    def reset(self) -> None:
        self.roll = _Roll()
        self.mode = "unwind" if self.unwind_first else "pull"
        self.retry, self.events = 0, []
        self.readvance_to = self.target_rot = 0.0
        self.slow = 1.0

    @property
    def done(self) -> bool:
        return self.mode in ("out", "stuck")

    def _crossing(self, ins_mm: float):
        if self.guard is None or self.env is None or not self.env.bridge.noisy["guidewire"]:
            return None
        from stroke_nav.safety.wall import guidewire_bodies

        c = self.guard.check(guidewire_bodies(self.env.bridge.noisy["guidewire"][-1]), ins_mm)
        return c if c.crossing else None

    def act(self, obs: dict) -> np.ndarray:
        s = np.asarray(obs["state"], np.float64)
        ins, rot = float(s[GW_INSERTION]) * 1000.0, self.roll.update(s)
        a = np.zeros(4, np.float32)
        if ins <= self.out_mm and self.mode != "unwind":
            self.mode = "out"
        if self.mode in ("out", "stuck"):
            return a
        if self.mode == "unwind":
            err = self.start_roll - rot
            if abs(err) > math.radians(3.0):
                a[1] = np.clip(err / RAD_PER_STEP, -1.0, 1.0)
                return a
            self.mode = "pull"
        c = self._crossing(ins)
        if c is not None and self.mode == "pull":
            self.events.append({"event": "wall_crossing", "ins_mm": round(ins, 1), "depth_mm": c.depth_mm, "kind": c.kind})
            if self.retry >= len(self.reroll_deg):
                self.mode = "stuck"
                self.events.append({"event": "stuck", "ins_mm": round(ins, 1)})
                return a
            self.target_rot = rot + math.radians(self.reroll_deg[self.retry])
            self.retry += 1
            self.readvance_to, self.mode, self.slow = ins + self.readvance_mm, "readvance", 0.5
        if self.mode == "readvance":
            if ins < self.readvance_to:
                a[0] = 0.3
                return a
            self.mode = "reroll"
        if self.mode == "reroll":
            err = self.target_rot - rot
            if abs(err) > math.radians(3.0):
                a[1] = np.clip(err / RAD_PER_STEP, -1.0, 1.0)
                return a
            self.mode = "pull"
        v = self.speed * self.slow
        if self.speed_by_depth is not None:
            v = min(v, float(self.speed_by_depth(ins)))
        a[0] = -v
        return a


class InsertThenWithdraw:
    """Navigate with ``navigator`` until ``insert_to_mm`` (or until insertion stops progressing for 10 steps), hold
    ``hold_steps`` steps, then ``withdrawal``."""

    def __init__(self, navigator, withdrawal: WithdrawalController, insert_to_mm: float, hold_steps: int = 5):
        self.nav, self.out, self.insert_to, self.hold_steps = navigator, withdrawal, float(insert_to_mm), int(hold_steps)

    def bind(self, env) -> None:
        for c in (self.nav, self.out):
            if hasattr(c, "bind"):
                c.bind(env)

    def reset(self) -> None:
        self.nav.reset()
        self.out.reset()
        self.phase, self.held, self.max_ins, self.progress = "insert", 0, 0.0, []

    @property
    def done(self) -> bool:
        return self.phase == "withdraw" and self.out.done

    @property
    def events(self) -> list:
        return list(getattr(self.nav, "events", [])) + self.out.events

    def act(self, obs: dict) -> np.ndarray:
        ins = float(obs["state"][GW_INSERTION]) * 1000.0
        self.max_ins = max(self.max_ins, ins)
        if self.phase == "insert":
            a = np.asarray(self.nav.act(obs), np.float32)
            self.progress.append(ins)
            stalled = len(self.progress) > 10 and self.progress[-1] - self.progress[-11] < 0.2 and ins > 1.0
            if ins < self.insert_to - 0.5 and not stalled:  # reached within 0.5 mm (insertion comes back as float32 metres)
                a[0] = min(a[0], max((self.insert_to - ins) / 2.0, 0.0))  # 2 mm per step at |v| = 1
                return a
            self.phase = "hold"
        if self.phase == "hold":
            self.held += 1
            if self.held < self.hold_steps:
                return np.zeros(4, np.float32)
            self.phase = "withdraw"
        return self.out.act(obs)
