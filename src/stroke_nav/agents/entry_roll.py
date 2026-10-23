"""Scripted expert for the organiser's rod physics: roll the guidewire at the entry, then push.

On their XPBD rods, re-steering at each bifurcation fails (stored twist releases in whips), but a
single roll at the entry decides the branch (docs/ORGANISER_SETUP.md §12): 120 deg then a 260 mm
push ends about 4 mm from the cas-reference goal. ``EntryRollExpert`` does that from the
organiser observation alone (``Dict(image, state(43))``), so a policy can be cloned from it:

* roll phase: turn the guidewire to ``roll_deg`` (proportional, from the observed shaft rotation)
* push phase: push the guidewire, slowing inside ``slow_radius_mm`` of the goal, holding inside
  ``stop_radius_mm``, and backing off slowly once it has passed the goal by ``overshoot_mm``
* the catheter is not moved (module 2 stays at home)

State indices follow ``openng3_rl.observation.ObservationBuilder``: guidewire insertion (m) at 17,
sin/cos of its shaft rotation (rad) at 18/19, goal minus noisy guidewire tip (m) at 36:39.
Action order ``[v_guidewire, w_guidewire, v_catheter, w_catheter]`` in [-1, 1]; at |a| = 1 the
robot moves 10 mm/s and 1 rad/s, i.e. 2 mm and 0.2 rad per 5 Hz step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

GW_INSERTION, GW_SIN, GW_COS = 17, 18, 19
GOAL_DELTA = slice(36, 39)
RAD_PER_STEP = 0.2  # rotation at |w| = 1 per control step
MM_PER_STEP = 2.0  # insertion at |v| = 1 per control step


@dataclass
class EntryRollExpert:
    roll_deg: float = 120.0
    roll_tol_deg: float = 2.0
    slow_radius_mm: float = 25.0
    min_push: float = 0.25
    stop_radius_mm: float = 2.5
    overshoot_mm: float = 4.0
    hold_roll: bool = True

    def reset(self) -> None:
        self.phase = "roll"
        self.rot = 0.0  # unwrapped guidewire rotation (rad)
        self.last_raw: float | None = None
        self.best_mm = math.inf

    def _rotation(self, state: np.ndarray) -> float:
        raw = math.atan2(float(state[GW_SIN]), float(state[GW_COS]))
        if self.last_raw is None:
            self.rot = raw
        else:
            self.rot += (raw - self.last_raw + math.pi) % (2.0 * math.pi) - math.pi
        self.last_raw = raw
        return self.rot

    def act(self, obs: dict) -> np.ndarray:
        if not hasattr(self, "phase"):
            self.reset()
        state = np.asarray(obs["state"], dtype=np.float64)
        rot = self._rotation(state)
        dist_mm = float(np.linalg.norm(state[GOAL_DELTA])) * 1000.0
        err = math.radians(self.roll_deg) - rot
        a = np.zeros(4, dtype=np.float32)
        if self.phase == "roll":
            a[1] = np.clip(err / RAD_PER_STEP, -1.0, 1.0)
            if abs(err) < math.radians(self.roll_tol_deg):
                self.phase = "push"
            return a
        if self.hold_roll and abs(err) > math.radians(3 * self.roll_tol_deg):
            a[1] = np.clip(err / RAD_PER_STEP, -0.5, 0.5)
        self.best_mm = min(self.best_mm, dist_mm)
        if dist_mm <= self.stop_radius_mm:
            return a  # hold in the goal
        if dist_mm > self.best_mm + self.overshoot_mm and self.best_mm < self.slow_radius_mm:
            a[0] = -self.min_push  # passed the goal: back off slowly
            return a
        a[0] = np.clip(dist_mm / self.slow_radius_mm, self.min_push, 1.0)
        return a
