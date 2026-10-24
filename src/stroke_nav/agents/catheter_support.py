"""Catheter-supported navigation on the organiser's physics: the catheter follows the guidewire as support.

On their rod an unsupported guidewire pushed through the aortic arch bows out and slips through the wall
between its collision bodies (docs/SAFETY.md §7). ``CatheterSupportedRoll`` keeps the free wire short: the
guidewire is driven by the entry-roll controller (``EntryRollExpert``: roll at the entry, then push); once the
roll is done, the 5F catheter tracks ``lead_mm`` behind the wire tip, over the wire, until it parks at
``park_mm`` of insertion (in a wide vessel before the target); from there the wire goes on alone. If the
catheter falls more than ``slack_mm`` behind its place, the wire waits for it.

Insertions are read from the organiser observation (guidewire index 17, catheter index 28, metres); actions
``[v_guidewire, w_guidewire, v_catheter, w_catheter]`` in [-1, 1], 2 mm per step at |v| = 1. The catheter is
not rotated.
"""

from __future__ import annotations

import numpy as np

from stroke_nav.agents.entry_roll import GW_INSERTION, EntryRollExpert

CATH_INSERTION = 28  # catheter aux block starts at 25: tip velocity (3), insertion (m)
MM_PER_STEP = 2.0


class CatheterSupportedRoll:
    def __init__(self, wire: EntryRollExpert, lead_mm: float = 25.0, park_mm: float = 200.0, slack_mm: float = 10.0):
        self.wire, self.lead_mm, self.park_mm, self.slack_mm = wire, float(lead_mm), float(park_mm), float(slack_mm)

    def reset(self) -> None:
        self.wire.reset()

    @property
    def phase(self) -> str:
        return getattr(self.wire, "phase", "push")

    def act(self, obs: dict) -> np.ndarray:
        a = np.asarray(self.wire.act(obs), np.float32).copy()
        s = np.asarray(obs["state"], np.float64)
        w_ins, c_ins = float(s[GW_INSERTION]) * 1000.0, float(s[CATH_INSERTION]) * 1000.0
        a[2] = a[3] = 0.0
        if self.phase == "roll":
            return a  # the catheter waits until the wire has turned
        target = max(0.0, min(w_ins - self.lead_mm, self.park_mm))  # never behind its home position
        a[2] = float(np.clip((target - c_ins) / MM_PER_STEP, -1.0, 1.0))
        if c_ins < self.park_mm - 2.0 and w_ins - c_ins > self.lead_mm + self.slack_mm:
            a[0] = min(a[0], 0.0)  # the wire waits for its support
        return a
