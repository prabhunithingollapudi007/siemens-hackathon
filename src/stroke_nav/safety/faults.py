"""Tracking fault injection: noise, latency and bursty, motion-correlated dropouts.

Real X-ray tip tracking loses frames mostly when the wire moves fast (motion blur at 4-10 Hz
fluoroscopy, Eyberg et al. 2022), and losses come in bursts. The organiser's noise layer uses
sigma 0.1 mm, 7 % dropout with hold-last-sample and 50-150 ms latency. ``FaultModel.organiser()``
reproduces that; ``FaultModel.stress()`` is a harsher setting for safety testing.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class FaultModel:
    sigma_mm: float = 0.0
    dropout_p: float = 0.0  # chance per step that a dropout burst starts
    burst_max: int = 1  # burst length drawn uniformly from 1..burst_max
    fast_mm_s: float = 25.0  # above this tip speed ...
    fast_extra_p: float = 0.0  # ... the dropout chance rises by this much (motion blur)
    latency_steps: int = 0  # delivered measurement is this many control steps old

    @classmethod
    def clean(cls) -> "FaultModel":
        return cls()

    @classmethod
    def organiser(cls) -> "FaultModel":
        return cls(sigma_mm=0.1, dropout_p=0.07, burst_max=1, latency_steps=1)

    @classmethod
    def stress(cls) -> "FaultModel":
        return cls(sigma_mm=1.0, dropout_p=0.08, burst_max=5, fast_extra_p=0.25, latency_steps=1)

    @property
    def active(self) -> bool:
        return self.sigma_mm > 0 or self.dropout_p > 0 or self.fast_extra_p > 0 or self.latency_steps > 0


class FaultInjector:
    """Turns true tracking points into delivered (noisy, delayed, sometimes missing) measurements."""

    def __init__(self, model: FaultModel, rng: np.random.Generator):
        self.m = model
        self.rng = rng
        self.reset()

    def reset(self) -> None:
        self.burst = 0
        self.queue: deque = deque()
        self.dropouts = 0

    def deliver(self, points: np.ndarray, tip_speed_mm_s: float) -> tuple[np.ndarray, bool]:
        """points: (devices, 3, 3) true tracking points in mm. Returns (delivered points, dropped)."""
        m = self.m
        noisy = points + (self.rng.normal(0.0, m.sigma_mm, points.shape) if m.sigma_mm > 0 else 0.0)
        self.queue.append(noisy)
        while len(self.queue) > m.latency_steps + 1:
            self.queue.popleft()
        delivered = self.queue[0]
        if self.burst > 0:
            self.burst -= 1
            dropped = True
        else:
            p = m.dropout_p + (m.fast_extra_p if tip_speed_mm_s > m.fast_mm_s else 0.0)
            dropped = bool(p > 0 and self.rng.random() < p)
            if dropped:
                self.burst = int(self.rng.integers(0, max(m.burst_max, 1)))
        self.dropouts += int(dropped)
        return delivered, dropped
