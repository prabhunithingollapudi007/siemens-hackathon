"""Entry-roll table for the organiser's rod physics: which entry roll reaches which point.

On their XPBD rods the roll set at the entry decides the branch the guidewire takes
(docs/ORGANISER_SETUP.md §12). ``scripts/calibrate_rolls.py`` turns the guidewire to each roll at
the entry and pushes it to the end of the robot's travel in the lockstep env, recording the true
tip path (world metres) for several noise seeds. Here:

* a point is reachable with a roll when the tip path of every seed of that roll passes within
  ``reach_mm`` of it (so the outcome does not hinge on one lucky run);
* ``RollTable.reach`` picks the roll whose worst seed passes closest, or (``prefer_small_within_mm``)
  the smallest |roll| among those within that distance. With signed rolls (-180, 180] the policy can
  then read its progress towards the roll from the observed sin/cos of the shaft rotation, which
  cannot tell 345 deg from -15 deg;
* ``RollTable.candidate_goals`` samples reachable goal points along the recorded paths, which is
  where generic targets for demonstrations and evaluation come from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RollPath:
    roll_deg: float
    seed: int
    tip_m: np.ndarray  # (N, 3) true guidewire tip after each control step, world metres
    ins_mm: np.ndarray  # (N,) observed guidewire insertion after each step


def path_distance_m(path: np.ndarray, point: np.ndarray) -> float:
    """Shortest distance from ``point`` to the polyline ``path`` (both world metres)."""
    a, b = path[:-1], path[1:]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", point - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18), 0.0, 1.0)
    return float(np.min(np.linalg.norm(a + t[:, None] * ab - point, axis=1))) if len(path) > 1 else \
        float(np.linalg.norm(path[0] - point))


class RollTable:
    def __init__(self, paths: list[RollPath]):
        if not paths:
            raise ValueError("RollTable needs recorded paths")
        self.paths = paths
        self.rolls = sorted({p.roll_deg for p in paths})

    @classmethod
    def load(cls, file: str | Path, roll_range: tuple[float, float] | None = None) -> "RollTable":
        """``file``: a sweep file or several, comma-separated; ``roll_range`` ``(lo, hi]``: keep only those rolls."""
        paths = []
        for f in str(file).split(","):
            doc = json.loads(Path(f.strip()).read_text(encoding="utf-8"))
            paths += [RollPath(float(p["roll_deg"]), int(p["seed"]), np.asarray(p["tip_m"], dtype=float),
                               np.asarray(p["ins_mm"], dtype=float)) for p in doc["paths"]]
        if roll_range is not None:
            paths = [p for p in paths if roll_range[0] < p.roll_deg <= roll_range[1]]
        return cls(paths)

    def worst_mm(self, roll_deg: float, goal_m) -> float:
        """Distance (mm) by which the worst seed of ``roll_deg`` misses ``goal_m``."""
        goal = np.asarray(goal_m, dtype=float)
        return max(path_distance_m(p.tip_m, goal) for p in self.paths if p.roll_deg == roll_deg) * 1000.0

    def reach(self, goal_m, prefer_small_within_mm: float | None = None,
              robust_within_mm: float | None = None) -> tuple[float, float]:
        """(roll in degrees, worst-seed miss in mm) of the roll that reaches ``goal_m`` best.

        ``prefer_small_within_mm``: the smallest |roll| among those that miss by at most that much.
        ``robust_within_mm``: the **centre** of the longest contiguous run of rolls (on the sweep's
        grid) that miss by at most that much. Neighbouring rolls usually land in the same vessel, so
        the run is the arc of rolls that works; its edges are where a slightly different executed
        roll (noise, the robot's deadband) takes another branch. For the cas-reference goal the
        closest roll was 150 deg (half the episodes failed), the arc centre 120 deg (all succeeded).
        """
        misses = [(self.worst_mm(r, goal_m), r) for r in self.rolls]
        if prefer_small_within_mm is not None:
            ok = [(abs(r), m, r) for m, r in misses if m <= prefer_small_within_mm]
            if ok:
                _, miss, roll = min(ok)
                return roll, miss
        if robust_within_mm is not None:
            good = [m <= robust_within_mm for m, _ in misses]
            if any(good):
                n = len(good)
                step = min(np.diff(self.rolls)) if n > 1 else 0.0
                # runs over the sorted rolls, allowing a gap of one grid step and the wrap at +-180 / 360
                best, run = [], []

                def score(r):  # longer arc first; among equal arcs the one whose rolls miss least
                    return (len(r), -float(np.mean([misses[j % n][0] for j in r])))

                for i in range(2 * n):
                    k = i % n
                    if good[k] and (not run or (i == run[-1] + 1 and
                                                 (self.rolls[k] - self.rolls[run[-1] % n]) % 360.0 <= 1.5 * step + 1e-9)):
                        run.append(i)
                    else:
                        run = [i] if good[k] else []
                    if not best or score(run) > score(best):
                        best = list(run)
                best = best[:n]
                # inside the arc (its edges are where a slightly different executed roll takes another
                # branch), the roll whose recorded paths pass closest; ties (within 0.5 mm) go to the centre
                interior = best[1:-1] if len(best) > 2 else best
                centre = len(best) // 2
                k = min(interior, key=lambda j: (round(misses[j % n][0] * 2.0) / 2.0, abs(best.index(j) - centre))) % n
                return self.rolls[k], misses[k][0]
        miss, roll = min(misses)
        return roll, miss

    def roll_for(self, goal_m, reach_mm: float = 2.5) -> float | None:
        roll, miss = self.reach(goal_m)
        return roll if miss <= reach_mm else None

    def candidate_goals(self, spacing_mm: float = 5.0, reach_mm: float = 2.5, min_ins_mm: float = 40.0,
                        prefer_small_roll: bool = False) -> list[dict]:
        """Reachable goal points about ``spacing_mm`` apart along the recorded paths (first seed of
        each roll), at least ``min_ins_mm`` of insertion from the entry; each with its best roll (or,
        ``prefer_small_roll``, the smallest |roll| that reaches it)."""
        goals: list[dict] = []
        kept = np.empty((0, 3))
        for roll in self.rolls:
            p = next(q for q in self.paths if q.roll_deg == roll)
            seg = np.linalg.norm(np.diff(p.tip_m, axis=0), axis=1)
            arc = np.concatenate([[0.0], np.cumsum(seg)]) * 1000.0
            for s in np.arange(0.0, arc[-1], spacing_mm):
                k = int(np.searchsorted(arc, s))
                if p.ins_mm[min(k, len(p.ins_mm) - 1)] < min_ins_mm:
                    continue
                point = p.tip_m[min(k, len(p.tip_m) - 1)]
                if len(kept) and np.min(np.linalg.norm(kept - point, axis=1)) * 1000.0 < spacing_mm:
                    continue  # paths overlap in the aorta and the shared trunks
                best_roll, miss = self.reach(point, reach_mm if prefer_small_roll else None)
                if miss > reach_mm:
                    continue
                goals.append({"goal_m": point.tolist(), "roll_deg": best_roll, "miss_mm": round(miss, 2),
                              "ins_mm": round(float(p.ins_mm[min(k, len(p.ins_mm) - 1)]), 1)})
                kept = np.vstack([kept, point])
        return goals


def load_goal_set(file: str | Path, splits: str = "train") -> list[dict]:
    """Goals written by ``scripts/make_goals.py`` whose split is in the comma-separated ``splits``
    (``train``, ``test``, ``cas``); each has ``goal_m`` (world metres), ``roll_deg`` and ``branch``."""
    wanted = {s.strip() for s in splits.split(",")}
    goals = [g for g in json.loads(Path(file).read_text(encoding="utf-8"))["goals"] if g["split"] in wanted]
    if not goals:
        raise ValueError(f"no goals with split {splits!r} in {file}")
    return goals
