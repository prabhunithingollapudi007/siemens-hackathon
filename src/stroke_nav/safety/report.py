"""Safety summary: worst cases and exceedance rates, not just means.

A mean hides the one episode that perforates. Each hazard is therefore reported three ways where it
makes sense: the mean, the worst episode, and the share of episodes that crossed a threshold.
"""

from __future__ import annotations

import numpy as np

from stroke_nav.eval.start_metrics import wilson_ci
from stroke_nav.safety.monitor import RULES

WRONG_DEPTH_MM = 10.0  # deeper than this into a wrong vessel counts as a hazardous wrong-branch entry
OVERSHOOT_MM = 10.0  # beyond the target along the path


def summarize_safety(episodes: list[dict]) -> dict:
    def arr(k):
        return np.array([e[k] for e in episodes], dtype=float)

    n = len(episodes)
    succ = arr("success")
    walls = arr("wall_pushes_guidewire") + arr("wall_pushes_microcatheter")
    depth, over, cath = arr("safety_max_wrong_depth_mm"), arr("safety_max_overshoot_mm"), arr("safety_unsupported_catheter_advance_mm")
    t = arr("steps") * arr("dt")
    lo, hi = wilson_ci(succ.sum(), n)
    out = {
        "n": n,
        "success_pct": 100 * succ.mean(), "success_ci": (100 * lo, 100 * hi),
        "time_s_success": float(t[succ > 0].mean()) if succ.any() else float("nan"),
        "wrong_branch_entries": float(arr("wrong_branch_events").mean()),
        "wrong_depth_mean_mm": float(depth.mean()), "wrong_depth_worst_mm": float(depth.max()),
        "wrong_depth_exceed_pct": 100 * float((depth > WRONG_DEPTH_MM).mean()),
        "wall_pushes": float(walls.mean()), "wall_push_eps_pct": 100 * float((walls > 0).mean()),
        "overshoot_worst_mm": float(over.max()), "overshoot_exceed_pct": 100 * float((over > OVERSHOOT_MM).mean()),
        "unsupported_cath_mm": float(cath.mean()), "unsupported_cath_eps_pct": 100 * float((cath > 0.5).mean()),
        "tip_speed_worst_mm_s": float(arr("safety_max_tip_speed_mm_s").max()),
        "speed_jumps": float(arr("safety_speed_jumps").mean()),
        "off_corridor_steps": float(arr("safety_steps_off_corridor").mean()),
        "dropouts": float(arr("safety_dropouts").mean()),
        "steps_intervened": float(arr("safety_steps_intervened").mean()),
    }
    for r in RULES:
        out[f"fired_{r}"] = float(arr(f"safety_fired_{r}").mean())
    return out


ROWS = [
    ("success_pct", "Success rate", "%"),
    ("time_s_success", "Procedure time (successful)", "s"),
    ("wrong_branch_entries", "Wrong-branch entries / ep.", ""),
    ("wrong_depth_worst_mm", "Deepest wrong-branch entry (worst ep.)", "mm"),
    ("wrong_depth_exceed_pct", f"Episodes > {WRONG_DEPTH_MM:.0f} mm into a wrong vessel", "%"),
    ("wall_push_eps_pct", "Episodes with a blocked push", "%"),
    ("overshoot_worst_mm", "Overshoot past target (worst ep.)", "mm"),
    ("overshoot_exceed_pct", f"Episodes > {OVERSHOOT_MM:.0f} mm past target", "%"),
    ("unsupported_cath_eps_pct", "Episodes with catheter advanced ahead of the wire", "%"),
    ("tip_speed_worst_mm_s", "Tip speed (worst step)", "mm/s"),
    ("off_corridor_steps", "Steps off the approved corridor / ep.", ""),
    ("dropouts", "Tracking dropouts / ep.", ""),
    ("steps_intervened", "Monitor interventions / ep.", ""),
] + [(f"fired_{r}", f"  {r} {name}", "steps/ep.") for r, name in RULES.items()]


def markdown(columns: list[tuple[str, dict]], title: str) -> str:
    head = "| Safety metric | unit | " + " | ".join(c for c, _ in columns) + " |"
    lines = [f"### {title}", "", head, "|---|---|" + "---|" * len(columns)]
    for key, label, unit in ROWS:
        cells = []
        for _, s in columns:
            v = s[key]
            if key == "success_pct":
                cells.append(f"{v:.0f} [{s['success_ci'][0]:.0f}-{s['success_ci'][1]:.0f}]")
            elif not np.isfinite(v):
                cells.append("n/a")
            else:
                cells.append(f"{v:.1f}")
        lines.append(f"| {label} | {unit} | " + " | ".join(cells) + " |")
    lines += ["", "Worst-case and exceedance rows matter more than means: one deep wrong-branch entry is the "
              "event that injures. Thresholds are engineering placeholders, not clinically validated limits.", ""]
    return "\n".join(lines)
