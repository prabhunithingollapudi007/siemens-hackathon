"""Loop-and-hook into an arch branch on the organiser's physics (experiment, numeric feedback only).

Pushing forward from the descending aorta, the wire reaches the left CCA opening at the wrong angle and slips
through the arch wall (docs/SAFETY.md §7). The reverse-curve technique: push past the branches into the wide
ascending aorta until the wire tip turns back on itself, then pull back so the reversed tip drops into the
branch; pulling then drives it up the branch.

    python scripts/loop_hook.py sweep                      # which entry rolls carry the wire into the ascending aorta
    python scripts/loop_hook.py run --roll 90 --loop-arc 35
    python scripts/loop_hook.py run --roll 90 --visible --record results/loop_hook/loop_hook.mp4

Scenario ``0028_H_ABAO_H-arch``: the 0028 mesh entered in the descending thoracic aorta (Aorta@200 mm from the
aortic root) heading to the arch; frame ``data/processed/organiser_frame_0028_H_ABAO_H-arch.json``. Positions
along the aorta are arc mm from the root (branch origins: brachiocephalic 67, left CCA 74; root open at 0).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stroke_nav.agents.entry_roll import GW_INSERTION, EntryRollExpert  # noqa: E402
from stroke_nav.data.anatomy import Anatomy  # noqa: E402
from stroke_nav.organiser.frame import OrganiserFrame  # noqa: E402
from stroke_nav.organiser.live.lockstep import kill_stray_pages, make_env  # noqa: E402
from stroke_nav.organiser.live.rollout import quat_z, wall_guard_for  # noqa: E402
from stroke_nav.safety.wall import guidewire_bodies  # noqa: E402

SCEN, TARGET = "0028_H_ABAO_H-arch", "Left_Common_Carotid@30"
ANAT = Anatomy.load("0028_H_ABAO_H")
FRAME = OrganiserFrame.load(ROOT / "data" / "processed" / f"organiser_frame_{SCEN}.json")
GOAL_VMR = ANAT.nodes[ANAT.resolve(TARGET)]
GOAL_M = FRAME.vmr_mm_to_world_m(GOAL_VMR)
AO = ANAT.branch_nodes("Aorta")


def where(tip_world_m) -> tuple[str, float]:
    """Vessel and arc (mm from the branch start; Aorta: from the root) of a world point."""
    n = ANAT.nearest_node(FRAME.world_m_to_vmr_mm(np.asarray(tip_world_m, float)))
    b = ANAT.branch_of(n)
    return b, float(ANAT.arc[n] - ANAT.arc[ANAT.branch_nodes(b)].min())


def aorta_dir_world(arc_mm: float) -> np.ndarray:
    """Unit direction of the aorta at ``arc_mm``, pointing towards the root (world frame)."""
    a = AO[np.argmin(abs(ANAT.arc[AO] - arc_mm))]
    b = AO[np.argmin(abs(ANAT.arc[AO] - (arc_mm - 8.0)))]
    d = FRAME.vmr_dir_to_world(ANAT.nodes[b] - ANAT.nodes[a])
    return d / np.linalg.norm(d)


class LoopHook:
    """Phases: ``roll`` + ``advance`` (entry roll, push into the ascending aorta), ``loop`` (push on until the tip
    points away from the root, never below ``floor_arc`` mm), ``hook`` (pull back until the tip is in the
    branch), ``drive`` (keep pulling while the tip gets closer to the goal), ``hold``."""

    def __init__(self, roll_deg: float, loop_arc: float = 35.0, floor_arc: float = 22.0, pull: float = 0.4,
                 branch: str = "Left_Common_Carotid", flipped_cos: float = 0.0):
        self.wire = EntryRollExpert(roll_deg=roll_deg, slow_radius_mm=1e-3, stop_radius_mm=-1.0)
        self.loop_arc, self.floor_arc, self.pull, self.branch, self.flipped_cos = loop_arc, floor_arc, pull, branch, flipped_cos

    def reset(self) -> None:
        self.wire.reset()
        self.phase, self.best, self.log = "advance", np.inf, []

    def act(self, obs: dict) -> np.ndarray:
        s = np.asarray(obs["state"], float)
        a = np.asarray(self.wire.act(obs), np.float32).copy()
        a[2] = a[3] = 0.0
        if self.wire.phase == "roll":
            return a
        tip, tdir = s[0:3], quat_z(s[3:7])
        b, arc = where(tip)
        dist = float(np.linalg.norm(s[36:39])) * 1000.0
        toward_root = float(tdir @ aorta_dir_world(arc)) if b == "Aorta" else np.nan
        prev = self.phase
        if self.phase == "advance" and b == "Aorta" and arc < self.loop_arc:
            self.phase = "loop"
        if self.phase == "loop" and ((b == "Aorta" and toward_root < self.flipped_cos) or (b == "Aorta" and arc < self.floor_arc)):
            self.phase = "hook"  # the tip turned back (or we are near the open root): start pulling
        if self.phase == "hook" and b == self.branch:
            self.phase = "drive"
        if self.phase == "drive":
            if dist < self.best - 0.2:
                self.best = dist
            elif dist > self.best + 3.0 or dist < 2.5:
                self.phase = "hold"
        if self.phase != prev:
            self.log.append({"phase": self.phase, "ins_mm": round(float(s[GW_INSERTION]) * 1000, 1), "tip_in": b,
                             "arc_mm": round(arc, 1), "toward_root": None if np.isnan(toward_root) else round(toward_root, 2),
                             "dist_mm": round(dist, 1)})
        if self.phase in ("advance", "loop"):
            a[0], a[1] = 1.0 if self.phase == "advance" else 0.5, a[1] * 0.5
        elif self.phase in ("hook", "drive"):
            a[0], a[1] = -self.pull, 0.0
        else:
            a[:] = 0.0
        return a


def episode(worker: int, roll: float, args, controller=None, record=None) -> dict:
    env = make_env(worker=worker, seed=worker, goals=[GOAL_M.tolist()], max_steps=args.max_steps, visible=args.visible,
                   scenario_name=SCEN, scenario_root=(ROOT / "data" / "scenarios").resolve())
    guard = wall_guard_for(SCEN, ROOT / "data" / "scenarios")
    ctl = controller or EntryRollExpert(roll_deg=roll, slow_radius_mm=1e-3, stop_radius_mm=-1.0)
    if controller is None and args.cath:
        from stroke_nav.agents.catheter_support import CatheterSupportedRoll

        ctl = CatheterSupportedRoll(ctl, lead_mm=args.cath_lead, park_mm=1e9)  # the catheter follows all the way
    rows, frames = [], []
    try:
        obs, _ = env.reset(seed=args.seed)
        ctl.reset()
        for step in range(args.max_steps):
            a = ctl.act(obs)
            obs, _, term, trunc, info = env.step(a)
            t = env.bridge.true["guidewire"][-1]
            b, arc = where(t["tip"])
            c = guard.check(guidewire_bodies(t), t["ins"] * 1000.0)
            rows.append({"step": step + 1, "ins": round(t["ins"] * 1000, 1), "tip_in": b, "arc": round(arc, 1),
                         "dist": round(info["eval/true_distance_m"] * 1000, 1), "cross": c.depth_mm if c.crossing else 0.0,
                         "phase": getattr(ctl, "phase", "")})
            if record is not None:
                from demo_lockstep import compose  # noqa: E402

                frames.append(compose(env.bridge.physics.screenshot(), env.bridge.last_viewer_jpeg,
                                      [getattr(ctl, "label", f"loop and hook, entry roll {roll:g} deg"), f"phase {rows[-1]['phase']}",
                                       f"tip in {b} @{arc:.0f} mm", f"distance to goal {rows[-1]['dist']:.1f} mm",
                                       f"guidewire in {rows[-1]['ins']:.0f} mm", f"wall crossed {rows[-1]['cross']:.1f} mm"],
                                      "GOAL REACHED" if info.get("success") else None))
            if term or trunc or getattr(ctl, "phase", "") == "hold" or (controller is None and t["ins"] * 1000 > args.push_mm):
                break
    finally:
        env.close()
    if record is not None and frames:
        import imageio.v2 as imageio

        Path(record).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(record, frames + [frames[-1]] * 20, fps=10, quality=8, macro_block_size=1)
    asc = [r["arc"] for r in rows if r["tip_in"] == "Aorta"]
    return {"roll": roll, "steps": len(rows), "end_in": rows[-1]["tip_in"], "end_arc": rows[-1]["arc"],
            "min_aorta_arc": min(asc) if asc else None, "min_dist": min(r["dist"] for r in rows), "final_dist": rows[-1]["dist"],
            "cross_steps": sum(r["cross"] > 0 for r in rows), "max_cross_mm": max(r["cross"] for r in rows),
            "success": bool(info.get("success")), "log": getattr(ctl, "log", []), "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("sweep", "run"))
    ap.add_argument("--rolls", default="0:360:30")
    ap.add_argument("--roll", type=float, default=90.0)
    ap.add_argument("--loop-arc", type=float, default=35.0)
    ap.add_argument("--floor-arc", type=float, default=22.0)
    ap.add_argument("--pull", type=float, default=0.4)
    ap.add_argument("--push-mm", type=float, default=230.0, help="sweep: push this far")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--cath", action="store_true", help="sweep with the catheter following the wire (support)")
    ap.add_argument("--cath-lead", type=float, default=25.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--visible", action="store_true")
    ap.add_argument("--record", default=None)
    args = ap.parse_args()
    kill_stray_pages()
    out = ROOT / "results" / "loop_hook"
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == "sweep":
        a, b, c = (float(v) for v in args.rolls.split(":"))
        rolls = list(np.arange(a, b, c))
        n = min(args.workers, len(rolls))
        with ThreadPoolExecutor(n) as ex:
            res = list(ex.map(lambda w: [episode(w, r, args) for r in rolls[w::n]], range(n)))
        res = sorted((r for part in res for r in part), key=lambda r: r["roll"])
        for r in res:
            print(f"roll {r['roll']:5.0f}: ends in {r['end_in']:24s} @{r['end_arc']:5.0f} | deepest into the ascending aorta: Aorta@"
                  f"{r['min_aorta_arc']} mm | closest to goal {r['min_dist']:5.1f} mm | wall crossed {r['cross_steps']} steps")
        (out / ("sweep_cath.json" if args.cath else "sweep.json")).write_text(json.dumps([{k: v for k, v in r.items() if k != "rows"} for r in res], indent=1))
    else:
        r = episode(0, args.roll, args, LoopHook(args.roll, args.loop_arc, args.floor_arc, args.pull), args.record)
        for e in r["log"]:
            print("  ", e)
        print({k: v for k, v in r.items() if k not in ("rows", "log")})
        (out / f"run_roll{args.roll:g}.json").write_text(json.dumps(r))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.exit(main())
