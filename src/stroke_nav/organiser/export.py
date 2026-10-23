"""Write work in the organiser's formats.

* ``write_run_dir``: a folder that drops into ``HCMR-EV-RL/runs/<run_id>/`` and passes their
  evaluation checks (resolved-config.json schema 1, device-parameters.json + sha256, scenario.json
  identical to their cas-reference manifest, final-model.zip). Their showcase/evaluation command
  then works on it unchanged.
* ``trajectory_device_commands`` / ``trajectory_twists``: a planned joint trajectory converted to
  their ROS 2 control interfaces (``catforce_interfaces/DeviceCommand`` absolute setpoints in mm/deg,
  or Twist velocities in mm/s and rad/s on ``/control/autonomous/module_{1,2}/cmd_vel``).
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from stroke_nav.organiser.frame import OrganiserFrame, write_goals_txt
from stroke_nav.organiser.spec import CAS_REFERENCE_MANIFEST, DEFAULT_SPEC, OrganiserSpec, device_parameters_hash

BRIDGE_LIMITS = {"translation_mm_s": 20.0, "rotation_rad_s": 5.0}  # ros_to_udp clamp (protocol.hpp)


def new_run_id(prefix: str = "stroke-nav-sac") -> str:
    return datetime.now(timezone.utc).strftime(f"{prefix}-%Y%m%d-%H%M%S-%fZ")


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover
        return None


def write_run_dir(run_dir: str | Path, spec: OrganiserSpec = DEFAULT_SPEC, model_path: str | Path | None = None,
                  frame: OrganiserFrame | None = None, extra: dict | None = None,
                  scenario_manifest: dict | None = None) -> Path:
    """Create/complete an organiser-format run folder. ``model_path`` is copied to final-model.zip.
    ``scenario_manifest``: the scenario the run is evaluated on (their ``scenario.json``; cas-reference
    by default, another one for a deployment on a variant mesh)."""
    manifest = scenario_manifest or CAS_REFERENCE_MANIFEST
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device_doc = spec.device_parameters()
    (run_dir / "device-parameters.json").write_text(json.dumps(device_doc, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "scenario.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    cfg = spec.online_rl_config()
    resolved = {
        "action": cfg["action"],
        "device_parameters": {"file": "device-parameters.json", "sha256": device_parameters_hash(device_doc),
                              "source": "training_startup"},
        "environment": cfg["environment"],
        "goals_file": cfg["goals_file"],
        "run_id": run_dir.name,
        "scenario": {"mesh_sha256": manifest["mesh_sha256"], "name": manifest["name"]},
        "schema_version": 1,
        "training": cfg["training"],
    }
    (run_dir / "resolved-config.json").write_text(json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8")
    if model_path is not None and Path(model_path).resolve() != (run_dir / "final-model.zip").resolve():
        shutil.copyfile(model_path, run_dir / "final-model.zip")
    (run_dir / "latest-run.txt").write_text(run_dir.name + "\n", encoding="utf-8")
    provenance = {
        "producer": "stroke-nav (kinematic simulator on VMR 0248, organiser-compatible interface)",
        "git_commit": _git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frame": (frame.to_dict() if frame is not None else None),
        **(extra or {}),
    }
    (run_dir / "stroke-nav.json").write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    return run_dir


def check_run_dir(run_dir: str | Path, scenario_manifest: dict | None = None,
                  model: str = "final-model.zip") -> list[str]:
    """Problems that would make the organiser's evaluation reject this run folder (empty = OK).
    ``scenario_manifest``: the scenario it must match (cas-reference by default)."""
    manifest = scenario_manifest or CAS_REFERENCE_MANIFEST
    run_dir = Path(run_dir)
    problems = []
    for name in ("resolved-config.json", "device-parameters.json", "scenario.json", model):
        if not (run_dir / name).is_file():
            problems.append(f"missing {name}")
    if problems:
        return problems
    cfg = json.loads((run_dir / "resolved-config.json").read_text(encoding="utf-8"))
    if cfg.get("schema_version") != 1:
        problems.append("resolved-config.json schema_version must be 1")
    dev = json.loads((run_dir / "device-parameters.json").read_text(encoding="utf-8"))
    try:
        h = device_parameters_hash(dev)
        if cfg.get("device_parameters", {}).get("sha256") != h:
            problems.append("device-parameters.json hash differs from resolved-config.json")
    except ValueError as e:
        problems.append(f"device-parameters.json invalid: {e}")
    scen = json.loads((run_dir / "scenario.json").read_text(encoding="utf-8"))
    if scen != manifest:
        problems.append(f"scenario.json differs from the {manifest['name']} manifest")
    if cfg.get("scenario", {}).get("mesh_sha256") != manifest["mesh_sha256"]:
        problems.append("resolved-config.json scenario mesh hash differs")
    if cfg.get("scenario", {}).get("name") != manifest["name"]:
        problems.append("resolved-config.json scenario name differs")
    return problems


# ---------------------------------------------------------------------- trajectories
def _resample(t: np.ndarray, y: np.ndarray, rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    tt = np.arange(0.0, float(t[-1]) + 1e-9, 1.0 / rate_hz)
    return tt, np.interp(tt, t, y)


def trajectory_device_commands(traj, rate_hz: float = 20.0) -> list[dict]:
    """Rows for ``/endovascular/{guidewire,catheter}/command`` (DeviceCommand: absolute mm / deg).

    Insertion is measured from our start point, which coincides with their device entry (origin)
    to within 2 mm; their follower also latches the first joint state as zero.
    """
    t = traj.t
    rows = []
    frames = int(round(60.0 / rate_hz))  # simulator advances in fixed 60 Hz frames
    for device, L, phi in (("guidewire", traj.L_wire, np.unwrap(traj.phi_wire)), ("catheter", traj.L_cath, np.unwrap(traj.phi_cath))):
        tt, Lr = _resample(t, L, rate_hz)
        _, pr = _resample(t, np.degrees(phi), rate_hz)
        vL = np.gradient(Lr, tt) if len(tt) > 1 else np.zeros_like(Lr)
        vp = np.gradient(pr, tt) if len(tt) > 1 else np.zeros_like(pr)
        for i in range(len(tt)):
            rows.append({"t_s": round(float(tt[i]), 4), "device": device,
                         "insertion_setpoint_mm": round(float(Lr[i]), 3), "rotation_setpoint_deg": round(float(pr[i]), 3),
                         "insertion_velocity_mm_s": round(float(vL[i]), 3), "rotation_velocity_deg_s": round(float(vp[i]), 3),
                         "frames": frames, "enabled": True})
    rows.sort(key=lambda r: (r["t_s"], r["device"]))
    return rows


def trajectory_twists(traj, rate_hz: float = 5.0, spec: OrganiserSpec = DEFAULT_SPEC) -> tuple[list[dict], dict]:
    """Velocity commands per module for ``/control/autonomous/module_{1,2}/cmd_vel``
    (linear.x mm/s, angular.z rad/s) plus a report of limit violations."""
    t = traj.t
    tt = np.arange(0.0, float(t[-1]) + 1e-9, 1.0 / rate_hz)
    out, report = [], {"rate_hz": rate_hz, "max_abs": {}, "exceeds_rl_action_limit": False, "exceeds_bridge_limit": False}
    series = {}
    for module, L, phi in ((1, traj.L_wire, np.unwrap(traj.phi_wire)), (2, traj.L_cath, np.unwrap(traj.phi_cath))):
        Lr = np.interp(tt, t, L)
        pr = np.interp(tt, t, phi)
        v = np.diff(Lr, append=Lr[-1]) * rate_hz  # mm/s held over the next interval
        w = np.diff(pr, append=pr[-1]) * rate_hz  # rad/s
        series[module] = (v, w)
        report["max_abs"][f"module_{module}"] = {"linear_x_mm_s": float(np.abs(v).max()), "angular_z_rad_s": float(np.abs(w).max())}
        if np.abs(v).max() > spec.action.max_translation_m_s * 1000 + 1e-6 or np.abs(w).max() > spec.action.max_rotation_rad_s + 1e-6:
            report["exceeds_rl_action_limit"] = True
        if np.abs(v).max() > BRIDGE_LIMITS["translation_mm_s"] + 1e-6 or np.abs(w).max() > BRIDGE_LIMITS["rotation_rad_s"] + 1e-6:
            report["exceeds_bridge_limit"] = True
    for i, ti in enumerate(tt):
        for module in (1, 2):
            v, w = series[module]
            out.append({"t_s": round(float(ti), 4), "module": module, "topic": f"/control/autonomous/module_{module}/cmd_vel",
                        "linear_x_mm_s": round(float(v[i]), 4), "angular_z_rad_s": round(float(w[i]), 5)})
    return out, report


def trajectory_policy_actions(traj, spec: OrganiserSpec = DEFAULT_SPEC) -> np.ndarray:
    """The same trajectory as normalised RL actions [-1,1]^4 at their 5 Hz control rate."""
    rows, _ = trajectory_twists(traj, spec.environment.control_hz, spec)
    n = len(rows) // 2
    a = np.zeros((n, 4))
    for i in range(n):
        m1, m2 = rows[2 * i], rows[2 * i + 1]
        a[i] = [m1["linear_x_mm_s"] / (spec.action.max_translation_m_s * 1000), m1["angular_z_rad_s"] / spec.action.max_rotation_rad_s,
                m2["linear_x_mm_s"] / (spec.action.max_translation_m_s * 1000), m2["angular_z_rad_s"] / spec.action.max_rotation_rad_s]
    return np.clip(a, -1.0, 1.0)


def write_csv(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def write_goal_for_target(path: str | Path, frame: OrganiserFrame, vmr_point_mm, label: str = "") -> Path:
    """A goals.txt (their mesh cm) for a target chosen in our anatomy."""
    return write_goals_txt(path, frame.vmr_mm_to_obj_cm(np.asarray(vmr_point_mm)[None]), label)
