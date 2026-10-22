"""Run an action sequence on the organiser's live stack and get the recorded device states back.

The ROS side (``probe_ros.py``) runs inside the EndoNav ros2-bridge container, which has rclpy and
the catforce message types. This module copies it in with ``docker cp``, runs it and parses the CSV.
"""

from __future__ import annotations

import csv
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

BRIDGE_CONTAINER = "hcmr-endonavsimdocker-ros2-bridge-1"
PROBE_SCRIPT = Path(__file__).with_name("probe_ros.py")
ROS_ENV = "source /opt/ros/jazzy/setup.bash; source /workspace/install/setup.bash; export ROS_DOMAIN_ID=42"
ACTION_KEYS = ("gw_trans", "gw_rot", "cath_trans", "cath_rot")


def _docker(*args: str, timeout: float = 600.0) -> subprocess.CompletedProcess:
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout, env=env)


def run_probe(actions: np.ndarray, dt: float = 0.2, container: str = BRIDGE_CONTAINER, home: bool = True,
              log=print) -> dict[str, np.ndarray]:
    """Play ``actions`` (N x 4, normalised [-1, 1]) on their robot; return the recording as columns."""
    actions = np.asarray(actions, dtype=float).reshape(-1, 4)
    with tempfile.TemporaryDirectory() as tmp:
        a_csv, o_csv = Path(tmp) / "actions.csv", Path(tmp) / "out.csv"
        with open(a_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(ACTION_KEYS)
            w.writerows(np.round(actions, 6).tolist())
        for src, dst in ((PROBE_SCRIPT, "/tmp/stroke_nav_probe.py"), (a_csv, "/tmp/stroke_nav_actions.csv")):
            r = _docker("cp", str(src), f"{container}:{dst}")
            if r.returncode:
                raise RuntimeError(f"docker cp failed: {r.stderr.strip()}")
        cmd = (f"{ROS_ENV}; python3 /tmp/stroke_nav_probe.py /tmp/stroke_nav_actions.csv /tmp/stroke_nav_out.csv "
               f"--dt {dt}" + ("" if home else " --no-home"))
        r = _docker("exec", container, "bash", "-c", cmd, timeout=len(actions) * dt + 300)
        for line in (r.stdout + r.stderr).splitlines():
            if line.strip():
                log(f"  [probe] {line}")
        if r.returncode:
            raise RuntimeError(f"probe failed (exit {r.returncode})")
        r = _docker("cp", f"{container}:/tmp/stroke_nav_out.csv", str(o_csv))
        if r.returncode:
            raise RuntimeError(f"could not fetch the recording: {r.stderr.strip()}")
        return read_recording(o_csv)


def read_recording(path: str | Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    keys = rows[0].keys() if rows else []
    out = {}
    for k in keys:
        out[k] = np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows])
    return out


def sample_at_steps(rec: dict[str, np.ndarray], n_steps: int, dt: float = 0.2) -> dict[str, np.ndarray]:
    """The recording at the end of each control step (t = (k+1) * dt), like an RL observation."""
    t = rec["t_s"]
    tq = (np.arange(n_steps) + 1) * dt
    idx = np.clip(np.searchsorted(t, tq), 0, len(t) - 1)
    return {k: v[idx] for k, v in rec.items()}
