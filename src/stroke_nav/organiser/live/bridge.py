"""Closed-loop access to the organiser's live simulator from the host.

``LiveBridge`` keeps ``relay_ros.py`` running inside the EndoNav ros2-bridge container
(``docker exec -i``) and exchanges one JSON line per request: reset, step(action), close.
Each step holds the normalised action for one control period (0.2 s, 5 Hz) and returns the latest
true and noisy device states, so controllers on the host (planner followers, policies) can run
against their XPBD physics in real time.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

from stroke_nav.organiser.live.probe import BRIDGE_CONTAINER, ROS_ENV, _docker

RELAY_SCRIPT = Path(__file__).with_name("relay_ros.py")


class LiveBridge:
    def __init__(self, container: str = BRIDGE_CONTAINER, dt: float = 0.2):
        self.container = container
        self.dt = dt
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> "LiveBridge":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    def open(self) -> "LiveBridge":
        r = _docker("cp", str(RELAY_SCRIPT), f"{self.container}:/tmp/stroke_nav_relay.py")
        if r.returncode:
            raise RuntimeError(f"docker cp failed: {r.stderr.strip()}")
        env = dict(os.environ, MSYS_NO_PATHCONV="1")
        self.proc = subprocess.Popen(
            ["docker", "exec", "-i", self.container, "bash", "-c", f"{ROS_ENV}; exec python3 -u /tmp/stroke_nav_relay.py"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env)
        return self

    def _request(self, req: dict) -> dict:
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("relay is not running")
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("relay exited")
        rep = json.loads(line)
        if not rep.get("ok"):
            raise RuntimeError(rep.get("error", "relay error"))
        return rep

    def reset(self) -> dict:
        rep = self._request({"op": "reset"})
        return rep["obs"] | {"message": rep.get("message", "")}

    def step(self, action) -> dict:
        a = np.clip(np.asarray(action, dtype=float).reshape(4), -1.0, 1.0)
        return self._request({"op": "step", "action": a.tolist(), "dt": self.dt})["obs"]

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                self._request({"op": "close"})
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.proc = None
