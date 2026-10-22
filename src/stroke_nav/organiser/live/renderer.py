"""Run the organiser's browser physics page headless and inspect it.

Their page (``catheter_guidewire_ros2.html``) runs the XPBD rod physics in a
``requestAnimationFrame`` loop, which browsers pause in hidden tabs. Their RL tooling
(``HCMR-EV-RL/scripts/headless-renderer``) therefore launches Chrome headless with background
throttling disabled; ``start_physics_page`` uses the same flags. Without ``scenario_url`` the page
loads its default scenario ``meshes/Mesh_1/0248_H_AOCERE_CAS_reference_10k.obj`` + ``goal.txt``
(identical to the cas-reference scenario).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

CHROME_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium",
)
RENDERER_FLAGS = (  # HCMR-EV-RL/scripts/headless-renderer
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
    "--enable-gpu", "--enable-webgl", "--ignore-gpu-blocklist",
    "--no-first-run", "--window-size=1400,1000",
)
PAGE = "http://localhost:{port}/catheter_guidewire_ros2.html?ws_port_base=18765&main_port={port}&noisy_port=8001"


def _chrome() -> str:
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
    raise FileNotFoundError("Chrome or Edge not found")


class PhysicsPage:
    """Handle to the physics page running in a dedicated (headless) Chrome."""

    def __init__(self, debug_port: int = 9222, page_port: int = 8000):
        self.debug_port = debug_port
        self.page_port = page_port

    # ------------------------------------------------------------------ lifecycle
    def start(self, headless: bool = True, profile: str | None = None, timeout_s: float = 60.0) -> "PhysicsPage":
        if self._target() is None:
            profile = profile or os.path.join(os.environ.get("LOCALAPPDATA", "/tmp"), "stroke-nav", "chrome-physics-profile")
            os.makedirs(profile, exist_ok=True)
            args = [_chrome(), *RENDERER_FLAGS, f"--remote-debugging-port={self.debug_port}", f"--user-data-dir={profile}"]
            if headless:
                args.append("--headless=new")
            args.append(PAGE.format(port=self.page_port))
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
        self.wait_ready(timeout_s)
        return self

    def stop(self) -> None:
        t = self._target()
        if t is not None:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.debug_port}/json/close/{t['id']}", timeout=5).read()
            except Exception:
                pass

    def reload(self, timeout_s: float = 60.0) -> None:
        self.evaluate("location.reload(); true")
        time.sleep(2.0)
        self.wait_ready(timeout_s)

    def wait_ready(self, timeout_s: float = 60.0) -> dict:
        """Wait until the page has loaded its mesh and goal."""
        deadline = time.monotonic() + timeout_s
        last = {}
        while time.monotonic() < deadline:
            try:
                last = self.state()
                if last.get("mesh_loaded"):
                    return last
            except Exception as e:  # page not up yet
                last = {"error": str(e)}
            time.sleep(1.0)
        raise TimeoutError(f"physics page not ready: {last}")

    # ------------------------------------------------------------------ inspection
    def state(self) -> dict:
        js = r"""(() => {
          const t = id => (document.getElementById(id) || {}).textContent || '';
          const info = t('meshInfo');
          return {mesh_info: info.trim(), mesh_loaded: /triangles/.test(info),
                  goal_info: t('goalInfo').trim(),
                  guidewire_insertion: t('depthOut'), guidewire_roll: t('rollOut'),
                  catheter_insertion: t('cathDepthOut'), catheter_roll: t('cathRollOut')};
        })()"""
        return self.evaluate(js)

    def evaluate(self, expression: str, timeout_s: float = 15.0):
        import websocket  # websocket-client

        t = self._target()
        if t is None:
            raise RuntimeError("physics page is not open in the debug-port Chrome")
        ws = websocket.create_connection(t["webSocketDebuggerUrl"], timeout=timeout_s, suppress_origin=True)
        try:
            ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate",
                                "params": {"expression": expression, "returnByValue": True, "awaitPromise": True}}))
            while True:
                msg = json.loads(ws.recv())
                if msg.get("id") == 1:
                    break
        finally:
            ws.close()
        result = msg.get("result", {})
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"].get("text", "evaluation failed"))
        return result.get("result", {}).get("value")

    def _target(self) -> dict | None:
        try:
            targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{self.debug_port}/json/list", timeout=2).read())
        except Exception:
            return None
        for t in targets:
            if t.get("type") == "page" and "catheter_guidewire_ros2.html" in t.get("url", ""):
                return t
        return None
