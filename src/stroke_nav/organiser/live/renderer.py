"""Run the organiser's browser physics page headless and inspect it.

Their page (``catheter_guidewire_ros2.html``) runs the XPBD rod physics in a
``requestAnimationFrame`` loop, which browsers pause in hidden tabs. Their RL tooling
(``HCMR-EV-RL/scripts/headless-renderer``) therefore launches Chrome headless with background
throttling disabled; ``start_physics_page`` uses the same flags. Without ``scenario_url`` the page
loads its default scenario ``meshes/Mesh_1/0248_H_AOCERE_CAS_reference_10k.obj`` + ``goal.txt``
(identical to the cas-reference scenario).

Their RL env also needs the noisy viewer page (``noisy_visualization_ros2.html`` on port 8001),
which renders the "fluoro" image published on ``/endovascular/imaging/noisy/compressed``. Opened on
its own it renders the devices without the vessel and from the wrong camera: it takes mesh scale,
start frame and camera from the physics page. ``RLRenderer`` reproduces their RL renderer instead:
their scenario asset server (``openng3_rl.asset_server``, stdlib + NumPy, run from the organiser
sources) on port 8002, then the physics page and the viewer in two separate headless Chromes (debug
ports 9222 and 9223), both opened with ``scenario_url``. The viewer then syncs through the asset
server, crops the square RL image, and the runtime status file holds the mesh-to-world transform
their goal sampler needs. (Opening the viewer as a popup of the physics page also syncs it, but puts
the physics page in a background tab, which drops its physics to about 11 frames/s.)
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
HCMR_EV = Path(os.environ.get("HCMR_EV_ROOT", REPO.parent / "HCMR-EV"))  # organiser sources (siblings)

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
PHYSICS_HTML = "catheter_guidewire_ros2.html"
NOISY_HTML = "noisy_visualization_ros2.html"
PAGE = "http://localhost:{port}/{html}?ws_port_base=18765&main_port=8000&noisy_port=8001"
ASSET_PORT = 8002


def _chrome() -> str:
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
    raise FileNotFoundError("Chrome or Edge not found")


class PhysicsPage:
    """Handle to the physics page (or, with ``html=NOISY_HTML``, the noisy viewer) running in a
    dedicated (headless) Chrome."""

    def __init__(self, debug_port: int = 9222, page_port: int = 8000, html: str = PHYSICS_HTML):
        self.debug_port = debug_port
        self.page_port = page_port
        self.html = html

    # ------------------------------------------------------------------ lifecycle
    def start(self, headless: bool = True, profile: str | None = None, timeout_s: float = 60.0,
              query: str = "") -> "PhysicsPage":
        """Open the page unless it is already open on this debug port; ``query`` is appended to its URL."""
        if self._target() is None:
            profile = profile or os.path.join(os.environ.get("LOCALAPPDATA", "/tmp"), "stroke-nav",
                                   "chrome-physics-profile" if self.html == PHYSICS_HTML else "chrome-noisy-profile")
            os.makedirs(profile, exist_ok=True)
            args = [_chrome(), *RENDERER_FLAGS, f"--remote-debugging-port={self.debug_port}", f"--user-data-dir={profile}"]
            if headless:
                args.append("--headless=new")
            args.append(PAGE.format(port=self.page_port, html=self.html) + query)
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
        raise TimeoutError(f"{self.html} not ready: {last}")

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

    def evaluate(self, expression: str, timeout_s: float = 15.0, user_gesture: bool = False):
        import websocket  # websocket-client

        t = self._target()
        if t is None:
            raise RuntimeError(f"{self.html} is not open in the debug-port Chrome")
        ws = websocket.create_connection(t["webSocketDebuggerUrl"], timeout=timeout_s, suppress_origin=True)
        try:
            ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate",
                                "params": {"expression": expression, "returnByValue": True, "awaitPromise": True,
                                           "userGesture": user_gesture}}))
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
            if t.get("type") == "page" and self.html in t.get("url", ""):
                return t
        return None



class RLRenderer:
    """Their RL renderer: scenario asset server + physics page + noisy viewer, all headless.

    ``status()`` is the asset server's runtime status (``physical_ready``, ``noisy_ready``,
    ``settings_synced``, ``mesh_to_world``, ``device_parameters_hash``), written to ``status_file``.
    """

    def __init__(self, scenario: str = "cas-reference", device_parameters: str | Path | None = None,
                 status_file: str | Path | None = None):
        self.scenario = scenario
        from stroke_nav.organiser.spec import device_parameters_path

        self.device_parameters = Path(device_parameters or device_parameters_path())  # follows STROKE_NAV_ROBOT
        self.status_file = Path(status_file or REPO / "runs" / "live" / "scenario-runtime.json")
        self.physics = PhysicsPage(9222, 8000, PHYSICS_HTML)
        self.viewer = PhysicsPage(9223, 8001, NOISY_HTML)
        self.server: subprocess.Popen | None = None

    def start(self, timeout_s: float = 90.0) -> "RLRenderer":
        if not (HCMR_EV / "HCMR-EV-RL" / "src" / "openng3_rl").is_dir():
            raise FileNotFoundError(f"organiser sources not found under {HCMR_EV} (set HCMR_EV_ROOT)")
        if self.physics._target() is not None or self.viewer._target() is not None:
            raise RuntimeError("a physics page or noisy viewer is already open on debug port 9222/9223; "
                               "close it first so both pages load the same scenario")
        if self._health():
            raise RuntimeError(f"port {ASSET_PORT} is already serving; stop that server first")
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        self.status_file.unlink(missing_ok=True)
        rl = HCMR_EV / "HCMR-EV-RL"
        env = dict(os.environ, PYTHONPATH=str(rl / "src"))
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.server = subprocess.Popen(
            [sys.executable, "-m", "openng3_rl.asset_server", "--scenario", self.scenario,
             "--scenario-root", str(rl / "scenarios"), "--status-file", str(self.status_file),
             "--host", "127.0.0.1", "--port", str(ASSET_PORT), "--device-parameters", str(self.device_parameters)],
            cwd=str(rl), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
        deadline = time.monotonic() + timeout_s
        while not self._health():
            if self.server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("scenario asset server did not start")
            time.sleep(0.2)
        query = f"&scenario_url=http://localhost:{ASSET_PORT}/scenario.json&renderer_session=stroke-nav-{int(time.time())}"
        self.physics.start(query=query, timeout_s=timeout_s)
        self._wait("physical_ready", deadline)
        self.viewer.start(query=query, timeout_s=max(1.0, deadline - time.monotonic()))
        for attempt in range(3):  # the viewer's first scenario fetch sometimes fails ("Failed to fetch"): reload
            try:
                self._wait("settings_synced", min(deadline, time.monotonic() + 20.0))
                break
            except TimeoutError:
                if attempt == 2:
                    raise
                self.viewer.evaluate("location.reload(); true")
        return self

    def status(self) -> dict:
        try:
            return json.loads(self.status_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def stop(self) -> None:
        self.viewer.stop()
        self.physics.stop()
        if self.server is not None and self.server.poll() is None:
            self.server.terminate()

    def _wait(self, field: str, deadline: float) -> None:
        while not self.status().get(field):
            if time.monotonic() > deadline:
                raise TimeoutError(f"renderer not ready: {field} is false; status {self.status()}")
            time.sleep(0.5)

    @staticmethod
    def _health() -> bool:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{ASSET_PORT}/health", timeout=1).read()
            return True
        except Exception:
            return False
