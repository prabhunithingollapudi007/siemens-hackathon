"""Lockstep RL env on the organiser's own physics, without their ROS stack or wall clock.

Their live stack runs the XPBD physics in a browser page against a wall-clock robot; on this laptop
the page manages about 0.18x real time with their containers running, so their ``OpenNG3Env``
refuses to start (images below 4.5 Hz) and every rollout is distorted. Here the same pages run
headless with ``lockstep_page.js`` injected: we run their animation frames ourselves (each is
1/60 s of simulated time) and speak their ROS-bridge WebSocket protocol in-page. What stays theirs:

* physics: ``catheter_guidewire_ros2.html`` with the cas-reference scenario and the device
  parameters from their asset server (``openng3_rl.asset_server``, fed ``device-parameters.json``)
* image: ``noisy_visualization_ros2.html`` in RL mode (square 480 px JPEG, decoded to 128x128 grey
  exactly as ``openng3_rl.ros_io``)
* env, observation, reward, success, goals: their ``openng3_rl`` package, imported from the
  organiser sources under the alias ``hcmr_openng3_rl`` (it does not clash with ``compat/``)

What is emulated (``LockstepBridge``), from their sources and our measurements:

* Monadnock robot: joint velocity = command after a 50 ms delay, arbiter deadband 0.05 mm/s and
  0.05 rad/s, module 1 travel 1.200-1.4645 m, module 2 from 0.900 m and >= 70.8 mm behind module 1
  (``spec.RobotSpec``); the joint-state follower sends joint - home as setpoints (mm, deg).
* ``sensor_sim/noise_injection_node.py``: per-sample Gaussian noise, 7 % tip dropout (hold last
  vision), 50-150 ms release latency in true-time order.
* ``openng3_rl.ros_io`` timing: commands are held until the 5 Hz tick; the observation is the first
  image after it (an image is rendered when a noisy guidewire sample is released, at most every
  50 ms, stamped with that sample's true time) paired with the catheter sample closest in true time
  (skew <= 0.1 s). The reward uses the latest true guidewire state.

Simulated time is the page's ``simulated_time``; the host's wall clock plays no part.
"""

from __future__ import annotations

import base64
import functools
import importlib
import importlib.util
import io
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from stroke_nav.organiser.live.renderer import HCMR_EV, PHYSICS_HTML, NOISY_HTML, REPO, RENDERER_FLAGS, _chrome
from stroke_nav.organiser.spec import DEFAULT_SPEC

INIT_SCRIPT = Path(__file__).with_name("lockstep_page.js")
FRAME_DT = 1.0 / 60.0  # the physics page's fixed step
STATE_EVERY_FRAMES = 3  # it publishes device states every 3rd frame (20 Hz)
WS_PORT_BASE = 18765
ALIAS = "hcmr_openng3_rl"

PHYSICS_WEB = HCMR_EV / "HCMR-EndoNavSimDocker" / "src" / "catforce_ros_bridge" / "web"
VIEWER_WEB = HCMR_EV / "HCMR-EndoNavSimDocker" / "src" / "sensor_sim" / "web"
RL_ROOT = HCMR_EV / "HCMR-EV-RL"


# ---------------------------------------------------------------------- their package
def organiser_rl(module: str = ""):
    """Import ``openng3_rl[.module]`` from the organiser sources, as ``hcmr_openng3_rl``."""
    if ALIAS not in sys.modules:
        pkg = RL_ROOT / "src" / "openng3_rl"
        if not (pkg / "__init__.py").is_file():
            raise FileNotFoundError(f"organiser RL sources not found at {pkg} (set HCMR_EV_ROOT)")
        spec = importlib.util.spec_from_file_location(ALIAS, pkg / "__init__.py", submodule_search_locations=[str(pkg)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[ALIAS] = mod
        spec.loader.exec_module(mod)
    return importlib.import_module(f"{ALIAS}.{module}" if module else ALIAS)


# ---------------------------------------------------------------------- servers
@dataclass(frozen=True)
class Ports:
    web: int = 8100  # physics page; the noisy viewer is served on web + 1
    asset: int = 8102  # their scenario asset server


def _serving(port: int, path: str = "/") -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=1).read()
        return True
    except Exception:
        return False


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass


_SERVER_LOCK = threading.Lock()


def ensure_servers(ports: Ports = Ports(), scenario: str = "cas-reference",
                   device_parameters: Path | None = None) -> Path:
    """Serve both pages and their asset server in this process unless something already serves
    them (e.g. the parent of a worker process). Returns the asset server's status file."""
    with _SERVER_LOCK:
        return _ensure_servers(ports, scenario, device_parameters)


def _ensure_servers(ports: Ports, scenario: str, device_parameters: Path | None) -> Path:
    status = REPO / "runs" / "live" / f"lockstep-runtime-{ports.asset}.json"
    for port, root in ((ports.web, PHYSICS_WEB), (ports.web + 1, VIEWER_WEB)):
        if not _serving(port, f"/{PHYSICS_HTML if root == PHYSICS_WEB else NOISY_HTML}"):
            srv = ThreadingHTTPServer(("127.0.0.1", port), functools.partial(_QuietHandler, directory=str(root)))
            threading.Thread(target=srv.serve_forever, daemon=True).start()
    if not _serving(ports.asset, "/health"):
        asset = organiser_rl("asset_server")
        dp = organiser_rl("device_parameters")
        params = dp.load_device_parameters(Path(device_parameters or os.environ.get("STROKE_NAV_DEVICE_PARAMS")
                                                or REPO / "device-parameters.json"))
        status.parent.mkdir(parents=True, exist_ok=True)

        class Runtime(asset.ScenarioRuntime):
            def _write_status(self) -> None:
                # several pages acknowledge at once; on Windows os.replace fails while another
                # thread or a virus scanner has the file open (WinError 5/32): retry briefly
                for attempt in range(50):
                    try:
                        return super()._write_status()
                    except PermissionError:
                        if attempt == 49:
                            raise
                        time.sleep(0.02)

        runtime = Runtime(scenario, (RL_ROOT / "scenarios").resolve(), status, params, "training_run")
        srv = ThreadingHTTPServer(("127.0.0.1", ports.asset), asset._handler(runtime))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    return status


# ---------------------------------------------------------------------- browser pages
class Cdp:
    """A persistent Chrome DevTools session on one page target."""

    def __init__(self, ws_url: str, timeout_s: float = 120.0):
        import websocket  # websocket-client

        self.ws = websocket.create_connection(ws_url, timeout=timeout_s, suppress_origin=True)
        self.next_id = 0

    def call(self, method: str, params: dict | None = None) -> dict:
        self.next_id += 1
        self.ws.send(json.dumps({"id": self.next_id, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.next_id:
                break
        if "error" in msg:
            raise RuntimeError(f"{method}: {msg['error']}")
        return msg.get("result", {})

    def eval(self, expression: str):
        r = self.call("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True})
        if "exceptionDetails" in r:
            d = r["exceptionDetails"]
            raise RuntimeError(d.get("exception", {}).get("description") or d.get("text", "evaluation failed"))
        return r.get("result", {}).get("value")

    def close(self) -> None:
        try:
            self.ws.close()
        except Exception:
            pass


class LockstepPage:
    """One of their pages in its own Chrome (headless unless ``window`` = (x, y, w, h) is given), with
    the lockstep harness injected. A visible page only redraws when we run its frames."""

    def __init__(self, url: str, debug_port: int, profile: Path, window: tuple[int, int, int, int] | None = None):
        self.url, self.debug_port, self.profile, self.window = url, debug_port, profile, window
        self.proc: subprocess.Popen | None = None
        self.cdp: Cdp | None = None

    def start(self, timeout_s: float = 60.0) -> "LockstepPage":
        self.profile.mkdir(parents=True, exist_ok=True)
        if self.window is None:
            show = ["--headless=new"]
        else:
            x, y, w, h = self.window
            show = [f"--window-position={x},{y}", f"--window-size={w},{h}", "--new-window"]
        args = [_chrome(), *RENDERER_FLAGS, *show, f"--remote-debugging-port={self.debug_port}",
                f"--user-data-dir={self.profile}", "about:blank"]
        self.proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        deadline = time.monotonic() + timeout_s
        target = None
        while target is None:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Chrome on debug port {self.debug_port} did not start")
            try:
                targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{self.debug_port}/json/list", timeout=2).read())
                target = next((t for t in targets if t.get("type") == "page"), None)
            except Exception:
                time.sleep(0.3)
        self.cdp = Cdp(target["webSocketDebuggerUrl"])
        self.cdp.call("Page.enable")
        self.cdp.call("Page.addScriptToEvaluateOnNewDocument", {"source": INIT_SCRIPT.read_text(encoding="utf-8")})
        self.cdp.call("Page.navigate", {"url": self.url})
        return self

    def reload(self) -> None:
        self.cdp.call("Page.navigate", {"url": self.url})

    def wait(self, js_condition: str, timeout_s: float, what: str) -> None:
        deadline = time.monotonic() + timeout_s
        last = None
        while time.monotonic() < deadline:
            try:
                last = self.cdp.eval(js_condition)
                if last is True:
                    return
            except RuntimeError as e:  # page still loading
                last = str(e)
            time.sleep(0.3)
        raise TimeoutError(f"{what} not ready on debug port {self.debug_port}: {last}")

    def mesh_info(self) -> str:
        return self.cdp.eval("(document.getElementById('meshInfo') || {}).textContent || ''") or ""

    def screenshot(self, quality: int = 80) -> bytes:
        """JPEG of the page as currently drawn."""
        r = self.cdp.call("Page.captureScreenshot", {"format": "jpeg", "quality": quality})
        return base64.b64decode(r["data"])

    def close(self) -> None:
        if self.cdp is not None:
            self.cdp.close()
        if self.proc is not None and self.proc.poll() is None:
            if os.name == "nt":  # Chrome's GPU and renderer children otherwise outlive it and keep the debug port
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], capture_output=True)
            else:
                self.proc.kill()


def _mesh_ready_js(sockets: int) -> str:
    return (f"(() => typeof __ls !== 'undefined' && __ls.connected().length === {sockets} && "
            "/triangles/.test((document.getElementById('meshInfo') || {}).textContent || ''))()")


# ---------------------------------------------------------------------- robot + noise
@dataclass(frozen=True)
class RobotModel:
    """Monadnock as seen through the arbiter and the joint-state follower (``spec.RobotSpec``)."""

    home: tuple[float, float, float, float] = (1.200, 0.0, 0.900, 0.0)  # m1 (m), r1 (rad), m2, r2
    travel_m: float = DEFAULT_SPEC.robot.guidewire_travel_mm / 1000.0
    module_gap_m: float = 0.0708  # module 2 stays this far behind module 1
    delay_s: float = 0.05  # command to joint motion, measured (rotation 50-70 ms, translation 40-75 ms)
    deadband_mm_s: float = 0.05  # openng3_rl.ros_io ARBITER_TRANSLATION_ACTIVE_NATIVE
    deadband_rad_s: float = 0.05


class Robot:
    def __init__(self, model: RobotModel = RobotModel()):
        self.m = model
        self.delay_frames = max(0, round(model.delay_s / FRAME_DT))
        self.reset()

    def reset(self) -> None:
        self.q = np.array(self.m.home, dtype=float)
        self.pipeline = deque([np.zeros(4)] * self.delay_frames)
        self.command = np.zeros(4)

    def set_command(self, commands: dict) -> None:
        """``commands``: {module: (linear m/s, angular rad/s)} as ``ActionMapper.to_commands``."""
        v = np.zeros(4)
        for module, (lin, ang) in commands.items():
            i = 2 * (int(module) - 1)
            if abs(lin) * 1000.0 <= self.m.deadband_mm_s and abs(ang) <= self.m.deadband_rad_s:
                lin = ang = 0.0  # inside the deadband the arbiter holds the module still
            v[i], v[i + 1] = lin, ang
        self.command = v

    def advance(self) -> tuple[float, float, float, float]:
        """One physics frame; returns the follower's setpoints (mm, deg, mm, deg)."""
        self.pipeline.append(self.command.copy())
        v = self.pipeline.popleft()
        m1, r1, m2, r2 = self.q + v * FRAME_DT
        h1, _, h2, _ = self.m.home
        m1 = min(max(m1, h1, self.q[2] + self.m.module_gap_m), h1 + self.m.travel_m)  # module 2 blocks retraction
        m2 = min(max(m2, h2), m1 - self.m.module_gap_m)
        self.q = np.array([m1, r1, m2, r2])
        return ((m1 - h1) * 1000.0, math.degrees(r1), (m2 - h2) * 1000.0, math.degrees(r2))


@dataclass(frozen=True)
class NoiseModel:
    """``HCMR-EndoNavSimDocker/src/sensor_sim/config/noise_params.yaml`` (noise_injection_node)."""

    position_std: float = 0.0001
    orientation_std: float = 0.001
    force_std: float = 0.005
    insertion_std: float = 0.00005
    rotation_std: float = 0.001
    proximal_std: float = 0.002
    tip_dropout: float = 0.07
    latency_min_s: float = 0.05
    latency_max_s: float = 0.15


def _perturb_quat(q, std: float, rng: random.Random):
    dx, dy, dz = rng.gauss(0.0, std), rng.gauss(0.0, std), rng.gauss(0.0, std)
    angle = math.sqrt(dx * dx + dy * dy + dz * dz)
    if angle == 0.0:
        bx, by, bz, bw = 0.0, 0.0, 0.0, 1.0
    else:
        s = math.sin(angle / 2.0) / angle
        bx, by, bz, bw = dx * s, dy * s, dz * s, math.cos(angle / 2.0)
    x, y, z, w = _normalise(*q)
    return _normalise(bw * x + bx * w + by * z - bz * y, bw * y - bx * z + by * w + bz * x,
                      bw * z + bx * y - by * x + bz * w, bw * w - bx * x - by * y - bz * z)


def _normalise(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    return (0.0, 0.0, 0.0, 1.0) if n == 0.0 else (x / n, y / n, z / n, w / n)


class NoiseChannel:
    """noise_injection_node for one device: true states in, noisy measurements with release times out."""

    def __init__(self, model: NoiseModel, seed: int):
        self.m = model
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        self.last_vision: dict | None = None
        self.last_release = -math.inf

    def measure(self, s: dict) -> dict:
        g, m = self.rng.gauss, self.m
        out = {"true_t": s["t"], "frame": s["frame"], "ins": s["ins"] + g(0.0, m.insertion_std),
               "rot": s["rot"] + g(0.0, m.rotation_std), "proximal": s["proximal"] + g(0.0, m.proximal_std),
               "contacts": int(s["contacts"]), "dropout": False}
        if self.rng.random() < m.tip_dropout:
            v = self.last_vision
            if v is not None:
                out.update({k: v[k] for k in ("tip", "quat", "vel", "omg", "seg", "tip_force", "axial", "peak")})
            else:  # no vision yet: their message keeps default (zero) fields
                out.update({"tip": [0.0, 0.0, 0.0], "quat": [0.0, 0.0, 0.0, 1.0], "vel": [0.0, 0.0, 0.0],
                            "omg": [0.0, 0.0, 0.0], "seg": [], "tip_force": 0.0, "axial": 0.0, "peak": 0.0})
            out["dropout"] = True
        else:
            out["tip"] = [c + g(0.0, m.position_std) for c in s["tip"]]
            out["quat"] = list(_perturb_quat(s["quat"], m.orientation_std, self.rng))
            out["vel"], out["omg"] = list(s["vel"]), list(s["omg"])
            seg = list(s["seg"])
            for i in range(0, len(seg), 7):
                seg[i], seg[i + 1], seg[i + 2] = seg[i] + g(0.0, m.position_std), seg[i + 1] + g(0.0, m.position_std), seg[i + 2] + g(0.0, m.position_std)
                seg[i + 3:i + 7] = _perturb_quat(seg[i + 3:i + 7], m.orientation_std, self.rng)
            out["seg"] = seg
            out["tip_force"] = s["tip_force"] + g(0.0, m.force_std)
            out["axial"] = s["axial"] + g(0.0, m.force_std)
            out["peak"] = s["peak"] + g(0.0, m.force_std)
            self.last_vision = out
        # released after a random latency, but never before an earlier sample (their heap is keyed by true time)
        release = s["t"] + self.rng.uniform(m.latency_min_s, m.latency_max_s)
        self.last_release = max(self.last_release, release)
        out["release_t"] = self.last_release
        out["sim_t"] = s["t"]
        return out


def _ns(t: float) -> int:
    return int(round(t * 1e9))


def _stamp(t: float) -> dict:
    ns = _ns(t)
    return {"sec": ns // 1_000_000_000, "nanosec": ns % 1_000_000_000}


def _viewer_message(device: str, n: dict) -> dict:
    """noisy_visualization_node._message_to_json for one noisy measurement."""
    seg = n["seg"]
    return {"type": "noisy_device_state", "device": device, "header_stamp": _stamp(n["release_t"]),
            "true_header_stamp": _stamp(n["true_t"]), "insertion_depth": n["ins"], "shaft_rotation": n["rot"],
            "tip_position": n["tip"], "tip_quaternion": n["quat"], "tip_velocity": n["vel"],
            "tip_angular_velocity": n["omg"],
            "segments": [{"position": seg[i:i + 3], "orientation": seg[i + 3:i + 7]} for i in range(0, len(seg), 7)],
            "force": {"proximal": n["proximal"], "tip": n["tip_force"], "axial": n["axial"], "peak": n["peak"],
                      "contacts": n["contacts"]},
            "simulated_time": n["sim_t"], "physics_frame": n["frame"], "tip_dropout": n["dropout"],
            "force_dropout": False}


# ---------------------------------------------------------------------- bridge
class LockstepBridge:
    """Their ``openng3_rl.env.Bridge`` protocol on the lockstep pages (see the module docstring)."""

    control_period_s = 1.0 / DEFAULT_SPEC.environment.control_hz
    image_min_interval_s = 0.05  # noisy viewer: at most one image per 50 ms

    def __init__(self, worker: int = 0, seed: int = 0, ports: Ports = Ports(), robot: RobotModel = RobotModel(),
                 noise: NoiseModel = NoiseModel(), timeout_s: float = 120.0, visible: bool = False):
        self.ports = ports
        self.status_file = ensure_servers(ports)
        obs = organiser_rl("observation")
        self._NoisySample, self._TrueSample, self._Packet = obs.NoisySample, obs.TrueSample, obs.ObservationPacket
        self._image_shape = obs.IMAGE_SHAPE
        query = (f"?scenario_url=http://localhost:{ports.asset}/scenario.json&renderer_session=lockstep-{worker}"
                 f"&ws_port_base={WS_PORT_BASE}&main_port={ports.web}&noisy_port={ports.web + 1}")
        profiles = Path(os.environ.get("LOCALAPPDATA", "/tmp")) / "stroke-nav" / "lockstep"
        dbg = 9330 + 2 * worker
        self.physics = LockstepPage(f"http://localhost:{ports.web}/{PHYSICS_HTML}{query}", dbg, profiles / f"physics-{worker}",
                                    window=(0, 0, 1100, 900) if visible else None)
        self.viewer = LockstepPage(f"http://localhost:{ports.web + 1}/{NOISY_HTML}{query}", dbg + 1, profiles / f"viewer-{worker}",
                                   window=(1100, 0, 700, 900) if visible else None)
        self.last_viewer_jpeg: bytes | None = None  # the noisy viewer's latest RL image (480 px JPEG)
        self.robot = Robot(robot)
        self.noise = {"guidewire": NoiseChannel(noise, seed * 2 + 1), "catheter": NoiseChannel(noise, seed * 2 + 2)}
        self.timeout_s = timeout_s
        try:
            self._start_pages()
        except Exception:
            self.close()
            raise
        self._clear()

    # ------------------------------------------------------------------ setup
    def _start_pages(self) -> None:
        self.physics.start()
        self.physics.wait(_mesh_ready_js(4), self.timeout_s, "physics page")
        self.physics.cdp.eval("__ls.skipCapture(true); true")
        self.viewer.start()
        for attempt in range(3):  # its first scenario fetch sometimes fails ("Failed to fetch"): reload
            try:
                self.viewer.wait(_mesh_ready_js(1), 30.0, "noisy viewer")
                break
            except TimeoutError:
                if attempt == 2 or "could not" not in self.viewer.mesh_info():
                    raise
                self.viewer.reload()
        # a few frames so both pages have baked the mesh and laid out the rods
        self.physics.cdp.eval("for (let i = 0; i < 5; i++) __ls.frame(); true")
        self.viewer.cdp.eval("__ls.frame(); true")

    def _clear(self) -> None:
        self.t = 0.0  # simulated time of the last physics frame
        self.now = 0.0  # simulated time the env last received a packet
        self.tick = 0.0
        self.true: dict[str, list[dict]] = {"guidewire": [], "catheter": []}
        self.noisy: dict[str, list[dict]] = {"guidewire": [], "catheter": []}
        self.images: list[dict] = []  # viewer images not yet consumed: {"t": send time, "gw": noisy index}
        self.image_t = -math.inf  # viewer: time of its last image
        self.image_gw = -1  # viewer: noisy guidewire index shown in its last image
        self.sequence = 0
        self.latest_true_sample = None

    # ------------------------------------------------------------------ Bridge protocol
    def set_command(self, commands: dict) -> None:
        self.robot.set_command(commands)

    def reset_episode(self, timeout_s: float, max_skew_s: float):
        self.robot.reset()
        for ch in self.noise.values():
            ch.reset()
        self._clear()
        self._ingest(self.physics.cdp.eval("__ls.reset()"))
        packet = self._advance_until_packet(after_t=-math.inf, max_skew_s=max_skew_s, timeout_s=timeout_s)
        self.tick = self.now
        return packet

    def command_and_wait(self, commands: dict, after_sequence: int, timeout_s: float, max_skew_s: float,
                         not_before: float | None = None):
        del after_sequence, not_before  # simulated-time ticks replace the env's wall-clock grid
        self.set_command(commands)
        tick = self.tick + self.control_period_s
        if tick <= self.now:  # late step: restart the grid with a full period, like OpenNG3Env.step
            tick = self.now + self.control_period_s
        self.tick = tick
        self._run_until(tick)
        return self._advance_until_packet(after_t=tick, max_skew_s=max_skew_s, timeout_s=timeout_s)

    def latest_true(self):
        return self.latest_true_sample

    def close(self) -> None:
        self.viewer.close()
        self.physics.close()

    # ------------------------------------------------------------------ simulation
    def _run_until(self, t_end: float) -> None:
        n = max(0, int(math.floor((t_end - self.t) / FRAME_DT + 1e-6)))
        if n:
            self._frames(n)

    def _frames(self, n: int) -> None:
        plan = [self.robot.advance() for _ in range(n)]
        states = self.physics.cdp.eval(f"__ls.step({json.dumps(plan)})")
        self.t += n * FRAME_DT
        self._ingest(states)

    def _ingest(self, states: list[dict]) -> None:
        for s in states:
            dev = s["device"]
            if self.true[dev] and s["t"] <= self.true[dev][-1]["t"]:
                continue  # their state node and noise heap keep strictly increasing stamps
            self.true[dev].append(s)
            self.noisy[dev].append(self.noise[dev].measure(s))
        self._viewer_images()

    def _viewer_images(self) -> None:
        """The noisy viewer's image events up to ``self.t``: it renders at 60 frames/s and sends an
        image when a guidewire sample with a new stamp has arrived and 50 ms have passed since its
        last image; the image shows the newest guidewire sample released by then."""
        gw = self.noisy["guidewire"]
        while self.image_gw + 1 < len(gw):
            t = max(gw[self.image_gw + 1]["release_t"], self.image_t + self.image_min_interval_s)
            t = math.ceil(t / FRAME_DT - 1e-9) * FRAME_DT  # next rendered frame
            if t > self.t:
                return  # samples released later may still change what this image shows
            k = self.image_gw + 1
            while k + 1 < len(gw) and gw[k + 1]["release_t"] <= t:
                k += 1
            self.image_t, self.image_gw = t, k
            self.images.append({"t": t, "gw": k})

    def _advance_until_packet(self, after_t: float, max_skew_s: float, timeout_s: float):
        """Run frame by frame until the env's packet rule is met at simulated time <= self.t."""
        deadline = self.t + timeout_s
        while True:
            packet = self._packet(after_t, max_skew_s)
            if packet is not None:
                return packet
            if self.t > deadline:
                raise TimeoutError("timed out waiting for a fresh synchronized noisy image/state packet")
            self._frames(1)

    def _packet(self, after_t: float, max_skew_s: float):
        """openng3_rl.ros_io: at each moment T, the first image acquired after ``after_t`` whose stamp
        has a catheter sample (released by T) within ``max_skew_s`` forms the packet."""
        images = [i for i in self.images if i["t"] > after_t]
        if not images:
            return None
        gw_list, ca_list = self.noisy["guidewire"], self.noisy["catheter"]
        events = sorted({i["t"] for i in images} | {c["release_t"] for c in ca_list if c["release_t"] > images[0]["t"]})
        for T in events:
            if T > self.t:
                break
            available = [c for c in ca_list if c["release_t"] <= T]
            for img in images:
                if img["t"] > T or not available:
                    break
                stamp = gw_list[img["gw"]]["true_t"]
                ca = min(available, key=lambda c: abs(c["true_t"] - stamp))
                if abs(ca["true_t"] - stamp) > max_skew_s:
                    continue
                return self._emit(img, ca, T, stamp)
        return None

    def _emit(self, img: dict, ca: dict, T: float, stamp: float):
        self.now = T
        image = self._render(img["t"])
        true_gw = [s for s in self.true["guidewire"] if s["t"] <= T]
        if true_gw:
            tg = true_gw[-1]
            self.latest_true_sample = self._TrueSample(tip_position=tuple(tg["tip"]), peak_force=float(tg["peak"]),
                                                       source_stamp_ns=_ns(tg["t"]))
        self.sequence += 1
        packet = self._Packet(guidewire=self._sample(self.noisy["guidewire"][img["gw"]]), catheter=self._sample(ca),
                              image=image, image_stamp_ns=_ns(stamp), sequence=self.sequence)
        self.images = [i for i in self.images if i["t"] > img["t"]]
        self._prune(T - 1.0)
        return packet

    def _prune(self, t_old: float) -> None:
        """Drop samples older than ``t_old`` (already delivered to the viewer), keeping indices valid."""
        gw = self.noisy["guidewire"]
        k = 0
        while k < len(gw) - 1 and gw[k]["true_t"] < t_old and gw[k].get("_sent") and k < self.image_gw:
            k += 1
        if k:
            del gw[:k]
            self.image_gw -= k
            for i in self.images:
                i["gw"] -= k
        ca = self.noisy["catheter"]
        j = 0
        while j < len(ca) - 1 and ca[j]["true_t"] < t_old and ca[j].get("_sent"):
            j += 1
        del ca[:j]
        for dev, states in self.true.items():
            keep = [x for x in states if x["t"] >= t_old]
            self.true[dev] = keep or states[-1:]

    def _sample(self, n: dict):
        return self._NoisySample(
            tip_position=tuple(n["tip"]), tip_orientation=tuple(n["quat"]), tip_linear_velocity=tuple(n["vel"]),
            insertion_depth=n["ins"], shaft_rotation=n["rot"], tip_force=n["tip_force"], peak_force=n["peak"],
            contact_count=n["contacts"], tip_dropout=n["dropout"], force_dropout=False, simulated_time=n["sim_t"],
            physics_frame=int(n["frame"]), source_stamp_ns=_ns(n["true_t"]), available_stamp_ns=_ns(n["release_t"]))

    def _render(self, t_img: float) -> np.ndarray:
        """Deliver every noisy message released by ``t_img`` to the viewer in release order, render one
        frame and decode its JPEG the way openng3_rl.ros_io does."""
        from PIL import Image

        msgs = [(n["release_t"], d, n) for d in ("guidewire", "catheter") for n in self.noisy[d] if n["release_t"] <= t_img]
        msgs.sort(key=lambda m: (m[0], m[1] == "guidewire"))  # guidewire last on ties: it triggers the image
        new = [_viewer_message(d, n) for _, d, n in msgs if not n.get("_sent")]
        for _, _, n in msgs:
            n["_sent"] = True
        b64 = self.viewer.cdp.eval(f"__ls.render({json.dumps(new)})")
        for _ in range(3):  # image throttle or a frame still pending: render again
            if b64:
                break
            b64 = self.viewer.cdp.eval("__ls.render([])")
        if not b64:
            v = self.viewer.cdp
            raise RuntimeError(f"noisy viewer did not produce an image: errors {v.eval('__ls.errors()')}, "
                               f"queued frames {v.eval('__ls.queued()')}, sockets {v.eval('__ls.connected()')}, "
                               f"mesh {self.viewer.mesh_info()!r}")
        self.last_viewer_jpeg = base64.b64decode(b64)
        with Image.open(io.BytesIO(self.last_viewer_jpeg)) as decoded:
            img = np.asarray(decoded.convert("L").resize((self._image_shape[2], self._image_shape[1])), dtype=np.uint8)
        return img[None, :, :]


# ---------------------------------------------------------------------- env
def make_env(worker: int = 0, seed: int = 0, goal_world_m=None, max_steps: int | None = None, visible: bool = False):
    """Their ``OpenNG3Env`` on a ``LockstepBridge``; goals from the cas-reference scenario unless
    ``goal_world_m`` (world metres) is given. Configuration as ``HCMR-EV-RL/configs/online_rl.json``.
    ``visible`` shows both pages in Chrome windows instead of running them headless."""
    env_mod, action, goal, scenario = (organiser_rl(m) for m in ("env", "action", "goal", "scenario"))
    e = DEFAULT_SPEC.environment
    a = DEFAULT_SPEC.action
    # one asset server per worker: it holds a single "authoritative" page session, so workers sharing
    # one got each other's mesh-to-world transform (wrong goals, every episode failed)
    bridge = LockstepBridge(worker=worker, seed=seed, ports=Ports(web=8100 + 10 * worker, asset=8102 + 10 * worker),
                            visible=visible)
    if goal_world_m is not None:
        sampler = goal.FixedGoal(tuple(float(v) for v in goal_world_m))
    else:  # their training: ScenarioGoalSampler with the physics page's acknowledged transform
        auth = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{bridge.ports.asset}/authoritative", timeout=5).read())
        sc = scenario.load_scenario(DEFAULT_SPEC.scenario_name, (RL_ROOT / "scenarios").resolve())
        sampler = scenario.ScenarioGoalSampler(sc, np.asarray(auth["mesh_to_world"], dtype=float).reshape(4, 4))
    cfg = env_mod.EnvConfig(
        control_hz=e.control_hz, max_steps=max_steps or e.max_steps, success_radius_m=e.success_radius_m,
        success_steps=e.success_steps, unsafe_force_n=e.unsafe_force_n, reset_timeout_s=e.reset_timeout_s,
        transition_timeout_s=e.transition_timeout_s, max_sync_skew_s=e.max_sync_skew_s,
        actions=action.ActionMapper(modules=(1, 2), max_translation_m_s=a.max_translation_m_s,
                                    max_rotation_rad_s=a.max_rotation_rad_s),
        reward_from_ground_truth=True)
    return env_mod.OpenNG3Env(bridge, cfg, sampler)
