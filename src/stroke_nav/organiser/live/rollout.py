"""Rollouts on the organiser's physics (lockstep env) with any controller, recorded as ``EpisodeTrace``.

Controllers implement ``reset()`` and ``act(obs) -> action`` on the organiser observation
(``Dict(image, state(43))``): the scripted experts, ``SACPolicy`` (an SB3 model), the action-chunking
``ChunkExecutor`` and ``RouteFollowerOnObservation`` (the A* route follower steering on the noisy
tip state). ``lockstep_pool`` runs a list of episodes on several workers, each with its own pages
and ports, restarts a worker's pages when a headless Chrome dies, and retries the episode.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from stroke_nav.agents.entry_roll import GW_INSERTION
from stroke_nav.agents.route_follower import ClosedLoopRouteFollower, FollowerParams, TipObs
from stroke_nav.planning.candidates import EpisodeTrace

WIRE_ONLY = np.array([1.0, 1.0, 0.0, 0.0])
CATH_INSERTION = 28  # catheter aux block starts at 25: velocity (3), insertion
PAGE_ERRORS = (ConnectionError, OSError, RuntimeError, TimeoutError)


def scenario_dir(name: str = "cas-reference", root=None):
    """Folder of a scenario their pages load: ``root/name``, or their own ``HCMR-EV-RL/scenarios/name``."""
    from pathlib import Path

    from stroke_nav.organiser.live.lockstep import RL_ROOT

    return Path(root) / name if root is not None else RL_ROOT / "scenarios" / name


def wall_guard_for(name: str = "cas-reference", root=None):
    """``WallGuard`` for a scenario, with the model's centerlines and frame for the shortcut test when they
    are built: ``cas-reference`` is 0248 (fitted frame), ``<model>-<variant>[-...]`` a 0248 variant, else ``<model>``."""
    from stroke_nav.data.anatomy import Anatomy
    from stroke_nav.organiser.frame import frame_for
    from stroke_nav.safety.wall import WallGuard

    model, variant = ("0248_H_AOCERE_CAS", None) if name == "cas-reference" else (name.split("-")[0], None)
    if name != "cas-reference" and "-" in name:
        variant = name.split("-")[1]
    try:
        anat, frame = Anatomy.load(model, variant), frame_for(model)
    except (FileNotFoundError, KeyError, ValueError):
        anat = frame = None  # depth test only
    return WallGuard.from_scenario(scenario_dir(name, root), anat=anat, frame=frame)


def log(msg: str) -> None:
    print(f"[lockstep] {msg}", flush=True)


def quat_z(q) -> np.ndarray:
    """Local +Z axis (the tip direction) of a quaternion (x, y, z, w)."""
    x, y, z, w = q
    return np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])


class SACPolicy:
    def __init__(self, path: str):
        from stable_baselines3 import SAC

        self.model = SAC.load(path, device="cpu")

    def reset(self) -> None:
        pass

    def act(self, obs: dict) -> np.ndarray:
        return self.model.predict(obs, deterministic=True)[0]


class RouteFollowerOnObservation:
    """``ClosedLoopRouteFollower`` driven by the organiser's noisy state: tip position and quaternion
    (indices 0:7), guidewire and catheter insertion. Holds still once the route's travel limit is
    reached. ``tip_vmr_mm`` / ``tip_dir_vmr`` keep the last measured pose (for replanning)."""

    def __init__(self, anat, plan, params: FollowerParams, frame, goal_vmr_mm):
        self.follower = ClosedLoopRouteFollower(anat, plan, params)
        self.frame, self.goal = frame, np.asarray(goal_vmr_mm, float)
        self.last_dir = np.array([0.0, 0.0, 1.0])
        self.tip_vmr_mm: np.ndarray | None = None

    def reset(self) -> None:
        self.follower.reset()
        self.tip_vmr_mm = None

    def act(self, obs: dict) -> np.ndarray:
        s = np.asarray(obs["state"], float)
        if not np.any(s[0:3]):
            return np.zeros(4, np.float32)  # no vision sample yet (their message keeps zero fields): wait
        tip = self.frame.world_m_to_vmr_mm(s[0:3])
        d = quat_z(s[3:7]) @ self.frame.M.T
        n = float(np.linalg.norm(d))
        self.last_dir = d / n if n > 1e-9 else self.last_dir
        self.tip_vmr_mm = tip
        if self.follower.state.mode == "travel_limit":
            return np.zeros(4, np.float32)
        o = TipObs(tip, self.last_dir, s[GW_INSERTION] * 1000.0, s[CATH_INSERTION] * 1000.0, self.goal)
        return np.asarray(self.follower.act(o), np.float32)

    @property
    def wrong_vessel_events(self) -> int:
        return sum(1 for e in self.follower.state.log if e["event"] == "wrong_vessel")


def run_lockstep_episode(env, policy, seed: int, sigma: float = 0.0, rng: np.random.Generator | None = None,
                         record: bool = False, guard=None) -> tuple[EpisodeTrace, dict | None]:
    """One episode in the lockstep env. ``policy.act(obs)`` is the label; with ``sigma`` Gaussian noise is
    added to the executed guidewire action (DART), the label stays the policy's. ``record`` also returns
    the arrays ``record_demos.py`` saves, so the episode can serve as a demonstration. ``guard``
    (``stroke_nav.safety.wall.WallGuard``) checks the true wire against the vessel wall after every step;
    a policy with ``bind(env)`` (the ``WallSafeController`` safety layer) is given the env first."""
    if hasattr(policy, "bind"):
        policy.bind(env)
    obs, info = env.reset(seed=seed)
    policy.reset()
    cross_steps, depth = 0, 0.0
    t0 = time.monotonic()
    tips, ins, acts, dists, rewards, images, states, labels = [], [], [], [], [], [], [], []
    peak, done = 0.0, False
    while not done:
        label = np.asarray(policy.act(obs), np.float32)
        noise = rng.normal(0.0, sigma, 4) * WIRE_ONLY if sigma and rng is not None else 0.0
        a = np.clip(label + noise, -1.0, 1.0).astype(np.float32)
        if record:
            images.append(obs["image"])
            states.append(obs["state"])
            labels.append(label)
        obs, r, term, trunc, info = env.step(a)
        if guard is not None and env.bridge.true["guidewire"]:
            from stroke_nav.safety.wall import guidewire_bodies

            crossed = False
            for dev in ("guidewire", "catheter"):  # the catheter's bodies are 16 mm apart: it can cross walls too
                s = env.bridge.true[dev][-1] if env.bridge.true[dev] else None
                if s is None or (dev == "catheter" and s["ins"] * 1000.0 < 2.0):
                    continue
                check = guard.check(guidewire_bodies(s), float(s["ins"]) * 1000.0)
                crossed |= check.crossing
                depth = max(depth, check.depth_mm)
            cross_steps += int(crossed)
        true = env.bridge.latest_true()
        tips.append(np.asarray(true.tip_position, float))
        peak = max(peak, float(true.peak_force))
        ins.append(float(obs["state"][GW_INSERTION]) * 1000.0)
        acts.append(a)
        dists.append(float(info["eval/true_distance_m"]) * 1000.0)
        rewards.append(float(r))
        done = term or trunc
    trace = EpisodeTrace(seed=seed, success=bool(info["success"]), tip_m=np.array(tips), insertion_mm=np.array(ins),
                         actions=np.array(acts), dist_mm=np.array(dists), peak_force_n=peak,
                         reason=str(info.get("reset_reason", "")), wall_s=time.monotonic() - t0,
                         wall_cross_steps=cross_steps, wall_depth_mm=depth, events=list(getattr(policy, "events", [])))
    raw = None
    if record:
        raw = {"images": np.stack(images), "states": np.stack(states), "labels": np.stack(labels),
               "actions": np.stack(acts), "true_dist_m": np.asarray(dists, np.float32) / 1000.0,
               "rewards": np.asarray(rewards, np.float32), "success": trace.success, "steps": trace.steps}
    return trace, raw


def lockstep_pool(jobs: list[dict], make_policy, workers: int, max_steps: int, goal_m, deadline: float | None = None,
                  tag: str = "", on_episode=None, **env_kwargs) -> list[tuple[dict, EpisodeTrace, dict | None]]:
    """Run ``jobs`` (dicts with ``seed``, optional ``sigma`` and ``record``) on ``workers`` lockstep envs, each
    with its own pages and ports. A job is started only before ``deadline`` (monotonic seconds).
    ``make_policy(worker)`` builds the controller; ``on_episode(job, trace, policy)`` may inspect each
    finished episode (e.g. read the follower's events). ``env_kwargs`` go to ``make_env`` (scenario ...)."""
    from stroke_nav.organiser.live.lockstep import make_env

    lock, queue, results = threading.Lock(), list(jobs), []
    guard_args = (env_kwargs.get("scenario_name", "cas-reference"), env_kwargs.get("scenario_root"))

    def fresh_env(worker: int, attempts: int = 3):
        """Their pages sometimes fail to come up (a stale Chrome, a reset socket): try a few times."""
        for k in range(attempts):
            try:
                return make_env(worker=worker, seed=worker, max_steps=max_steps, goal_world_m=goal_m, **env_kwargs)
            except PAGE_ERRORS as e:
                log(f"{tag} worker {worker}: pages did not start ({type(e).__name__}: {str(e)[:100]}); "
                    f"{'retrying' if k + 1 < attempts else 'giving up on this worker'}")
                time.sleep(2.0)
        return None

    def work(worker: int) -> None:
        guard = wall_guard_for(*guard_args)  # one per worker: VTK locators are not shared across threads
        env = fresh_env(worker)
        if env is None:
            return  # the other workers take its jobs
        if env_kwargs.get("visible"):
            from stroke_nav.organiser.live.lockstep import show_goal

            show_goal(env.bridge.physics, goal_m)  # the goal marker in the visible physics page
        policy = make_policy(worker)
        try:
            while True:
                with lock:
                    if not queue or (deadline is not None and time.monotonic() > deadline):
                        return
                    job = queue.pop(0)
                try:
                    trace, raw = run_lockstep_episode(env, policy, job["seed"], job.get("sigma", 0.0),
                                                      np.random.default_rng(job["seed"]), job.get("record", False), guard)
                except PAGE_ERRORS as e:
                    # a headless Chrome can die mid-run (WinError 10054): restart this worker's pages, retry the episode
                    attempts = job.get("attempts", 0) + 1
                    log(f"{tag} seed {job['seed']}: {type(e).__name__}: {str(e)[:120]}; restarting worker {worker}'s pages"
                        f"{'' if attempts < 3 else '; giving up on this seed'}")
                    env.close()
                    if attempts < 3:
                        with lock:
                            queue.insert(0, {**job, "attempts": attempts})
                    env = fresh_env(worker)
                    if env is None:
                        return
                    policy = make_policy(worker)
                    continue
                if on_episode is not None:
                    on_episode(job, trace, policy)
                with lock:
                    results.append((job, trace, raw))
                log(f"{tag} seed {job['seed']} sigma {job.get('sigma', 0.0):g}: {'success' if trace.success else trace.reason} "
                    f"in {trace.steps} steps, min {trace.min_dist_mm:.1f} mm, final {trace.final_dist_mm:.1f} mm, "
                    f"{trace.wall_s:.0f} s"
                    + (f", WALL CROSSED in {trace.wall_cross_steps} steps (up to {trace.wall_depth_mm:.1f} mm)" if trace.wall_cross_steps else "")
                    + (f", safety events {[e['event'] for e in trace.events]}" if trace.events else ""))
        finally:
            if env is not None:
                try:
                    env.close()
                except PAGE_ERRORS as e:  # a page already gone (visible windows closed, a reset socket)
                    log(f"{tag} worker {worker}: close: {type(e).__name__}")

    n = max(1, min(workers, len(jobs)))
    with ThreadPoolExecutor(n) as ex:
        list(ex.map(work, range(n)))
    return sorted(results, key=lambda r: r[0]["seed"])
