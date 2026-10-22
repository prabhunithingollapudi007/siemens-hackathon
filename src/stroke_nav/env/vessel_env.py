"""Gymnasium environment: steer a guidewire + microcatheter to a target in a VMR anatomy.

Action (4 floats in [-1, 1], the learnable robot parameters):
    [v_guidewire, w_guidewire, v_microcatheter, w_microcatheter]
    v = translation speed (x v_max mm/s, + advances, - retracts), w = rotation speed (x w_max deg/s).

Observation (46 floats, coordinates normalised to the anatomy's bounding box):
    guidewire tip tracking points (3 x 3) now and at the previous step  (18)
    microcatheter tip tracking points (3 x 3) now and previous          (18)
    target position (3), target minus success-device tip (3), previous action (4)

Reward (dense, Robertshaw et al. 2024 style):
    r = -0.005 - progress_coef * (D_t - D_{t-1}) + 1.0 * success - wrong_branch_penalty * entries
    D = geodesic distance (mm, along centerlines) from the success device's tip to the target.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stroke_nav.data.anatomy import Anatomy
from stroke_nav.env.devices import CATH, WIRE, SimParams, TrackSim, device_sim_params


@dataclass
class EnvConfig:
    model: str = "0248_H_AOCERE_CAS"
    variant: str | None = None
    target: object = "random:left_carotid@20-70|right_carotid@20-70"
    start: object = None  # None = anatomy default (distal descending aorta for 0248)
    success_device: str = "microcatheter"  # "microcatheter" | "guidewire"
    success_threshold_mm: float = 5.0
    max_steps: int = 400
    progress_coef: float = 0.01  # paper value 0.001 (see README)
    step_penalty: float = 0.005
    success_reward: float = 1.0
    wrong_branch_penalty: float = 0.0
    include_catheter_obs: bool = True
    plan_routes: bool = True  # route = device-aware plan for the success device (else plain shortest path)
    sim: SimParams = field(default_factory=device_sim_params)  # devices from device-parameters.json


class VesselNavEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 5.0}

    def __init__(self, config: EnvConfig | None = None, anatomy: Anatomy | None = None, **overrides):
        super().__init__()
        cfg = config or EnvConfig()
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise TypeError(f"unknown EnvConfig field {k!r}")
            setattr(cfg, k, v)
        self.cfg = cfg
        self.anat = anatomy or Anatomy.load(cfg.model, cfg.variant)
        self.sim = TrackSim(self.anat, cfg.sim)
        self.success_dev = CATH if cfg.success_device.startswith("micro") else WIRE

        n_obs = (36 if cfg.include_catheter_obs else 18) + 3 + 3 + 4
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self._target_spec = cfg.target
        self._start_spec = cfg.start
        self._plan_cache: dict[tuple[int, int], object] = {}
        self.plan = None

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if "target" in options:
            self._target_spec = options["target"]
        if "start" in options:
            self._start_spec = options["start"]
        a = self.anat
        self.start_node = a.default_start() if self._start_spec is None else a.resolve(self._start_spec, self.np_random)
        reason = ""
        for _ in range(100):
            self.target_node = a.resolve(self._target_spec, self.np_random)
            if not np.isfinite(a.distances_from(self.target_node)[0][self.start_node]):
                reason = "target not connected to the start"
                continue
            self.plan = self._planned(self.start_node, self.target_node)
            if self.plan is None or self.plan.feasible:
                break
            reason = self.plan.message
            if not a.is_random(self._target_spec):
                break
        if reason and (self.plan is not None and not self.plan.feasible):
            raise ValueError(f"no feasible route to target {self._target_spec!r}: {reason}")
        self.target_pos = a.nodes[self.target_node].copy()
        self.dist_to_target, _ = a.distances_from(self.target_node)
        self.route = list(self.plan.nodes) if self.plan is not None else a.route(self.start_node, self.target_node)
        self.route_set = set(self.route) | set(self._continuation(self.target_node, self.route, 20.0))
        self.route_junctions = [n for n in self.route[1:-1] if a.degree[n] >= 3]

        self.sim.reset(self.start_node, rng=self.np_random)
        self.steps = 0
        self.prev_action = np.zeros(4)
        self.prev_points = self._points()
        self.D = self._geo(self.success_dev)
        self.D0 = self.D
        self.D_min = self.D
        self._ep = {
            "path_length_mm": np.zeros(2),
            "retractions": np.zeros(2, dtype=int),
            "last_dir": np.zeros(2),
            "wrong_branch_events": 0,
            "tip_speeds": [],
            "rewards": {"step": 0.0, "progress": 0.0, "success": 0.0, "wrong_branch": 0.0},
            "trajectory": [self._points()[: 2 if self.cfg.include_catheter_obs else 1, 0].copy()],
        }
        return self._obs(), {"target_node": self.target_node, "start_node": self.start_node}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(4), -1.0, 1.0)
        prev_tip = self.sim.position_at(self.sim.L[self.success_dev])
        events = self.sim.step(action)
        self.steps += 1
        ep = self._ep

        # bookkeeping for START metrics
        ep["path_length_mm"] += events["moved"]
        for i, v in ((WIRE, action[0]), (CATH, action[2])):
            d = np.sign(v) if abs(v) > 0.1 else 0.0
            if d < 0 and ep["last_dir"][i] > 0:
                ep["retractions"][i] += 1
            if d != 0:
                ep["last_dir"][i] = d
        wrong = 0
        for (u, chosen, _leader) in self.sim.decisions[len(ep.setdefault("_seen_decisions", [])):]:
            ep["_seen_decisions"].append((u, chosen))
            if u in self.route_set and chosen not in self.route_set:
                wrong += 1
        ep["wrong_branch_events"] += wrong
        tip = self.sim.position_at(self.sim.L[self.success_dev])
        ep["tip_speeds"].append(float(np.linalg.norm(tip - prev_tip)) / self.cfg.sim.dt)

        D_new = self._geo(self.success_dev)
        success = bool(D_new <= self.cfg.success_threshold_mm)
        c = self.cfg
        r_step = -c.step_penalty
        r_prog = -c.progress_coef * (D_new - self.D)
        r_succ = c.success_reward if success else 0.0
        r_wrong = -c.wrong_branch_penalty * wrong
        reward = r_step + r_prog + r_succ + r_wrong
        for k, v in (("step", r_step), ("progress", r_prog), ("success", r_succ), ("wrong_branch", r_wrong)):
            ep["rewards"][k] += v
        self.D = D_new
        self.D_min = min(self.D_min, D_new)
        self.prev_action = action
        obs = self._obs()
        self.prev_points = self._points()
        ep["trajectory"].append(self.prev_points[: 2, 0].copy())

        terminated = success
        truncated = self.steps >= c.max_steps
        info = {"distance_mm": D_new, "success": success, "events": events}
        if terminated or truncated:
            info["episode_metrics"] = self.episode_metrics(success)
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ helpers
    def _planned(self, start: int, target: int):
        if not self.cfg.plan_routes:
            return None
        key = (int(start), int(target))
        if key not in self._plan_cache:
            from stroke_nav.planning.planner import PathPlanner

            spec = self.cfg.sim.device(self.success_dev).spec
            if not hasattr(self, "_planner"):
                self._planner = PathPlanner(self.anat, spec)
            if len(self._plan_cache) > 2048:
                self._plan_cache.clear()
            self._plan_cache[key] = self._planner.plan(start, target)
        return self._plan_cache[key]

    def _geo(self, device: int) -> float:
        i, f = self.sim.index_at(self.sim.L[device])
        tr = self.sim.track
        d0 = self.dist_to_target[tr[i]]
        d1 = self.dist_to_target[tr[min(i + 1, len(tr) - 1)]]
        return float((1 - f) * d0 + f * d1)

    def _points(self) -> np.ndarray:
        return np.stack([self.sim.tracking_points(WIRE), self.sim.tracking_points(CATH)])  # (2, 3, 3)

    def _norm(self, x: np.ndarray) -> np.ndarray:
        return (x - self.anat.center) / self.anat.half_extent

    def _obs(self) -> np.ndarray:
        pts = self._points()
        devs = 2 if self.cfg.include_catheter_obs else 1
        parts = [self._norm(pts[:devs]).ravel(), self._norm(self.prev_points[:devs]).ravel()]
        tip = pts[self.success_dev, 0]
        parts += [self._norm(self.target_pos), (self.target_pos - tip) / self.anat.half_extent, self.prev_action]
        return np.clip(np.concatenate(parts), -10.0, 10.0).astype(np.float32)

    def _continuation(self, node: int, route: list[int], length: float) -> list[int]:
        """Nodes straight beyond the target (so overshooting the target is not a 'wrong branch')."""
        a = self.anat
        if len(route) < 2:
            return []
        prev, cur, acc, out = route[-2], node, 0.0, []
        while acc < length:
            nxt = [w for w in a.neighbors[cur] if w != prev]
            if not nxt:
                break
            w = max(nxt, key=lambda w: float(np.dot(a.nodes[w] - a.nodes[cur], a.nodes[cur] - a.nodes[prev])))
            acc += float(np.linalg.norm(a.nodes[w] - a.nodes[cur]))
            out.append(w)
            prev, cur = cur, w
        return out

    def episode_metrics(self, success: bool | None = None) -> dict:
        """Raw per-episode quantities for START evaluation (see stroke_nav.eval.start_metrics)."""
        ep = self._ep
        if success is None:
            success = bool(self.D <= self.cfg.success_threshold_mm)
        passed = sum(1 for n in self.route_junctions if n in set(self.sim.track[: self.sim.index_at(self.sim.L[self.success_dev])[0] + 1]))
        return {
            "success": bool(success),
            "steps": int(self.steps),
            "dt": float(self.cfg.sim.dt),
            "phases_total": len(self.route_junctions),
            "phases_passed": int(passed),
            "wrong_branch_events": int(ep["wrong_branch_events"]),
            "retractions_guidewire": int(ep["retractions"][WIRE]),
            "retractions_microcatheter": int(ep["retractions"][CATH]),
            "wall_pushes_guidewire": int(self.sim.wall_pushes[WIRE]),
            "wall_pushes_microcatheter": int(self.sim.wall_pushes[CATH]),
            "path_length_mm_guidewire": float(ep["path_length_mm"][WIRE]),
            "path_length_mm_microcatheter": float(ep["path_length_mm"][CATH]),
            "tip_speed_mean_mm_s": float(np.mean(ep["tip_speeds"])) if ep["tip_speeds"] else 0.0,
            "tip_speed_max_mm_s": float(np.max(ep["tip_speeds"])) if ep["tip_speeds"] else 0.0,
            "initial_distance_mm": float(self.D0),
            "final_distance_mm": float(self.D),
            "min_distance_mm": float(self.D_min),
            "centerline_error_pct": 0.0,  # devices ride the centerline in this kinematic model
            "contact_force_N": float("nan"),  # not modelled yet (wall pushes are the proxy)
            "target_node": int(self.target_node),
            "target_branch": self.anat.branch_of(self.target_node),
            "return_terms": dict(ep["rewards"]),
        }

    @property
    def trajectory(self) -> np.ndarray:
        """(T+1, 2, 3) tip positions of guidewire and microcatheter over the episode."""
        return np.array(self._ep["trajectory"])
