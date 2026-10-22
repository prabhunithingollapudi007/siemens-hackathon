"""``OrganiserCompatEnv``: the organiser's RL interface on top of our fast simulator.

Matches ``openng3_rl.env.OpenNG3Env`` (HCMR-EV-RL) so that a policy trained here loads and runs in
their evaluation unchanged:

* action ``Box(-1, 1, (4,), float32)`` = [gw_trans, gw_rot, cath_trans, cath_rot],
  scaled to +-0.01 m/s and +-1 rad/s, held for one control period of 0.2 s (5 Hz)
* observation ``Dict(image=Box(0,255,(1,128,128),uint8), state=Box(-10,10,(43,),float32))``,
  state layout and units exactly as ``openng3_rl.observation.ObservationBuilder``
  (positions in their world frame, metres; noise and dropouts like their sensor_sim)
* reward = ``openng3_rl.reward.compute_reward`` on the ground-truth Euclidean distance from the
  guidewire tip to the goal; success = within 5 mm for 3 consecutive steps; 600-step limit

Differences that remain (documented in docs/HANDOVER.md): our devices are kinematic (they ride the
centerlines, rotation picks the branch), there are no contact forces unless ``force_proxy_n`` is
set, and the image is an approximate re-render.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np

from stroke_nav.data.anatomy import Anatomy
from stroke_nav.env.devices import CATH, WIRE, SimParams, device_sim_params
from stroke_nav.env.vessel_env import EnvConfig, VesselNavEnv
from stroke_nav.organiser.fluoro import FluoroRenderer, GreyLevels
from stroke_nav.organiser.frame import OrganiserFrame, default_frame
from stroke_nav.organiser.spec import (
    CAS_REFERENCE_GOALS_CM,
    CURRICULUM_GOALS_M,
    DEFAULT_SPEC,
    IMAGE_SHAPE,
    OBS_CLIP,
    STATE_SIZE,
    OrganiserSpec,
)


def organiser_sim_params(spec: OrganiserSpec = DEFAULT_SPEC, tracking_tip_mm: float | None = None,
                         robot_limits: bool = True) -> SimParams:
    """Our simulator configured from the device parameters (see ``env.devices.device_sim_params``)."""
    return device_sim_params(spec, tracking_tip_mm, robot_limits)


@dataclass
class CompatConfig:
    model: str = "0248_H_AOCERE_CAS"
    variant: str | None = None
    goal_mode: str = "scenario"  # "scenario" (cas-reference goals.txt) | "curriculum" | "target"
    target: object = "random:left_carotid@20-70|right_carotid@20-70"  # used when goal_mode == "target"
    curriculum_group: str = "all"  # near | medium | far | all
    image_mode: str = "render"  # "render" | "blank"
    noise: bool = True
    force_proxy_n: float = 0.0  # N reported while a device pushes against a dead end (0 = no force)
    tracking_tip_mm: float | None = None  # None: tip lengths from the device parameters (30 / 28 mm)
    robot_limits: bool = True  # Monadnock travel limits (spec.robot)
    spec: OrganiserSpec = field(default_factory=lambda: DEFAULT_SPEC)


class OrganiserCompatEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: CompatConfig | None = None, anatomy: Anatomy | None = None,
                 frame: OrganiserFrame | None = None, renderer: FluoroRenderer | None = None):
        super().__init__()
        self.cfg = cfg = config or CompatConfig()
        self.organiser_spec = cfg.spec
        self.anat = anatomy or Anatomy.load(cfg.model, cfg.variant)
        self.frame = frame or default_frame()
        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, IMAGE_SHAPE, dtype=np.uint8),
            "state": gym.spaces.Box(-OBS_CLIP, OBS_CLIP, (STATE_SIZE,), dtype=np.float32),
        })
        self.inner = VesselNavEnv(
            EnvConfig(model=cfg.model, variant=cfg.variant, target=self.anat.default_start(),
                      success_device="guidewire", success_threshold_mm=-1.0, max_steps=10**9,
                      plan_routes=False, sim=organiser_sim_params(self.organiser_spec, cfg.tracking_tip_mm, cfg.robot_limits)),
            anatomy=self.anat,
        )
        self.renderer = renderer
        if cfg.image_mode == "render" and self.renderer is None:
            self.renderer = make_renderer(self.anat, self.frame)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        a = self.anat
        # first reset the inner sim at the start so we know the reset tip (curriculum goals are relative to it)
        self.inner.reset(seed=int(self.np_random.integers(2**31)), options={"target": a.default_start()})
        tip0_world = self._true_tip_world(WIRE)
        goal_world = options.get("goal_world_m")
        if goal_world is None:
            goal_world = self._sample_goal(tip0_world)
        goal_world = np.asarray(goal_world, dtype=float)
        goal_node = a.nearest_node(self.frame.world_m_to_vmr_mm(goal_world))
        if self.cfg.goal_mode in ("curriculum", "target"):
            goal_world = self.frame.vmr_mm_to_world_m(a.nodes[goal_node])  # reachable on our centerlines
        self.goal = goal_world
        self.inner.reset(seed=int(self.np_random.integers(2**31)), options={"target": int(goal_node)})
        self.goal_node = int(goal_node)
        self._steps = 0
        self._in_goal = 0
        self._prev_action = np.zeros(4)
        self._roll = np.zeros(2)  # unwrapped shaft rotation (rad)
        self._held: dict[int, dict] = {}
        self._prev_true = {d: self._true_tip_world(d) for d in (WIRE, CATH)}
        self._blocked = np.zeros(2, dtype=bool)
        self._prev_dist = self._true_distance()
        obs = self._observe()
        return obs, {"goal": self.goal.copy(), "goal_node": self.goal_node,
                     "goal_branch": a.branch_of(self.goal_node), "goal_mode": self.cfg.goal_mode}

    def step(self, action):
        spec = self.organiser_spec
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(4), -1.0, 1.0)
        phi_before = self.inner.sim.phi.copy()
        _, _, _, _, info_inner = self.inner.step(action)
        self._roll += _wrap(self.inner.sim.phi - phi_before)
        self._blocked = np.asarray(info_inner["events"]["blocked"], dtype=bool)
        self._steps += 1
        dist = self._true_distance()
        force = self.cfg.force_proxy_n if self._blocked[WIRE] else 0.0
        self._in_goal = self._in_goal + 1 if dist <= spec.environment.success_radius_m else 0
        success = self._in_goal >= spec.environment.success_steps
        unsafe = force > spec.environment.unsafe_force_n
        terms = compute_reward(self._prev_dist, dist, force, spec.dt, action, self._prev_action, success, unsafe, spec)
        terminated = success or unsafe
        truncated = self._steps >= spec.environment.max_steps
        obs = self._observe()
        self._prev_action = action
        self._prev_dist = dist
        info = {
            "goal": self.goal.copy(),
            "reward_terms": terms,
            "noisy_distance_m": float(np.linalg.norm(self.goal - self._held[WIRE]["pos"])),
            "eval/true_distance_m": dist,
            "success": success,
            "unsafe": unsafe,
            "stale_abort": False,
            "episode_steps": self._steps,
        }
        if terminated or truncated:
            m = self.inner.episode_metrics(success)
            m["final_distance_m_euclidean"] = dist
            m["goal_branch"] = self.anat.branch_of(self.goal_node)
            info["episode_metrics"] = m
        return obs, float(terms["total"]), terminated, truncated, info

    # ------------------------------------------------------------------ observation
    def _observe(self) -> dict:
        poses, aux = [], []
        for dev in (WIRE, CATH):
            true = self._true_tip_world(dev)
            vel = (true - self._prev_true[dev]) / self.organiser_spec.dt
            self._prev_true[dev] = true
            q = self._tip_quaternion_world(dev)
            sample = {"pos": true.copy(), "quat": q, "vel": vel}
            tip_drop = force_drop = False
            if self.cfg.noise:
                sample["pos"] = true + self.np_random.normal(0.0, self.organiser_spec.noise.tip_position_sigma_m, 3)
                tip_drop = bool(self.np_random.random() < self.organiser_spec.noise.tip_dropout_probability)
                force_drop = bool(self.np_random.random() < self.organiser_spec.noise.force_dropout_probability)
            held = self._held.get(dev)
            if held is not None and tip_drop:
                sample = {"pos": held["pos"], "quat": held["quat"], "vel": held["vel"]}
            self._held[dev] = sample
            force = self.cfg.force_proxy_n if self._blocked[dev] else 0.0
            contacts = 1 if self._blocked[dev] else 0
            poses.append(np.concatenate([sample["pos"], sample["quat"]]))
            aux.append(np.concatenate([
                sample["vel"],
                [self.inner.sim.L[dev] / 1000.0],
                [np.sin(self._roll[dev]), np.cos(self._roll[dev])],
                [np.log1p(force), np.log1p(force), np.log1p(contacts)],
                [float(tip_drop), float(force_drop)],
            ]))
        state = np.concatenate([*poses, *aux, self.goal - self._held[WIRE]["pos"], self._prev_action])
        state = np.clip(np.nan_to_num(state, nan=0.0, posinf=OBS_CLIP, neginf=-OBS_CLIP), -OBS_CLIP, OBS_CLIP)
        return {"image": self._image(), "state": state.astype(np.float32)}

    def _image(self) -> np.ndarray:
        if self.renderer is None:
            return np.zeros(IMAGE_SHAPE, dtype=np.uint8)
        sim, lv = self.inner.sim, self.renderer.levels
        devices = []
        for dev, level in ((CATH, lv.catheter), (WIRE, lv.guidewire)):
            if sim.L[dev] > 0.5:
                devices.append((self.frame.vmr_mm_to_world_m(sim.device_polyline(dev, step_mm=4.0)), level))
        return self.renderer.render(devices, self.goal)

    # ------------------------------------------------------------------ helpers
    def _true_tip_world(self, dev: int) -> np.ndarray:
        return self.frame.vmr_mm_to_world_m(self.inner.sim.tracking_points(dev)[0])

    def _true_distance(self) -> float:
        return float(np.linalg.norm(self.goal - self._true_tip_world(WIRE)))

    def _tip_quaternion_world(self, dev: int) -> np.ndarray:
        """Tip frame with local +Z along the tip direction (their convention), as (x, y, z, w)."""
        sim = self.inner.sim
        z = self.frame.vmr_dir_to_world(sim.tip_direction(dev))
        _, n1, n2 = sim.frame_at(sim.L[dev])
        x = np.cos(sim.phi[dev]) * n1 + np.sin(sim.phi[dev]) * n2
        x = self.frame.vmr_dir_to_world(x)
        x = x - np.dot(x, z) * z
        if np.linalg.norm(x) < 1e-9:
            x = np.array([1.0, 0.0, 0.0]) - z[0] * z
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        return _quat_from_matrix(np.stack([x, y, z], axis=1))

    def _sample_goal(self, tip0_world: np.ndarray) -> np.ndarray:
        mode = self.cfg.goal_mode
        if mode == "scenario":
            goals = np.asarray(CAS_REFERENCE_GOALS_CM) * self.frame.meters_per_unit
            return goals[int(self.np_random.integers(len(goals)))]
        if mode == "curriculum":
            groups = CURRICULUM_GOALS_M if self.cfg.curriculum_group == "all" else {
                self.cfg.curriculum_group: CURRICULUM_GOALS_M[self.cfg.curriculum_group]}
            offsets = np.concatenate([np.asarray(v) for v in groups.values()])
            return tip0_world + offsets[int(self.np_random.integers(len(offsets)))]
        if mode == "target":
            node = self.anat.resolve(self.cfg.target, self.np_random)
            return self.frame.vmr_mm_to_world_m(self.anat.nodes[node])
        raise ValueError(f"unknown goal_mode {mode!r}")


# ---------------------------------------------------------------------- shared pieces
def compute_reward(prev_dist, dist, force, dt, action, prev_action, success, unsafe, spec: OrganiserSpec = DEFAULT_SPEC) -> dict:
    """Same terms as ``openng3_rl.reward.compute_reward`` (returned as a dict)."""
    r = spec.reward
    progress = r.progress_weight * float(np.clip((prev_dist - dist) / r.progress_scale_m, -1.0, 1.0))
    contact = -r.contact_weight * (force * dt / r.contact_exposure_scale_n_s)
    action_change = -r.action_change_weight * float(np.mean(np.square(np.asarray(action) - np.asarray(prev_action))))
    time = -r.time_cost
    terminal = (r.success_bonus if success else 0.0) - (r.unsafe_penalty if unsafe else 0.0)
    total = progress + contact + action_change + time + terminal
    return {"total": float(total), "progress": progress, "contact": contact, "action_change": action_change,
            "time": time, "terminal": terminal}


def make_renderer(anatomy: Anatomy, frame: OrganiserFrame, levels: GreyLevels | None = None) -> FluoroRenderer:
    """Renderer for our vessel surface, transformed into organiser world metres."""
    import pyvista as pv

    src = anatomy.meta.get("surface_file")
    if not src or not Path(src).exists():
        raise FileNotFoundError("surface model not found; run scripts/fetch_data.py first or use image_mode='blank'")
    surf = pv.read(src).extract_surface(algorithm="dataset_surface").triangulate()
    surf.points = surf.points * float(anatomy.meta.get("unit_scale", 1.0))
    if surf.n_cells > 60_000:
        surf = surf.decimate(1.0 - 60_000 / surf.n_cells)
    surf.points = frame.vmr_mm_to_world_m(np.asarray(surf.points))
    return FluoroRenderer(surf, levels=levels)


def _wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def _quat_from_matrix(m: np.ndarray) -> np.ndarray:
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)
