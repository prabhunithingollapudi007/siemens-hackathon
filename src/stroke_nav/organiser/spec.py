"""The organiser's robot, devices, RL environment and training settings, as data.

Values are copied from the organiser's code and run snapshots (paths relative to their
``Hackathon/`` folder) so that our training matches theirs and our run folders validate in their
evaluation:

* device controls ......... HCMR-EV-RL/runs/online-rl-pool-20261021-145046-707530Z/device-parameters.json
* RL environment/training . HCMR-EV-RL/configs/online_rl.json
* action limits ........... HCMR-EV-RL/src/openng3_rl/action.py
* observation layout ...... HCMR-EV-RL/src/openng3_rl/observation.py
* reward .................. HCMR-EV-RL/src/openng3_rl/reward.py
* scenario manifest ....... HCMR-EV-RL/scenarios/cas-reference/scenario.json
* noise ................... HCMR-EndoNavSimDocker/src/sensor_sim/config/noise_params.yaml

``load_from_repo(path)`` re-reads the live files if a checkout of their repo is available, so a
newer organiser configuration can be picked up without editing this file.

Device parameters: ``device-parameters.json`` at the repository root is the source of truth for
both devices in every simulator, planner and robot path of this repo (``DEFAULT_SPEC`` is loaded
from it at import). Set ``STROKE_NAV_DEVICE_PARAMS`` to use another file. The built-in copy
``DEFAULT_DEVICE_CONTROLS`` is only a fallback when no file is found (with a warning).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

# ---------------------------------------------------------------- device parameters (schema 1)
DEVICE_CONTROL_NAMES = (
    "preset", "young", "density", "poisson", "diameter", "bendScale", "twistScale",
    "stretchScale", "wireLength", "bodies", "tipLength", "tipAngle", "railStiff", "friction",
    "tipSoftLength", "tipBendScale", "tipTwistScale", "cathPreset", "cathYoung", "cathDensity",
    "cathPoisson", "cathDiameter", "cathBendScale", "cathTwistScale", "cathStretchScale",
    "cathLength", "cathBodies", "cathTipLength", "cathTipAngle", "cathRailStiff", "cathFriction",
    "gravity", "damping", "substeps", "iterations", "collIters", "collide", "bloodViscosity",
    "clearance", "cathCouple", "cathLumenClear", "cathGwFriction",
)
_BOOLEAN = {"collide", "cathCouple"}
_PRESET = {"preset", "cathPreset"}
_INTEGER = {"bodies", "cathBodies", "substeps", "iterations", "collIters"}

# Snapshot used by the organiser's 4-worker SAC run (radifocus 0.035" wire + judkins 5F catheter).
# Units follow the browser controls: diameters in mm, lengths of wire/catheter in m, tip lengths in
# mm, tip angles in deg; young/bend/twist scales are log10 multipliers of the preset defaults.
DEFAULT_DEVICE_CONTROLS: dict = {
    "bendScale": -1.3, "bloodViscosity": 0.0035, "bodies": 160.0, "cathBendScale": -0.8,
    "cathBodies": 56.0, "cathCouple": True, "cathDensity": 1300.0, "cathDiameter": 1.67,
    "cathFriction": 0.3, "cathGwFriction": 0.12, "cathLength": 0.9, "cathLumenClear": 0.1,
    "cathPoisson": 0.4, "cathPreset": "judkins", "cathRailStiff": 0.7, "cathStretchScale": 0.0,
    "cathTipAngle": 70.0, "cathTipLength": 28.0, "cathTwistScale": 0.5, "cathYoung": -1.5,
    "clearance": 0.5, "collIters": 2.0, "collide": True, "damping": 2.5, "density": 6450.0,
    "diameter": 0.89, "friction": 0.25, "gravity": 0.5, "iterations": 2.0, "poisson": 0.33,
    "preset": "radifocus", "railStiff": 0.7, "stretchScale": 0.0, "substeps": 5.0,
    "tipAngle": 45.0, "tipBendScale": -2.6, "tipLength": 30.0, "tipSoftLength": 0.0,
    "tipTwistScale": 1.3, "twistScale": 1.3, "wireLength": 1.2, "young": 0.2,
}


def validate_device_parameters(document) -> dict:
    """Same rules as the organiser's ``device_parameters.validate_device_parameters``."""
    if not isinstance(document, dict) or set(document) != {"schema_version", "controls"}:
        raise ValueError("device parameters must contain schema_version and controls")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("unsupported device parameter schema")
    controls = document["controls"]
    if not isinstance(controls, dict) or set(controls) != set(DEVICE_CONTROL_NAMES):
        missing = set(DEVICE_CONTROL_NAMES) - set(controls or {})
        extra = set(controls or {}) - set(DEVICE_CONTROL_NAMES)
        raise ValueError(f"device parameters contain missing {sorted(missing)} or unknown {sorted(extra)} controls")
    normalized = {}
    for name in DEVICE_CONTROL_NAMES:
        value = controls[name]
        if name in _BOOLEAN:
            if type(value) is not bool:
                raise ValueError(f"device control {name} must be boolean")
        elif name in _PRESET:
            if not isinstance(value, str) or not value:
                raise ValueError(f"device control {name} must be a preset identifier")
        else:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"device control {name} must be finite and numeric")
            if name in _INTEGER and (value < 1 or int(value) != value):
                raise ValueError(f"device control {name} must be a positive integer")
            value = float(value)
        normalized[name] = value
    return {"schema_version": 1, "controls": normalized}


def device_parameters_hash(document) -> str:
    """SHA-256 the organiser's evaluation compares against ``resolved-config.json``."""
    encoded = json.dumps(validate_device_parameters(document), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- environment / training
@dataclass(frozen=True)
class ActionSpec:
    """Normalised action [-1, 1]^4 = [gw_trans, gw_rot, cath_trans, cath_rot] (module 1 = guidewire)."""

    max_translation_m_s: float = 0.01
    max_rotation_rad_s: float = 1.0


@dataclass(frozen=True)
class EnvironmentSpec:
    control_hz: float = 5.0
    max_steps: int = 600
    success_radius_m: float = 0.005
    success_steps: int = 3
    unsafe_force_n: float = 3.0
    reset_timeout_s: float = 20.0
    transition_timeout_s: float = 2.0
    max_sync_skew_s: float = 0.1
    readiness_sample_s: float = 5.0
    readiness_timeout_s: float = 30.0


@dataclass(frozen=True)
class RewardSpec:
    progress_scale_m: float = 0.01
    contact_exposure_scale_n_s: float = 0.01
    progress_weight: float = 1.0
    contact_weight: float = 1.0
    action_change_weight: float = 0.02
    time_cost: float = 0.001
    success_bonus: float = 10.0
    unsafe_penalty: float = 10.0


@dataclass(frozen=True)
class TrainingSpec:
    algorithm: str = "SAC"
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    buffer_size: int = 50_000
    learning_starts: int = 2000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    checkpoint_every_steps: int = 10_000
    seed: int = 0
    device: str = "auto"
    torch_threads: int = 4


@dataclass(frozen=True)
class NoiseSpec:
    """Measurement noise of their ``sensor_sim`` layer applied to observations (not to reward)."""

    tip_position_sigma_m: float = 0.0001
    tip_dropout_probability: float = 0.07
    force_dropout_probability: float = 0.0


@dataclass(frozen=True)
class RobotSpec:
    """Monadnock robot limits measured on their stack (2026-10-22, scripts/measure_gap.py).

    Module 1 (guidewire) starts at 1.200 m and stops at 1.4645 m; module 2 (catheter) starts at
    0.900 m and must stay 70.8 mm behind module 1. The EndoNav follower maps joint travel 1:1 onto
    device insertion, so these are insertion limits from the reset position.
    """

    guidewire_travel_mm: float = 264.5
    catheter_lead_max_mm: float = 229.2  # catheter insertion <= guidewire insertion + this
    guidewire_lag_mm: float = 5.0  # measured mean lag of their rod behind the commanded insertion
    catheter_lag_mm: float = 1.5


IMAGE_SHAPE = (1, 128, 128)
OBS_CLIP = 10.0
STATE_SIZE = 43  # 2 x (pose 7 + aux 11) + goal delta 3 + previous action 4


@dataclass(frozen=True)
class OrganiserSpec:
    action: ActionSpec = field(default_factory=ActionSpec)
    environment: EnvironmentSpec = field(default_factory=EnvironmentSpec)
    reward: RewardSpec = field(default_factory=RewardSpec)
    training: TrainingSpec = field(default_factory=TrainingSpec)
    noise: NoiseSpec = field(default_factory=NoiseSpec)
    robot: RobotSpec = field(default_factory=RobotSpec)
    device_controls: dict = field(default_factory=lambda: dict(DEFAULT_DEVICE_CONTROLS))
    scenario_name: str = "cas-reference"
    goals_file: str = "goals.json"
    insertion_limits_mm: tuple[float, float] = (-50.0, 800.0)  # browser slider range
    roll_limits_deg: tuple[float, float] = (-1080.0, 1080.0)

    # ---------------------------------------------------------------- derived values
    @property
    def dt(self) -> float:
        return 1.0 / self.environment.control_hz

    def device_parameters(self) -> dict:
        return validate_device_parameters({"schema_version": 1, "controls": copy.deepcopy(self.device_controls)})

    def online_rl_config(self) -> dict:
        """``configs/online_rl.json`` (schema 1) for these settings."""
        return {
            "schema_version": 1,
            "goals_file": self.goals_file,
            "action": asdict(self.action),
            "environment": asdict(self.environment),
            "training": asdict(self.training),
        }

    def guidewire_geometry(self) -> dict:
        c = self.device_controls
        return {"preset": c["preset"], "outer_diameter_mm": c["diameter"], "length_m": c["wireLength"],
                "tip_length_mm": c["tipLength"], "tip_angle_deg": c["tipAngle"], "bodies": int(c["bodies"])}

    def catheter_geometry(self) -> dict:
        c = self.device_controls
        return {"preset": c["cathPreset"], "outer_diameter_mm": c["cathDiameter"], "length_m": c["cathLength"],
                "tip_length_mm": c["cathTipLength"], "tip_angle_deg": c["cathTipAngle"], "bodies": int(c["cathBodies"])}

    def with_training(self, **overrides) -> "OrganiserSpec":
        return replace(self, training=replace(self.training, **overrides))


ORGANISER_DEVICE_PARAMETERS_HASH = "38b1ea532f0a929ed5d36111e2e0f0efb28ca1610a40ade0bccc987813a65fbc"


def load_device_parameters(path: str | Path, spec: "OrganiserSpec | None" = None) -> "OrganiserSpec":
    """Spec with device controls from a ``device-parameters.json`` (the physics page's settings).

    The guidewire and catheter geometry (diameter, length, tip length and angle) used by our
    simulator, planner and follower all derive from these controls.
    """
    doc = validate_device_parameters(json.loads(Path(path).read_text(encoding="utf-8")))
    return replace(spec or OrganiserSpec(), device_controls=doc["controls"])


def load_from_repo(hcmr_ev_rl_dir: str | Path, run_id: str | None = None) -> OrganiserSpec:
    """Read their live ``configs/online_rl.json`` (and a run's ``device-parameters.json``) if present."""
    root = Path(hcmr_ev_rl_dir)
    spec = OrganiserSpec()
    cfg_path = root / "configs" / "online_rl.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if cfg.get("schema_version") != 1:
            raise ValueError(f"unsupported online_rl.json schema in {cfg_path}")
        spec = replace(
            spec,
            action=ActionSpec(**cfg.get("action", {})),
            environment=EnvironmentSpec(**cfg.get("environment", {})),
            training=TrainingSpec(**{k: v for k, v in cfg.get("training", {}).items() if k in TrainingSpec.__dataclass_fields__}),
            goals_file=cfg.get("goals_file", spec.goals_file),
        )
    if run_id:
        dp = root / "runs" / run_id / "device-parameters.json"
        if dp.is_file():
            doc = validate_device_parameters(json.loads(dp.read_text(encoding="utf-8")))
            spec = replace(spec, device_controls=doc["controls"])
    return spec


# ---------------------------------------------------------------- scenario (cas-reference)
CAS_REFERENCE_MANIFEST = {
    "bake": {"axis": "authored", "flip": False, "roll_180": False},
    "goal_frame": "mesh",
    "goals_file": "goals.txt",
    "goals_sha256": "157398134b5120edd359d7ee5bc352d120f79aa17ff1c93831a3d6ce6d795f71",
    "mesh_file": "vessel.obj",
    "mesh_format": "obj",
    "mesh_sha256": "04fbd581c7564c7540de614aadda4a2b95931a8d288a7c030f24fd97992bd3be",
    "meters_per_unit": 0.01,
    "name": "cas-reference",
    "schema_version": 1,
    "selection": "uniform",
    "units": "cm",
}
# goals.txt of cas-reference, in mesh (OBJ) cm; world metres = mesh * meters_per_unit
CAS_REFERENCE_GOALS_CM = ((7.001, 9.756, 15.350),)

# Curriculum goals (configs/goals.json): offsets in metres from the post-reset guidewire tip.
CURRICULUM_GOALS_M = {
    "near": ((0.0, 0.0, 0.015), (0.0, 0.0, 0.025), (0.0, 0.0, 0.035)),
    "medium": ((0.0, 0.0, 0.05), (0.0, 0.0, 0.07), (0.0, 0.0, 0.09)),
    "far": ((0.0, 0.0, 0.12), (0.0, 0.0, 0.15), (0.0, 0.0, 0.18)),
}

REPO_ROOT = Path(__file__).resolve().parents[3]
DEVICE_PARAMETERS_ENV = "STROKE_NAV_DEVICE_PARAMS"
BUILTIN_SPEC = OrganiserSpec()  # organiser snapshot, used only when no device-parameters.json is found


def device_parameters_path() -> Path:
    """The device-parameters file in use: ``$STROKE_NAV_DEVICE_PARAMS`` or ``<repo>/device-parameters.json``."""
    env = os.environ.get(DEVICE_PARAMETERS_ENV)
    return Path(env) if env else REPO_ROOT / "device-parameters.json"


def load_active_spec() -> OrganiserSpec:
    """Spec whose device controls come from ``device_parameters_path()`` (fallback: the built-in copy)."""
    path = device_parameters_path()
    if path.is_file():
        return load_device_parameters(path)
    if os.environ.get(DEVICE_PARAMETERS_ENV):
        raise FileNotFoundError(f"{DEVICE_PARAMETERS_ENV} points to a missing file: {path}")
    warnings.warn(f"{path} not found; using the built-in organiser device parameters", stacklevel=2)
    return BUILTIN_SPEC


def describe_devices(spec: OrganiserSpec | None = None) -> str:
    """One line naming the device file, its hash status and the geometry it sets."""
    spec = spec or DEFAULT_SPEC
    path = device_parameters_path()
    h = device_parameters_hash(spec.device_parameters())
    gw, ca = spec.guidewire_geometry(), spec.catheter_geometry()
    src = path.name if path.is_file() else "built-in copy"
    status = "matches the organiser's run" if h == ORGANISER_DEVICE_PARAMETERS_HASH else "DIFFERS from the organiser's run"
    return (f"devices from {src} (sha256 {h[:12]}, {status}): {gw['preset']} wire {gw['outer_diameter_mm']} mm, "
            f"tip {gw['tip_length_mm']:.0f} mm @ {gw['tip_angle_deg']:.0f} deg; {ca['preset']} catheter "
            f"{ca['outer_diameter_mm']} mm, tip {ca['tip_length_mm']:.0f} mm @ {ca['tip_angle_deg']:.0f} deg")


DEFAULT_SPEC = load_active_spec()
