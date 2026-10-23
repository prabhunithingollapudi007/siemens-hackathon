"""device-parameters.json is the source of truth for both devices everywhere."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from stroke_nav.env.devices import CATH, WIRE, TrackSim, device_sim_params
from stroke_nav.env.vessel_env import EnvConfig
from stroke_nav.organiser.spec import (BUILTIN_SPEC, DEFAULT_SPEC, DEVICE_PARAMETERS_ENV, ORGANISER_DEVICE_PARAMETERS_HASH, ROBOT_ENV,
                                       device_parameters_hash, device_parameters_path, validate_device_parameters)
from stroke_nav.planning.constraints import CATHETER, GUIDEWIRE, get_device

ROOT = Path(__file__).resolve().parents[1]
REPO_FILE = ROOT / "device-parameters.json"


@pytest.mark.skipif(not REPO_FILE.is_file(), reason="no device-parameters.json in the repo")
def test_repo_file_is_the_active_source_and_matches_the_organiser():
    if os.environ.get(DEVICE_PARAMETERS_ENV) or os.environ.get(ROBOT_ENV, "organiser") != "organiser":
        pytest.skip("another device file or robot profile is selected")
    assert device_parameters_path() == REPO_FILE
    doc = json.loads(REPO_FILE.read_text(encoding="utf-8"))
    assert device_parameters_hash(doc) == ORGANISER_DEVICE_PARAMETERS_HASH  # hand-over invariant
    assert DEFAULT_SPEC.device_parameters() == validate_device_parameters(doc)
    assert device_parameters_hash(BUILTIN_SPEC.device_parameters()) == ORGANISER_DEVICE_PARAMETERS_HASH


def test_every_default_uses_the_device_parameters():
    gw, ca = DEFAULT_SPEC.guidewire_geometry(), DEFAULT_SPEC.catheter_geometry()
    for sp in (EnvConfig().sim, device_sim_params()):
        assert sp.guidewire.tip_angle_deg == gw["tip_angle_deg"] and sp.guidewire.tip_length_mm == gw["tip_length_mm"]
        assert sp.microcatheter.tip_angle_deg == ca["tip_angle_deg"] and sp.microcatheter.tip_length_mm == ca["tip_length_mm"]
        assert (sp.guidewire.spec, sp.microcatheter.spec) == (GUIDEWIRE, CATHETER)
        assert sp.dt == pytest.approx(DEFAULT_SPEC.dt)
        assert sp.wire_travel_mm == DEFAULT_SPEC.robot.guidewire_travel_mm
    assert get_device(GUIDEWIRE).outer_diameter_mm == gw["outer_diameter_mm"]
    assert get_device(CATHETER).outer_diameter_mm == ca["outer_diameter_mm"]


def test_robot_travel_limits_in_the_simulator(toy_anatomy):
    sp = device_sim_params()
    sp.wire_travel_mm, sp.cath_lead_max_mm, sp.enforce_lumen = 50.0, 10.0, False
    sim = TrackSim(toy_anatomy, sp)
    sim.reset(toy_anatomy.default_start(), np.random.default_rng(0))
    for _ in range(40):
        sim.step(np.array([1.0, 0.0, 1.0, 0.0]))
    assert sim.L[WIRE] == pytest.approx(50.0)
    assert sim.L[CATH] == pytest.approx(60.0)  # at most 10 mm ahead of the wire
    off = device_sim_params(robot_limits=False)
    assert off.wire_travel_mm is None and off.cath_lead_max_mm is None


def test_another_file_changes_every_consumer(tmp_path):
    doc = json.loads(REPO_FILE.read_text(encoding="utf-8")) if REPO_FILE.is_file() else BUILTIN_SPEC.device_parameters()
    doc["controls"].update(tipAngle=60, tipLength=20, diameter=0.46, cathTipAngle=40, cathDiameter=1.2)
    f = tmp_path / "devices.json"
    f.write_text(json.dumps(doc), encoding="utf-8")
    code = ("from stroke_nav.env.vessel_env import EnvConfig; from stroke_nav.planning.constraints import get_device; "
            "s = EnvConfig().sim; print(s.guidewire.tip_angle_deg, s.guidewire.tip_length_mm, s.microcatheter.tip_angle_deg, "
            "get_device('radifocus_035').outer_diameter_mm, get_device('judkins_5f').outer_diameter_mm)")
    env = dict(os.environ, **{DEVICE_PARAMETERS_ENV: str(f)}, PYTHONPATH=str(ROOT / "src"))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True).stdout.split()
    assert [float(x) for x in out] == [60.0, 20.0, 40.0, 0.46, 1.2]


def test_missing_selected_file_is_an_error(tmp_path):
    env = dict(os.environ, **{DEVICE_PARAMETERS_ENV: str(tmp_path / "nope.json")}, PYTHONPATH=str(ROOT / "src"))
    r = subprocess.run([sys.executable, "-c", "import stroke_nav.organiser.spec"], capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "missing file" in r.stderr
