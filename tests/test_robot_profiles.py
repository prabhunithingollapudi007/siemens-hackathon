"""Robot profiles under robots/: device parameters their page accepts, robot limits, selection by name."""

import json
from pathlib import Path

import pytest

from stroke_nav.organiser.spec import (DEVICE_PARAMETERS_ENV, ORGANISER_DEVICE_PARAMETERS_HASH, PAGE_INSERTION_MAX_MM,
                                       ROBOT_ENV, ROBOTS_DIR, RobotSpec, device_parameters_hash, device_parameters_path,
                                       load_active_spec, load_robot, validate_device_parameters)

PROFILES = sorted(p for p in ROBOTS_DIR.iterdir() if (p / "device-parameters.json").is_file()) if ROBOTS_DIR.is_dir() else []
# their physics page rejects values outside its sliders (catheter_guidewire_ros2.html, applyDeviceParameters)
PAGE_MAX = {"wireLength": 1.20, "cathLength": 1.20, "bodies": 160, "cathBodies": 80}


@pytest.mark.parametrize("profile", PROFILES, ids=[p.name for p in PROFILES])
def test_profile_files_are_valid_for_their_page(profile):
    doc = validate_device_parameters(json.loads((profile / "device-parameters.json").read_text(encoding="utf-8")))
    for name, hi in PAGE_MAX.items():
        assert doc["controls"][name] <= hi, f"{name} beyond their page's slider"
    assert device_parameters_hash(doc) != ORGANISER_DEVICE_PARAMETERS_HASH  # a profile is not their robot
    robot = load_robot(profile / "robot.json")
    assert robot.name == profile.name and robot.guidewire_travel_mm <= PAGE_INSERTION_MAX_MM


def test_robot_env_selects_the_profile(monkeypatch):
    monkeypatch.delenv(DEVICE_PARAMETERS_ENV, raising=False)
    monkeypatch.setenv(ROBOT_ENV, "long-reach")
    assert device_parameters_path() == ROBOTS_DIR / "long-reach" / "device-parameters.json"
    spec = load_active_spec()
    assert spec.robot.name == "long-reach" and spec.robot.guidewire_travel_mm == 750.0
    monkeypatch.setenv(ROBOT_ENV, "organiser")
    assert load_active_spec().robot == RobotSpec()
    monkeypatch.setenv(ROBOT_ENV, "no-such-robot")
    with pytest.raises(FileNotFoundError):
        device_parameters_path()


def test_robot_json_next_to_any_device_file(tmp_path, monkeypatch):
    src = ROBOTS_DIR / "long-reach"
    (tmp_path / "device-parameters.json").write_bytes((src / "device-parameters.json").read_bytes())
    (tmp_path / "robot.json").write_text(json.dumps({"schema": "stroke-nav/robot/1", "name": "x", "guidewire_travel_mm": 500}))
    monkeypatch.setenv(DEVICE_PARAMETERS_ENV, str(tmp_path / "device-parameters.json"))
    assert load_active_spec().robot.guidewire_travel_mm == 500
    (tmp_path / "robot.json").write_text(json.dumps({"schema": "stroke-nav/robot/1", "guidewire_travel_mm": 900}))
    with pytest.raises(ValueError):  # beyond their page's insertion range
        load_active_spec()
