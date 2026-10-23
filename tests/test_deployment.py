"""Deployment bundle pieces: one-goal scenario, run folders on another scenario, the bundled controller loader."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from stroke_nav.organiser.export import check_run_dir, write_run_dir
from stroke_nav.organiser.scenario import read_goals_cm, scenario_with_goals

ROOT = Path(__file__).resolve().parents[1]


def _source_scenario(tmp_path: Path) -> Path:
    src = tmp_path / "src" / "variant"
    src.mkdir(parents=True)
    (src / "vessel.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    (src / "goals.txt").write_text("# two lesions\n0.100,0.200,0.300\n0.400,0.500,0.600\n", encoding="utf-8")
    return src


def test_scenario_with_goals_keeps_mesh_and_one_goal(tmp_path):
    out = scenario_with_goals(_source_scenario(tmp_path), tmp_path / "scenarios", "variant-left", [1], validate=False)
    manifest = json.loads((out / "scenario.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "variant-left" and manifest["selection"] == "uniform"
    assert (out / "vessel.obj").read_bytes() == (tmp_path / "src" / "variant" / "vessel.obj").read_bytes()
    np.testing.assert_allclose(read_goals_cm(out), [[0.4, 0.5, 0.6]])


def test_run_dir_on_another_scenario(tmp_path):
    out = scenario_with_goals(_source_scenario(tmp_path), tmp_path / "scenarios", "variant-left", [0], validate=False)
    manifest = json.loads((out / "scenario.json").read_text(encoding="utf-8"))
    run = write_run_dir(tmp_path / "runs" / "controller-run", scenario_manifest=manifest)
    (run / "controller.json").write_text("{}", encoding="utf-8")
    assert check_run_dir(run, manifest, model="controller.json") == []
    assert any("cas-reference" in p for p in check_run_dir(run, model="controller.json"))
    cfg = json.loads((run / "resolved-config.json").read_text(encoding="utf-8"))
    assert cfg["scenario"] == {"mesh_sha256": manifest["mesh_sha256"], "name": "variant-left"}


class _Env:
    def __init__(self):
        self.resets = 0

    def reset(self, seed=None):
        self.resets += 1
        return {}, {}


def test_bundled_controller_loader(tmp_path):
    # the run folder as export_deployment.py writes it: the loader and entry_roll.py side by side
    shutil.copyfile(ROOT / "src/stroke_nav/organiser/deploy/evaluate_controller.py", tmp_path / "evaluate_controller.py")
    shutil.copyfile(ROOT / "src/stroke_nav/agents/entry_roll.py", tmp_path / "entry_roll.py")
    (tmp_path / "controller.json").write_text(json.dumps({
        "schema": "stroke-nav/entry-roll-controller/1", "controller": "EntryRollExpert",
        "params": {"roll_deg": -15.0, "slow_radius_mm": 8.0}}), encoding="utf-8")
    import importlib.util

    spec = importlib.util.spec_from_file_location("bundled_controller", tmp_path / "evaluate_controller.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    env = _Env()
    policy = mod.load_controller(tmp_path / "controller.json", env=env)
    state = np.zeros(43, dtype=np.float32)
    state[19] = 1.0  # cos(roll) = 1: shaft at 0 deg
    state[36] = 0.1  # goal 100 mm away
    action, _ = policy.predict({"state": state}, deterministic=True)
    assert action.shape == (4,) and action[1] < 0 and action[0] == 0  # rolls towards -15 deg first, no push yet
    policy.controller.phase = "push"
    env.reset(seed=0)  # their run_evaluation resets the env per episode: the controller restarts in its roll phase
    assert env.resets == 1 and policy.controller.phase == "roll"
