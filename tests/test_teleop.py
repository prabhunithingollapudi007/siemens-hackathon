"""Keyboard teleop controller for the lockstep env (no browser)."""

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("teleop_lockstep", Path(__file__).resolve().parents[1] / "scripts" / "teleop_lockstep.py")
teleop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(teleop)


def test_push_is_latched_and_turns_are_queued():
    t = teleop.Teleop()
    for k in map(ord, "dd"):
        t.handle(k)
    t.handle(ord("w"))
    assert t.action().tolist() == [1.0, 1.0, 0.0, 0.0]
    assert t.action().tolist() == [1.0, 1.0, 0.0, 0.0]
    assert t.action().tolist() == [1.0, 0.0, 0.0, 0.0]  # queue empty: keeps pushing, no turn
    t.handle(ord("2"))
    t.handle(ord("a"))
    t.handle(ord("S"))
    assert t.action().tolist() == [-0.5, -0.5, 0.0, 0.0]
    t.handle(ord(" "))
    assert t.action().tolist() == [0.0, 0.0, 0.0, 0.0]


def test_episode_keys_and_queue_cap():
    t = teleop.Teleop()
    assert t.handle(13) == "save" and t.handle(8) == "discard" and t.handle(27) == "quit"
    for _ in range(100):  # a held, auto-repeating key queues only a few steps
        t.handle(ord("d"))
    assert t.turn == t.max_turn_queue
    turns = [t.action()[1] for _ in range(t.max_turn_queue + 2)]
    assert turns[-2:] == [0.0, 0.0]


def test_hold_mode_follows_held_keys():
    t = teleop.Teleop()
    t.handle(ord("w"), hold_mode=True)  # motion events are ignored in hold mode
    assert t.action(set()).tolist() == [0.0, 0.0, 0.0, 0.0]
    assert t.action({"w", "d"}).tolist() == [1.0, 1.0, 0.0, 0.0]
    t.handle(ord("1"), hold_mode=True)
    assert t.action({"s", "a"}).tolist() == [-0.25, -0.25, 0.0, 0.0]
    assert t.action(set()).tolist() == [0.0, 0.0, 0.0, 0.0]  # released: stops at once


def test_scripted_keys():
    k = teleop.ScriptedKeys("d*2,w,_*3")
    steps = [k.poll(0.0)[0] for _ in range(7)]
    assert steps == [[ord("d")], [ord("d")], [ord("w")], [], [], [], [13]]
    t = teleop.Teleop()
    acts = []
    for keys in steps[:-1]:
        for key in keys:
            t.handle(key)
        acts.append(t.action())
    assert np.allclose(np.stack(acts)[:, :2], [[0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0]])
