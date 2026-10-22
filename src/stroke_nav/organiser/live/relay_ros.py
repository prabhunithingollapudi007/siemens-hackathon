#!/usr/bin/env python3
"""Line-based relay between a host process and the organiser's ROS graph.

Runs INSIDE the EndoNav ros2-bridge container (rclpy + catforce_interfaces); self-contained.
The host (``stroke_nav.organiser.live.bridge``) writes one JSON request per line on stdin and reads
one JSON reply per line on stdout:

    {"op": "reset"}                                   -> {"ok": true, "message": ..., "obs": {...}}
    {"op": "step", "action": [a0, a1, a2, a3], "dt": 0.2} -> {"ok": true, "obs": {...}}
    {"op": "close"}                                   -> {"ok": true}

``action`` is the organiser's normalised action (x 10 mm/s, x 1 rad/s); it is republished at 20 Hz
for ``dt`` seconds on /control/autonomous/module_{1,2}/cmd_vel, like openng3_rl. ``obs`` holds the
latest true and noisy DeviceState of both devices (world metres) and the robot joints.
Reset uses the organiser's /hcmr/episode/reset (reset coordinator).
"""

import json
import sys
import time

import rclpy
from catforce_interfaces.msg import DeviceState
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

MAX_TRANS_MM_S = 10.0
MAX_ROT_RAD_S = 1.0
DEVICES = ("guidewire", "catheter")


def state_dict(m):
    if m is None:
        return None
    p, q = m.tip_pose.position, m.tip_pose.orientation
    return {"insertion_m": m.insertion_depth, "roll_rad": m.shaft_rotation, "tip": [p.x, p.y, p.z],
            "quat": [q.x, q.y, q.z, q.w], "peak_force_n": m.peak_force, "contact_force_n": m.contact_force,
            "tip_force_n": m.tip_force, "contacts": m.contact_count, "sim_time": m.simulated_time}


class StreamLock:
    """Accept DeviceState from one physics page only.

    A second page connected to the bridge (e.g. a background browser tab) publishes on the same
    topics. After a reset the page that just restarted has the smallest simulated_time; lock onto
    it and accept only messages whose simulated_time continues that stream.
    """

    def __init__(self):
        self.t = None
        self.candidates = []

    def clear(self):
        self.t, self.candidates = None, []

    def offer(self, msg):
        st = float(msg.simulated_time)
        if self.t is None:
            self.candidates.append(st)
            return True
        if self.t - 0.25 <= st <= self.t + 2.0:
            self.t = max(self.t, st)
            return True
        return False

    def lock(self):
        if self.t is None and self.candidates:
            self.t = min(self.candidates)
        return self.t


class Relay(Node):
    def __init__(self):
        super().__init__("stroke_nav_relay")
        be = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pubs = {m: self.create_publisher(Twist, f"/control/autonomous/module_{m}/cmd_vel", 10) for m in (1, 2)}
        self.true = {d: None for d in DEVICES}
        self.noisy = {d: None for d in DEVICES}
        self.locks = {d: StreamLock() for d in DEVICES}
        self.noisy_locks = {d: StreamLock() for d in DEVICES}
        self.joints = None
        for d in DEVICES:
            self.create_subscription(DeviceState, f"/endovascular/{d}/state", lambda m, d=d: self._on(self.true, self.locks, d, m), be)
            self.create_subscription(DeviceState, f"/endovascular/{d}/state/noisy", lambda m, d=d: self._on(self.noisy, self.noisy_locks, d, m), be)
        self.create_subscription(JointState, "/ng3/joint_states", lambda m: setattr(self, "joints", m), be)
        self.reset_client = self.create_client(Trigger, "/hcmr/episode/reset")

    @staticmethod
    def _on(store, locks, d, m):
        if locks[d].offer(m):
            store[d] = m

    def command(self, a):
        for module, (v, w) in ((1, (a[0], a[1])), (2, (a[2], a[3]))):
            msg = Twist()
            msg.linear.x = float(max(-1.0, min(1.0, v))) * MAX_TRANS_MM_S
            msg.angular.z = float(max(-1.0, min(1.0, w))) * MAX_ROT_RAD_S
            self.pubs[module].publish(msg)

    def hold(self, a, seconds, rate=20.0):
        end = time.monotonic() + seconds
        while True:
            self.command(a)
            left = end - time.monotonic()
            if left <= 0:
                break
            rclpy.spin_once(self, timeout_sec=min(1.0 / rate, left))

    def obs(self):
        j = self.joints
        return {"true": {d: state_dict(self.true[d]) for d in DEVICES},
                "noisy": {d: state_dict(self.noisy[d]) for d in DEVICES},
                "joints": list(j.position[:4]) if j is not None else None, "wall_time": time.time()}

    def reset(self):
        self.hold([0, 0, 0, 0], 0.5)
        if not self.reset_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("/hcmr/episode/reset unavailable: start the reset coordinator (tools/organiser_stack.ps1 start)")
        fut = self.reset_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=30.0)
        r = fut.result()
        if r is None or not r.success:
            raise RuntimeError(f"episode reset failed: {getattr(r, 'message', 'timeout')}")
        for lk in list(self.locks.values()) + list(self.noisy_locks.values()):
            lk.clear()
        self.hold([0, 0, 0, 0], 1.5)
        for lk in list(self.locks.values()) + list(self.noisy_locks.values()):
            lk.lock()
        self.true = {d: None for d in DEVICES}
        self.noisy = {d: None for d in DEVICES}
        self.hold([0, 0, 0, 0], 0.3)
        deadline = time.monotonic() + 10.0
        while any(self.true[d] is None for d in DEVICES) and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if any(self.true[d] is None for d in DEVICES):
            raise RuntimeError("no DeviceState after reset; is the physics page running?")
        return r.message


def main():
    rclpy.init()
    node = Relay()
    out = sys.stdout
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                op = req.get("op")
                if op == "reset":
                    msg = node.reset()
                    rep = {"ok": True, "message": msg, "obs": node.obs()}
                elif op == "step":
                    node.hold(req["action"], float(req.get("dt", 0.2)))
                    rep = {"ok": True, "obs": node.obs()}
                elif op == "close":
                    node.hold([0, 0, 0, 0], 0.3)
                    out.write(json.dumps({"ok": True}) + "\n")
                    out.flush()
                    break
                else:
                    rep = {"ok": False, "error": f"unknown op {op!r}"}
            except Exception as e:  # report to the host, keep serving
                node.command([0, 0, 0, 0])
                rep = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            out.write(json.dumps(rep) + "\n")
            out.flush()
    finally:
        node.command([0, 0, 0, 0])
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
