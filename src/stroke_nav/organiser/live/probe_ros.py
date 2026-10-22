#!/usr/bin/env python3
"""Drive the organiser's robot with an action sequence and record ground-truth device states.

Runs INSIDE the EndoNav ros2-bridge container (rclpy + catforce_interfaces). Self-contained: it does
not import stroke_nav. Copied in and started by ``stroke_nav.organiser.live.probe``.

    python3 probe_ros.py actions.csv out.csv [--dt 0.2] [--rate 20]

actions.csv: one row per control step with columns gw_trans, gw_rot, cath_trans, cath_rot in [-1, 1]
(the organiser's normalised action; scaled to 10 mm/s and 1 rad/s like openng3_rl.action).

Reset: the organiser's own ``/hcmr/episode/reset`` (HCMR-EV-Integration reset coordinator: inhibit
commands, follower off, restart Monadnock so the robot is back at 1.2 m / 0.9 m, browser and noise
reset, follower on). If that coordinator is not running, a manual fallback drives the robot home
with the follower off, then resets the browser and re-enables the follower.

Module 1 (guidewire) can travel only to 1.4645 m, i.e. 264.5 mm of insertion from home.
"""

import argparse
import csv
import time

import rclpy
from catforce_interfaces.msg import DeviceState
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool, Trigger

MAX_TRANS_MM_S = 10.0  # 0.01 m/s, sent in OpenNG3's native mm/s
MAX_ROT_RAD_S = 1.0
DEVICES = ("guidewire", "catheter")
HOME = (1.2, 0.0, 0.9, 0.0)  # Monadnock joint positions after a restart (m, rad, m, rad)


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


class Probe(Node):
    def __init__(self):
        super().__init__("stroke_nav_probe")
        be = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pubs = {m: self.create_publisher(Twist, f"/control/autonomous/module_{m}/cmd_vel", 10) for m in (1, 2)}
        self.state = {d: None for d in DEVICES}
        self.locks = {d: StreamLock() for d in DEVICES}
        self.joints = None
        for d in DEVICES:
            self.create_subscription(DeviceState, f"/endovascular/{d}/state", lambda msg, d=d: self._on_state(d, msg), be)
        self.create_subscription(JointState, "/ng3/joint_states", lambda msg: setattr(self, "joints", msg), be)
        self.follower = self.create_client(SetBool, "/endovascular/follower/enabled")
        self.reset_page = self.create_client(Trigger, "/endovascular/simulation/reset_checked")
        self.episode_reset = self.create_client(Trigger, "/hcmr/episode/reset")

    def _on_state(self, d, msg):
        if self.locks[d].offer(msg):
            self.state[d] = msg

    def command(self, gw_trans, gw_rot, cath_trans, cath_rot, clip=1.0):
        for module, (v, w) in ((1, (gw_trans, gw_rot)), (2, (cath_trans, cath_rot))):
            msg = Twist()
            msg.linear.x = float(max(-clip, min(clip, v))) * MAX_TRANS_MM_S
            msg.angular.z = float(max(-clip, min(clip, w))) * MAX_ROT_RAD_S
            self.pubs[module].publish(msg)

    def call(self, client, request, timeout=10.0):
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"service {client.srv_name} not available")
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if future.result() is None:
            raise RuntimeError(f"service {client.srv_name} timed out")
        return future.result()

    def spin_for(self, seconds, action=None, rate=20.0):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if action is not None:
                self.command(*action)
            rclpy.spin_once(self, timeout_sec=1.0 / rate)

    def reset(self):
        self.spin_for(0.5, action=(0, 0, 0, 0))
        if self.episode_reset.wait_for_service(timeout_sec=2.0):
            r = self.call(self.episode_reset, Trigger.Request(), timeout=30.0)
            if not r.success:
                raise RuntimeError(f"episode reset failed: {r.message}")
            return self._settle(r.message)
        # fallback without the coordinator: home the robot, then reset the page and the follower
        self.wait_joints()
        self.go_home(HOME)
        self.call(self.follower, SetBool.Request(data=False))
        r = self.call(self.reset_page, Trigger.Request())
        if not r.success:
            raise RuntimeError(f"browser reset failed: {r.message}")
        self.spin_for(1.0)
        r = self.call(self.follower, SetBool.Request(data=True))
        return self._settle(r.message + " (manual reset)")

    def wait_joints(self, timeout=10.0):
        deadline = time.monotonic() + timeout
        while self.joints is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.joints is None:
            raise RuntimeError("no /ng3/joint_states received")

    def _settle(self, message):
        self.state = {d: None for d in DEVICES}
        for lk in self.locks.values():
            lk.clear()
        self.spin_for(1.5, action=(0, 0, 0, 0))
        for d in DEVICES:
            self.locks[d].lock()
        self.state = {d: None for d in DEVICES}
        self.spin_for(0.3, action=(0, 0, 0, 0))
        deadline = time.monotonic() + 10.0
        while (any(self.state[d] is None for d in DEVICES) or self.joints is None) and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if any(self.state[d] is None for d in DEVICES):
            raise RuntimeError("no DeviceState received; is the physics page running?")
        return message

    def joint_vector(self):
        p = self.joints.position
        return [p[0], p[1], p[2], p[3]]

    def go_home(self, home, timeout=90.0, rate=20.0):
        """Closed-loop return of both modules to ``home`` (m, rad, m, rad) with the follower off."""
        self.call(self.follower, SetBool.Request(data=False))
        deadline = time.monotonic() + timeout
        err = [0.0] * 4
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=1.0 / rate)
            q = self.joint_vector()
            err = [h - x for h, x in zip(home, q)]
            if abs(err[0]) < 5e-4 and abs(err[2]) < 5e-4 and abs(err[1]) < 0.02 and abs(err[3]) < 0.02:
                break
            # bridge clamps at 20 mm/s and 5 rad/s; send in mm/s and rad/s, normalised by command()'s scale
            v1 = max(-20.0, min(20.0, 2000.0 * err[0])) / MAX_TRANS_MM_S
            v2 = max(-20.0, min(20.0, 2000.0 * err[2])) / MAX_TRANS_MM_S
            w1 = max(-2.0, min(2.0, 2.0 * err[1])) / MAX_ROT_RAD_S
            w2 = max(-2.0, min(2.0, 2.0 * err[3])) / MAX_ROT_RAD_S
            self.command(v1, w1, v2, w2, clip=2.0)
        self.spin_for(0.5, action=(0, 0, 0, 0))
        return err


def row_of(msg: DeviceState, prefix: str) -> dict:
    p, q = msg.tip_pose.position, msg.tip_pose.orientation
    return {
        f"{prefix}_insertion_m": msg.insertion_depth, f"{prefix}_roll_rad": msg.shaft_rotation,
        f"{prefix}_tip_x": p.x, f"{prefix}_tip_y": p.y, f"{prefix}_tip_z": p.z,
        f"{prefix}_qx": q.x, f"{prefix}_qy": q.y, f"{prefix}_qz": q.z, f"{prefix}_qw": q.w,
        f"{prefix}_peak_force_n": msg.peak_force, f"{prefix}_contact_force_n": msg.contact_force,
        f"{prefix}_tip_force_n": msg.tip_force, f"{prefix}_contacts": msg.contact_count,
        f"{prefix}_sim_time": msg.simulated_time,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("actions")
    ap.add_argument("out")
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--rate", type=float, default=20.0)
    ap.add_argument("--no-home", action="store_true", help="leave the robot where the run ended (no reset after)")
    args = ap.parse_args()
    actions = [[float(r[k]) for k in ("gw_trans", "gw_rot", "cath_trans", "cath_rot")]
               for r in csv.DictReader(open(args.actions, encoding="utf-8"))]
    rclpy.init()
    node = Probe()
    rows = []
    try:
        print("reset:", node.reset(), flush=True)
        home = node.joint_vector()
        print("joints:", list(node.joints.name), [round(x, 4) for x in home], flush=True)
        t0 = time.monotonic()
        total = len(actions) * args.dt
        while True:
            t = time.monotonic() - t0
            if t >= total + 1.0:
                break
            k = int(t / args.dt)
            action = actions[k] if k < len(actions) else [0.0, 0.0, 0.0, 0.0]
            node.command(*action)
            rclpy.spin_once(node, timeout_sec=1.0 / args.rate)
            row = {"t_s": round(t, 4), "step": k, "gw_trans": action[0], "gw_rot": action[1],
                   "cath_trans": action[2], "cath_rot": action[3]}
            for d, prefix in (("guidewire", "gw"), ("catheter", "cath")):
                if node.state[d] is not None:
                    row.update(row_of(node.state[d], prefix))
            if node.joints is not None and len(node.joints.position) >= 4:
                row.update({"joint_m1_m": node.joints.position[0], "joint_m1_rad": node.joints.position[1],
                            "joint_m2_m": node.joints.position[2], "joint_m2_rad": node.joints.position[3]})
            rows.append(row)
        node.spin_for(1.0, action=(0, 0, 0, 0))
        if not args.no_home:
            print("reset after run:", node.reset(), flush=True)
    finally:
        node.command(0, 0, 0, 0)
        node.destroy_node()
        rclpy.shutdown()
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("t_s", "step"), k))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"recorded {len(rows)} samples over {rows[-1]['t_s'] if rows else 0:.1f} s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
