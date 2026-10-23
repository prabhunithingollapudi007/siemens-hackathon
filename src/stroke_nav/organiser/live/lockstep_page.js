// Lockstep harness for the organiser's browser pages (catheter_guidewire_ros2.html and
// noisy_visualization_ros2.html), injected before their scripts run
// (Page.addScriptToEvaluateOnNewDocument). Their code is not modified:
//
// * requestAnimationFrame only queues callbacks; __ls.frame() runs one animation frame. The physics
//   page advances a fixed FRAME_DT (1/60 s) per frame, so N frames are N/60 s of simulated time
//   however long they take.
// * WebSocket is replaced by an in-page socket. The pages speak their usual ROS-bridge protocol
//   (ws_port_base + 0..4: guidewire state/command, catheter state/command, noisy viewer) to the
//   Python side through __ls instead of to the EndoNav containers.
// * In the noisy viewer, performance.now() is a virtual clock so its 50 ms image throttle follows
//   simulated time.
(() => {
  const queue = [];
  window.requestAnimationFrame = cb => { queue.push(cb); return queue.length; };
  window.cancelAnimationFrame = () => {};

  const viewer = location.pathname.includes('noisy_visualization');
  let clockMs = 1000;  // never 0: the viewer's render(scheduleNext = true) takes the rAF timestamp
  if (viewer) performance.now = () => clockMs;

  const sockets = {};  // port -> socket
  class LockstepSocket extends EventTarget {
    constructor(url) {
      super();
      this.url = String(url);
      this.port = Number(new URL(this.url).port);
      this.readyState = LockstepSocket.CONNECTING;
      this.sent = [];
      this.onopen = this.onmessage = this.onclose = this.onerror = null;
      sockets[this.port] = this;
      setTimeout(() => {
        this.readyState = LockstepSocket.OPEN;
        const e = new Event('open');
        this.dispatchEvent(e);
        if (this.onopen) this.onopen(e);
      }, 0);
    }
    send(data) { this.sent.push(data); }
    close() {}
    deliver(obj) {
      const e = new MessageEvent('message', { data: JSON.stringify(obj) });
      this.dispatchEvent(e);
      if (this.onmessage) this.onmessage(e);
    }
    drain() { return this.sent.splice(0); }
  }
  LockstepSocket.CONNECTING = 0; LockstepSocket.OPEN = 1; LockstepSocket.CLOSING = 2; LockstepSocket.CLOSED = 3;
  window.WebSocket = LockstepSocket;

  const base = () => Number(new URLSearchParams(location.search).get('ws_port_base') || 18765);
  const sock = offset => sockets[base() + offset];

  const errors = [];
  function frame() {
    const callbacks = queue.splice(0);
    const now = performance.now();
    for (const cb of callbacks) {
      try { cb(now); } catch (e) { errors.push(String(e && e.stack || e)); if (errors.length > 20) errors.shift(); }
    }
    return callbacks.length;
  }

  // Physics-page state messages, compacted: the guidewire message also carries a JPEG of the
  // physics view and contact lists, which the noise layer and the RL env do not use.
  function compact(raw) {
    const m = JSON.parse(raw);
    if (!m.type || !m.type.endsWith('_state')) return null;
    const seg = new Array(m.segments.length * 7);
    m.segments.forEach((s, i) => {
      for (let k = 0; k < 3; k++) seg[i * 7 + k] = s.position[k];
      for (let k = 0; k < 4; k++) seg[i * 7 + 3 + k] = s.orientation[k];
    });
    const f = m.force || {};
    return { device: m.type.replace('_state', ''), mode: m.control_mode, t: m.simulated_time, frame: m.physics_frame,
             ins: m.insertion_depth, rot: m.shaft_rotation, tip: m.tip_position, quat: m.tip_quaternion,
             vel: m.tip_velocity, omg: m.tip_angular_velocity, proximal: f.proximal || 0, tip_force: f.tip || 0,
             axial: f.axial || 0, peak: f.peak || 0, contacts: f.contacts || 0, contact: f.contact || 0, seg };
  }

  function drainStates() {
    const out = [];
    for (const offset of [0, 2]) {
      const s = sock(offset);
      if (!s) continue;
      for (const raw of s.drain()) {
        const c = compact(raw);
        if (c) out.push(c);
      }
    }
    return out;
  }

  window.__ls = {
    connected: () => [0, 1, 2, 3, 4].filter(o => sock(o) && sock(o).readyState === 1).map(o => base() + o),
    frame,
    errors: () => errors.slice(),
    queued: () => queue.length,
    // Skip the physics page's per-state JPEG of its own view (toDataURL), unused here.
    skipCapture(on) {
      if (on && !HTMLCanvasElement.prototype.__lsToDataURL) {
        HTMLCanvasElement.prototype.__lsToDataURL = HTMLCanvasElement.prototype.toDataURL;
        HTMLCanvasElement.prototype.toDataURL = function (...args) {
          return window.__lsSkip ? 'data:image/jpeg;base64,' : this.__lsToDataURL(...args);
        };
      }
      window.__lsSkip = !!on;
    },
    // Physics page: reset both rods, return the states it publishes.
    reset() {
      sock(0).drain(); sock(2).drain();
      sock(1).deliver({ type: 'reset' });
      sock(1).drain(); sock(3).drain();
      return drainStates();
    },
    // Physics page: one frame per plan row [gw_ins_mm, gw_rot_deg, cath_ins_mm, cath_rot_deg]
    // (DeviceCommand setpoints, as the joint-state follower sends them); returns published states.
    step(plan) {
      const out = [];
      for (const p of plan) {
        sock(1).deliver({ type: 'command', insertion_setpoint: p[0], rotation_setpoint: p[1], enabled: true });
        sock(3).deliver({ type: 'command', insertion_setpoint: p[2], rotation_setpoint: p[3], enabled: true });
        frame();
        for (const s of drainStates()) out.push(s);
      }
      sock(1).drain(); sock(3).drain();
      return out;
    },
    // Noisy viewer: deliver noisy_device_state messages, render one frame, return its JPEG (base64).
    render(messages) {
      const s = sock(4);
      s.drain();
      for (const m of messages) s.deliver(m);
      clockMs += 1000;
      frame();
      const sent = s.drain();
      for (let i = sent.length - 1; i >= 0; i--) {
        const m = JSON.parse(sent[i]);
        if (m.type === 'noisy_image') return m.data_base64;
      }
      return null;
    },
  };
})();
