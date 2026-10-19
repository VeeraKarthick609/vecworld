// The character and the third-person orbit camera, ported from the free-roam spike (/tmp/pano3d/index.html).
// Positions are in the world's current frame (a THREE.Vector2 of x, z); k is the world's view scale (world.js),
// so the character's ground speed on screen always matches its stride.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

export const MODEL = new URL("../vendor/RobotExpressive.glb", import.meta.url).href;
export const CLIPS = { idle: "Idle", walk: "Walking" };   // clip-name mapping, so another character can be swapped in
export const HERO_SCALE = 0.11, STRIDE = 1.1, RUN = 1.8, TURN = 10, ACCEL = 5;
export const CAM_DIST = 1.9, CAM_H = 0.55, LOOK_H = 0.35, PITCH_MIN = -0.1, PITCH_MAX = 0.45;

// Bearing (degrees, 0 = −z, clockwise) of a direction (dx, dz), and back.
export const bearingOf = (dx, dz) => ((Math.atan2(dx, -dz) * 180 / Math.PI) + 360) % 360;

export class Hero {
  static async load(scene, url = MODEL) {
    return new Hero(scene, await new GLTFLoader().loadAsync(url));
  }

  constructor(scene, gltf) {
    this.object = gltf.scene; this.object.scale.setScalar(HERO_SCALE); scene.add(this.object);
    this.shadow = new THREE.Mesh(new THREE.CircleGeometry(0.16, 32),
      new THREE.MeshBasicMaterial({ color: 0, transparent: true, opacity: 0.35, depthWrite: false }));
    this.shadow.rotation.x = -Math.PI / 2; this.shadow.renderOrder = 1; scene.add(this.shadow);
    this.mixer = new THREE.AnimationMixer(this.object);
    const act = n => this.mixer.clipAction(gltf.animations.find(a => a.name === n));
    this.idle = act(CLIPS.idle); this.walk = act(CLIPS.walk);
    this.idle.play(); this.walk.play(); this.walk.setEffectiveWeight(0);
    this.walkSpeed = STRIDE / this.walk.getClip().duration;   // ground speed matched to the stride
    this.pos = new THREE.Vector2(0, 0);
    this.face = Math.PI;      // rotation.y; π faces −z (north)
    this.speed = 0;           // display units per second
    this.walked = 0;          // display distance walked in total
  }

  // Bearing the character faces, in degrees.
  get heading() { return bearingOf(Math.sin(this.face), Math.cos(this.face)); }

  // input: {fwd, side} in −1..1 and run; camYaw from the orbit camera. Moves pos (frame units).
  update(dt, input, camYaw, k = 1) {
    const moving = input.fwd !== 0 || input.side !== 0;
    const target = moving ? this.walkSpeed * (input.run ? RUN : 1) : 0;
    this.speed += (target - this.speed) * Math.min(1, dt * ACCEL);
    if (moving) {
      const a = camYaw + Math.atan2(input.side, input.fwd), dx = -Math.sin(a), dz = -Math.cos(a);
      this.pos.x += dx * this.speed * dt / k; this.pos.y += dz * this.speed * dt / k;
      this.walked += this.speed * dt;
      let d = Math.atan2(dx, dz) - this.face; d = Math.atan2(Math.sin(d), Math.cos(d));
      this.face += d * Math.min(1, dt * TURN);
    }
    const w = Math.min(1, this.speed / this.walkSpeed);
    this.walk.setEffectiveWeight(w); this.idle.setEffectiveWeight(1 - w);
    this.walk.timeScale = Math.max(1, this.speed / this.walkSpeed);
    this.mixer.update(dt);
  }

  // Place the model at display position (k·pos) on the ground (y = 0).
  place(k = 1) {
    this.object.position.set(k * this.pos.x, 0, k * this.pos.y); this.object.rotation.y = this.face;
    this.shadow.position.set(k * this.pos.x, 0.01, k * this.pos.y);
  }
}

// WASD/arrows (+ Shift) held keys as {fwd, side, run}.
export class Keys {
  constructor(target = window) {
    this.down = new Set();
    target.addEventListener("keydown", e => this.down.add(e.key.toLowerCase()));
    target.addEventListener("keyup", e => this.down.delete(e.key.toLowerCase()));
    target.addEventListener("blur", () => this.down.clear());
  }
  get input() {
    const h = k => this.down.has(k);
    return { fwd: (h("w") || h("arrowup")) - (h("s") || h("arrowdown")),
             side: (h("d") || h("arrowright")) - (h("a") || h("arrowleft")), run: h("shift") };
  }
}

// Orbit camera behind and above the character: drag, or click to capture the pointer (Esc releases).
export class OrbitCam {
  constructor(dom) {
    this.yaw = 0; this.pitch = 0.18; this.dragging = false;
    dom.addEventListener("pointerdown", () => { this.dragging = true; dom.requestPointerLock?.(); });
    addEventListener("pointerup", () => (this.dragging = false));
    addEventListener("pointermove", e => {
      if (!this.dragging && document.pointerLockElement !== dom) return;
      this.turn(-e.movementX * 0.004, e.movementY * 0.003);
    });
    addEventListener("keydown", e => { if (e.key === "Escape") document.exitPointerLock?.(); });
  }
  turn(dYaw, dPitch) { this.yaw += dYaw; this.setPitch(this.pitch + dPitch); }
  setPitch(p) { this.pitch = Math.min(PITCH_MAX, Math.max(PITCH_MIN, p)); }
  // The camera yaw that looks along a bearing (degrees).
  static yawFor(bearing) { return -bearing * Math.PI / 180; }
  apply(camera, at) {
    const ch = Math.cos(this.pitch);
    camera.position.set(at.x + Math.sin(this.yaw) * CAM_DIST * ch, at.y + CAM_H + Math.sin(this.pitch) * CAM_DIST,
                        at.z + Math.cos(this.yaw) * CAM_DIST * ch);
    camera.lookAt(at.x, at.y + LOOK_H, at.z);
  }
}
