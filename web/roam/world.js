// The world of spots: each spot's mesh placed by origin (x, z), heading and scale, in the frame of the current spot.
//
// Frames and re-rooting. Spot.scale and Spot.distance are relative to the parent only (src/vecworld/roam/spot.py).
// The world keeps every spot in the CURRENT spot's frame: the current spot sits at (0, 0) with scale 1. A child is
// placed in its parent's frame (placeChild), and when the player crosses into it the world re-roots on it: every
// origin, scale and the player's position are re-expressed in the child's frame, so nothing accumulates along a
// journey. The re-root is invisible: the world's view scale k takes over the child's scale at that instant, and then
// eases back to 1 about the player (a slow, slight zoom of the scenery when depth scales differ between spots).
//
// Headings are world bearings in degrees, 0 = north (−z, the first spot's forward), clockwise (east = +x). Every
// spot's mesh is vertically placed so its fitted ground sits at y = 0, where the character walks.
import * as THREE from "three";

export const NEAR = 4, FAR = 50, VSPAN = Math.PI / 2;
export const GX = 512, GY = 128;               // the depth mesh grid (spec: 512×128)
export const LIMIT = 0.55 * NEAR;              // walkable radius around the current spot's origin (spike)
export const FADE_MS = 600;                    // crossfade length at a switch
export const EDGE_AT = 0.98;                   // at |p| ≥ EDGE_AT·LIMIT the player is at the edge of the current spot
const HYSTERESIS = 0.08;                       // a spot must be this much nearer than the current one to switch to
const AHEAD = 45;                              // degrees: a spot is "ahead" when its bearing is this close to travel
const SCALE_EASE = 0.6;                        // per second: how fast the view scale returns to 1 after a re-root
const RAD = Math.PI / 180;

export function bearingVec(deg) { return { x: Math.sin(deg * RAD), z: -Math.cos(deg * RAD) }; }

const angleBetween = (ax, az, bx, bz) =>
  Math.acos(Math.max(-1, Math.min(1, (ax * bx + az * bz) / (Math.hypot(ax, az) * Math.hypot(bx, bz) || 1)))) / RAD;

// The placement of a child spot in the parent's (current) frame: origin = the parent origin + distance along
// exitHeading (distance and scale being parent-relative, they are multiplied by the parent's scale in this frame).
export function childPlacement(parent, exitHeading, distance, scale) {
  const v = bearingVec(exitHeading), d = distance * parent.scale;
  return { origin: { x: parent.origin.x + v.x * d, z: parent.origin.z + v.z * d }, heading: exitHeading,
           scale: parent.scale * scale };
}

const groundVert = /* glsl */`
  varying vec2 vp;
  void main() { vp = position.xy; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
// The ground plane, textured by projecting the panorama onto it: a ground point at horizontal distance ρ and height
// `ground` is the pixel on the ray of row v with (0.5 − v)·VSPAN·ρ = ground (the mesh's own mapping). Inside ρ0 the
// panorama never saw the ground (it spans only 45° down), so that disc is filled with copies of the real projected
// ground from further out, translated (not radially unwrapped, which streaks), weighted toward each point's own side.
const groundFrag = /* glsl */`
  uniform sampler2D map; uniform float ground; uniform float opacity;
  varying vec2 vp;
  const float PI = 3.141592653589793, VSPAN = PI / 2.0;
  vec3 proj(vec2 p) {                            // p = (x, -z)·(-1): plane coords, see below
    float rho = max(length(p), 1e-3);
    float u = atan(p.x, p.y) / (2.0 * PI) + 0.5;
    float t = 0.5 + ground / (VSPAN * rho);      // texture t = 1 - v
    float ua = fract(u + 0.5) - 0.5;             // continuous across the u = 0/1 seam, for the derivatives
    vec2 g = vec2(abs(dFdx(u)) < abs(dFdx(ua)) ? dFdx(u) : dFdx(ua), dFdx(t));
    vec2 h = vec2(abs(dFdy(u)) < abs(dFdy(ua)) ? dFdy(u) : dFdy(ua), dFdy(t));
    return textureGrad(map, vec2(u, max(t, 0.002)), g, h).rgb;
  }
  void main() {
    // PlaneGeometry lies in xy; rotated −90° about x, plane y becomes world −z: forward (−z) is +y here.
    vec2 p = vec2(vp.x, vp.y);
    float rho = length(p), rho0 = -2.0 * ground / VSPAN;
    // every sample is taken unconditionally: texture derivatives are undefined in divergent branches
    vec3 c = proj(p);
    vec2 n = p / max(rho, 1e-3);
    vec3 acc = vec3(0.0); float ws = 0.0;
    for (int k = 0; k < 6; k++) {
      float a = float(k) * PI / 3.0 + 0.5;
      vec2 dir = vec2(sin(a), cos(a));
      float f = smoothstep(0.0, 0.6, rho);         // at the very centre every copy weighs the same
      float w = 1e-4 + f * pow(0.5 + 0.5 * dot(n, dir), 12.0) + (1.0 - f);
      acc += w * proj(p + dir * (1.2 * rho0)); ws += w;   // lands just outside rho0, where the pano is sharpest
    }
    c = mix(acc / ws, c, smoothstep(rho0 - 0.8, rho0 + 0.2, rho));   // done by where the mesh's bottom edge starts
    gl_FragColor = vec4(c, opacity);
    #include <colorspace_fragment>
  }`;

const fadeFrag = /* glsl */`
  uniform sampler2D a; uniform sampler2D b; uniform float t; uniform vec3 haze; uniform float hazeAmt;
  varying vec2 vUv;
  void main() {
    vec4 c = mix(texture2D(a, vUv), texture2D(b, vUv), t);
    c.rgb = mix(c.rgb, haze, hazeAmt * sin(3.141592653589793 * t));
    gl_FragColor = c;
    #include <colorspace_fragment>
  }`;

// A soft, low-resolution copy of the panorama for the far backdrop: the cut gaps show a blur, not ghost trunks.
function blurredTexture(image) {
  const c = document.createElement("canvas"); c.width = 256; c.height = 64;
  const g = c.getContext("2d");
  g.filter = "blur(1.5px)";
  for (const dx of [-256, 0, 256]) g.drawImage(image, dx, 0, 256, 64);   // wrap so the blur crosses the seam
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.wrapS = THREE.RepeatWrapping;
  return t;
}

// The backdrop at FAR: the panorama's cylinder (same mapping as the mesh) continued above into a dome of its top row.
function backdropGeometry() {
  const cols = 96, rows = 24, cap = 8, pos = [], uv = [], idx = [];
  const topY = 0.5 * VSPAN * FAR, topAng = Math.atan2(topY, FAR);
  for (let y = 0; y <= rows + cap; y++) for (let x = 0; x <= cols; x++) {
    const u = x / cols, th = (u - 0.5) * Math.PI * 2;
    let r, h, v;
    if (y < cap) {                                        // dome rows, zenith first
      const ang = Math.PI / 2 - (Math.PI / 2 - topAng) * (y / cap);
      const R = Math.hypot(FAR, topY);
      r = Math.max(1e-3, Math.cos(ang) * R); h = Math.sin(ang) * R; v = 1;
    } else {
      const vv = (y - cap) / rows; r = FAR; h = (0.5 - vv) * VSPAN * FAR; v = 1 - vv;
    }
    pos.push(Math.sin(th) * r, h, -Math.cos(th) * r); uv.push(u, Math.min(v, 0.995));
  }
  for (let y = 0; y < rows + cap; y++) for (let x = 0; x < cols; x++) {
    const a = y * (cols + 1) + x, b = a + 1, c = a + cols + 1, e = c + 1;
    idx.push(a, c, b, b, c, e);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2)); g.setIndex(idx);
  return g;
}

export class World {
  constructor(scene, workerUrl = new URL("./spotmesh.worker.js", import.meta.url)) {
    this.scene = scene;
    this.root = new THREE.Group(); scene.add(this.root);
    this.worker = new Worker(workerUrl);
    this.jobs = new Map(); this.nextJob = 0; this.workerBuilt = 0;
    this.onError = err => console.error(err);   // worker failures no pending build can carry (the page wires it up)
    this.worker.onmessage = e => {
      const job = this.jobs.get(e.data?.id);
      if (!job) { this.onError(new Error(`spot mesh worker replied for unknown job ${JSON.stringify(e.data?.id)}`)); return; }
      this.jobs.delete(e.data.id);
      if (e.data.error) job.reject(new Error(e.data.error)); else { this.workerBuilt++; job.resolve(e.data); }
    };
    const fail = msg => {
      const err = new Error(msg);
      if (this.jobs.size) { for (const j of this.jobs.values()) j.reject(err); this.jobs.clear(); } else this.onError(err);
    };
    this.worker.onerror = e => { e.preventDefault?.(); fail(`spot mesh worker: ${e.message || "failed to load"}`); };
    this.worker.onmessageerror = () => fail("spot mesh worker: a reply could not be deserialised");
    this.spots = [];
    this.current = null;
    this.edge = null;               // {bearing} while the player is at the edge with no spot ahead (fog goes here)
    this._r = 0;                    // the player's distance from the current origin last frame
    this.fade = null;               // {from, to, elapsed, ms} (milliseconds)
    this.k = 1;                     // view scale (see the header)
    this.abs = { x: 0, z: 0, s: 1 };  // the current frame inside the first spot's frame (debug: absolute positions)
    this.haze = new THREE.Color("#dfe3ea"); this.hazeAmt = 0;
    this._rt = null; this._fadeMat = null; this.override = null;
  }

  // Mesh one spot (panorama image + depth image) in the worker. Returns a spot ready for addSpot.
  async buildSpot(image, depthImage, meta = {}) {
    const c = document.createElement("canvas"); c.width = GX; c.height = GY;
    const g = c.getContext("2d", { willReadFrequently: true }); g.drawImage(depthImage, 0, 0, GX, GY);
    const rgba = g.getImageData(0, 0, GX, GY).data, depth = new Uint8Array(GX * GY);
    for (let i = 0; i < depth.length; i++) depth[i] = rgba[i * 4];
    const id = this.nextJob++;
    const m = await new Promise((resolve, reject) => {
      this.jobs.set(id, { resolve, reject });
      this.worker.postMessage({ id, depth, gx: GX, gy: GY }, [depth.buffer]);
    });
    const tex = new THREE.Texture(image); tex.colorSpace = THREE.SRGBColorSpace; tex.wrapS = THREE.RepeatWrapping;
    tex.anisotropy = 4; tex.needsUpdate = true;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(m.position, 3));
    geo.setAttribute("uv", new THREE.BufferAttribute(m.uv, 2));
    geo.setAttribute("color", new THREE.BufferAttribute(m.color, 4, true));
    geo.setIndex(new THREE.BufferAttribute(m.index, 1));
    const group = new THREE.Group();
    const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
      map: tex, vertexColors: true, side: THREE.DoubleSide, alphaToCoverage: true, transparent: false }));
    const back = new THREE.Mesh(backdropGeometry(), new THREE.MeshBasicMaterial({
      map: blurredTexture(image), side: THREE.DoubleSide, depthWrite: false }));
    back.renderOrder = -2;
    const groundMat = new THREE.ShaderMaterial({
      uniforms: { map: { value: tex }, ground: { value: m.ground }, opacity: { value: 1 } },
      vertexShader: groundVert, fragmentShader: groundFrag,
      polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 8 });
    const plane = new THREE.Mesh(new THREE.CircleGeometry(0.9 * FAR, 96), groundMat);
    plane.rotation.x = -Math.PI / 2; plane.position.y = m.ground; plane.renderOrder = -1;
    group.add(back, plane, mesh);
    return { group, mesh, ground: m.ground, triangles: m.triangles, texture: tex, meta, index: meta.index };
  }

  // Put a built spot into the world at origin (x, z) of the current frame, its u = 0.5 facing `heading`, × scale.
  addSpot(spot, origin, heading, scale) {
    const e = Object.assign(spot, { origin: { x: origin.x, z: origin.z }, heading, scale });
    this.root.add(e.group); this._pose(e); this.spots.push(e);
    if (!this.current) { this.current = e; e.group.visible = true; }
    return e;
  }

  // Place `child` (optional) relative to the parent world entry; returns the placement (and the entry if added).
  placeChild(parent, exitHeading, distance, scale, child) {
    const p = childPlacement(parent, exitHeading, distance, scale);
    if (!child) return p;
    child.parent = parent;
    return this.addSpot(child, p.origin, p.heading, p.scale);
  }

  _pose(e) {
    e.group.position.set(e.origin.x, -e.scale * e.ground, e.origin.z);
    e.group.rotation.y = -e.heading * RAD;
    e.group.scale.setScalar(e.scale);
  }

  // The walkable clamp alone (idempotent; update() applies it first). Returns the player's distance from the origin.
  constrain(p) {
    const r = p.length();
    if (r > LIMIT && r > this._r) p.multiplyScalar(Math.max(LIMIT, this._r) / r);
    return p.length();
  }

  // Keep the player in the world and choose its spot. p: position (frame units, THREE.Vector2 of x, z), adjusted in
  // place by the walkable clamp and on a re-root. dir: the unit direction of travel (x, z), or null when standing.
  //  - Clamp: the player can't move outward past LIMIT (or past where it already is, just after a switch at the edge
  //    into a spot whose origin is further than LIMIT away; walking inward is always allowed).
  //  - Switch (crossfade + re-root) to another spot when moving toward it and either inside its walkable radius and
  //    nearer to it than to the current origin (the bisector), or at the edge with it ahead: its bearing within AHEAD°
  //    of travel, on the player's side. (Inside its radius: after an edge switch between far-apart spots the player is
  //    nearer the old origin than the new one, and must not be switched straight back.)
  //  - At the edge with no spot ahead, world.edge = {bearing} (where the game shows dream fog and waits).
  // Returns the spot switched to, or null.
  update(p, dir = null) {
    if (!this.current) return null;
    const r = this.constrain(p);
    const moving = dir && (dir.x || dir.y);
    const dist = e => Math.hypot(p.x - e.origin.x, p.y - e.origin.z);
    const toward = e => moving && dir.x * (e.origin.x - p.x) + dir.y * (e.origin.z - p.y) > 0;
    let best = null, bd = dist(this.current) - HYSTERESIS;
    for (const e of this.spots) {
      if (e !== this.current && toward(e) && dist(e) <= LIMIT && dist(e) < bd) { best = e; bd = dist(e); }
    }
    const atEdge = r >= EDGE_AT * LIMIT;
    if (!best && atEdge && moving) {
      let ba = AHEAD;
      for (const e of this.spots) {
        if (e === this.current || e.origin.x * p.x + e.origin.z * p.y <= 0) continue;
        const a = angleBetween(dir.x, dir.y, e.origin.x, e.origin.z);
        if (a <= ba) { best = e; ba = a; }
      }
    }
    const outward = moving && dir.x * p.x + dir.y * p.y > 0;
    this.edge = !best && atEdge && outward ? { bearing: (Math.atan2(p.x, -p.y) / RAD + 360) % 360 } : null;
    if (!best) { this._r = r; return null; }
    this.enter(best, p);
    return best;
  }

  // Switch to spot e now, wherever the player stands: re-root on it and crossfade from the current spot. update()
  // does this at a switch; the game also does it when the server hands it the next spot at an edge.
  enter(e, p) {
    if (e === this.current) return;
    const from = this.current;
    this._reroot(e, p);
    this.crossfade(e, FADE_MS, from);
    this._r = p.length(); this.edge = null;
    this.prune();
  }

  _reroot(c, p) {
    const o = { ...c.origin }, s = c.scale;
    this.abs.x += this.abs.s * o.x; this.abs.z += this.abs.s * o.z; this.abs.s *= s;
    for (const e of this.spots) {
      e.origin = { x: (e.origin.x - o.x) / s, z: (e.origin.z - o.z) / s }; e.scale /= s; this._pose(e);
    }
    if (p) p.set((p.x - o.x) / s, (p.y - o.z) / s);
    this.k *= s; this.root.scale.setScalar(this.k);   // nothing on screen moves at the re-root
    this.current = c;
  }

  // Jump straight to a spot (no fade): the player goes to its origin.
  jumpTo(e, p) {
    this._reroot(e, p);
    if (p) p.set(0, 0);
    this.fade = null; this._r = 0; this.edge = null;
    this.prune();
  }

  // Fade the view from one spot to another (render() draws both during it); `to` becomes current at once.
  crossfade(to, ms = FADE_MS, from = this.current) {
    this.fade = { from, to, elapsed: 0, ms };   // advanced by tick(dt), so a paused game pauses it
    this.current = to;
  }

  get fading() { return this.fade ? Math.min(1, this.fade.elapsed / this.fade.ms) : null; }

  // Advance the view scale's ease back to 1; finish a done crossfade. Call once per frame.
  tick(dt) {
    const k1 = this.k + (1 - this.k) * Math.min(1, dt * SCALE_EASE);
    this.k = Math.abs(k1 - 1) < 1e-4 ? 1 : k1;
    this.root.scale.setScalar(this.k);
    if (this.fade) this.fade.elapsed += dt * 1000;
    if (this.fade && this.fading >= 1) this.fade = null;
  }

  // Free every spot but the current one, the one being faded from (normally its parent, so walking back still
  // works until the next switch) and the current one's placed children. Runs on every re-root.
  prune() {
    for (const e of [...this.spots]) {
      if (e === this.current || (this.fade && e === this.fade.from) || e.parent === this.current) continue;
      this.remove(e);
    }
  }

  remove(e) {
    this.root.remove(e.group);
    e.group.traverse(o => { if (o.isMesh) { o.geometry.dispose(); o.material.map?.dispose(); o.material.dispose(); } });
    this.spots = this.spots.filter(s => s !== e);
    if (this.override === e) this.override = null;
  }

  render(renderer, scene, camera) {
    const show = s => { for (const e of this.spots) e.group.visible = e === s; };
    if (this.override) { show(this.override); renderer.render(scene, camera); return; }
    if (!this.fade) { show(this.current); renderer.render(scene, camera); return; }
    const t0 = this.fading, t = t0 * t0 * (3 - 2 * t0);
    const size = renderer.getDrawingBufferSize(new THREE.Vector2());
    if (!this._rt || this._rt[0].width !== size.x || this._rt[0].height !== size.y) {
      this._rt?.forEach(r => r.dispose());
      this._rt = [0, 1].map(() => new THREE.WebGLRenderTarget(size.x, size.y, { samples: 4, type: THREE.HalfFloatType }));
      this._fadeMat ??= new THREE.ShaderMaterial({
        uniforms: { a: { value: null }, b: { value: null }, t: { value: 0 }, haze: { value: this.haze },
                    hazeAmt: { value: 0 } },
        vertexShader: "varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }",
        fragmentShader: fadeFrag, depthTest: false, depthWrite: false });
      this._quad ??= new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this._fadeMat);
      this._quad.frustumCulled = false;
      this._quadScene ??= new THREE.Scene(); this._quadScene.add(this._quad);
      this._quadCam ??= new THREE.Camera();
    }
    const prev = renderer.getRenderTarget();
    show(this.fade.from); renderer.setRenderTarget(this._rt[0]); renderer.render(scene, camera);
    show(this.fade.to); renderer.setRenderTarget(this._rt[1]); renderer.render(scene, camera);
    renderer.setRenderTarget(prev);
    const u = this._fadeMat.uniforms;
    u.a.value = this._rt[0].texture; u.b.value = this._rt[1].texture; u.t.value = t; u.hazeAmt.value = this.hazeAmt;
    renderer.render(this._quadScene, this._quadCam);
  }
}
