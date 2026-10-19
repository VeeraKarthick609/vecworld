// Dream Roam: type a place, then roam a dreamed world of 360° painted spots in any direction.
//
// The server holds the journey; the page holds the world around the player (web/roam/world.js). The first spot
// comes from /start. Walking to the edge of a spot with nothing ahead asks /next for the spot that way; while it
// comes the character eases to a stop in a dream fog, and when it lands it is placed beside its parent (distance,
// heading and scale are parent-relative) and the world crossfades into it. The server pre-dreams around the
// player's heading, which the page hints at most every HINT_MS when it has turned by more than HINT_DEG.
//
// The new spot is always placed from the spot the player is in. That is normally the server's spot; after walking
// back into the previous one (possible until the next switch) it is that one, so the new spot appears ahead of the
// player rather than ahead of a spot behind them.
import * as THREE from "three";
import { World, LIMIT } from "./roam/world.js";
import { Hero, Keys, OrbitCam } from "./roam/hero.js";
import { Signposts } from "./roam/signposts.js";
import { Compass, rel } from "./roam/compass.js";

const HINT_MS = 300, HINT_DEG = 20;   // heading hints: at most this often, and only past this much turn
const EASE_FROM = 0.85;               // while a spot is on its way, walking outward stops past this fraction of LIMIT
const LOOK_X = 0.004, LOOK_Y = 0.003; // radians of camera turn per pixel of drag (as OrbitCam)
const STICK_R = 44;                   // px the phone stick's knob travels
const TOAST_MS = 3500;
const STILL = Object.freeze({ fwd: 0, side: 0, run: false });

const $ = id => document.getElementById(id);
const input = $("place");
const chips = [...document.querySelectorAll(".chip")];

// --- the scene ------------------------------------------------------------------

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(2, devicePixelRatio));
renderer.setSize(innerWidth, innerHeight);
const canvas = renderer.domElement;
$("stage").append(canvas);
const scene = new THREE.Scene();
scene.background = new THREE.Color("#0b0d12");
const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.05, 300);
addEventListener("resize", () => {
  renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
});
scene.add(new THREE.HemisphereLight(0xfff3dd, 0x3a4a2a, 2.2));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(-3, 8, 4);
scene.add(sun);

const world = new World(scene);
world.onError = err => { roamErrors.push(String(err.message || err)); toast("The dream stumbled, try again."); };
const signposts = new Signposts(scene);
const compass = new Compass($("compass"));
// OrbitCam's own listeners are left inert on a detached element: the page handles the pointer itself (below),
// because a pointer-lock request can reject and touch drags carry no movementX. Its Esc-releases-lock stays.
const cam = new OrbitCam(document.createElement("div"));
const keys = new Keys();
let hero = null;
const ready = Hero.load(scene).then(h => { hero = h; hero.object.visible = hero.shadow.visible = false; });

// --- server -------------------------------------------------------------------

async function api(path, body) {
  let r;
  try {
    r = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new Error("Lost the connection to the dream, try again.");
  }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw Object.assign(new Error(data.message || `Request failed (${r.status})`), { code: data.error });
  return data;
}

let sid = null;
async function session() {
  if (!sid) sid = (await api("/api/session")).session_id;
  return sid;
}
const call = async (what, body) => api(`/api/session/${await session()}/${what}`, body);

function loadImage(url) {
  return new Promise((ok, fail) => {
    const img = new Image();
    img.onload = () => ok(img);
    img.onerror = () => fail(new Error("The dream image didn't load, try again."));
    img.src = url;
  });
}

async function build(v) {
  const [pano, depth] = await Promise.all([loadImage(v.image_url), loadImage(v.depth_url)]);
  return world.buildSpot(pano, depth, { spot_id: v.spot_id });
}

// --- HUD ----------------------------------------------------------------------

let toastTimer;
function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), TOAST_MS);
}

const activeMoods = () => chips.filter(c => c.getAttribute("aria-pressed") === "true").map(c => c.dataset.mood);
const typing = () => document.activeElement === input;
const camBearing = () => ((-cam.yaw * 180 / Math.PI) % 360 + 360) % 360;

// --- the journey ----------------------------------------------------------------

let started = false, starting = false;
let view = null;          // the server's current spot (SpotOut)
let stepping = false;     // a /next is in flight, or its spot is being built
let stuck = false;        // a /next failed at this edge: no retry until the player leaves the edge
let exitBearing = null;   // where the pending /next leads (for the fog)
let cameFrom = null;      // the bearing back the way the player came
let hint = { at: -Infinity, heading: 0 }, hintCount = 0;
const hints = [];         // the last hints sent, {t, heading}, for the tests

// The server's spot v is now `entry` in the world: signposts, compass and arrival follow it.
function settle(v, entry) {
  view = v;
  signposts.set(entry, v.neighbours);
  compass.set(v.neighbours, v.place, cameFrom);
  if (v.arrived) toast(`Arrived at ${v.arrived}`);
}

async function begin(place) {
  starting = true;
  input.disabled = true;
  try {
    await ready;
    const v = await call("start", { place, moods: activeMoods() });
    const s = await build(v);
    world.addSpot(s, { x: 0, z: 0 }, v.heading, 1);
    hero.pos.set(0, 0);
    hero.object.visible = hero.shadow.visible = true;
    cam.yaw = OrbitCam.yawFor(v.heading);
    hint = { at: -Infinity, heading: v.heading };   // the server pre-dreams around the spot heading to begin with
    settle(v, s);
    started = true;
    $("intro").hidden = true;
    $("compass").hidden = $("keys").hidden = $("stick").hidden = false;
    document.body.classList.add("roaming");
    input.value = "";
    input.placeholder = "drift toward another place…";
  } catch (err) {
    toast(err.message);
  } finally {
    starting = false;
    input.disabled = false;
    if (started) input.blur();
  }
}

async function step(exit) {
  stepping = true;
  exitBearing = exit;
  const sent = hintCount;
  try {
    const v = await call("next", { exit_heading: exit });
    if (hintCount === sent) hint = { at: performance.now(), heading: exit };   // the server now pre-dreams this way
    const s = await build(v);
    world.placeChild(world.current, v.heading, v.distance, v.scale, s);
    world.enter(s, hero.pos);
    cameFrom = (v.heading + 180) % 360;
    settle(v, s);
  } catch (err) {
    toast(err.message);      // the current spot stays as it was, and playable
    stuck = true;
  } finally {
    stepping = false;
    exitBearing = null;
  }
}

function sendHint(now) {
  const h = hero.heading;
  if (now - hint.at < HINT_MS || Math.abs(rel(hint.heading, h)) <= HINT_DEG) return;
  hint = { at: now, heading: h };
  hintCount++;
  hints.push({ t: now, heading: h });
  if (hints.length > 64) hints.shift();
  call("heading", { heading: h }).catch(() => {});   // best effort: only steers the pre-dreaming
}

// --- controls -----------------------------------------------------------------

const stick = { id: null, cx: 0, cy: 0, input: STILL };

// Keys or the stick; nothing while typing; and, while the next spot is on its way, nothing that walks on outward
// near the edge, so the character eases to a stop there instead of pushing against it.
function control() {
  if (typing()) return STILL;
  const c = stick.id !== null ? stick.input : keys.input;
  if (!stepping || (!c.fwd && !c.side)) return c;
  const p = hero.pos, a = cam.yaw + Math.atan2(-c.side, c.fwd);
  const outward = -Math.sin(a) * p.x - Math.cos(a) * p.y > 0;
  return outward && p.length() >= EASE_FROM * LIMIT ? STILL : c;
}

// Look: drag anywhere on the world (mouse, pen or touch), or click to capture the mouse (Esc lets go).
const look = { id: null, x: 0, y: 0 };
const turn = (dx, dy) => cam.turn(-dx * LOOK_X, dy * LOOK_Y);
canvas.addEventListener("pointerdown", e => {
  if (typing()) input.blur();
  if (!started || look.id !== null) return;
  Object.assign(look, { id: e.pointerId, x: e.clientX, y: e.clientY });
  if (e.pointerType === "mouse" && document.pointerLockElement !== canvas) {
    try { canvas.requestPointerLock()?.catch?.(() => {}); } catch { /* no pointer lock: dragging still looks */ }
  }
});
addEventListener("pointermove", e => {
  if (document.pointerLockElement === canvas) { turn(e.movementX, e.movementY); return; }
  if (e.pointerId !== look.id) return;
  turn(e.clientX - look.x, e.clientY - look.y);
  look.x = e.clientX; look.y = e.clientY;
});
const endLook = e => { if (e.pointerId === look.id) look.id = null; };
addEventListener("pointerup", endLook);
addEventListener("pointercancel", endLook);

// The phone stick: push the knob the way to walk, relative to the camera; all the way out runs.
const stickEl = $("stick"), knob = $("knob");
function moveStick(e) {
  let dx = e.clientX - stick.cx, dy = e.clientY - stick.cy;
  const m = Math.hypot(dx, dy);
  if (m > STICK_R) { dx *= STICK_R / m; dy *= STICK_R / m; }
  knob.style.transform = `translate(${dx}px, ${dy}px)`;
  stick.input = m < 0.2 * STICK_R ? STILL : { fwd: -dy / STICK_R, side: dx / STICK_R, run: m > 1.4 * STICK_R };
}
stickEl.addEventListener("pointerdown", e => {
  e.preventDefault();
  if (typing()) input.blur();
  const r = stickEl.getBoundingClientRect();
  Object.assign(stick, { id: e.pointerId, cx: r.left + r.width / 2, cy: r.top + r.height / 2 });
  stickEl.setPointerCapture(e.pointerId);
  moveStick(e);
});
stickEl.addEventListener("pointermove", e => { if (e.pointerId === stick.id) moveStick(e); });
const endStick = e => {
  if (e.pointerId !== stick.id) return;
  Object.assign(stick, { id: null, input: STILL });
  knob.style.transform = "";
};
stickEl.addEventListener("pointerup", endStick);
stickEl.addEventListener("pointercancel", endStick);

addEventListener("keydown", e => {
  if (!typing() && started && e.key.startsWith("Arrow")) e.preventDefault();
});

// The first place starts the dream; later ones drift it toward them.
$("theme").addEventListener("submit", async e => {
  e.preventDefault();
  const place = input.value.trim();
  if (!place) return;
  if (!started) { if (!starting) begin(place); return; }
  try {
    await call("theme", { place });
    input.value = "";
    input.blur();
    toast(`The dream drifts toward ${place}`);
  } catch (err) {
    toast(err.message);
  }
});

// Mood toggles: before the start they go with it; after, straight away.
for (const chip of chips) {
  chip.addEventListener("click", async () => {
    const on = chip.getAttribute("aria-pressed") !== "true";
    chip.setAttribute("aria-pressed", String(on));
    if (!started) return;
    try {
      await call("moods", { moods: activeMoods() });
    } catch (err) {
      chip.setAttribute("aria-pressed", String(!on));
      toast(err.message);
    }
  });
}

// --- the frame ------------------------------------------------------------------

const fogged = () => stepping || !!world.edge;

function play(dt, now) {
  hero.update(dt, control(), cam.yaw, world.k);
  world.update(hero.pos, hero.dir.lengthSq() ? hero.dir : null);
  world.tick(dt);
  if (!world.edge) stuck = false;
  else if (!stepping && !stuck) step(world.edge.bearing);
  sendHint(now);
  const fog = fogged(), toward = exitBearing ?? world.edge?.bearing;
  document.body.classList.toggle("fogged", fog);
  if (fog && toward != null) {
    const d = rel(camBearing(), toward) * Math.PI / 180;   // the fog thickens on the side it lies
    document.body.style.setProperty("--fog-x", `${50 + 40 * Math.sin(d)}%`);
  }
}

function pose() {
  const k = world.k, p = hero.pos;
  hero.place(k);
  signposts.place(k);
  cam.apply(camera, new THREE.Vector3(k * p.x, 0, k * p.y));
  scene.updateMatrixWorld();
  camera.updateMatrixWorld();
}

function draw() {
  if (!started) { renderer.render(scene, camera); return; }
  pose();
  world.render(renderer, scene, camera);
  compass.update(camBearing());
}

let last = performance.now();
renderer.setAnimationLoop(t => {
  const dt = Math.min(0.05, (t - last) / 1000);
  last = t;
  if (started) play(dt, t);
  draw();
});

// --- state for the tests (read-only) and their hooks ----------------------------

window.roam = Object.freeze(Object.defineProperties({}, {
  ready: { get: () => hero !== null, enumerable: true },
  started: { get: () => started, enumerable: true },
  spotId: { get: () => view?.spot_id ?? null, enumerable: true },
  place: { get: () => view?.place ?? null, enumerable: true },
  neighbours: { get: () => (view?.neighbours ?? []).map(n => ({ ...n })), enumerable: true },
  pos: { get: () => hero && { x: hero.pos.x, z: hero.pos.y }, enumerable: true },
  world: { get: () => hero && { x: world.abs.x + world.abs.s * hero.pos.x, z: world.abs.z + world.abs.s * hero.pos.y },
           enumerable: true },
  heading: { get: () => hero?.heading ?? null, enumerable: true },
  camYaw: { get: () => cam.yaw, enumerable: true },
  camBearing: { get: () => camBearing(), enumerable: true },
  stepping: { get: () => stepping, enumerable: true },
  fog: { get: () => started && fogged(), enumerable: true },
  edge: { get: () => world.edge && { ...world.edge }, enumerable: true },
  built: { get: () => world.spots.length, enumerable: true },
  fading: { get: () => !!world.fade, enumerable: true },
  signposts: { get: () => [...signposts.names], enumerable: true },
  hints: { get: () => hints.map(h => ({ ...h })), enumerable: true },
}));

const gl = renderer.getContext();
window.roamDev = {
  world,
  get hero() { return hero; },
  // Look along a bearing (degrees), at a pitch.
  look(bearing, pitch) { if (bearing != null) cam.yaw = OrbitCam.yawFor(bearing); if (pitch != null) cam.setPitch(pitch); },
  // Render now and read the pixel at (fx, fy), fractions of the screen from the top left.
  pixel(fx, fy) {
    draw();
    const px = new Uint8Array(4);
    gl.readPixels(Math.round(fx * (gl.drawingBufferWidth - 1)), Math.round((1 - fy) * (gl.drawingBufferHeight - 1)),
                  1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
    return [px[0], px[1], px[2]];
  },
  // Where the signpost's boards are on screen, as fractions from the top left.
  signpostOnScreen() {
    pose();
    const v = signposts.group.localToWorld(new THREE.Vector3(0, 0.5, 0)).project(camera);
    return [(v.x + 1) / 2, (1 - v.y) / 2];
  },
};

session().catch(() => {});    // warm up; retried on first use if it failed
