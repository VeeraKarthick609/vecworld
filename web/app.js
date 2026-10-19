// Dream Roam: type a place, then roam a dreamed world of 360° painted spots in any direction.
//
// The server holds the journey; the page holds the world around the player (web/roam/world.js). The first spot
// comes from /start. Walking out toward the rim of a spot with nothing ahead asks /next for the spot that way, late:
// once the rim is within the coast distance plus LEAD seconds of walking, so a player who turns back never commits.
// It says how far from the spot's centre the player will leave it (walked: where the next spot is dreamed from;
// each spot's rim comes from the server, see world.js).
// The new spot is placed beside the server's spot (distance, heading and scale are parent-relative) and the player
// switches into it on walking in (World.update). If it isn't there in time, the character coasts to a stop at the
// rim in a dream fog and is moved into it as it lands. The server pre-dreams around the player's heading, which the
// page hints at most every HINT_MS when it has turned by more than HINT_DEG.
//
// `here` is the world entry of the server's current spot. Walking back into the previous spot (World.update can
// switch there until the next step) tells the server with /enter, so every /next is asked from the spot the player
// is in.

import * as THREE from "three";
import { World } from "./roam/world.js";
import { ACCEL, Hero, Keys, OrbitCam } from "./roam/hero.js";
import { Signposts } from "./roam/signposts.js";
import { Compass, rel } from "./roam/compass.js";

const HINT_MS = 300, HINT_DEG = 20;   // heading hints: at most this often, and only past this much turn
// Seconds of walking before the coast to the rim, when /next goes out: enough for the answer, its two images and
// the mesh to land before the rim (measured ~0.4 s on the Tailscale LAN), so a ready spot is walked into without
// braking.
const LEAD = 0.8;
const RETRY_FROM = 0.6;               // after a failed /next, back inside this fraction of the rim the way opens again
const RETRY_TURN = 45;                // ... or heading for an exit this many degrees from the failed one
const REBUILD_TRIES = 2;              // a spot /next returned that fails to build this often is given up
const CLOSED = "The way is closed — try another direction.";
const FADED = "The way ahead faded — step back and try again.";
const FOG_RIM = 0.95;                 // a pending spot shows the fog once the player is past this fraction of the rim
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
  return world.buildSpot(pano, depth, { spot_id: v.spot_id, limits: v.limits });
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
let here = null;          // ... and its world entry
let stepping = false;     // a /next is in flight, or its spot is being built
let pending = null;       // a SpotOut /next returned whose spot didn't build: built again before asking anew
let pendingFails = 0;     // ... and how often building it failed (dropped after REBUILD_TRIES)
let stuck = null;         // the exit bearing of a failed step: the way is closed (see RETRY_*)
let braking = false;      // the character is coasting to a stop at the rim
let entering = false;     // an /enter is in flight
let enterWanted = null;   // a spot the player walked into that the server should step into
let switches = 0;         // world switches, for the tests
let exitBearing = null;   // where the pending /next leads (for the fog)
let moodsPending = false; // a chip changed while /start was in flight
const steps = [];         // every step taken, for the tests
let cameFrom = null;      // the bearing back the way the player came
let hint = { at: -Infinity, heading: 0 }, hintCount = 0;
const hints = [];         // the last hints sent, {t, heading}, for the tests

// The server's spot v is now `entry` in the world: signposts, compass and arrival follow it.
function settle(v, entry) {
  view = v;
  here = entry;
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
  if (started && moodsPending) sendMoods();
}

// The world's absolute position (the first spot's frame) of a point of the current frame, for the step log.
const absolute = (x, z) => ({ x: world.abs.x + world.abs.s * x, z: world.abs.z + world.abs.s * z });

// Fog and pacing, for the tests and the pacing measurement: every fog spell {t0, t1} (ms), and how each /next
// was served (the server's own timing is in its log).
const fogs = [];
const nexts = [];

async function step(exit, walked) {
  stepping = true;
  exitBearing = exit;
  const rebuild = pending !== null;
  steps.push({ exit, walked, from: here.meta.spot_id, inServerSpot: world.current === here, rebuild,
               origin: absolute(here.origin.x, here.origin.z), player: absolute(hero.pos.x, hero.pos.y) });
  const sent = hintCount;
  let giveUp = false;
  try {
    let v;
    try {
      const t0 = performance.now();
      v = pending ?? await call("next", { exit_heading: exit, walked });
      if (!rebuild) nexts.push({ t0, ms: performance.now() - t0, distance: v.distance, walked, served: v.served });
    } catch (err) {
      toast(`${err.message} ${CLOSED}`);   // the current spot stays as it was, and playable
      stuck = exit;
      return;
    }
    pending = v;              // the server has moved on: until its spot is built, building it is the next step
    if (hintCount === sent) hint = { at: performance.now(), heading: exit };   // the server pre-dreams this way
    let s;
    try {
      s = await build(v);
    } catch (err) {
      toast(`${err.message} ${FADED}`);
      stuck = exit;
      giveUp = ++pendingFails >= REBUILD_TRIES;
      return;
    }
    pending = null; pendingFails = 0;
    const from = here;
    world.placeChild(from, v.heading, v.distance, v.scale, s);
    // Waiting at the rim: in now. Otherwise the player switches on walking in (World.update).
    if (braking && world.current === from) { world.enter(s, hero.pos); switches++; }
    cameFrom = (v.heading + 180) % 360;
    settle(v, s);
  } finally {
    stepping = false;
    exitBearing = null;
    if (giveUp) {           // the server stays on a spot that won't build: take it back to the player's spot
      pending = null; pendingFails = 0;
      enterServer(here);
    }
  }
}

// Tell the server the player is in `entry` (walked back into the previous spot, or into a placed child after
// stepping back): it steps into that spot, no painting. A spot the server no longer holds is dropped from the world
// (so it can't be walked into again) and the world re-roots on the server's spot.
async function enterServer(entry) {
  entering = true;
  const from = here;
  try {
    const v = await call("enter", { spot_id: entry.meta.spot_id });
    if (from.parent === entry) cameFrom = from.heading;                       // walked back from the child
    else if (entry.parent === from) cameFrom = (entry.heading + 180) % 360;   // on into the child
    pending = null; pendingFails = 0; stuck = null;
    settle(v, entry);
    if (world.current !== entry) enterWanted = world.current;   // crossed on meanwhile: that spot next
  } catch (err) {
    toast(err.message);
    if (world.spots.includes(here) && world.current !== here) { world.enter(here, hero.pos); switches++; }
    if (err.code === "unknown_spot" && entry !== here && world.spots.includes(entry)) world.remove(entry);
  } finally {
    entering = false;
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

// Keys or the stick; nothing while typing.
const control = () => typing() ? STILL : stick.id !== null ? stick.input : keys.input;

// The unit direction (x, z) input c walks in, or null.
function travel(c) {
  if (!c.fwd && !c.side) return null;
  const a = cam.yaw + Math.atan2(-c.side, c.fwd);
  return new THREE.Vector2(-Math.sin(a), -Math.cos(a));
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

// Mood toggles: before the start they go with it; during it they follow it; after, straight away.
async function sendMoods(chip) {
  moodsPending = false;
  try {
    await call("moods", { moods: activeMoods() });
  } catch (err) {
    if (chip) chip.setAttribute("aria-pressed", String(chip.getAttribute("aria-pressed") !== "true"));
    toast(err.message);
  }
}
for (const chip of chips) {
  chip.addEventListener("click", () => {
    chip.setAttribute("aria-pressed", String(chip.getAttribute("aria-pressed") !== "true"));
    if (started) sendMoods(chip);
    else if (starting) moodsPending = true;
  });
}

// --- the frame ------------------------------------------------------------------

const fogged = () => stepping && (braking || hero.pos.length() >= FOG_RIM * world.rimAt(hero.pos.x, hero.pos.y));

function play(dt, now) {
  const p = hero.pos, c = control(), want = travel(c);
  const exit = world.exitAhead(p, want);
  const busy = stepping || entering, coast = hero.speed / ACCEL / world.k;
  if (stuck !== null && (p.length() < RETRY_FROM * world.rimAt(p.x, p.y)
                         || (exit && Math.abs(rel(stuck, exit.bearing)) > RETRY_TURN)))
    stuck = null;
  // Why the way out at this exit is closed, if it is: a failed step, or a spot the server moved to lying elsewhere.
  const closed = !exit ? null
    : pending && (stuck !== null || Math.abs(rel(pending.heading, exit.bearing)) > RETRY_TURN) ? FADED
    : stuck !== null ? CLOSED : null;
  if (exit && !closed && !busy && exit.gap <= coast + hero.speed * LEAD / world.k + 0.02) {
    if (world.current !== here) enterWanted = world.current;   // the server first steps into this spot
    else step(exit.bearing, exit.r);
  }
  // Waiting at a rim (a step or /enter on its way, or the way closed): stop walking on once the coast would just
  // reach it.
  braking = !!exit && (busy || !!closed || world.current !== here) && exit.gap <= coast + 0.02;
  if (braking && closed && $("toast").hidden) toast(closed);
  const before = p.clone();
  hero.move(dt, braking ? STILL : c, cam.yaw, world.k);
  world.constrain(p);
  const moved = p.distanceTo(before);       // frame units, before a switch can re-root p
  hero.collide(moved, dt, world.k);
  const dir = hero.dir.lengthSq() ? hero.dir.clone() : null;
  const to = world.update(p, dir);
  if (to) { switches++; if (to !== here) enterWanted = to; }
  if (enterWanted && !stepping && !entering) {
    if (enterWanted === world.current && enterWanted !== here) enterServer(enterWanted);
    enterWanted = null;
  }
  hero.animate(dt);
  world.tick(dt);
  sendHint(now);
  const fog = fogged();
  if (fog && !wasFog) fogs.push({ t0: now, t1: null });
  if (!fog && wasFog) fogs[fogs.length - 1].t1 = now;
  if (fogs.length > 200) fogs.shift();
  wasFog = fog;
  document.body.classList.toggle("fogged", fog);
  if (fog && exitBearing != null) {
    const d = rel(camBearing(), exitBearing) * Math.PI / 180;   // the fog thickens on the side the way lies
    document.body.style.setProperty("--fog-x", `${50 + 40 * Math.sin(d)}%`);
  }
  samples.push({ t: now, ground: dt > 0 ? moved * world.k / dt : 0, walk: hero.walk.getEffectiveWeight() });
  if (samples.length > 600) samples.shift();
}
const samples = [];       // per frame: ground speed and walk-clip weight, for the tests
let wasFog = false;

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
  rim: { get: () => hero && started ? world.rimAt(hero.pos.x, hero.pos.y) : null, enumerable: true },
  fogs: { get: () => fogs.map(f => ({ ...f })), enumerable: true },
  nexts: { get: () => nexts.map(n => ({ ...n })), enumerable: true },
  fog: { get: () => started && fogged(), enumerable: true },
  braking: { get: () => braking, enumerable: true },
  stuck: { get: () => stuck !== null, enumerable: true },
  inServerSpot: { get: () => world.current === here, enumerable: true },
  parentSpotId: { get: () => here?.parent?.meta.spot_id ?? null, enumerable: true },
  samples: { get: () => samples.map(x => ({ ...x })), enumerable: true },
  steps: { get: () => steps.map(x => ({ ...x, origin: { ...x.origin }, player: { ...x.player } })), enumerable: true },
  switches: { get: () => switches, enumerable: true },
  entering: { get: () => entering, enumerable: true },
  edge: { get: () => world.edge && { ...world.edge }, enumerable: true },
  built: { get: () => world.spots.length, enumerable: true },
  spotIds: { get: () => world.spots.map(e => e.meta.spot_id), enumerable: true },
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
