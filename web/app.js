"use strict";

// Dream Walk viewer: a 360° cylindrical panorama you look around in (client-side) and walk through
// (the server paints the next spot, centred on the direction you moved).

const FOV = 90;              // horizontal degrees shown, narrowed on tall screens so the image keeps its aspect
const TURN_SPEED = 150;      // degrees per second while A/D or ←/→ is held
const MOVE_MS = 900;         // dolly + crossfade between spots
const START_MS = 600;        // fade in of the first spot
const DOLLY = 1.2;           // zoom reached by the old spot as you walk into it
const SETTLE_MS = 500;       // still this long before the spot ahead is prefetched
const PREFETCH_TOLERANCE = 45;
const DRAG_CLICK = 6;        // px of drag after which a release is not a click

const $ = (id) => document.getElementById(id);
const canvas = $("view"), ctx = canvas.getContext("2d");
const labelsEl = $("labels"), input = $("theme-text");
const chips = [...document.querySelectorAll(".chip")];

let sid = null;
let view = null;             // the ViewOut on screen
let labels = [];             // its labels
let labelEls = [];           // [element, label] pairs on screen
let image = null;            // its loaded panorama
let yaw = 0;                 // degrees, 0 = the centre of the panorama, positive = right
let busy = false;            // a start or move is in flight (request, image load or transition)
let anim = null;             // {from, fromYaw, to, dir, t0, ms, done}
let lastPrefetch = null;     // yaw of the last prefetch for this spot and theme/moods
let settleTimer = 0;

const deepFreeze = (o) => { Object.values(o).forEach((v) => v && typeof v === "object" && deepFreeze(v)); return Object.freeze(o); };
window.vecworld = Object.freeze({
  get yaw() { return yaw; },
  get view() { return view; },
  get labels() { return labels; },
});

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

// --- HUD ----------------------------------------------------------------------

function setBusy(on) {
  busy = on;
  $("spinner").hidden = !on || anim !== null;
  document.body.classList.toggle("busy", on);
}

let toastTimer;
function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 3500);
}

const activeMoods = () => chips.filter((c) => c.getAttribute("aria-pressed") === "true").map((c) => c.dataset.mood);

// --- drawing ------------------------------------------------------------------

function resize() {
  const dpr = devicePixelRatio || 1;
  canvas.width = Math.round(innerWidth * dpr);
  canvas.height = Math.round(innerHeight * dpr);
}
addEventListener("resize", resize);
resize();

const wrapYaw = (a) => ((a + 180) % 360 + 360) % 360 - 180;
const angleBetween = (a, b) => Math.abs(wrapYaw(a - b));

// Degrees across the screen: 90°, or less on a screen taller than the panorama allows at 90°.
function fov() {
  const aspect = image ? image.naturalHeight / image.naturalWidth : 0.25;
  return Math.min(FOV, 360 * aspect * innerWidth / innerHeight);
}

// Draw the window of a cylindrical panorama centred on yawDeg, wrapping around the seam, zoomed by zoom (≥ 1).
function drawPano(img, yawDeg, zoom = 1, alpha = 1) {
  const W = img.naturalWidth, H = img.naturalHeight, cw = canvas.width, ch = canvas.height;
  const sw = W * fov() / 360 / zoom, sh = Math.min(H, sw * ch / cw);
  const cx = ((W / 2 + yawDeg / 360 * W) % W + W) % W, sy = (H - sh) / 2;
  ctx.globalAlpha = alpha;
  const sx = cx - sw / 2;
  for (const off of [-W, 0, W]) {              // up to three copies cover the wrap
    const x = sx + off;
    if (x + sw <= 0 || x >= W) continue;
    const a = Math.max(0, x), b = Math.min(W, x + sw);
    ctx.drawImage(img, a, sy, b - a, sh, (a - x) / sw * cw, 0, (b - a) / sw * cw, ch);
  }
  ctx.globalAlpha = 1;
}

function frame(t) {
  ctx.fillStyle = "#0b0d12";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (anim) {
    const k = Math.min(1, (t - anim.t0) / anim.ms);
    const fade = anim.from ? Math.max(0, (k - 0.35) / 0.65) : k;
    if (anim.dir > 0) {
      // Walking on: dolly into the old spot while the new one, already a little closer, settles in.
      if (anim.from) drawPano(anim.from, anim.fromYaw, 1 + (DOLLY - 1) * k);
      drawPano(anim.to, 0, 1 + (DOLLY - 1) * 0.3 * (1 - k), fade);
    } else {
      // Turning back: the old spot falls away as the new one pulls back from close up.
      if (anim.from) drawPano(anim.from, anim.fromYaw, 1, 1 - fade);
      drawPano(anim.to, 0, 1 + (DOLLY - 1) * (1 - k), fade);
    }
    if (k >= 1) { const done = anim.done; anim = null; done(); }
  } else if (image) {
    drawPano(image, yaw);
  }
  placeLabels();
  requestAnimationFrame(frame);
}

// --- labels -------------------------------------------------------------------

function renderLabels() {
  labelEls = labels.map((l) => {
    const el = document.createElement("button");
    el.type = "button";
    el.className = "label";
    el.dataset.id = l.id;
    el.setAttribute("aria-label", `approach ${l.text}`);
    const text = document.createElement("span");
    text.textContent = l.text;
    el.append(text);
    el.addEventListener("click", () => { if (dragged <= DRAG_CLICK) move("approach", { label: l.id }); });
    return [el, l];
  });
  labelsEl.replaceChildren(...labelEls.map(([el]) => el));
}

// Place each label from its panorama box (u = centre, y = top, w, h as fractions) relative to the yaw.
function placeLabels() {
  labelsEl.hidden = anim !== null;
  if (anim) return;
  const f = fov() / 360, cw = innerWidth, ch = innerHeight;
  const ratio = image ? image.naturalWidth / image.naturalHeight : 4;
  const vis = Math.min(1, f * ratio * ch / cw);    // fraction of the panorama's height on screen
  const top = (1 - vis) / 2;
  const centre = (0.5 + yaw / 360) % 1;
  for (const [el, l] of labelEls) {
    // Nearest copy of the label, so one just across the seam shows on the side it really is on.
    const d = ((l.u - centre + 0.5) % 1 + 1) % 1 - 0.5;
    if (Math.abs(d) - l.w / 2 >= f / 2) { el.style.display = "none"; continue; }
    el.style.display = "";
    el.style.left = `${(0.5 + (d - l.w / 2) / f) * cw}px`;
    el.style.width = `${l.w / f * cw}px`;
    el.style.top = `${(l.y - top) / vis * ch}px`;
    el.style.height = `${l.h / vis * ch}px`;
  }
}

// --- moving -------------------------------------------------------------------

function transition(img, dir, ms) {
  return new Promise((done) => {
    // Labels for the new spot go up in the frame the transition ends, never over the old one.
    anim = { from: image, fromYaw: yaw, to: img, dir, t0: performance.now(), ms, done: () => { renderLabels(); done(); } };
    $("spinner").hidden = true;
  });
}

async function arrive(v, dir) {
  const img = await loadImage(v.image_url);
  const t = transition(img, dir, view ? MOVE_MS : START_MS);
  view = deepFreeze(v);
  labels = view.labels;
  image = img;
  yaw = 0;                    // the server centred the new spot on the direction you moved
  await t;
  lastPrefetch = null;
  settled();
}

async function move(what, body) {
  if (busy || !view) return;
  setBusy(true);
  try {
    const v = await call(what, body);
    await arrive(v, what === "back" ? -1 : 1);
  } catch (e) {
    toast(e.message);         // nothing changed: the previous view and labels are still on screen
  } finally {
    setBusy(false);
  }
}

const walk = () => move("walk", { yaw });
const back = () => move("back", { yaw });

// --- prefetch -----------------------------------------------------------------

function settled() {
  clearTimeout(settleTimer);
  settleTimer = setTimeout(() => {
    if (!view || busy) return;
    if (lastPrefetch !== null && angleBetween(yaw, lastPrefetch) <= PREFETCH_TOLERANCE) return;
    lastPrefetch = yaw;
    call("prefetch", { yaw }).catch(() => {});   // best effort; the server drops stale ones
  }, SETTLE_MS);
}

function turn(by) {
  yaw = wrapYaw(yaw + by);
  settled();
}

// --- input --------------------------------------------------------------------

const held = new Set();
const typing = () => document.activeElement === input;

addEventListener("keydown", (e) => {
  if (typing() || e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k.startsWith("arrow")) e.preventDefault();
  if (busy || !view) return;
  if (["a", "d", "arrowleft", "arrowright"].includes(k)) held.add(k);
  else if ((k === "w" || k === "arrowup") && !e.repeat) walk();
  else if ((k === "s" || k === "arrowdown") && !e.repeat) back();
});
addEventListener("keyup", (e) => held.delete(e.key.toLowerCase()));
addEventListener("blur", () => held.clear());
input.addEventListener("focus", () => held.clear());

let last = performance.now();
function turning(t) {
  const dt = Math.min(0.1, (t - last) / 1000);
  last = t;
  if (busy || !view) held.clear();
  const dir = (held.has("d") || held.has("arrowright") ? 1 : 0) - (held.has("a") || held.has("arrowleft") ? 1 : 0);
  if (dir) turn(dir * TURN_SPEED * dt);
  requestAnimationFrame(turning);
}
requestAnimationFrame(turning);
requestAnimationFrame(frame);

// Drag to look (mouse, pen or touch). Window listeners, not pointer capture, so a tap on a label still clicks it.
let drag = null, dragged = 0;
$("stage").addEventListener("pointerdown", (e) => {
  if (e.button !== 0) return;
  if (typing()) input.blur();
  drag = e.clientX;
  dragged = 0;
});
addEventListener("pointermove", (e) => {
  if (drag === null) return;
  const dx = e.clientX - drag;
  drag = e.clientX;
  dragged += Math.abs(dx);
  if (!busy && view) turn(-dx * fov() / innerWidth);
});
addEventListener("pointerup", () => (drag = null));
addEventListener("pointercancel", () => (drag = null));

$("go-walk").addEventListener("click", walk);
$("go-back").addEventListener("click", back);

// The first theme starts the dream; later ones drift it.
$("theme").addEventListener("submit", async (e) => {
  e.preventDefault();
  const theme = input.value.trim();
  if (!theme) return;
  if (view) {
    try {
      await call("theme", { theme });
      input.value = "";
      input.blur();
      lastPrefetch = null;    // the server dropped the prefetch built for the old theme
      settled();
      toast(`the dream is drifting toward ${theme}`);
    } catch (err) {
      toast(err.message);
    }
    return;
  }
  if (busy) return;
  setBusy(true);
  input.disabled = true;
  try {
    const v = await call("start", { theme, moods: activeMoods() });
    for (const c of chips) c.setAttribute("aria-pressed", String(v.moods.includes(c.dataset.mood)));
    $("intro").hidden = true;
    const landing = arrive(v, 1);
    $("keys").hidden = false;
    $("pad").hidden = false;
    document.body.classList.add("dreaming");
    input.value = "";
    input.placeholder = "drift the dream toward…";
    await landing;
  } catch (err) {
    toast(err.message);
  } finally {
    input.disabled = false;
    setBusy(false);
    if (view) input.blur();
  }
});

// Mood toggles: before the start they are sent with it; after, straight away.
for (const chip of chips) {
  chip.addEventListener("click", async () => {
    const on = chip.getAttribute("aria-pressed") !== "true";
    chip.setAttribute("aria-pressed", String(on));
    if (!view) return;
    try {
      await call("moods", { moods: activeMoods() });
      lastPrefetch = null;    // the server dropped the prefetch built for the old moods
      settled();
    } catch (err) {
      chip.setAttribute("aria-pressed", String(!on));
      toast(err.message);
    }
  });
}

session().catch(() => {});    // warm up; retried on first use if it failed
