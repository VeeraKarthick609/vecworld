"use strict";

const $ = (id) => document.getElementById(id);
const stage = $("stage"), hotspotsEl = $("hotspots"), input = $("jump-text");
const layers = [$("scene-a"), $("scene-b")];
let front = 0;            // index of the visible scene layer
let sid = null;
let busy = false;

async function api(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw Object.assign(new Error(data.message || `Request failed (${r.status})`), { code: data.error });
  return data;
}

async function newSession() {
  sid = (await api("/api/session")).session_id;
}

function setBusy(on) {
  busy = on;
  input.disabled = on;
  $("spinner").hidden = !on;
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

function loadImage(url) {
  return new Promise((ok, fail) => {
    const img = new Image();
    img.onload = () => ok(url);
    img.onerror = fail;
    img.src = url;
  });
}

// Crossfade from the current layer to a fresh one holding the new scene.
async function show(scene) {
  await loadImage(scene.image_url);
  const old = layers[front], next = layers[1 - front];
  next.style.transition = "none";
  next.style.transform = "";
  next.src = scene.image_url;
  void next.offsetWidth;
  next.style.transition = "";
  next.classList.add("shown");
  old.classList.remove("shown");
  setTimeout(() => { old.style.transform = ""; }, 350);
  front = 1 - front;

  $("backdrop").style.backgroundImage = `url(${scene.image_url})`;
  $("intro").hidden = true;
  renderHotspots(scene.hotspots);
  renderMap(scene.mindmap);
}

function renderHotspots(hotspots) {
  hotspotsEl.replaceChildren(...hotspots.map((h) => {
    const el = document.createElement("div");
    el.className = "hotspot";
    el.tabIndex = 0;
    el.setAttribute("role", "button");
    el.setAttribute("aria-label", `go toward ${h.concept}`);
    Object.assign(el.style, {
      left: `${h.x * 100}%`, top: `${h.y * 100}%`,
      width: `${h.w * 100}%`, height: `${h.h * 100}%`,
    });
    const label = document.createElement("span");
    label.textContent = h.concept;
    el.append(label);
    const go = () => step(h);
    el.addEventListener("click", go);
    el.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
    return el;
  }));
}

const map = $("mindmap"), ctx = map.getContext("2d");

function renderMap(m) {
  map.hidden = false;
  const W = map.width, H = map.height;
  const scale = m.reach_radius > 0 ? (0.25 * W) / m.reach_radius : W / 2;
  const toPx = ([x, y]) => [W / 2 + (x - m.me[0]) * scale, H / 2 - (y - m.me[1]) * scale];

  ctx.clearRect(0, 0, W, H);

  ctx.fillStyle = "rgba(143, 179, 255, 0.7)";
  for (const n of m.neighbours) {
    const [px, py] = toPx([n.x, n.y]);
    ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2); ctx.fill();
  }

  if (m.trail.length > 1) {
    ctx.strokeStyle = "rgba(255, 207, 92, 0.6)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    m.trail.forEach((p, i) => { const [px, py] = toPx(p); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
    ctx.stroke();
  }

  ctx.setLineDash([6, 6]);
  ctx.strokeStyle = "rgba(255, 207, 92, 0.7)";
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(W / 2, H / 2, m.reach_radius * scale, 0, Math.PI * 2); ctx.stroke();
  ctx.setLineDash([]);

  ctx.shadowColor = "#ffcf5c"; ctx.shadowBlur = 12;
  ctx.fillStyle = "#ffcf5c";
  ctx.beginPath(); ctx.arc(W / 2, H / 2, 7, 0, Math.PI * 2); ctx.fill();
  ctx.shadowBlur = 0;
}

async function withSession(fn) {
  if (busy) return;
  setBusy(true);
  try {
    if (!sid) await newSession();
    try {
      await fn();
    } catch (e) {
      if (e.code !== "unknown_session") throw e;
      await newSession();
      toast("Your session expired, starting fresh. Type a word to jump in.");
    }
  } catch (e) {
    if (e.code !== "busy") toast(e.message);
    layers[front].style.transform = "";
  } finally {
    setBusy(false);
  }
}

$("jump").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  withSession(async () => {
    layers[front].classList.remove("shown");
    await show(await api(`/api/session/${sid}/jump`, { text }));
    input.value = "";
  });
});

// Holding animation while the next scene generates: zoom toward what was clicked.
function step(h) {
  withSession(async () => {
    const cur = layers[front];
    cur.style.transformOrigin = `${(h.x + h.w / 2) * 100}% ${(h.y + h.h / 2) * 100}%`;
    cur.style.transform = "scale(1.6)";
    hotspotsEl.replaceChildren();
    await show(await api(`/api/session/${sid}/step`, { concept: h.concept }));
  });
}
