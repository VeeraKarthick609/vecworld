// Builds one spot's depth mesh off the main thread (a classic worker; also importable for its maths).
//
// Geometry (the plan's Global Constraints): column u -> heading θ = (u − 0.5)·2π, u = 0.5 facing −z (the spot's
// forward); row v -> height (0.5 − v)·π/2 on the unit cylinder; radius = 1/(d·(1/NEAR − 1/FAR) + 1/FAR) for the
// 0..1 depth d (1 = near). Triangles whose radius ratio exceeds EDGE are cut.
//
// Over the spike: the depth is median-filtered before cutting (fewer ragged cut lines), vertices next to a cut get
// a soft alpha (so a trunk's edge fades instead of tearing), and every point at or below the fitted ground plane is
// slid down its own ray onto it (no ridge where the ground meets the trees; the ground plane takes over below).

const NEAR = 4, FAR = 50, EDGE = 1.35, VSPAN = Math.PI / 2;
const GROUND_ROWS = 12;           // bottom grid rows the ground plane is fitted from
const SNAP_FROM = 0.8;            // points within this fraction of the ground ray's length start sliding onto it

function radius(d) { return 1 / (d * (1 / NEAR - 1 / FAR) + 1 / FAR); }

function median9(a) { const s = Array.from(a).sort((x, y) => x - y); return s[4]; }

// depth: gx·gy values 0..255 (row-major, top row first). Returns transferable typed arrays + the ground level.
function buildSpotMesh(depth, gx, gy) {
  // 1. depth -> radius, through a 3×3 median (wrapping round the cylinder, clamped top and bottom)
  const d = new Float32Array(gx * gy), win = new Float32Array(9);
  for (let y = 0; y < gy; y++) for (let x = 0; x < gx; x++) {
    let n = 0;
    for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) {
      const yy = Math.min(gy - 1, Math.max(0, y + j)), xx = (x + i + gx) % gx;
      win[n++] = depth[yy * gx + xx];
    }
    d[y * gx + x] = median9(win) / 255;
  }
  const nx = gx + 1, nv = nx * gy;                        // one extra column closes the cylinder (u = 1)
  const rad = new Float32Array(nv);
  for (let y = 0; y < gy; y++) for (let x = 0; x <= gx; x++) rad[y * nx + x] = radius(d[y * gx + (x % gx)]);
  const rowAngle = y => (0.5 - y / (gy - 1)) * VSPAN;    // the height factor of grid row y

  // 2. ground level = the median height of the bottom rows (the spike's fit)
  const hs = [];
  for (let y = gy - GROUND_ROWS; y < gy; y++) for (let x = 0; x < gx; x += 2) hs.push(rowAngle(y) * rad[y * nx + x]);
  hs.sort((a, b) => a - b);
  const ground = hs[hs.length >> 1];

  // 3. slide ground-level and below-ground points down their ray onto the plane: row y's ray meets it at -ground/(-a)
  for (let y = 0; y < gy; y++) {
    const a = rowAngle(y);
    if (a >= -1e-6) continue;
    const rg = Math.min(FAR, ground / a);
    for (let x = 0; x <= gx; x++) {
      const k = y * nx + x, f = rad[k] / rg;
      if (f <= SNAP_FROM) continue;
      const t = Math.min(1, (f - SNAP_FROM) / (1 - SNAP_FROM)), w = t * t * (3 - 2 * t);
      rad[k] = f >= 1 ? rg : rad[k] + (rg - rad[k]) * w;
    }
  }

  // 4. positions + uvs
  const position = new Float32Array(nv * 3), uv = new Float32Array(nv * 2);
  for (let y = 0; y < gy; y++) for (let x = 0; x <= gx; x++) {
    const k = y * nx + x, u = x / gx, r = rad[k], th = (u - 0.5) * Math.PI * 2;
    position[k * 3] = Math.sin(th) * r; position[k * 3 + 1] = rowAngle(y) * r; position[k * 3 + 2] = -Math.cos(th) * r;
    uv[k * 2] = u; uv[k * 2 + 1] = 1 - y / (gy - 1);
  }

  // 5. triangles, cutting the stretched ones; mark the vertices of every cut triangle
  const index = new Uint32Array((gy - 1) * gx * 6), cut = new Uint8Array(nv);
  let n = 0;
  for (let y = 0; y < gy - 1; y++) for (let x = 0; x < gx; x++) {
    const a = y * nx + x, b = a + 1, c = a + nx, e = c + 1;
    for (const [p, q, s] of [[a, c, b], [b, c, e]]) {
      const lo = Math.min(rad[p], rad[q], rad[s]), hi = Math.max(rad[p], rad[q], rad[s]);
      if (hi / lo < EDGE) { index[n++] = p; index[n++] = q; index[n++] = s; }
      else cut[p] = cut[q] = cut[s] = 1;
    }
  }
  // the seam column is one vertex twice: share the mark
  for (let y = 0; y < gy; y++) { const l = y * nx, r = l + gx; cut[l] = cut[r] = cut[l] | cut[r]; }

  // 6. soft edge: alpha 0 on cut vertices, 0.5 on their neighbours, 1 elsewhere (RGBA, white)
  const color = new Uint8Array(nv * 4).fill(255);
  for (let y = 0; y < gy; y++) for (let x = 0; x <= gx; x++) {
    const k = y * nx + x;
    let al = 255;
    if (cut[k]) al = 0;
    else {
      const xl = x === 0 ? gx - 1 : x - 1, xr = x === gx ? 1 : x + 1;
      const nb = [y * nx + xl, y * nx + xr, y > 0 ? k - nx : k, y < gy - 1 ? k + nx : k];
      if (nb.some(m => cut[m])) al = 128;
    }
    color[k * 4 + 3] = al;
  }
  return { position, uv, color, index: index.slice(0, n), ground, triangles: n / 3 };
}

if (typeof self !== "undefined" && typeof window === "undefined") {
  self.onmessage = e => {
    const { id, depth, gx, gy } = e.data;
    try {
      const m = buildSpotMesh(depth, gx, gy);
      self.postMessage({ id, ...m }, [m.position.buffer, m.uv.buffer, m.color.buffer, m.index.buffer]);
    } catch (err) {
      self.postMessage({ id, error: String(err && err.stack || err) });
    }
  };
}
