// The HUD compass: a strip along the top showing the bearings within ±RANGE of where the camera looks, with the
// cardinal points, one marker per neighbouring place and a "came from" marker. Of the places off the strip (or too
// near its ends to show whole), the one nearest the view on each side is pinned under that end with an arrow,
// pointing the way to turn.
export const RANGE = 75;                   // degrees either side of the view shown across the strip
const PIN = RANGE - 16;                    // places further off than this are pinned (half a pill is ~15°)

const CARDINALS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];

// Signed degrees from a to b, in (−180, 180].
export const rel = (a, b) => { const d = ((b - a) % 360 + 540) % 360 - 180; return d === -180 ? 180 : d; };

export class Compass {
  constructor(el) {
    this.el = el;
    this.strip = el.querySelector(".strip");
    this.placeEl = el.querySelector(".here");
    this.ticks = CARDINALS.map((name, i) => {
      const t = document.createElement("span");
      t.className = "tick" + (i % 2 ? " minor" : "");
      t.textContent = name;
      this.strip.append(t);
      return { el: t, bearing: i * 45 };
    });
    this.back = document.createElement("span");
    this.back.className = "came-from";
    this.back.textContent = "came from";
    this.back.hidden = true;
    this.strip.append(this.back);
    this.backBearing = null;
    this.places = [];
  }

  // neighbours: [{name, bearing}]; here: the current place; cameFrom: the bearing back the way you came, or null.
  set(neighbours, here, cameFrom = null) {
    for (const p of this.places) p.el.remove();
    this.places = neighbours.map(n => {
      const el = document.createElement("span");
      el.className = "place";
      el.dataset.name = n.name;
      el.title = `${n.name}, ${Math.round(n.bearing)}°`;
      const arrow = document.createElement("i");
      const label = document.createElement("b");
      label.textContent = n.name;
      el.append(arrow, label);
      this.el.append(el);                  // over the strip, or under its ends when pinned
      return { el, arrow, bearing: n.bearing };
    });
    this.placeEl.textContent = here;
    this.backBearing = cameFrom;
    this.back.hidden = cameFrom == null;
  }

  // Lay the strip out for the camera looking along `view` (a bearing, degrees).
  update(view) {
    const at = (el, d) => { el.style.left = `${50 + 50 * d / RANGE}%`; };
    for (const t of this.ticks) {
      const d = rel(view, t.bearing);
      t.el.style.visibility = Math.abs(d) <= RANGE ? "visible" : "hidden";
      at(t.el, d);
    }
    // Off the strip, only the place nearest to view on each side is pinned to that end (they would overlap).
    const nearest = {};
    for (const p of this.places) {
      p.d = rel(view, p.bearing);
      const side = p.d < 0 ? "left" : "right";
      const best = nearest[side];
      if (Math.abs(p.d) > PIN && (!best || Math.abs(p.d) < Math.abs(best.d))) nearest[side] = p;
    }
    for (const p of this.places) {
      const off = Math.abs(p.d) > PIN, side = p.d < 0 ? "left" : "right";
      p.el.classList.toggle("off-left", off && side === "left");
      p.el.classList.toggle("off-right", off && side === "right");
      p.el.style.visibility = !off || nearest[side] === p ? "visible" : "hidden";
      p.arrow.textContent = off ? (side === "left" ? "‹" : "›") : "";
      if (off) p.el.style.left = "";
      else at(p.el, p.d);
    }
    if (this.backBearing != null) {
      const d = rel(view, this.backBearing);
      this.back.style.visibility = Math.abs(d) <= RANGE ? "visible" : "hidden";
      at(this.back, d);
    }
  }
}
