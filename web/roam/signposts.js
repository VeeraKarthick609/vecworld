// The signpost: a wooden post near the spot centre with one arrow-shaped board per neighbouring place, each
// pointing along that place's bearing and carrying its name on both faces. Client-side 3D, never painted.
//
// It stands OFFSET units from the spot origin, toward the spot heading + SIDE degrees (ahead and to the right of
// where the character appears), in the spot's own frame, so it travels with the spot through a re-root. Its size
// is the character's scale, not the spot's: it is a prop beside the character.
import * as THREE from "three";
import { bearingVec } from "./world.js";

export const OFFSET = 0.6, SIDE = 35;
const POST_H = 0.62, BOARD_L = 0.4, BOARD_H = 0.075, BOARD_T = 0.012, GAP = 0.085;
const RAD = Math.PI / 180;
const WOOD = 0x7a5a38, POST = 0x5b4128;

function labelTexture(name) {
  const c = document.createElement("canvas"); c.width = 512; c.height = 96;
  const g = c.getContext("2d");
  g.fillStyle = "#8a6842"; g.fillRect(0, 0, c.width, c.height);
  g.fillStyle = "rgba(0,0,0,0.12)";
  for (let y = 8; y < c.height; y += 14) g.fillRect(0, y, c.width, 2);    // grain
  let size = 64;
  g.font = `600 ${size}px system-ui, sans-serif`;
  while (g.measureText(name).width > c.width - 90 && size > 24) { size -= 4; g.font = `600 ${size}px system-ui, sans-serif`; }
  g.fillStyle = "#f6ecd8"; g.textBaseline = "middle";
  g.fillText(name, 22, c.height / 2 + 2);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4;
  return t;
}

// An arrow-shaped plank lying along +x from the post (the tip at +x), BOARD_H tall, centred on y = 0.
function boardShape() {
  const s = new THREE.Shape(), h = BOARD_H / 2, tip = 0.05, x0 = 0.015;
  s.moveTo(x0, -h); s.lineTo(BOARD_L - tip, -h); s.lineTo(BOARD_L, 0); s.lineTo(BOARD_L - tip, h); s.lineTo(x0, h);
  s.closePath();
  return s;
}

export class Signposts {
  constructor(scene) {
    this.group = new THREE.Group(); scene.add(this.group);
    this.group.visible = false;
    this.wood = new THREE.MeshLambertMaterial({ color: WOOD });
    this.boardGeo = new THREE.ExtrudeGeometry(boardShape(), { depth: BOARD_T, bevelEnabled: false });
    this.boardGeo.translate(0, 0, -BOARD_T / 2);
    this.faceGeo = new THREE.PlaneGeometry(BOARD_L - 0.07, BOARD_H * 0.86);
    const post = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.022, POST_H, 10),
      new THREE.MeshLambertMaterial({ color: POST }));
    post.position.y = POST_H / 2;
    const cap = new THREE.Mesh(new THREE.ConeGeometry(0.03, 0.04, 10), post.material);
    cap.position.y = POST_H + 0.02;
    const shadow = new THREE.Mesh(new THREE.CircleGeometry(0.07, 24),
      new THREE.MeshBasicMaterial({ color: 0, transparent: true, opacity: 0.3, depthWrite: false }));
    shadow.rotation.x = -Math.PI / 2; shadow.position.y = 0.008; shadow.renderOrder = 1;
    this.fixed = [post, cap, shadow];
    this.group.add(...this.fixed);
    this.boards = [];
    this.names = [];
    this.spot = null;
  }

  // Stand the post at world entry `spot` (see world.js) with one board per neighbour {name, bearing}.
  set(spot, neighbours) {
    this.spot = spot;
    for (const b of this.boards) {
      this.group.remove(b);
      b.traverse(o => { if (o.isMesh && o.material !== this.wood) { o.material.map.dispose(); o.material.dispose(); } });
    }
    this.boards = neighbours.map((n, i) => {
      const board = new THREE.Group();
      board.add(new THREE.Mesh(this.boardGeo, this.wood));
      const mat = new THREE.MeshLambertMaterial({ map: labelTexture(n.name) });
      const front = new THREE.Mesh(this.faceGeo, mat), back = new THREE.Mesh(this.faceGeo, mat);
      const cx = 0.02 + (BOARD_L - 0.07) / 2;
      front.position.set(cx, 0, BOARD_T / 2 + 0.001);
      back.position.set(cx, 0, -BOARD_T / 2 - 0.001);
      back.rotation.y = Math.PI;              // faces the other way, and still reads left to right from there
      board.add(front, back);
      // local +x turned by θ about y points along (cos θ, −sin θ) in (x, z); the bearing's direction is
      // (sin b, −cos b), so θ = 90° − b.
      board.rotation.y = (90 - n.bearing) * RAD;
      board.position.y = POST_H - 0.06 - i * GAP;
      board.userData.name = n.name;
      this.group.add(board);
      return board;
    });
    this.names = neighbours.map(n => n.name);
    this.group.visible = true;
  }

  // Where the post stands in the current frame (frame units), or null.
  get origin() {
    const s = this.spot;
    if (!s) return null;
    const v = bearingVec(s.heading + SIDE);
    return { x: s.origin.x + s.scale * OFFSET * v.x, z: s.origin.z + s.scale * OFFSET * v.z };
  }

  // Follow the spot (its origin moves on every re-root); k is the world's view scale. Call once per frame.
  place(k) {
    const o = this.origin;
    if (!o) return;
    this.group.position.set(k * o.x, 0, k * o.z);
  }
}
