# vecworld: Dream Roam design

Date: 2026-10-19 · Status: draft for review · Supersedes the Road Trip draft (same day). Replaces Dream Walk as the main game; Dream Walk stays in git history on `dreamwalk`.

## 1. Intent

You control a small character in third person and roam a painted, dreamlike world in **any direction**. W/A/S/D move relative to the camera, and the mouse turns the camera around your character. The world is organised by meaning:
- you start in a place you type ("haunted forest");
- nearby, related places lie in different directions ("Village to the east, Lakeshore to the north, Old Mines to the south-west");
- signposts point the way, and walking toward a place makes the scenery drift into it until you arrive.

There's no goal and no score: the point is travelling. It's a dream, so the world isn't stored, and only where you're walking right now has to feel connected.

**Why this design:**
- Dream Walk felt like "a random experiment with vision models". It was first person, with no sense of place and no way to head somewhere.
- The user wants a **third-person journey through related places, with signposts**.
- On the road spike they asked to **move in all directions and use the mouse to turn around**.
- They walked the free-roam spike on 2026-10-19 and said "good".

Goals that carry over:
- **Smooth movement:** walking and looking are continuous at browser frame rate, and generation happens ahead of you.
- **Serious engineering on modest hardware:** one laptop with a 4 GB GPU. Phase 2 (§7) is about running better, heavier models through memory engineering.

## 2. Goals

- **Start:** type a starting place, optionally toggle mood chips (foggy, night, rain, snow, eerie, warm), and appear in it with your character in front of you.
- **Move:**
  - W/A/S/D (or arrows) move relative to the camera, and Shift runs.
  - The character turns to face its direction of travel and plays walk/run/idle animations blended by speed, with ground speed matched to stride and a contact shadow.
  - On phones, an on-screen stick and a drag area are a bonus.
- **Look:** the mouse orbits the camera around the character. Drag, or click to capture the pointer (Esc releases). Pitch is limited so you never look under the world.
- **The world is a chain of spots:**
  - Each spot is a seamless 360° painted panorama with depth, around which you can walk with real parallax.
  - Walking out of a spot in any direction dreams the next spot that way, grown from what lies in that direction, and the world continues without a cut.
- **Places, directions and signposts:**
  - The current place has 2–4 related neighbouring places at compass directions, chosen by an LLM to fit where you've been.
  - Each has a signpost in the scene pointing toward it, and a small HUD compass/arrow.
  - Walking roughly toward a place drifts the scenery into it, and after a few spots you **arrive**: it becomes the current place, with new neighbours.
  - Walking elsewhere keeps you in variations of the current place.
- **The text bar:** typing a new place mid-roam drifts the world toward it, as Dream Walk's drift does.
- About 5 friends can each roam their own world at once.

## 3. Non-goals

- A persistent or shared world, a stored map, or returning to exactly the same place.
- Generated characters. The character is a fixed rigged 3D model, because image models can't keep a character consistent.
- Quests, scores, inventory, multiplayer interaction. Hosted or paid model APIs.
- Discovery labels or CLIPSeg (signposts replace them). Things can still appear in the scenery, unlabelled.

## 4. Anti-goals

- **The character looks like it's flying or sliding.** The user rejected a flat sprite for exactly this. Feet must plant, and speed must match the stride.
- **Moving into the next spot pops, or doesn't feel like the same world.**
- **Places never actually change.** Walking toward "Village" must visibly arrive at a village.
- **Looking at a hole:** void under or around the world.

## 5. Decided, with spike evidence (2026-10-19, on the laptop)

Spikes: free roam at `/tmp/pano3d/` (`gen.py`, `index.html`, served on `:8092`), and the earlier road walk at `/tmp/roadspike/` (`:8091`).

- **Panorama:**
  - The `dreamwalk` painter, `TurboPainter.panorama`: SD-Turbo with wrap-around MultiDiffusion at 2048×512, cylindrical 360° × ~90°, with the final-review fixes (overlapping VAE tiles, warped dolly, LRU embedding cache).
  - Measured **2.8 s** per panorama.
  - Steps in a direction use `TurboPainter.step`: roll to the walk direction, dolly, regenerate the region ahead, apply the palette pull. Measured 3.9–5.7 s in Dream Walk.
  - SDXL-Turbo is **not** used for panoramas: its MultiDiffusion panorama took 21 s.
- **360° depth:**
  - **Depth Anything V2 Small** runs on 8 overlapping 512-px slices of the panorama (stride 256, wrapping).
  - Each slice is least-squares-aligned (scale + shift) to the previous one on their overlap.
  - The loop-closure error is spread linearly around the circle, then the slices are blended with triangular weights and normalised by the 1st/99th percentile.
  - Measured **0.55 s**. The wrap seam in depth was 0.0053 against 0.0042 typical, which is invisible.
- **The viewer (three.js 0.160):**
  - **Mesh:** a cylinder of depth around the spot centre, a 512×128 grid. Column u maps to heading θ = (u − 0.5)·2π, row v to height (0.5 − v)·π/2. Radius = `1/(d·(1/NEAR − 1/FAR) + 1/FAR)` with NEAR = 4, FAR = 50.
  - **Depth edges:** triangles whose radius ratio exceeds 1.35 are cut, and a backdrop cylinder at FAR fills the gaps.
  - **Ground:** the ground level is the median height of the bottom rows. A ground disc (radius 3·NEAR) is textured by unwrapping the bottom 18% of the panorama radially. That fixed the hole below, but it looks streaky.
  - **Character:** CC0 `RobotExpressive.glb`, scale 0.11, with Idle/Walking blended by speed. Stride 1.1 units per walk cycle, Shift ×1.8.
  - **Camera:** orbits at distance 1.9, height 0.55, with pitch limited to −0.1…0.45.
  - **Movement:** camera-relative, and the character turns toward its direction of travel. Movement is clamped to a radius LIMIT = 0.55·NEAR.
  - **Known spike artefacts:** torn trunk edges, a raised ridge where ground meets the trees, and a streaky ground disc.

- **Measured end to end on the built game** (2026-10-19, Task 9; real server on the laptop, headless Chrome on the Mac driving the keyboard, brisk walking toward a neighbour; server build times from a timing wrapper):

  | journey | spots | start → first spot | spot build (server) | fogs while walking | fog lengths | arrived at |
  |---|---|---|---|---|---|---|
  | haunted forest + night | 8 in 43.6 s | 3.9 s | 4.96–5.53 s, median 5.30 | 8 (11/min), 15.1 s in all | 0.03–5.0 s | lake (spot 3), then marsh |
  | desert oasis | 8 in 58.3 s | 2.8 s | 5.17–5.44 s, median 5.22 | 9 (9.3/min), 20.5 s | 0.02–5.1 s | desert dunes (spot 3), then oasis |
  | snowy mountains | 8 in 52.4 s | 5.6 s | 5.21–5.60 s, median 5.32 | 9 (10.3/min), 19.0 s | 0.08–7.9 s | snowy peak trail (spot 3), then village |
  | haunted forest + night, **running** | 8 in 52.9 s | 5.7 s | 5.23–5.59 s, median 5.33 | 10 (11.3/min), 33.4 s | 0.03–7.1 s | haunted glen (spot 3), then village |

  - The first spot builds in 2.6–3.6 s; the rest of "start → first spot" is the depth/image download and the mesh.
  - Spots follow each other every 5.2–5.7 s while walking (10–10.5 s after an on-demand build). The game is GPU-throughput bound: about 5.3 s per spot against at most ~5.7 s of walking rim to rim, so the heading pre-dream is usually still running at the edge ("waited" served 28 of 34 moves, "demand" 5, a finished pre-dream 1). Running only lengthens the fogs.
  - Peak GPU memory 3764 MiB of 4096 (2662 MiB with the server idle).
  - Frame time on the Mac's headless Chrome (Metal): p50 16.7 ms, p99 16.8 ms (vsync-bound 60 fps).
  - The places LLM takes 9.4–22.4 s per answer on the CPU. The first LLM places landed 12–23 s after the start build began; in two of the four journeys the player had already arrived at a fallback neighbour by then, so the LLM's names for the start place were never seen.
  - Walking 180° back returned into the kept spots without a fog in two of three journeys; in the snowy one a forward step (toward farm) was still in flight when the player turned back, and it landed while the player waited at the back rim: the viewer entered it from there, moving that spot to the player, with an "Arrived at farm" toast and a 4.5 s fog. Fixed since: a landed spot is entered at once only from the rim it was asked for, and its arrival is told when the player enters it.

## 6. How it works

- **Spots and transitions:**
  - The player roams within a spot's walkable radius. When their position passes about 40% of the radius heading outward, the server pre-generates the **next spot in that direction**, using Dream Walk's prefetch with the 45° tolerance.
  - The next spot = the current panorama rolled to that heading and dollied in (the warped dolly, so nothing tears behind you), with the region ahead regenerated (focus prompt) and the palette pulled to the place's reference. Then 360° depth is computed.
  - At the radius edge, the world switches to the new spot. The new mesh is centred where the old one's dolly implies, ahead along the heading. It crossfades in while the player keeps walking, so there's no stop and no cut. The builder works out the exact maths so the scene doesn't jump, and checks it by eye.
  - If the next spot isn't ready at the edge, the character eases to a stop in a soft "dream fog" and continues as soon as it arrives.
- **Places:**
  - The session holds the current place, its 2–4 neighbours (name + compass bearing), and a drift state per neighbour.
  - Each spot built while heading within ±45° of a neighbour's bearing advances that neighbour's drift (text-embedding weights from the current place to the neighbour over about 3 spots, Dream Walk's drift mechanism). Moving away resets it.
  - When a drift completes, you have **arrived**: the neighbour becomes the current place, the colour reference resets, and new neighbours are requested.
  - The opposite direction always includes the place you came from, so you can go back. It's a dream, so it may look different.
- **The dreamer LLM** (Qwen2.5-1.5B-Instruct on the CPU, from Dream Walk, single instance, capped threads, ~30 s per call):
  - It proposes the neighbours for a place, given the journey so far, and gives each a short scenery description for the in-between spots.
  - Only parsed phrases leave it.
  - While it hasn't answered, a built-in fallback table of related places is used, with generic in-between scenery.
- **Signposts:** a simple 3D signpost model is placed near the spot centre, with one board per neighbour pointing at its bearing. The HUD shows a small compass with the neighbour names. Signposts are client-side 3D objects, not painted by the model.
- **Prompt:** "{place description}, {in-between scenery if drifting}, {mood words}, {STYLE}", reusing Dream Walk's MOOD_WORDS and STYLE. The panorama prompt also asks for open ground near the camera, so the walkable area isn't a tree trunk.
- **GPU sharing:** one GPU worker shared by all players: on-demand jobs before prefetch, fair across players, one queued prefetch per player replaced in place. The painter and depth model both live on the GPU if the combined peak stays under 3.6 GB; otherwise depth runs on the CPU. The builder measures.
- **Carried over from the `dreamwalk` branch:**
  - The painter, panorama maths, the GPU queue, sessions with TTL/LRU, errors and 503 on out-of-memory, the dreamer and its parser, palette, mood words, the text bar, mood chips, and the test markers.
  - Run/deploy notes, JPEG scenes, and file cleanup.
- **Retired:** the labels, CLIPSeg, the 2D panorama canvas viewer, `approach`/`back`, and the Road Trip draft.

### 6.1 Settled after review

- **Pace and speculative pre-dreaming (the user chose this, option 3 of 3):**
  - Keep the spike's brisk walking speed (stride 1.1 units per cycle, Shift ×1.8).
  - As soon as the player enters a spot, the server pre-dreams the next spot in **three directions at once**: the current heading and ±60° from it. The heading is done first.
  - As the heading changes, the set re-targets: new directions replace queued, not-started ones, and finished ones are kept while they're still within 45° of a candidate direction.
  - At the edge, the finished spot closest to the exit heading (within 45°) is used. Otherwise, dream fog while the exact direction is generated at demand priority.
  - The GPU queue must allow up to 3 queued prefetches per player, replaced as a set, instead of build 2's single replace-in-place prefetch.
  - The user accepted the cost: with several friends roaming, the GPU saturates and fog pauses grow. Phase 2's faster models are what make this scale.
  - Make the walkable radius as large as the depth allows without walking into the painted geometry, and measure how often fog happens.
- **Transition first:** milestone 1 of the build is a spot-to-spot transition prototype with real models, checked by eye. Depth scale differs per spot, so the transition aligns the new spot's depth scale to the old one on their shared view direction, matching the median radius of the overlapping region. If a seamless switch can't be achieved, a soft dissolve through a light dream haze (well under 1 s) is the accepted fallback, since it's a dream. *(Claude's call.)*
- **Bearings are assigned by code, not the LLM.**
  - Neighbours are spread around the compass, starting from a random offset.
  - The place you came from is always at the bearing behind your arrival direction.
  - The LLM only names neighbouring places and their in-between scenery. The fallback table covers the first ~30 s, and that's acceptable.
  - One QwenDreamer serves all players, with capped threads, and text encoding for painting has priority.
  *(Claude's call.)*
- **Arrival and palette:**
  - Arrival is judged by eye on the reference journeys (§9). No CLIP harness.
  - On arrival, the colour reference moves to the new place gradually over the drift spots, not in one jump, so nothing pops.
  *(Claude's call.)*
- **Sharpness metric:** mean absolute difference between adjacent pixels over the whole panorama (the metric used in builds 1–2), compared with the journey's first spot. *(Claude's call.)*
- **Depth on CPU fallback:** only if both models can't share the GPU under 3.6 GB. If CPU depth breaks the timing, reduce the depth slice resolution first. *(Claude's call.)*
- **Assets:** the character GLB and the three.js modules are bundled with the app (served locally), not loaded from a CDN. *(Claude's call.)*
- **Players:** "about 5 friends at once" is supported through the fair queue with fog waits. It's not a throughput guarantee.

## 7. Phases

- **Phase 1 (this design):** the playable Dream Roam on SD-Turbo panoramas plus Depth Anything.
- **Phase 2 (separate design, later): heavy-model engineering.** The user explicitly wants better, heavier image models running on 4 GB through clever engineering, for example:
  - streaming transformer/UNet blocks from pinned RAM to the GPU just in time on a side CUDA stream
  - 4-bit weights
  - CPU/GPU splits
  - weights paged from the SSD through the `io_uring` + `O_DIRECT` layer from the original vecworld design
  Phase 1 keeps the painter behind its `Painter` interface so phase 2 can swap in a better model.

## 8. Constraints

- **Hardware:** RTX 3050 Ti 4 GB (about 3.7 GB usable). Peak allocated VRAM **< 3.6 GB**. ~14 GB RAM. Laptop `ssh gpu`, repo `~/dev/vecworld`.
- **Latency targets** (single player; with several players they queue fairly, and the dream fog covers waits):
  - Start to first spot: ≤ 6 s.
  - When walking steadily outward, the next spot is ready before the player reaches the edge. At walking speed, the time from 40% to 100% of the radius must exceed the measured spot time; the builder adjusts the trigger or radius to make that true.
- **Frame rate:** the viewer runs at full browser frame rate on a desktop. Spot meshes are built off the main thread (a Web Worker), so a new spot never hitches movement.
- **Transfer:** each spot is sent as a JPEG panorama plus an 8-bit PNG 360° depth map.
- **Everything local:** no paid APIs, no Docker. Commits carry no Claude attribution. The character is the CC0 robot for v1, behind a single model setting with clip-name mapping, so a human traveller can be swapped in later.
- **Clients:** desktop browsers on the Tailscale network come first. Phones are a bonus.

## 9. Pass criteria for visual quality

- **Three reference journeys of 8 spots each**, walking toward a neighbour each time:
  - "haunted forest" with night;
  - "desert oasis";
  - "snowy mountains".
  Each must:
  - arrive at a visibly different place within ~3–4 spots;
  - keep sharpness ≥ 0.6× the first spot's;
  - show no visible seam or hole in normal looking.
- The controller checks them by eye, and so does the user, by playing.
- The torn trunk edges, the ground ridge and the streaky ground disc must be visibly improved over the spike, compared side by side.

## 10. Left to the builder

The module layout (a new `vecworld/roam/` package beside the reused pieces), the exact spot-transition maths and crossfade, the edge-tear and ground-ridge fixes (for example depth smoothing, soft alpha at cut edges, or a ground plane fitted from depth), how the ground texture is made, the walkable radius and prefetch trigger tuning, the signpost model and HUD compass design, the fallback places table, the LLM prompts, how the character model is loaded (bundled vs CDN), and test strategy following the earlier conventions (unit / gpu / ui markers, fakes, a puppeteer-driven end-to-end check from the Mac).
