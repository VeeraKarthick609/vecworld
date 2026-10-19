# vecworld: Road Trip design

Date: 2026-10-19 · Status: draft for review · Replaces Dream Walk as the main game (Dream Walk stays in git history on `dreamwalk`)

## 1. Intent

You walk a small character, seen from behind, down a painted road. The world changes as you travel: a deep forest thins out into farmland, the farmland becomes a village, and the village grows into a city. At crossroads, signposts offer nearby places ("Village", "Lakeshore", "Old Mines"). You pick one, and the road leads there. There's no goal and no score; the point is the journey. It's a dream, so places aren't stored, and only the road you're on has to feel continuous.

**Why this direction:** Dream Walk felt like "a random experiment with vision models". It was first person, with no sense of place, and no way to leave the forest and head somewhere. The user wants a **third-person journey through related places with signposts**. They walked the road spike (2026-10-19) and confirmed it feels right once the character really walks.

Goals that carry over:
- **Smooth movement.** Walking is continuous at browser frame rate; generation happens ahead of you.
- **Serious engineering on modest hardware.** It runs on one laptop with a 4 GB GPU. The user wants to run better, heavier models with clever memory engineering (see §7, phase 2).

## 2. Goals

- **Start:** type a starting place ("haunted forest", "desert oasis") and appear on a road there, your character standing in front of you.
- **Walk:**
  - Hold W, ↑, or the mouse or touch button to walk forward along the road. **Shift** runs.
  - The camera follows behind. The character plays a real walk or run cycle, with ground speed matched to its stride and a contact shadow.
  - The scene moves with depth parallax, so near trees slide past.
- **The road is continuous:**
  - The road is a chain of **stretches**, each a generated image plus a depth map.
  - Each stretch grows out of the previous one, and stretches crossfade as you walk.
  - The scenery drifts from where you are toward where you're heading over several stretches (forest → forest edge → farmland → village outskirts → village).
- **Arriving and forks:**
  - When you reach a destination, the road arrives at a crossroads with 2–3 **signposts**.
  - Each signpost names a nearby, related place that an LLM picked to fit where you've been.
  - A/D (←/→) or clicking selects a sign; W walks that road.
- **The place name shows in the HUD.** Optionally, a small journey trail lists the places visited.
- About 5 friends can each travel their own road at once.

## 3. Non-goals

- A persistent or shared world, a map, or walking back to exactly the same place.
- Free roaming off the road, or turning around and walking back. You only go forward. S, ↓ or releasing W stops.
- Generated characters. The character is a fixed rigged 3D model, because image models can't keep a character consistent.
- Quests, scores, inventory, multiplayer interaction.
- Hosted or paid model APIs.

## 4. Anti-goals

- **The character looks like it's flying or sliding.** Feet must plant, and the walk speed must match the stride. The user rejected the flat sprite for exactly this.
- **A stretch transition that pops or doesn't feel like the same road.**
- **Places that never actually change.** If you walk toward "village", a village must visibly arrive.

## 5. Decided, with spike evidence (2026-10-19, on the laptop)

The throwaway spike lives at `/tmp/roadspike/` (generator `road.py`, viewer `viewer/index.html`, served on `:8091`).

- **Image model: SDXL-Turbo**, set up as follows:
  - UNet quantized to int8 (optimum-quanto), on the GPU.
  - Both text encoders on the CPU in fp32. Embeddings are computed there and moved to the GPU.
  - `madebyollin/sdxl-vae-fp16-fix` VAE, fp16 on the GPU.
  - 896×512.
  - Measured: first stretch **2.2 s**, each later stretch **1.4 s**, peak **3519 MiB**.
  - In the spike, the pipeline needed fp16 casts on the UNet and VAE-encoder inputs because the quantized UNet confuses dtype inference. The real build should solve this cleanly.
- **Depth: Depth Anything V2 Small** (`depth-anything/Depth-Anything-V2-Small-hf`), on the GPU, **~0.15 s** per stretch.
  - Output is relative inverse depth, normalised to 0..1 (1 = near).
  - The viewer maps it to distance with `1/(d·(1/NEAR − 1/FAR) + 1/FAR)`, NEAR = 5, FAR = 60.
- **A stretch** is generated as follows:
  1. Dolly the previous image into the vanishing point (spike: 1.35×, crop centre slightly above middle).
  2. img2img at strength 0.7, 4 steps, prompt `"{ROAD}, {place description}, {STYLE}"`.
  - ROAD is "a dirt road leading straight ahead into the distance, seen from the middle of the road".
  - STYLE is "impressionist oil painting, visible brushstrokes, dreamlike atmosphere, wide angle, eye level".
  - **Known problem:** later stretches go hazy and blurry, from compounding the zoom. Use a smaller dolly, a palette/detail anchor (Dream Walk's `match_palette`), or a periodic fresh txt2img re-anchor. The builder tunes this and checks it by eye.
- **Viewer** (three.js 0.160):
  - Each stretch is a depth mesh: a 224×128 grid along camera rays.
  - Triangles whose depth ratio exceeds 1.35 are **cut**.
  - A **backdrop** copy of the image sits at FAR to fill the gaps.
  - The camera walks about 65% of the stretch (STRETCH = 4.5 units), then crossfades to the next stretch, which starts where its dolly began.
  - Known artefact: jagged, torn edges on near trunks.
- **Character:**
  - Spike placeholder: three.js's CC0 `RobotExpressive.glb`, with Idle/Walking blended by speed, scale 0.11, 2.6 units ahead of the camera on the road.
  - The camera sits behind and looks slightly down the road, bobbing once per footstep.
  - Stride is 1.1 units per walk cycle. Shift runs at 1.8× with the animation sped up.
  - The user said "good now, increase the speed", so the speed was doubled.

## 6. How it works

- **Places and roads:**
  - The journey is a sequence of places. A road between place A and place B is a sequence of stretches. My call: 6 stretches per road.
  - Along it, the place description blends from A to B. The text-embedding weights move A→B (Dream Walk's drift mechanism), and the prompt names the in-between scenery an LLM wrote for that road (e.g. "forest thinning, light ahead", "farmland, a village on a hill").
  - The last stretch is the arrival scene at B, at a crossroads.
- **The dreamer LLM** (Qwen2.5-1.5B-Instruct on the CPU, from Dream Walk, single instance, ~30 s per call) does two things:
  - (a) **Proposes 2–3 next places** related to the current place and the journey so far, for the signposts.
  - (b) **Writes the in-between scenery** for a chosen road: about 4 short descriptions from A to B.
  - It works in the background as soon as you arrive somewhere, or even earlier. Only parsed phrases leave it (the existing parser rules).
  - While it hasn't answered, a small built-in fallback table of related places and transitions is used.
- **Pre-generation:**
  - The player only moves forward, so the next stretches are perfectly predictable. Keep a buffer of up to 3 stretches ahead of the player (image + depth).
  - At a fork, generate the first stretch of every offered road, so whichever sign is chosen, walking starts instantly.
  - All of this goes through Dream Walk's single-worker **GPU queue**: on-demand before prefetch, fair across players, one queued prefetch per player replaced in place.
- **Walking never blocks on generation if the buffer is full.** If the player catches up with generation (e.g. sprinting), the character slows to a stop at the end of the last ready stretch and resumes when the next arrives, with a subtle "dream fog" or holding cue. My call.
- **Carried over from the `dreamwalk` branch:** the FastAPI structure, sessions with TTL/LRU, the error format and 503 on GPU out-of-memory, the GPU queue, the dreamer and its parser, the palette helper, the test markers, and the run and deploy notes.
- **Retired:** the panorama painter, panorama maths, labels and CLIPSeg (signposts replace discovery labels), and the 360° viewer.

### 6.1 Settled after review

- **Several players:** timing targets are single-player (the user decided this for Dream Walk, and it carries over). With several players walking, they share the GPU through the fair queue. If someone's buffer runs dry, their character eases to a stop in "dream fog" at the end of the last ready stretch and resumes when the next arrives. No player cap, no batching.
- **Walk speed:** the spike's final values. Stride 1.1 units per walk cycle, at the robot's Walking clip duration. Shift runs at 1.8× with the animation sped up to match. STRETCH = 4.5 and 65% of each stretch is walked before the crossfade. *(Claude's call.)*
- **CPU sharing:** text encoding for the next stretch has priority over the dreamer LLM. The LLM runs with a capped thread count in the background, and embeddings are cached per prompt text. If the LLM's next places or scenery aren't ready when needed, the fallback table is used for that road. Fallback scenery is a fixed 4-step transition template per place pair, e.g. "{A}, thinning out" → "between {A} and {B}" → "approaching {B}" → "the edge of {B}". *(Claude's call.)*
- **Pass criteria for continuity and anti-blur:**
  - Three reference journeys, each 12 stretches: "haunted forest" → village → city; "desert oasis" → bazaar → port; "snowy mountains" → monastery → valley.
  - Every stretch's sharpness (mean adjacent-pixel difference) must stay at least 0.6× the first stretch's.
  - The controller checks them by eye, and so does the user, by playing.
  - A periodic fresh re-anchor is allowed only if it doesn't pop visibly at the crossfade.
  *(Claude's call.)*
- **GPU memory:** the 3519 MiB spike peak did **not** include depth (depth ran after the image model was freed). Depth Anything V2 Small stays on the GPU if the combined peak stays under 3.6 GB. Otherwise it runs on the CPU, estimated ~0.3 s. The builder measures and decides.
- **Character (the user said "your call"):** version 1 keeps the CC0 RobotExpressive. It already plants its feet with real walk and run clips. It sits behind a single character-model setting with clip-name mapping, so a human traveller can be dropped in later.
- **Clients:** desktop browsers on the Tailscale network come first. Phones are a bonus: touch walk button plus one mobile-viewport browser test. Depth meshes are built off the main thread (a Web Worker) so walking never hitches. Each stretch is sent as a JPEG plus an 8-bit PNG depth map.
- **Nothing is saved after a session.** Stretches are dropped as the player passes them, keeping only a few behind for the crossfade. The journey trail holds only place names.

## 7. Phases

- **Phase 1 (this design): the playable Road Trip** on SDXL-Turbo int8 as measured above.
- **Phase 2 (separate design, later): heavy-model engineering.** The user wants better, heavier models made to run on 4 GB through clever engineering, for example:
  - streaming UNet/transformer blocks from pinned RAM to the GPU just in time on a side CUDA stream
  - 4-bit weights
  - CPU/GPU splits for text encoders
  - weights paged from the SSD through the `io_uring` + `O_DIRECT` storage layer from the original vecworld design
  Candidate models: FLUX.1-schnell-class or SD3.5-class. Phase 1 keeps the image model behind a `StretchPainter` interface so phase 2 can swap it.

## 8. Constraints

- **Hardware:** RTX 3050 Ti 4 GB (about 3.7 GB usable). Peak allocated VRAM **< 3.6 GB**. ~14 GB RAM. Ubuntu laptop `ssh gpu`, repo `~/dev/vecworld`.
- **Latency targets** (single player):
  - Start to first frame: ≤ 6 s.
  - A buffered stretch is always ready before the player reaches it while walking. My call: 3 stretches of buffer at walking speed is about 3 × (4.5 × 0.65) / speed s.
  - At a fork, the first stretch of every offered road is ready within ~5 s of arriving.
  - With several players, they queue fairly.
- **Frame rate:** the viewer runs at full browser frame rate on a laptop. Mesh building happens on the client and must not hitch the walk.
- **Everything local:** no paid APIs, no Docker. Commits carry no Claude attribution.

## 9. Left to the builder

The module layout (new `vecworld/road/` package beside the reused pieces), how continuity and anti-blur are done, dolly amount, prompt wording, the fallback places table, the LLM prompts, the exact buffer logic, the depth-edge cut threshold and backdrop handling, the signpost visuals (HTML overlay or 3D sign models in the scene), the HUD and journey trail, how the character model is loaded (bundled vs CDN, choosing a CC0 model), and test strategy following the earlier conventions (unit / gpu / ui markers, fakes).
