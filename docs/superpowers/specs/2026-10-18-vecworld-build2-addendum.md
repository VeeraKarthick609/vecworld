# vecworld build 2 addendum: walking

Date: 2026-10-18 · Builds on: `2026-10-18-vecworld-design.md` and the slice 1 plan.
Scope approved by the user. This note covers what changes and the calls made.

## Measured starting point (build 1, seed 1)

On 8 jump words (guitar, elephant, kitchen, anger, volcano, library, bicycle,
friendship), **13 of 32 hotspots (41%) were fallback slots**, meaning boxes for
concepts CLIPSeg couldn't find. anger and friendship came out as misty
landscapes where nothing could be located. kitchen, library and their
neighbours each got a box covering 96–99% of the frame. A 7-step walk from
lighthouse went beacon → lighthouse → beacon … and drifted to saturated orange.
Jumps took ~1.95 s and steps ~2.2 s, with a peak of 3.1 GiB allocated.
CLIPSeg alone took ~1.7 s for 7 prompts because it ran the vision tower once per
prompt on the same image.

## What changes

1. **Hotspots are only what CLIPSeg finds.** The prompt names the centre plus
   5 neighbours. CLIPSeg is asked about the centre plus 6 candidate neighbours,
   with **one vision pass per image** (0.25 s instead of 1.7 s, same output).
   - A box is the bounding box of the **connected blob around the heatmap
     peak** (pixels ≥ 0.5 × peak), not of every pixel above that level.
   - A neighbour is offered only if its peak is ≥ `HOTSPOT_MIN`, its box covers
     ≤ `MAX_HOTSPOT_AREA = 0.5` of the frame (a whole-room box isn't an object),
     and its IoU with every box already offered is ≤ `MAX_HOTSPOT_IOU = 0.6`.
     Candidates are taken in meaning order, up to 3. Fallback slots are gone.
   - **Call: the centre concept always gets a hotspot.** If CLIPSeg finds it
     with a sensible box, that box is used. If it isn't found, or its box covers
     more than half the frame (a kitchen *is* the scene), it gets a fixed
     central box `CENTRE_BOX = (0.35, 0.35, 0.3, 0.3)`. It's what the prompt put
     "in the centre" and where the zoom lands. *If wrong:* the centre box is
     sometimes over empty space, which is the one remaining non-located hotspot.
   - Labels are always visible, at the top of each box. Boxes keep the build 1
     largest-first order, and the IoU rule stops two boxes from coinciding.
2. **Abstract words get real scenes.** STYLE loses "landscape". A WordNet
   check decides the framing: the word is concrete when its first noun sense
   descends from `physical_entity.n.01`.
   - concrete: `a scene of a {c} in the centre, with {others} around it, {STYLE}`
   - abstract: `a symbolic scene evoking {c}, with {others}, {STYLE}`
   - *Call:* use only the dominant sense. fire, sunset, beacon and night count
     as abstract, which still renders them fine ("evoking fire"). *If wrong:* a
     few concrete things get a moodier scene.
   - The subject now comes before the style, so CLIP weights it more.
3. **Walking with keys.** One press is one step, and auto-repeat is ignored.
   - **W / ↑ forward:** the session keeps a *heading*, the unit tangent of the
     last step's great circle at the landing point. Forward moves exactly
     `MAX_STEP_ANGLE` along it. The new centre is the nearest concept not
     recently visited, and the zoom-crop is the image centre.
     *Call: with no heading yet* (just after a jump or a back), the heading
     points at the most similar concept not recently visited, which is usually
     the first offered neighbour. *If wrong:* the first W can go toward
     something that isn't drawn.
   - **A / ← and D / →:** step toward the leftmost (or rightmost) non-centre
     hotspot whose box centre is left (or right) of the frame's midline. If
     there isn't one, a toast says so and nothing moves. It's the same as
     clicking that object. The choice happens in the browser because it's
     about screen layout.
   - **S / ↓ back:** restores the previous state exactly (position, heading,
     recent list, scene, hotspots, trail) from a bounded history of
     `HISTORY = 10` snapshots. There's no GPU work. The PNG is already on disk,
     and the session reloads it as the next img2img source. Back with an empty
     history returns 409 `no_history`. Every jump, step and forward pushes a
     snapshot.
   - Keys are ignored while the jump box has focus, while a move is in flight,
     with Ctrl/Alt/Meta held, and before the first scene.
4. **No cluster looping.** The session keeps the last `RECENT = 4` centres.
   They're excluded from the neighbour candidates and from forward's choice of
   centre. A clicked concept is still pinned even if it's recent. A jump resets
   the list to its own centre.
5. **Colour drift fix.** Each step scene is pulled toward the palette of the
   **current walk's jump scene** using a Reinhard mean/std transfer in CIELAB
   at `PALETTE_STRENGTH`, tuned visually. *Call: the reference resets on every
   jump*, not once per session, because a jump is allowed to teleport to a new
   palette. *If wrong:* a palette can change after a jump, which is what a jump
   is for anyway.

## API additions

- `POST /api/session/{sid}/forward` and `POST /api/session/{sid}/back` take no
  body and return the same scene JSON. They use the same busy guard and error
  codes, plus 409 `no_history`.
- The scene JSON gains `"centre": "<concept>"`, the pinned concept.

## Not in this build

Real latent morphs, smooth gliding, prefetch, SmartBatch, caching, a public
tunnel. Scene PNGs are still never deleted, as in build 1.
