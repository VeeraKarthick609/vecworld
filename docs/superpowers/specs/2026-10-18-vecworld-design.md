# vecworld: walking through an LLM's meaning-space

Date: 2026-10-18 · Status: draft for review · Working name: vecworld

## 1. Intent

A browser game where you explore the inside of a real LLM. Where you stand is a
vector in the model's embedding space. What you see around you is a generated
picture of the concepts the model places near that vector. Moving through the
world means moving through meaning-space.

Two reasons it exists, and both matter equally:

1. **To see how an LLM thinks.** The fun is in the surprises: why "ocean"
   sits close to "loneliness", where "storm" leads. The pictures have to be
   honest. The real vectors decide *what* appears, and the image model only
   decides *how it looks*.
2. **To show off serious engineering on modest hardware.** It all runs on one
   laptop with a 4 GB GPU, about 14 GB of usable RAM and an NVMe SSD, serving a
   few people at once. The serving, caching and storage layers are part of the
   point, not an afterthought.

Audience: about 5 friends and work colleagues opening a shared link, all at
once in the worst case. Not a public launch.

## 2. Goals

- A player types a word or phrase and lands in a generated scene of that
  region of meaning-space.
- A player clicks an object in the scene and steps toward that concept. The
  scene morphs smoothly into the next one.
- Every clickable object in a scene stands for a real nearby concept, taken
  from the vectors, not invented by the image model.
- Steps feel connected (see the anti-goal).
- A corner mind-map shows where you are in meaning-space.
- It runs entirely on the `gpu` laptop with no paid APIs, and handles 5
  concurrent players without falling over.

## 3. Non-goals

- Scores, goals, quests, winning. It's solo exploration.
- Players seeing or meeting each other. Five people can play at once, but
  each one is alone in their own world.
- Accounts, logins, saved profiles.
- Continuous free movement like a 3D game, which isn't realistic on this
  hardware. Movement is in discrete steps, made to feel continuous by morphs.
- Hosted model APIs (OpenAI, Gemini, FLUX, and so on).
- Scale beyond about 5 concurrent players for now.
- Mobile polish. It should work in a desktop browser. Phones are a bonus.

## 4. Anti-goal

**Scenes that don't feel connected when you step.** If a colleague clicks the
ship and the next scene feels like a random teleport, the project has failed,
however good the engineering underneath. Continuity is the quality bar every
other choice gets measured against.

"Connected" means:
- **Same world look:** one art style, palette and lighting mood across scenes.
- **The clicked object stays and becomes the centre of the next scene.**
  (The user's words.)
- **You can tell why you're here:** the new scene is visibly related to the
  last one.
- **The trip is visible:** the morph shows the path between the two scenes.

A *jump* (typing a destination) is allowed to feel like teleporting, because
the player asked for it. Only *steps* must feel connected.

## 5. The player experience

### 5.1 Starting
The player opens the link and sees an empty, styled screen with a typing bar.
They type a first word or phrase, and that's where they land. There's no
default spawn point. *(Claude's call.)*

### 5.2 The play screen (layout B, chosen from mockups)
- A full-screen generated scene.
- Clickable objects in the scene. Hovering one shows its concept word.
- A typing bar for jumping anywhere.
- A **mind-map** in the corner: a small 2D map of meaning-space with the player
  (highlighted), their trail, nearby concept dots, and a ring showing how far
  one step can reach.

### 5.3 Moving
**Jump (type):** the player types text. The text is embedded with the LLM, the
player moves to that vector, and a fresh scene is generated there. A short
wait is acceptable.

**Step (click):** the player clicks an object, which is a concept C.
- The player's vector moves toward C's vector, but **never more than a fixed
  maximum distance per step**. The user's example was "5 points maximum in a
  single click". The cap is a tunable constant.
- The movement can go in any direction, so everywhere reachable in one step
  forms a small ball around the player.
- The clicked object stays and becomes the centre of the next scene.
- **When the cap stops the player short of C**, C is still pinned at the
  centre. Everything else in the scene comes from the neighbours of where the
  player actually landed. *(Decided after review.)*
- The scene **morphs** into the next one over about a second, rather than
  cutting or crossfading.

### 5.4 Art style
One fixed style for the whole world: painterly and dreamlike. It suits fast
small image models' rougher output and helps continuity. *(Claude's call. The
exact style prompt is up to the builder.)*

## 6. How it works

### 6.1 Meaning-space from a real LLM
- Vectors come from a **real small LLM** (for example Qwen 0.5B-class), not a
  dedicated embedding model. The user chose this because the point is to see
  how an *LLM* thinks, and accepts that the space will be messier.
- A **fixed vocabulary of concepts** (a few thousand words and short
  phrases) is embedded once, up front, and kept in RAM. "What's near me" means
  nearest-neighbour search over this vocabulary.
- Free-text jumps are embedded live and cached by query.
- The mind-map uses a 2D projection of the vocabulary that's computed once.
- Left to the builder: which model, which layer or pooling to turn hidden
  states into vectors, how to build the vocabulary, the distance metric, how to
  express the step cap in that metric, and the projection method. The builder
  should check that nearest neighbours look sensible before building on top.

### 6.2 Scenes
- A scene is generated from the concepts nearest the player's vector, plus the
  fixed style.
- **Building x+1 from x (user's idea):** a step's scene is built from the
  previous scene, not from scratch. This serves continuity and speed at once.
  Candidate techniques: image-to-image from the previous scene with a fixed
  seed, zooming or cropping toward the clicked object so it becomes the centre,
  and blending the two scenes' latents. The builder chooses and tunes.
- **What "honest" means (decided after review):** every *clickable* object
  must be a real current neighbour, or the pinned clicked concept. Background
  scenery carried over from earlier scenes is allowed, and should fade out over
  a few steps. That permits heavy image-to-image for continuity.
- **Shared vs private (decided after review):** jump landing scenes are
  shared across players and sessions. Positions are snapped to a grid with a
  deterministic seed per cell, so the world fills in as people explore, and
  that is the SSD tier's main reuse path. Step scenes are private to each
  player's path, because they're built from that player's previous scene. They
  are still written to the SSD for revisits within a session.
- **The morph** is a short sequence of in-between frames from interpolating
  latents and denoising briefly, so each frame looks like a real place. A
  stepped-to scene's morph frames should be generated ahead along with the
  scene.
- **Clickable objects:** the system has to know where each nearby concept
  appears in the picture so it can place a hotspot and a label. Approaches:
  control placement during generation (regional prompts or layout control), or
  detect the objects afterward with an open-vocabulary detector or image-text
  similarity. Builder's choice, within the VRAM budget.
- Image model: a fast small diffusion model that fits in 4 GB alongside the
  LLM (SD-Turbo class, about 512×512). Roughly 1–2 s per image is an estimate
  that still needs measuring.

### 6.3 The sliding window (prefetch)
Because a step's reach is bounded, the set of possible next scenes is small
and known. While the player looks at the current scene, the system
pre-generates the scenes (and morphs) for the most likely next steps, most
importantly one per clickable object. Then a click plays a morph right away
instead of waiting. Each player gets a prefetch budget so one player can't
starve the others on the shared GPU.

**GPU budget (decided after review):** the GPU can't prefetch everything for
5 players, so:
- 3–4 clickable objects per scene
- prefetch only the 1–2 most likely next steps per player
- morph frames are cheap (low resolution, few denoising steps)
- targets: a click shows motion within **2 s**, and a jump lands within
  **5 s**

An instant morph is best-effort. If a click hits a step that wasn't
prefetched, show a cheap holding animation (for example a zoom toward the
clicked object) while the real scene generates.

### 6.4 Memory tiers
The user's framing: **GPU for runtime, RAM for precomputation, and the SSD
worked hard as a large, fast RAM extension for smooth rollout.**

| Tier | Holds |
|---|---|
| VRAM (4 GB) | The loaded models (LLM + image model) and in-flight generation |
| RAM | The vocabulary vectors, its projection, hot scenes, prefetched neighbours, query-embedding cache |
| SSD | Every scene ever generated, stored as compact latents and as images |

Note: a 512×512 SD latent (4×64×64 at fp16) is 32 KB, which is exactly one
fixed, aligned slot.

### 6.5 The SSD storage layer
- **Chosen:** `io_uring` + `O_DIRECT` on a **raw partition**, with no
  filesystem. The layout is fixed-size aligned slots, an index in RAM, and
  large sequential writes to keep flash wear low. Prefetches for a player's
  neighbourhood go out as one batch of reads so the SSD works at high queue
  depth.
- **Not chosen:** SPDK full kernel bypass, because it needs a whole second
  SSD.
- Written as a small native library (C, C++ or Rust; builder's choice) that
  the Python backend calls.
- Runs directly on the host, not inside a default Docker container, because
  Docker's seccomp profile blocks `io_uring`.
- `io_uring` power-ups (SQPOLL, IOPOLL, registered buffers and files) are
  options to measure, not requirements.

### 6.6 Serving
- **Python** backend.
- **SmartBatch** (the user's own dynamic-batching inference server,
  github.com/VeeraKarthick609/SmartBatch) serves the models: the LLM embedder
  and the image model, registered as separate models. It batches concurrent
  requests and sheds load with 429s rather than running out of GPU memory.
- Use SmartBatch's current API wherever it fits. If morphs, image-to-image
  or latent in/out need something it doesn't support, extend SmartBatch on a
  branch in its own repo rather than building around it. *(Claude's call.)*
- The game backend is a separate Python service that calls SmartBatch over
  HTTP.
- Front end: a browser page served by the backend. Framework is the builder's
  choice.
- Friends outside the user's Tailscale network reach it through Tailscale
  Funnel or a Cloudflare Tunnel.

## 7. Constraints

- **Hardware (the `gpu` laptop, `ssh gpu`):** RTX 3050 Ti laptop GPU with
  4 GB, about 14 GB of usable RAM, a 477 GB NVMe, Ubuntu with kernel
  7.0.0-34, Python 3.14. Dual-boot with Windows.
- The laptop must stay awake and online while people play. Never restart
  `systemd-logind` on it.
- Development happens on the `gpu` laptop (the user's instruction).
- Fully local. No paid APIs.

## 8. Known blockers (found 2026-10-18)

1. ~~**The NVIDIA driver isn't working.**~~ **Fixed 2026-10-18.** Kernel
   `-34` had no matching prebuilt NVIDIA module. The laptop moved to kernel
   `7.0.0-38` with driver `595.99.02` (CUDA 13.2), and `nvidia-smi` shows the
   RTX 3050 Ti with 4096 MiB.
2. **There's no free partition for the raw storage layer.** The NVMe holds
   EFI, Windows NTFS partitions (325 GB, 917 MB, 10.3 GB) and Ubuntu's ext4
   root (140 GB, 71 GB free). Getting a raw partition means shrinking either
   Windows or the ext4 root (which can't be shrunk while mounted, so it needs a
   live USB). Either way it's risky, and the user runs it themselves. Until
   then, the storage layer can be developed against a large preallocated file
   opened with `O_DIRECT`, keeping the same slot layout, so switching to the
   partition later is a path change. **The first "done" version runs on the
   file. Moving to the raw partition is a separate later milestone, after the
   user resizes the disk.** *(Claude's call.)*

The user asked for blocker 1 (the driver) to be fixed first, before other
work.

## 9. Left to the builder

The LLM choice and vector extraction, vocabulary construction, distance metric
and step-cap units, projection method, the image model, how continuity and
morphs are made, how hotspots are located, prefetch policy, cache eviction,
the native library's language, the front-end framework, and tunnel choice.
Where a choice affects continuity, continuity wins.
