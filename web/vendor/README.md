# Vendored browser assets

Bundled so the Dream Roam viewer makes no CDN requests at runtime (spec §6.1, "Assets"). The pages map them with an
import map: `"three"` → `vendor/three.module.min.js`, `"three/addons/"` → `vendor/`. `loaders/` and `utils/` keep
three.js's `examples/jsm/` layout, because `GLTFLoader.js` imports `../utils/BufferGeometryUtils.js`. The files are
unmodified copies, fetched once on 2026-10-19.

| File | Source | Licence |
| --- | --- | --- |
| `three.module.min.js` | https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.min.js | MIT (`LICENSE-three.txt`) |
| `loaders/GLTFLoader.js` | https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/GLTFLoader.js | MIT (`LICENSE-three.txt`) |
| `utils/BufferGeometryUtils.js` | https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/utils/BufferGeometryUtils.js | MIT (`LICENSE-three.txt`) |
| `LICENSE-three.txt` | https://cdn.jsdelivr.net/npm/three@0.160.0/LICENSE | — |
| `RobotExpressive.glb` | https://cdn.jsdelivr.net/gh/mrdoob/three.js@r160/examples/models/gltf/RobotExpressive/RobotExpressive.glb | CC0 1.0, by Tomás Laulhé (Quaternius), modifications by Don McCurdy (per the three.js examples' model README) |
