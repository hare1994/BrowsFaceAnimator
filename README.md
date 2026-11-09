# BrowsFaceAnimator

Upload a front-facing photo of a head. The app:

1. **Finds the face**: 478 points from MediaPipe Face Landmarker.
2. **Removes the eyebrows**: finds the actual brow hairs pixel by pixel (never the eyelid, eyeshadow or hairline). It then repaints them as skin with the MI-GAN AI inpainting model, followed by a clean-up pass for any leftover hairs. If the model can't load, it falls back to a classic skin fill.
3. **Animates the mapping**:
   - centre line
   - yellow verticals from the nostril edges (brow **start**)
   - brow height lines
   - white verticals at the outer iris (**arch**)
   - white diagonals from nostril through the outer eye corner (**tail**)
   - cyan tail markers
   - pink points, then the connected outline
4. **Draws the new brows**: hair strokes and/or powder, coloured from the photo.

You can then adjust thickness, arch height and position, tail length, colour and style. You can also save a PNG or a video of the whole animation.

Everything runs on the user's device. Photos are never uploaded.

## Project layout

```
public/            static site (this is what gets deployed)
  index.html
  style.css
  app.js           face detection, AI repaint, mapping, animation
  removal.js       brow-hair detection + classic fill (no DOM; runs in removal-worker.js)
  ai-worker.js     runs the MI-GAN model in its own thread (fast multi-threaded mode,
                   automatic fallback to single-threaded mode)
  vendor/mediapipe MediaPipe Tasks Vision 0.10.14 (JS + WASM, self-hosted)
  vendor/ort       onnxruntime-web 1.22.0 (WASM build)
  models/          face_landmarker.task, migan.part0/part1 (MI-GAN, split to stay under 25 MB per file)
wrangler.jsonc     Cloudflare Workers static-assets config
```

There is no build step.

## Run locally

```bash
cd public && python3 -m http.server 8080
# open http://localhost:8080
```

## Deploy to Cloudflare

**Option A: Workers (recommended)**
1. Cloudflare dashboard → Workers & Pages → Create → Import a repository → pick this repo.
2. Leave the build command empty. The deploy command is `npx wrangler deploy`.

Or from your own computer: `npx wrangler deploy`.

**Option B: Pages**
1. Workers & Pages → Create → Pages → Connect to Git → pick this repo.
2. Framework preset: *None*. Build command: *(empty)*. Build output directory: `public`.

## Credits and licences

- **MediaPipe Face Landmarker**, Google (Apache 2.0)
- **MI-GAN** inpainting model, Picsart AI Research (MIT). Uses `migan_pipeline_v2.onnx`, taken from the npm package `migan-onnx` and split in two.
- **onnxruntime-web**, Microsoft (MIT)

`public/_headers` turns on cross-origin isolation, which lets the AI model use several CPU threads (about 2x faster).
