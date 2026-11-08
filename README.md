# BrowsFaceAnimator

Upload a front-facing photo of a head. The app:

1. **Finds the face**: 478 points from MediaPipe Face Landmarker.
2. **Removes the eyebrows**: masks them and fills the area with matching skin, all in the browser.
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
  app.js           detection, brow removal, mapping, animation
  vendor/mediapipe MediaPipe Tasks Vision 0.10.14 (JS + WASM, self-hosted)
  models/          face_landmarker.task
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
