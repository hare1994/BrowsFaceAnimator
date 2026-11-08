import { FaceLandmarker, FilesetResolver } from './vendor/mediapipe/vision_bundle.mjs';

/* ------------------------------------------------------------------ *
 * Brow Mapping Studio
 *  1. detect the face (MediaPipe Face Landmarker, 478 points)
 *  2. remove the existing brows (in-browser inpainting)
 *  3. animate the mapping guides (nose, pupils, eye corners)
 *  4. draw new brows from the mapped points
 * Everything runs locally in the browser.
 * ------------------------------------------------------------------ */

const MAX_SIDE = 1280;
const IRIS_MM = 11.7;

// MediaPipe face mesh indices. "R" = subject's right = image left.
const LM = {
  chin: 152, top: 10,
  eyeOutR: 33, eyeInR: 133, eyeOutL: 263, eyeInL: 362,
  irisR: 468, irisL: 473,
  irisRingR: [469, 470, 471, 472], irisRingL: [474, 475, 476, 477],
  alarR: 64, alarL: 294,
  browInBotR: 55, browInBotL: 285,
  browPolyR: [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
  browPolyL: [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
};

const COLORS = { yellow: '#ffe600', white: '#ffffff', cyan: '#31e6ff', pink: '#ff2bd6' };

const $ = (id) => document.getElementById(id);
const els = {
  intro: $('intro'), studio: $('studio'), drop: $('drop'), file: $('file'),
  stage: $('stage'), caption: $('caption'), status: $('status'),
  replay: $('replay'), compare: $('compare'), controls: $('controls'),
  thick: $('thick'), lift: $('lift'), arch: $('arch'), tail: $('tail'), gap: $('gap'),
  color: $('color'), style: $('style'),
  measure: $('measure'), measureList: $('measureList'),
  savePng: $('savePng'), saveVid: $('saveVid'), newPhoto: $('newPhoto'),
};
const ctx = els.stage.getContext('2d');

const state = {
  W: 0, H: 0,
  orig: null,      // canvas: original photo
  clean: null,     // canvas: brows removed
  face: null,      // face frame + raw measurements
  brows: null,     // computed brow geometry + strokes
  timeline: null,
  t: 0, playing: false, start: 0,
  showOriginal: false,
  hairColor: '#3b2a22',
};

/* ------------------------------ helpers ------------------------------ */

const lerp = (a, b, t) => a + (b - a) * t;
const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const smooth = (t) => { t = clamp(t); return t * t * (3 - 2 * t); };
const easeOut = (t) => 1 - Math.pow(1 - clamp(t), 3);
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const pt = (x, y) => ({ x, y });

function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return c;
}

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}
function rgbToHex({ r, g, b }) {
  return '#' + [r, g, b].map((v) => Math.round(clamp(v, 0, 255)).toString(16).padStart(2, '0')).join('');
}
const lum = (r, g, b) => 0.299 * r + 0.587 * g + 0.114 * b;

function cubic(p0, p1, p2, p3, n, out) {
  for (let i = 1; i <= n; i++) {
    const t = i / n, u = 1 - t;
    out.push(pt(
      u * u * u * p0.x + 3 * u * u * t * p1.x + 3 * u * t * t * p2.x + t * t * t * p3.x,
      u * u * u * p0.y + 3 * u * u * t * p1.y + 3 * u * t * t * p2.y + t * t * t * p3.y,
    ));
  }
  return out;
}

function polylineLength(pts) {
  let L = 0;
  for (let i = 1; i < pts.length; i++) L += dist(pts[i - 1], pts[i]);
  return L;
}

/* --------------------------- face detection -------------------------- */

let landmarkerPromise = null;
function getLandmarker() {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const files = await FilesetResolver.forVisionTasks(new URL('./vendor/mediapipe/wasm', import.meta.url).href);
      return FaceLandmarker.createFromOptions(files, {
        baseOptions: { modelAssetPath: new URL('./models/face_landmarker.task', import.meta.url).href, delegate: 'CPU' },
        runningMode: 'IMAGE',
        numFaces: 1,
      });
    })();
    landmarkerPromise.catch(() => { landmarkerPromise = null; });
  }
  return landmarkerPromise;
}

/* Builds a face-aligned frame: rotation removes head roll so the
 * mapping lines are truly vertical/horizontal relative to the face. */
function buildFace(lmNorm, W, H) {
  const P = lmNorm.map((p) => pt(p.x * W, p.y * H));
  const irisR = P[LM.irisR], irisL = P[LM.irisL];
  const angle = Math.atan2(irisL.y - irisR.y, irisL.x - irisR.x);
  const C = pt((irisR.x + irisL.x) / 2, (irisR.y + irisL.y) / 2);
  const cos = Math.cos(-angle), sin = Math.sin(-angle);
  const F = (p) => pt(C.x + (p.x - C.x) * cos - (p.y - C.y) * sin, C.y + (p.x - C.x) * sin + (p.y - C.y) * cos);
  const Q = (i) => F(P[i]);

  const inR = Q(LM.eyeInR), inL = Q(LM.eyeInL), outR = Q(LM.eyeOutR), outL = Q(LM.eyeOutL);
  const midX = (inR.x + inL.x) / 2;
  const avgHalf = (a, b) => (Math.abs(a.x - midX) + Math.abs(b.x - midX)) / 2;
  const avgY = (a, b) => (a.y + b.y) / 2;

  const iR = Q(LM.irisR), iL = Q(LM.irisL);
  const ringRadius = (center, ring) => Math.max(...ring.map((i) => dist(Q(i), center)));
  const irisRad = (ringRadius(iR, LM.irisRingR) + ringRadius(iL, LM.irisRingL)) / 2;
  const aR = Q(LM.alarR), aL = Q(LM.alarL);

  const face = {
    P, angle, C, F,
    midX,
    eyeY: avgY(iR, iL),
    eyeW: (dist(inR, outR) + dist(inL, outL)) / 2,
    irisHalf: avgHalf(iR, iL),
    irisRad,
    innerCornerHalf: avgHalf(inR, inL),
    innerCornerY: avgY(inR, inL),
    outerHalf: avgHalf(outR, outL),
    outerY: avgY(outR, outL),
    noseHalf: avgHalf(aR, aL),
    nostrilY: avgY(aR, aL),
    browInBotY: avgY(Q(LM.browInBotR), Q(LM.browInBotL)),
    topY: Q(LM.top).y,
    chinY: Q(LM.chin).y,
    mmPerPx: IRIS_MM / (2 * irisRad),
    asym: {
      eyeY: Math.abs(iR.y - iL.y),
      nose: Math.abs(Math.abs(aR.x - midX) - Math.abs(aL.x - midX)),
      outer: Math.abs(Math.abs(outR.x - midX) - Math.abs(outL.x - midX)),
    },
  };
  return face;
}

/* ------------------------------ brow removal ------------------------------ */

// Fills unknown pixels (unk=1) so each equals the mean of its 4 neighbours.
// Solved coarse-to-fine so large holes converge quickly.
function solveHarmonic(img, unk, w, h) {
  const anyUnknown = unk.some((v) => v);
  if (!anyUnknown) return;
  if (w > 16 && h > 16) {
    const w2 = Math.ceil(w / 2), h2 = Math.ceil(h / 2);
    const img2 = new Float32Array(w2 * h2 * 3), unk2 = new Uint8Array(w2 * h2);
    for (let y = 0; y < h2; y++) for (let x = 0; x < w2; x++) {
      let r = 0, g = 0, b = 0, n = 0, holes = 0;
      for (let oy = 0; oy < 2; oy++) for (let ox = 0; ox < 2; ox++) {
        const xx = Math.min(w - 1, 2 * x + ox), yy = Math.min(h - 1, 2 * y + oy), j = yy * w + xx;
        if (unk[j]) { holes++; continue; }
        r += img[j * 3]; g += img[j * 3 + 1]; b += img[j * 3 + 2]; n++;
      }
      const j2 = y * w2 + x;
      if (holes) unk2[j2] = 1;
      else { img2[j2 * 3] = r / n; img2[j2 * 3 + 1] = g / n; img2[j2 * 3 + 2] = b / n; }
    }
    solveHarmonic(img2, unk2, w2, h2);
    // bilinear upsample of the coarse solution as the starting guess
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const j = y * w + x;
      if (!unk[j]) continue;
      const fx = Math.min(w2 - 1, Math.max(0, (x - 0.5) / 2)), fy = Math.min(h2 - 1, Math.max(0, (y - 0.5) / 2));
      const xa = Math.floor(fx), ya = Math.floor(fy), xb = Math.min(w2 - 1, xa + 1), yb = Math.min(h2 - 1, ya + 1);
      const tx = fx - xa, ty = fy - ya;
      for (let c = 0; c < 3; c++) {
        const v = (img2[(ya * w2 + xa) * 3 + c] * (1 - tx) + img2[(ya * w2 + xb) * 3 + c] * tx) * (1 - ty)
                + (img2[(yb * w2 + xa) * 3 + c] * (1 - tx) + img2[(yb * w2 + xb) * 3 + c] * tx) * ty;
        img[j * 3 + c] = v;
      }
    }
  } else {
    let r = 0, g = 0, b = 0, n = 0;
    for (let j = 0; j < w * h; j++) if (!unk[j]) { r += img[j * 3]; g += img[j * 3 + 1]; b += img[j * 3 + 2]; n++; }
    if (!n) { r = g = b = 180; n = 1; }
    for (let j = 0; j < w * h; j++) if (unk[j]) { img[j * 3] = r / n; img[j * 3 + 1] = g / n; img[j * 3 + 2] = b / n; }
  }
  const idx = [];
  for (let j = 0; j < w * h; j++) if (unk[j]) idx.push(j);
  for (let it = 0; it < 40; it++) {
    for (const j of idx) {
      const x = j % w, y = (j / w) | 0;
      const l = x > 0 ? j - 1 : j + 1, r = x < w - 1 ? j + 1 : j - 1;
      const u = y > 0 ? j - w : j + w, dn = y < h - 1 ? j + w : j - w;
      for (let c = 0; c < 3; c++) {
        img[j * 3 + c] = 0.25 * (img[l * 3 + c] + img[r * 3 + c] + img[u * 3 + c] + img[dn * 3 + c]);
      }
    }
  }
}

function removeBrows(src, face) {
  const { W, H } = state;
  const { P, eyeW } = face;
  const mask = makeCanvas(W, H);
  const m = mask.getContext('2d');
  m.fillStyle = '#fff'; m.strokeStyle = '#fff';
  m.lineJoin = 'round'; m.lineCap = 'round';
  m.lineWidth = eyeW * 0.36;
  const brows = [LM.browPolyR, LM.browPolyL];
  for (const poly of brows) {
    m.beginPath();
    poly.forEach((i, k) => (k ? m.lineTo(P[i].x, P[i].y) : m.moveTo(P[i].x, P[i].y)));
    m.closePath(); m.fill(); m.stroke();
  }

  const sctx = src.getContext('2d', { willReadFrequently: true });
  const img = sctx.getImageData(0, 0, W, H);
  const d = img.data;
  const md = m.getImageData(0, 0, W, H).data;
  const inMask = (x, y) => x >= 0 && y >= 0 && x < W && y < H && md[(y * W + x) * 4 + 3] > 127;

  // Hair colour: darkest pixels inside the (un-dilated) brow polygons.
  const tight = makeCanvas(W, H).getContext('2d');
  tight.fillStyle = '#fff';
  for (const poly of brows) {
    tight.beginPath();
    poly.forEach((i, k) => (k ? tight.lineTo(P[i].x, P[i].y) : tight.moveTo(P[i].x, P[i].y)));
    tight.closePath(); tight.fill();
  }
  const td = tight.getImageData(0, 0, W, H).data;

  let x0 = W, y0 = H, x1 = 0, y1 = 0;
  const browPx = [];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const k = (y * W + x) * 4;
    if (md[k + 3] > 127) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    if (td[k + 3] > 127) browPx.push([d[k], d[k + 1], d[k + 2], lum(d[k], d[k + 1], d[k + 2])]);
  }

  // Robust sample of skin next to the mask (brighter half of a small patch).
  function sample(x, y, dx, dy) {
    const px = [];
    for (let s = 2; s <= 12; s++) {
      for (let o = -1; o <= 1; o++) {
        const sx = x + dx * s + (dy ? o : 0), sy = y + dy * s + (dx ? o : 0);
        if (sx < 0 || sy < 0 || sx >= W || sy >= H || inMask(sx, sy)) continue;
        const k = (sy * W + sx) * 4;
        px.push([d[k], d[k + 1], d[k + 2], lum(d[k], d[k + 1], d[k + 2])]);
      }
    }
    if (!px.length) return null;
    px.sort((a, b) => a[3] - b[3]);
    const from = Math.floor(px.length * 0.4), to = Math.max(from + 1, Math.floor(px.length * 0.85));
    let r = 0, g = 0, b = 0, n = 0;
    for (let i = from; i < to; i++) { r += px[i][0]; g += px[i][1]; b += px[i][2]; n++; }
    return [r / n, g / n, b / n];
  }

  if (x1 < x0 || y1 < y0) return { canvas: src, hair: null };
  const t0 = performance.now();

  // Work on a padded region around both brows.
  const pad = Math.round(eyeW * 0.5);
  const rx0 = Math.max(0, x0 - pad), ry0 = Math.max(0, y0 - pad);
  const rx1 = Math.min(W - 1, x1 + pad), ry1 = Math.min(H - 1, y1 + pad);
  const rw = rx1 - rx0 + 1, rh = ry1 - ry0 + 1;
  const N = rw * rh;
  const img3 = new Float32Array(N * 3);
  const unk = new Uint8Array(N);
  for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) {
    const j2 = y * rw + x, k = ((y + ry0) * W + (x + rx0)) * 4;
    img3[j2 * 3] = d[k]; img3[j2 * 3 + 1] = d[k + 1]; img3[j2 * 3 + 2] = d[k + 2];
    unk[j2] = md[k + 3] > 127 ? 1 : 0;
  }

  // 1) Smooth base: multi-scale harmonic fill (each missing pixel becomes the
  //    average of its neighbours), so light and shading flow in seamlessly
  //    from the clean skin around the brows.
  solveHarmonic(img3, unk, rw, rh);

  // 2) Skin texture: copy fine detail (pores, freckles, grain) from the
  //    forehead right above the brows onto the smooth base.
  const R = Math.max(2, Math.round(eyeW * 0.03));
  const shiftUp = (y1 - y0) + Math.round(eyeW * 0.12);
  const detail = (x, y, c) => {
    let sum = 0, n = 0;
    for (let oy = -R; oy <= R; oy++) {
      const yy = y + oy; if (yy < 0 || yy >= H) continue;
      for (let ox = -R; ox <= R; ox++) {
        const xx = x + ox; if (xx < 0 || xx >= W) continue;
        sum += d[(yy * W + xx) * 4 + c]; n++;
      }
    }
    return d[(y * W + x) * 4 + c] - sum / n;
  };
  for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) {
    const j2 = y * rw + x;
    if (!unk[j2]) continue;
    const ix = x + rx0, iy = y + ry0;
    let sy = iy - shiftUp;
    // mirror back down if the forehead patch runs off the image or into the mask
    if (sy < R || inMask(ix, sy)) sy = iy - Math.round(shiftUp * 0.6);
    if (sy < R || inMask(ix, sy)) continue;
    for (let c = 0; c < 3; c++) img3[j2 * 3 + c] += clamp(detail(ix, sy, c), -40, 40) * 0.9;
  }

  const filled = makeCanvas(W, H);
  const fctx = filled.getContext('2d');
  const out = fctx.createImageData(W, H);
  out.data.set(d);
  for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) {
    const j2 = y * rw + x;
    if (!unk[j2]) continue;
    const k = ((y + ry0) * W + (x + rx0)) * 4;
    out.data[k] = img3[j2 * 3]; out.data[k + 1] = img3[j2 * 3 + 1]; out.data[k + 2] = img3[j2 * 3 + 2];
  }
  fctx.putImageData(out, 0, 0);

  // 3) Narrow feather so there is no visible seam.
  const feather = makeCanvas(W, H);
  const fe = feather.getContext('2d');
  fe.filter = `blur(${Math.max(1, eyeW * 0.015)}px)`;
  fe.drawImage(mask, 0, 0);
  fctx.globalCompositeOperation = 'destination-in';
  fctx.drawImage(feather, 0, 0);
  fctx.globalCompositeOperation = 'source-over';

  const result = makeCanvas(W, H);
  const rctx = result.getContext('2d');
  rctx.drawImage(src, 0, 0);
  rctx.drawImage(filled, 0, 0);
  if (window.__brow) window.__brow.lastRemoveMs = Math.round(performance.now() - t0);

  // Hair colour estimate.
  let hair = null;
  if (browPx.length > 20) {
    browPx.sort((a, b) => a[3] - b[3]);
    const n = Math.max(5, Math.floor(browPx.length * 0.25));
    let rr = 0, gg = 0, bb = 0;
    for (let i = 0; i < n; i++) { rr += browPx[i][0]; gg += browPx[i][1]; bb += browPx[i][2]; }
    const c = { r: rr / n, g: gg / n, b: bb / n };
    const skin = sample(Math.round((x0 + x1) / 2), y0, 0, -1);
    const contrast = skin ? lum(...skin) - lum(c.r, c.g, c.b) : 0;
    if (contrast > 45) {
      // deepen slightly: drawn brows read lighter than real hair
      hair = rgbToHex({ r: c.r * 0.8, g: c.g * 0.78, b: c.b * 0.76 });
    } else if (skin) {
      // no real brows: a soft brown derived from skin tone
      hair = rgbToHex({ r: skin[0] * 0.12 + 22, g: skin[1] * 0.09 + 14, b: skin[2] * 0.07 + 10 });
    }
  }
  // faint or very light brows give a washed-out colour: keep it deep enough to read
  if (hair) {
    const c = hexToRgb(hair), L = lum(c.r, c.g, c.b);
    if (L > 80) { const k = 80 / L; hair = rgbToHex({ r: c.r * k, g: c.g * k, b: c.b * k }); }
  }
  return { canvas: result, hair };
}

/* ------------------------------ brow mapping ------------------------------ */

function settings() {
  return {
    thick: +els.thick.value,
    lift: +els.lift.value,
    arch: +els.arch.value,
    tail: +els.tail.value,
    gap: +els.gap.value,
    color: els.color.value,
    style: els.style.value,
  };
}

function mapBrows(face, s) {
  const { midX, eyeY, eyeW } = face;

  // Proportions below are calibrated against the reference mapping
  // (all in eye-widths, measured from the pupil line).
  //   inner bottom 0.52 · inner top 0.91 · arch top 1.05 · tail 0.47

  // Brow base, nudged towards where her own brow bone sits.
  const idealBot = eyeY - 0.53 * eyeW;
  const ownBot = face.browInBotY - 0.09 * eyeW;
  const innerBotY = Math.abs(ownBot - idealBot) < 0.2 * eyeW ? lerp(idealBot, ownBot, 0.5) : idealBot;

  const thickness = 0.33 * eyeW * s.thick;
  const innerTopY = innerBotY - thickness;
  const archTopY = innerTopY - 0.14 * eyeW * s.lift;
  const archThick = thickness * 0.66;

  const ax = face.noseHalf, ay = face.nostrilY;

  // 1) start: up from the nostril edge, pulled in towards the bridge of the
  //    nose (never wider than the inner eye corner)
  const innerHalf = Math.min(ax, face.innerCornerHalf) * 0.72 * s.gap;
  // 2) arch: line from the nostril edge through the centre of the pupil,
  //    where it crosses the arch height
  const archSlope = (face.irisHalf - ax) / (ay - eyeY);
  const archHalf = ax + archSlope * (ay - archTopY) * s.arch;
  // 3) tail: line from the nostril edge past the outer eye corner
  //    (the reference line clears the corner by ~0.21 eye-widths)
  const tailY = innerBotY + 0.05 * eyeW;
  const slope = (face.outerHalf + 0.21 * eyeW - ax) / (ay - face.outerY); // half-x gained per px climbed
  const tailHalfIdeal = ax + slope * (ay - tailY);
  const tailHalf = innerHalf + (tailHalfIdeal - innerHalf) * s.tail;

  const sides = [-1, 1].map((sg) => {
    const X = (h) => midX + sg * h;
    const IT = pt(X(innerHalf), innerTopY);
    const IB = pt(X(innerHalf), innerBotY);
    const AT = pt(X(archHalf), archTopY);
    const AB = pt(X(archHalf), archTopY + archThick);
    const T = pt(X(tailHalf), tailY);
    const span1 = archHalf - innerHalf, span2 = tailHalf - archHalf;

    // top edge: soft rounded front, then an almost straight rise to the arch
    const frontTop = pt(X(innerHalf + span1 * 0.12), innerTopY - (innerTopY - archTopY) * 0.22);
    const top = [];
    cubic(IT, pt(X(innerHalf + span1 * 0.02), IT.y - thickness * 0.08), pt(X(innerHalf + span1 * 0.07), frontTop.y + thickness * 0.02), frontTop, 8, top);
    cubic(frontTop, pt(X(innerHalf + span1 * 0.5), frontTop.y - (frontTop.y - AT.y) * 0.7), pt(X(archHalf - span1 * 0.18), AT.y), AT, 24, top);
    cubic(AT, pt(X(archHalf + span2 * 0.28), AT.y), pt(X(tailHalf - span2 * 0.3), T.y - (T.y - AT.y) * 0.5), T, 20, top);
    top.unshift(IT);

    // bottom edge: nearly straight, lifting slightly under the arch, then to the tail
    const bottom = [T];
    cubic(T, pt(X(tailHalf - span2 * 0.4), T.y - (T.y - AB.y) * 0.3), pt(X(archHalf + span2 * 0.3), AB.y), AB, 20, bottom);
    cubic(AB, pt(X(archHalf - span1 * 0.4), AB.y), pt(X(innerHalf + span1 * 0.3), IB.y - (IB.y - AB.y) * 0.1), IB, 24, bottom);

    // rounded inner end, slightly fuller at the bottom like a natural brow head
    const front = [];
    cubic(IB, pt(IB.x - sg * thickness * 0.22, IB.y - thickness * 0.1), pt(IT.x - sg * thickness * 0.2, IT.y + thickness * 0.35), IT, 10, front);

    const outline = [...top, ...bottom.slice(1), ...front];
    const path = new Path2D();
    outline.forEach((p, i) => (i ? path.lineTo(p.x, p.y) : path.moveTo(p.x, p.y)));
    path.closePath();

    return { sg, IT, IB, AT, AB, T, outline, path, outlineLen: polylineLength(outline) };
  });

  const geo = {
    innerHalf, archHalf, tailHalf, innerTopY, innerBotY, archTopY, tailY, thickness, archThick,
    slope, archSlope, sides,
  };
  geo.strokes = makeStrokes(face, geo, s);
  return geo;
}

function makeStrokes(face, geo, s) {
  const { eyeW, midX } = face;
  const probe = makeCanvas(2, 2).getContext('2d');
  const rnd = mulberry32(1234);
  const strokes = [];
  const base = hexToRgb(s.color);
  const archU = (geo.archHalf - geo.innerHalf) / (geo.tailHalf - geo.innerHalf);
  const span = geo.tailHalf - geo.innerHalf;

  // Three layers, like real brows / microblading:
  //   0 soft under-hairs (lighter, wider, low opacity)
  //   1 main hairs
  //   2 fine dark crisp hairs on top
  const layers = [
    { share: 0.3, len: [0.08, 0.15], w: 1.1, alpha: [0.22, 0.4], shade: [1.1, 1.45] },
    { share: 0.45, len: [0.07, 0.14], w: 0.95, alpha: [0.35, 0.65], shade: [0.9, 1.2] },
    { share: 0.25, len: [0.05, 0.11], w: 0.75, alpha: [0.5, 0.85], shade: [0.7, 0.95] },
  ];

  for (const side of geo.sides) {
    const xs = side.outline.map((p) => p.x), ys = side.outline.map((p) => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const total = Math.round(1700 * s.thick);
    let made = 0, tries = 0;
    while (made < total && tries < total * 50) {
      tries++;
      const x = lerp(minX, maxX, rnd()), y = lerp(minY, maxY, rnd());
      if (!probe.isPointInPath(side.path, x, y)) continue;
      const u = clamp((Math.abs(x - midX) - geo.innerHalf) / span);
      // sparse, soft head; slightly sparser tail tip
      if (u < 0.22 && rnd() > 0.1 + 0.9 * smooth(u / 0.22)) continue;
      if (u > 0.9 && rnd() > 1 - (u - 0.9) * 5) continue;
      made++;

      const r = rnd();
      const L = r < layers[0].share ? 0 : r < layers[0].share + layers[1].share ? 1 : 2;
      const ly = layers[L];

      // vertical position inside the brow at this x (0 = bottom, 1 = top)
      const topY = lerp(geo.innerTopY, geo.archTopY, smooth(u / Math.max(archU, 0.01)));
      const botY = lerp(geo.innerBotY, geo.archTopY + geo.archThick, smooth(u / Math.max(archU, 0.01)));
      const v = clamp((botY - y) / Math.max(1, botY - topY));

      let a; // degrees above horizontal, pointing towards the tail
      if (u < 0.18) a = lerp(88, 62, u / 0.18) - v * 12;           // head: hairs stand up, fanning out
      else if (u < archU) a = lerp(55, 18, smooth((u - 0.18) / Math.max(0.01, archU - 0.18)));
      else a = lerp(12, -24, smooth((u - archU) / Math.max(0.01, 1 - archU)));
      if (u >= 0.18) a += (0.5 - v) * 34;  // herringbone: lower hairs rise, upper hairs lie down
      a += (rnd() - 0.5) * 14;
      const rad = (a * Math.PI) / 180;
      const len = eyeW * lerp(ly.len[0], ly.len[1], rnd()) * (u < 0.15 ? 0.75 : 1);
      const dx = side.sg * Math.cos(rad) * len, dy = -Math.sin(rad) * len;
      // hairs curve gently towards the tail
      const bend = len * lerp(0.08, 0.22, rnd());
      const nx = side.sg * Math.sin(rad), ny = Math.cos(rad);
      const shade = lerp(ly.shade[0], ly.shade[1], rnd());
      const col = { r: clamp(base.r * shade + (L === 0 ? 18 : 0), 0, 255), g: clamp(base.g * shade + (L === 0 ? 10 : 0), 0, 255), b: clamp(base.b * shade, 0, 255) };
      strokes.push({
        sg: side.sg, u, L,
        order: u * 0.75 + L * 0.1 + rnd() * 0.05,
        x, y,
        cx: x + dx * 0.5 + nx * bend, cy: y + dy * 0.5 + ny * bend,
        ex: x + dx, ey: y + dy,
        w: Math.max(0.45, eyeW * 0.0068) * ly.w * lerp(0.75, 1.25, rnd()),
        alpha: lerp(ly.alpha[0], ly.alpha[1], rnd()) * (u < 0.12 ? 0.7 : 1),
        color: `rgb(${col.r | 0},${col.g | 0},${col.b | 0})`,
      });
    }
  }
  strokes.sort((a, b) => a.L - b.L);
  return strokes;
}

/* ------------------------------ timeline ------------------------------ */

function buildTimeline(face, geo) {
  const { midX, eyeW, topY, chinY } = face;
  const ext = eyeW * 0.35;
  const longHalf = geo.tailHalf + eyeW * 0.35;
  const L = [];
  const add = (o) => L.push({ w: 1, ...o });

  // centre line, grows up from the chin
  add({ color: COLORS.yellow, a: pt(midX, chinY), b: pt(midX, topY), t0: 1.9, d: 0.9, tip: true });
  // nostril verticals -> brow start
  for (const sg of [-1, 1]) {
    const x = midX + sg * geo.innerHalf;
    add({ color: COLORS.yellow, a: pt(x, chinY - eyeW * 0.1), b: pt(x, topY + eyeW * 0.2), t0: 2.8 + (sg > 0 ? 0.25 : 0), d: 0.9, tip: true });
  }
  // brow-top (short), brow-bottom (short), arch line (long)
  const short = face.noseHalf + eyeW * 0.55;
  add({ color: COLORS.yellow, a: pt(midX - short, geo.innerTopY), b: pt(midX + short, geo.innerTopY), t0: 4.1, d: 0.6, fromCenter: true });
  add({ color: COLORS.yellow, a: pt(midX - short, geo.innerBotY), b: pt(midX + short, geo.innerBotY), t0: 4.6, d: 0.6, fromCenter: true });
  add({ color: COLORS.yellow, a: pt(midX - longHalf, geo.archTopY), b: pt(midX + longHalf, geo.archTopY), t0: 5.2, d: 0.8, fromCenter: true });
  // white: arch verticals through the outer iris
  for (const sg of [-1, 1]) {
    const x = midX + sg * geo.archHalf;
    add({ color: COLORS.white, a: pt(x, geo.archTopY - eyeW * 0.6), b: pt(x, face.nostrilY + eyeW * 0.25), t0: 6.1 + (sg > 0 ? 0.3 : 0), d: 0.8, tip: true });
  }
  // white: diagonals nostril -> outer eye corner -> tail
  for (const sg of [-1, 1]) {
    const ax = midX + sg * face.noseHalf, ay = face.nostrilY;
    const ey = geo.archTopY - eyeW * 0.05;
    const ex = midX + sg * (face.noseHalf + geo.slope * (ay - ey));
    add({ color: COLORS.white, a: pt(ax, ay), b: pt(ex, ey), t0: 7.2 + (sg > 0 ? 0.35 : 0), d: 0.9, tip: true });
  }
  // cyan tail markers
  for (const sg of [-1, 1]) {
    const x = midX + sg * geo.tailHalf;
    add({ color: COLORS.cyan, a: pt(x, geo.tailY + ext), b: pt(x, geo.tailY - ext * 1.4), t0: 8.6 + (sg > 0 ? 0.2 : 0), d: 0.5, tip: true });
  }

  const dots = [];
  let td = 9.4;
  for (const side of geo.sides) {
    for (const k of ['IT', 'IB', 'AT', 'T']) { dots.push({ p: side[k], t0: td }); td += 0.09; }
  }

  return {
    removeStart: 0.3, removeEnd: 1.4,
    gridStart: 1.3, gridEnd: 1.9,
    lines: L, dots,
    outlineStart: 10.3, outlineEnd: 11.8,
    guidesFadeStart: 12.0, guidesFadeEnd: 12.8,
    fillStart: 12.6, fillEnd: 15.0,
    end: 15.6,
    captions: [
      [0, 'Removing the brows'],
      [1.8, 'Centre line'],
      [2.8, 'Start: inner nose line, straight up'],
      [4.1, 'Brow height'],
      [6.1, 'Arch: outer edge of the iris'],
      [7.2, 'Tail: nostril → outer eye corner'],
      [9.4, 'Mapping points'],
      [10.3, 'Connecting the shape'],
      [12.6, 'Filling in'],
      [15.0, ''],
    ],
  };
}

/* ------------------------------ rendering ------------------------------ */

function faceTransform(c) {
  const { C, angle } = state.face;
  c.translate(C.x, C.y); c.rotate(angle); c.translate(-C.x, -C.y);
}

function glowLine(c, a, b, color, lw, alpha) {
  c.save();
  c.globalAlpha = alpha;
  c.lineCap = 'round';
  c.strokeStyle = color; c.shadowColor = color;
  c.shadowBlur = lw * 5; c.lineWidth = lw * 1.6; c.globalAlpha = alpha * 0.55;
  c.beginPath(); c.moveTo(a.x, a.y); c.lineTo(b.x, b.y); c.stroke();
  c.shadowBlur = lw * 2; c.lineWidth = lw; c.globalAlpha = alpha;
  c.beginPath(); c.moveTo(a.x, a.y); c.lineTo(b.x, b.y); c.stroke();
  c.restore();
}

function dot(c, p, r, color, alpha) {
  c.save();
  c.globalAlpha = alpha;
  c.fillStyle = color; c.shadowColor = color; c.shadowBlur = r * 2.5;
  c.beginPath(); c.arc(p.x, p.y, r, 0, Math.PI * 2); c.fill();
  c.restore();
}

function drawGrid(c, alpha) {
  if (alpha <= 0) return;
  const { W, H } = state;
  const step = state.face.eyeW / 3.2;
  const diag = Math.hypot(W, H);
  const { C } = state.face;
  c.save();
  c.globalAlpha = alpha * 0.22;
  c.strokeStyle = '#000'; c.lineWidth = Math.max(0.5, W / 1400);
  c.beginPath();
  for (let x = C.x - diag; x <= C.x + diag; x += step) { c.moveTo(x, C.y - diag); c.lineTo(x, C.y + diag); }
  for (let y = C.y - diag; y <= C.y + diag; y += step) { c.moveTo(C.x - diag, y); c.lineTo(C.x + diag, y); }
  c.stroke();
  c.restore();
}

const layerCache = { hair: null, powder: null, mask: null, maskKey: '' };

function layerCanvas(key) {
  const { W, H } = state;
  let cv = layerCache[key];
  if (!cv || cv.width !== W || cv.height !== H) cv = layerCache[key] = makeCanvas(W, H);
  return cv;
}

// Soft-edged mask of both brows (image space), so hairs fade out at the
// edges instead of being cut along a hard outline.
function browMask() {
  const geo = state.brows;
  if (layerCache.mask && layerCache.maskKey === geo) return layerCache.mask;
  const cv = layerCanvas('mask');
  const m = cv.getContext('2d');
  m.setTransform(1, 0, 0, 1, 0, 0);
  m.clearRect(0, 0, cv.width, cv.height);
  m.save();
  faceTransform(m);
  m.filter = `blur(${Math.max(0.8, state.face.eyeW * 0.018)}px)`;
  m.fillStyle = '#000'; m.strokeStyle = '#000';
  m.lineWidth = state.face.eyeW * 0.03; m.lineJoin = 'round';
  for (const side of geo.sides) { m.fill(side.path); m.stroke(side.path); }
  m.restore();
  layerCache.mask = cv; layerCache.maskKey = geo;
  return cv;
}

function drawBrows(c, progress) {
  if (progress <= 0) return;
  const s = settings();
  const geo = state.brows;
  const rgb = hexToRgb(s.color);
  const mask = browMask();
  const eyeW = state.face.eyeW;

  if (s.style !== 'strokes') {
    // powder: soft ombré tint, multiplied so skin texture shows through
    const pw = layerCanvas('powder');
    const p = pw.getContext('2d');
    p.setTransform(1, 0, 0, 1, 0, 0);
    p.clearRect(0, 0, pw.width, pw.height);
    p.save();
    faceTransform(p);
    p.filter = `blur(${Math.max(1, eyeW * 0.04)}px)`;
    const k = s.style === 'powder' ? 1 : 0.5;
    for (const side of geo.sides) {
      const g = p.createLinearGradient(side.IT.x, 0, side.T.x, 0);
      g.addColorStop(0, `rgba(${rgb.r},${rgb.g},${rgb.b},0)`);
      g.addColorStop(0.18, `rgba(${rgb.r},${rgb.g},${rgb.b},${0.35 * k})`);
      g.addColorStop(0.55, `rgba(${rgb.r},${rgb.g},${rgb.b},${0.75 * k})`);
      g.addColorStop(1, `rgba(${rgb.r},${rgb.g},${rgb.b},${0.6 * k})`);
      p.fillStyle = g;
      p.fill(side.path);
    }
    p.restore();
    c.save();
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.globalAlpha = smooth(progress * 1.3);
    c.globalCompositeOperation = 'multiply';
    c.drawImage(pw, 0, 0);
    c.restore();
  }

  if (s.style !== 'powder') {
    const hv = layerCanvas('hair');
    const h = hv.getContext('2d');
    h.setTransform(1, 0, 0, 1, 0, 0);
    h.clearRect(0, 0, hv.width, hv.height);
    h.save();
    faceTransform(h);
    h.lineCap = 'round';
    const reveal = clamp(progress) * 1.05;
    for (const k of geo.strokes) {
      if (k.order > reveal) continue;
      h.strokeStyle = k.color;
      // tapered hair: thick root, fine tip (3 segments along the curve)
      let px = k.x, py = k.y;
      for (let i = 1; i <= 3; i++) {
        const t = i / 3, u = 1 - t;
        const qx = u * u * k.x + 2 * u * t * k.cx + t * t * k.ex;
        const qy = u * u * k.y + 2 * u * t * k.cy + t * t * k.ey;
        h.globalAlpha = k.alpha * (1 - (i - 1) * 0.18);
        h.lineWidth = k.w * (1.15 - (i - 1) * 0.38);
        h.beginPath(); h.moveTo(px, py); h.lineTo(qx, qy); h.stroke();
        px = qx; py = qy;
      }
    }
    h.restore();
    // fade hairs at the brow edges
    h.globalCompositeOperation = 'destination-in';
    h.drawImage(mask, 0, 0);
    h.globalCompositeOperation = 'source-over';

    c.save();
    c.setTransform(1, 0, 0, 1, 0, 0);
    // a hint of softness so hairs sit in the photo instead of on top of it
    c.filter = `blur(${Math.max(0.25, eyeW * 0.0025)}px)`;
    c.drawImage(hv, 0, 0);
    c.restore();
  }
}

function drawOutline(c, side, frac, color, lw, alpha) {
  if (frac <= 0 || alpha <= 0) return;
  const pts = side.outline;
  const target = side.outlineLen * clamp(frac);
  c.save();
  c.globalAlpha = alpha;
  c.strokeStyle = color; c.lineWidth = lw; c.lineJoin = 'round'; c.lineCap = 'round';
  c.shadowColor = 'rgba(0,0,0,0.35)'; c.shadowBlur = lw;
  c.beginPath(); c.moveTo(pts[0].x, pts[0].y);
  let L = 0;
  for (let i = 1; i < pts.length; i++) {
    const seg = dist(pts[i - 1], pts[i]);
    if (L + seg >= target) {
      const t = (target - L) / seg;
      c.lineTo(lerp(pts[i - 1].x, pts[i].x, t), lerp(pts[i - 1].y, pts[i].y, t));
      break;
    }
    c.lineTo(pts[i].x, pts[i].y); L += seg;
  }
  if (frac >= 1) c.closePath();
  c.stroke();
  c.restore();
}

function render(t) {
  const { W, H, face } = state;
  const tl = state.timeline;
  const c = ctx;
  c.setTransform(1, 0, 0, 1, 0, 0);
  c.clearRect(0, 0, W, H);

  if (state.showOriginal) {
    c.drawImage(state.orig, 0, 0);
    return;
  }

  // 1. brow removal: a scan sweeps down, revealing the cleaned photo
  c.drawImage(state.orig, 0, 0);
  const rp = smooth((t - tl.removeStart) / (tl.removeEnd - tl.removeStart));
  if (rp > 0) {
    c.save();
    c.globalAlpha = rp;
    c.drawImage(state.clean, 0, 0);
    c.restore();
  }

  const lw = Math.max(2, face.eyeW * 0.035);
  const guidesAlpha = 1 - smooth((t - tl.guidesFadeStart) / (tl.guidesFadeEnd - tl.guidesFadeStart));

  c.save();
  faceTransform(c);

  drawGrid(c, smooth((t - tl.gridStart) / (tl.gridEnd - tl.gridStart)) * guidesAlpha);
  c.restore();

  // final brows (under the guides while they fade)
  drawBrows(c, (t - tl.fillStart) / (tl.fillEnd - tl.fillStart));

  c.save();
  faceTransform(c);

  if (guidesAlpha > 0) {
    for (const ln of tl.lines) {
      const p = easeOut((t - ln.t0) / ln.d);
      if (p <= 0) continue;
      let a, b;
      if (ln.fromCenter) {
        const m = pt((ln.a.x + ln.b.x) / 2, (ln.a.y + ln.b.y) / 2);
        a = pt(lerp(m.x, ln.a.x, p), lerp(m.y, ln.a.y, p));
        b = pt(lerp(m.x, ln.b.x, p), lerp(m.y, ln.b.y, p));
      } else {
        a = ln.a;
        b = pt(lerp(ln.a.x, ln.b.x, p), lerp(ln.a.y, ln.b.y, p));
      }
      glowLine(c, a, b, ln.color, lw * (ln.color === COLORS.cyan ? 1.1 : 0.9), guidesAlpha);
      if (ln.tip && p < 1) dot(c, b, lw * 1.4, ln.color === COLORS.yellow ? '#fff7a8' : ln.color, guidesAlpha);
    }
    for (const dt of tl.dots) {
      const p = (t - dt.t0) / 0.25;
      if (p <= 0) continue;
      const pop = p < 1 ? 1 + Math.sin(p * Math.PI) * 0.6 : 1;
      dot(c, dt.p, lw * 1.25 * pop, COLORS.pink, guidesAlpha);
    }
  }

  // outline: draws on, stays a moment after guides fade, then sinks under the fill
  const op = (t - tl.outlineStart) / (tl.outlineEnd - tl.outlineStart);
  const outlineAlpha = 1 - smooth((t - tl.fillStart - 0.6) / 1.2);
  for (const side of state.brows.sides) {
    drawOutline(c, side, op, 'rgba(40,28,24,0.85)', Math.max(1, lw * 0.35), outlineAlpha);
  }
  c.restore();

  // caption
  let cap = '';
  for (const [ct, text] of tl.captions) if (t >= ct) cap = text;
  if (els.caption.textContent !== cap) els.caption.textContent = cap;
}

/* ------------------------------ playback ------------------------------ */

function play(onDone) {
  state.playing = true;
  state.start = performance.now();
  els.replay.disabled = true;
  const tick = (now) => {
    if (!state.playing) return;
    state.t = (now - state.start) / 1000;
    render(state.t);
    if (state.t >= state.timeline.end) {
      state.playing = false;
      state.t = state.timeline.end;
      render(state.t);
      finished();
      onDone && onDone();
      return;
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function finished() {
  els.replay.disabled = false;
  els.compare.disabled = false;
  els.controls.disabled = false;
  els.savePng.disabled = false;
  els.saveVid.disabled = !('MediaRecorder' in window) || !els.stage.captureStream;
  showMeasurements();
  document.documentElement.dataset.browState = 'done';
}

function rebuild() {
  if (!state.face) return;
  state.brows = mapBrows(state.face, settings());
  state.timeline = buildTimeline(state.face, state.brows);
  if (!state.playing) render(state.timeline.end);
  showMeasurements();
}

function showMeasurements() {
  const f = state.face, g = state.brows;
  if (!f || !g) return;
  const mm = (px) => `${(px * f.mmPerPx).toFixed(1)} mm`;
  const rows = [
    ['Brow start from centre', mm(g.innerHalf)],
    ['Arch from centre', mm(g.archHalf)],
    ['Tail from centre', mm(g.tailHalf)],
    ['Brow length', mm(g.tailHalf - g.innerHalf)],
    ['Thickness at start', mm(g.thickness)],
    ['Arch lift', mm(g.innerTopY - g.archTopY)],
    ['Gap above pupil', mm(f.eyeY - g.archTopY)],
    ['Face symmetry', symmetryScore(f)],
  ];
  els.measureList.innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
  els.measure.hidden = false;
}

function symmetryScore(f) {
  const e = (f.asym.eyeY + f.asym.nose + f.asym.outer) / (3 * f.eyeW);
  return `${Math.round(clamp(1 - e * 6) * 100)}%`;
}

/* ------------------------------ loading ------------------------------ */

function setStatus(msg, error = false) {
  els.status.textContent = msg;
  els.status.classList.toggle('error', error);
}

async function loadFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  state.playing = false;
  document.documentElement.dataset.browState = 'loading';
  els.intro.hidden = true;
  els.studio.hidden = false;
  els.measure.hidden = true;
  [els.replay, els.compare, els.savePng, els.saveVid].forEach((b) => (b.disabled = true));
  els.controls.disabled = true;
  els.caption.textContent = '';

  let bmp;
  try {
    bmp = await createImageBitmap(file, { imageOrientation: 'from-image' });
  } catch {
    bmp = await new Promise((res, rej) => {
      const img = new Image();
      img.onload = () => res(img); img.onerror = rej;
      img.src = URL.createObjectURL(file);
    });
  }
  const scale = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
  const W = Math.round(bmp.width * scale), H = Math.round(bmp.height * scale);
  state.W = W; state.H = H;
  els.stage.width = W; els.stage.height = H;
  const orig = makeCanvas(W, H);
  orig.getContext('2d', { willReadFrequently: true }).drawImage(bmp, 0, 0, W, H);
  state.orig = orig;
  ctx.drawImage(orig, 0, 0);

  try {
    setStatus('Loading face model…');
    const lmk = await getLandmarker();
    setStatus('Finding her features…');
    await new Promise((r) => setTimeout(r, 30));
    const res = lmk.detect(orig);
    if (!res.faceLandmarks || !res.faceLandmarks.length) {
      setStatus('No face found. Try a well-lit, front-facing photo.', true);
      document.documentElement.dataset.browState = 'error';
      return;
    }
    state.face = buildFace(res.faceLandmarks[0], W, H);
    setStatus('Removing brows…');
    await new Promise((r) => setTimeout(r, 30));
    const { canvas, hair } = removeBrows(orig, state.face);
    state.clean = canvas;
    if (hair) els.color.value = hair;
    syncOutputs();
    state.brows = mapBrows(state.face, settings());
    state.timeline = buildTimeline(state.face, state.brows);
    setStatus('');
    play();
  } catch (err) {
    console.error(err);
    setStatus('Something went wrong: ' + (err && err.message ? err.message : err), true);
    document.documentElement.dataset.browState = 'error';
  }
}

/* ------------------------------ export ------------------------------ */

function savePng() {
  render(state.timeline.end);
  els.stage.toBlob((blob) => download(blob, 'brow-mapping.png'), 'image/png');
}

function download(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
}

function saveVideo() {
  const types = ['video/mp4;codecs=avc1', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm'];
  const type = types.find((t) => MediaRecorder.isTypeSupported(t));
  if (!type) { setStatus('Video recording is not supported in this browser.', true); return; }
  const stream = els.stage.captureStream(30);
  const rec = new MediaRecorder(stream, { mimeType: type, videoBitsPerSecond: 6_000_000 });
  const chunks = [];
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  rec.onstop = () => {
    setStatus('');
    els.saveVid.disabled = false;
    download(new Blob(chunks, { type }), `brow-mapping.${type.includes('mp4') ? 'mp4' : 'webm'}`);
  };
  els.saveVid.disabled = true;
  rec.start();
  play(() => setTimeout(() => rec.stop(), 1200));
}

/* ------------------------------ UI wiring ------------------------------ */

function syncOutputs() {
  $('thickV').textContent = `${Math.round(+els.thick.value * 100)}%`;
  $('liftV').textContent = `${Math.round(+els.lift.value * 100)}%`;
  $('archV').textContent = `${Math.round(+els.arch.value * 100)}%`;
  $('tailV').textContent = `${Math.round(+els.tail.value * 100)}%`;
  $('gapV').textContent = `${Math.round(+els.gap.value * 100)}%`;
}

els.file.addEventListener('change', () => loadFile(els.file.files[0]));
['dragenter', 'dragover'].forEach((ev) => els.drop.addEventListener(ev, (e) => { e.preventDefault(); els.drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((ev) => els.drop.addEventListener(ev, (e) => { e.preventDefault(); els.drop.classList.remove('over'); }));
els.drop.addEventListener('drop', (e) => loadFile(e.dataTransfer.files[0]));

els.replay.addEventListener('click', () => play());
const holdOn = (e) => { e.preventDefault(); state.showOriginal = true; render(state.t); };
const holdOff = () => { if (!state.showOriginal) return; state.showOriginal = false; render(state.t); };
els.compare.addEventListener('pointerdown', holdOn);
['pointerup', 'pointerleave', 'pointercancel'].forEach((ev) => els.compare.addEventListener(ev, holdOff));

[els.thick, els.lift, els.arch, els.tail, els.gap, els.color, els.style].forEach((el) =>
  el.addEventListener('input', () => { syncOutputs(); rebuild(); }));

els.savePng.addEventListener('click', savePng);
els.saveVid.addEventListener('click', saveVideo);
els.newPhoto.addEventListener('click', () => { els.file.value = ''; els.file.click(); });

syncOutputs();
getLandmarker().catch(() => {}); // warm up while the user picks a photo

// test hook: render any moment of the animation
window.__brow = { seek: (t) => { state.playing = false; render(t); }, state };
