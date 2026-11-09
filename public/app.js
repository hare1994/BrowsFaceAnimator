import { FaceLandmarker, FilesetResolver } from './vendor/mediapipe/vision_bundle.mjs';
import * as ort from './vendor/ort/ort.wasm.min.mjs';

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

/* ------------------------------ brow removal ------------------------------
 * 1. Find the actual brow hairs: start from the landmark brow shape and grow
 *    into every connected pixel that is clearly darker than the skin around
 *    it. The search never leaves the face, never goes down to the eyelid and
 *    never strays far from the brow, so nothing else on the face is touched.
 * 2. Fill: a smooth skin base (harmonic fill) whose tone comes from the brow
 *    bone, plus real skin texture (pores, grain) copied block by block from
 *    clean skin nearby.
 * ------------------------------------------------------------------------- */

const FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
  152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
const LID_R = [33, 246, 161, 160, 159, 158, 157, 173, 133];
const LID_L = [263, 466, 388, 387, 386, 385, 384, 398, 362];
const BROW_LOW_R = [46, 53, 52, 65, 55], BROW_LOW_L = [276, 283, 282, 295, 285];

// --- small image helpers (all on a w*h grid) ---
function dilateMask(m, w, h, r) {
  if (r <= 0) return m.slice();
  const tmp = new Uint8Array(w * h), out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    let cnt = 0; const row = y * w;
    for (let x = 0; x <= Math.min(r, w - 1); x++) cnt += m[row + x];
    for (let x = 0; x < w; x++) {
      tmp[row + x] = cnt > 0 ? 1 : 0;
      const add = x + r + 1, rem = x - r;
      if (add < w) cnt += m[row + add];
      if (rem >= 0) cnt -= m[row + rem];
    }
  }
  for (let x = 0; x < w; x++) {
    let cnt = 0;
    for (let y = 0; y <= Math.min(r, h - 1); y++) cnt += tmp[y * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = cnt > 0 ? 1 : 0;
      const add = y + r + 1, rem = y - r;
      if (add < h) cnt += tmp[add * w + x];
      if (rem >= 0) cnt -= tmp[rem * w + x];
    }
  }
  return out;
}
function erodeMask(m, w, h, r) {
  const inv = new Uint8Array(w * h);
  for (let j = 0; j < w * h; j++) inv[j] = m[j] ? 0 : 1;
  const d = dilateMask(inv, w, h, r);
  for (let j = 0; j < w * h; j++) d[j] = d[j] ? 0 : 1;
  return d;
}
// separable box blur of a multi-channel float image
function boxBlur(src, w, h, ch, r) {
  const tmp = new Float32Array(src.length), out = new Float32Array(src.length);
  for (let y = 0; y < h; y++) for (let c = 0; c < ch; c++) {
    let s = 0, n = 0;
    for (let x = 0; x <= Math.min(r, w - 1); x++) { s += src[(y * w + x) * ch + c]; n++; }
    for (let x = 0; x < w; x++) {
      tmp[(y * w + x) * ch + c] = s / n;
      const add = x + r + 1, rem = x - r;
      if (add < w) { s += src[(y * w + add) * ch + c]; n++; }
      if (rem >= 0) { s -= src[(y * w + rem) * ch + c]; n--; }
    }
  }
  for (let x = 0; x < w; x++) for (let c = 0; c < ch; c++) {
    let s = 0, n = 0;
    for (let y = 0; y <= Math.min(r, h - 1); y++) { s += tmp[(y * w + x) * ch + c]; n++; }
    for (let y = 0; y < h; y++) {
      out[(y * w + x) * ch + c] = s / n;
      const add = y + r + 1, rem = y - r;
      if (add < h) { s += tmp[(add * w + x) * ch + c]; n++; }
      if (rem >= 0) { s -= tmp[(rem * w + x) * ch + c]; n--; }
    }
  }
  return out;
}
// y of a polyline (landmark points) at x, flat beyond its ends
function lineY(pts, x) {
  const s = [...pts].sort((a, b) => a.x - b.x);
  if (x <= s[0].x) return s[0].y;
  if (x >= s[s.length - 1].x) return s[s.length - 1].y;
  for (let i = 1; i < s.length; i++) if (x <= s[i].x) {
    const t = (x - s[i - 1].x) / Math.max(1e-6, s[i].x - s[i - 1].x);
    return lerp(s[i - 1].y, s[i].y, t);
  }
  return s[0].y;
}

function removeBrows(src, face) {
  const { W, H } = state;
  const { P, eyeW } = face;
  const t0 = performance.now();
  const sctx = src.getContext('2d', { willReadFrequently: true });
  const d = sctx.getImageData(0, 0, W, H).data;

  // ---- working box: brows + a band of forehead above (texture source) ----
  const browIdx = [...LM.browPolyR, ...LM.browPolyL];
  const bxs = browIdx.map((i) => P[i].x), bys = browIdx.map((i) => P[i].y);
  const lidYs = [...LID_R, ...LID_L].map((i) => P[i].y);
  const bx0 = Math.max(0, Math.floor(Math.min(...bxs) - eyeW * 0.6));
  const bx1 = Math.min(W - 1, Math.ceil(Math.max(...bxs) + eyeW * 0.6));
  const by0 = Math.max(0, Math.floor(Math.min(...bys) - eyeW * 1.1));
  const by1 = Math.min(H - 1, Math.ceil(Math.max(...lidYs) + eyeW * 0.1));
  const bw = bx1 - bx0 + 1, bh = by1 - by0 + 1, N = bw * bh;

  const draw = (fn) => {
    const c = makeCanvas(bw, bh), g = c.getContext('2d');
    g.translate(-bx0, -by0);
    g.fillStyle = '#fff'; g.strokeStyle = '#fff'; g.lineJoin = 'round'; g.lineCap = 'round';
    fn(g);
    const a = g.getImageData(0, 0, bw, bh).data, m = new Uint8Array(N);
    for (let j = 0; j < N; j++) m[j] = a[j * 4 + 3] > 127 ? 1 : 0;
    return m;
  };
  const path = (g, idx, dy = 0) => {
    g.beginPath();
    idx.forEach((i, k) => (k ? g.lineTo(P[i].x, P[i].y + dy) : g.moveTo(P[i].x, P[i].y + dy)));
    g.closePath();
  };

  // face, shrunk a little so hairline / temples are never touched
  const oval = draw((g) => {
    path(g, FACE_OVAL); g.fill();
    g.globalCompositeOperation = 'destination-out'; g.lineWidth = eyeW * 0.1; path(g, FACE_OVAL); g.stroke();
  });
  // where brow hair may be searched for
  const search = draw((g) => {
    g.lineWidth = eyeW * 0.62;
    for (const p of [LM.browPolyR, LM.browPolyL]) { path(g, p); g.fill(); g.stroke(); }
  });
  // certainly brow
  const coreShape = draw((g) => {
    g.lineWidth = eyeW * 0.07;
    for (const p of [LM.browPolyR, LM.browPolyL]) { path(g, p); g.fill(); g.stroke(); }
  });
  // texture-source exclusion (generous brow area)
  const nearBrow = draw((g) => {
    g.lineWidth = eyeW * 0.75;
    for (const p of [LM.browPolyR, LM.browPolyL]) { path(g, p); g.fill(); g.stroke(); }
  });

  // floor: never below the brow's lower edge + margin, never near the eyelid
  const lowR = BROW_LOW_R.map((i) => P[i]), lowL = BROW_LOW_L.map((i) => P[i]);
  const lidR = LID_R.map((i) => P[i]), lidL = LID_L.map((i) => P[i]);
  const cR = P[LM.irisR].x, cL = P[LM.irisL].x;
  const floorAt = new Float32Array(bw), lidAt = new Float32Array(bw);
  for (let x = 0; x < bw; x++) {
    const ix = x + bx0, right = Math.abs(ix - cR) < Math.abs(ix - cL);
    const fb = lineY(right ? lowR : lowL, ix) + eyeW * 0.3;
    const fl = lineY(right ? lidR : lidL, ix) - eyeW * 0.1;
    floorAt[x] = Math.min(fb, fl) - by0;
    lidAt[x] = lineY(right ? lidR : lidL, ix) - by0;
  }

  const rgb = new Float32Array(N * 3), L = new Float32Array(N);
  for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
    const j = y * bw + x, k = ((y + by0) * W + (x + bx0)) * 4;
    rgb[j * 3] = d[k]; rgb[j * 3 + 1] = d[k + 1]; rgb[j * 3 + 2] = d[k + 2];
    L[j] = lum(d[k], d[k + 1], d[k + 2]);
  }
  const allowed = new Uint8Array(N);
  for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
    const j = y * bw + x;
    allowed[j] = search[j] && oval[j] && y <= floorAt[x] ? 1 : 0;
  }

  // ---- 1. find the brow hair ----
  let mask = new Uint8Array(N);

  const ringR = Math.max(3, Math.round(eyeW * 0.07));
  const skinLum = (m) => {
    const f = new Float32Array(rgb);
    const hide = dilateMask(m, bw, bh, ringR);
    solveHarmonic(f, hide, bw, bh);
    const out = new Float32Array(N);
    for (let j = 0; j < N; j++) out[j] = lum(f[j * 3], f[j * 3 + 1], f[j * 3 + 2]);
    return out;
  };
  // fine texture (hair is grainy, shadows are smooth)
  const hp = new Float32Array(N);
  { const b1 = boxBlur(L, bw, bh, 1, 1); for (let k = 0; k < N; k++) hp[k] = Math.abs(L[k] - b1[k]); }
  const Tex = boxBlur(hp, bw, bh, 1, 2);
  // skin tone just above the search area, per column (forehead reference)
  const skinAbove = new Float32Array(bw);
  const lowLine = new Float32Array(bw);
  {
    const has = new Uint8Array(bw);
    for (let x = 0; x < bw; x++) {
      const ix = x + bx0, right = Math.abs(ix - cR) < Math.abs(ix - cL);
      lowLine[x] = lineY(right ? lowR : lowL, ix) - by0;
      let t = -1;
      for (let y = 0; y < bh; y++) if (search[y * bw + x]) { t = y; break; }
      if (t < 0) continue;
      const vals = [];
      for (let y = Math.max(0, t - 10); y < t - 1; y++) vals.push(L[y * bw + x]);
      if (!vals.length) continue;
      vals.sort((a, b) => a - b);
      skinAbove[x] = vals[Math.floor(vals.length * 0.6)]; has[x] = 1;
    }
    let last = -1;
    for (let x = 0; x < bw; x++) {
      if (has[x]) { if (last < 0) for (let k = 0; k < x; k++) skinAbove[k] = skinAbove[x]; last = x; }
      else if (last >= 0) skinAbove[x] = skinAbove[last];
    }
    skinAbove.set(boxBlur(skinAbove, bw, 1, 1, Math.max(3, Math.round(eyeW * 0.1))));
  }
  // typical skin graininess (forehead) -> what counts as "hair texture"
  let texThr;
  {
    const v = [];
    for (let y = 0; y < bh; y += 2) for (let x = 0; x < bw; x += 2) {
      const k = y * bw + x;
      if (oval[k] && !search[k] && y < lowLine[x]) v.push(Tex[k]);
    }
    v.sort((a, b) => a - b);
    texThr = (v.length ? v[Math.floor(v.length * 0.6)] : 3) * 1.6 + 1;
  }
  // hair test for single pixels: very dark, or darker *and* grainy
  const isHair = (k, ref) => {
    const r = L[k] / ref;
    return r < 0.36 || (r < 0.93 && ref - L[k] > 7 && Tex[k] > texThr);
  };

  // ---- the brow band, column by column ----
  // Smooth the darkness a little, find the darkest point near the landmark
  // brow line, and take the connected stretch that stays clearly dark.
  // Eyeshadow, lid skin and the crease are lighter than the hair, so the
  // band stops where the brightness comes back.
  const upR = [70, 63, 105, 66, 107].map((i) => P[i]), upL = [300, 293, 334, 296, 336].map((i) => P[i]);
  const Ls = boxBlur(boxBlur(L, bw, bh, 1, Math.max(1, Math.round(eyeW * 0.012))), bw, bh, 1, 0);
  const Lh = (() => { // extra horizontal smoothing so sparse hairs read as one band
    const r = Math.max(2, Math.round(eyeW * 0.04)), out = new Float32Array(N);
    for (let y = 0; y < bh; y++) {
      let sum = 0, n = 0;
      for (let x = 0; x <= Math.min(r, bw - 1); x++) { sum += Ls[y * bw + x]; n++; }
      for (let x = 0; x < bw; x++) {
        out[y * bw + x] = sum / n;
        if (x + r + 1 < bw) { sum += Ls[y * bw + x + r + 1]; n++; }
        if (x - r >= 0) { sum -= Ls[y * bw + x - r]; n--; }
      }
    }
    return out;
  })();
  const bandM = new Uint8Array(N);
  const bandTop = new Int32Array(bw).fill(-1), bandBot = new Int32Array(bw).fill(-1);
  for (let x = 0; x < bw; x++) {
    const ix = x + bx0, right = Math.abs(ix - cR) < Math.abs(ix - cL);
    const uy = lineY(right ? upR : upL, ix) - by0, ly = lowLine[x];
    const ref = skinAbove[x];
    if (!ref) continue;
    const ya = Math.max(0, Math.floor(uy - eyeW * 0.15)), yb = Math.min(bh - 1, Math.ceil(ly + eyeW * 0.06));
    let best = -1, bestD = 0;
    for (let y = ya; y <= yb; y++) {
      const k = y * bw + x;
      if (!allowed[k]) continue;
      const D = (ref - Lh[k]) / ref;
      if (D > bestD) { bestD = D; best = y; }
    }
    if (best < 0 || bestD < 0.1) continue;
    const t = bestD * 0.55;
    const inBand = (y) => { const k = y * bw + x; return allowed[k] && (ref - Ls[k]) / ref > t; };
    let top = best, bot = best;
    const gap = Math.max(1, Math.round(eyeW * 0.015));
    for (let y = best - 1, miss = 0; y >= 0 && miss <= gap; y--) { if (inBand(y)) { top = y; miss = 0; } else miss++; }
    for (let y = best + 1, miss = 0; y < bh && miss <= gap; y++) { if (inBand(y)) { bot = y; miss = 0; } else miss++; }
    // a brow is never much thicker than the landmark brow (crease / shadow below)
    const maxT = Math.max((ly - uy) * 1.3, eyeW * 0.15);
    bandTop[x] = top; bandBot[x] = Math.min(bot, Math.round(ly + eyeW * 0.08), Math.round(top + maxT));
  }
  // remove column-to-column jumps (median over neighbouring columns)
  {
    const r = Math.max(2, Math.round(eyeW * 0.06));
    const med = (arr, x) => {
      const v = [];
      for (let k = x - r; k <= x + r; k++) if (k >= 0 && k < bw && arr[k] >= 0) v.push(arr[k]);
      if (v.length < r) return -1;
      v.sort((a, b) => a - b); return v[v.length >> 1];
    };
    const t2 = new Int32Array(bw), b2 = new Int32Array(bw);
    for (let x = 0; x < bw; x++) {
      if (bandTop[x] < 0) { t2[x] = b2[x] = -1; continue; }
      const mt = med(bandTop, x), mb = med(bandBot, x);
      const tol = eyeW * 0.04;
      t2[x] = mt < 0 ? bandTop[x] : Math.round(Math.max(bandTop[x], mt - tol));
      b2[x] = mb < 0 ? bandBot[x] : Math.round(Math.min(bandBot[x], mb + tol));
      // a column whose band sits far below its neighbours is not brow
      if (mt >= 0 && bandTop[x] > mb) { t2[x] = b2[x] = -1; }
    }
    for (let x = 0; x < bw; x++) {
      bandTop[x] = t2[x]; bandBot[x] = b2[x];
      if (t2[x] >= 0) for (let y = t2[x]; y <= b2[x]; y++) bandM[y * bw + x] = 1;
    }
  }
  // the lowest hairs: keep going down while it is still hair-dark (not crease-dark)
  for (let x = 0; x < bw; x++) {
    if (bandBot[x] < 0) continue;
    const ref = skinAbove[x], lim = Math.min(bh - 1, Math.round(bandBot[x] + eyeW * 0.06), Math.floor(floorAt[x]));
    let y = bandBot[x] + 1;
    while (y <= lim && Ls[y * bw + x] / ref < 0.45) { bandM[y * bw + x] = 1; bandBot[x] = y; y++; }
  }
  // brows have no holes: smooth the band outline across columns
  {
    const r = Math.max(1, Math.round(eyeW * 0.02));
    let b2 = erodeMask(dilateMask(bandM, bw, bh, r), bw, bh, r);
    b2 = dilateMask(erodeMask(b2, bw, bh, 1), bw, bh, 1); // drop single-pixel specks
    for (let k = 0; k < N; k++) mask[k] = b2[k] && allowed[k] ? 1 : 0;
  }
  // brow heads / ends where no band was found: landmark shape + hair test
  for (let k = 0; k < N; k++) {
    const x = k % bw;
    if (!mask[k] && bandTop[x] < 0 && coreShape[k] && allowed[k] && isHair(k, skinAbove[x])) mask[k] = 1;
  }
  // stray hairs just outside the band (connected growth, limited distance)
  {
    const near = dilateMask(mask, bw, bh, Math.max(2, Math.round(eyeW * 0.05)));
    const skin = skinLum(mask);
    const q = [];
    for (let k = 0; k < N; k++) if (mask[k]) q.push(k);
    for (let qi = 0; qi < q.length; qi++) {
      const k = q[qi], x = k % bw, y = (k / bw) | 0;
      for (const nb of [x > 0 ? k - 1 : -1, x < bw - 1 ? k + 1 : -1, y > 0 ? k - bw : -1, y < bh - 1 ? k + bw : -1]) {
        if (nb < 0 || mask[nb] || !near[nb] || !allowed[nb]) continue;
        const nx = nb % bw, ny = (nb / bw) | 0;
        // never extend downwards past the band (that is lid / eyeshadow)
        if (bandBot[nx] >= 0 && ny > bandBot[nx]) continue;
        if (isHair(nb, Math.max(skin[nb], skinAbove[nx] * 0.97))) { mask[nb] = 1; q.push(nb); }
      }
    }
  }
  // drop small isolated specks (keep the two brows)
  {
    const lab = new Int32Array(N).fill(-1), sizes = [];
    for (let k = 0; k < N; k++) {
      if (!mask[k] || lab[k] >= 0) continue;
      const id = sizes.length, st = [k]; lab[k] = id; let n = 0;
      while (st.length) {
        const c = st.pop(); n++;
        const x = c % bw, y = (c / bw) | 0;
        for (let oy = -1; oy <= 1; oy++) for (let ox = -1; ox <= 1; ox++) {
          const xx = x + ox, yy = y + oy;
          if (xx < 0 || yy < 0 || xx >= bw || yy >= bh) continue;
          const nb = yy * bw + xx;
          if (mask[nb] && lab[nb] < 0) { lab[nb] = id; st.push(nb); }
        }
      }
      sizes.push(n);
    }
    const big = Math.max(0, ...sizes);
    for (let k = 0; k < N; k++) if (mask[k] && sizes[lab[k]] < big * 0.12) mask[k] = 0;
  }
  if (window.__brow) window.__brow.dbgBand = { bandTop, bandBot };
  // tidy: close small gaps, then cover the soft (anti-aliased) hair edges
  {
    const rc = Math.max(1, Math.round(eyeW * 0.025));
    mask = erodeMask(dilateMask(mask, bw, bh, rc), bw, bh, rc);
    mask = dilateMask(mask, bw, bh, Math.max(1, Math.round(eyeW * 0.02)));
    // fine hairs sit along the top edge and the ends: extend up (forehead side)
    // and sideways, never down towards the eye
    const up = Math.max(2, Math.round(eyeW * 0.06)), side = Math.max(1, Math.round(eyeW * 0.03));
    const ext = new Uint8Array(N);
    for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
      if (!mask[y * bw + x]) continue;
      for (let dy = 0; dy <= up; dy++) { const yy = y - dy; if (yy < 0) break; ext[yy * bw + x] = 1; }
    }
    const extS = dilateMask(ext, bw, bh, side);
    for (let k = 0; k < N; k++) if (extS[k] && (k / bw | 0) <= (k % bw >= 0 ? floorAt[k % bw] : 0)) mask[k] = mask[k] || (oval[k] && search[k] ? 1 : 0);
    for (let j = 0; j < N; j++) if (!(allowed[j] || coreShape[j] && oval[j])) mask[j] = 0;
  }

  // ---- mask for the AI inpainter: generous, covers every hair + the brow shape ----
  let mlMask;
  {
    const u = new Uint8Array(N);
    for (let k = 0; k < N; k++) u[k] = mask[k] || coreShape[k] ? 1 : 0;
    mlMask = dilateMask(u, bw, bh, Math.max(2, Math.round(eyeW * 0.035)));
    for (let k = 0; k < N; k++) {
      const x = k % bw, y = (k / bw) | 0;
      if (!oval[k] || y > lidAt[x] - eyeW * 0.07) mlMask[k] = 0;
    }
  }
  if (window.__brow) {
    const mc = makeCanvas(W, H), mg = mc.getContext('2d'), md = mg.createImageData(bw, bh);
    for (let k = 0; k < N; k++) if (mlMask[k]) { md.data[k * 4] = md.data[k * 4 + 1] = md.data[k * 4 + 2] = 255; md.data[k * 4 + 3] = 255; }
    mg.putImageData(md, bx0, by0);
    window.__brow.lastMlMask = mc;
  }

  // ---- 2a. smooth skin base ----
  // H1: all surrounding skin. H2: pixels *below* the brow replaced with the
  // brow-bone tone so lid shadow / eyeshadow does not bleed upwards.
  const H1 = new Float32Array(rgb), H2 = new Float32Array(rgb);
  const topY = new Int32Array(bw).fill(-1), botY = new Int32Array(bw).fill(-1);
  for (let x = 0; x < bw; x++) for (let y = 0; y < bh; y++) if (mask[y * bw + x]) { if (topY[x] < 0) topY[x] = y; botY[x] = y; }
  {
    const ref = new Float32Array(bw * 3), has = new Uint8Array(bw);
    for (let x = 0; x < bw; x++) {
      if (topY[x] < 0) continue;
      let n = 0;
      for (let y = Math.max(0, topY[x] - 8); y < topY[x] - 1; y++) {
        const j = y * bw + x; if (mask[j]) continue;
        for (let c = 0; c < 3; c++) ref[x * 3 + c] += rgb[j * 3 + c];
        n++;
      }
      if (n) { for (let c = 0; c < 3; c++) ref[x * 3 + c] /= n; has[x] = 1; }
    }
    let last = -1;
    for (let x = 0; x < bw; x++) {
      if (has[x]) { if (last < 0) for (let k = 0; k < x; k++) for (let c = 0; c < 3; c++) ref[k * 3 + c] = ref[x * 3 + c]; last = x; }
      else if (last >= 0) for (let c = 0; c < 3; c++) ref[x * 3 + c] = ref[last * 3 + c];
    }
    const sm = boxBlur(ref, bw, 1, 3, Math.max(3, Math.round(eyeW * 0.12)));
    for (let x = 0; x < bw; x++) {
      if (botY[x] < 0) continue;
      for (let y = botY[x] + 1; y < bh; y++) {
        const j = y * bw + x;
        for (let c = 0; c < 3; c++) H2[j * 3 + c] = sm[x * 3 + c];
      }
    }
  }
  solveHarmonic(H1, mask, bw, bh);
  solveHarmonic(H2, mask, bw, bh);
  const base = new Float32Array(N * 3);
  const band = Math.max(3, eyeW * 0.05);
  for (let x = 0; x < bw; x++) {
    for (let y = 0; y < bh; y++) {
      const j = y * bw + x;
      if (!mask[j]) continue;
      // distance to the nearest known pixel below in this column
      let db = 1e9;
      for (let yy = y + 1; yy < bh; yy++) if (!mask[yy * bw + x]) { db = yy - y; break; }
      const w = 1 - smooth(db / band);
      for (let c = 0; c < 3; c++) base[j * 3 + c] = lerp(H2[j * 3 + c], H1[j * 3 + c], w);
    }
  }

  // ---- 2b. skin texture, copied block-by-block from clean skin ----
  const R = Math.max(2, Math.round(eyeW * 0.025));
  const blurred = boxBlur(rgb, bw, bh, 3, R);
  const Lbig = boxBlur(L, bw, bh, 1, Math.max(4, Math.round(eyeW * 0.08)));
  const good = new Uint8Array(N);
  for (let j = 0; j < N; j++) {
    good[j] = oval[j] && !nearBrow[j] && Math.abs(L[j] - Lbig[j]) < 28 && L[j] > Lbig[j] * 0.72 ? 1 : 0;
  }
  const B = Math.max(8, Math.round(eyeW * 0.14)), S = Math.max(4, B >> 1);
  const tex = new Float32Array(N * 3), tw = new Float32Array(N);
  const rnd = mulberry32(4242);
  let mx0 = bw, mx1 = 0, my0 = bh, my1 = 0;
  for (let j = 0; j < N; j++) if (mask[j]) { const x = j % bw, y = (j / bw) | 0; if (x < mx0) mx0 = x; if (x > mx1) mx1 = x; if (y < my0) my0 = y; if (y > my1) my1 = y; }
  for (let ty = my0 - S; ty <= my1; ty += S) for (let tx = mx0 - S; tx <= mx1; tx += S) {
    // does this block touch the mask?
    let touches = false;
    for (let y = Math.max(0, ty); y < Math.min(bh, ty + B) && !touches; y += 2)
      for (let x = Math.max(0, tx); x < Math.min(bw, tx + B); x += 2) if (mask[y * bw + x]) { touches = true; break; }
    if (!touches) continue;
    // pick the cleanest nearby source block (prefer close, above)
    let best = null, bestScore = -1;
    for (let k = 0; k < 40; k++) {
      const sx = Math.round(tx + (rnd() - 0.5) * eyeW * 0.9);
      const sy = Math.round(ty - eyeW * (0.25 + rnd() * 0.75));
      if (sx < 0 || sy < 0 || sx + B > bw || sy + B > bh) continue;
      let ok = 0, n = 0;
      for (let y = 0; y < B; y += 2) for (let x = 0; x < B; x += 2) { ok += good[(sy + y) * bw + sx + x]; n++; }
      const score = ok / n - Math.abs(sx - tx) / (eyeW * 6);
      if (score > bestScore) { bestScore = score; best = [sx, sy]; }
      if (ok === n && k > 10) break;
    }
    if (!best || bestScore < 0.6) continue;
    const [sx, sy] = best;
    for (let y = 0; y < B; y++) for (let x = 0; x < B; x++) {
      const X = tx + x, Y = ty + y;
      if (X < 0 || Y < 0 || X >= bw || Y >= bh || !mask[Y * bw + X]) continue;
      const sj = (sy + y) * bw + sx + x;
      if (!good[sj]) continue;
      const wgt = (1 - Math.abs((x + 0.5) / B * 2 - 1)) * (1 - Math.abs((y + 0.5) / B * 2 - 1)) + 1e-3;
      const tj = Y * bw + X;
      const gain = clamp(lum(base[tj * 3], base[tj * 3 + 1], base[tj * 3 + 2]) / Math.max(30, Lbig[sj]), 0.6, 1.4);
      for (let c = 0; c < 3; c++) tex[tj * 3 + c] += clamp(rgb[sj * 3 + c] - blurred[sj * 3 + c], -45, 45) * gain * wgt;
      tw[tj] += wgt;
    }
  }
  // fallback grain where no clean source was found
  let grain = 0, gn = 0;
  for (let j = 0; j < N; j += 3) if (good[j]) { grain += Math.abs(L[j] - lum(blurred[j * 3], blurred[j * 3 + 1], blurred[j * 3 + 2])); gn++; }
  grain = gn ? grain / gn : 3;

  // ---- compose with a 1–2 px feathered edge ----
  const alpha = boxBlur(Float32Array.from(mask), bw, bh, 1, 1);
  const out = sctx.createImageData(W, H);
  out.data.set(d);
  for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
    const j = y * bw + x;
    const a = mask[j] ? 1 : alpha[j];
    if (a <= 0) continue;
    const k = ((y + by0) * W + (x + bx0)) * 4;
    let fill;
    if (mask[j]) {
      fill = [0, 1, 2].map((c) => base[j * 3 + c] + (tw[j] > 0 ? tex[j * 3 + c] / tw[j] : (rnd() + rnd() - 1) * grain * 1.2));
    } else {
      // just outside the mask: soften the seam with the neighbouring fill
      fill = [d[k], d[k + 1], d[k + 2]];
    }
    for (let c = 0; c < 3; c++) out.data[k + c] = lerp(d[k + c], fill[c], a);
  }
  const result = makeCanvas(W, H);
  result.getContext('2d').putImageData(out, 0, 0);

  // debug / test hook: the final mask in image space
  if (window.__brow) {
    const mc = makeCanvas(W, H), mg = mc.getContext('2d'), md = mg.createImageData(bw, bh);
    for (let j = 0; j < N; j++) if (mask[j]) { md.data[j * 4] = md.data[j * 4 + 1] = md.data[j * 4 + 2] = 255; md.data[j * 4 + 3] = 255; }
    mg.putImageData(md, bx0, by0);
    window.__brow.lastMask = mc;
    window.__brow.lastRemoveMs = Math.round(performance.now() - t0);
  }

  // ---- hair colour from the darkest brow pixels ----
  let hair = null;
  {
    const px = [];
    let sr = 0, sg = 0, sb = 0, sn = 0;
    for (let j = 0; j < N; j++) {
      if (!coreShape[j]) continue;
      px.push([rgb[j * 3], rgb[j * 3 + 1], rgb[j * 3 + 2], L[j]]);
      sr += H1[j * 3]; sg += H1[j * 3 + 1]; sb += H1[j * 3 + 2]; sn++;
    }
    if (px.length > 20 && sn) {
      px.sort((a, b) => a[3] - b[3]);
      const n = Math.max(5, Math.floor(px.length * 0.25));
      let rr = 0, gg = 0, bb = 0;
      for (let i = 0; i < n; i++) { rr += px[i][0]; gg += px[i][1]; bb += px[i][2]; }
      const c = { r: rr / n, g: gg / n, b: bb / n };
      const skinC = [sr / sn, sg / sn, sb / sn];
      if (lum(...skinC) - lum(c.r, c.g, c.b) > 45) hair = rgbToHex({ r: c.r * 0.72, g: c.g * 0.7, b: c.b * 0.68 });
      else hair = rgbToHex({ r: skinC[0] * 0.12 + 22, g: skinC[1] * 0.09 + 14, b: skinC[2] * 0.07 + 10 });
      const hc = hexToRgb(hair), HL = lum(hc.r, hc.g, hc.b);
      if (HL > 70) { const k = 70 / HL; hair = rgbToHex({ r: hc.r * k, g: hc.g * k, b: hc.b * k }); }
    }
  }
  return { canvas: result, hair, brow: { mask, bx0, by0, bw, bh } };
}


/* ------------------------------ AI inpainting ------------------------------
 * MI-GAN (Picsart AI Research, MIT licence) running on onnxruntime-web.
 * It repaints the brow area as skin using the whole face as context. The
 * model ships in two parts (hosting limit of 25 MB per file).
 * -------------------------------------------------------------------------- */
let miganPromise = null;
function getMigan() {
  if (!miganPromise) {
    miganPromise = (async () => {
      ort.env.wasm.wasmPaths = new URL('./vendor/ort/', import.meta.url).href;
      ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
      ort.env.wasm.proxy = true; // run in a worker so the page (and its animation) stays smooth
      const parts = await Promise.all(['migan.part0', 'migan.part1'].map(async (f) => {
        const r = await fetch(new URL('./models/' + f, import.meta.url));
        if (!r.ok) throw new Error('model download failed');
        return new Uint8Array(await r.arrayBuffer());
      }));
      const buf = new Uint8Array(parts[0].length + parts[1].length);
      buf.set(parts[0], 0); buf.set(parts[1], parts[0].length);
      return ort.InferenceSession.create(buf, { executionProviders: ['wasm'] });
    })();
    miganPromise.catch(() => { miganPromise = null; });
  }
  return miganPromise;
}

// Repaints the brow mask with MI-GAN, one brow at a time.
async function aiRemoveBrows(src, brow, face, onStep = () => {}) {
  const { W, H } = state;
  const { mask, bx0, by0, bw, bh } = brow;
  const session = await getMigan();
  // a little extra margin around the hair for the model
  const m2 = dilateMask(mask, bw, bh, Math.max(2, Math.round(face.eyeW * 0.025)));
  const sctx = src.getContext('2d', { willReadFrequently: true });
  const img = sctx.getImageData(0, 0, W, H);
  const d = img.data;
  const midX = face.midX - bx0;
  // the model only needs the area around the brows (plus context)
  const cx0 = Math.max(0, bx0 - Math.round(face.eyeW * 0.4)), cy0 = Math.max(0, by0 - Math.round(face.eyeW * 0.3));
  const cx1 = Math.min(W, bx0 + bw + Math.round(face.eyeW * 0.4)), cy1 = Math.min(H, by0 + bh + Math.round(face.eyeW * 0.5));
  const cw = cx1 - cx0, ch = cy1 - cy0;
  const yieldFrame = () => new Promise((r) => setTimeout(r, 0));
  const runMask = async (holeAt) => { // holeAt(x, y) in box coords
    const known = new Uint8Array(cw * ch).fill(255);
    let any = false;
    for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
      if (!holeAt(x, y)) continue;
      const X = x + bx0 - cx0, Y = y + by0 - cy0;
      if (X < 0 || Y < 0 || X >= cw || Y >= ch) continue;
      known[Y * cw + X] = 0; any = true;
    }
    if (!any) return;
    const n = cw * ch, chw = new Uint8Array(3 * n);
    for (let Y = 0; Y < ch; Y++) for (let X = 0; X < cw; X++) {
      const k = (Y + cy0) * W + (X + cx0), j = Y * cw + X;
      chw[j] = d[k * 4]; chw[n + j] = d[k * 4 + 1]; chw[2 * n + j] = d[k * 4 + 2];
    }
    await yieldFrame();
    const out = await session.run({
      image: new ort.Tensor('uint8', chw, [1, 3, ch, cw]),
      mask: new ort.Tensor('uint8', known, [1, 1, ch, cw]),
    });
    await yieldFrame();
    const r = out.result.data;
    // composite with a 1 px soft edge
    for (let Y = 1; Y < ch - 1; Y++) for (let X = 1; X < cw - 1; X++) {
      const j = Y * cw + X;
      let a;
      if (!known[j]) a = 1;
      else {
        let m = 0;
        for (let oy = -1; oy <= 1; oy++) for (let ox = -1; ox <= 1; ox++) if (!known[j + oy * cw + ox]) m++;
        a = m / 9 * 0.6;
      }
      if (a <= 0) continue;
      const k = ((Y + cy0) * W + (X + cx0)) * 4;
      for (let c = 0; c < 3; c++) d[k + c] = lerp(d[k + c], r[c * n + j], a);
    }
  };
  const snapshot = () => { const c = makeCanvas(W, H); c.getContext('2d').putImageData(img, 0, 0); return c; };
  // one brow at a time (more resolution, cleaner context)
  onStep('side', 0);
  await runMask((x, y) => m2[y * bw + x] && x < midX);
  onStep('done', 0, snapshot());
  onStep('side', 1);
  await runMask((x, y) => m2[y * bw + x] && x >= midX);
  onStep('done', 1, snapshot());
  onStep('polish');
  // safety pass: hair-like dark specks left (or invented) inside the brow area
  for (let pass = 0; pass < 2; pass++) {
    const Lr = new Float32Array(bw * bh);
    for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
      const k = ((y + by0) * W + (x + bx0)) * 4;
      Lr[y * bw + x] = lum(d[k], d[k + 1], d[k + 2]);
    }
    const Lb = boxBlur(Lr, bw, bh, 1, Math.max(3, Math.round(face.eyeW * 0.06)));
    const spots = new Uint8Array(bw * bh);
    let cnt = 0;
    for (let k = 0; k < bw * bh; k++) {
      if (m2[k] && Lr[k] < Lb[k] - Math.max(14, Lb[k] * 0.09)) { spots[k] = 1; cnt++; }
    }
    if (cnt < 3) break;
    const sm = dilateMask(spots, bw, bh, Math.max(2, Math.round(face.eyeW * 0.03)));
    await runMask((x, y) => sm[y * bw + x] && x < midX);
    await runMask((x, y) => sm[y * bw + x] && x >= midX);
  }
  const result = makeCanvas(W, H);
  result.getContext('2d').putImageData(img, 0, 0);
  return result;
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

  // Proportions (eye-widths from the pupil line) follow a full, natural,
  // softly angled brow: low full head, steady rise to an arch above the
  // outer eye corner, tail finishing a little above the head.
  //   head 0.40–0.67 · arch 0.62–0.89 · tail 0.46

  // Brow base: mostly where her own brow bone / brow sits.
  const idealBot = eyeY - 0.45 * eyeW;
  const ownBot = face.browInBotY + 0.06 * eyeW;
  const innerBotY = Math.abs(ownBot - idealBot) < 0.3 * eyeW ? lerp(idealBot, ownBot, 0.85) : idealBot;

  const thickness = 0.27 * eyeW * s.thick;
  const innerTopY = innerBotY - thickness;
  const archTopY = innerTopY - 0.22 * eyeW * s.lift;
  const archThick = thickness * 0.85;

  const ax = face.noseHalf, ay = face.nostrilY;

  // 1) start: up from the nostril edge, pulled in towards the bridge of the
  //    nose (never wider than the inner eye corner)
  const innerHalf = Math.min(ax, face.innerCornerHalf) * 0.8 * s.gap;
  // 2) arch: straight above the outer corner of the eye
  const archHalf = face.outerHalf * 1.03 * s.arch;
  const archSlope = (archHalf - ax) / Math.max(1, ay - archTopY);
  // 3) tail: line from the nostril edge past the outer eye corner
  //    (clears the corner by ~0.21 eye-widths)
  const tailY = innerBotY - 0.06 * eyeW;
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
  // Realistic hair: long, fine, tapered hairs following the natural flow of
  // a brow - upright fan at the head, lower hairs rising diagonally and
  // upper hairs lying down (meeting along a spine), everything converging
  // into the tail. Soft edges come from hair tips, not from a cut outline.
  const { eyeW, midX } = face;
  const probe = makeCanvas(2, 2).getContext('2d');
  const rnd = mulberry32(1234);
  const strokes = [];
  const base = hexToRgb(s.color);
  const span = geo.tailHalf - geo.innerHalf;
  const archU = clamp((geo.archHalf - geo.innerHalf) / span, 0.3, 0.85);

  for (const side of geo.sides) {
    const sg = side.sg;
    // vertical extent of the brow at each position along it
    const BINS = 80, topB = new Float32Array(BINS).fill(1e9), botB = new Float32Array(BINS).fill(-1e9);
    const ol = side.outline;
    for (let i = 0; i < ol.length; i++) { // walk the outline densely so every slot is filled
      const p0 = ol[i], p1 = ol[(i + 1) % ol.length];
      const steps = Math.max(1, Math.ceil(dist(p0, p1)));
      for (let t = 0; t <= steps; t++) {
        const px = lerp(p0.x, p1.x, t / steps), py = lerp(p0.y, p1.y, t / steps);
        const u = clamp((Math.abs(px - midX) - geo.innerHalf) / span);
        const b = Math.min(BINS - 1, Math.floor(u * BINS));
        topB[b] = Math.min(topB[b], py); botB[b] = Math.max(botB[b], py);
      }
    }
    for (let b = 0; b < BINS; b++) if (topB[b] > 1e8) { // fill gaps from neighbours
      let l = b - 1, r = b + 1;
      while (l >= 0 && topB[l] > 1e8) l--;
      while (r < BINS && topB[r] > 1e8) r++;
      const src = l >= 0 ? l : r;
      if (src >= 0 && src < BINS) { topB[b] = topB[src]; botB[b] = botB[src]; }
    }
    const vAt = (u, y) => {
      const b = Math.min(BINS - 1, Math.floor(u * BINS));
      return clamp((botB[b] - y) / Math.max(1, botB[b] - topB[b]));
    };
    // direction of the brow's centre line at each position (degrees, towards the tail)
    const axis = new Float32Array(BINS);
    { // least-squares slope of the centre line over a window of slots
      const cen = Array.from({ length: BINS }, (_, b) => (topB[b] + botB[b]) / 2);
      const R = 6;
      for (let b = 0; b < BINS; b++) {
        let sx = 0, sy = 0, sxx = 0, sxy = 0, n = 0;
        for (let k = Math.max(2, b - R); k <= Math.min(BINS - 3, b + R); k++) {
          const xk = (k / BINS) * span, yk = cen[k];
          sx += xk; sy += yk; sxx += xk * xk; sxy += xk * yk; n++;
        }
        const slope = n > 1 ? (n * sxy - sx * sy) / Math.max(1e-6, n * sxx - sx * sx) : 0;
        axis[b] = Math.atan(-slope) * 180 / Math.PI;
      }
    }
    const axisAt = (u) => axis[Math.min(BINS - 1, Math.floor(u * BINS))];
    // area -> number of hairs
    let area = 0;
    const o = side.outline;
    for (let i = 0; i < o.length; i++) { const a = o[i], b = o[(i + 1) % o.length]; area += a.x * b.y - b.x * a.y; }
    area = Math.abs(area) / 2;
    const total = Math.round(clamp(area / (eyeW * eyeW) * 4000, 500, 3600) * s.thick);
    const xs = o.map((p) => p.x), ys = o.map((p) => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);

    let made = 0, tries = 0;
    while (made < total && tries < total * 60) {
      tries++;
      const x = lerp(minX, maxX, rnd()), y = lerp(minY, maxY, rnd());
      if (!probe.isPointInPath(side.path, x, y)) continue;
      const u = clamp((Math.abs(x - midX) - geo.innerHalf) / span);
      const v = vAt(u, y);
      // soft, airy head and a fine tail tip
      if (u < 0.12 && rnd() > 0.25 + 0.75 * smooth(u / 0.12)) continue;
      if (u > 0.92 && rnd() > 1 - (u - 0.92) * 8) continue;
      if (v > 0.88 && rnd() > 0.55) continue; // airy top edge
      made++;

      // ---- direction (degrees above horizontal, pointing towards the tail) ----
      // body hairs follow the brow's centre line: lower hairs rise a little
      // more steeply, upper hairs lie flat (they meet along a soft spine)
      // hairs rooted near the top edge lift slightly so their tips break the outline
      const body = axisAt(u) + lerp(26, -5, smooth((v - 0.15) / 0.75)) + (v > 0.8 ? (v - 0.8) * 25 : 0);
      const headW = 1 - smooth(u / 0.2);
      let a = lerp(body, 78 - v * 10, headW);   // head: upright, slightly fanned
      a += (rnd() - 0.5) * 8;
      const rad = (a * Math.PI) / 180;

      // ---- length / shape ----
      const lenK = u < 0.12 ? lerp(0.06, 0.11, rnd())
        : u > 0.88 ? lerp(0.1, 0.17, rnd())
        : lerp(0.16, 0.3, rnd()) * (0.75 + 0.25 * smooth(u / 0.35));
      const len = eyeW * lenK;
      const dx = sg * Math.cos(rad) * len, dy = -Math.sin(rad) * len;
      // hairs curve clockwise (tip bends towards the tail / downwards)
      const bend = len * lerp(0.08, 0.18, rnd());
      const nx = sg * Math.sin(rad), ny = Math.cos(rad);

      // ---- look ----
      const r = rnd();
      let L, shade, alpha, w;
      if (r < 0.22) { L = 0; shade = lerp(1.05, 1.35, rnd()); alpha = lerp(0.18, 0.32, rnd()); w = 1.15; }     // soft under-hairs
      else if (r < 0.86) { L = 1; shade = lerp(0.5, 1.12, rnd()); alpha = lerp(0.45, 0.92, rnd()); w = 1.0; }   // main, varied
      else { L = 2; shade = lerp(1.5, 2.0, rnd()); alpha = lerp(0.3, 0.5, rnd()); w = 0.8; }                    // sheen highlights
      const col = L === 2
        ? { r: base.r * shade + 25, g: base.g * shade + 22, b: base.b * shade + 22 }
        : { r: base.r * shade, g: base.g * shade, b: base.b * shade };
      strokes.push({
        sg, u, L,
        order: u * 0.85 + (L === 0 ? 0 : L === 1 ? 0.05 : 0.1) + rnd() * 0.08,
        x, y,
        cx: x + dx * 0.5 + nx * bend, cy: y + dy * 0.5 + ny * bend,
        ex: x + dx + nx * bend * 0.6, ey: y + dy + ny * bend * 0.6,
        w: Math.max(0.35, eyeW * 0.0058) * w * lerp(0.75, 1.25, rnd()),
        alpha: alpha * (u < 0.1 ? 0.75 : 1),
        color: `rgb(${clamp(col.r, 0, 255) | 0},${clamp(col.g, 0, 255) | 0},${clamp(col.b, 0, 255) | 0})`,
      });
    }
    // a few stray hairs just outside the top and bottom edges
    const strays = Math.round(total * 0.008);
    for (let i = 0, t = 0; i < strays && t < strays * 80; t++) {
      const x = lerp(minX, maxX, rnd()), y = lerp(minY - eyeW * 0.04, maxY, rnd());
      if (probe.isPointInPath(side.path, x, y)) continue;
      const u = clamp((Math.abs(x - midX) - geo.innerHalf) / span);
      if (u < 0.15 || u > 0.9) continue;
      const v = vAt(u, y);
      if (v < 1 || v > 1.35) continue;
      i++;
      const a = axisAt(u) - 4 + (rnd() - 0.5) * 10;
      const rad = (a * Math.PI) / 180, len = eyeW * lerp(0.08, 0.15, rnd());
      const dx = sg * Math.cos(rad) * len, dy = -Math.sin(rad) * len;
      strokes.push({
        sg, u, L: 1, order: u * 0.85 + 0.12 + rnd() * 0.05, x, y,
        cx: x + dx * 0.5, cy: y + dy * 0.5, ex: x + dx, ey: y + dy,
        w: Math.max(0.3, eyeW * 0.0045), alpha: lerp(0.18, 0.35, rnd()),
        color: `rgb(${base.r | 0},${base.g | 0},${base.b | 0})`,
      });
    }
  }
  strokes.sort((a, b) => a.order - b.order);
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
      [6.1, 'Arch: above the outer eye corner'],
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
const hairLayer = { cv: null, key: null, drawn: 0 };

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
    // Hairs are drawn at 2x resolution (fine, anti-aliased) onto a cached
    // layer; while the brows grow in, only the newly revealed hairs are added.
    const { W, H } = state;
    const SS = Math.min(2, Math.floor(4096 / Math.max(W, H)) || 1);
    if (!hairLayer.cv || hairLayer.cv.width !== W * SS || hairLayer.cv.height !== H * SS) {
      hairLayer.cv = makeCanvas(W * SS, H * SS); hairLayer.key = null;
    }
    const strokes = geo.strokes;
    const reveal = clamp(progress) * 1.05;
    let target = 0;
    while (target < strokes.length && strokes[target].order <= reveal) target++;
    const h = hairLayer.cv.getContext('2d');
    if (hairLayer.key !== strokes || target < hairLayer.drawn) {
      h.setTransform(1, 0, 0, 1, 0, 0);
      h.clearRect(0, 0, hairLayer.cv.width, hairLayer.cv.height);
      hairLayer.key = strokes; hairLayer.drawn = 0;
    }
    if (target > hairLayer.drawn) {
      h.save();
      h.setTransform(SS, 0, 0, SS, 0, 0);
      faceTransform(h);
      h.lineCap = 'round';
      for (let n = hairLayer.drawn; n < target; n++) {
        const k = strokes[n];
        h.strokeStyle = k.color;
        // tapered hair: thick root, fine tip (4 segments along the curve)
        let px = k.x, py = k.y;
        for (let i = 1; i <= 4; i++) {
          const t = i / 4, u = 1 - t;
          const qx = u * u * k.x + 2 * u * t * k.cx + t * t * k.ex;
          const qy = u * u * k.y + 2 * u * t * k.cy + t * t * k.ey;
          h.globalAlpha = k.alpha * (1 - (i - 1) * 0.15);
          h.lineWidth = k.w * (1.1 - (i - 1) * 0.28);
          h.beginPath(); h.moveTo(px, py); h.lineTo(qx, qy); h.stroke();
          px = qx; py = qy;
        }
      }
      h.restore();
      hairLayer.drawn = target;
    }
    c.save();
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.imageSmoothingQuality = 'high';
    c.drawImage(hairLayer.cv, 0, 0, W, H);
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

/* ------------------------------ working animation ------------------------------
 * Runs while the photo is being analysed and the brows are erased:
 * a soft scanning beam, the face mesh twinkling in, particles rising from the
 * brow being erased, and a progress bar whose label and hint change over time.
 * --------------------------------------------------------------------------- */
const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const busy = {
  on: false, raf: 0, t0: 0, last: 0, maxGap: 0,
  base: null, next: null, fadeStart: 0,
  face: null, faceT: 0, spawn: null, side: -1, midX: 0,
  particles: [],
  from: 0, to: 0, stageStart: 0, stageMs: 1000, shown: 0,
  subs: [], subIdx: -1, subT: 0,
};
const busyTimes = { side: 3000 };
const busyEls = { box: $('busy'), text: $('busyText'), pct: $('busyPct'), fill: $('busyFill'), sub: $('busySub') };

function busyStart(img) {
  busy.on = true; busy.base = img; busy.next = null; busy.face = null; busy.spawn = null;
  busy.particles = []; busy.shown = 0; busy.maxGap = 0;
  busy.t0 = busy.last = performance.now();
  busyEls.box.hidden = false;
  cancelAnimationFrame(busy.raf);
  busy.raf = requestAnimationFrame(busyFrame);
}
function busyStage(text, from, to, ms, subs = []) {
  busy.from = Math.max(from, busy.shown); busy.to = to; busy.stageMs = Math.max(200, ms);
  busy.stageStart = performance.now();
  busy.subs = subs; busy.subIdx = -1; busy.subT = 0;
  busyEls.text.textContent = text;
}
function busySetImage(canvas) { busy.next = canvas; busy.fadeStart = performance.now(); }
function busySetFace(face) { busy.face = face; busy.faceT = performance.now(); }
function busySetBrow(brow, face) {
  const pts = [];
  const { mask, bx0, by0, bw, bh } = brow;
  for (let k = 0; k < mask.length; k += 3) if (mask[k]) pts.push([(k % bw) + bx0, ((k / bw) | 0) + by0]);
  busy.spawn = pts; busy.midX = face.midX;
}
function busyFinish() {
  return new Promise((resolve) => {
    busyStage('Done', busy.shown, 1, 250);
    busy.to = 1; busy.from = 1;
    setTimeout(() => {
      busyEls.box.hidden = true;
      busy.on = false;
      cancelAnimationFrame(busy.raf);
      resolve();
    }, 420);
  });
}
function busyFail() { busy.on = false; cancelAnimationFrame(busy.raf); busyEls.box.hidden = true; }

function busyFrame(now) {
  if (!busy.on) return;
  busy.maxGap = Math.max(busy.maxGap, now - busy.last);
  busy.last = now;
  const t = (now - busy.t0) / 1000;
  const { W, H } = state;
  const c = ctx;

  // progress: eases towards the end of the current stage, never stalls, never goes back
  const e = 1 - Math.exp(-(now - busy.stageStart) / busy.stageMs);
  const target = busy.from + (busy.to - busy.from) * (busy.to >= 1 ? 1 : e * 0.97);
  busy.shown = Math.max(busy.shown, busy.shown + (target - busy.shown) * 0.12);
  const pct = Math.min(100, Math.round(busy.shown * 100));
  busyEls.pct.textContent = pct + '%';
  busyEls.fill.style.width = (busy.shown * 100).toFixed(1) + '%';

  // rotating hint under the bar
  if (busy.subs.length && (busy.subIdx < 0 || now - busy.subT > 2300)) {
    busy.subIdx = (busy.subIdx + 1) % busy.subs.length; busy.subT = now;
    busyEls.sub.style.opacity = 0;
    const msg = busy.subs[busy.subIdx];
    setTimeout(() => { busyEls.sub.textContent = msg; busyEls.sub.style.opacity = 1; }, 180);
  } else if (!busy.subs.length && busyEls.sub.textContent) busyEls.sub.textContent = '';

  // ---- picture ----
  c.setTransform(1, 0, 0, 1, 0, 0);
  c.clearRect(0, 0, W, H);
  if (busy.base) c.drawImage(busy.base, 0, 0);
  if (busy.next) {
    const a = smooth((now - busy.fadeStart) / 700);
    c.save(); c.globalAlpha = a; c.drawImage(busy.next, 0, 0); c.restore();
    if (a >= 1) { busy.base = busy.next; busy.next = null; }
  }
  c.fillStyle = 'rgba(18, 14, 13, 0.16)';
  c.fillRect(0, 0, W, H);
  if (reduceMotion) { busy.raf = requestAnimationFrame(busyFrame); return; }

  const f = busy.face;
  const unit = f ? f.eyeW : Math.min(W, H) * 0.12;
  // scanning beam over the face (or the whole photo before the face is found)
  let y0 = 0, y1 = H;
  if (f) { y0 = Math.max(0, f.P[10].y - unit * 0.4); y1 = Math.min(H, f.P[152].y + unit * 0.2); }
  const by = y0 + (y1 - y0) * (0.5 - 0.5 * Math.cos(t * Math.PI * 2 / 3.2));
  const bh = unit * 0.5;
  c.save();
  c.globalCompositeOperation = 'lighter';
  const g = c.createLinearGradient(0, by - bh, 0, by + bh);
  g.addColorStop(0, 'rgba(255,230,0,0)'); g.addColorStop(0.5, 'rgba(255,230,0,0.13)'); g.addColorStop(1, 'rgba(255,230,0,0)');
  c.fillStyle = g; c.fillRect(0, by - bh, W, bh * 2);
  c.fillStyle = 'rgba(255,240,150,0.35)'; c.fillRect(0, by - 0.75, W, 1.5);
  c.restore();

  // face mesh twinkling in
  if (f) {
    const fin = smooth((now - busy.faceT) / 900);
    c.save();
    c.globalCompositeOperation = 'lighter';
    const r = Math.max(0.8, unit * 0.011);
    for (let i = 0; i < 468; i += 3) {
      const p = f.P[i];
      const near = 1 - smooth(Math.abs(p.y - by) / (unit * 1.2));
      const a = fin * (0.18 + 0.22 * (0.5 + 0.5 * Math.sin(t * 2.6 + i * 1.7)) + 0.5 * near);
      c.fillStyle = `rgba(255,248,210,${a.toFixed(3)})`;
      c.beginPath(); c.arc(p.x, p.y, r, 0, Math.PI * 2); c.fill();
    }
    c.restore();
  }

  // particles rising from the brow being erased
  if (busy.spawn && busy.spawn.length && busy.side >= 0) {
    const pool = busy.side === 2 ? busy.spawn : busy.spawn.filter((p) => (busy.side === 0 ? p[0] < busy.midX : p[0] >= busy.midX));
    if (pool.length) {
      for (let n = 0; n < 5; n++) {
        const p = pool[(Math.random() * pool.length) | 0];
        busy.particles.push({ x: p[0], y: p[1], vx: (Math.random() - 0.5) * unit * 0.004, vy: -unit * (0.004 + Math.random() * 0.008), life: 0, max: 0.7 + Math.random() * 0.9, r: unit * (0.008 + Math.random() * 0.016) });
      }
    }
  }
  if (busy.particles.length) {
    c.save();
    c.globalCompositeOperation = 'lighter';
    const keep = [];
    for (const q of busy.particles) {
      q.life += 1 / 60; q.x += q.vx; q.y += q.vy; q.vy *= 0.995;
      if (q.life > q.max) continue;
      keep.push(q);
      const k = q.life / q.max, a = Math.sin(k * Math.PI) * 0.85;
      const col = k < 0.5 ? `rgba(255,230,0,${a.toFixed(3)})` : `rgba(255,${Math.round(230 - (k - 0.5) * 2 * 180)},${Math.round((k - 0.5) * 2 * 214)},${a.toFixed(3)})`;
      c.fillStyle = col;
      c.beginPath(); c.arc(q.x, q.y, q.r * (1 - k * 0.5), 0, Math.PI * 2); c.fill();
    }
    busy.particles = keep.length > 900 ? keep.slice(-900) : keep;
    c.restore();
  }
  busy.raf = requestAnimationFrame(busyFrame);
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
    setStatus('');
    busyStart(orig);
    busyStage('Getting ready', 0, 0.1, 1500, ['Warming up the face model', 'Loading the skin AI']);
    const migan = getMigan().catch((e) => { throw e; });
    migan.catch(() => {});
    const lmk = await getLandmarker();
    busyStage('Finding her face', 0.1, 0.18, 500, ['Locating 478 face points']);
    await new Promise((r) => setTimeout(r, 60));
    const res = lmk.detect(orig);
    if (!res.faceLandmarks || !res.faceLandmarks.length) {
      busyFail();
      setStatus('No face found. Try a well-lit, front-facing photo.', true);
      document.documentElement.dataset.browState = 'error';
      return;
    }
    state.face = buildFace(res.faceLandmarks[0], W, H);
    busySetFace(state.face);
    busyStage('Mapping brow hairs', 0.18, 0.26, 600, ['Tracing every hair', 'Protecting lids and eyeshadow']);
    await new Promise((r) => setTimeout(r, 450));
    const { canvas, hair, brow } = removeBrows(orig, state.face);
    state.clean = canvas;
    busySetBrow(brow, state.face);
    // AI repaint (better skin, shading and texture); the classic fill stays as fallback
    const sideMs = busyTimes.side;
    try {
      busyStage('Erasing brow 1 of 2', 0.26, 0.56, sideMs + 1200, ['Rebuilding the skin underneath', 'Matching skin tone', 'Recreating pores and texture']);
      busy.side = 0;
      await migan;
      busy.maxGap = 0; // measure smoothness of the AI phase only
      const t0 = performance.now();
      let tSide = t0;
      state.clean = await aiRemoveBrows(orig, brow, state.face, (step, side, snap) => {
        if (step === 'side' && side === 1) {
          busyStage('Erasing brow 2 of 2', 0.56, 0.86, busyTimes.side, ['Blending with the forehead', 'Keeping the eyelid untouched', 'Softening the edges']);
          busy.side = 1;
        } else if (step === 'done') {
          busyTimes.side = performance.now() - tSide; tSide = performance.now();
          busySetImage(snap);
        } else if (step === 'polish') {
          busyStage('Polishing', 0.86, 0.97, busyTimes.side, ['Checking for any leftover hairs']);
          busy.side = 2;
        }
      });
      if (window.__brow) window.__brow.aiMs = Math.round(performance.now() - t0);
    } catch (e) {
      console.warn('AI brow removal unavailable, using classic fill', e);
      if (window.__brow) window.__brow.aiError = String(e && e.message || e);
    }
    busy.side = -1;
    busySetImage(state.clean);
    await new Promise((r) => setTimeout(r, 650));
    if (window.__brow) window.__brow.maxGap = Math.round(busy.maxGap);
    await busyFinish();
    if (hair) els.color.value = hair;
    syncOutputs();
    state.brows = mapBrows(state.face, settings());
    state.timeline = buildTimeline(state.face, state.brows);
    play();
  } catch (err) {
    console.error(err);
    busyFail();
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
getMigan().catch(() => {});

// test hook: render any moment of the animation
window.__brow = { seek: (t) => { state.playing = false; render(t); }, state,
  prep: (src) => (window.__brow._r = removeBrows(src, state.face)),
  ai: async (src) => aiRemoveBrows(src, window.__brow._r.brow, state.face, (step) => { (window.__brow.log = window.__brow.log || []).push([step, Math.round(performance.now())]); }) };
