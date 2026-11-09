import { FaceLandmarker, FilesetResolver } from './vendor/mediapipe/vision_bundle.mjs';
import * as ort from './vendor/ort/ort.wasm.min.mjs';
import { removeBrowsData, dilateMask, boxBlur } from './removal.js';

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

/* ------------------------------ brow removal ------------------------------
 * Detection + classic fill live in removal.js and run in a Web Worker
 * (removal-worker.js) so the page never freezes; inline as a fallback.
 * -------------------------------------------------------------------------- */
let remWorker = null, remSeq = 0;
function removeBrows(src, face) {
  const { W, H } = state;
  const d = src.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, W, H).data;
  const P = face.P.map((p) => ({ x: p.x, y: p.y }));
  const finish = (r) => {
    const out = new ImageData(new Uint8ClampedArray(r.out.buffer), W, H);
    const canvas = makeCanvas(W, H);
    canvas.getContext('2d').putImageData(out, 0, 0);
    const brow = { mask: r.mask, bx0: r.bx0, by0: r.by0, bw: r.bw, bh: r.bh };
    if (window.__brow) { // test hook: mask in image space
      const mc = makeCanvas(W, H), mg = mc.getContext('2d'), md = mg.createImageData(r.bw, r.bh);
      for (let k = 0; k < r.mask.length; k++) if (r.mask[k]) { md.data[k * 4] = md.data[k * 4 + 1] = md.data[k * 4 + 2] = md.data[k * 4 + 3] = 255; }
      mg.putImageData(md, r.bx0, r.by0);
      window.__brow.lastMask = mc;
    }
    return { canvas, hair: r.hair, brow };
  };
  const inline = () => finish(removeBrowsData(new Uint8ClampedArray(d), W, H, P, face.eyeW));
  if (typeof Worker === 'undefined' || typeof OffscreenCanvas === 'undefined') return Promise.resolve().then(inline);
  try {
    if (!remWorker) remWorker = new Worker(new URL('./removal-worker.js', import.meta.url), { type: 'module' });
  } catch { return Promise.resolve().then(inline); }
  const id = ++remSeq;
  return new Promise((resolve, reject) => {
    const onMsg = (e) => {
      if (e.data.id !== id) return;
      remWorker.removeEventListener('message', onMsg);
      if (e.data.ok) resolve(finish(e.data.r)); else reject(new Error(e.data.error));
    };
    remWorker.addEventListener('message', onMsg);
    remWorker.addEventListener('error', () => { try { resolve(inline()); } catch (er) { reject(er); } }, { once: true });
    const copy = new Uint8ClampedArray(d);
    remWorker.postMessage({ id, d: copy, W, H, P, eyeW: face.eyeW }, [copy.buffer]);
  });
}

/* ------------------------------ AI inpainting ------------------------------
 * MI-GAN (Picsart AI Research, MIT licence) running on onnxruntime-web.
 * It repaints the brow area as skin using the whole face as context. The
 * model ships in two parts (hosting limit of 25 MB per file).
 * -------------------------------------------------------------------------- */
let miganPromise = null;
const miganLoad = { loaded: 0, total: 0, ready: false };
function getMigan() {
  if (!miganPromise) {
    miganPromise = (async () => {
      ort.env.wasm.wasmPaths = new URL('./vendor/ort/', import.meta.url).href;
      ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
      ort.env.wasm.proxy = true; // run in a worker so the page (and its animation) stays smooth
      const parts = await Promise.all(['migan.part0', 'migan.part1'].map(async (f) => {
        const r = await fetch(new URL('./models/' + f, import.meta.url));
        if (!r.ok) throw new Error('model download failed');
        const size = +r.headers.get('content-length') || 14e6;
        miganLoad.total += size;
        if (!r.body || !r.body.getReader) { const b = new Uint8Array(await r.arrayBuffer()); miganLoad.loaded += b.length; return b; }
        const reader = r.body.getReader(), chunks = [];
        let got = 0;
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value); got += value.length; miganLoad.loaded += value.length;
        }
        const b = new Uint8Array(got);
        let o = 0; for (const c of chunks) { b.set(c, o); o += c.length; }
        return b;
      }));
      const buf = new Uint8Array(parts[0].length + parts[1].length);
      buf.set(parts[0], 0); buf.set(parts[1], parts[0].length);
      const sess = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] });
      miganLoad.ready = true;
      return sess;
    })();
    miganPromise.catch(() => { miganPromise = null; miganLoad.loaded = miganLoad.total = 0; });
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
    removeStart: -2, removeEnd: -1, // brows are already erased during the working animation
    gridStart: 1.3, gridEnd: 1.9,
    lines: L, dots,
    outlineStart: 10.3, outlineEnd: 11.8,
    guidesFadeStart: 12.0, guidesFadeEnd: 12.8,
    fillStart: 12.6, fillEnd: 15.0,
    end: 15.6,
    captions: [
      [0, ''],
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
  busy.particles = []; busy.shown = 0; busy.maxGap = 0; busy.dl = false;
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
  // hyperbolic ease: quick at first, then keeps creeping forward even if a step runs long
  const e = 1 - 1 / (1 + ((now - busy.stageStart) / busy.stageMs) * 2.2);
  const frac = busy.dl ? (miganLoad.total ? miganLoad.loaded / miganLoad.total : 0) : e * 0.97;
  const target = busy.from + (busy.to - busy.from) * (busy.to >= 1 ? 1 : frac);
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
    const { canvas, hair, brow } = await removeBrows(orig, state.face);
    state.clean = canvas;
    busySetBrow(brow, state.face);
    // AI repaint (better skin, shading and texture); the classic fill stays as fallback
    const sideMs = busyTimes.side;
    try {
      let from = 0.26;
      if (!miganLoad.ready) { // first visit: show the real download progress
        busyStage('Downloading the skin AI', 0.26, 0.42, 1000, ['One-time download (~28 MB)', 'Saved on this device for next time']);
        busy.dl = true;
        await migan;
        busy.dl = false;
        from = 0.42;
      }
      busyStage('Erasing brow 1 of 2', from, 0.58, sideMs + 800, ['Rebuilding the skin underneath', 'Matching skin tone', 'Recreating pores and texture']);
      busy.side = 0;
      await migan;
      const t0 = performance.now();
      let tSide = t0;
      state.clean = await aiRemoveBrows(orig, brow, state.face, (step, side, snap) => {
        if (step === 'side' && side === 1) {
          busyStage('Erasing brow 2 of 2', 0.58, 0.86, busyTimes.side, ['Blending with the forehead', 'Keeping the eyelid untouched', 'Softening the edges']);
          busy.side = 1;
        } else if (step === 'done') {
          busyTimes.side = performance.now() - tSide; tSide = performance.now();
          busySetImage(snap);
        } else if (step === 'polish') {
          busyStage('Polishing', 0.86, 0.97, busyTimes.side * 1.5, ['Checking for any leftover hairs']);
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
  prep: async (src) => (window.__brow._r = await removeBrows(src, state.face)),
  ai: async (src) => aiRemoveBrows(src, window.__brow._r.brow, state.face, (step) => { (window.__brow.log = window.__brow.log || []).push([step, Math.round(performance.now())]); }) };
