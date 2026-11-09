import { FaceLandmarker, FilesetResolver } from './vendor/mediapipe/vision_bundle.mjs';
import { removeBrowsData, dilateMask, boxBlur } from './removal.js';
import { t as tr, applyLang, setLang, getLang, onLangChange } from './i18n.js';

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
  format: $('format'), savePng: $('savePng'), saveVid: $('saveVid'), newPhoto: $('newPhoto'),
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
      remWorker.removeEventListener('error', onErr);
      if (e.data.ok) resolve(finish(e.data.r)); else reject(new Error(e.data.error));
    };
    remWorker.addEventListener('message', onMsg);
    const onErr = () => { remWorker.removeEventListener('message', onMsg); try { resolve(inline()); } catch (er) { reject(er); } };
    remWorker.addEventListener('error', onErr, { once: true });
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
const miganLoad = { loaded: 0, total: 0, ready: false, mode: '', error: '' };
const MIGAN_BYTES = 28079181;

// Starts the model in a worker. First choice: multi-threaded (fast). If that
// fails or hangs, a fresh worker in single-threaded mode is tried; a failure of
// the fast mode is remembered so the next visit starts in the safe mode.
function getMigan() {
  if (!miganPromise) {
    miganPromise = (async () => {
      let safe = false;
      try { safe = localStorage.getItem('brow-ai-safe') === '1'; } catch {}
      const canThread = self.crossOriginIsolated && (navigator.hardwareConcurrency || 1) > 1;
      const modes = [];
      if (canThread && !safe) modes.push({ engine: 'fast', threads: Math.min(4, navigator.hardwareConcurrency || 1) });
      modes.push({ engine: 'light', threads: 1 });
      const errors = [];
      for (const mode of modes) {
        try {
          const w = await startAiWorker(mode.threads, mode.engine);
          miganLoad.ready = true; miganLoad.mode = mode.engine + ' ' + mode.threads;
          return w;
        } catch (e) {
          errors.push(`${mode.engine} mode: ${e.message || e}`);
          console.warn('AI start failed', mode, e);
          if (mode.engine === 'fast') { try { localStorage.setItem('brow-ai-safe', '1'); } catch {} }
          miganLoad.loaded = 0; miganLoad.total = 0;
        }
      }
      throw new Error(errors.join(' | '));
    })();
    miganPromise.catch((e) => { miganPromise = null; miganLoad.error = String(e.message || e); });
  }
  return miganPromise;
}

function startAiWorker(threads, engine = threads > 1 ? 'fast' : 'light') {
  return new Promise((resolve, reject) => {
    let w;
    if (engine === 'fast' && /[?&]aitest=failfast/.test(location.search)) { reject(new Error('forced failure (test)')); return; }
    try { w = new Worker(new URL('./ai-worker.js', import.meta.url), { type: 'module' }); }
    catch (e) { reject(new Error('this browser cannot start the AI worker')); return; }
    const pending = new Map();
    let seq = 0, ready = false, lastSign = performance.now();
    const fail = (msg) => {
      clearInterval(watch);
      try { w.terminate(); } catch {}
      for (const p of pending.values()) p.reject(new Error(msg));
      pending.clear();
      if (!ready) reject(new Error(msg));
    };
    // a worker that dies (e.g. out of memory) sends nothing: watch for silence
    const watch = setInterval(() => {
      const quiet = performance.now() - lastSign;
      if (!ready && quiet > 45000) fail('AI did not start (no response)');
      else if (ready && pending.size && quiet > 60000) fail('AI stopped responding');
    }, 2000);
    w.onerror = (e) => fail('AI worker crashed' + (e && e.message ? ': ' + e.message : ''));
    w.onmessage = (e) => {
      const m = e.data;
      lastSign = performance.now();
      if (m.type === 'progress') { miganLoad.loaded = m.loaded; miganLoad.total = m.total; return; }
      if (m.type === 'ready') {
        ready = true;
        resolve({
          threads,
          run(chw, known, cw, ch) {
            return new Promise((res, rej) => {
              const id = ++seq;
              pending.set(id, { resolve: res, reject: rej });
              lastSign = performance.now();
              w.postMessage({ type: 'run', id, chw, known, cw, ch }, [chw.buffer, known.buffer]);
            });
          },
        });
        return;
      }
      if (m.type === 'result' || m.type === 'error') {
        if (!ready && m.type === 'error') { fail(m.error); return; }
        const p = pending.get(m.id);
        if (!p) return;
        pending.delete(m.id);
        if (m.type === 'result') p.resolve(m.r); else p.reject(new Error(m.error));
      }
    };
    w.postMessage({
      type: 'init', id: 0, threads, engine, expected: MIGAN_BYTES,
      wasmPaths: new URL('./vendor/ort/', import.meta.url).href,
      wasmPathsLight: new URL('./vendor/ort118/', import.meta.url).href,
      urls: ['migan.part0', 'migan.part1'].map((f) => new URL('./models/' + f, import.meta.url).href),
    });
  });
}

class AbortError extends Error { constructor() { super('cancelled'); this.name = 'AbortError'; } }
// the model must never run twice at the same time: queue every call
let modelQueue = Promise.resolve();
function runModel(model, chw, known, cw, ch) {
  const job = modelQueue.then(() => model.run(chw, known, cw, ch));
  modelQueue = job.catch(() => {});
  return job;
}

// Repaints the brow mask with MI-GAN, one brow at a time.
async function aiRemoveBrows(src, brow, face, onStep = () => {}, alive = () => true) {
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
    if (!alive()) throw new AbortError();
    let r;
    try { r = await runModel(session, chw.slice(), known.slice(), cw, ch); }
    catch (e) { console.warn('AI step failed, retrying once', e); r = await runModel(session, chw.slice(), known.slice(), cw, ch); }
    if (!alive()) throw new AbortError();
    await yieldFrame();
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
  try { for (let pass = 0; pass < 2; pass++) {
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
  } } catch (e) {
    if (e instanceof AbortError) throw e;
    console.warn('polish pass failed; keeping the erased result', e);
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

// Bottom edge of her real brow head (face frame), measured from the hair mask
// found while erasing. null when there is no mask.
function ownBrowBottom(face) {
  const b = state.browInfo;
  if (!b) return null;
  // depends only on the photo: computed once per photo
  const memo = (b._memo = b._memo || {});
  if ('ownBrowBottom' in memo) return memo['ownBrowBottom'];
  return (memo['ownBrowBottom'] = ownBrowBottomCalc(face, b));
}
function ownBrowBottomCalc(face, b) {
  const { mask, bx0, by0, bw, bh } = b;
  const { midX, eyeW } = face;
  const h0 = Math.min(face.noseHalf, face.innerCornerHalf) * 0.8;
  const vals = [];
  for (let x = 0; x < bw; x++) {
    let bot = -1;
    for (let y = bh - 1; y >= 0; y--) if (mask[y * bw + x]) { bot = y; break; }
    if (bot < 0) continue;
    const q = face.F(pt(x + bx0, bot + by0));
    const half = Math.abs(q.x - midX);
    if (half >= h0 && half <= h0 + eyeW * 0.4) vals.push(q.y);
  }
  if (vals.length < 6) return null;
  vals.sort((a, b) => a - b);
  // the mask reaches a little past the hairs; step back to the hair edge
  return vals[vals.length >> 1] - eyeW * 0.03;
}

// How far out her real brow reaches (face frame, half-distance from the centre),
// averaged over both sides. null when there is no mask.
function ownBrowOuterHalf(face) {
  const b = state.browInfo;
  if (!b) return null;
  // depends only on the photo: computed once per photo
  const memo = (b._memo = b._memo || {});
  if ('ownBrowOuterHalf' in memo) return memo['ownBrowOuterHalf'];
  return (memo['ownBrowOuterHalf'] = ownBrowOuterHalfCalc(face, b));
}
function ownBrowOuterHalfCalc(face, b) {
  const { mask, bx0, by0, bw, bh } = b;
  let maxL = 0, maxR = 0;
  for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
    if (!mask[y * bw + x]) continue;
    const q = face.F(pt(x + bx0, y + by0));
    const h = q.x - face.midX;
    if (h < 0) maxL = Math.max(maxL, -h); else maxR = Math.max(maxR, h);
  }
  if (!maxL || !maxR) return null;
  // the mask reaches a little past the hairs
  return (maxL + maxR) / 2 - face.eyeW * 0.03;
}

function mapBrows(face, s) {
  const { midX, eyeY, eyeW } = face;

  // Proportions (eye-widths from the pupil line) follow a full, natural,
  // softly angled brow: low full head, steady rise to an arch above the
  // outer eye corner, tail finishing a little above the head.
  //   head 0.40–0.67 · arch 0.62–0.89 · tail 0.46

  // Brow base: mostly where her own brow bone / brow sits.
  // The real brow (found while erasing) is the most reliable guide: the new
  // brow head sits where her own brow head was, even for brows set very
  // close to the eyes. Landmarks are only a fallback.
  const idealBot = eyeY - 0.45 * eyeW;
  const measured = ownBrowBottom(face);
  const ownBot = measured != null ? measured : face.browInBotY + 0.06 * eyeW;
  const innerBotY = measured != null ? lerp(idealBot, ownBot, 0.9)
    : Math.abs(ownBot - idealBot) < 0.3 * eyeW ? lerp(idealBot, ownBot, 0.85) : idealBot;

  const thickness = 0.27 * eyeW * s.thick;
  const innerTopY = innerBotY - thickness;
  const archTopY = innerTopY - 0.22 * eyeW * s.lift;
  const archThick = thickness * 0.85;

  const ax = face.noseHalf, ay = face.nostrilY;

  // 1) start: up from the nostril edge, pulled in towards the bridge of the
  //    nose (never wider than the inner eye corner)
  const innerHalf = Math.min(ax, face.innerCornerHalf) * 0.8 * s.gap;
  // 3) tail (standard mapping): the straight line from the outer edge of the
  //    nostril through (just past) the outer corner of the eye, at tail height
  //    (tail sits a little above the head, never lower)
  const tailY = innerBotY - 0.06 * eyeW;
  // the line clears the eye corner by ~0.1 eye-widths: halfway between the
  // strict corner line (too short) and the reel's line (too long)
  const slope = (face.outerHalf + 0.105 * eyeW - ax) / Math.max(1, ay - face.outerY); // half-x gained per px climbed
  const tailHalfIdeal = ax + slope * (ay - tailY);
  let tailHalf = innerHalf + (tailHalfIdeal - innerHalf) * s.tail;
  // Brows set very low (close to the eye) and naturally long: draw them a
  // touch longer (up to +5%, never past where her own brow ended). All other
  // faces are untouched (weight 0 when the brow sits 0.21+ eye-widths up).
  {
    const low = (eyeY - ownBot) / eyeW;
    const wLow = measured != null ? smooth((0.21 - low) / 0.05) : 0;
    const reach = ownBrowOuterHalf(face);
    if (wLow > 0 && reach != null && reach > tailHalf) {
      tailHalf = Math.min(reach, tailHalf + (tailHalf - innerHalf) * 0.05 * wLow);
    }
  }
  if (window.__brow) window.__brow.metrics = {
    low: +((eyeY - ownBot) / eyeW).toFixed(3),
    tailHalf: +tailHalf.toFixed(2), tailHalfIdeal: +(innerHalf + (tailHalfIdeal - innerHalf) * s.tail).toFixed(2),
  };
  // 2) arch (standard mapping): straight above the outer edge of the iris,
  //    which lands about two-thirds along the brow
  const irisEdge = face.irisHalf + face.irisRad;
  const archHalf = clamp(irisEdge * s.arch, innerHalf + (tailHalf - innerHalf) * 0.5, innerHalf + (tailHalf - innerHalf) * 0.8);
  const archSlope = (archHalf - ax) / Math.max(1, ay - archTopY);

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
  // drawn hair strokes are only needed by the 'strokes' styles: build them lazily
  let strokes = null;
  Object.defineProperty(geo, 'strokes', { get: () => strokes || (strokes = makeStrokes(face, geo, s)) });
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


/* ------------------------------ colour matching ------------------------------
 * Makes the new brows the same colour as the brows that were erased: measure
 * the original brow hair (darker half of the brow pixels), render the new
 * brows off-screen, measure them the same way, adjust the tint, repeat.
 * --------------------------------------------------------------------------- */
// Colour matching compares the overall look of the two brows: the mean colour
// over each brow's footprint (hair plus the skin showing between hairs).
// Old brow: pixels inside the erase mask that the brow darkened by > 10 levels.
// New brow: pixels the new brow covers (fixed area, so the match converges).
function originalBrowColour() {
  const info = state.browInfo;
  if (!info || !state.clean) return null;
  const { mask, bx0, by0, bw, bh } = info;
  const a = state.orig.getContext('2d', { willReadFrequently: true }).getImageData(bx0, by0, bw, bh).data;
  const b = state.clean.getContext('2d', { willReadFrequently: true }).getImageData(bx0, by0, bw, bh).data;
  const px = [];
  for (let k = 0; k < bw * bh; k++) {
    if (!mask[k]) continue;
    const la = lum(a[k * 4], a[k * 4 + 1], a[k * 4 + 2]);
    if (lum(b[k * 4], b[k * 4 + 1], b[k * 4 + 2]) - la < 10) continue;
    px.push([a[k * 4], a[k * 4 + 1], a[k * 4 + 2], la]);
  }
  return colourStats(px);
}
// overall look (mean) and the hair itself (mean of the darkest quarter)
function colourStats(px) {
  if (px.length < 30) return null;
  let r = 0, g = 0, b = 0;
  for (const p of px) { r += p[0]; g += p[1]; b += p[2]; }
  const mean = [r / px.length, g / px.length, b / px.length];
  px.sort((p, q) => p[3] - q[3]);
  const m = Math.max(10, Math.floor(px.length * 0.25));
  r = g = b = 0;
  for (let i = 0; i < m; i++) { r += px[i][0]; g += px[i][1]; b += px[i][2]; }
  return { mean, hair: [r / m, g / m, b / m] };
}
function renderedBrowColour(hex) {
  const { W, H } = state;
  const SS = Math.min(2, Math.floor(4096 / Math.max(W, H)) || 1);
  const big = makeCanvas(W * SS, H * SS), bg = big.getContext('2d');
  bg.setTransform(SS, 0, 0, SS, 0, 0); faceTransform(bg); bg.imageSmoothingQuality = 'high';
  drawNaturalBrows(bg, state.brows, state.face, hex, 1, 1);
  const layer = makeCanvas(W, H), lg = layer.getContext('2d', { willReadFrequently: true });
  lg.imageSmoothingQuality = 'high';
  lg.drawImage(big, 0, 0, W, H);
  const comp = makeCanvas(W, H), cg = comp.getContext('2d', { willReadFrequently: true });
  cg.drawImage(state.clean, 0, 0);
  cg.drawImage(layer, 0, 0);
  let x0 = W, y0 = H, x1 = 0, y1 = 0;
  for (const side of state.brows.sides) for (const p0 of side.outline) {
    const q = faceToImage(p0);
    x0 = Math.min(x0, q.x); x1 = Math.max(x1, q.x); y0 = Math.min(y0, q.y); y1 = Math.max(y1, q.y);
  }
  const pad = state.face.eyeW * 0.25;
  x0 = Math.max(0, Math.floor(x0 - pad)); y0 = Math.max(0, Math.floor(y0 - pad));
  x1 = Math.min(W, Math.ceil(x1 + pad)); y1 = Math.min(H, Math.ceil(y1 + pad));
  const w = x1 - x0, h = y1 - y0;
  if (w <= 0 || h <= 0) return null;
  const al = lg.getImageData(x0, y0, w, h).data, c = cg.getImageData(x0, y0, w, h).data;
  const px = [];
  for (let k = 0; k < w * h; k++) {
    if (al[k * 4 + 3] < 64) continue;
    px.push([c[k * 4], c[k * 4 + 1], c[k * 4 + 2], lum(c[k * 4], c[k * 4 + 1], c[k * 4 + 2])]);
  }
  return colourStats(px);
}
function faceToImage(q) { // inverse of face.F
  const { C, angle } = state.face;
  const c = Math.cos(angle), s = Math.sin(angle);
  return pt(C.x + (q.x - C.x) * c - (q.y - C.y) * s, C.y + (q.x - C.x) * s + (q.y - C.y) * c);
}
// Finds the tint for which the new brows' `key` measurement equals the old one.
function solveTint(start, target, key) {
  let col = { ...start };
  for (let it = 0; it < 8; it++) {
    const got = renderedBrowColour(rgbToHex(col));
    if (!got) return null;
    const g = got[key], t = target[key];
    if (Math.abs(g[0] - t[0]) + Math.abs(g[1] - t[1]) + Math.abs(g[2] - t[2]) < 3) break;
    // the visible colour is a blend of hair and skin: step a bit further than the gap
    col = { r: clamp(col.r + (t[0] - g[0]) * 1.6, 0, 255), g: clamp(col.g + (t[1] - g[1]) * 1.6, 0, 255), b: clamp(col.b + (t[2] - g[2]) * 1.6, 0, 255) };
  }
  return col;
}
// The eye judges a brow both by its hairs and by its overall look: the hairs
// alone made dense new brows look too dark, the overall average made very dark
// brows look too light. The tint is the midpoint of the two exact matches.
function matchBrowColour() {
  const target = originalBrowColour();
  if (!target) return null;
  const start = hexToRgb(els.color.value);
  const byHair = solveTint(start, target, 'hair');
  const byLook = solveTint(start, target, 'mean');
  if (!byHair || !byLook) return null;
  const col = { r: (byHair.r + byLook.r) / 2, g: (byHair.g + byLook.g) / 2, b: (byHair.b + byLook.b) / 2 };
  const hex = rgbToHex(col);
  els.color.value = hex;
  state.browOpacity = 1;
  const fin = renderedBrowColour(hex);
  const R = (v) => v.map(Math.round);
  const res = { hex, target: { hair: R(target.hair), mean: R(target.mean) }, rendered: { hair: R(fin.hair), mean: R(fin.mean) } };
  if (window.__brow) window.__brow.colour = res;
  return res;
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
      [6.1, 'Arch: above the outer edge of the iris'],
      [7.2, 'Tail: nostril past the outer eye corner'],
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
const natLayer = { cv: null, key: null, color: '', reveal: -1 };

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

/* ------------------------------ procedural realistic brows ------------------------------
 * Every hair is generated from the structure of a brow, not copied from a photo:
 *  - roots: stratified (even but irregular) over the brow area, denser in the
 *    body, sparse and soft at the head, thinning towards the tail tip
 *  - direction: head hairs stand up and fan out; in the body the lower hairs
 *    rise and the upper hairs lie down along the brow (herringbone) and meet
 *    in a soft spine; tail hairs converge into the tip
 *  - length: proportional to the local brow height (shorter at the head)
 *  - shape: a gentle curve towards the tail and a taper from root to tip
 *  - tone: several layers (soft under-hairs, main hairs, fine dark hairs,
 *    a few light sheen hairs) so overlapping hairs build depth
 * Hairs are batched by layer/tone into a few Path2D fills: fast enough to
 * redraw on every slider step.
 * Seeds are made once per photo in normalised brow coordinates, so the same
 * hairs follow the brow when the shape sliders move.
 * --------------------------------------------------------------------------- */
function browEdges(side, geo, face, BINS = 64) {
  const span = geo.tailHalf - geo.innerHalf;
  const topB = new Float32Array(BINS + 1).fill(1e9), botB = new Float32Array(BINS + 1).fill(-1e9);
  const ol = side.outline;
  for (let i = 0; i < ol.length; i++) {
    const p0 = ol[i], p1 = ol[(i + 1) % ol.length];
    const steps = Math.max(1, Math.ceil(dist(p0, p1)));
    for (let t = 0; t <= steps; t++) {
      const px = lerp(p0.x, p1.x, t / steps), py = lerp(p0.y, p1.y, t / steps);
      const b = Math.round(clamp((Math.abs(px - face.midX) - geo.innerHalf) / span) * BINS);
      topB[b] = Math.min(topB[b], py); botB[b] = Math.max(botB[b], py);
    }
  }
  for (let b = 0; b <= BINS; b++) if (topB[b] > 1e8) {
    let l = b - 1, r = b + 1;
    while (l >= 0 && topB[l] > 1e8) l--;
    while (r <= BINS && topB[r] > 1e8) r++;
    const src = l >= 0 ? l : r;
    topB[b] = topB[src]; botB[b] = botB[src];
  }
  return { topB, botB, BINS };
}

// layers: [share, tone range, alpha, width factor, length factor]
const HAIR_LAYERS = [
  { name: 'under', share: 0.18, tone: [1.1, 1.4],  alpha: 0.22, w: 1.5,  len: 0.9 },
  { name: 'main',  share: 0.56, tone: [0.75, 1.05], alpha: 0.88, w: 1.0,  len: 1.0 },
  { name: 'fine',  share: 0.2,  tone: [0.5, 0.72],  alpha: 0.95, w: 0.65, len: 1.1 },
  { name: 'sheen', share: 0.06, tone: [1.6, 2.0],   alpha: 0.45, w: 0.7,  len: 0.95 },
];
const TONE_STEPS = 3;

let hairSeedCache = null;
function hairSeeds() {
  const key = state.browInfo || state.face;
  if (hairSeedCache && hairSeedCache.key === key) return hairSeedCache.sides;
  const sides = [0, 1].map((si) => {
    const rnd = mulberry32(9001 + si * 77);
    const list = [];
    const COLS = 150, ROWS = 26;
    for (let i = 0; i < COLS; i++) for (let j = 0; j < ROWS; j++) {
      const u = (i + rnd()) / COLS, v = (j + rnd()) / ROWS;
      // density: soft head, thinning tail tip, airy top and bottom edges
      let d = 0.25 + 0.75 * smooth(u / 0.2);
      d *= 1 - 0.75 * smooth((u - 0.86) / 0.14);
      d *= 1 - 0.55 * smooth((v - 0.82) / 0.18);
      d *= 1 - 0.35 * smooth((0.12 - v) / 0.12);
      if (rnd() > d * 0.34) continue;
      const r = rnd();
      let L = 0, acc = 0;
      for (; L < HAIR_LAYERS.length - 1; L++) { acc += HAIR_LAYERS[L].share; if (r < acc) break; }
      list.push({ u, v, L, tone: Math.min(TONE_STEPS - 1, (rnd() * TONE_STEPS) | 0), ang: (rnd() - 0.5), len: rnd(), bend: rnd(), w: rnd() });
    }
    // draw order: by position, so the brow grows in from head to tail
    list.sort((a, b) => a.u - b.u);
    return list;
  });
  hairSeedCache = { key, sides };
  return sides;
}

// builds the hair geometry for the current brow shape: per bucket, one Path2D
function buildHairPaths(geo, face, reveal) {
  const eyeW = face.eyeW;
  const span = geo.tailHalf - geo.innerHalf;
  const seeds = hairSeeds();
  const CH = 8; // chunks along the brow: many small paths rasterise faster than one huge one
  const buckets = HAIR_LAYERS.map(() => Array.from({ length: TONE_STEPS }, () => Array.from({ length: CH * 2 }, () => new Path2D())));
  geo.sides.forEach((side, si) => {
    const sg = side.sg;
    const { topB, botB, BINS } = browEdges(side, geo, face);
    const at = (arr, u) => { const f = clamp(u) * BINS, i = Math.min(BINS - 1, Math.floor(f)), t = f - i; return lerp(arr[i], arr[i + 1], t); };
    // centre-line slope (degrees, towards the tail)
    const axis = (u) => {
      const a = Math.max(0, u - 0.06), b = Math.min(1, u + 0.06);
      const ya = (at(topB, a) + at(botB, a)) / 2, yb = (at(topB, b) + at(botB, b)) / 2;
      return Math.atan2(-(yb - ya), (b - a) * span) * 180 / Math.PI;
    };
    const tipX = face.midX + sg * geo.tailHalf, tipY = (at(topB, 1) + at(botB, 1)) / 2;
    const archU = clamp((geo.archHalf - geo.innerHalf) / span, 0.3, 0.85);
    for (const h of seeds[si]) {
      if (h.u > reveal * 1.05) break;
      const top = at(topB, h.u), bot = at(botB, h.u), hh = Math.max(1, bot - top);
      const x = face.midX + sg * (geo.innerHalf + h.u * span);
      const y = bot - h.v * hh;
      const ax = axis(h.u);
      // direction
      let a;
      const headW = 1 - smooth((h.u - 0.04) / 0.16);
      const body = ax + lerp(30, -2, smooth((h.v - 0.1) / 0.8));
      a = lerp(body, 68 - h.v * 14 + ax * 0.3, headW);
      if (h.u > archU) { // tail: converge into the tip
        const toTip = Math.atan2(-(tipY - y), Math.abs(tipX - x) + 1e-3) * 180 / Math.PI;
        a = lerp(a, toTip, smooth((h.u - archU) / (1 - archU)) * 0.7);
      }
      a += h.ang * 7;
      const lay = HAIR_LAYERS[h.L];
      // length: tied to the local brow height
      let len = hh * lerp(1.0, 1.75, h.len) * lay.len;
      len *= lerp(0.45, 1, smooth(h.u / 0.22));              // short head hairs
      len = clamp(len, eyeW * 0.05, eyeW * 0.36);
      const rad = a * Math.PI / 180;
      const dx = sg * Math.cos(rad), dy = -Math.sin(rad);
      // groomed silhouette: tips stay close to the top / bottom edge
      if (dy < -0.05) len = Math.min(len, Math.max(hh * 0.25, (y - (top - hh * 0.12)) / -dy));
      else if (dy > 0.05) len = Math.min(len, Math.max(hh * 0.25, ((bot + hh * 0.08) - y) / dy));
      // curve towards the tail / downwards at the tip
      const bend = len * lerp(0.04, 0.16, h.bend);
      const nx = sg * Math.sin(rad), ny = Math.cos(rad);
      const cx = x + dx * len * 0.5 + nx * bend, cy = y + dy * len * 0.5 + ny * bend;
      const ex = x + dx * len + nx * bend * 1.6, ey = y + dy * len + ny * bend * 1.6;
      // tapered ribbon along the quadratic curve
      const w0 = Math.max(0.3, eyeW * 0.0048) * lay.w * lerp(0.75, 1.25, h.w) * lerp(0.7, 1, smooth(h.u / 0.15));
      const p = buckets[h.L][h.tone][si * CH + Math.min(CH - 1, (h.u * CH) | 0)];
      const N = 4, left = [], right = [];
      for (let k = 0; k <= N; k++) {
        const t = k / N, u1 = 1 - t;
        const px = u1 * u1 * x + 2 * u1 * t * cx + t * t * ex, py = u1 * u1 * y + 2 * u1 * t * cy + t * t * ey;
        const tx = 2 * u1 * (cx - x) + 2 * t * (ex - cx), ty = 2 * u1 * (cy - y) + 2 * t * (ey - cy);
        const tl = Math.hypot(tx, ty) || 1;
        const half = w0 * 0.5 * Math.pow(1 - t, 0.85);
        left.push(px - ty / tl * half, py + tx / tl * half);
        right.push(px + ty / tl * half, py - tx / tl * half);
      }
      p.moveTo(left[0], left[1]);
      for (let k = 2; k < left.length; k += 2) p.lineTo(left[k], left[k + 1]);
      for (let k = right.length - 2; k >= 0; k -= 2) p.lineTo(right[k], right[k + 1]);
    }
  });
  return buckets;
}

// draws both brows (face-frame transform already applied to g)
function drawNaturalBrows(g, geo, face, hex, reveal, opacity = state.browOpacity || 1) {
  const base = hexToRgb(hex);
  const buckets = buildHairPaths(geo, face, reveal);
  g.save();
  HAIR_LAYERS.forEach((lay, L) => {
    for (let t = 0; t < TONE_STEPS; t++) {
      const k = lerp(lay.tone[0], lay.tone[1], (t + 0.5) / TONE_STEPS);
      const add = lay.name === 'sheen' ? 30 : 0;
      g.fillStyle = `rgb(${clamp(base.r * k + add, 0, 255) | 0},${clamp(base.g * k + add, 0, 255) | 0},${clamp(base.b * k + add, 0, 255) | 0})`;
      g.globalAlpha = lay.alpha * opacity;
      for (const p of buckets[L][t]) g.fill(p);
    }
  });
  g.restore();
}

function drawBrows(c, progress) {
  if (progress <= 0) return;
  const s = settings();
  const geo = state.brows;
  const rgb = hexToRgb(s.color);
  const eyeW = state.face.eyeW;

  if (s.style === 'natural') {
    // photo-real: drawn only in a box around the brows, at 2x (1x while a
    // slider is being dragged), and redrawn only when something changed
    const { W, H } = state;
    const SS = state.dragging ? 1 : Math.min(2, Math.floor(4096 / Math.max(W, H)) || 1);
    const reveal = clamp(progress * 1.08);
    let box = natLayer.boxFor === geo ? natLayer.box : null;
    if (!box) {
      let x0 = W, y0 = H, x1 = 0, y1 = 0;
      for (const side of geo.sides) for (const q of side.outline) {
        const p = faceToImage(q);
        x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y);
      }
      const pad = eyeW * 0.35;
      x0 = Math.max(0, Math.floor(x0 - pad)); y0 = Math.max(0, Math.floor(y0 - pad));
      x1 = Math.min(W, Math.ceil(x1 + pad)); y1 = Math.min(H, Math.ceil(y1 + pad));
      box = { x0, y0, w: Math.max(1, x1 - x0), h: Math.max(1, y1 - y0) };
      natLayer.box = box; natLayer.boxFor = geo;
    }
    if (!natLayer.cv) natLayer.cv = makeCanvas(1, 1);
    if (natLayer.key !== geo || natLayer.color !== s.color || natLayer.reveal !== reveal || natLayer.op !== state.browOpacity || natLayer.ss !== SS) {
      const cv = natLayer.cv;
      if (cv.width !== box.w * SS || cv.height !== box.h * SS) { cv.width = box.w * SS; cv.height = box.h * SS; }
      const g = cv.getContext('2d');
      g.setTransform(1, 0, 0, 1, 0, 0);
      g.clearRect(0, 0, cv.width, cv.height);
      g.save();
      g.setTransform(SS, 0, 0, SS, -box.x0 * SS, -box.y0 * SS);
      faceTransform(g);
      g.imageSmoothingQuality = 'high';
      drawNaturalBrows(g, geo, state.face, s.color, reveal);
      g.restore();
      natLayer.key = geo; natLayer.color = s.color; natLayer.reveal = reveal; natLayer.op = state.browOpacity; natLayer.ss = SS;
    }
    c.save();
    c.setTransform(1, 0, 0, 1, 0, 0);
    if (SS === 1) c.imageSmoothingEnabled = false; // 1:1 copy, no filtering needed
    else c.imageSmoothingQuality = 'high';
    c.drawImage(natLayer.cv, box.x0, box.y0, box.w, box.h);
    c.restore();
    return;
  }

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

// c: target canvas context (the stage by default; an off-screen one when
// saving a video). opts.burnCaption draws the caption into the picture.
function render(t, c = ctx, opts = {}) {
  const { W, H, face } = state;
  const tl = state.timeline;
  c.setTransform(1, 0, 0, 1, 0, 0);
  c.clearRect(0, 0, W, H);

  if (state.showOriginal && c === ctx) {
    c.drawImage(state.orig, 0, 0);
    return;
  }

  // 1. brow removal: a scan sweeps down, revealing the cleaned photo
  const rp = smooth((t - tl.removeStart) / (tl.removeEnd - tl.removeStart));
  if (rp < 1) c.drawImage(state.orig, 0, 0);
  if (rp >= 1) c.drawImage(state.clean, 0, 0);
  else if (rp > 0) {
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
  for (const [ct, text] of tl.captions) if (t >= ct) cap = tr(text);
  if (opts.onCaption) opts.onCaption(cap);
  else if (opts.burnCaption) burnCaption(c, cap);
  else if (c === ctx && els.caption.textContent !== cap) els.caption.textContent = cap;
}

function burnCaption(c, text) {
  if (!text) return;
  const { W, H } = state;
  const size = Math.round(Math.max(14, Math.min(W, H) * 0.038));
  c.save();
  c.setTransform(1, 0, 0, 1, 0, 0);
  c.font = `700 ${size}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
  c.textAlign = 'center'; c.textBaseline = 'bottom';
  c.direction = getLang() === 'he' ? 'rtl' : 'ltr';
  c.shadowColor = 'rgba(0,0,0,0.6)'; c.shadowBlur = size * 0.5;
  c.fillStyle = '#fff';
  c.fillText(text, W / 2, H - size * 1.1);
  c.restore();
}

/* ------------------------------ erasing intro (saved video) ------------------------------
 * The first seconds of a saved video: the original photo, a scanning beam,
 * the face mesh twinkling in and particles rising while each brow fades away.
 * Fully deterministic (same video every time).
 * ----------------------------------------------------------------------------------------- */
// the mapping animation plays at this speed (lower = slower, more time to read each step)
const ANIM_SPEED = 0.62;
const INTRO = 5;
let introCache = null;
function introParticles() {
  if (introCache && introCache.key === state.browInfo) return introCache.list;
  const list = [];
  const b = state.browInfo, face = state.face;
  if (b) {
    const pts = [];
    for (let k = 0; k < b.mask.length; k += 3) if (b.mask[k]) pts.push([(k % b.bw) + b.bx0, ((k / b.bw) | 0) + b.by0]);
    const rnd = mulberry32(77), unit = face.eyeW;
    for (let i = 0; i < 700 && pts.length; i++) {
      const p = pts[(rnd() * pts.length) | 0];
      const left = p[0] < face.midX;
      const t0 = (left ? 1.1 : 2.1) + rnd() * 1.1;
      list.push({ x: p[0], y: p[1], t0, life: 0.6 + rnd() * 0.9, vx: (rnd() - 0.5) * unit * 0.25, vy: -unit * (0.25 + rnd() * 0.45), r: unit * (0.008 + rnd() * 0.016) });
    }
  }
  introCache = { key: state.browInfo, list };
  return list;
}
function renderIntro(c, t, opts = {}) {
  t = t * 4.2 / INTRO; // designed for 4.2 s, played over INTRO s
  const { W, H, face } = state;
  c.setTransform(1, 0, 0, 1, 0, 0);
  c.clearRect(0, 0, W, H);
  c.drawImage(state.orig, 0, 0);
  // each brow fades to the erased skin, one after the other
  const sides = [[1.2, 2.3, 0, face.midX], [2.2, 3.3, face.midX, W]];
  for (const [a0, a1, x0, x1] of sides) {
    const a = smooth((t - a0) / (a1 - a0));
    if (a <= 0) continue;
    c.save();
    c.beginPath(); c.rect(x0, 0, x1 - x0, H); c.clip();
    c.globalAlpha = a; c.drawImage(state.clean, 0, 0);
    c.restore();
  }
  const unit = face.eyeW;
  const fade = 1 - smooth((t - 3.6) / 0.6);   // effects fade out at the end
  c.fillStyle = `rgba(18,14,13,${(0.16 * fade).toFixed(3)})`;
  c.fillRect(0, 0, W, H);
  // scanning beam
  const y0 = Math.max(0, face.P[10].y - unit * 0.4), y1 = Math.min(H, face.P[152].y + unit * 0.2);
  const by = y0 + (y1 - y0) * (0.5 - 0.5 * Math.cos(t * Math.PI * 2 / 3.2));
  const bh = unit * 0.5;
  c.save();
  c.globalCompositeOperation = 'lighter';
  c.globalAlpha = fade;
  const g = c.createLinearGradient(0, by - bh, 0, by + bh);
  g.addColorStop(0, 'rgba(255,230,0,0)'); g.addColorStop(0.5, 'rgba(255,230,0,0.13)'); g.addColorStop(1, 'rgba(255,230,0,0)');
  c.fillStyle = g; c.fillRect(0, by - bh, W, bh * 2);
  c.fillStyle = 'rgba(255,240,150,0.35)'; c.fillRect(0, by - 0.75, W, 1.5);
  // face mesh
  const fin = smooth((t - 0.3) / 0.9) * fade, r = Math.max(0.8, unit * 0.011);
  for (let i = 0; i < 468; i += 3) {
    const p = face.P[i];
    const near = 1 - smooth(Math.abs(p.y - by) / (unit * 1.2));
    c.fillStyle = `rgba(255,248,210,${(fin * (0.18 + 0.22 * (0.5 + 0.5 * Math.sin(t * 2.6 + i * 1.7)) + 0.5 * near)).toFixed(3)})`;
    c.beginPath(); c.arc(p.x, p.y, r, 0, Math.PI * 2); c.fill();
  }
  // particles rising from the brows
  for (const q of introParticles()) {
    const k = (t - q.t0) / q.life;
    if (k <= 0 || k >= 1) continue;
    const a = Math.sin(k * Math.PI) * 0.85;
    const age = t - q.t0;
    c.fillStyle = k < 0.5 ? `rgba(255,230,0,${a.toFixed(3)})` : `rgba(255,${Math.round(230 - (k - 0.5) * 360)},${Math.round((k - 0.5) * 428)},${a.toFixed(3)})`;
    c.beginPath(); c.arc(q.x + q.vx * age, q.y + q.vy * age, q.r * (1 - k * 0.5), 0, Math.PI * 2); c.fill();
  }
  c.restore();
  const cap = tr(t < 1.0 ? 'Analysing the face' : 'Erasing the brows');
  if (opts.onCaption) opts.onCaption(cap); else burnCaption(c, cap);
}

/* ------------------------------ playback ------------------------------ */

function play(onDone) {
  state.playing = true;
  state.start = performance.now();
  els.replay.disabled = true;
  const tick = (now) => {
    if (!state.playing) return;
    state.t = (now - state.start) / 1000 * ANIM_SPEED;
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

// slider moves arrive much faster than frames: rebuild at most once per frame
let rebuildQueued = false;
function scheduleRebuild() {
  if (rebuildQueued) return;
  rebuildQueued = true;
  requestAnimationFrame(() => { rebuildQueued = false; rebuild(); });
}

// image-space box around both brows (with margin)
function browsBox(geo, pad) {
  const { W, H } = state;
  let x0 = W, y0 = H, x1 = 0, y1 = 0;
  for (const side of geo.sides) for (const q of side.outline) {
    const p = faceToImage(q);
    x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y);
  }
  x0 = Math.max(0, Math.floor(x0 - pad)); y0 = Math.max(0, Math.floor(y0 - pad));
  x1 = Math.min(W, Math.ceil(x1 + pad)); y1 = Math.min(H, Math.ceil(y1 + pad));
  return { x0, y0, x1, y1 };
}

function rebuild() {
  if (!state.face) return;
  const prev = state.brows;
  state.brows = mapBrows(state.face, settings());
  state.timeline = buildTimeline(state.face, state.brows);
  if (!state.playing) {
    const atEnd = state.t >= state.timeline.end - 1e-6;
    if (atEnd && prev && !state.showOriginal) {
      // final frame only changes around the brows: repaint just that area
      const pad = state.face.eyeW * 0.6;
      const a = browsBox(prev, pad), b = browsBox(state.brows, pad);
      const x0 = Math.min(a.x0, b.x0), y0 = Math.min(a.y0, b.y0), x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
      ctx.save();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.beginPath(); ctx.rect(x0, y0, x1 - x0, y1 - y0); ctx.clip();
      ctx.drawImage(state.clean, x0, y0, x1 - x0, y1 - y0, x0, y0, x1 - x0, y1 - y0);
      drawBrows(ctx, 1);
      ctx.restore();
    } else render(state.timeline.end);
  }
  scheduleMeasurements();
}
// the measurement table is not needed on every slider frame
let measureTimer = 0;
function scheduleMeasurements() {
  clearTimeout(measureTimer);
  measureTimer = setTimeout(showMeasurements, state.dragging ? 150 : 0);
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
  els.measureList.innerHTML = rows.map(([k, v]) => `<dt>${tr(k)}</dt><dd dir="ltr">${v}</dd>`).join('');
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
  busy.textKey = text;
  busyEls.text.textContent = tr(text);
}
function busySetImage(canvas) {
  // if a fade is still running, freeze what is on screen right now as the new
  // starting point, so the picture never jumps back to an older frame
  if (busy.next && busy.base) {
    const a = smooth((performance.now() - busy.fadeStart) / 700);
    const mid = makeCanvas(state.W, state.H), g = mid.getContext('2d');
    g.drawImage(busy.base, 0, 0);
    g.globalAlpha = a; g.drawImage(busy.next, 0, 0);
    busy.base = mid;
  }
  busy.next = canvas; busy.fadeStart = performance.now();
}
function busySetFace(face) { busy.face = face; busy.faceT = performance.now(); }
function busySetBrow(brow, face) {
  const pts = [];
  const { mask, bx0, by0, bw, bh } = brow;
  for (let k = 0; k < mask.length; k += 3) if (mask[k]) pts.push([(k % bw) + bx0, ((k / bw) | 0) + by0]);
  busy.spawn = pts; busy.midX = face.midX;
}
function busyFinish(alive = () => true) {
  return new Promise((resolve) => {
    busyStage('Done', busy.shown, 1, 250);
    busy.to = 1; busy.from = 1;
    setTimeout(() => {
      if (!alive()) { resolve(); return; }
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
    const msg = tr(busy.subs[busy.subIdx]);
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

let loadRun = 0;
async function loadFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  // a newer photo cancels any run still in progress
  const myRun = ++loadRun;
  const alive = () => myRun === loadRun;
  state.playing = false;
  state.browInfo = null;
  state.browOpacity = 1;
  showAiNotice('');
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
  if (!alive()) return;
  const scale = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
  const W = Math.round(bmp.width * scale), H = Math.round(bmp.height * scale);
  state.W = W; state.H = H;
  els.stage.width = W; els.stage.height = H;
  requestAnimationFrame(placeStageTools);
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
    if (!alive()) return;
    busyStage('Finding her face', 0.1, 0.18, 500, ['Locating 478 face points']);
    await new Promise((r) => setTimeout(r, 60));
    if (!alive()) return;
    const res = lmk.detect(orig);
    if (!res.faceLandmarks || !res.faceLandmarks.length) {
      busyFail();
      setStatus(tr('No face found. Try a well-lit, front-facing photo.'), true);
      document.documentElement.dataset.browState = 'error';
      return;
    }
    state.face = buildFace(res.faceLandmarks[0], W, H);
    busySetFace(state.face);
    busyStage('Mapping brow hairs', 0.18, 0.26, 600, ['Tracing every hair', 'Protecting lids and eyeshadow']);
    await new Promise((r) => setTimeout(r, 450));
    if (!alive()) return;
    const face = state.face;
    const { canvas, hair, brow } = await removeBrows(orig, face);
    if (!alive()) return;
    state.clean = canvas;
    busySetBrow(brow, state.face);
    state.browInfo = brow;
    // AI repaint (better skin, shading and texture); the classic fill stays as fallback
    let aiFailed = false;
    const sideMs = busyTimes.side;
    try {
      let from = 0.26;
      if (!miganLoad.ready) { // first visit: show the real download progress
        busyStage('Downloading the skin AI', 0.26, 0.42, 1000, ['One-time download (~28 MB)', 'Saved on this device for next time']);
        busy.dl = true;
        await migan;
        if (!alive()) return;
        busy.dl = false;
        from = 0.42;
      }
      busyStage('Erasing brow 1 of 2', from, 0.58, sideMs + 800, ['Rebuilding the skin underneath', 'Matching skin tone', 'Recreating pores and texture']);
      busy.side = 0;
      await migan;
      if (!alive()) return;
      const t0 = performance.now();
      let tSide = t0;
      const aiResult = await aiRemoveBrows(orig, brow, face, (step, side, snap) => {
        if (!alive()) return;
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
      }, alive);
      if (!alive()) return;
      state.clean = aiResult;
      if (window.__brow) window.__brow.aiMs = Math.round(performance.now() - t0);
    } catch (e) {
      if (!alive() || e instanceof AbortError) return;
      console.warn('AI brow removal unavailable, using classic fill', e);
      if (window.__brow) window.__brow.aiError = String(e && e.message || e);
      aiFailed = true;
    }
    busy.side = -1;
    busySetImage(state.clean);
    await new Promise((r) => setTimeout(r, 650));
    if (!alive()) return;
    if (window.__brow) window.__brow.maxGap = Math.round(busy.maxGap);
    await busyFinish(alive);
    if (!alive()) return;
    if (aiFailed) showAiNotice(tr('AI erase failed, so a basic erase was used. Reason: ') + (window.__brow && window.__brow.aiError || 'unknown') + tr('. Open this page with ?diag=1 at the end of the address for a full check.'));
    if (hair) els.color.value = hair;
    syncOutputs();
    if (!alive()) return;
    state.brows = mapBrows(state.face, settings());
    // new brows in exactly the colour of the brows that were erased
    try { matchBrowColour(); } catch (e) { console.warn('colour match skipped', e); }
    state.timeline = buildTimeline(state.face, state.brows);
    play();
  } catch (err) {
    if (!alive()) return;
    console.error(err);
    busyFail();
    setStatus(tr('Something went wrong: ') + (err && err.message ? err.message : err), true);
    document.documentElement.dataset.browState = 'error';
  }
}


/* the Replay / original buttons sit on the photo's top-right corner, wherever it is drawn */
function placeStageTools() {
  const tools = document.querySelector('.stage-tools'), wrap = document.querySelector('.stage-wrap');
  if (!tools || !wrap) return;
  const b = els.stage.getBoundingClientRect(), w = wrap.getBoundingClientRect();
  // the picture is letterboxed inside the canvas box (object-fit: contain)
  const iw = els.stage.width || 1, ih = els.stage.height || 1;
  const k = Math.min(b.width / iw, b.height / ih), pw = iw * k, ph = ih * k;
  const s = { top: b.top + (b.height - ph) / 2, right: b.right - (b.width - pw) / 2, width: pw };
  tools.style.top = (s.top - w.top + 10) + 'px';
  tools.style.right = (w.right - s.right + 10) + 'px';
  tools.style.maxWidth = Math.max(120, s.width - 20) + 'px';
}
if (typeof ResizeObserver !== 'undefined') new ResizeObserver(placeStageTools).observe(els.stage);
addEventListener('resize', placeStageTools);

/* ------------------------------ AI notice + diagnostics ------------------------------ */
function showAiNotice(text) {
  let el = document.getElementById('aiNotice');
  if (!el) {
    el = document.createElement('p');
    el.id = 'aiNotice';
    el.style.cssText = 'margin:10px 0 0;padding:10px 12px;border-radius:10px;background:rgba(160,30,40,.85);color:#fff;font-size:13px;line-height:1.4;user-select:text';
    document.querySelector('.stage-wrap').after(el);
  }
  el.textContent = text;
  el.hidden = !text;
}

// ?diag=1 : checks every file the AI needs and tries each AI mode, in this browser
async function runDiagnostics() {
  const box = document.createElement('pre');
  box.style.cssText = 'position:fixed;inset:12px;z-index:99;overflow:auto;margin:0;padding:14px;border-radius:12px;background:#111;color:#e8e8e8;font:12px/1.5 ui-monospace,monospace;white-space:pre-wrap';
  document.body.appendChild(box);
  const log = (t) => { box.textContent += t + '\n'; };
  log('Brow app diagnostics — screenshot this and send it\n');
  log('browser: ' + navigator.userAgent);
  log('crossOriginIsolated: ' + self.crossOriginIsolated + ' | cores: ' + navigator.hardwareConcurrency + ' | memory(GB): ' + (navigator.deviceMemory || '?'));
  log('OffscreenCanvas: ' + (typeof OffscreenCanvas !== 'undefined') + ' | SharedArrayBuffer: ' + (typeof SharedArrayBuffer !== 'undefined'));
  try { log('safe-mode flag: ' + localStorage.getItem('brow-ai-safe')); } catch { log('safe-mode flag: (storage blocked)'); }
  log('\nfiles:');
  const files = ['app.js', 'ai-worker.js', 'removal-worker.js', 'removal.js', 'vendor/ort/ort.wasm.min.mjs', 'vendor/ort/ort-wasm-simd-threaded.mjs',
    'vendor/ort/ort-wasm-simd-threaded.wasm', 'vendor/ort118/ort.wasm.min.js', 'vendor/ort118/ort-wasm-simd.wasm', 'models/migan.part0', 'models/migan.part1', 'models/face_landmarker.task'];
  for (const f of files) {
    try {
      const r = await fetch(new URL('./' + f, import.meta.url), { cache: 'no-store' });
      const b = await r.arrayBuffer();
      log(`  ${f}: ${r.status} ${r.headers.get('content-type') || '-'} ${b.byteLength} bytes` + (r.headers.get('cross-origin-embedder-policy') ? ' coep' : ''));
    } catch (e) { log(`  ${f}: FAILED ${e.message || e}`); }
  }
  for (const [engine, threads] of [['fast', Math.min(4, navigator.hardwareConcurrency || 1)], ['light', 1]]) {
    log(`\nAI test, ${engine} engine, ${threads} thread(s):`);
    const t0 = performance.now();
    try {
      const m = await startAiWorker(threads, engine);
      log(`  started in ${Math.round(performance.now() - t0)} ms`);
      const n = 64 * 64, chw = new Uint8Array(3 * n).fill(170), known = new Uint8Array(n).fill(255);
      for (let y = 24; y < 40; y++) for (let x = 20; x < 44; x++) known[y * 64 + x] = 0;
      const t1 = performance.now();
      await m.run(chw, known, 64, 64);
      log(`  test image OK in ${Math.round(performance.now() - t1)} ms`);
    } catch (e) { log('  FAILED: ' + (e.message || e)); }
  }
  log('\ndone.');
}
if (/[?&]diag=1/.test(location.search)) runDiagnostics();

/* ------------------------------ export ------------------------------ */

function savePng() {
  const { W, H } = state;
  const rc = document.createElement('canvas'); rc.width = W; rc.height = H;
  render(state.timeline.end, rc.getContext('2d'), { onCaption: () => {} });
  const [OW, OH] = exportSize();
  const comp = makeComposer(OW, OH);
  comp.frame(rc, '');
  comp.canvas.toBlob((blob) => download(blob, `brow-mapping-${els.format.value.replace(':', 'x')}.png`), 'image/png');
}

function download(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
}

/* ------------------------------ export (video + image) ------------------------------
 * The animation is rendered at photo size, then composed onto an output frame
 * in the chosen format (9:16 Reels/TikTok, 16:9, 1:1 or original): the photo is
 * fitted over a blurred copy of itself, captions are large and inside the
 * platform's safe area, the optional logo sits in a corner, and the video ends
 * with a before/after split.
 * ------------------------------------------------------------------------------------- */
const EXPORT_FORMATS = { '9:16': [1080, 1920], '16:9': [1920, 1080], '1:1': [1080, 1080] };
const exportLogo = { img: null };
const OUTRO = 3.2;

function exportSize() {
  const f = EXPORT_FORMATS[els.format.value];
  if (f) return f;
  // original: photo aspect, long side 1080..1920
  const { W, H } = state, k = Math.min(1920 / Math.max(W, H), Math.max(1, 1080 / Math.min(W, H)));
  return [Math.round(W * k / 2) * 2, Math.round(H * k / 2) * 2];
}

function makeComposer(OW, OH) {
  const out = document.createElement('canvas');
  out.width = OW; out.height = OH;
  const o = out.getContext('2d');
  const { W, H } = state;
  const k = Math.min(OW / W, OH / H), dw = W * k, dh = H * k;
  const dx = (OW - dw) / 2, dy = (OH - dh) / 2;
  const vertical0 = OH > OW * 1.2;
  // when the photo does not fill the frame it is shown as a card on a
  // designed backdrop (built once): a deep gradient in the photo's own tones,
  // a soft glow behind the card, a vignette and fine grain
  const letterbox = true;
  let bg = null, card = { x: dx, y: dy, w: dw, h: dh, r: 0 };
  if (letterbox) {
    // the card never covers the caption: a slot above it (logo / labels) and
    // below it (captions) are always kept free
    const inset = Math.min(OW, OH) * 0.05;
    const topSlot = vertical0 ? OH * 0.11 : OH * 0.1, botSlot = vertical0 ? OH * 0.29 : OH * 0.17;
    const availW = OW - inset * 2, availH = OH - topSlot - botSlot;
    const k2 = Math.min(availW / W, availH / H);
    card = { w: W * k2, h: H * k2, r: Math.min(OW, OH) * 0.035 };
    card.x = (OW - card.w) / 2;
    card.y = topSlot + (availH - card.h) / 2;
    // average colour of the photo
    const t = document.createElement('canvas'); t.width = 16; t.height = 16;
    const tg = t.getContext('2d', { willReadFrequently: true }); tg.drawImage(state.clean, 0, 0, 16, 16);
    const px = tg.getImageData(0, 0, 16, 16).data;
    let r = 0, g = 0, b = 0;
    for (let k = 0; k < px.length; k += 4) { r += px[k]; g += px[k + 1]; b += px[k + 2]; }
    const n = px.length / 4; r /= n; g /= n; b /= n;
    const tone = (m, add = 0) => `rgb(${clamp(r * m + add, 0, 255) | 0},${clamp(g * m + add, 0, 255) | 0},${clamp(b * m + add, 0, 255) | 0})`;
    bg = document.createElement('canvas'); bg.width = OW; bg.height = OH;
    const c2 = bg.getContext('2d');
    const lin = c2.createLinearGradient(0, 0, OW * 0.3, OH);
    lin.addColorStop(0, tone(0.34, 6)); lin.addColorStop(0.55, tone(0.2, 4)); lin.addColorStop(1, tone(0.1, 2));
    c2.fillStyle = lin; c2.fillRect(0, 0, OW, OH);
    const cx = OW / 2, cy = card.y + card.h / 2;
    const glow = c2.createRadialGradient(cx, cy, 0, cx, cy, Math.max(card.w, card.h) * 0.9);
    glow.addColorStop(0, tone(0.75, 30).replace('rgb', 'rgba').replace(')', ',0.55)'));
    glow.addColorStop(1, 'rgba(0,0,0,0)');
    c2.fillStyle = glow; c2.fillRect(0, 0, OW, OH);
    const vig = c2.createRadialGradient(OW / 2, OH / 2, Math.min(OW, OH) * 0.35, OW / 2, OH / 2, Math.hypot(OW, OH) * 0.62);
    vig.addColorStop(0, 'rgba(0,0,0,0)'); vig.addColorStop(1, 'rgba(0,0,0,0.55)');
    c2.fillStyle = vig; c2.fillRect(0, 0, OW, OH);
    // fine grain (avoids colour banding in the video)
    const gr = c2.getImageData(0, 0, OW, OH), gd = gr.data, rnd = mulberry32(5);
    for (let k = 0; k < gd.length; k += 4) { const v = (rnd() - 0.5) * 9; gd[k] += v; gd[k + 1] += v; gd[k + 2] += v; }
    c2.putImageData(gr, 0, 0);
    // card shadow
    c2.save();
    c2.shadowColor = 'rgba(0,0,0,0.55)'; c2.shadowBlur = Math.min(OW, OH) * 0.05; c2.shadowOffsetY = Math.min(OW, OH) * 0.015;
    c2.fillStyle = '#000';
    c2.beginPath(); c2.roundRect ? c2.roundRect(card.x, card.y, card.w, card.h, card.r) : c2.rect(card.x, card.y, card.w, card.h); c2.fill();
    c2.restore();
  }
  const vertical = OH > OW * 1.2;
  function caption(text) {
    if (!text) return;
    const size = Math.round(Math.min(OW, OH) * (vertical ? 0.048 : 0.042));
    o.save();
    o.font = `700 ${size}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
    o.direction = getLang() === 'he' ? 'rtl' : 'ltr';
    o.textAlign = 'center'; o.textBaseline = 'middle';
    // Reels/TikTok cover the bottom ~20% with their own buttons: stay above it
    // always in the free slot under the card (never over the face)
    const y = card.y + card.h + (OH - card.y - card.h) * (vertical ? 0.33 : 0.5);
    const words = text.split(' '), lines = [];
    let line = '';
    for (const w of words) {
      const test = line ? line + ' ' + w : w;
      if (o.measureText(test).width > OW * 0.84 && line) { lines.push(line); line = w; } else line = test;
    }
    lines.push(line);
    const lh = size * 1.25, padX = size * 0.7, padY = size * 0.45;
    const boxW = Math.max(...lines.map((l) => o.measureText(l).width)) + padX * 2;
    const boxH = lines.length * lh + padY * 2;
    const top = y - boxH / 2;
    o.fillStyle = 'rgba(15,12,11,0.62)';
    o.beginPath();
    o.roundRect ? o.roundRect(OW / 2 - boxW / 2, top, boxW, boxH, size * 0.5) : o.rect(OW / 2 - boxW / 2, top, boxW, boxH);
    o.fill();
    o.fillStyle = '#fff';
    lines.forEach((l, i) => o.fillText(l, OW / 2, top + padY + lh * (i + 0.5)));
    o.restore();
  }
  function logo(atBottom) {
    const img = exportLogo.img;
    if (!img) return;
    const slot = atBottom ? OH - card.y - card.h : card.y;
    const maxW = OW * (vertical ? 0.34 : 0.2), maxH = slot * 0.62;
    const k3 = Math.min(maxW / img.width, maxH / img.height);
    const w = img.width * k3, h = img.height * k3;
    const y = atBottom ? card.y + card.h + (slot - h) * (vertical ? 0.33 : 0.5) : (slot - h) / 2;
    o.save();
    o.globalAlpha = 0.95;
    o.shadowColor = 'rgba(0,0,0,0.35)'; o.shadowBlur = Math.min(OW, OH) * 0.015;
    o.drawImage(img, (OW - w) / 2, y, w, h);
    o.restore();
  }
  // Before / After labels in the top slot, each centred over its half
  function labels(split) {
    const size = Math.round(Math.min(OW, OH) * (vertical ? 0.06 : 0.05));
    o.save();
    o.globalAlpha = split.alpha;
    o.font = `800 ${size}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
    o.textAlign = 'center'; o.textBaseline = 'middle';
    const xm = card.x + split.x * (card.w / state.W);
    const y = card.y / 2;
    for (const [text, cx] of [[tr('Before'), (card.x + xm) / 2], [tr('After'), (xm + card.x + card.w) / 2]]) {
      const tw = o.measureText(text).width, pw = tw + size * 1.2, ph = size * 1.6;
      o.fillStyle = 'rgba(15,12,11,0.7)';
      o.beginPath(); o.roundRect ? o.roundRect(cx - pw / 2, y - ph / 2, pw, ph, ph / 2) : o.rect(cx - pw / 2, y - ph / 2, pw, ph); o.fill();
      o.fillStyle = '#fff'; o.fillText(text, cx, y + size * 0.04);
    }
    o.restore();
  }
  return {
    canvas: out,
    frame(src, cap, split) {
      o.setTransform(1, 0, 0, 1, 0, 0);
      o.imageSmoothingQuality = 'high';
      if (bg) {
        o.drawImage(bg, 0, 0);
        o.save();
        o.beginPath(); o.roundRect ? o.roundRect(card.x, card.y, card.w, card.h, card.r) : o.rect(card.x, card.y, card.w, card.h); o.clip();
        o.drawImage(src, card.x, card.y, card.w, card.h);
        o.restore();
        // thin light edge around the card
        o.save(); o.strokeStyle = 'rgba(255,255,255,0.14)'; o.lineWidth = Math.max(1, Math.min(OW, OH) * 0.0015);
        o.beginPath(); o.roundRect ? o.roundRect(card.x, card.y, card.w, card.h, card.r) : o.rect(card.x, card.y, card.w, card.h); o.stroke(); o.restore();
      } else o.drawImage(src, dx, dy, dw, dh);
      caption(cap);
      logo(!!split);
      if (split) labels(split);
    },
  };
}

// before | after split on the work canvas (t: seconds into the outro).
// The split follows the face's own vertical axis through the nose (tilted
// with the head), and the labels sit at the bottom so a logo never covers them.
function renderOutro(c, t) {
  const { W, H, face } = state;
  render(state.timeline.end, c, { onCaption: () => {} });
  // face axis: through the bridge (168) and the nose tip (1), perpendicular to the eye line
  const top = face.P[168], tip = face.P[1];
  const ang = face.angle; // head roll
  const ux = -Math.sin(ang), uy = Math.cos(ang); // face "down" direction
  const cx = (top.x + tip.x) / 2, cy = (top.y + tip.y) / 2;
  const xAt = (y) => cx + (y - cy) * ux / uy;   // x of the axis at height y
  const shift = lerp(W * 1.2, 0, smooth(t / 0.9)); // wipe in from the right
  const xTop = xAt(0) + shift, xBot = xAt(H) + shift;
  c.save();
  c.setTransform(1, 0, 0, 1, 0, 0);
  c.beginPath(); c.moveTo(0, 0); c.lineTo(xTop, 0); c.lineTo(xBot, H); c.lineTo(0, H); c.closePath(); c.clip();
  c.drawImage(state.orig, 0, 0);
  c.restore();
  c.save();
  c.strokeStyle = '#fff'; c.lineWidth = Math.max(3, W * 0.006);
  c.shadowColor = 'rgba(0,0,0,0.4)'; c.shadowBlur = c.lineWidth * 2;
  c.beginPath(); c.moveTo(xTop, 0); c.lineTo(xBot, H); c.stroke();
  c.restore();
  return { x: (xTop + xBot) / 2, alpha: smooth((t - 0.8) / 0.4) };
}

// Records the whole story (erasing intro + mapping + before/after) off-screen:
// the picture on screen does not replay, the button shows progress.
let recording = false;
function saveVideo() {
  if (recording) return;
  const types = ['video/mp4;codecs=avc1', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm'];
  const type = typeof MediaRecorder !== 'undefined' && types.find((t) => MediaRecorder.isTypeSupported(t));
  if (!type) { setStatus(tr('Video recording is not supported in this browser.'), true); return; }
  const { W, H } = state;
  const rc = document.createElement('canvas');
  rc.width = W; rc.height = H;
  const rctx = rc.getContext('2d');
  const [OW, OH] = exportSize();
  const comp = makeComposer(OW, OH);
  let cap = '';
  const onCaption = (txt) => { cap = txt; };
  renderIntro(rctx, 0, { onCaption }); comp.frame(rc, cap);
  const stream = comp.canvas.captureStream(30);
  const rec = new MediaRecorder(stream, { mimeType: type, videoBitsPerSecond: 10_000_000 });
  const chunks = [];
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  const labelEl = $('saveVidLabel');
  const main = state.timeline.end / ANIM_SPEED; // seconds of video
  const total = INTRO + main + 0.6 + OUTRO;
  recording = true;
  els.saveVid.disabled = true;
  rec.onstop = () => {
    recording = false;
    els.saveVid.disabled = false;
    labelEl.textContent = tr('Save animation');
    download(new Blob(chunks, { type }), `brow-mapping-${els.format.value.replace(':', 'x')}.${type.includes('mp4') ? 'mp4' : 'webm'}`);
  };
  rec.start();
  const t0 = performance.now();
  const frame = (now) => {
    const t = (now - t0) / 1000;
    cap = '';
    if (t < INTRO) renderIntro(rctx, t, { onCaption });
    else if (t < INTRO + main + 0.6) render(Math.min((t - INTRO) * ANIM_SPEED, state.timeline.end), rctx, { onCaption });
    let split = null;
    if (t >= INTRO + main + 0.6) { split = renderOutro(rctx, t - INTRO - main - 0.6); cap = ''; }
    comp.frame(rc, cap, split);
    labelEl.textContent = `${tr('Saving…')} ${Math.min(99, Math.round(t / total * 100))}%`;
    if (t < total) requestAnimationFrame(frame);
    else rec.stop();
  };
  requestAnimationFrame(frame);
}

/* ------------------------------ UI wiring ------------------------------ */

// export settings: format and optional logo (remembered on this device)
try { const f = localStorage.getItem('brow-format'); if (f && [...els.format.options].some((o) => o.value === f)) els.format.value = f; } catch {}
els.format.addEventListener('change', () => { try { localStorage.setItem('brow-format', els.format.value); } catch {} });
function setLogo(img, dataUrl) {
  exportLogo.img = img;
  $('logoPreview').hidden = !img;
  $('logoRemove').hidden = !img;
  if (img) $('logoPreview').src = dataUrl;
  try { if (dataUrl) localStorage.setItem('brow-logo', dataUrl); else localStorage.removeItem('brow-logo'); } catch {}
}
$('logoFile').addEventListener('change', () => {
  const f = $('logoFile').files[0];
  if (!f) return;
  const r = new FileReader();
  r.onload = () => { const i = new Image(); i.onload = () => setLogo(i, r.result); i.src = r.result; };
  r.readAsDataURL(f);
});
$('logoRemove').addEventListener('click', () => { $('logoFile').value = ''; setLogo(null, null); });
try { const d = localStorage.getItem('brow-logo'); if (d) { const i = new Image(); i.onload = () => setLogo(i, d); i.src = d; } } catch {}

// language
$('lang').addEventListener('click', () => setLang(getLang() === 'he' ? 'en' : 'he'));
onLangChange(() => {
  if (busy.on && busy.textKey) busyEls.text.textContent = tr(busy.textKey);
  if (state.brows && !els.measure.hidden) showMeasurements();
  if (state.timeline && !state.playing && !busy.on) render(state.t);
  requestAnimationFrame(placeStageTools);
});

// phone controls: one control at a time, picked from the chip row
const chips = [...document.querySelectorAll('#chips .chip')];
function selectChip(key) {
  for (const c of chips) c.classList.toggle('on', c.dataset.for === key);
  for (const el of document.querySelectorAll('#panel .ctl')) el.classList.toggle('on', el.dataset.ctl === key);
  els.studio.dataset.ctl = key;
}
for (const c of chips) c.addEventListener('click', () => { selectChip(c.dataset.for); c.scrollIntoView({ block: 'nearest', inline: 'center', behavior: 'smooth' }); });
selectChip('thick');
// keep the photo sized to the space left above the control dock
if (typeof ResizeObserver !== 'undefined') new ResizeObserver(() => {
  document.documentElement.style.setProperty('--dock-h', $('panel').offsetHeight + 'px');
  placeStageTools();
}).observe($('panel'));
applyLang();

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
  el.addEventListener('input', () => { syncOutputs(); scheduleRebuild(); }));
for (const el of [els.thick, els.lift, els.arch, els.tail, els.gap]) {
  el.addEventListener('pointerdown', () => { state.dragging = true; });
  const end = () => { if (!state.dragging) return; state.dragging = false; scheduleRebuild(); };
  el.addEventListener('pointerup', end); el.addEventListener('pointercancel', end); el.addEventListener('change', end);
}

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
