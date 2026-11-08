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
  thick: $('thick'), lift: $('lift'), arch: $('arch'), tail: $('tail'),
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

function removeBrows(src, face) {
  const { W, H } = state;
  const { P, eyeW } = face;
  const mask = makeCanvas(W, H);
  const m = mask.getContext('2d');
  m.fillStyle = '#fff'; m.strokeStyle = '#fff';
  m.lineJoin = 'round'; m.lineCap = 'round';
  m.lineWidth = eyeW * 0.3;
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

  const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
  if (bw <= 0 || bh <= 0) return { canvas: src, hair: null };
  const acc = new Float32Array(bw * bh * 3);
  const wsum = new Float32Array(bw * bh);

  const addRun = (coords, a, b, len) => {
    const w = 1 / (len * len);
    for (let i = 0; i < coords.length; i++) {
      const t = (i + 1) / (coords.length + 1);
      const [x, y] = coords[i];
      const j = (y - y0) * bw + (x - x0);
      const ca = a || b, cb = b || a;
      acc[j * 3] += lerp(ca[0], cb[0], t) * w;
      acc[j * 3 + 1] += lerp(ca[1], cb[1], t) * w;
      acc[j * 3 + 2] += lerp(ca[2], cb[2], t) * w;
      wsum[j] += w;
    }
  };

  // Vertical runs (dominant: short, skin above + below).
  for (let x = x0; x <= x1; x++) {
    let y = y0;
    while (y <= y1) {
      if (!inMask(x, y)) { y++; continue; }
      const start = y; const coords = [];
      while (y <= y1 && inMask(x, y)) { coords.push([x, y]); y++; }
      const a = sample(x, start, 0, -1), b = sample(x, y - 1, 0, 1);
      if (a || b) addRun(coords, a, b, coords.length);
    }
  }
  // Horizontal runs (smooth along the brow).
  for (let y = y0; y <= y1; y++) {
    let x = x0;
    while (x <= x1) {
      if (!inMask(x, y)) { x++; continue; }
      const start = x; const coords = [];
      while (x <= x1 && inMask(x, y)) { coords.push([x, y]); x++; }
      const a = sample(start, y, -1, 0), b = sample(x - 1, y, 1, 0);
      if (a || b) addRun(coords, a, b, coords.length * 1.6);
    }
  }

  // Normalise + light box blur to remove streaks.
  const fill = new Float32Array(bw * bh * 3);
  for (let j = 0; j < bw * bh; j++) {
    if (wsum[j] > 0) for (let c = 0; c < 3; c++) fill[j * 3 + c] = acc[j * 3 + c] / wsum[j];
  }
  const blurPass = (horizontal, r) => {
    const out = new Float32Array(fill.length);
    for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
      const j = y * bw + x;
      if (!wsum[j]) continue;
      let s0 = 0, s1 = 0, s2 = 0, n = 0;
      for (let o = -r; o <= r; o++) {
        const xx = horizontal ? x + o : x, yy = horizontal ? y : y + o;
        if (xx < 0 || yy < 0 || xx >= bw || yy >= bh) continue;
        const jj = yy * bw + xx;
        if (!wsum[jj]) continue;
        s0 += fill[jj * 3]; s1 += fill[jj * 3 + 1]; s2 += fill[jj * 3 + 2]; n++;
      }
      out[j * 3] = s0 / n; out[j * 3 + 1] = s1 / n; out[j * 3 + 2] = s2 / n;
    }
    fill.set(out);
  };
  const r = Math.max(2, Math.round(eyeW * 0.03));
  blurPass(true, r); blurPass(false, r); blurPass(true, r);

  // Skin grain: measure fine texture on the forehead just above the mask.
  let grain = 0, gn = 0;
  for (let y = Math.max(1, y0 - Math.round(eyeW * 0.5)); y < y0 - 2; y++) {
    for (let x = x0 + Math.round(bw * 0.3); x < x1 - Math.round(bw * 0.3); x += 2) {
      const k = (y * W + x) * 4, k2 = k + 8;
      grain += Math.abs(lum(d[k], d[k + 1], d[k + 2]) - lum(d[k2], d[k2 + 1], d[k2 + 2])); gn++;
    }
  }
  grain = gn ? clamp(grain / gn, 1, 10) : 3;

  const filled = makeCanvas(W, H);
  const fctx = filled.getContext('2d');
  const out = fctx.createImageData(W, H);
  out.data.set(d);
  const rnd = mulberry32(7);
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    const j = (y - y0) * bw + (x - x0);
    if (!wsum[j]) continue;
    const k = (y * W + x) * 4;
    const n = (rnd() + rnd() - 1) * grain * 1.1;
    out.data[k] = fill[j * 3] + n;
    out.data[k + 1] = fill[j * 3 + 1] + n;
    out.data[k + 2] = fill[j * 3 + 2] + n;
  }
  fctx.putImageData(out, 0, 0);

  // Feather the patch into the photo.
  const feather = makeCanvas(W, H);
  const fe = feather.getContext('2d');
  fe.filter = `blur(${Math.max(1.5, eyeW * 0.035)}px)`;
  fe.drawImage(mask, 0, 0);
  fctx.globalCompositeOperation = 'destination-in';
  fctx.drawImage(feather, 0, 0);
  fctx.globalCompositeOperation = 'source-over';

  const result = makeCanvas(W, H);
  const rctx = result.getContext('2d');
  rctx.drawImage(src, 0, 0);
  rctx.drawImage(filled, 0, 0);

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
      hair = rgbToHex({ r: skin[0] * 0.2 + 18, g: skin[1] * 0.15 + 12, b: skin[2] * 0.12 + 9 });
    }
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

  const thickness = 0.36 * eyeW * s.thick;
  const innerTopY = innerBotY - thickness;
  const archTopY = innerTopY - 0.14 * eyeW * s.lift;
  const archThick = thickness * 0.68;

  const ax = face.noseHalf, ay = face.nostrilY;

  // 1) start: straight up from the nostril edge
  const innerHalf = ax;
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

    const top = [IT];
    cubic(IT, pt(X(innerHalf + span1 * 0.3), IT.y - (IT.y - AT.y) * 0.85), pt(X(archHalf - span1 * 0.3), AT.y), AT, 24, top);
    cubic(AT, pt(X(archHalf + span2 * 0.3), AT.y), pt(X(tailHalf - span2 * 0.35), T.y - (T.y - AT.y) * 0.45), T, 20, top);

    const bottom = [T];
    cubic(T, pt(X(tailHalf - span2 * 0.45), T.y - (T.y - AB.y) * 0.25), pt(X(archHalf + span2 * 0.35), AB.y), AB, 20, bottom);
    cubic(AB, pt(X(archHalf - span1 * 0.35), AB.y), pt(X(innerHalf + span1 * 0.35), IB.y - (IB.y - AB.y) * 0.05), IB, 24, bottom);

    // rounded inner end
    const front = [];
    cubic(IB, pt(IB.x - sg * thickness * 0.18, IB.y - thickness * 0.15), pt(IT.x - sg * thickness * 0.12, IT.y + thickness * 0.2), IT, 10, front);

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

  for (const side of geo.sides) {
    const xs = side.outline.map((p) => p.x), ys = side.outline.map((p) => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const count = Math.round(560 * s.thick);
    let tries = 0;
    while (strokes.filter((k) => k.sg === side.sg).length < count && tries < count * 40) {
      tries++;
      const x = lerp(minX, maxX, rnd()), y = lerp(minY, maxY, rnd());
      if (!probe.isPointInPath(side.path, x, y)) continue;
      const u = clamp((Math.abs(x - midX) - geo.innerHalf) / (geo.tailHalf - geo.innerHalf));
      // thin out the very front for a soft, natural start
      if (u < 0.12 && rnd() > 0.35 + u * 4) continue;

      // vertical position inside the brow at this x (0 = bottom, 1 = top)
      const topY = lerp(geo.innerTopY, geo.archTopY, smooth(u / Math.max(archU, 0.01)));
      const v = clamp((geo.innerBotY - y) / Math.max(1, geo.innerBotY - topY));

      let a; // degrees above horizontal, pointing towards the tail
      if (u < archU) a = lerp(78, 14, smooth(u / archU));
      else a = lerp(14, -22, smooth((u - archU) / (1 - archU)));
      if (u > 0.15) a += (0.5 - v) * 26; // herringbone: low hairs rise, high hairs lie down
      a += (rnd() - 0.5) * 16;
      const rad = (a * Math.PI) / 180;
      const len = eyeW * (u < 0.15 ? lerp(0.05, 0.1, rnd()) : lerp(0.07, 0.13, rnd()));
      const dx = side.sg * Math.cos(rad) * len, dy = -Math.sin(rad) * len;
      const bend = (rnd() - 0.3) * 0.25 * len;
      const shade = lerp(0.82, 1.25, rnd());
      strokes.push({
        sg: side.sg, u,
        x, y, cx: x + dx * 0.5 + side.sg * bend * Math.sin(rad), cy: y + dy * 0.5 + bend * Math.cos(rad) * 0.5,
        ex: x + dx, ey: y + dy,
        w: Math.max(0.5, eyeW * 0.0075) * lerp(0.7, 1.3, rnd()),
        alpha: lerp(0.35, 0.8, rnd()) * (u < 0.1 ? 0.6 : 1),
        color: `rgb(${clamp(base.r * shade, 0, 255) | 0},${clamp(base.g * shade, 0, 255) | 0},${clamp(base.b * shade, 0, 255) | 0})`,
      });
    }
  }
  strokes.sort((a, b) => a.u - b.u);
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
    const x = midX + sg * face.noseHalf;
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
      [2.8, 'Start: nostril edge, straight up'],
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

function drawBrows(c, progress) {
  if (progress <= 0) return;
  const s = settings();
  const geo = state.brows;
  const rgb = hexToRgb(s.color);

  if (s.style !== 'strokes') {
    // powder base with an ombré front
    for (const side of geo.sides) {
      const g = c.createLinearGradient(side.IT.x, 0, side.T.x, 0);
      const k = s.style === 'powder' ? 1 : 0.7;
      g.addColorStop(0, `rgba(${rgb.r},${rgb.g},${rgb.b},${0.12 * k})`);
      g.addColorStop(0.3, `rgba(${rgb.r},${rgb.g},${rgb.b},${0.6 * k})`);
      g.addColorStop(1, `rgba(${rgb.r},${rgb.g},${rgb.b},${0.72 * k})`);
      c.save();
      c.globalAlpha = smooth(progress * 1.4);
      c.filter = `blur(${Math.max(1, state.face.eyeW * 0.022)}px)`;
      c.fillStyle = g;
      c.fill(side.path);
      c.restore();
    }
  }
  if (s.style !== 'powder') {
    const n = Math.floor(geo.strokes.length * clamp(progress));
    c.save();
    // keep hairs inside the mapped shape (slightly grown so edges stay soft)
    const clip = new Path2D();
    for (const side of geo.sides) clip.addPath(side.path);
    c.clip(clip);
    c.lineCap = 'round';
    for (let i = 0; i < n; i++) {
      const k = geo.strokes[i];
      c.globalAlpha = k.alpha;
      c.strokeStyle = k.color;
      c.lineWidth = k.w;
      c.beginPath(); c.moveTo(k.x, k.y); c.quadraticCurveTo(k.cx, k.cy, k.ex, k.ey); c.stroke();
    }
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

  // final brows (under the guides while they fade)
  drawBrows(c, (t - tl.fillStart) / (tl.fillEnd - tl.fillStart));

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

[els.thick, els.lift, els.arch, els.tail, els.color, els.style].forEach((el) =>
  el.addEventListener('input', () => { syncOutputs(); rebuild(); }));

els.savePng.addEventListener('click', savePng);
els.saveVid.addEventListener('click', saveVideo);
els.newPhoto.addEventListener('click', () => { els.file.value = ''; els.file.click(); });

syncOutputs();
getLandmarker().catch(() => {}); // warm up while the user picks a photo

// test hook: render any moment of the animation
window.__brow = { seek: (t) => { state.playing = false; render(t); }, state };
