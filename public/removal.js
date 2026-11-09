/* Brow detection + classic skin fill, free of any page (DOM) dependencies so it
 * can run inside a Web Worker and keep the page animation smooth. */

const LM = {
  irisR: 468, irisL: 473,
  browPolyR: [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
  browPolyL: [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
};
const lerp = (a, b, t) => a + (b - a) * t;
const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const smooth = (t) => { t = clamp(t); return t * t * (3 - 2 * t); };
const lum = (r, g, b) => 0.299 * r + 0.587 * g + 0.114 * b;
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function makeCanvas(w, h) {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h);
  const c = document.createElement('canvas'); c.width = w; c.height = h; return c;
}
function hexToRgb(hex) { const n = parseInt(hex.slice(1), 16); return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 }; }
function rgbToHex({ r, g, b }) {
  return '#' + [r, g, b].map((v) => Math.round(clamp(v, 0, 255)).toString(16).padStart(2, '0')).join('');
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

let INNER_DROP = 0.5;
export function setInnerDrop(v) { INNER_DROP = v; }
const FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
  152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
const LID_R = [33, 246, 161, 160, 159, 158, 157, 173, 133];
const LID_L = [263, 466, 388, 387, 386, 385, 384, 398, 362];
const BROW_LOW_R = [46, 53, 52, 65, 55], BROW_LOW_L = [276, 283, 282, 295, 285];

// --- small image helpers (all on a w*h grid) ---
export function dilateMask(m, w, h, r) {
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
export function boxBlur(src, w, h, ch, r) {
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

export function removeBrowsData(d, W, H, P, eyeW) {

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
  // eyelid height; between the inner eye corner and the nose there is no lid,
  // so keep the height of the lid a little way into the eye (not the low corner)
  const midXi = (cR + cL) / 2;
  const lidY = (ix, right) => {
    const pts = right ? lidR : lidL;
    const inner = right ? P[133] : P[362], outer = right ? P[33] : P[263];
    const x22 = lerp(inner.x, outer.x, 0.22);
    if (Math.abs(ix - midXi) < Math.abs(x22 - midXi)) return lerp(lineY(pts, x22), lineY(pts, ix), INNER_DROP);
    return lineY(pts, ix);
  };
  for (let x = 0; x < bw; x++) {
    const ix = x + bx0, right = Math.abs(ix - cR) < Math.abs(ix - cL);
    const fb = lineY(right ? lowR : lowL, ix) + eyeW * 0.3;
    const ly = lidY(ix, right);
    floorAt[x] = Math.min(fb, ly - eyeW * 0.1) - by0;
    lidAt[x] = ly - by0;
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

  // ---- where the lashes end (per column) ----
  // From the lash line walk upwards while pixels are lash-black. The brow may
  // be erased down to just above that point and never below it, so brows set
  // right on the eye get fully erased while lashes and the lid stay intact.
  {
    const eyeSpan = (pts) => [Math.min(...pts.map((p) => p.x)) - bx0, Math.max(...pts.map((p) => p.x)) - bx0];
    const spans = [eyeSpan(lidR), eyeSpan(lidL)];
    const lashFloor = new Float32Array(bw).fill(NaN);
    for (let x = 0; x < bw; x++) {
      if (!spans.some(([a, b]) => x >= a && x <= b)) continue;
      const ref = skinAbove[x] || 200, lid = Math.round(lidAt[x]);
      const top = Math.max(0, Math.round(lidAt[x] - eyeW * 0.25));
      let lastDark = lid, miss = 0, y = lid;
      for (; y >= top; y--) {
        const k = y * bw + x;
        if (L[k] / ref < 0.45) { lastDark = y; miss = 0; } else if (++miss > 2) break;
      }
      // still dark all the way up: lashes run into the brow, assume typical lash length
      const lashTop = y < top ? lidAt[x] - eyeW * 0.13 : lastDark;
      lashFloor[x] = Math.min(lashTop - eyeW * 0.015, lidAt[x] - eyeW * 0.04);
    }
    // smooth across neighbouring columns, then apply
    const r = Math.max(2, Math.round(eyeW * 0.03));
    for (let x = 0; x < bw; x++) {
      if (Number.isNaN(lashFloor[x])) continue;
      const v = [];
      for (let k = x - r; k <= x + r; k++) if (k >= 0 && k < bw && !Number.isNaN(lashFloor[k])) v.push(lashFloor[k]);
      v.sort((a, b) => a - b);
      const ix = x + bx0, right = Math.abs(ix - cR) < Math.abs(ix - cL);
      const fb = lineY(right ? lowR : lowL, ix) + eyeW * 0.3 - by0;
      floorAt[x] = Math.min(fb, v[v.length >> 1]);
    }
    for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
      const k = y * bw + x;
      allowed[k] = search[k] && oval[k] && y <= floorAt[x] ? 1 : 0;
    }
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
  const TexS = boxBlur(Tex, bw, bh, 1, 1);
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
      // hair-like darkness wins over a smooth shadow (eye socket, lid fold)
      const hairy = TexS[k] > texThr * 0.7 || Ls[k] / ref < 0.32;
      const D = (ref - Lh[k]) / ref * (hairy ? 1 : 0.55);
      if (D > bestD) { bestD = D; best = y; }
    }
    if (best < 0 || bestD < 0.1) continue;
    const t = bestD * 0.55;
    const inBand = (y) => { const k = y * bw + x; return allowed[k] && (ref - Ls[k]) / ref > t; };
    let top = best, bot = best;
    const gap = Math.max(1, Math.round(eyeW * 0.015));
    for (let y = best - 1, miss = 0; y >= 0 && miss <= gap; y--) { if (inBand(y)) { top = y; miss = 0; } else miss++; }
    // going down, a smooth shadow (lid fold) is not brow: it must be very dark
    // or grainy like hair to count
    const inBandBelow = (y) => {
      if (!inBand(y)) return false;
      const k = y * bw + x, r = Ls[k] / ref;
      return r < 0.32 || Tex[k] > texThr * 0.75;
    };
    for (let y = best + 1, miss = 0; y < bh && miss <= gap; y++) { if (inBandBelow(y)) { bot = y; miss = 0; } else miss++; }
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
  // lowest allowed row per column: the band bottom, or (at the brow ends where
  // no band was found) the nearest band column's bottom / the brow line
  const botLimit = new Float32Array(bw);
  for (let x = 0; x < bw; x++) {
    if (bandBot[x] >= 0) { botLimit[x] = bandBot[x]; continue; }
    let best = -1, bd = 1e9;
    for (let k = 1; k < bw; k++) {
      if (x - k >= 0 && bandBot[x - k] >= 0) { best = bandBot[x - k]; bd = k; break; }
      if (x + k < bw && bandBot[x + k] >= 0) { best = bandBot[x + k]; bd = k; break; }
    }
    botLimit[x] = best >= 0 && bd < eyeW * 0.4 ? best : lowLine[x];
  }
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
        if (ny > botLimit[nx]) continue;
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
  const out = { data: new Uint8ClampedArray(d) };
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
  return { out: out.data, hair, mask, bx0, by0, bw, bh, dbg: { bandTop: Array.from(bandTop), bandBot: Array.from(bandBot), floorAt: Array.from(floorAt), lowLine: Array.from(lowLine) } };
}


