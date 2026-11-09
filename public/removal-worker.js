import { removeBrowsData } from './removal.js';

self.onmessage = (e) => {
  const { id, d, W, H, P, eyeW } = e.data;
  try {
    const r = removeBrowsData(d, W, H, P, eyeW);
    self.postMessage({ id, ok: true, r }, [r.out.buffer, r.mask.buffer]);
  } catch (err) {
    self.postMessage({ id, ok: false, error: String((err && err.message) || err) });
  }
};
