// Runs the MI-GAN skin model in its own thread. A fresh worker is a fresh
// JavaScript realm, so if one configuration fails (e.g. multi-threading on a
// memory-limited phone) the page can simply start another one.
// two engines: 'fast' = onnxruntime-web 1.22 (multi-threaded, reserves a lot of
// memory up front); 'light' = 1.18 single-threaded SIMD build, which does not,
// for phones / browsers that report "out of memory"
let ort = null;

let session = null;

async function download(urls, expected) {
  const parts = [];
  let loaded = 0;
  for (const url of urls) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`model download failed (${r.status})`);
    const type = r.headers.get('content-type') || '';
    if (type.includes('text/html')) throw new Error('model file missing on the server (got a web page instead)');
    const reader = r.body && r.body.getReader ? r.body.getReader() : null;
    if (!reader) {
      const b = new Uint8Array(await r.arrayBuffer());
      parts.push(b); loaded += b.length;
      self.postMessage({ type: 'progress', loaded, total: expected });
      continue;
    }
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value); loaded += value.length;
      self.postMessage({ type: 'progress', loaded, total: expected });
    }
  }
  if (loaded !== expected) throw new Error(`model download incomplete (${loaded} of ${expected} bytes)`);
  const buf = new Uint8Array(loaded);
  let o = 0;
  for (const p of parts) { buf.set(p, o); o += p.length; }
  return buf;
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      ort = m.engine === 'light'
        ? await import('./vendor/ort118/ort.wasm.min.js')
        : await import('./vendor/ort/ort.wasm.min.mjs');
      ort.env.wasm.wasmPaths = m.engine === 'light' ? m.wasmPathsLight : m.wasmPaths;
      ort.env.wasm.numThreads = m.engine === 'light' ? 1 : m.threads;
      ort.env.wasm.proxy = false;
      const buf = await download(m.urls, m.expected);
      // no pre-allocated memory arenas: much lower peak memory
      session = await ort.InferenceSession.create(buf, {
        executionProviders: ['wasm'],
        enableCpuMemArena: false,
        enableMemPattern: false,
        executionMode: 'sequential',
      });
      // smoke test on a tiny image so a broken setup fails here, not mid-photo
      const n = 48 * 48, img = new Uint8Array(3 * n).fill(180), known = new Uint8Array(n).fill(255);
      for (let y = 20; y < 28; y++) for (let x = 16; x < 32; x++) known[y * 48 + x] = 0;
      await session.run({ image: new ort.Tensor('uint8', img, [1, 3, 48, 48]), mask: new ort.Tensor('uint8', known, [1, 1, 48, 48]) });
      self.postMessage({ type: 'ready', id: m.id });
    } else if (m.type === 'run') {
      if (!session) throw new Error('model not ready');
      const out = await session.run({
        image: new ort.Tensor('uint8', m.chw, [1, 3, m.ch, m.cw]),
        mask: new ort.Tensor('uint8', m.known, [1, 1, m.ch, m.cw]),
      });
      const r = new Uint8Array(out.result.data);
      self.postMessage({ type: 'result', id: m.id, r }, [r.buffer]);
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: m.id, error: String((err && err.message) || err) });
  }
};
