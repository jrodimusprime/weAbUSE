// Runs the game without a browser: stubs the handful of browser APIs the
// engine touches, so the real game code can be driven from Node (training,
// tests, level inspection). Nothing is drawn.
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const store = new Map();
globalThis.addEventListener ??= () => {};
globalThis.requestAnimationFrame ??= () => 0;
// defined outright: newer Node versions have a localStorage of their own that
// warns when touched without a backing file
Object.defineProperty(globalThis, 'localStorage', {
  configurable: true,
  value: {
    getItem: (k) => store.get(k) ?? null,
    setItem: (k, v) => { store.set(k, String(v)); },
    removeItem: (k) => { store.delete(k); },
  },
});
// the game fetches its data relative to the source files: read them from disk
globalThis.fetch = async (url) => {
  try {
    const buf = await readFile(fileURLToPath(url));
    return {
      ok: true,
      text: async () => buf.toString('latin1'),
      arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
    };
  } catch {
    return { ok: false, status: 404, text: async () => '', arrayBuffer: async () => new ArrayBuffer(0) };
  }
};

const gl = new Proxy({}, { get: () => () => 1 }); // a WebGL context that accepts everything
const canvas = {
  getContext: () => gl,
  addEventListener: () => {},
  getBoundingClientRect: () => ({ left: 0, top: 0, width: 320, height: 200 }),
};

export const LEVELS = Array.from({ length: 22 }, (_, i) => `level${String(i).padStart(2, '0')}.spe`);

// A ready Game on the given level. The Lisp reader logs a couple of harmless
// debug messages while loading the game's scripts; they are silenced here.
export async function makeGame(level = LEVELS[0]) {
  const { Game } = await import('../src/game.js');
  const g = new Game(canvas, { textContent: '' });
  const debug = console.debug;
  console.debug = () => {};
  try { await g.init(); } finally { console.debug = debug; }
  g.autoPause = true; // the caller steps the simulation itself
  await g.start(level);
  return g;
}
