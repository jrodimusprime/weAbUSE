// Reader for Abuse SPEC1.0 archives and the records stored in them.

export const T = {
  COLOR_TABLE: 1, PALETTE: 2, IMAGE: 4, FORETILE: 5, BACKTILE: 6, CHARACTER: 7,
  DATA_ARRAY: 20, CHARACTER2: 21,
};

const cache = new Map();

export async function loadSpec(url) {
  if (!cache.has(url)) {
    cache.set(url, fetch(url).then(async (r) => {
      if (!r.ok) throw new Error(`${r.status} ${url}`);
      return new Spec(await r.arrayBuffer());
    }));
  }
  return cache.get(url);
}

export class Spec {
  constructor(buffer) {
    this.dv = new DataView(buffer);
    this.u8 = new Uint8Array(buffer);
    const sig = String.fromCharCode(...this.u8.subarray(0, 7));
    if (sig !== 'SPEC1.0') throw new Error('not a SPEC file');
    const count = this.dv.getUint16(8, true);
    this.entries = [];
    this.byName = new Map();
    let p = 10;
    for (let i = 0; i < count; i++) {
      const type = this.u8[p++];
      const nameLen = this.u8[p++];
      const name = String.fromCharCode(...this.u8.subarray(p, p + nameLen)).replace(/\0+$/, '');
      p += nameLen;
      const flags = this.u8[p++];
      if (flags & 1) { p += 1 + this.u8[p]; continue; }
      const size = this.dv.getUint32(p, true);
      const offset = this.dv.getUint32(p + 4, true);
      p += 8;
      const e = { type, name, size, offset };
      this.entries.push(e);
      if (!this.byName.has(name)) this.byName.set(name, e);
    }
  }

  find(name) { return this.byName.get(name); }
  ofType(type) { return this.entries.filter((e) => e.type === type); }
  bytes(e) { return this.u8.subarray(e.offset, e.offset + e.size); }
}

class Reader {
  constructor(spec, offset) { this.dv = spec.dv; this.u8 = spec.u8; this.p = offset; }
  u8v() { return this.u8[this.p++]; }
  i8() { const v = this.dv.getInt8(this.p); this.p += 1; return v; }
  u16() { const v = this.dv.getUint16(this.p, true); this.p += 2; return v; }
  bytes(n) { const b = this.u8.subarray(this.p, this.p + n); this.p += n; return b; }
}

function readImage(r) {
  const w = r.u16();
  const h = r.u16();
  return { w, h, pix: r.bytes(w * h) };
}

function readPoints(r) {
  const n = r.u8v();
  const pts = [];
  for (let i = 0; i < n; i++) pts.push([r.u8v(), r.u8v()]);
  return pts;
}

export function readPalette(spec, e) {
  const r = new Reader(spec, e.offset);
  const n = r.u16();
  const rgb = new Uint8Array(256 * 4);
  const raw = r.bytes(n * 3);
  for (let i = 0; i < Math.min(n, 256); i++) {
    rgb[i * 4] = raw[i * 3]; rgb[i * 4 + 1] = raw[i * 3 + 1]; rgb[i * 4 + 2] = raw[i * 3 + 2]; rgb[i * 4 + 3] = 255;
  }
  return rgb;
}

export function readBackTile(spec, e) {
  const r = new Reader(spec, e.offset);
  const img = readImage(r);
  img.next = r.u16();
  return img;
}

export function readForeTile(spec, e) {
  const r = new Reader(spec, e.offset);
  const img = readImage(r);
  img.next = r.u16();
  img.damage = r.u8v();
  img.points = readPoints(r);
  img.mask = rasterize(img.points, img.w, img.h);
  return img;
}

export function readCharacter(spec, e) {
  const r = new Reader(spec, e.offset);
  const img = readImage(r);
  img.hitDamage = r.u8v();
  img.xcfg = r.u8v();
  if (e.type === T.CHARACTER2) img.advance = r.i8(); else readPoints(r);
  img.boundary = readPoints(r);
  img.hit = readPoints(r);
  return img;
}

// Even-odd fill of a tile's boundary polygon; null when the tile has no boundary (not solid).
function rasterize(pts, w, h) {
  if (pts.length < 3) return null;
  const mask = new Uint8Array(w * h);
  let any = false;
  for (let y = 0; y < h; y++) {
    const cy = y + 0.5;
    const xs = [];
    for (let i = 0; i < pts.length - 1; i++) {
      const [x1, y1] = pts[i];
      const [x2, y2] = pts[i + 1];
      if ((y1 <= cy && y2 > cy) || (y2 <= cy && y1 > cy)) {
        xs.push(x1 + ((cy - y1) / (y2 - y1)) * (x2 - x1));
      }
    }
    xs.sort((a, b) => a - b);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      for (let x = Math.ceil(xs[k] - 0.5); x < xs[k + 1] - 0.5 && x < w; x++) {
        if (x >= 0) { mask[y * w + x] = 1; any = true; }
      }
    }
  }
  return any ? mask : null;
}
