// CPU light map: Abuse's per-pixel light level (0-63), sampled on a 4 px grid and
// interpolated by the GPU. The shader darkens each colour by (63 - level).

export const CELL = 4;
export const LIGHT_W = 81;
export const LIGHT_H = 51;

// Bounding box in which a light has any effect; types 1-8 are half or quarter planes.
export function lightRange(l) {
  const ox = l.outer >> l.xs, oy = l.outer >> l.ys;
  const { x, y } = l;
  switch (l.type) {
    case 0: return [x - ox, y - oy, x + ox, y + oy];
    case 1: return [x - ox, y - oy, x + ox, y];
    case 2: return [x - ox, y, x + ox, y + oy];
    case 3: return [x, y - oy, x + ox, y + oy];
    case 4: return [x - ox, y - oy, x, y + oy];
    case 5: return [x, y - oy, x + ox, y];
    case 6: return [x - ox, y - oy, x, y];
    case 7: return [x - ox, y, x, y + oy];
    case 8: return [x, y, x + ox, y + oy];
    default: return [x, y, x + l.xs, y + l.ys]; // type 9: constant-level rectangle
  }
}

export class LightMap {
  constructor() {
    this.data = new Uint8Array(LIGHT_W * LIGHT_H);
  }

  update(lights, camX, camY, ambient) {
    const view = [];
    for (const l of lights) {
      const r = lightRange(l);
      if (r[0] > camX + LIGHT_W * CELL || r[2] < camX || r[1] > camY + LIGHT_H * CELL || r[3] < camY) continue;
      view.push({ l, r, mul: l.outer > l.inner ? Math.floor(65536 / (l.outer - l.inner)) * 64 : 0 });
    }
    const d = this.data;
    for (let j = 0; j < LIGHT_H; j++) {
      const py = camY + j * CELL;
      for (let i = 0; i < LIGHT_W; i++) {
        const px = camX + i * CELL;
        let lv = ambient;
        for (const { l, r, mul } of view) {
          if (px < r[0] || px > r[2] || py < r[1] || py > r[3]) continue;
          if (l.type === 9) { lv = l.inner; break; }
          const dx = Math.abs(l.x - px) << l.xs;
          const dy = Math.abs(l.y - py) << l.ys;
          const r2 = dx < dy ? dx + dy - (dx >> 1) : dx + dy - (dy >> 1);
          if (r2 < l.outer) lv += Math.floor(((l.outer - r2) * mul) / 65536);
        }
        d[j * LIGHT_W + i] = lv > 63 ? 63 : lv < 0 ? 0 : lv;
      }
    }
    return d;
  }
}
