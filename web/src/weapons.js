// Player weapons and projectiles (player and enemy).

// cop.cpp: small_fire_off / large_fire_off — x & y offset from the character to
// the end of the gun, per upper-body frame (24 frames around the clock).
export const SMALL_FIRE_OFF = [
  17, 20, 17, 23, 17, 28, 15, 33, 11, 39, 7, 43, -3, 44, -10, 42,
  -16, 39, -20, 34, -20, 28, -20, 25, -19, 20, -19, 16, -16, 14, -14, 11,
  -11, 9, -7, 8, -3, 8, 2, 8, 6, 9, 10, 10, 14, 13, 16, 15,
];
export const LARGE_FIRE_OFF = [
  18, 25, 17, 30, 15, 34, 14, 36, 10, 39, 7, 41, 4, 42, -3, 41,
  -8, 39, -11, 37, -14, 33, -16, 30, -18, 25, -17, 21, -14, 17, -11, 15,
  -7, 13, -4, 12, 3, 12, 9, 12, 12, 15, 14, 16, 15, 18, 16, 21,
];

export const WEAPONS = {
  // delay = original fire_delay1 (ticks at 15 Hz) converted to seconds.
  MGUN: { top: 'MGUN_TOP', label: 'Machine gun', kind: 'bullet', delay: 3 / 15, aim: 1 },
  PGUN: { top: 'PGUN_TOP', label: 'Plasma', kind: 'plasma', delay: 2 / 15 },
  GRENADE: { top: 'GRENADE_TOP', label: 'Grenades', kind: 'grenade', delay: 6 / 15 },
  ROCKET: { top: 'ROCKET_TOP', label: 'Rockets', kind: 'rocket', delay: 6 / 15, large: true },
  FIREBOMB: { top: 'FIREBOMB_TOP', label: 'Fire bombs', kind: 'grenade', delay: 6 / 15, fire: true },
  DFRIS: { top: 'DFRIS_TOP', label: 'Death frisbee', kind: 'plasma', delay: 6 / 15, big: true, large: true },
  LSABER: { top: 'LIGHT_SABER', label: 'Light saber', kind: 'plasma', delay: 1 / 15, big: true },
};
export const WEAPON_ORDER = ['MGUN', 'PGUN', 'GRENADE', 'ROCKET', 'FIREBOMB', 'DFRIS', 'LSABER'];

// 1px Bresenham line in a palette colour (the original draws bullets with draw_line).
function dline(r, x0, y0, x1, y1, idx) {
  let x = Math.round(x0), y = Math.round(y0);
  const tx = Math.round(x1), ty = Math.round(y1);
  const dx = Math.abs(tx - x), dy = Math.abs(ty - y);
  const sx = x < tx ? 1 : -1, sy = y < ty ? 1 : -1;
  let err = dx - dy;
  for (let i = 0; i < 400; i++) {
    r.rect(x, y, 1, 1, idx);
    if (x === tx && y === ty) break;
    const e2 = 2 * err;
    if (e2 > -dy) { err -= dy; x += sx; }
    if (e2 < dx) { err += dx; y += sy; }
  }
}

const ICON_WEAPON = {
  MBULLET: 'MGUN', PLASMA: 'PGUN', GRENADE: 'GRENADE', ROCKET: 'ROCKET', FBOMB: 'FIREBOMB', DFRIS: 'DFRIS', LSABER: 'LSABER',
};

export function pickupFor(typeName) {
  const m = /^([A-Z]+)_ICON(\d+)$/.exec(typeName);
  if (!m || !ICON_WEAPON[m[1]]) return null;
  return { weapon: ICON_WEAPON[m[1]], amount: parseInt(m[2], 10) };
}

export function firePlayer(g) {
  const p = g.player;
  const w = WEAPONS[p.weapon];
  if (!w || p.cooldown > 0 || p.dead) return;
  if (!g.god) {
    if ((p.ammo[p.weapon] || 0) <= 0) return;
    p.ammo[p.weapon]--;
  }
  p.cooldown = w.delay;
  p.justFired = true; // people.lsp: player flashes with bright_tint after firing
  // p.aimAngle is the original's point_angle in degrees (cop.cpp top_ai).
  const ang = p.aimAngle * Math.PI / 180;
  const c = Math.cos(ang), s = -Math.sin(ang);
  // player_fire_weapon (cop.cpp): the muzzle is a per-frame table offset from
  // the character (fire_off), with x shifted +4 when the body faces left.
  const foff = w.large ? LARGE_FIRE_OFF : SMALL_FIRE_OFF;
  const frame = ((p.aim % 24) + 24) % 24;
  const ox = (p.dir < 0 ? p.x + 4 : p.x) + foff[frame * 2];
  const oy = p.y - foff[frame * 2 + 1];
  const vx0 = p.vx / 15;
  if (w.kind === 'bullet') {
    // SHOTGUN_BULLET (guns.lsp fire_object type 10): speed 15 + creator xvel/2,
    // lifetime 6, red palette colours (find_rgb 255 0 0 / 150 0 0). The C++
    // sgun_ai accelerates it 6/5 every tick along a fixed angle — that is what
    // gives the original its range, and there is no spread.
    const speed = 15 + p.vx / 30;
    g.projs.push({
      kind: 'bullet', x: ox, y: oy, px: ox, py: oy, speed, angDeg: p.aimAngle,
      vx: c * speed, vy: s * speed, life: 6, dmg: 5, mine: true,
    });
    g.sound('mgun');
  } else if (w.kind === 'plasma') {
    g.projs.push({ kind: 'plasma', x: ox, y: oy, px: ox, py: oy, vx: c * 34, vy: s * 34, life: 20, dmg: w.big ? 18 : 9, mine: true, big: !!w.big });
    g.sound('plasma');
  } else if (w.kind === 'grenade') {
    g.projs.push({
      kind: 'grenade', x: ox, y: oy, px: ox, py: oy, vx: c * 13 + vx0, vy: s * 13 - 3, gravity: 2, life: 40, dmg: w.fire ? 45 : 40,
      radius: w.fire ? 60 : 50, mine: true, def: w.fire ? 'FIREBOMB' : 'GRENADE', bounces: 0,
    });
    g.sound('throw');
  } else if (w.kind === 'rocket') {
    g.projs.push({ kind: 'rocket', x: ox, y: oy, px: ox, py: oy, vx: c * 18, vy: s * 18, ang, life: 60, dmg: 35, radius: 55, mine: true, def: 'ROCKET', smoke: 0 });
    g.sound('rocket');
  }
}

export function enemyShot(g, x, y, tx, ty, speed, dmg, kind = 'acid') {
  const dx = tx - x, dy = ty - y;
  const d = Math.hypot(dx, dy) || 1;
  g.projs.push({ kind, x, y, px: x, py: y, vx: (dx / d) * speed, vy: (dy / d) * speed, life: 40, dmg, mine: false });
  g.sound('enemyshot', x, y);
}

function hitAt(g, b) {
  if (g.tileSolid(Math.floor(b.x), Math.floor(b.y))) return { tile: true };
  if (b.mine) {
    for (const e of g.entities) {
      if (e.dead || !e.shootable) continue;
      const r = g.rectOf(e);
      if (r && b.x >= r.x0 && b.x <= r.x1 && b.y >= r.y0 && b.y <= r.y1) return { entity: e };
    }
  } else {
    const r = g.playerRect();
    if (b.x >= r.x0 && b.x <= r.x1 && b.y >= r.y0 && b.y <= r.y1) return { player: true };
  }
  return null;
}

function detonate(g, b) {
  g.explode(b.x, b.y, b.radius, b.dmg, b.mine);
  b.dead = true;
}

export function updateProjectiles(g) {
  for (const b of g.projs) {
    b.px = b.x; b.py = b.y;
    const bullet = b.kind === 'bullet';
    if (bullet) {
      // C++ sgun_ai (cop.cpp): the bullet accelerates 6/5 each tick and moves
      // along a fixed angle. It dies when lifetime reaches 0, after moving.
      b.speed = b.speed * 6 / 5;
      const a = b.angDeg * Math.PI / 180;
      b.vx = Math.cos(a) * b.speed;
      b.vy = -Math.sin(a) * b.speed;
    } else if (--b.life <= 0) {
      if (b.kind === 'grenade') detonate(g, b); else b.dead = true;
      continue;
    }
    if (b.gravity) b.vy += b.gravity;
    if (b.kind === 'rocket') { if (!b.straight) { b.vx *= 1.04; b.vy *= 1.04; } if (++b.smoke % 2 === 0) g.effect('SMALL_LIGHT_CLOUD', b.x, b.y); }
    const steps = Math.max(1, Math.ceil(Math.hypot(b.vx, b.vy) / 3));
    for (let i = 0; i < steps && !b.dead; i++) {
      const nx = b.x + b.vx / steps, ny = b.y + b.vy / steps;
      const prevX = b.x, prevY = b.y;
      b.x = nx; b.y = ny;
      const hit = hitAt(g, b);
      if (!hit) continue;
      if (b.kind === 'grenade' && hit.tile) {
        // bounce off whichever axis is blocked
        b.x = prevX; b.y = prevY;
        const blockedY = g.tileSolid(Math.floor(prevX), Math.floor(prevY + Math.sign(b.vy || 1) * 2));
        if (blockedY) b.vy = -b.vy * 0.45; else b.vx = -b.vx * 0.45;
        b.vx *= 0.7;
        if (++b.bounces > 5) detonate(g, b);
        break;
      }
      if (b.kind === 'grenade' || b.kind === 'rocket') { detonate(g, b); break; }
      if (bullet) {
        // sgun_ai: a wall hit pops EXPLODE5; a hurtable hit pops EXPLODE3 and
        // deals 5 damage with a push along the bullet's direction.
        if (hit.tile) {
          g.effect('EXPLODE5', b.x + g.rand(4), b.y + g.rand(4));
        } else {
          g.effect('EXPLODE3', b.x + g.rand(4), b.y + g.rand(4));
          const a = b.angDeg * Math.PI / 180;
          g.damage(hit.entity, 5, Math.cos(a) * 10, Math.sin(a) * 10);
        }
        b.dead = true;
        break;
      }
      if (hit.entity) g.damage(hit.entity, b.dmg, b);
      if (hit.player) g.hurtPlayer(b.dmg);
      g.effect(b.mine ? 'EXPLODE6' : 'EXPLODE7', b.x, b.y);
      b.dead = true;
    }
    // sgun_ai decrements lifetime after the move; the bullet dies at 0.
    if (bullet && !b.dead && --b.life <= 0) b.dead = true;
    if (b.kind === 'grenade' && !b.dead) b.dir = b.vx >= 0 ? 1 : -1;
  }
  g.projs = g.projs.filter((b) => !b.dead);
}

export function drawProjectiles(g, alpha) {
  const r = g.r;
  const cx = Math.floor(g.cam.x), cy = Math.floor(g.cam.y);
  const col = g.colors;
  for (const b of g.projs) {
    const x = b.px + (b.x - b.px) * alpha - cx;
    const y = b.py + (b.y - b.py) * alpha - cy;
    if (b.kind === 'bullet') {
      // Original sgun_draw: bright centre line with medium-red neighbours.
      const lx = Math.round(b.px - cx), ly = Math.round(b.py - cy);
      const tx = Math.round(x), ty = Math.round(y);
      dline(r, lx, ly - 1, tx, ty - 1, col.redDark);
      dline(r, lx, ly + 1, tx, ty + 1, col.redDark);
      dline(r, lx - 1, ly, tx - 1, ty, col.redBright);
      dline(r, lx + 1, ly, tx + 1, ty, col.redDark);
      dline(r, lx, ly, tx, ty, col.redBright);
    } else if (b.kind === 'plasma' || b.kind === 'acid' || b.kind === 'spark') {
      const c1 = b.kind === 'plasma' ? col.cyan : b.kind === 'acid' ? col.green : col.orange;
      const n = 3;
      for (let i = 0; i < n; i++) {
        const t = i / n;
        const sx = Math.round(x - b.vx * t * 0.6), sy = Math.round(y - b.vy * t * 0.6);
        const s = b.big || b.kind === 'acid' ? 3 : 2;
        r.rect(sx - 1, sy - 1, s, s, i === 0 ? col.white : c1);
      }
    } else {
      const def = g.assets.defs.get(b.def);
      const frames = def?.states.get('stopped');
      if (!def || !frames) continue;
      let f = 0;
      if (b.kind === 'rocket' && frames.length > 1) {
        const a = Math.atan2(-b.vy, b.vx);
        f = Math.round(((a + 2 * Math.PI) % (2 * Math.PI)) / (2 * Math.PI) * frames.length) % frames.length;
      }
      const img = g.assets.sprite(def.file, frames[f]);
      if (!img) continue;
      r.draw(img, Math.round(x - img.w / 2), Math.round(y - img.h / 2));
    }
  }
}
