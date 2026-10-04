// Player weapons and projectiles (player and enemy).

export const WEAPONS = {
  MGUN: { top: 'MGUN_TOP', label: 'Machine gun', kind: 'bullet', delay: 0.1, aim: 1 },
  PGUN: { top: 'PGUN_TOP', label: 'Plasma', kind: 'plasma', delay: 0.2 },
  GRENADE: { top: 'GRENADE_TOP', label: 'Grenades', kind: 'grenade', delay: 0.55 },
  ROCKET: { top: 'ROCKET_TOP', label: 'Rockets', kind: 'rocket', delay: 0.6 },
  FIREBOMB: { top: 'FIREBOMB_TOP', label: 'Fire bombs', kind: 'grenade', delay: 0.6, fire: true },
  DFRIS: { top: 'DFRIS_TOP', label: 'Death frisbee', kind: 'plasma', delay: 0.3, big: true },
  LSABER: { top: 'LIGHT_SABER', label: 'Light saber', kind: 'plasma', delay: 0.12, big: true },
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
  const ang = p.aimAngle;
  const c = Math.cos(ang), s = -Math.sin(ang);
  // Muzzle from the original player_fire_weapon:
  //   firex = x + cos(angle)*17 + xvel, firey = y - sin(angle)*16 - 20 + yvel
  const ox = p.x + c * 17 + p.vx / 15;
  const oy = p.y - 20 + s * 16 + p.vy / 15;
  const vx0 = p.vx / 15;
  if (w.kind === 'bullet') {
    const spread = (g.rand(100) - 50) / 1500;
    const a = ang + spread;
    // SHOTGUN_BULLET (guns.lsp type 10): speed 15 + creator xvel/2, lifetime 6.
    // It spawns at the muzzle from player_fire_weapon — sgun_ufun is never called
    // (user_fun is an explicit call and guns.lsp doesn't invoke it), so no extra offset.
    const speed = 15 + p.vx / 30;
    g.projs.push({ kind: 'bullet', x: ox, y: oy, px: ox, py: oy, vx: Math.cos(a) * speed, vy: -Math.sin(a) * speed, life: 6, dmg: 3, mine: true });
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
    if (--b.life <= 0) { if (b.kind === 'grenade') detonate(g, b); else b.dead = true; continue; }
    if (b.gravity) b.vy += b.gravity;
    if (b.kind === 'rocket') { b.vx *= 1.04; b.vy *= 1.04; if (++b.smoke % 2 === 0) g.effect('SMALL_LIGHT_CLOUD', b.x, b.y); }
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
      if (hit.entity) g.damage(hit.entity, b.dmg, b);
      if (hit.player) g.hurtPlayer(b.dmg);
      g.effect(b.kind === 'bullet' ? 'SMALL_LIGHT_CLOUD' : b.mine ? 'EXPLODE6' : 'EXPLODE7', b.x, b.y);
      b.dead = true;
    }
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
