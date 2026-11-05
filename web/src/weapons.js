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
  // delay = original fire_delay1 (cop.cpp ufuns), in 15 Hz ticks.
  MGUN: { top: 'MGUN_TOP', label: 'Machine gun', kind: 'bullet', delay: 3, aim: 1 },
  PGUN: { top: 'PGUN_TOP', label: 'Plasma', kind: 'plasma', delay: 2 },
  GRENADE: { top: 'GRENADE_TOP', label: 'Grenades', kind: 'grenade', delay: 6 },
  ROCKET: { top: 'ROCKET_TOP', label: 'Rockets', kind: 'rocket', delay: 6, large: true },
  FIREBOMB: { top: 'FIREBOMB_TOP', label: 'Fire bombs', kind: 'grenade', delay: 6, fire: true },
  DFRIS: { top: 'DFRIS_TOP', label: 'Death frisbee', kind: 'plasma', delay: 6, big: true, large: true },
  LSABER: { top: 'LIGHT_SABER', label: 'Light saber', kind: 'plasma', delay: 1, big: true },
};
export const WEAPON_ORDER = ['MGUN', 'PGUN', 'GRENADE', 'ROCKET', 'FIREBOMB', 'DFRIS', 'LSABER'];

// abuse.lsp bad_guy_list (the types the original's seekers look for), as far as this game has them
const BAD_GUYS = new Set(['ANT_ROOF', 'TRACK_GUN', 'SPRAY_GUN', 'JUGGER', 'ROB1', 'WHO', 'FLYER', 'GREEN_FLYER', 'BOSS_ANT']);
const badGuyNear = (g, x, y, rx, ry, skip = null) => {
  let best = null, bd = Infinity;
  for (const e of g.entities) {
    if (e.dead || e === skip || !BAD_GUYS.has(e.type) || !e.shootable) continue;
    const dx = Math.abs(e.x - x), dy = Math.abs(e.y - y);
    if (dx >= rx || dy >= ry) continue;
    if (g.apart?.(skip, e)) continue; // (a 3D host: only what is at the shot's depth)
    const d = dx * dx + dy * dy;
    if (d < bd) { bd = d; best = e; }
  }
  return best;
};

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
  p.cooldown = w.delay;
  if (!g.god) {
    if ((p.ammo[p.weapon] || 0) > 0) {
      p.ammo[p.weapon]--;
      if (p.ammo[p.weapon] === 0) g.outOfAmmo();
    } else if (p.weapon === 'MGUN') {
      // cop.cpp laser_ufun: the machine gun still fires with no ammo, at the
      // slower fire_delay1 of 5 ticks.
      p.cooldown = 5;
    } else {
      // An empty weapon is never left selected (view.cpp add_ammo).
      p.cooldown = 0;
      g.outOfAmmo();
      return;
    }
  }
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
    // SHOTGUN_BULLET (guns.lsp fire_object type 10): speed 15,
    // lifetime 6, red palette colours (find_rgb 255 0 0 / 150 0 0). The C++
    // sgun_ai accelerates it 6/5 every tick along a fixed angle — that is what
    // gives the original its range, and there is no spread.
    // (The "+ xvel / 2" in fire_object is evaluated inside with_object on the
    // new bullet, whose own xvel is 0: the speed is 15 whichever way the
    // shooter is running.)
    const speed = 15;
    g.projs.push({
      kind: 'bullet', x: ox, y: oy, px: ox, py: oy, speed, angDeg: p.aimAngle,
      vx: c * speed, vy: s * speed, life: 6, dmg: 5, mine: true,
    });
    g.sound('mgun');
  } else if (!g.originalWeapons) {
    // (this game's own weapons, which its recorded runs were played with: see Game.originalWeapons)
    if (w.kind === 'plasma') {
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
  } else if (p.weapon === 'PGUN') {
    // guns.lsp fire_object 4, PLASMAGUN_BULLET: no bolt but a line, at once: set_course angle 200 and one
    // bmove, stopping at the first thing in the way: a wall pops EXPLODE5, something hurtable EXPLODE3 and
    // 10 damage. What is left lives 5 ticks to be drawn (pgun_draw: a scatter line fading pink).
    // (Traced on the next projectile update, when a 3D host has given it its course across the hall.)
    g.projs.push({ kind: 'beam', style: 'plasma', x: ox, y: oy, px: ox, py: oy, ox, oy, angDeg: p.aimAngle, reach: 200, dmg: 10, life: 5, pending: true, mine: true });
    g.sound('plasma');
  } else if (p.weapon === 'LSABER') {
    // fire_object 7, LSABER_BULLET: the same at reach 45 for 30 damage, fired every tick (fire_delay1 1) at
    // the aim plus (tick & 7) - 8 degrees (cop.cpp lsaber_ufun): a flickering blade. Drawn for one tick.
    const a = p.aimAngle + ((g.tickCount || 0) & 7) - 8;
    g.projs.push({ kind: 'beam', style: 'saber', x: ox, y: oy, px: ox, py: oy, ox, oy, angDeg: a, reach: 45, dmg: 30, life: 1, pending: true, mine: true });
    g.sound('plasma02');
  } else if (p.weapon === 'DFRIS') {
    // fire_object 6, DFRIS_BULLET (dfris_ai): a disc that flies at 12 a tick (25 on the first), turning up to
    // 35 degrees a tick towards the player's pointer, trailing smoke; it blows up (do_white_explo 40 45) on
    // anything in its way or a bad guy within 7.
    g.projs.push({ kind: 'disc', x: ox, y: oy, px: ox, py: oy, angDeg: p.aimAngle, speed: 25, vx: c * 25, vy: s * 25, life: 300, dmg: 45, radius: 40, mine: true, def: 'DFRIS_BULLET', frame: 0 });
    g.sound('rocket');
  } else if (p.weapon === 'FIREBOMB') {
    // fire_object 5, FIREBOMB (firebomb_ai): set_course angle 20, plus the thrower's fall. Not a grenade: an
    // unseen fire that runs on (it falls, and slides to a stop once down) for up to 20 ticks, every tick
    // bursting into flame (EXPLODE1) and burning all round it (hurt_radius 60 40, push 10), until a wall.
    g.projs.push({ kind: 'firebomb', x: ox, y: oy, px: ox, py: oy, vx: c * 20, vy: s * 20 + p.vy / 15, t: 0, dmg: 40, radius: 60, mine: true });
    g.sound('firebmb1');
  } else if (w.kind === 'grenade') {
    // GRENADE: do_explo 40 36 (weapons.lsp grenade_ai), the same as the ones enemies throw
    g.projs.push({
      kind: 'grenade', x: ox, y: oy, px: ox, py: oy, vx: c * 13 + vx0, vy: s * 13 - 3, gravity: 2, life: 40, dmg: 36,
      radius: 40, mine: true, def: 'GRENADE', bounces: 0,
    });
    g.sound('throw');
  } else if (w.kind === 'rocket') {
    // cop.cpp player_rocket_ufun: locks on to the nearest bad guy within 160 x 130 as it fires; fire_object
    // 3: speed 5, gaining 2 a tick to 14 (the player's), turning up to 23 degrees a tick towards its target
    // (rocket_ai), straight on without one; do_explo 40 50 on a hit or within 10 of the target.
    const b = { kind: 'rocket', x: ox, y: oy, px: ox, py: oy, vx: c * 5, vy: s * 5, life: 90, dmg: 50, radius: 40, mine: true, def: 'ROCKET', smoke: 0, homing: true, speed: 5, maxSpeed: 14, angDeg: ((p.aimAngle % 360) + 360) % 360, hp: 4 };
    b.target = badGuyNear(g, p.x, p.y, 160, 130);
    g.projs.push(b);
    g.sound('rocket');
  }
}

// guns.lsp fire_object, for everything that is not the player: gun turrets,
// ants and flyers all shoot through this, and what comes out is decided by the
// shooter's aitype, set per object in the level:
//   0 fast bullet (speed 15, 6 ticks)      3 rocket (homes in, to speed 10)
//   1 slow bullet (speed 6, 40 ticks)      4 plasma
//   2 grenade                              5 fire bomb
// Bullets do 5 damage (cop.cpp sgun_ai), plasma 10; explosives use the
// do_explo figures from weapons.lsp. `angle` is in degrees, anticlockwise.
export function enemyFire(g, type, x, y, angle) {
  const a = angle * Math.PI / 180, c = Math.cos(a), s = -Math.sin(a);
  const base = { x, y, px: x, py: y, mine: false };
  if (type === 5 && g.originalWeapons) {
    // FIREBOMB (firebomb_ai): see firePlayer
    g.projs.push({ ...base, kind: 'firebomb', vx: c * 20, vy: s * 20, t: 0, dmg: 40, radius: 60 });
    g.sound('firebmb1', x, y);
  } else if (type === 2 || type === 5) {
    // set_course angle 20, then the grenade's own gravity
    g.projs.push({ ...base, kind: 'grenade', vx: c * 20, vy: s * 20, gravity: 2, life: 40, dmg: 36, radius: 40, def: type === 5 ? 'FIREBOMB' : 'GRENADE', bounces: 0 });
    g.sound('throw', x, y);
  } else if (type === 3) {
    // rocket_ai: starts at speed 5, gains 2 a tick up to 10 (14 for the
    // player's), turns up to 23 degrees a tick towards its target
    g.projs.push({ ...base, kind: 'rocket', vx: c * 5, vy: s * 5, life: 90, dmg: 50, radius: 40, def: 'ROCKET', smoke: 0, homing: true, speed: 5, angDeg: ((angle % 360) + 360) % 360, hp: 4 });
    g.sound('rocket', x, y);
  } else if (type === 4 && g.originalWeapons) {
    // PLASMAGUN_BULLET: a line, at once (see firePlayer)
    g.projs.push({ ...base, kind: 'beam', style: 'plasma', ox: x, oy: y, angDeg: angle, reach: 200, dmg: 10, life: 5, pending: true });
    g.sound('plasma', x, y);
  } else if (type === 4) {
    g.projs.push({ ...base, kind: 'plasma', vx: c * 34, vy: s * 34, life: 20, dmg: 10 });
    g.sound('plasma', x, y);
  } else {
    const slow = type === 1;
    g.projs.push({
      ...base, kind: 'bullet', speed: slow ? 6 : 15, angDeg: angle, vx: 0, vy: 0, life: slow ? 40 : 6, dmg: 5,
      bright: slow ? g.colors.orange : g.colors.yellow, dark: slow ? g.colors.redBright : g.colors.yellow,
    });
    g.sound('enemyshot', x, y);
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
      if (g.apart?.(b, e)) continue; // a 3D host's depth check (WEaBuse3D); not set in this game
      const r = g.rectOf(e);
      if (r && b.x >= r.x0 && b.x <= r.x1 && b.y >= r.y0 && b.y <= r.y1) return { entity: e };
    }
    // A rocket is hurtable in the original (weapons.lsp ROCKET: hurtable,
    // start_hp 4), so the player's fire can shoot an enemy's down.
    for (const q of g.projs) {
      if (q === b || q.dead || q.mine || !isRocket(q)) continue;
      if (Math.hypot(q.x - b.x, q.y - b.y) < 9) return { rocket: q };
    }
  } else {
    const r = g.playerRect();
    if (b.x >= r.x0 && b.x <= r.x1 && b.y >= r.y0 && b.y <= r.y1 && !g.apart?.(b, g.player)) return { player: true };
    // ...and the same the other way: the player's rockets are the same
    // ROCKET, and an enemy's shot can bring one down.
    for (const q of g.projs) {
      if (q === b || q.dead || !q.mine || !isRocket(q)) continue;
      if (Math.hypot(q.x - b.x, q.y - b.y) < 9) return { rocket: q };
    }
  }
  return null;
}

function detonate(g, b) {
  if (b.dead) return;
  b.dead = true;
  g.blast = b; // what is blowing up, for a 3D host's depth check (see Game.explode)
  // hurt_radius excludes the shot's creator (explo.lsp do_explo: get_object 0): the player's own blasts
  // do not hurt the player. The frisbee's is do_white_explo (EXPLODE8).
  const spare = !!b.mine && !!g.originalWeapons;
  g.explode(b.x, b.y, b.radius, b.dmg, b.mine, false, null, spare, b.kind === 'disc' ? 'EXPLODE8' : 'EXPLODE1');
  g.blast = null;
}

// Which projectiles are the original's ROCKET (weapons.lsp: hurtable, start_hp
// 4): the player's rockets and the homing ones enemies fire. The flyer's
// STRAIT_ROCKET (ant.lsp) is a different object, not hurtable.
const isRocket = (q) => q.kind === 'rocket' && !q.straight;

// weapons.lsp ROCKET is hurtable with start_hp 4: a direct hit reduces its hp,
// and at 0 it blows (rocket_ai: do_explo, with its own radius and damage).
function damageRocket(g, q, amount) {
  q.hp = Math.max(0, (q.hp ?? 4) - amount);
  if (q.hp <= 0) detonate(g, q);
}
// hurt_radius (level.cpp) damages every hurtable object in a blast, whoever
// set it off; rockets are hurtable, so an explosion destroys any it catches,
// the player's own included.
export function hurtRocketsInBlast(g, x, y, radius, dmg) {
  for (const q of g.projs) {
    if (q.dead || !isRocket(q)) continue;
    const d = Math.hypot(q.x - x, q.y - y);
    if (d < radius) damageRocket(g, q, Math.max(1, ((radius - d) * dmg) / radius));
  }
}

// A plasma or saber line (fire_object 4 and 7): one bmove from the gun along the angle, up to its reach,
// stopping at the first thing in the way.
function traceBeam(g, b) {
  b.pending = false;
  const a = b.angDeg * Math.PI / 180, c = Math.cos(a), s = -Math.sin(a);
  for (let d = 1; d <= b.reach; d++) {
    b.x = b.ox + c * d; b.y = b.oy + s * d;
    if (g.shotStopped?.(b)) break; // (a 3D host: a frame or side wall of the hall in the way; not set in this game)
    const hit = hitAt(g, b);
    if (!hit) continue;
    if (hit.tile) { if (b.style === 'plasma') g.effect('EXPLODE5', b.x - g.rand(5), b.y - g.rand(5)); }
    else {
      if (b.style === 'plasma') g.effect('EXPLODE3', b.x - g.rand(5), b.y - g.rand(5));
      if (hit.player) g.hurtPlayer(b.dmg);
      else if (hit.rocket) damageRocket(g, hit.rocket, b.dmg);
      else g.damage(hit.entity, b.dmg, c * 20, s * 10);
    }
    break;
  }
  b.px = b.ox; b.py = b.oy;
}

// weapons.lsp firebomb_ai, one tick: flame and burn where it is, then move on (it falls; once down it slows
// by its stop_accel, 4, ability.cpp) while it is young or still moving, for 20 ticks, until blocked.
function fireBombTick(g, b) {
  g.effect('EXPLODE1', b.x - g.rand(5), b.y + g.rand(20));
  g.blast = b;
  g.hurtRadius(b.x, b.y, b.radius, b.dmg, b.mine, false, null, !!b.mine);
  g.blast = null;
  if (!((b.t < 3 || b.vx !== 0) && b.t < 20)) { b.dead = true; return; }
  b.t++;
  if (!b.down) b.vy += 2;
  const steps = Math.max(1, Math.ceil(Math.hypot(b.vx, b.vy) / 3));
  for (let i = 0; i < steps; i++) {
    const nx = b.x + b.vx / steps;
    if (g.tileSolid(Math.floor(nx), Math.floor(b.y))) { b.dead = true; return; } // blocked: it stops burning
    b.x = nx;
    const ny = b.y + b.vy / steps;
    if (b.vy > 0 && g.tileSolid(Math.floor(b.x), Math.floor(ny))) { b.down = true; b.vy = 0; }
    else if (b.vy < 0 && g.tileSolid(Math.floor(b.x), Math.floor(ny))) b.vy = 0;
    else b.y = ny;
  }
  if (b.down) {
    if (!g.tileSolid(Math.floor(b.x), Math.floor(b.y) + 1)) b.down = false; // off an edge: falls again
    else b.vx = b.vx > 0 ? Math.max(0, b.vx - 4) : Math.min(0, b.vx + 4);
  }
}

// weapons.lsp dfris_ai: towards the player's pointer, by up to 35 degrees a tick (get_fris_angle)
function discTurn(g, b) {
  if (!g.mouse || !b.mine) return;
  const px = g.mouse.x + g.cam.x, py = g.mouse.y + g.cam.y;
  let want = Math.atan2(b.y - py - 4, px - b.x) * 180 / Math.PI;
  if (want < 0) want += 360;
  let diff = want - b.angDeg;
  if (diff > 180) diff -= 360; else if (diff < -180) diff += 360;
  b.angDeg = (b.angDeg + Math.max(-35, Math.min(35, diff)) + 360) % 360;
}

export function updateProjectiles(g) {
  for (const b of g.projs) {
    if (b.dead) continue; // (shot down or caught in a blast earlier this tick)
    if (b.kind === 'beam') { if (b.pending) traceBeam(g, b); else if (--b.life <= 0) b.dead = true; continue; }
    if (b.kind === 'firebomb') { b.px = b.x; b.py = b.y; fireBombTick(g, b); continue; }
    b.px = b.x; b.py = b.y;
    const bullet = b.kind === 'bullet';
    if (bullet) {
      // C++ sgun_ai (cop.cpp): the bullet accelerates 6/5 each tick and moves
      // along a fixed angle. It dies when lifetime reaches 0, after moving.
      b.speed = Math.trunc(b.speed * 6 / 5); // whole numbers, as the original's integer speed
      const a = b.angDeg * Math.PI / 180;
      b.vx = Math.cos(a) * b.speed;
      b.vy = -Math.sin(a) * b.speed;
    } else if (--b.life <= 0) {
      if (b.kind === 'grenade') detonate(g, b); else b.dead = true;
      continue;
    }
    if (b.gravity) b.vy += b.gravity;
    if (b.kind === 'disc') {
      // dfris_ai: smoke every other tick, then on at 12 towards the pointer
      if ((g.tickCount || 0) % 2 === 0) g.effect('SMALL_LIGHT_CLOUD', b.x, b.y);
      if (b.frame++ > 0) { discTurn(g, b); b.speed = 12; }
      const a = b.angDeg * Math.PI / 180;
      b.vx = Math.cos(a) * b.speed; b.vy = -Math.sin(a) * b.speed;
      if (b.mine) {
        const near = badGuyNear(g, b.x, b.y, 7, 7, b);
        if (near) { detonate(g, b); continue; }
      }
    }
    if (b.kind === 'rocket') {
      if (b.homing) {
        // weapons.lsp rocket_ai: steer towards the target (an enemy's: the
        // player; the player's: the bad guy it locked on to), 15 px above the
        // feet, by up to 23 degrees a tick, not at all within 5; with no
        // target (or one gone) straight on
        const t = b.mine ? (b.target && !b.target.dead ? b.target : null) : g.player;
        if (t) {
          let want = Math.atan2(b.y - (t.y - 15), t.x - b.x) * 180 / Math.PI;
          if (want < 0) want += 360;
          const clock = want < b.angDeg ? b.angDeg - want : b.angDeg + (360 - want);
          const closest = clock > 180 ? 360 - clock : clock;
          const add = closest > 23 ? 23 : closest < 5 ? 0 : closest;
          b.angDeg = clock > 180 ? (b.angDeg + add) % 360 : (b.angDeg - add + 360) % 360;
        }
        if (b.speed < (b.maxSpeed ?? 10)) b.speed += 2;
        const a = b.angDeg * Math.PI / 180;
        b.vx = Math.cos(a) * b.speed; b.vy = -Math.sin(a) * b.speed;
        // and it blows within 10 of its target
        if (t && b.mine && Math.abs(t.x - b.x) < 10 && Math.abs(t.y - b.y - 15) < 10) { detonate(g, b); continue; }
      } else if (!b.straight) { b.vx *= 1.04; b.vy *= 1.04; }
      if (++b.smoke % 2 === 0) g.effect('SMALL_LIGHT_CLOUD', b.x, b.y);
    }
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
      if (b.kind === 'grenade' || b.kind === 'rocket' || b.kind === 'disc') { detonate(g, b); break; }
      if (bullet) {
        // sgun_ai: a wall hit pops EXPLODE5; a hurtable hit pops EXPLODE3 and
        // deals 5 damage with a push along the bullet's direction.
        if (hit.tile) {
          g.effect('EXPLODE5', b.x + g.rand(4), b.y + g.rand(4));
        } else if (hit.player) {
          g.effect('EXPLODE3', b.x + g.rand(4), b.y + g.rand(4));
          g.hurtPlayer(b.dmg);
        } else if (hit.rocket) {
          g.effect('EXPLODE3', b.x + g.rand(4), b.y + g.rand(4));
          damageRocket(g, hit.rocket, b.dmg);
        } else {
          g.effect('EXPLODE3', b.x + g.rand(4), b.y + g.rand(4));
          const a = b.angDeg * Math.PI / 180;
          g.damage(hit.entity, 5, Math.cos(a) * 10, Math.sin(a) * 10);
        }
        b.dead = true;
        break;
      }
      if (hit.rocket) damageRocket(g, hit.rocket, b.dmg);
      else if (hit.entity) g.damage(hit.entity, b.dmg, b);
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
      // the player's are red; enemy bullets carry their own colours (yellow, or orange for the slow kind)
      const bright = b.bright ?? col.redBright, dark = b.dark ?? col.redDark;
      dline(r, lx, ly - 1, tx, ty - 1, dark);
      dline(r, lx, ly + 1, tx, ty + 1, dark);
      dline(r, lx - 1, ly, tx - 1, ty, bright);
      dline(r, lx + 1, ly, tx + 1, ty, dark);
      dline(r, lx, ly, tx, ty, bright);
    } else if (b.kind === 'beam') {
      if (b.pending) continue;
      // pgun_draw: a line fading pink over its 5 ticks; lsaber_draw: white with a blue-grey edge
      const ax = Math.round(b.ox - cx), ay = Math.round(b.oy - cy), bx = Math.round(b.x - cx), by = Math.round(b.y - cy);
      if (b.style === 'plasma') dline(r, ax, ay, bx, by, col.plasma[Math.max(0, Math.min(4, 5 - b.life))]);
      else { dline(r, ax, ay - 1, bx, by - 1, col.saberBlue); dline(r, ax, ay + 1, bx, by + 1, col.saberDark); dline(r, ax, ay, bx, by, col.white); }
    } else if (b.kind === 'firebomb') {
      continue; // fb_draw: nothing; its flames are its EXPLODE1s
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
