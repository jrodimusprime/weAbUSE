// Native JS versions of the original Lisp/C++ object AI, keyed by the def_char ai_fun name.
// Each takes (entity, game) once per 15 Hz tick and returns false to remove the entity.
import { pickupFor, enemyShot } from './weapons.js';

export const LOGIC_AI = new Set([
  'delay_ai', 'or_ai', 'and_ai', 'xor_ai', 'not_ai', 'pulse_ai', 'sensor_ai', 'switcher_ai', 'switch_once_ai',
  'sdoor_ai', 'strap_door_ai', 'indicator_ai', 'hwall_ai', 'big_wall_ai', 'switch_delay_ai', 'switch_dim_ai', 'switch_mover_ai',
]);

// Door-like objects block the player until they reach their fully open ('blocking') state.
export const SOLID_AI = new Set(['sdoor_ai', 'strap_door_ai', 'hwall_ai', 'big_wall_ai', 'platform_ai']);

const HIDDEN_DRAW = new Set(['dev_draw', 'sensor_draw']);
export const isHiddenInPlay = (def) => HIDDEN_DRAW.has(def.funs.get('draw_fun'));

const GRAVITY = 2;
const link0 = (e) => e.links[0];
// Unlinked objects count as activated; linked ones follow their first link.
const activated = (e) => (e.links.length ? link0(e).aistate !== 0 : true);
const goState = (e, s) => { e.aistate = s; e.stateTime = 0; };

function fall(e, g) {
  e.vy = Math.min(e.vy + GRAVITY, 26);
  const m = g.moveEntity(e, e.vx, e.vy);
  if (m.down || m.up) e.vy = 0;
  if (m.blockedX) e.vx = 0;
  return m;
}

const nearestDist = (e, p) => ({ dx: p.x - e.x, dy: p.y - e.y });

function killEffects(e, g, big) {
  g.effect(big ? 'EXPLODE1' : 'EXPLODE2', e.x, e.y - 12);
  g.effect('EXPLODE3', e.x + g.rand(10) - 5, e.y - 6);
  g.sound('explode', e.x, e.y);
}

// ---- ants ----
function antSpit(e, g) {
  const p = g.player;
  const fx = e.x + e.dir * 15, fy = e.y - 15;
  if (!g.sees(e.x, e.y - 15, fx, fy) || !g.sees(fx, fy, p.x, p.y - 15)) return false;
  enemyShot(g, fx, fy, p.x + p.vx / 15 * 6, p.y - 15, 16, 7, 'acid');
  e.setState('weapon_fire');
  return true;
}

function ant(e, g) {
  const p = g.player, a = e.a;
  if (!a.st) a.st = e.type === 'HIDDEN_ANT' || e.lv?.hide_flag ? 'hiding' : 'start';
  if (e.hp <= 0) { killEffects(e, g, false); g.sound('antdie', e.x, e.y); return false; }
  if (a.cd > 0) a.cd--;
  const speed = e.def.abilities.get('run_top_speed') ?? 7;
  const { dx, dy } = nearestDist(e, p);
  const face = () => { e.dir = dx > 0 ? 1 : -1; };
  const contact = () => {
    if (!a.cd && g.touchesPlayer(e)) { g.hurtPlayer(e.state === 'run_jump' ? 10 : 6); a.cd = 6; }
  };
  const startJump = (vx, vy) => { e.vx = vx; e.vy = vy; e.setState('run_jump'); a.st = 'jump'; };
  if (e.state === 'flinch_up' || e.state === 'flinch_down') {
    if (!e.nextPicture()) e.setState('stopped');
    return true;
  }

  switch (a.st) {
    case 'start': e.setState('hanging'); a.st = 'hanging'; break;
    case 'hiding':
      e.hidden = true;
      e.shootable = false;
      if (Math.abs(dx) < 130 && e.y < p.y) {
        e.hidden = false;
        if (e.type === 'HIDDEN_ANT') g.changeType(e, 'ANT_ROOF');
        e.shootable = true;
        e.setState('falling');
        a.st = 'fall';
      }
      break;
    case 'hanging':
      e.shootable = false;
      if (Math.abs(dx) < 130 && e.y < p.y) {
        e.shootable = true;
        e.setState('fall_start');
        a.st = 'fall';
      } else if (g.rand(128) === 0) g.sound('antscare', e.x, e.y);
      break;
    case 'fall': {
      e.setState('falling');
      const m = fall(e, g);
      if (m.down) { e.setState('landing'); g.sound('antland', e.x, e.y); a.st = 'landing'; e.vx = 0; }
      break;
    }
    case 'landing':
      if (!e.nextPicture()) {
        if (!g.solidAt(e.x, e.y + 2)) { a.st = 'fall'; } else { e.setState('stopped'); a.st = 'running'; }
      }
      break;
    case 'running': {
      contact();
      if (g.rand(16) === 0) a.dodge = 1;
      if (a.dodge) {
        a.dodge = 0;
        if (g.rand(2) === 0) { startJump(e.dir * 11, -10); break; }
      }
      const toward = (dx > 0 && e.dir === 1) || (dx < 0 && e.dir === -1);
      if (!toward) { face(); e.setState('landing'); a.st = 'landing'; break; }
      e.nextPicture();
      if (g.rand(4) === 0 && Math.abs(dx) < 180 && Math.abs(dy) < 100 && g.sees(e.x + e.dir * 15, e.y - 15, p.x, p.y - 15)) {
        e.setState('fire_wait'); a.st = 'fire';
      } else if (Math.abs(dx) < 100 && Math.abs(dy) < 10 && g.rand(4) === 0) { e.setState('pounce_wait'); a.st = 'pounce'; a.t = 0; }
      else if (Math.abs(dx) > 140 && g.rand(3) === 0) startJump(e.dir * 11, -9);
      else {
        if (e.state !== 'running') e.setState('running');
        const m = g.moveEntity(e, e.dir * speed, 0);
        if (m.blockedX) {
          const up = g.moveEntity(e, 0, -speed);
          if (up.up) { e.dir = -e.dir; startJump(e.dir * 11, -9); }
          else g.moveEntity(e, e.dir * speed, 0);
        }
        const d = g.moveEntity(e, 0, 10);
        if (!d.down) a.st = 'fall';
      }
      break;
    }
    case 'pounce':
      contact();
      e.setState('pounce_wait');
      if (++a.t > 3) { g.sound('antslash', e.x, e.y); startJump(e.dir * 13, -8); }
      break;
    case 'jump': {
      contact();
      e.vy = Math.min(e.vy + GRAVITY, 24);
      const m = g.moveEntity(e, e.vx, e.vy);
      if (m.blockedX) e.vx = 0;
      if (m.up) e.vy = 0;
      e.setState(e.vy > 2 ? 'run_jump_fall' : 'run_jump');
      if (m.down) { e.vx = 0; e.vy = 0; e.setState('stopped'); a.st = 'running'; }
      break;
    }
    case 'fire':
      if (e.state === 'fire_wait') {
        if (!e.nextPicture()) { antSpit(e, g); if (e.state !== 'weapon_fire') e.setState('stopped'); a.st = 'running'; }
      }
      break;
    default: a.st = 'running';
  }
  return true;
}

// ---- flyers ----
function flyer(e, g) {
  const p = g.player, a = e.a;
  if (e.hp <= 0) { killEffects(e, g, true); return false; }
  if (a.cd > 0) a.cd--;
  if (e.state === 'flinch_up') { if (!e.nextPicture()) e.setState('running'); } else e.nextPicture();
  const tx = p.x + (e.x < p.x ? -90 : 90), ty = p.y - 70;
  const maxvx = e.lv?.max_xvel || 8, maxvy = e.lv?.max_yvel || 5;
  e.vx += Math.sign(tx - e.x) * 1.5; e.vy += Math.sign(ty - e.y) * 1;
  e.vx = Math.max(-maxvx, Math.min(maxvx, e.vx)); e.vy = Math.max(-maxvy, Math.min(maxvy, e.vy));
  const m = g.moveEntity(e, e.vx, e.vy);
  if (m.blockedX) e.vx = -e.vx * 0.5;
  if (m.up || m.down) e.vy = -e.vy * 0.5;
  e.dir = p.x > e.x ? 1 : -1;
  const delay = e.lv?.fire_delay || 18;
  if (!a.cd && Math.abs(p.x - e.x) < 260 && Math.abs(p.y - e.y) < 200 && g.sees(e.x, e.y - 8, p.x, p.y - 15)) {
    enemyShot(g, e.x + e.dir * 12, e.y - 8, p.x, p.y - 15, 16, 6, 'spark');
    a.cd = delay;
  }
  return true;
}

// ---- gun turrets ----
function turret(e, g) {
  const p = g.player, a = e.a;
  if (e.hp <= 0) { killEffects(e, g, true); return false; }
  const aimFrames = e.def.states.get('spinning') || e.def.states.get('spray.aim');
  const n = aimFrames?.length || 24;
  // Muzzle from the original spray_fire: (x + cos*20, y - 21 - sin*22)
  const baseY = e.y - 21;
  const canSee = Math.hypot(p.x - e.x, p.y - 15 - baseY) < 320 && g.sees(e.x, baseY, p.x, p.y - 15);
  const want = Math.atan2(-(p.y - 15 - baseY), p.x - e.x);
  if (canSee) {
    const target = Math.round(((want + 2 * Math.PI) % (2 * Math.PI)) / (2 * Math.PI) * n) % n;
    const diff = ((target - (a.frame ?? 0) + n * 1.5) % n) - n / 2;
    a.frame = (((a.frame ?? 0) + Math.sign(diff) * Math.min(1, Math.abs(diff)) + n) % n);
    a.aim = want;
    if (a.cd > 0) a.cd--;
    else if (Math.abs(diff) < 1.5) {
      const dmg = e.type === 'SPRAY_GUN' ? 5 : 8;
      enemyShot(g, e.x + Math.cos(want) * 20, baseY - Math.sin(want) * 22, p.x, p.y - 15, 18, dmg, 'spark');
      a.cd = e.type === 'SPRAY_GUN' ? 3 : (e.lv?.fire_delay || 8);
    }
  }
  e.state = aimFrames === e.def.states.get('spinning') ? 'spinning' : 'spray.aim';
  e.frame = Math.round(a.frame ?? 0) % n;
  return true;
}

// ---- items ----
const items = {
  hp_up(e, g) {
    e.nextPicture();
    if (g.touchesPlayer(e) && g.giveHealth(20)) { g.sound('health'); return false; }
    return true;
  },
  weapon_icon_ai(e, g) {
    if (!e.a.settled) { const m = g.moveEntity(e, 0, 10); if (m.down) e.a.settled = true; }
    if (!g.touchesPlayer(e)) return true;
    const pk = pickupFor(e.type);
    if (pk) g.giveAmmo(pk.weapon, pk.amount);
    g.sound('ammo');
    return false;
  },
  health_power_ai(e, g) {
    if (g.touchesPlayer(e)) { g.player.hp = g.player.maxhp; g.sound('health'); return false; }
    return true;
  },
  lava_ai(e, g) {
    e.nextPicture();
    if (g.touchesPlayer(e) && e.stateTime % 20 === 0) g.hurtPlayer(6);
    return true;
  },
  spring_ai(e, g) {
    if (e.links.length && link0(e).aistate === 0) return true;
    if (e.aistate === 0) {
      if (g.touchesPlayer(e)) {
        const p = g.player;
        p.vy = Math.max(-560, p.vy - Math.abs(e.yvel || 15) * 15);
        p.ground = false;
        e.setState('running');
        e.aistate = 1;
        g.sound('spring', e.x, e.y);
      }
    } else if (!e.nextPicture()) { e.setState('stopped'); e.aistate = 0; }
    return true;
  },
};

// ---- explosives ----
function bomb(e, g) {
  const big = e.type === 'BIG_BOMB';
  if (e.aistate === 0) {
    if (e.a.hit || activated(e, g)) { goState(e, 1); e.lv ??= {}; e.a.blink = e.a.hit ? 3 : (e.lv.blink_time ?? 14); }
    return true;
  }
  const t = e.a.blink;
  if (t < 1) {
    g.explode(e.x, e.y - 8, big ? 100 : 55, big ? 80 : 45, false);
    g.sound('explode', e.x, e.y);
    return false;
  }
  if (t < 10 || (t < 18 && t % 2 === 0) || (t < 30 && t % 3 === 0) || (t < 50 && t % 4 === 0)) { e.nextPicture(); g.sound('tick', e.x, e.y); }
  e.a.blink--;
  return true;
}

function mine(e, g) {
  const p = g.player;
  if (Math.abs(p.x - e.x) < 22 && Math.abs(p.y - 10 - e.y) < 28) {
    g.explode(e.x, e.y, 45, 30, false);
    g.sound('explode', e.x, e.y);
    return false;
  }
  e.nextPicture();
  return true;
}

// ---- doors, switches, logic ----
function door(e, g) {
  switch (e.aistate) {
    case 0:
      if (e.links.length && link0(e).aistate !== 0) { e.setState('running'); g.sound('swish', e.x, e.y); goState(e, 1); } else e.setState('stopped');
      break;
    case 1:
      if (!e.nextPicture()) { e.setState('blocking'); goState(e, 2); }
      break;
    case 2:
      if (e.links.length && link0(e).aistate === 0) { e.setState('walking'); g.sound('swish', e.x, e.y); goState(e, 3); }
      break;
    case 3:
      if (!e.nextPicture()) { e.setState('stopped'); goState(e, 0); }
      break;
    default:
  }
  return true;
}

function switcher(once) {
  return (e, g) => {
    e.nextPicture();
    const p = g.player;
    const near = Math.abs(p.x - e.x) < 20 && Math.abs(p.y - e.y) < 30;
    const act = g.pressed('action');
    switch (e.aistate) {
      case 0:
        if (near && act) { g.sound('switch', e.x, e.y); e.setState('running'); e.aistate = 1; }
        break;
      case 1: if (!act) e.aistate = once ? 3 : 2; break;
      case 2:
        if (near && act) { g.sound('switch', e.x, e.y); e.setState('stopped'); e.aistate = 4; }
        break;
      case 4: if (!act) e.aistate = 0; break;
      default:
    }
    return true;
  };
}

function sensor(e, g) {
  const p = g.player;
  if (e.aistate === 0) {
    if (Math.abs(p.x - e.x) < e.xvel && Math.abs(p.y - e.y) < e.yvel) e.aistate = e.hp || 1;
  } else if (!e.lv?.unoffable) {
    if (!e.hp) { if (Math.abs(e.x - p.x) > e.xacel || Math.abs(e.y - p.y) > e.yacel) e.aistate = 0; }
    else e.aistate--;
  }
  return true;
}

function deathSensor(e) {
  e.links = e.links.filter((target) =>
    !target.dead && target.state !== 'dead' && target.state !== 'blown_back_dead');
  if (e.links.length === 0) {
    e.setState('running');
    e.aistate = 1;
  }
  return true;
}

function setOnState(e, on) {
  e.aistate = on ? 1 : 0;
  e.setState(on ? 'on_state' : 'stopped');
}

const gates = {
  or_ai(e) { setOnState(e, e.links.some((l) => l.aistate !== 0)); return true; },
  and_ai(e) { setOnState(e, e.links.length > 0 && e.links.every((l) => l.aistate !== 0)); return true; },
  xor_ai(e) { setOnState(e, e.links.filter((l) => l.aistate !== 0).length % 2 === 1); return true; },
  not_ai(e) { if (e.links.length) setOnState(e, link0(e).aistate === 0); return true; },
  indicator_ai(e) { if (e.links.length) setOnState(e, link0(e).aistate !== 0); return true; },
  delay_ai(e) {
    const time = e.lv?.delay_time ?? 1;
    if (e.a.count > 0) {
      if (--e.a.count === 0) setOnState(e, e.aistate === 0);
    } else if (e.links.length && (e.aistate === 0) !== (link0(e).aistate === 0)) e.a.count = time || 1;
    return true;
  },
  pulse_ai(e) {
    const speed = e.lv?.pulse_speed || e.xvel || 10;
    e.a.t = (e.a.t || 0) + 1;
    if (e.a.t >= speed) { e.a.t = 0; setOnState(e, e.aistate === 0); }
    return true;
  },
};

function wall(big) {
  return (e, g) => {
    if (e.hp <= 0 || (e.links.length === 1 && link0(e).aistate !== 0)) {
      g.effect('EXPLODE1', e.x + 15, e.y - 7);
      if (big) { g.effect('EXPLODE1', e.x - 15, e.y - 22); g.effect('EXPLODE1', e.x, e.y - 20); }
      g.sound('hwall', e.x, e.y);
      g.explode(e.x, e.y - 12, big ? 110 : 50, big ? 40 : 20, false, true);
      return false;
    }
    return true;
  };
}

function nextLevel(e, g) {
  if (g.touchesPlayer(e) && g.pressed('action') && !g.transitioning) g.nextLevel(e.aistate);
  return true;
}

function teleporter(e, g) {
  if (!e.links.length) return true;
  if (e.aistate === 0) {
    if (g.touchesPlayer(e) && g.pressed('action')) { e.setState('running'); g.sound('teleport', e.x, e.y); e.aistate = 1; }
  } else if (!e.nextPicture()) {
    const dest = link0(e);
    g.teleportPlayer(dest.x, dest.y, 16);
    e.setState('stopped'); e.aistate = 0;
  }
  return true;
}

function effect(e) {
  return e.nextPicture();
}

const HINTS = [
  'Aim gun with mouse, fire with left mouse button',
  'Collect ammo to increase firing speed',
  'Press the down key to activate objects. This is a switch.',
  'This console saves the state of the game, press down',
  'Press down to activate platform',
  'Hold down the right mouse button to use special powers',
  'Use the CTRL & INS keys (or 1-7, or the mouse wheel) to select weapons',
  'Press the up key to climb ladders',
  'Press the down key to start!',
  'Shoot hidden walls to destroy them',
  'Shoot switch ball to activate',
  'Press down to teleport',
];
const HINT_VOICE = [
  'aimsave', 'ammosave', 'switch_1', 'savesave', 'platfo_1', 'poweru_1', 'weapon_1', 'ladder_1', 'starts_1', 'wallss_1', 'switch_2', 'telepo_1',
];

function trainMessage(e, g) {
  if (e.aistate === 0) {
    if (activated(e)) { g.showHelp(HINTS[e.aitype] ?? '', HINT_VOICE[e.aitype]); e.aistate = 1; }
    return true;
  }
  if (e.aistate === 100) return false;
  g.showHelp(HINTS[e.aitype] ?? '', HINT_VOICE[e.aitype]);
  e.aistate++;
  return true;
}

const AMBIENT = [
  'ambtech1', 'ambtech2', 'ambtech3', 'ambcave1', 'ambcave2', 'ambcave3', 'ambcave4', 'ambfrst2', 'scream02', 'scream03',
  'scream08', 'adie03', 'amb11', 'amb13', 'amb16', 'amb07', 'amb10',
];

function ambientSound(e, g) {
  if (!activated(e)) { e.aistate = 0; return true; }
  if (e.aistate === 0) {
    g.sound(AMBIENT[e.aitype] ?? AMBIENT[0], e.x, e.y, (e.yvel || 127) / 127);
    e.aistate = e.xvel + g.rand(e.xacel + 1);
    return e.xvel > 0;
  }
  e.aistate--;
  return true;
}

// Slowly changes the radius of the linked light (or the ambient level) in response to a switch.
function dimmer(e, g) {
  const sw = link0(e);
  const light = e.lights[0];
  const getV = () => (light ? light.outer : g.ambient);
  const setV = (v) => {
    if (light) { if (v > light.inner) light.outer = v; } else if (v >= 0 && v < 64) g.ambient = v;
  };
  switch (e.aistate) {
    case 0: if (sw && sw.aistate !== 0) goState(e, 1); break;
    case 1: if (e.stateTime > e.yvel) goState(e, 2); else setV(getV() - e.xvel * e.dir); break;
    case 2: if (sw) sw.aistate = 1; e.aistate = 3; break;
    case 3: if (sw && sw.aistate === 0) goState(e, 4); break;
    case 4: if (e.stateTime > e.yvel) goState(e, 5); else setV(getV() + e.xvel * e.dir); break;
    case 5: if (sw) sw.aistate = 4; e.aistate = 0; break;
    default: e.aistate = 0;
  }
  return true;
}

// Puts the second linked object (a teleport beam) at this marker when the first link switches on.
function switchMover(e) {
  if (e.links.length < 2) return false;
  const [sw, target] = e.links;
  if (e.aistate === 0) {
    if (sw.aistate !== 0) {
      target.x = e.x; target.y = e.y;
      if (e.xvel === 0) { target.fade = 15; e.aistate = 1; } else return false;
    }
    return true;
  }
  if (target.fade === 0) return false;
  target.fade--;
  return true;
}

function teleBeam(e, g) {
  e.nextPicture();
  e.a.up ??= e.dir > 0;
  if (e.a.up) {
    if (e.fade >= 12) { g.sound('amb16', e.x, e.y, 0.8); e.a.up = false; } else e.fade++;
  } else if (e.fade <= 5) { g.sound('amb16', e.x, e.y, 0.8); e.a.up = true; } else e.fade--;
  return true;
}

function powerUp(name) {
  return (e, g) => {
    e.nextPicture();
    if (!g.touchesPlayer(e)) return true;
    g.player.power = name;
    g.sound('health');
    g.toast(name === 'FAST' ? 'Speed power: hold right mouse button' : name === 'FLY' ? 'Flight power: hold right mouse button' : 'Power acquired');
    return false;
  };
}

function saveStation(e, g) {
  if (e.aistate === 0) {
    if (e.state !== 'stopped') e.setState('stopped');
    e.nextPicture();
    if (g.touchesPlayer(e) && g.pressed('action')) { e.setState('running'); e.aistate = 1; e.a.t = 0; g.sound('switch', e.x, e.y); g.setCheckpoint(e.x, e.y); }
  } else {
    e.nextPicture();
    if (++e.a.t > 6) { e.setState('stopped'); e.aistate = 0; }
  }
  return true;
}

// Teleporting doors: open as the player nears, press the action key to hop to the linked door.
function tpDoor(e, g) {
  const p = g.player;
  const other = e.links[0];
  const near = Math.abs(p.x - e.x) < 100 && Math.abs(p.y - e.y) < 80;
  const frames = e.frames().length;
  e.a.opening = near || !!other?.a.opening;
  if (e.xvel >= 0 && e.xvel < 64) g.ambient = e.xvel;
  if (e.a.opening) e.frame = Math.min(frames - 1, e.frame + 1); else e.frame = Math.max(0, e.frame - 1);
  if (other && !g.tpLatch && g.pressed('action') && Math.abs(p.x - e.x) < 20 && Math.abs(p.y - e.y) < 30) {
    g.teleportPlayer(other.x, other.y);
    g.tpLatch = true;
    g.sound('teleport', e.x, e.y);
  }
  return true;
}

function forceField(e, g) {
  if (e.a.endY === undefined) {
    let y = Math.floor(e.y);
    while (y < e.y + 400 && !g.tileSolid(Math.floor(e.x), y)) y++;
    e.a.endY = y;
  }
  if (activated(e)) {
    e.a.solidRect = { x0: e.x - 2, x1: e.x + 2, y0: e.y, y1: e.a.endY };
    e.a.beam = true;
    if (g.tickCount % 4 === 0) g.sound('swish', e.x, e.y);
  } else { e.a.solidRect = null; e.a.beam = false; }
  return true;
}

function lightning(e, g) {
  if (!activated(e)) return true;
  if (e.aistate === 0) {
    if (e.stateTime < e.aitype * 2) e.setState('stopped');
    else { e.setState('running'); g.sound('teleport', e.x, e.y); goState(e, 1); }
  } else if (!e.nextPicture()) { e.aistate = 0; e.stateTime = 0; e.setState('stopped'); }
  else if (g.touchesPlayer(e)) g.hurtPlayer(6);
  return true;
}

function antCrack(e, g) {
  if (e.aistate === 0) {
    const p = g.player;
    const go = e.links.length ? link0(e).aistate !== 0 : Math.abs(p.x - e.x) < 50 && Math.abs(p.y - e.y) < 70;
    if (go) e.aistate = 1;
    return true;
  }
  e.a.total ??= e.lv?.create_total || 1;
  switch (e.frame) {
    case 4: break;
    case 3: {
      const ant = g.spawn('ANT_ROOF', e.x + e.dir * 20, e.y);
      if (ant) {
        ant.dir = e.dir; ant.vx = e.dir * 11; ant.vy = -8;
        ant.setState('run_jump');
        ant.a.st = 'jump';
      }
      if (e.a.total <= 1) e.frame = 4; else { e.a.total--; e.frame = 0; }
      break;
    }
    default: e.frame++;
  }
  return true;
}

function boulder(e, g) {
  const a = e.a;
  if (e.links.length && link0(e).aistate === 0) return true;
  if (e.hp <= 0) { g.explode(e.x, e.y - 10, 30, 10, false); return false; }
  if (!a.init) { e.vx = e.xvel || -4; e.vy = e.yvel || 0; a.init = true; }
  e.nextPicture();
  if (a.cd > 0) a.cd--;
  e.vy += 1;
  const ox = e.vx, oy = e.vy;
  const m = g.moveEntity(e, e.vx, e.vy);
  if (m.down || m.up) {
    if (Math.abs(oy) > 3) g.sound('antland', e.x, e.y);
    e.vy = oy > 1 ? 2 - oy : -oy;
  } else if (m.blockedX) e.vx = -ox;
  if (!a.cd && g.touchesPlayer(e)) { g.hurtPlayer(15); a.cd = 4; }
  return true;
}

function jugger(e, g) {
  const p = g.player, a = e.a;
  if (e.hp <= 0) {
    if (e.state === 'dieing') {
      if (!e.nextPicture()) { g.effect('EXPLODE1', e.x, e.y - 20); g.sound('explode', e.x, e.y); return false; }
      return true;
    }
    e.setState('dieing');
    return true;
  }
  if (a.cd > 0) a.cd--;
  if (!a.cd && g.touchesPlayer(e)) { g.hurtPlayer(8); a.cd = 6; }
  const stationary = e.lv?.stationary || 0;
  switch (e.aistate) {
    case 0:
      if (!stationary) { e.setState('running'); goState(e, 1); } else if (e.stateTime > (e.aitype || 8)) { e.setState('weapon_fire'); goState(e, 2); }
      break;
    case 1: {
      e.dir = p.x > e.x ? 1 : -1;
      const img = g.spriteOf(e);
      const step = Math.max(2, Math.abs(img?.advance || 3));
      const ox = e.x, oy = e.y;
      g.moveEntity(e, e.dir * step, 0);
      if (!g.moveEntity(e, 0, 10).down) { e.x = ox; e.y = oy; }
      if (!e.nextPicture()) { e.setState('weapon_fire'); goState(e, 2); }
      break;
    }
    case 2:
      if (e.stateTime > 3) {
        g.projs.push({
          kind: 'grenade', x: e.x, y: e.y - 24, px: e.x, py: e.y - 24,
          vx: (e.lv?.throw_xvel || 8) * e.dir, vy: e.lv?.throw_yvel || -8, gravity: 2, life: 40, dmg: 40, radius: 50, mine: false, def: 'GRENADE', bounces: 0,
        });
        g.sound('throw', e.x, e.y);
        goState(e, 3);
      } else e.nextPicture();
      break;
    case 3:
      if (!e.nextPicture()) e.aistate = 0;
      break;
    default: e.aistate = 0;
  }
  return true;
}

// Elevators: links[0] and links[1] are the two end points (usually sensors), links[2] an optional enable switch.
// The platform travels to links[aitype] when that end's sensor fires or the rider presses the action key.
function platform(e, g) {
  const n = e.links.length;
  if (!(n === 2 || (n === 3 && e.links[2].aistate !== 0))) { e.setState('stopped'); return true; }
  if (e.state === 'stopped') e.setState('running'); else e.nextPicture();
  const speed = () => (e.aistate === 0 || e.yacel === 0 ? e.xacel : e.yacel) || 20;
  switch (e.aistate) {
    case 0:
      if ((e.links[e.aitype] && e.links[e.aitype].aistate !== 0) || (g.touchesPlayer(e) && g.pressed('action'))) goState(e, 2);
      break;
    case 2:
      g.sound('swish', e.x, e.y);
      e.aitype = 1 - e.aitype;
      e.xvel = speed();
      goState(e, 3);
      break;
    case 3: {
      const src = e.links[e.aitype], dst = e.links[1 - e.aitype];
      if (!src || !dst) { e.aistate = 0; break; }
      let nx, ny;
      if (e.xvel <= 0) { nx = dst.x; ny = dst.y; e.aistate = 0; } else {
        const sp = speed();
        nx = dst.x - Math.trunc(((dst.x - src.x) * e.xvel) / sp);
        ny = dst.y - Math.trunc(((dst.y - src.y) * e.xvel) / sp);
        e.xvel--;
      }
      g.pushRiders(e, nx - e.x, ny - e.y);
      e.x = nx; e.y = ny;
      break;
    }
    default: e.aistate = 0;
  }
  return true;
}

export const behaviors = {
  ant_ai: ant,
  flyer_ai: flyer,
  track_ai: turret,
  spray_gun_ai: turret,
  bomb_ai: bomb,
  air_mine_ai: mine,
  mine_ai: mine,
  sdoor_ai: door,
  strap_door_ai: door,
  switcher_ai: switcher(false),
  switch_once_ai: switcher(true),
  sensor_ai: sensor,
  death_sen_ai: deathSensor,
  hwall_ai: wall(false),
  big_wall_ai: wall(true),
  next_level_ai: nextLevel,
  tp2_ai: teleporter,
  platform_ai: platform,
  train_ai: trainMessage,
  amb_sound_ai: ambientSound,
  switch_dim_ai: dimmer,
  switch_mover_ai: switchMover,
  tele_beam_ai: teleBeam,
  fast_ai: powerUp('FAST'),
  fly_power_ai: powerUp('FLY'),
  sneaky_power_ai: powerUp('SNEAKY'),
  do_nothing: (e) => { e.nextPicture(); return true; },
  restart_ai: saveStation,
  tpd_ai: tpDoor,
  ff_ai: forceField,
  lightin_ai: lightning,
  crack_ai: antCrack,
  bolder_ai: boulder,
  jug_ai: jugger,
  exp_ai: effect,
  animate_ai: (e) => { e.nextPicture(); return true; },
  ...items,
  ...gates,
};
