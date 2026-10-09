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
      if (g.rand(20) === 0) a.dodge = 1;
      if (a.dodge) {
        a.dodge = 0;
        if (g.rand(2) === 0) { startJump(e.dir * 11, -10); break; }
      }
      const toward = (dx > 0 && e.dir === 1) || (dx < 0 && e.dir === -1);
      if (!toward) { face(); e.setState('landing'); a.st = 'landing'; break; }
      e.nextPicture();
      // Original ant (aistate 2 -> 8): 1/5 chance to fire when facing you.
      if (g.rand(5) === 0 && Math.abs(dx) < 180 && Math.abs(dy) < 100 && g.sees(e.x + e.dir * 15, e.y - 15, p.x, p.y - 15)) {
        e.setState('fire_wait'); a.st = 'fire';
      } else if (Math.abs(dx) < 100 && Math.abs(dx) > 10 && Math.abs(dy) < 10 && g.rand(5) === 0) { e.setState('pounce_wait'); a.st = 'pounce'; a.t = 0; }
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

// ---- flyers (flyer.lsp) ----
function flyer(e, g) {
  const p = g.player, a = e.a;
  const lv = e.lv || {};
  const maxvx = lv.max_xvel ?? 10, maxvy = lv.max_yvel ?? 5;

  // flyer_damage: smoke trail + knock-up whenever we take a hit.
  if (a.lastHp === undefined) a.lastHp = e.hp;
  else if (e.hp < a.lastHp) {
    a.smokeTime = 30;
    e.vy -= 14;
    e.setState('flinch_up');
  }
  a.lastHp = e.hp;
  if (a.smokeTime > 0) {
    a.smokeTime--;
    if (a.smokeTime % 2 === 0) g.effect('SMALL_DARK_CLOUD', e.x, e.y);
  }

  // Wait for the trigger before activating (aistate 0 = stopped/unhurtable).
  if (e.aistate === 0) {
    if (e.links.length === 0 || e.links[0].aistate !== 0) {
      if (!e.nextPicture()) { e.shootable = true; e.setState('running'); e.aistate = 1; }
    } else {
      e.shootable = false;
      e.setState('stopped');
    }
    return true;
  }

  // Awake flyers are always hittable, also when a restored PPO checkpoint
  // put one straight into aistate 1 without the wake-up transition above.
  e.shootable = true;

  // Dead: three explosions, then remove ourselves.
  if (e.hp <= 0) {
    g.effect('EXPLODE1', e.x + g.rand(10), e.y + g.rand(10) - 20);
    g.effect('EXPLODE1', e.x - g.rand(10), e.y - g.rand(10) - 20);
    g.effect('EXPLODE1', e.x, e.y - g.rand(20) - 20);
    return false;
  }

  // Flyer hum every 5 ticks.
  if (e.stateTime % 5 === 0) g.sound('robot02', e.x, e.y);

  // Chase the player horizontally (accel 1/tick, capped by max_xvel).
  if (p.x > e.x) {
    e.vx = Math.min(e.vx + 1, maxvx);
    if (e.dir === -1) { e.dir = 1; e.setState('turn_around'); }
  } else if (p.x < e.x) {
    e.vx = Math.max(e.vx - 1, -maxvx);
    if (e.dir === 1) { e.dir = -1; e.setState('turn_around'); }
  }

  // Hover in the band between 70px and 50px above the player's head.
  if (p.y - 70 > e.y) e.vy = e.vy > maxvy ? e.vy - 1 : e.vy + 1;
  else if (p.y - 50 < e.y) e.vy = e.vy < -maxvy ? e.vy + 1 : e.vy - 1;

  // Random jitter, 1 in 5 each way.
  if (g.rand(5) === 0) e.vx += 1; else if (g.rand(5) === 0) e.vx -= 1;
  if (g.rand(5) === 0) e.vy += 1; else if (g.rand(5) === 0) e.vy -= 1;

  // Advance the current animation; loop back to running once it finishes.
  if (!e.nextPicture()) e.setState('running');

  // bounce_move: half the velocity on any wall/floor/ceiling contact.
  const m = g.moveEntity(e, e.vx, e.vy);
  if (m.blockedX) e.vx = Math.trunc(e.vx / 2);
  if (m.down || m.up) e.vy = Math.trunc(e.vy / 2);

  // Burst fire a straight rocket at the player (flyer_cons: burst_total 2,
  // burst_delay 3, fire_delay 20; aitype 9 = STRAIT_ROCKET).
  a.fireTime ??= 0; a.burstWait ??= 0; a.burstLeft ??= 0;
  const fireDelay = lv.fire_delay ?? 20, burstDelay = lv.burst_delay ?? 3, burstTotal = lv.burst_total ?? 2;
  if (a.fireTime > 0) {
    a.fireTime--;
    if (a.fireTime === 0) { a.burstLeft = burstTotal; a.burstWait = 0; }
  } else if (a.burstWait === 0) {
    const facing = e.dir === (p.x > e.x ? 1 : -1);
    if (Math.abs(p.x - e.x) < 150 && facing) {
      const firex = e.x + e.dir * 10, firey = e.y;
      const playerx = p.x + (p.vx / 15) * 4, playery = p.y - 15 + (p.vy / 15) * 2; // port velocities are px/s
      if (g.sees(e.x, e.y, firex, firey) && g.sees(firex, firey, playerx, playery)) {
        const ang = Math.atan2(firey - playery, playerx - firex);
        g.projs.push({ kind: 'rocket', x: firex, y: firey, px: firex, py: firey, vx: Math.cos(ang) * 15, vy: -Math.sin(ang) * 15, life: 60, dmg: 15, radius: 25, mine: false, def: 'ROCKET', smoke: 0, straight: true });
        g.sound('mgun', firex, firey);
        if (a.burstLeft <= 1) a.fireTime = fireDelay;
        else a.burstLeft--;
        a.burstWait = burstDelay;
      }
    }
  } else {
    a.burstWait--;
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
      // Original hurt_radius: hwall x+15*dir,y-7 r=50 m=60 ; big_wall x,y-15 r=110 m=120.
      // The blast damages the neighbouring walls, so the whole linked floor goes up at once.
      // Both pass (bg) as hurt_radius's excluded object (doors.lsp), so the
      // blast never hurts or throws the player who shot the wall.
      if (big) g.explode(e.x, e.y - 15, 110, 120, false, false, e, true);
      else g.explode(e.x + 15 * e.dir, e.y - 7, 50, 60, false, false, e, true);
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
    if (activated(e)) {
      // Original train_ai (general.lsp): the voice-over plays once when the
      // hint activates; the per-tick calls below only refresh the text.
      g.showHelp(HINTS[e.aitype] ?? '', HINT_VOICE[e.aitype]);
      e.aistate = 1;
    }
    return true;
  }
  if (e.aistate === 100) return false;
  g.showHelp(HINTS[e.aitype] ?? '');
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

// common.lsp push_char: a player level with or above the object, within
// `ya` of it vertically and `xa` horizontally, is shoved out sideways to
// exactly `xa` away (walls permitting).
function pushChar(e, g, xa, ya) {
  const p = g.player;
  // (+2: the port's player stands a pixel lower than an object on the same floor)
  if (p.dead || p.y > e.y + 2 || Math.abs(p.y - e.y) >= ya || Math.abs(p.x - e.x) >= xa) return;
  g.pushPlayer(p.x > e.x ? xa - (p.x - e.x) : (e.x - p.x) - xa);
}

// ROB1, the cleaner robot — jugger.lsp rob1_ai. It waits (hidden, if
// rob_hiden is set) for its link to switch on, then trundles the way it faces
// at xvel pixels a tick, shoving the player along in front of it, until it
// can no longer see 23 pixels ahead. It never turns round. It takes 70
// damage to destroy.
function rob1(e, g) {
  if (e.fade > 0) e.fade--;
  const hidden = e.lv?.rob_hiden === 1;
  switch (e.aistate) {
    case 0:
      if (hidden) { e.shootable = false; e.setState('rob_hiding'); } // the empty frame
      else { e.shootable = true; pushChar(e, g, 30, 55); }
      if (e.links.length < 1 || link0(e).aistate !== 0) {
        if (hidden) e.fade = 15;
        goState(e, 1);
      }
      return true;
    case 1: {
      e.shootable = true;
      pushChar(e, g, 30, 55);
      // the one-frame hiding sequence ends at once, and a finished sequence
      // falls back to "stopped" (objects.cpp next_sequence): the robot appears
      if (!e.nextPicture() && e.state === 'rob_hiding') e.setState('stopped');
      if (e.stateTime % 6 === 0) g.sound('cleaner', e.x, e.y);
      g.moveEntity(e, 0, 10); // (try_move 0 10): settle onto the floor
      const eye = e.y - 10;   // sight line at body height (e.y is the floor line here)
      const step = Math.abs(e.xvel) || 2;
      if (e.dir > 0) { if (g.sees(e.x, eye, e.x + step + 23, eye)) e.x += step; }
      else if (g.sees(e.x, eye, e.x - step - 23, eye)) e.x -= step;
      if (e.hp <= 0) {
        for (const [dx, dy] of [[5, 10], [-5, 15], [10, 2], [-10, 20], [20, 27], [-25, 30], [20, 5], [-3, 1]]) g.effect('EXPLODE1', e.x + dx, e.y - dy);
        g.sound('explode', e.x, e.y);
        goState(e, 2);
      }
      return true;
    }
    default:
      pushChar(e, g, 30, 55);
      return e.stateTime < 3;
  }
}

// BOLDER — duong.lsp bolder_ai on objects.cpp float_tick. The boulder hangs
// where it was placed until its first link (a sensor or switch) is on, then
// falls and bounces. float_tick makes one straight move and stops at the first
// thing it hits; it does not walk up steps like a character (so a boulder
// bounces off a ledge instead of rolling over it), and what it hit decides the
// bounce: off a floor or ceiling the fall is reversed (keeping all but 2 of
// its speed), off a wall the roll is.
function boulder(e, g) {
  const a = e.a;
  if (e.links.length && link0(e).aistate === 0) return true;
  if (e.hp <= 0) { g.explode(e.x, e.y - 10, 30, 10, false); return false; }
  // The level stores each boulder's own speed; 0 is a real value (a boulder
  // that drops straight down), not "use the default".
  if (!a.init) { e.vx = e.xvel; e.vy = e.yvel; a.init = true; }
  e.nextPicture();
  if (a.cd > 0) a.cd--;
  e.vy += 1;
  const ox = e.vx, oy = e.vy;
  // It collides with its sprite's boundary outline (bold.spe: 38 x 27, inside
  // a 51 x 43 picture), not the whole picture. Measured by the picture, a
  // boulder set in a shaft just taller than the outline is "already inside
  // the ceiling" and never moves at all.
  if (!a.box) {
    const r = g.rectOf(e), d = g.deckRect(e);
    a.box = r && d ? { hw: Math.max(e.x - d.x0, d.x1 - e.x), top: e.y - d.y0, bot: e.y - d.y1 } : { hw: 19, top: 35, bot: 8 };
  }
  const { hw, top, bot } = a.box;
  const free = (x, y) => !g.boxHits(x, y - bot, hw, top - bot, e);
  // the straight move, a pixel at a time
  const n = Math.max(Math.abs(ox), Math.abs(oy));
  let k = 0;
  while (k < n && free(e.x + (ox * (k + 1)) / n, e.y + (oy * (k + 1)) / n)) k++;
  const dx = n ? (ox * k) / n : 0, dy = n ? (oy * k) / n : 0;
  g.pushRiders(e, dx, dy); // platform_push: whoever stands on it goes with it
  e.x += dx; e.y += dy;
  if (k < n) {
    // which way was it blocked? (float_tick tests one pixel along each axis)
    let vert, horiz;
    if (ox === 0) { vert = true; horiz = false; } else if (oy === 0) { vert = false; horiz = true; } else {
      horiz = !free(e.x + Math.sign(ox), e.y);
      vert = !free(e.x, e.y + Math.sign(oy));
      if (!horiz && !vert) horiz = vert = true;
    }
    if (vert) {
      if (Math.abs(oy) > 3) g.sound('antland', e.x, e.y);
      e.vx = ox;
      e.vy = oy > 1 ? 2 - oy : -oy;
    } else if (horiz) { e.vy = oy; e.vx = -ox; }
  }
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
    case 0: {
      const sensorOn = e.links[e.aitype] && e.links[e.aitype].aistate !== 0;
      const boarding = g.touchesPlayer(e) && g.pressed('action');
      if (!sensorOn && !boarding) break;
      // platform.lsp: when the rider presses the action key while touching,
      // they are set to (y - start_accel) — 22 small, 26 big, 72 red — which is
      // on or just above the deck, and the engine's next tick() settles them
      // onto it (objects.cpp tick: drops of up to 11 px are taken at once).
      // The net effect is the rider standing on the deck when it departs.
      if (boarding) {
        const deck = g.deckRect(e);
        if (deck && g.overDeck(deck)) { g.player.y = deck.y0; g.player.vy = 0; g.player.ground = true; }
      }
      goState(e, 2);
      // Fall through: the original's go_state re-enters platform_ai in the same
      // tick, so departure sound + speed are set and the first step is taken now.
    }
    case 2:
      g.sound('eleacc01', e.x, e.y); // PLAT_A_SND
      e.aitype = 1 - e.aitype;
      e.xvel = speed();
      goState(e, 3);
      // Fall through to take the first movement step this same tick.
    case 3: {
      const src = e.links[e.aitype], dst = e.links[1 - e.aitype];
      if (!src || !dst) { e.aistate = 0; break; }
      let nx, ny;
      if (e.xvel <= 0) { nx = dst.x; ny = dst.y; e.aistate = 0; } else {
        if (e.xvel === 6) g.sound('eledec01', e.x, e.y); // PLAT_D_SND
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

// ---- common.lsp / general.lsp / duong.lsp environment objects ----

// OBJ_MOVER — the compiled C mover_ai (cop.cpp): the mover drags its second
// linked object towards its first link over `aitype` ticks, then hands it off.
function mover(e) {
  if (e.links.length !== 2) return true;
  const [dest, carried] = e.links;
  if (e.aistate < 2) {
    e.links = [dest];
    if (!dest.links.includes(carried)) dest.links.push(carried);
    dest.aistate = dest.aitype;
  } else {
    e.aistate--;
    const frames = e.aitype || 20; // mover_cons: (set_aitype 20)
    carried.x = dest.x - Math.trunc(((dest.x - e.x) * e.aistate) / frames);
    carried.y = dest.y - Math.trunc(((dest.y - e.y) * e.aistate) / frames);
  }
  return true;
}

// PUSHER — while its switch is on and the player touches it, shoves the player
// along its direction by pusher_speed per tick. In the original, `touching_bg`
// tests overlap with the nearest player and `(bg)` is that player (clisp.cpp
// case 4 / case 22), so the pusher pushes the player, not a block.
function pusher(e, g) {
  if (!activated(e)) return true;
  e.nextPicture();
  const r = g.rectOf(e), pr = g.playerRect();
  if (r && r.x0 <= pr.x1 && r.x1 >= pr.x0 && r.y0 <= pr.y1 && r.y1 >= pr.y0) {
    g.moveX(g.player, (e.dir > 0 ? 1 : -1) * (e.aistate || 4));
  }
  return true;
}

// OBJ_HOLDER — holds its first linked object at an (xvel, yvel) offset from the
// second linked object; the third link is an optional enable switch.
function holder(e) {
  const pin = (ref) => {
    e.links[0].x = ref.x + e.xvel;
    e.links[0].y = ref.y + e.yvel;
    e.x = ref.x + e.xvel;
    e.y = ref.y + e.yvel;
  };
  switch (e.links.length) {
    case 2: pin(e.links[1]); return true;
    case 3:
      if (e.links[2].aistate !== 0) { pin(e.links[1]); return true; }
      return e.xacel !== 1; // xacel==1 removes the holder until re-enabled
    case 4: case 5: case 6: return true;
    default: return false; // 0 or 1 links: die, like the original
  }
}

// BLOCK — destructible scenery brick (hp 30, can_block); crumbles when destroyed.
function block(e, g) {
  if (e.hp <= 0) {
    if (e.state !== 'dieing') { g.sound('crumble', e.x, e.y); e.setState('dieing'); return true; }
    return e.nextPicture();
  }
  return true;
}

// STEP — a step that exists while its switch is OFF (running = "step_gone").
function step(e) {
  if (e.links.length === 0 || link0(e).aistate !== 0) e.setState('stopped');
  else e.setState('running');
  return true;
}

// SWITCH_DELAY — press to toggle on, auto-resets after reset_time ticks
// (switch_delay_cons: (setq reset_time 14)).
function switchDelay(e, g) {
  e.a.reset ??= 14;
  switch (e.aistate) {
    case 0:
      e.nextPicture();
      if (Math.abs(g.player.x - e.x) < 20 && Math.abs(g.player.y - e.y) < 30 && g.pressed('action')) {
        g.sound('switch', e.x, e.y);
        e.setState('running');
        e.aistate = 1;
      }
      break;
    case 1: if (!g.pressed('action')) e.aistate = 2; break;
    case 2:
      if (e.stateTime > e.a.reset) { g.sound('switch', e.x, e.y); e.setState('stopped'); e.aistate = 0; }
      break;
    default:
  }
  return true;
}

// DEATH_RESPAWNER — when a watched linked object dies, spawn a fresh one of the
// first link's type where it fell. (Watched by reference: the engine unlinks
// dead objects at the end of the tick, before this would see them in links.)
function deathRespawner(e, g) {
  e.a.watch ??= e.links.slice(1);
  const watch = e.a.watch;
  for (const w of watch) {
    if (w.dead || w.state === 'dead' || w.state === 'blown_back_dead') {
      g.spawn(link0(e).type, w.x, w.y);
      watch.splice(watch.indexOf(w), 1);
      break;
    }
  }
  return true;
}

// LIGHTHOLD — follows its linked object and drags its light along.
function lightHold(e, g) {
  if (e.links.length) {
    const t = link0(e);
    e.x = t.x;
    const img = g.spriteOf(e);
    e.y = t.y - Math.floor((img?.h || 0) / 2);
  }
  if (e.lights && e.lights.length === 1) { e.lights[0].x = e.x; e.lights[0].y = e.y; }
  return true;
}

// NEXT_LEVEL_TOP — casts down up to 100 px to record how far the floor is.
function nextLevelTop(e, g) {
  if (!e.a.init) {
    const oy = e.y;
    g.moveEntity(e, 0, 100);
    e.a.floorYoff = e.y - oy;
    e.y = oy;
    e.a.init = true;
  }
  return true;
}

// LADDER — the climbable region is already computed once at level load
// (game.js builds this.ladders from these entities); nothing to run per tick.
const ladder = () => true;

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
  rob1_ai: rob1,
  jug_ai: jugger,
  exp_ai: effect,
  animate_ai: (e) => { e.nextPicture(); return true; },
  mover_ai: mover,
  pusher_ai: pusher,
  holder_ai: holder,
  block_ai: block,
  step_ai: step,
  switch_delay_ai: switchDelay,
  death_re_ai: deathRespawner,
  lhold_ai: lightHold,
  next_level_top_ai: nextLevelTop,
  latter_ai: ladder,
  ...items,
  ...gates,
};
