// Game loop, level state, player, collision and rendering.
import { Renderer } from './renderer.js';
import { Assets } from './assets.js';
import { loadLevel } from './level.js';
import { Entity } from './entity.js';
import { behaviors, LOGIC_AI, SOLID_AI, isHiddenInPlay } from './behaviors.js';
import { WEAPONS, WEAPON_ORDER, firePlayer, updateProjectiles, drawProjectiles, SMALL_FIRE_OFF, LARGE_FIRE_OFF } from './weapons.js';
import { Audio } from './audio.js';
import { LightMap } from './lighting.js';

const VIEW_W = 320;
const VIEW_H = 200;
const STEP = 1 / 60;
const TICK = 1 / 15; // original game logic rate

// Original physics constants are per 15 Hz tick; converted to px/s here.
const RUN_SPEED = 120;
const JUMP_VEL = 225;
const GRAVITY = 675;
const MAX_FALL = 420;
const HALF_W = 6;
const BODY_H = 36;
const CLIMB_SPEED = 90;

const EXTRA_DEFS = [
  'DARNEL', 'GRENADE', 'ROCKET', 'FIREBOMB', 'ANT_ROOF', 'HIDDEN_ANT',
  'EXPLODE1', 'EXPLODE2', 'EXPLODE3', 'EXPLODE4', 'EXPLODE5', 'EXPLODE6', 'EXPLODE7', 'EXPLODE8',
  'CLOUD', 'SMALL_DARK_CLOUD', 'SMALL_LIGHT_CLOUD',
  ...WEAPON_ORDER.map((w) => WEAPONS[w].top),
];
const MIDDLE_DRAW = new Set(['exp_draw', 'middle_draw']);

export class Game {
  constructor(canvas, hud) {
    this.r = new Renderer(canvas, VIEW_W, VIEW_H);
    this.viewW = VIEW_W;
    this.viewH = VIEW_H;
    this.assets = new Assets();
    this.audio = new Audio();
    this.hud = hud;
    this.keys = new Set();
    this.mouse = null;
    this.mouseDown = false;
    this.god = false;
    this.lightMap = new LightMap();
    this.lightsOn = true;
    this.ambient = 32;
    this.pan = { x: 0, y: 0 };
    this.rightDown = false;
    this.godDeaths = JSON.parse(localStorage.getItem('abuse.godDeaths') || '{}');
    this.level = null;
    this.entities = [];
    this.solids = [];
    this.ladders = [];
    this.projs = [];
    this.player = null;
    this.cam = { x: 0, y: 0 };
    this.acc = 0;
    this.tickAcc = 0;
    this.rng = 12345;

    addEventListener('keydown', (e) => {
      this.keys.add(e.code);
      if (e.code.startsWith('Arrow') || e.code === 'Space') e.preventDefault();
      if (e.code === 'KeyG' && !e.repeat) this.setGod(!this.god);
      if (e.code === 'KeyL' && !e.repeat) this.lightsOn = !this.lightsOn;
      if (e.code === 'Insert') this.cycleWeapon(1);
      if (e.code === 'ControlRight') this.cycleWeapon(-1);
      const n = e.code.startsWith('Digit') ? parseInt(e.code.slice(5), 10) : 0;
      if (n >= 1 && n <= WEAPON_ORDER.length) this.selectWeapon(WEAPON_ORDER[n - 1]);
    });
    addEventListener('keyup', (e) => this.keys.delete(e.code));
    addEventListener('blur', () => this.keys.clear());
    canvas.addEventListener('mousemove', (e) => {
      const b = canvas.getBoundingClientRect();
      this.mouse = { x: ((e.clientX - b.left) / b.width) * VIEW_W, y: ((e.clientY - b.top) / b.height) * VIEW_H };
    });
    canvas.addEventListener('mousedown', (e) => { if (e.button === 0) this.mouseDown = true; if (e.button === 2) this.rightDown = true; });
    addEventListener('mouseup', () => { this.mouseDown = false; this.rightDown = false; });
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.cycleWeapon(e.deltaY > 0 ? 1 : -1);
    }, { passive: false });
  }

  setGod(on) {
    this.god = on;
    this.onGod?.(on);
  }

  async init() {
    await this.assets.init();
    this.r.setPalette(this.assets.palette);
    const t = this.assets.foreTile(1) || this.assets.foreTile(0);
    this.tw = t.w;
    this.th = t.h;
    const b = this.assets.backTile(0) || [...this.assets.back.keys()].map((k) => this.assets.backTile(k))[0];
    this.bw = b?.w || this.tw;
    this.bh = b?.h || this.th;
    const nearest = (r, g, bl) => {
      let best = 0, bd = Infinity;
      const pal = this.assets.palette;
      for (let i = 1; i < 256; i++) {
        const d = (pal[i * 4] - r) ** 2 + (pal[i * 4 + 1] - g) ** 2 + (pal[i * 4 + 2] - bl) ** 2;
        if (d < bd) { bd = d; best = i; }
      }
      return best;
    };
    this.colors = {
      white: nearest(255, 255, 255), yellow: nearest(255, 230, 90), cyan: nearest(120, 220, 255),
      green: nearest(90, 255, 90), orange: nearest(255, 140, 40),
      redBright: nearest(255, 0, 0), redDark: nearest(150, 0, 0),
    };
    this.player = {
      x: 0, y: 0, vx: 0, vy: 0, dir: 1, ground: false, anim: 0, state: 'stopped', aim: 0, aimAngle: 0,
      hp: 100, maxhp: 100, weapon: 'MGUN', ammo: { MGUN: 100 }, owned: new Set(['MGUN']), cooldown: 0,
      dead: false, deadTime: 0, climbing: false, ladderExit: null,
    };
  }

  rand(n) {
    this.rng = (this.rng * 1103515245 + 12345) & 0x7fffffff;
    return (this.rng >> 8) % n;
  }

  sound(name, x, y, volume = 1) {
    const p = this.player;
    this.audio.play(name, x === undefined ? null : { dx: x - p.x, dy: (y ?? p.y) - p.y }, volume);
  }

  async start(file) {
    const level = await loadLevel(file);
    const types = new Set(level.objects.map((o) => o.type));
    EXTRA_DEFS.forEach((t) => types.add(t));
    await Promise.all([...types].map((t) => this.assets.preloadDef(this.assets.defs.get(t))));
    this.r.resetAtlas();
    this.level = level;
    this.projs = [];
    this.ambient = 32;
    this.pan = { x: 0, y: 0 };
    this.transitioning = false;
    this.onLevel?.(level.name);

    const byIndex = level.objects.map((o) => {
      const def = this.assets.defs.get(o.type);
      if (!def || o.type === 'START') return null;
      const e = new Entity(def, o);
      e.lv = o.lv || {};
      e.shootable = def.flags.get('hurtable') === 'T' || /BOMB$/.test(o.type);
      e.logic = LOGIC_AI.has(e.ai);
      e.fade = o.fade || 0;
      e.lights = o.lights.map((i) => level.lights[i]);
      return e;
    });
    level.objects.forEach((o, i) => { if (byIndex[i]) byIndex[i].links = o.links.map((j) => byIndex[j]).filter(Boolean); });
    this.entities = byIndex.filter(Boolean);
    this.ladders = this.entities
      .filter((e) => e.ai === 'latter_ai' && e.links.length)
      .map((e) => ({ x0: e.x, y0: e.y, x1: e.links[0].x, y1: e.links[0].y }));
    this.refreshSolids();

    const start = level.objects.find((o) => o.type === 'START');
    this.startPos = { x: start ? start.x : 100, y: start ? start.y : 100 };
    this.respawn();
    this.cam.x = this.player.x - VIEW_W / 2;
    this.cam.y = this.player.y - VIEW_H / 2;
    if (!this.running) { this.running = true; this.last = performance.now(); requestAnimationFrame((t) => this.frame(t)); }
  }

  nextLevel(n) {
    if (!(n >= 0 && n <= 21)) return;
    this.transitioning = true;
    this.start(`level${String(n).padStart(2, '0')}.spe`);
  }

  respawn() {
    const p = this.player;
    let x = this.startPos.x, y = this.startPos.y;
    if (this.boxHits(x, y, HALF_W, BODY_H, null)) {
      let found = false;
      for (let d = 1; d <= 120; d++) {
        if (!this.boxHits(x, y + d, HALF_W, BODY_H, null)) { y += d; found = true; break; }
      }
      for (let dx = 3; dx <= 240 && !found; dx += 3) {
        for (const offset of [-dx, dx]) {
          for (let dy = 0; dy <= 120; dy += 3) {
            if (!this.boxHits(x + offset, y + dy, HALF_W, BODY_H, null)) {
              x += offset; y += dy; found = true; break;
            }
          }
          if (found) break;
        }
      }
    }
    Object.assign(p, {
      x, y, vx: 0, vy: 0, hp: p.maxhp, dead: false, deadTime: 0, climbing: false, ladderExit: null, cooldown: 0,
    });
  }

  frame(now) {
    const dt = Math.min(0.1, (now - this.last) / 1000);
    this.last = now;
    this.acc += dt;
    while (this.acc >= STEP) { this.update(STEP); this.acc -= STEP; }
    this.render();
    this.updateHud();
    requestAnimationFrame((t) => this.frame(t));
  }

  // ---- input helpers ----
  pressed(name) {
    const k = this.keys;
    if (name === 'action') return k.has('ArrowDown') || k.has('KeyS') || k.has('KeyE');
    return false;
  }

  selectWeapon(w) {
    if (this.player.owned.has(w)) this.player.weapon = w;
  }

  cycleWeapon(d) {
    const owned = WEAPON_ORDER.filter((w) => this.player.owned.has(w));
    const i = owned.indexOf(this.player.weapon);
    this.player.weapon = owned[(i + d + owned.length) % owned.length];
  }

  // ---- collision ----
  tileSolid(px, py) {
    const lv = this.level;
    if (px < 0 || px >= lv.fgW * this.tw) return true;
    if (py < 0 || py >= lv.fgH * this.th) return false;
    const tx = Math.floor(px / this.tw);
    const ty = Math.floor(py / this.th);
    const tile = this.assets.foreTile(lv.fgmap[ty * lv.fgW + tx]);
    if (!tile || !tile.mask) return false;
    return tile.mask[(py - ty * this.th) * tile.w + (px - tx * this.tw)] === 1;
  }

  solidAt(px, py) {
    px = Math.floor(px); py = Math.floor(py);
    if (this.tileSolid(px, py)) return true;
    for (const s of this.solids) if (px >= s.x0 && px <= s.x1 && py >= s.y0 && py <= s.y1) return true;
    return false;
  }

  boxHits(x, y, hw, bh, ignore) {
    const l = Math.floor(x - hw), r = Math.floor(x + hw), top = Math.floor(y - bh), yy = Math.floor(y);
    for (const s of this.solids) {
      if (s.e !== ignore && r >= s.x0 && l <= s.x1 && yy - 1 >= s.y0 && top <= s.y1) return true;
    }
    for (let py = top; py < yy; py += 4) if (this.tileSolid(l, py) || this.tileSolid(r, py)) return true;
    if (this.tileSolid(l, yy - 1) || this.tileSolid(r, yy - 1)) return true;
    for (let px = l; px <= r; px += 3) if (this.tileSolid(px, top) || this.tileSolid(px, yy - 1)) return true;
    return this.tileSolid(r, top);
  }

  sees(x1, y1, x2, y2) {
    const n = Math.max(1, Math.ceil(Math.hypot(x2 - x1, y2 - y1) / 3));
    for (let i = 1; i < n; i++) {
      if (this.tileSolid(Math.floor(x1 + ((x2 - x1) * i) / n), Math.floor(y1 + ((y2 - y1) * i) / n))) return false;
    }
    return true;
  }

  // The collision body is measured from a locomotion frame rather than whichever
  // pose is playing: transient frames (e.g. a roof ant's tall falling sprite)
  // would otherwise be cached and leave the entity wedged inside the ceiling.
  bodySprite(e) {
    if (e.def.file) {
      for (const s of ['stopped', 'running', 'walking', 'top_walk', 'climbing']) {
        const f = e.def.states.get(s);
        if (!f?.length) continue;
        const img = this.assets.sprite(e.def.file, f[0]);
        if (img) return img;
      }
    }
    return this.spriteOf(e);
  }

  box(e) {
    if (!e.hw) {
      const img = this.bodySprite(e);
      e.hw = Math.max(4, Math.floor((img?.w || 16) * 0.35));
      e.bh = Math.max(6, (img?.h || 20) - 3);
    }
    return e;
  }

  moveEntity(e, dx, dy) {
    const out = { blockedX: false, down: false, up: false };
    const { hw, bh } = this.box(e);
    const nx = Math.ceil(Math.abs(dx)), sx = Math.sign(dx);
    for (let i = 0; i < nx; i++) {
      const x = e.x + sx * Math.min(1, Math.abs(dx) - i);
      if (!this.boxHits(x, e.y, hw, bh, e)) { e.x = x; continue; }
      let up = 1;
      while (up <= 5 && this.boxHits(x, e.y - up, hw, bh, e)) up++;
      if (up <= 5) { e.x = x; e.y -= up; } else { out.blockedX = true; break; }
    }
    const ny = Math.ceil(Math.abs(dy)), sy = Math.sign(dy);
    for (let i = 0; i < ny; i++) {
      const y = e.y + sy * Math.min(1, Math.abs(dy) - i);
      if (!this.verticalHits(e, y, hw, bh, sy > 0)) { e.y = y; continue; }
      if (sy > 0) out.down = true; else out.up = true;
      break;
    }
    return out;
  }

  // Vertical movement only tests the moving edge, matching the original engine's
  // feet/head spine test: a falling object is blocked by its feet, a rising one
  // by its head. Testing the whole body would wedge tall poses (e.g. a roof ant
  // hanging just under the ceiling) permanently against the ceiling tile.
  verticalHits(e, y, hw, bh, down) {
    const cx = Math.floor(e.x);
    const edge = Math.floor(down ? y : y - bh + 1);
    if (this.tileSolid(cx, edge)) return true;
    for (const s of this.solids) {
      if (s.e === e) continue;
      if (e.x + hw >= s.x0 && e.x - hw <= s.x1 && edge >= s.y0 && edge <= s.y1) return true;
    }
    return false;
  }

  // ---- entity helpers ----
  spriteOf(e) {
    const f = e.frames();
    if (!f || !e.def.file) return null;
    return this.assets.sprite(e.def.file, f[e.frame % f.length]);
  }

  rectOf(e) {
    const img = this.spriteOf(e);
    if (!img) return null;
    const x0 = Math.round(e.x) - (e.dir < 0 ? img.w - img.xcfg - 1 : img.xcfg);
    const y0 = Math.round(e.y) - img.h + 1;
    return { x0, y0, x1: x0 + img.w - 1, y1: y0 + img.h - 1 };
  }

  playerRect() {
    const p = this.player;
    return { x0: p.x - HALF_W, x1: p.x + HALF_W, y0: p.y - BODY_H, y1: p.y };
  }

  touchesPlayer(e) {
    let r;
    if (e.shootable && e.hw) r = { x0: e.x - e.hw, x1: e.x + e.hw, y0: e.y - e.bh, y1: e.y };
    else r = this.rectOf(e);
    if (!r) return false;
    const p = this.playerRect();
    return r.x0 <= p.x1 && r.x1 >= p.x0 && r.y0 <= p.y1 && r.y1 >= p.y0;
  }

  refreshSolids() {
    this.solids = [];
    for (const e of this.entities) {
      if (e.dead) continue;
      if (e.a.solidRect) { this.solids.push({ ...e.a.solidRect, e }); continue; }
      if (!SOLID_AI.has(e.ai) || e.state === 'blocking') continue;
      const r = this.rectOf(e);
      if (r) this.solids.push({ ...r, e });
    }
  }

  // Moves the player along with a platform they are standing on.
  pushRiders(e, dx, dy) {
    const p = this.player;
    const r = this.rectOf(e);
    if (!r || p.dead || (!dx && !dy)) return;
    if (p.x + HALF_W >= r.x0 && p.x - HALF_W <= r.x1 && Math.abs(p.y - r.y0) <= 4) {
      p.x += dx; p.y += dy; p.vy = 0; p.ground = true;
    }
  }

  spawn(typeName, x, y) {
    const def = this.assets.defs.get(typeName);
    if (!def) return null;
    const e = new Entity(def, { x, y });
    e.logic = false;
    e.shootable = def.flags.get('hurtable') === 'T';
    this.entities.push(e);
    return e;
  }

  effect(typeName, x, y) { return this.spawn(typeName, x, y); }

  changeType(e, name) {
    const def = this.assets.defs.get(name);
    if (!def) return;
    e.def = def;
    e.type = name;
    e.ai = def.funs.get('ai_fun');
    e.hp = def.abilities.get('start_hp') ?? e.hp;
    e.setState('stopped');
    e.hw = 0;
  }

  teleportPlayer(x, y, yOffset = 0) {
    const p = this.player;
    let sx = x, sy = y - yOffset;
    if (this.boxHits(sx, sy, HALF_W, BODY_H, null)) {
      let found = false;
      for (let d = 3; d <= VIEW_W * 2 && !found; d += 3) {
        for (const dx of [-d, d]) {
          if (!this.boxHits(x + dx, sy, HALF_W, BODY_H, null)) { sx = x + dx; found = true; break; }
        }
      }
      for (let d = 3; d <= VIEW_H && !found; d += 3) {
        for (const dy of [-d, d]) {
          if (!this.boxHits(x, sy + dy, HALF_W, BODY_H, null)) { sy += dy; found = true; break; }
        }
      }
      if (found) this.toast('Teleporter landing adjusted to clear the wall');
    }
    p.x = sx; p.y = sy; p.vx = 0; p.vy = 0; p.ground = false;
  }

  damage(e, amount, pushX = 0, pushY = 0) {
    if (/BOMB$/.test(e.type)) { e.a.hit = true; return; }
    if (e.type === 'SWITCH_BALL') {
      if (e.state === 'stopped') { e.aistate = 1; e.setState('running'); this.sound('switch', e.x, e.y); }
      return;
    }
    e.hp -= amount;
    // Original do_damage passes push velocities to the victim's hurt function;
    // entity velocities here are px/tick, the same unit the C++ uses.
    if (pushX || pushY) { e.vx += pushX; e.vy += pushY; }
    if (e.hp > 0 && this.rand(3) === 0) {
      if (e.def.states.has('flinch_up') && !/^(HIDDEN|TRACK|SPRAY)/.test(e.type)) e.setState('flinch_up');
    }
  }

  explode(x, y, radius, dmg, fromPlayer, noEntities = false, exclude = null) {
    this.effect('EXPLODE1', x, y);
    this.effect('EXPLODE3', x + this.rand(10) - 5, y + this.rand(10) - 5);
    this.sound('explode', x, y);
    if (!noEntities) {
      // Linear falloff like the original hurt_radius: damage = (r - d) * m / r.
      for (const e of this.entities) {
        if (e.dead || e === exclude || !e.shootable) continue;
        const d = Math.hypot(e.x - x, e.y - 10 - y);
        if (d < radius) this.damage(e, Math.max(1, ((radius - d) * dmg) / radius));
      }
    }
    const p = this.player;
    const d = Math.hypot(p.x - x, p.y - 18 - y);
    if (d < radius) {
      this.hurtPlayer(Math.max(1, ((radius - d) * dmg * (fromPlayer ? 0.5 : 1)) / radius));
      p.vx += Math.sign(p.x - x || 1) * (1 - d / radius) * 180;
      p.vy -= (1 - d / radius) * 120;
    }
  }

  hurtPlayer(amount) {
    const p = this.player;
    if (p.dead) return;
    p.hp -= amount;
    if (p.hp > 0) return;
    if (this.god) { this.countGodDeath(); p.hp = p.maxhp; return; }
    p.hp = 0; p.dead = true; p.deadTime = 2; p.vx = 0; this.sound('die');
  }

  // In god mode the player survives, but each would-be death is tallied per level.
  countGodDeath() {
    const name = this.level.name;
    this.godDeaths[name] = (this.godDeaths[name] || 0) + 1;
    localStorage.setItem('abuse.godDeaths', JSON.stringify(this.godDeaths));
    this.sound('die');
  }

  toast(text) {
    this.msg = text;
    this.msgTime = 45;
  }

  // Training hint shown over the game, with the original voice-over when there is one.
  showHelp(text, voice) {
    this.onHelp?.(text);
    this.helpTime = 60;
    if (voice && this.helpVoice !== voice) { this.helpVoice = voice; this.audio.play(`voice/${voice}`, null); }
  }

  // Save stations move the respawn point and restore health.
  setCheckpoint(x, y) {
    this.startPos = { x, y };
    this.player.hp = this.player.maxhp;
    this.toast('Checkpoint saved');
  }

  giveHealth(n) {
    const p = this.player;
    if (p.hp >= p.maxhp || p.dead) return false;
    p.hp = Math.min(p.maxhp, p.hp + n);
    return true;
  }

  giveAmmo(weapon, n) {
    const p = this.player;
    p.ammo[weapon] = Math.min(999, (p.ammo[weapon] || 0) + n);
    if (!p.owned.has(weapon)) { p.owned.add(weapon); p.weapon = weapon; }
  }

  // ---- simulation ----
  update(dt) {
    const p = this.player;
    if (!p || !this.level) return;
    this.updatePlayer(dt);
    this.tickAcc += dt;
    while (this.tickAcc >= TICK) { this.tickAcc -= TICK; this.tick(); }
  }

  tick() {
    const p = this.player;
    if (!this.pressed('action')) this.tpLatch = false;
    this.tickCount = (this.tickCount || 0) + 1;
    if (this.msgTime > 0) this.msgTime--;
    if (this.helpTime > 0 && --this.helpTime === 0) { this.onHelp?.(''); this.helpVoice = null; }
    this.applyArea();
    for (const e of this.entities) { e.px = e.x; e.py = e.y; }
    for (const e of this.entities.slice()) {
      if (e.dead || !e.ai) continue;
      const fn = behaviors[e.ai];
      if (!fn) continue;
      if (!e.logic) {
        const [rx, ry] = e.def.range;
        const off = e.x < this.cam.x - rx - 20 || e.x > this.cam.x + VIEW_W + rx + 20
          || e.y < this.cam.y - ry - 40 || e.y > this.cam.y + VIEW_H + ry + 80;
        if (off) continue;
      }
      e.stateTime++;
      if (fn(e, this) === false) e.dead = true;
      // Anything that falls out of the world is removed (as the original does).
      else if (e.y > this.level.fgH * this.th + 160) e.dead = true;
    }
    updateProjectiles(this);
    // Dying objects are unlinked from everything else, like level::remove_object,
    // so gates/sensors that count them re-evaluate and can fire their links.
    const gone = new Set(this.entities.filter((e) => e.dead));
    this.entities = this.entities.filter((e) => !e.dead);
    if (gone.size) for (const e of this.entities) if (e.links.length) e.links = e.links.filter((l) => !gone.has(l));
    this.refreshSolids();
  }

  // The smallest area containing the player steers ambient light and the camera pan.
  applyArea() {
    const p = this.player;
    let best = null, size = Infinity;
    for (const a of this.level.areas) {
      if (p.x >= a.x && p.y >= a.y && p.x <= a.x + a.w && p.y <= a.y + a.h && a.w * a.h < size) { best = a; size = a.w * a.h; }
    }
    if (!best) return;
    const step = (cur, target, speed) => (speed > 0 ? cur + Math.max(-speed, Math.min(speed, target - cur)) : target);
    if (best.ambient >= 0) this.ambient = step(this.ambient, best.ambient, best.ambientSpeed);
    this.pan.x = step(this.pan.x, best.panX, best.panXSpeed);
    this.pan.y = step(this.pan.y, best.panY, best.panYSpeed);
  }

  inLadder(p) {
    if (p.ladderExit) {
      const l = p.ladderExit;
      if (p.x < l.x0 - 10 || p.x > l.x1 + 10 || p.y < l.y0 - 10 || p.y > l.y1 + 10 || this.keys.has('ArrowDown') || this.keys.has('KeyS')) {
        p.ladderExit = null;
      } else return null;
    }
    return this.ladders.find((l) => p.x >= l.x0 - 5 && p.x <= l.x1 + 5 && p.y >= l.y0 && p.y <= l.y1);
  }

  updatePlayer(dt) {
    const p = this.player;
    const k = this.keys;
    p.cooldown = Math.max(0, p.cooldown - dt);

    if (p.dead) {
      p.deadTime -= dt;
      p.vy = Math.min(MAX_FALL, p.vy + GRAVITY * dt);
      this.moveY(p, p.vy * dt);
      if (p.deadTime <= 0) { p.power = null; this.respawn(); }
      return;
    }

    const left = k.has('ArrowLeft') || k.has('KeyA');
    const right = k.has('ArrowRight') || k.has('KeyD');
    const up = k.has('ArrowUp') || k.has('KeyW');
    const down = k.has('ArrowDown') || k.has('KeyS');
    const ladder = this.inLadder(p);
    const jump = k.has('Space') || k.has('KeyZ') || (up && !ladder && !p.ladderExit);

    if (ladder && (up || down) && !p.climbing) p.climbing = true;
    if (!ladder) p.climbing = false;

    if (p.climbing) {
      if (jump) { p.climbing = false; p.vy = -JUMP_VEL; } else {
        p.vx = 0; p.vy = 0;
        p.x += (((ladder.x0 + ladder.x1) / 2) - p.x) * Math.min(1, dt * 10);
        const dy = ((down ? 1 : 0) - (up ? 1 : 0)) * CLIMB_SPEED * dt;
        if (up && p.y - ladder.y0 < 32) {
          p.y = ladder.y0;
          p.vy = 0;
          p.climbing = false;
          p.ladderExit = ladder;
          p.state = 'climb_off';
        } else {
          p.y += dy;
          p.anim += Math.abs(dy) * 1.2;
          p.state = 'climbing';
        }
      }
    }

    if (!p.climbing) {
      const target = (right ? 1 : 0) - (left ? 1 : 0);
      // Right mouse activates a held power; god mode always grants the FAST run.
      const power = (this.rightDown && p.power) || (this.god ? 'FAST' : null);
      const run = RUN_SPEED * (power === 'FAST' ? 1.7 : 1);
      const accel = p.ground ? 1400 : 700;
      if (target) { p.dir = target; p.vx += target * accel * dt; p.vx = Math.max(-run, Math.min(run, p.vx)); }
      else p.vx -= Math.sign(p.vx) * Math.min(Math.abs(p.vx), 1800 * dt);

      if (power === 'FLY') p.vy = ((down ? 1 : 0) - (up || jump ? 1 : 0)) * 110;
      else {
        if (jump && p.ground) { p.vy = -JUMP_VEL; p.ground = false; }
        p.vy = Math.min(MAX_FALL, p.vy + GRAVITY * dt);
      }

      const wasGround = p.ground;
      this.moveX(p, p.vx * dt);
      if (wasGround && p.vy >= 0) {
        let d = 0;
        while (d < 6 && !this.boxHits(p.x, p.y + 1, HALF_W, BODY_H, null) && !this.boxHits(p.x, p.y, HALF_W, BODY_H, null)) { p.y++; d++; }
      }
      this.moveY(p, p.vy * dt);
      if (p.y > this.level.fgH * this.th + 100) {
        if (this.god) this.countGodDeath();
        p.y = this.startPos.y; p.x = this.startPos.x; p.vy = 0;
      }

      // The original advances the player's frame once per 15 Hz tick in every
      // state (cop.cpp player_move calls next_picture each tick), so the idle
      // and run cycles both run at the one fixed rate.
      p.anim += dt / TICK;
      p.state = !p.ground ? (p.vy < 0 ? 'run_jump' : 'run_jump_fall') : Math.abs(p.vx) > 10 ? 'running' : 'stopped';
      const moving = Math.abs(p.vx) > 10;
      if (power === 'FLY') p.state = moving ? 'fly_running' : 'fly_stopped';
      else if (power === 'FAST' && p.ground && moving) p.state = 'fast_running';
    }

    if (this.mouse) {
      // cop.cpp top_ai: pick the upper-body frame by angular distance to the
      // pointer, pivoted at the gun grip (fire_off[12], fire_off[1]). The fire
      // angle (point_angle) is measured from the barrel tip to the pointer, or
      // is the frame's own angle when the pointer sits right at the muzzle.
      // The body-facing shift (x+4 when facing left) is applied first, exactly
      // like top_ai and player_fire_weapon do.
      const w = WEAPONS[p.weapon];
      const foff = w && w.large ? LARGE_FIRE_OFF : SMALL_FIRE_OFF;
      const base = p.dir < 0 ? p.x + 4 : p.x;
      const ix = foff[12], iy = foff[1];
      const mwx = this.mouse.x + this.cam.x, mwy = this.mouse.y + this.cam.y;
      let bestDeg = Math.atan2(p.y - iy - mwy, mwx - (base + ix)) * 180 / Math.PI;
      if (bestDeg < 0) bestDeg += 360;
      let best = 0, bd = Infinity;
      for (let i = 0; i < 24; i++) {
        let ta = Math.atan2(foff[i * 2 + 1] - iy, foff[i * 2] - ix) * 180 / Math.PI;
        if (ta < 0) ta += 360;
        let d = Math.abs(ta - bestDeg);
        if (d > 180) d = 360 - d;
        if (d < bd) { bd = d; best = i; }
      }
      p.aim = best;
      const fbX = foff[best * 2], fbY = foff[best * 2 + 1];
      let point;
      if (Math.abs(p.y - fbY - mwy) < 45 && Math.abs(mwx - (base + fbX)) < 40) {
        point = Math.atan2(fbY - iy, fbX - ix) * 180 / Math.PI;
      } else {
        point = Math.atan2(p.y - fbY - mwy, mwx - (base + fbX)) * 180 / Math.PI;
      }
      if (point < 0) point += 360;
      p.aimAngle = point; // degrees, like lisp_atan2
    } else {
      p.aimAngle = p.dir > 0 ? 0 : 180;
      p.aim = p.dir > 0 ? 0 : 12;
    }

    if (this.mouseDown || k.has('KeyF') || k.has('ControlLeft')) firePlayer(this);

    const tx = p.x - VIEW_W / 2 + p.dir * 24 + this.pan.x;
    const ty = p.y - BODY_H / 2 - VIEW_H / 2 + this.pan.y;
    this.cam.x += (tx - this.cam.x) * Math.min(1, dt * 6);
    this.cam.y += (ty - this.cam.y) * Math.min(1, dt * 6);
    this.cam.x = Math.max(0, Math.min(this.cam.x, this.level.fgW * this.tw - VIEW_W));
    this.cam.y = Math.max(0, Math.min(this.cam.y, this.level.fgH * this.th - VIEW_H));
  }

  moveX(p, dx) {
    const n = Math.ceil(Math.abs(dx));
    const s = Math.sign(dx);
    for (let i = 0; i < n; i++) {
      const nx = p.x + s * Math.min(1, Math.abs(dx) - i);
      if (!this.boxHits(nx, p.y, HALF_W, BODY_H, null)) { p.x = nx; continue; }
      let up = 1;
      while (up <= 8 && this.boxHits(nx, p.y - up, HALF_W, BODY_H, null)) up++;
      if (up <= 8) { p.x = nx; p.y -= up; } else { p.vx = 0; break; }
    }
  }

  moveY(p, dy) {
    const n = Math.ceil(Math.abs(dy));
    const s = Math.sign(dy);
    p.ground = false;
    for (let i = 0; i < n; i++) {
      const ny = p.y + s * Math.min(1, Math.abs(dy) - i);
      if (!this.boxHits(p.x, ny, HALF_W, BODY_H, null)) { p.y = ny; continue; }
      if (s > 0) p.ground = true;
      p.vy = 0;
      break;
    }
  }

  // ---- drawing ----
  blit(def, stateName, frame, x, y, dir, middle = false) {
    const frames = def.states.get(stateName) || def.states.get('stopped');
    if (!frames || !def.file) return;
    const img = this.assets.sprite(def.file, frames[Math.floor(frame) % frames.length]);
    if (!img) return;
    const ox = dir < 0 ? img.w - img.xcfg - 1 : img.xcfg;
    const yy = middle ? y + img.h / 2 : y;
    this.r.draw(img, Math.round(x - ox - this.cam.x), Math.round(yy - img.h + 1 - this.cam.y), { flip: dir < 0 });
  }

  render() {
    const lv = this.level;
    if (!lv) return;
    const r = this.r;
    const cx = Math.floor(this.cam.x), cy = Math.floor(this.cam.y);
    const alpha = Math.min(1, this.tickAcc / TICK);
    r.begin();
    if (this.lightsOn) r.setLightMap(this.lightMap.update(lv.lights, cx, cy, this.ambient));
    r.setLit(this.lightsOn);

    if (lv.bgW) {
      const bx = Math.floor(cx * lv.bgRate.xmul / lv.bgRate.xdiv);
      const by = Math.floor(cy * lv.bgRate.ymul / lv.bgRate.ydiv);
      const t0x = Math.floor(bx / this.bw), t0y = Math.floor(by / this.bh);
      for (let ty = t0y; ty * this.bh - by < VIEW_H; ty++) {
        for (let tx = t0x; tx * this.bw - bx < VIEW_W; tx++) {
          const id = lv.bgmap[(((ty % lv.bgH) + lv.bgH) % lv.bgH) * lv.bgW + (((tx % lv.bgW) + lv.bgW) % lv.bgW)];
          const t = this.assets.backTile(id);
          if (t) r.draw(t, tx * this.bw - bx, ty * this.bh - by, { opaque: true });
        }
      }
    }

    const x0 = Math.floor(cx / this.tw), y0 = Math.floor(cy / this.th);
    for (let ty = y0; ty <= y0 + Math.ceil(VIEW_H / this.th) && ty < lv.fgH; ty++) {
      for (let tx = x0; tx <= x0 + Math.ceil(VIEW_W / this.tw) && tx < lv.fgW; tx++) {
        const t = this.assets.foreTile(lv.fgmap[ty * lv.fgW + tx]);
        if (t) r.draw(t, tx * this.tw - cx, ty * this.th - cy);
      }
    }

    const fx = [];
    for (const e of this.entities) {
      if (e.dead || e.hidden) continue;
      const still = e.ai === 'platform_ai'; // riders move in 15 Hz steps, so keep the platform in step with them
      const x = still ? e.x : e.px + (e.x - e.px) * alpha, y = still ? e.y : e.py + (e.y - e.py) * alpha;
      if (x < cx - 160 || x > cx + VIEW_W + 160 || y < cy - 80 || y > cy + VIEW_H + 200) continue;
      if (isHiddenInPlay(e.def)) continue;
      if (e.ai === 'tele_beam_ai' && e.fade < 8 && (this.tickCount & 1)) continue;
      if (MIDDLE_DRAW.has(e.def.funs.get('draw_fun'))) { fx.push([e, x, y]); continue; }
      this.blit(e.def, e.state, e.frame, x, y, e.dir);
      if (e.a.beam) this.drawBeam(e);
    }

    const p = this.player;
    const body = this.assets.defs.get('DARNEL');
    // just_fired: the original draws the player with the bright tint for the
    // frame after firing (people.lsp player_draw / bright_tint).
    if (p.justFired) r.setBright(1.8);
    if (body) this.blit(body, p.dead ? 'dead' : p.state, p.anim, p.x, p.y, p.dir);
    const top = !p.dead && this.assets.defs.get(WEAPONS[p.weapon]?.top);
    if (top && !p.climbing) this.blit(top, 'stopped', p.aim, p.dir > 0 ? p.x : p.x + 4, p.y, 1);
    if (p.justFired) { r.setBright(1); p.justFired = false; }

    r.setLit(false);
    drawProjectiles(this, alpha);
    for (const [e, x, y] of fx) this.blit(e.def, e.state, e.frame, x, y, e.dir, true);
    r.flush();
  }

  // Force-field beam from the emitter down to the floor.
  drawBeam(e) {
    const { x0, x1, y0, y1 } = e.a.solidRect;
    const sx = Math.round((x0 + x1) / 2 - this.cam.x);
    const top = Math.round(y0 - this.cam.y), h = Math.round(y1 - y0);
    if (sx < -4 || sx > VIEW_W + 4 || top > VIEW_H || top + h < 0) return;
    const flick = this.rand(3);
    this.r.rect(sx - 1, top, 3, h, flick ? this.colors.cyan : this.colors.white);
    this.r.rect(sx, top, 1, h, this.colors.white);
  }

  updateHud() {
    const p = this.player;
    const w = WEAPONS[p.weapon];
    const owned = WEAPON_ORDER.map((n, i) => `${p.owned.has(n) ? (n === p.weapon ? '[' : ' ') : '-'}${i + 1}${n === p.weapon ? ']' : ' '}`).join('');
    const here = this.godDeaths[this.level.name] || 0;
    const total = Object.values(this.godDeaths).reduce((a, b) => a + b, 0);
    const ammo = this.god ? '∞' : (p.ammo[p.weapon] || 0);
    this.hud.textContent = `${this.level.name}  HP ${Math.ceil(p.hp)}  ${w.label} ${ammo}  ${owned}${this.god ? `  GOD deaths: ${here} here, ${total} total` : ''}${this.msgTime > 0 ? `  -- ${this.msg}` : ''}`;
  }
}
