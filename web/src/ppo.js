// PPO autopilot for the demo mode: a policy network trained in the browser to
// reach the level exits.
//
// Training signal comes straight from the port's own navigation model: the
// same tile-grid Dijkstra the heuristic bot uses is run periodically and its
// distance-to-the-nearest-exit is the progress reward. On top of that:
//   + big bonus for touching an exit (level complete)
//   - penalty for dying
//   - small per-step cost, extra stagnation penalty for not improving
//   + per-enemy-killed bonus
// The agent sees a local tile window (solids, enemies, exits, specials) plus
// scalar state, and outputs one of 24 discrete input combos (move x jump x
// down x fire). Aiming is heuristic (nearest enemy in sight), like a human.
//
// PPO: clipped surrogate + GAE + entropy bonus, Adam, separate actor/critic.
// Weights persist to localStorage so the trained policy can drive the demo.

// Bumped whenever what the policy sees, is paid for, or the game's physics
// change, so stale weights and stale recorded runs are not silently reused
// (the previous ones stay in localStorage under their old keys).
export const PPO_KEY = 'abuse.ppo.v6'; // v6: the agent also sees the nearest health heart and weapon/ammo pickup (7 inputs more than v4)
export const BEST_KEY = 'abuse.ppo.bestrun5'; // 5: a record is a chain of pieces from the level's start to its best point
export const HIST_KEY = 'abuse.ppo.besthistory5';
export const CAMPAIGN_KEY = 'abuse.ppo.campaign1'; // the levels passed so far, and where the next one starts
export const STATIONS_KEY = 'abuse.ppo.stations1'; // save stations reached on the current level: the places runs may start from

const ACTS = 24;            // move(-1..1) x jump x down x fire
const ROLLOUT = 1024;       // decisions collected per policy update
const BATCH = 32;           // samples per Adam step
const EPOCHS = 2;           // passes over each rollout
const REPLAY_SPEED = 6;     // recorded attempts are replayed at 6x real time
const REPLAY_TAIL = 60 * 45;   // the last 45 game-seconds of a failed attempt are shown (60 Hz steps)
const REPLAY_WIN = 60 * 150;   // ...and up to the last 150 game-seconds of one that reached the exit
const HIDDEN = 1e12;        // renderThrottle value that never draws
// Reward weights. Exploring is paid on a par with following the compass, so a
// compass that points the wrong way (or at a door it can't open) cannot pin
// the agent in place: somewhere it has rarely been is always worth going to.
const COMPASS = 4;          // per cell closer to the exit along the walking route (was 8)
const EXPLORE = 3;          // for a 40 px cell no run has reached before; falls as 1/sqrt(runs that have)
const SWITCH_PULL = 1;       // per cell closer to the nearest switch still off, while the exit itself can be steered for
const SWITCH_ON = 10;       // for turning a switch on (once per switch per run)
const HEALTH = 0.1;          // per point of health regained (a heart restores 20, so +2)
const PICKUP = 1;           // for collecting ammo; a weapon the player did not have pays 3
const STATION = 5;          // for using a save station (once per station per run)
// When a run is given up on (all in 60 Hz game steps). "Progress" is reaching
// a cell new to this run, a new best distance, a switch or a save station.
//  - really stuck: no progress for 60 game-seconds AND it has not left the
//    area it is in (AREA pixels each way) in that time;
//  - going nowhere: no progress for 3 game-minutes, wherever it has wandered;
//  - and no level gets more than 20 game-minutes.
// A run that is still moving through the level, backtracking across rooms it
// has already seen, is therefore left alone for a good while.
const STALL = 15 * 60 * 4;
const STALL_ROAMING = 15 * 180 * 4;
const LEVEL_CAP = 15 * 1200 * 4;
const AREA = 220;
export const VISITS_KEY = 'abuse.ppo.visits1';
const WIN_C = 20, WIN_R = 12, CH = 4;
export const OBS_N = WIN_C * WIN_R * CH + 20;

const ENEMY_AI = new Set([
  'ant_ai', 'flyer_ai', 'track_ai', 'spray_gun_ai', 'jug_ai',
  'crack_ai', 'lightin_ai', 'bolder_ai', 'air_mine_ai', 'mine_ai',
]);

const rand = () => Math.random() * 2 - 1;

class Dense {
  constructor(inN, outN) {
    this.inN = inN;
    this.outN = outN;
    this.w = new Float32Array(inN * outN);
    this.b = new Float32Array(outN);
    const s = Math.sqrt(1 / inN);
    for (let i = 0; i < this.w.length; i++) this.w[i] = rand() * s;
    this.gw = new Float32Array(this.w.length);
    this.gb = new Float32Array(outN);
    this.mw = new Float32Array(this.w.length);
    this.vw = new Float32Array(this.w.length);
    this.mb = new Float32Array(outN);
    this.vb = new Float32Array(outN);
  }
  forward(x, out) {
    this.x = x;
    const n = this.outN, W = this.inN;
    for (let o = 0; o < n; o++) {
      let s = this.b[o];
      const off = o * W;
      for (let i = 0; i < W; i++) s += this.w[off + i] * x[i];
      out[o] = s;
    }
    return out;
  }
  backward(dOut) {
    const n = this.outN, W = this.inN;
    this.gb.set(dOut);
    for (let o = 0; o < n; o++) {
      const d = dOut[o];
      const off = o * W;
      for (let i = 0; i < W; i++) this.gw[off + i] = d * this.x[i];
    }
  }
  // One Adam step on the accumulated gradients (scaled, e.g. by 1/batch).
  applyGrads(lr, scale = 1) {
    for (let i = 0; i < this.w.length; i++) {
      const g = this.gw[i] * scale;
      this.mw[i] = 0.9 * this.mw[i] + 0.1 * g;
      this.vw[i] = 0.999 * this.vw[i] + 0.001 * g * g;
      this.w[i] -= lr * this.mw[i] / (Math.sqrt(this.vw[i]) + 1e-5);
      this.gw[i] = 0;
    }
    for (let i = 0; i < this.b.length; i++) {
      const g = this.gb[i] * scale;
      this.mb[i] = 0.9 * this.mb[i] + 0.1 * g;
      this.vb[i] = 0.999 * this.vb[i] + 0.001 * g * g;
      this.b[i] -= lr * this.mb[i] / (Math.sqrt(this.vb[i]) + 1e-5);
      this.gb[i] = 0;
    }
  }
}

class MLP {
  constructor(inN, hidden, outN) {
    this.inN = inN;
    this.hidden = hidden;
    this.outN = outN;
    this.d1 = new Dense(inN, hidden);
    this.d2 = new Dense(hidden, hidden);
    this.do = new Dense(hidden, outN);
    this.h1 = new Float32Array(hidden);
    this.h2 = new Float32Array(hidden);
    this.out = new Float32Array(outN);
    this.nz = new Int32Array(inN); // indices of the non-zero inputs of the last forward()
    this.nn = 0;
    this.dh1 = new Float32Array(hidden);
    this.dh2 = new Float32Array(hidden);
  }
  forward(x) {
    this.x = x;
    // The observation is mostly zeros (a tile window of 0/1 flags), so the
    // first layer only visits the inputs that are set.
    const nz = this.nz;
    let nn = 0;
    for (let j = 0; j < this.inN; j++) if (x[j] !== 0) nz[nn++] = j;
    this.nn = nn;
    for (let i = 0; i < this.hidden; i++) {
      let s = this.d1.b[i];
      const off = i * this.inN;
      for (let k = 0; k < nn; k++) { const j = nz[k]; s += this.d1.w[off + j] * x[j]; }
      this.h1[i] = Math.tanh(s);
    }
    for (let i = 0; i < this.hidden; i++) {
      let s = this.d2.b[i];
      const off = i * this.hidden;
      for (let j = 0; j < this.hidden; j++) s += this.d2.w[off + j] * this.h1[j];
      this.h2[i] = Math.tanh(s);
    }
    for (let i = 0; i < this.outN; i++) {
      let s = this.do.b[i];
      const off = i * this.hidden;
      for (let j = 0; j < this.hidden; j++) s += this.do.w[off + j] * this.h2[j];
      this.out[i] = s;
    }
    return this.out;
  }
  // Backprop dE/d(out) for the sample just passed through forward(), ADDING
  // to each layer's gradient arrays; applyGrads() then takes one Adam step for
  // the whole minibatch and clears them.
  backward(dOut) {
    const h = this.hidden;
    const dh2 = this.dh2.fill(0);
    for (let o = 0; o < this.outN; o++) {
      const d = dOut[o];
      const off = o * h;
      for (let i = 0; i < h; i++) {
        dh2[i] += this.do.w[off + i] * d;
        this.do.gw[off + i] += d * this.h2[i];
      }
      this.do.gb[o] += d;
    }
    const dh1 = this.dh1.fill(0);
    for (let i = 0; i < h; i++) {
      dh2[i] *= 1 - this.h2[i] * this.h2[i];
      this.d2.gb[i] += dh2[i];
      const off = i * h;
      for (let j = 0; j < h; j++) {
        dh1[j] += this.d2.w[off + j] * dh2[i];
        this.d2.gw[off + j] += dh2[i] * this.h1[j];
      }
    }
    const nz = this.nz, nn = this.nn, x = this.x;
    for (let i = 0; i < h; i++) {
      dh1[i] *= 1 - this.h1[i] * this.h1[i];
      this.d1.gb[i] += dh1[i];
      const off = i * this.inN;
      for (let k = 0; k < nn; k++) { const j = nz[k]; this.d1.gw[off + j] += dh1[i] * x[j]; }
    }
  }
  applyGrads(lr, scale = 1) {
    this.d1.applyGrads(lr, scale);
    this.d2.applyGrads(lr, scale);
    this.do.applyGrads(lr, scale);
  }
}

function softmax(logits) {
  let m = -Infinity;
  for (const v of logits) if (v > m) m = v;
  const p = new Float32Array(logits.length);
  let s = 0;
  for (let i = 0; i < logits.length; i++) { p[i] = Math.exp(logits[i] - m); s += p[i]; }
  for (let i = 0; i < p.length; i++) p[i] /= s;
  return p;
}

function sampleCat(p, r) {
  let acc = 0;
  for (let i = 0; i < p.length; i++) { acc += p[i]; if (r <= acc) return i; }
  return p.length - 1;
}

// scratch buffers for the PPO update loop (avoid per-sample allocations)
const _dOut = new Float32Array(ACTS);
const _critD = new Float32Array(1);

// ---- observation ----

export function buildObs(g, dist) {
  const p = g.player;
  const tw = g.tw, th = g.th;
  const pc = Math.floor(p.x / tw), pr = Math.floor(p.y / th);
  const obs = new Float32Array(OBS_N);
  let k = 0;
  const entCell = new Map();
  for (const e of g.entities) {
    if (e.dead) continue;
    const c = Math.floor(e.x / tw), r = Math.floor(e.y / th);
    const key = (r - pr) * WIN_C + (c - pc);
    if (c - pc < -WIN_C / 2 || c - pc >= WIN_C / 2 || r - pr < -3 || r - pr >= WIN_R - 3) continue;
    if (!entCell.has(key)) entCell.set(key, e);
    else if (e.ai === 'next_level_ai') entCell.set(key, e); // exit wins
  }
  for (let r = -3; r < WIN_R - 3; r++) {
    for (let c = -WIN_C / 2; c < WIN_C / 2; c++) {
      const cc = pc + c, rr = pr + r;
      obs[k++] = cc < 0 || cc >= g.level.fgW || rr < 0 || rr >= g.level.fgH
        ? 1 : (g.tileSolid(cc * tw + tw / 2, rr * th + th / 2) ? 1 : 0);
      const e = entCell.get(r * WIN_C + c);
      obs[k++] = e && ENEMY_AI.has(e.ai) ? 1 : 0;
      obs[k++] = e && e.ai === 'next_level_ai' ? 1 : 0;
      let special = 0;
      if (e && (e.ai === 'tp2_ai' || e.ai === 'tpd_ai' || e.ai === 'platform_ai' || e.ai === 'sdoor_ai' || e.ai === 'strap_door_ai' || e.ai === 'restart_ai' || SWITCH_AI.has(e.ai) || BREAK_AI.has(e.ai))) special = 1;
      for (const l of g.ladders) {
        if (cc * tw >= l.x0 - 5 && cc * tw <= l.x1 + 5 && rr * th >= l.y0 && rr * th <= l.y1) special = 1;
      }
      obs[k++] = special;
    }
  }
  // scalars
  let exit = null, ed = Infinity, enemy = null, end = Infinity;
  for (const e of g.entities) {
    if (e.dead) continue;
    if (e.ai === 'next_level_ai') {
      const d = Math.abs(e.x - p.x) + Math.abs(e.y - p.y);
      if (d < ed) { ed = d; exit = e; }
    } else if (ENEMY_AI.has(e.ai)) {
      // Nearest enemy that can be hurt; dormant ones (a flyer still waiting
      // for its sensor) only count when nothing hittable is around.
      const d = Math.abs(e.x - p.x) + Math.abs(e.y - p.y) + (e.shootable ? 0 : 1e6);
      if (d < end) { end = d; enemy = e; }
    }
  }
  obs[k++] = exit ? (exit.x - p.x) / 100 : 0;
  obs[k++] = exit ? (exit.y - p.y) / 100 : 0;
  obs[k++] = (isFinite(dist) ? Math.min(dist >= LOCKED ? (dist - LOCKED) % 1000 : dist >= OPEN ? dist - OPEN : dist, 5000) : 5000) / 50; // distance to the exit (or the next switch), capped
  obs[k++] = enemy ? (enemy.x - p.x) / 100 : 0;
  obs[k++] = enemy ? (enemy.y - p.y) / 100 : 0;
  obs[k++] = p.vx / 100;
  obs[k++] = p.vy / 100;
  obs[k++] = p.ground ? 1 : 0;
  obs[k++] = p.climbing ? 1 : 0;
  obs[k++] = p.hp / 100;
  obs[k++] = (p.ammo[p.weapon] || 0) > 0 ? 1 : 0;
  obs[k++] = p.cooldown > 0 ? 1 : 0;
  obs[k++] = 0; // stuck flag, set by trainer
  // the nearest switch that is still off: which way, and how far through the level
  const sw = switchDist(g);
  obs[k++] = sw.nearest ? (sw.nearest.x - p.x) / 100 : 0;
  obs[k++] = sw.nearest ? (sw.nearest.y - p.y) / 100 : 0;
  obs[k++] = (isFinite(sw.dist) ? Math.min(sw.dist, 500) : 500) / 50;
  // the nearest health heart and the nearest weapon / ammo pickup, if one is close by
  let heart = null, hd = 500, ammo = null, ad = 500;
  for (const e of g.entities) {
    if (e.dead) continue;
    const d = Math.abs(e.x - p.x) + Math.abs(e.y - p.y);
    if (e.ai === 'hp_up' && d < hd) { hd = d; heart = e; } else if (e.ai === 'weapon_icon_ai' && d < ad) { ad = d; ammo = e; }
  }
  obs[k++] = heart ? (heart.x - p.x) / 100 : 0;
  obs[k++] = heart ? (heart.y - p.y) / 100 : 0;
  obs[k++] = ammo ? (ammo.x - p.x) / 100 : 0;
  obs[k++] = ammo ? (ammo.y - p.y) / 100 : 0;
  return obs;
}

// ---- Dijkstra distance to the nearest exit (shared navigation model) ----

// MAX_CLIMB: a jump lifts the feet 51 px and the engine then steps the player
// up onto anything within 16 px, so a ledge 4 rows (60 px) up can be mounted.
const MAX_FALL = 80, MAX_JUMP = 8, MAX_CLIMB = 4;

let lastPathDebug = null;
export function pathDistDebug() { return lastPathDebug; }

// Diagnosing the compass: returns the first edge of the shortest path from
// the player's cell (kind + target cell), so we can see what move the reward
// is actually pointing at.
export function pathDistStep(g, nextNum = null) {
  pathDist(g, nextNum);
  const dbg = lastPathDebug;
  if (!dbg || dbg.foundGoal === null) return null;
  const W = g.level.fgW;
  // trace the shortest-path tree back from the goal to the cell just after start
  let cur = dbg.foundGoal;
  let hops = 0;
  while (hops++ < 100000 && dbg.prev[cur] !== dbg.start) {
    if (dbg.prev[cur] === undefined) return null;
    cur = dbg.prev[cur];
  }
  const entry = dbg.first.find((f) => f.to[0] === cur % W && f.to[1] === Math.floor(cur / W) && f.from[0] === dbg.start % W && f.from[1] === Math.floor(dbg.start / W));
  return { dist: dbg.dist, step: [cur % W, Math.floor(cur / W)], kind: entry ? entry.kind : '?' };
}

const DOOR_AI = new Set(['sdoor_ai', 'strap_door_ai']);
// Switches the player works with the action key. Off (aistate 0) until used.
const SWITCH_AI = new Set(['switcher_ai', 'switch_once_ai', 'switch_delay_ai']);
// Obstacles the player can get through: doors (opened by a switch or sensor)
// and anything that is destroyed by shooting it (walls, bricks, gun turrets).
const SOFT_AI = new Set(['sdoor_ai', 'strap_door_ai', 'hwall_ai', 'big_wall_ai', 'block_ai', 'ff_ai']);
// Solid objects that are not obstacles to the route: lifts are modelled by
// their stops (and leaving the moving deck out keeps the graph fixed while one
// travels); exits and teleporters are things the player walks into.
const NOT_WALL_AI = new Set(['platform_ai', 'next_level_ai', 'tp2_ai', 'tpd_ai']);
// A lift with a third link only runs while that object (a switch) is on (platform.lsp).
const liftRuns = (e) => e.links.length < 3 || e.links[2].aistate !== 0;
// A door, force field or bank of pushers that only a "key" will clear: found
// by following its links back through the logic gates to a switch, or to a
// death sensor (which fires when the creature it watches has been killed). A
// door worked by an ordinary sensor opens as the player walks up to it; this
// kind does not, so until it clears it is a wall, and the way past it is the
// way to its key.
const GATE_AI = new Set(['and_ai', 'or_ai', 'xor_ai', 'not_ai', 'delay_ai', 'pulse_ai', 'indicator_ai']);
function switchLocked(e, depth = 0, seen = new Set()) {
  for (const l of e.links) {
    if (seen.has(l)) continue;
    seen.add(l);
    if (SWITCH_AI.has(l.ai) || l.ai === 'death_sen_ai') return true;
    if (depth < 8 && GATE_AI.has(l.ai) && switchLocked(l, depth + 1, seen)) return true;
  }
  return false;
}
const LOCKABLE_AI = new Set(['sdoor_ai', 'strap_door_ai', 'ff_ai']);
// The keys still to be dealt with: switches that are off, and creatures a
// death sensor is waiting on.
function keysToGo(g) {
  const out = [];
  for (const e of g.entities) {
    if (e.dead) continue;
    if (e.aistate === 0 && SWITCH_AI.has(e.ai)) out.push(e);
    else if (e.ai === 'death_sen_ai' && e.aistate === 0) for (const l of e.links) if (!l.dead && !out.includes(l)) out.push(l);
  }
  return out;
}
// How many keys have been dealt with (switches on, death sensors satisfied): progress worth banking at a save station.
// (a death sensor that watched nothing from the start does not count: the level file says what it was wired to)
const keysDone = (g) => g.entities.reduce((n, e) => n + (!e.dead && e.aistate !== 0 && (SWITCH_AI.has(e.ai) || (e.ai === 'death_sen_ai' && g.level.objects[e.id]?.links.length > 0)) ? 1 : 0), 0);
// Pushers blowing across a passage (general.lsp pusher_ai) that only a key will switch off.
const lockedPushers = (g) => g.entities.filter((e) => !e.dead && e.ai === 'pusher_ai' && e.links.length && e.links[0].aistate !== 0 && switchLocked(e));
const isSoft = (e) => !!e && (SOFT_AI.has(e.ai) || !!e.shootable);
const SOFT_COST = 15;
const REACH = 3; // how many cells of walking one cell nearer the exit (through open space) is worth

// The navigation graph over the tile grid: which cells are solid, which can be
// stood on, and the moves out of each cell (walk, step up, climb, drop, jump,
// ladder, teleporter / lift / spring). Shared by the forward search (pathDist)
// and the cached distance field (navDist).
//
// A cell (c, r) stands for "feet resting on solid cell r of column c", with
// the body in the two cells above it.
export function navGraph(g, nextNum) {
  const W = g.level.fgW, H = g.level.fgH, tw = g.tw, th = g.th;
  const N = W * H;
  // grid: 1 = hard (tiles, fixed objects), 2 = soft (closed doors, breakable
  // walls). Soft cells can be stood on like any floor, but they don't wall the
  // route off: passing through one just costs extra, so the distance stays
  // finite, falls as the agent approaches the obstacle, and drops sharply once
  // it is opened or destroyed (which the PPO experiences as a progress burst).
  const grid = new Uint8Array(N);
  let locked = 0; // closed doors that only a switch will open
  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) if (g.tileSolid(c * tw + tw / 2, r * th + th / 2)) grid[r * W + c] = 1;
  for (const s of g.solids) {
    if (s.e && NOT_WALL_AI.has(s.e.ai)) continue;
    if (isSoft(s.e) && DOOR_AI.has(s.e.ai) && s.e.aistate !== 0) continue; // already opening
    const lockedDoor = LOCKABLE_AI.has(s.e?.ai) && switchLocked(s.e);
    if (lockedDoor) locked++;
    const soft = isSoft(s.e) && !lockedDoor;
    const c0 = Math.max(0, Math.floor(s.x0 / tw) - 1), c1 = Math.min(W - 1, Math.floor(s.x1 / tw) + 1);
    const r0 = Math.max(0, Math.floor(s.y0 / th) - 1), r1 = Math.min(H - 1, Math.floor(s.y1 / th) + 1);
    for (let r = r0; r <= r1; r++) for (let c = c0; c <= c1; c++) {
      const x = c * tw + tw / 2;
      if (x < s.x0 || x > s.x1 || (r + 1) * th <= s.y0 || r * th >= s.y1) continue;
      const idx = r * W + c;
      if (!soft) grid[idx] = 1; else if (grid[idx] === 0) grid[idx] = 2;
    }
  }
  // A bank of pushers the player cannot get past (they push as fast as the
  // player runs) is a wall for as long as its key keeps it on.
  for (const e of lockedPushers(g)) {
    const r = g.rectOf(e);
    if (!r) continue;
    locked++;
    for (let rr = Math.max(0, Math.floor(r.y0 / th) - 1); rr <= Math.min(H - 1, Math.floor(r.y1 / th) + 1); rr++) {
      for (let c = Math.max(0, Math.floor(r.x0 / tw)); c <= Math.min(W - 1, Math.floor(r.x1 / tw)); c++) grid[rr * W + c] = 1;
    }
  }
  // hard(): blocks the body. floorOK(): can be stood on (hard or soft).
  const hard = (c, r) => {
    if (c < 0 || c >= W) return true;
    if (r < 0 || r >= H) return false;
    return grid[r * W + c] === 1;
  };
  const floorOK = (c, r) => r >= 0 && r < H && c >= 0 && c < W && grid[r * W + c] !== 0;
  const bodyOK = (c, r) => {
    if (c < 0 || c >= W) return false;
    for (let rr = r - 1; rr >= r - 2; rr--) if (rr >= 0 && hard(c, rr)) return false;
    return true;
  };
  const softBody = (c, r) => (r >= 1 && grid[(r - 1) * W + c] === 2) || (r >= 2 && grid[(r - 2) * W + c] === 2);
  const cellOf = (e) => {
    const cx = Math.floor(e.x / tw), cy = Math.floor(e.y / th);
    let best = null, bd = Infinity;
    for (let c = cx - 4; c <= cx + 4; c++) {
      for (let r = cy - 8; r <= cy + 8; r++) {
        if (r < 0 || r >= H) continue;
        if (floorOK(c, r) && bodyOK(c, r)) {
          const d = Math.abs(c - cx) * 2 + Math.abs(r - cy);
          if (d < bd) { bd = d; best = r * W + c; }
        }
      }
    }
    return best;
  };
  const goals = [];
  for (const e of g.entities) {
    if (e.dead || e.ai !== 'next_level_ai') continue;
    // nextNum filters to the zones that actually advance to the next level
    // (original people.lsp: next_level_ai loads level{aistate}).
    if (nextNum !== null && e.aistate !== nextNum) continue;
    const cell = cellOf(e);
    if (cell !== null) goals.push(cell);
  }
  if (!goals.length && nextNum !== null) {
    // fall back to any exit if no zone advances to the requested level
    for (const e of g.entities) {
      if (e.dead || e.ai !== 'next_level_ai') continue;
      const cell = cellOf(e);
      if (cell !== null) goals.push(cell);
    }
  }
  const jumpTo = new Map();
  const addJump = (from, to, cost) => {
    if (from === null || to === null || from === to) return;
    if (!jumpTo.has(from)) jumpTo.set(from, []);
    jumpTo.get(from).push({ to, cost });
  };
  // Ridable platforms/trains: their stops are standable even in mid-air, and
  // riding connects the stops. cellOf() can't see mid-air stops, so stops use
  // the raw cell under the stop point (the player's feet rest on the platform).
  const standCells = new Set();
  const stopCell = (x, y) => {
    const c = Math.floor(x / tw), r = Math.floor(y / th);
    if (c < 0 || c >= W || r < 0 || r >= H) return null;
    if (!bodyOK(c, r)) return null; // stop buried in a wall
    const idx = r * W + c;
    standCells.add(idx);
    return idx;
  };
  const ride = (stops) => {
    for (const a of stops) for (const b of stops) if (a !== null && b !== null && a !== b) addJump(a, b, 8);
  };
  for (const e of g.entities) {
    if (e.dead) continue;
    if (e.ai === 'tp2_ai' || e.ai === 'tpd_ai') {
      addJump(cellOf(e), e.links[0] ? cellOf(e.links[0]) : null, 4);
    } else if (e.ai === 'platform_ai') {
      const stops = (e.links || []).slice(0, 2).map((l) => stopCell(l.x, l.y));
      if (!liftRuns(e)) continue; // waiting for its switch: somewhere to stand, not a ride
      ride(stops);
      // Vertical lifts pass through open chambers between their stops: the
      // player can hop off mid-ride, so connect intermediate rows too
      // (e.g. level00's elevator dismount into the east chamber).
      for (const s of stops) {
        if (s === null) continue;
        const sc = s % W, sr = Math.floor(s / W);
        for (const s2 of stops) {
          if (s2 === null || s2 === s) continue;
          const c2 = s2 % W, r2 = Math.floor(s2 / W);
          if (c2 !== sc) continue;
          const top = Math.min(sr, r2), bot = Math.max(sr, r2);
          for (let r = top + 3; r < bot; r += 3) {
            const idx = r * W + sc;
            if (!bodyOK(sc, r)) continue; // mid-ride cell inside a wall
            standCells.add(idx);
            addJump(s, idx, Math.min(8, Math.abs(r - sr) * 0.7 + 2));
            addJump(idx, s2, Math.min(8, Math.abs(r2 - r) * 0.7 + 2));
          }
        }
      }
    } else if (e.ai === 'spring_ai') {
      const f = cellOf(e);
      if (f !== null) {
        const c = f % W, r = Math.floor(f / W);
        addJump(f, r >= 9 ? (r - 9) * W + c : null, 6);
      }
    }
  }
  const ladders = g.ladders.map((l) => ({
    c0: Math.floor(l.x0 / tw), c1: Math.floor(l.x1 / tw),
    rTop: Math.max(0, Math.floor(l.y0 / th)), rBot: Math.min(H - 1, Math.floor(l.y1 / th)),
  }));

  // Calls emit(toCell, cost, kind) for every move out of `cur`.
  let emit = null, curSoft = false;
  const relax = (c, r, cost, kind) => {
    if (c < 0 || c >= W) return;
    const idx = r * W + c;
    if ((!floorOK(c, r) && !standCells.has(idx)) || !bodyOK(c, r)) return;
    // going into a door / breakable wall costs once, not per cell of its thickness
    emit(idx, !curSoft && softBody(c, r) ? cost + SOFT_COST : cost, kind);
  };
  const expand = (cur, emitFn) => {
    emit = emitFn;
    const c = cur % W, r = Math.floor(cur / W);
    curSoft = softBody(c, r);
    for (const s of [-1, 1]) {
      relax(c + s, r, 1, 'walk');
      if (r > 0) relax(c + s, r - 1, 1.5, 'stepup');
    }
    // Jump up onto a ledge beside the player (up to MAX_CLIMB rows higher),
    // when there is headroom above the take-off cell.
    for (let up = 2; up <= MAX_CLIMB; up++) {
      if (r - up - 2 >= 0 && hard(c, r - up - 2)) break; // head hits the ceiling first
      for (const s of [-1, 1]) relax(c + s, r - up, up + 0.5, 'climb');
    }
    // A soft floor (trap door, breakable wall) can be opened and dropped through.
    if (grid[cur] === 2) {
      let rr = r + 1;
      for (; rr < Math.min(H, r + MAX_FALL); rr++) if (grid[rr * W + c] !== 2) break;
      for (; rr < Math.min(H, r + MAX_FALL); rr++) if (grid[rr * W + c] !== 0) break;
      if (rr < H && grid[rr * W + c] !== 0) relax(c, rr, rr - r + SOFT_COST, 'through');
    }
    for (const s of [-1, 1]) {
      const nc = c + s;
      if (nc < 0 || nc >= W) continue;
      if (hard(nc, r - 1)) continue;
      let rr = r;
      for (; rr < Math.min(H, r + MAX_FALL); rr++) if (floorOK(nc, rr)) break;
      if (rr > r && rr < H && floorOK(nc, rr)) relax(nc, rr, rr - r + 0.5, 'drop'); // (level ground is a walk, above)
    }
    for (const s of [-1, 1]) {
      const n1 = c + s;
      if (n1 < 0 || n1 >= W) continue;
      if (floorOK(n1, r) && bodyOK(n1, r)) continue;
      for (let d = 2; d <= MAX_JUMP; d++) {
        const nc = c + s * d;
        if (nc < 0 || nc >= W) break;
        let clear = true;
        for (let i = 1; i < d; i++) {
          const cc = c + s * i;
          for (let rr = r - 6; rr <= r - 1; rr++) if (rr >= 0 && hard(cc, rr)) { clear = false; break; }
          if (!clear) break;
        }
        if (!clear) break;
        for (const dr of [-2, -1, 0, 1]) if (floorOK(nc, r + dr)) relax(nc, r + dr, d + Math.abs(dr), 'jump');
      }
    }
    for (const l of ladders) {
      if (c < l.c0 || c > l.c1 || r < l.rTop || r > l.rBot) continue;
      // Climbing is rect-based in the game (inLadder); tiles don't block it.
      for (let rr = l.rTop; rr <= l.rBot; rr++) if (rr !== r) emit(rr * W + c, Math.abs(rr - r) * 1.4, 'ladder');
    }
    const js = jumpTo.get(cur);
    if (js) for (const j of js) relax(j.to % W, Math.floor(j.to / W), j.cost, 'ride');
  };
  return { W, H, N, grid, goals, expand, cellOf, floorOK, bodyOK, jumpTo, locked };
}

// Binary min-heap of (distance, cell) pairs; stale entries are skipped by the caller.
function makeHeap() {
  const keys = [], cells = [];
  return {
    get size() { return keys.length; },
    key: 0,
    push(d, idx) {
      let i = keys.length;
      keys.push(d); cells.push(idx);
      while (i > 0) {
        const pi = (i - 1) >> 1;
        if (keys[pi] <= d) break;
        keys[i] = keys[pi]; cells[i] = cells[pi];
        i = pi;
      }
      keys[i] = d; cells[i] = idx;
    },
    // Returns the nearest cell; its distance is left in `key`.
    pop() {
      const top = cells[0];
      this.key = keys[0];
      const lk = keys.pop(), lc = cells.pop();
      const n = keys.length;
      if (n) {
        let i = 0;
        for (;;) {
          let ci = i, ck = lk;
          const a = i * 2 + 1, b = a + 1;
          if (a < n && keys[a] < ck) { ck = keys[a]; ci = a; }
          if (b < n && keys[b] < ck) { ci = b; }
          if (ci === i) break;
          keys[i] = keys[ci]; cells[i] = cells[ci];
          i = ci;
        }
        keys[i] = lk; cells[i] = lc;
      }
      return top;
    },
  };
}

const playerCell = (g) => {
  const W = g.level.fgW, H = g.level.fgH;
  const pc = Math.max(0, Math.min(W - 1, Math.floor(g.player.x / g.tw)));
  const pr = Math.max(0, Math.min(H - 1, Math.floor(g.player.y / g.th)));
  return pr * W + pc;
};

// Forward search from the player's cell to the nearest exit. Keeps the search
// tree for pathDistStep/pathDistDebug; the trainer uses navDist instead.
export function pathDist(g, nextNum = null) {
  const { W, N, goals, expand } = navGraph(g, nextNum);
  if (!goals.length) return Infinity;
  const goalSet = new Set(goals);
  const dist = new Float64Array(N).fill(Infinity);
  const prev = new Int32Array(N).fill(-1);
  const start = playerCell(g);
  const dbg = { start, goals, expansions: 0, goalReached: [], cells: [], first: [], prev, foundGoal: null, dist: null };
  dist[start] = 0;
  const heap = makeHeap();
  heap.push(0, start);
  let found = Infinity;
  let expansions = 0;
  let foundGoal = null;
  let cur = 0;
  const emit = (idx, cost, kind) => {
    const nd = dist[cur] + cost;
    if (nd < dist[idx]) {
      dist[idx] = nd;
      prev[idx] = cur;
      heap.push(nd, idx);
      dbg.cells.push([idx % W, Math.floor(idx / W)]);
      dbg.first.push({ to: [idx % W, Math.floor(idx / W)], from: [cur % W, Math.floor(cur / W)], kind });
      if (goalSet.has(idx)) dbg.goalReached.push([idx, nd]);
    }
  };
  while (heap.size && expansions++ < 200000) {
    cur = heap.pop();
    if (heap.key > dist[cur]) continue; // stale entry
    if (goalSet.has(cur)) { found = dist[cur]; foundGoal = cur; break; }
    expand(cur, emit);
  }
  dbg.expansions = expansions;
  dbg.reached = 0;
  dbg.rows = {};
  dbg.foundGoal = foundGoal;
  dbg.dist = found;
  for (let i = 0; i < N; i++) if (isFinite(dist[i])) { dbg.reached++; const rr = Math.floor(i / W); dbg.rows[rr] = (dbg.rows[rr] || 0) + 1; }
  lastPathDebug = dbg;
  return found;
}

// ---- cached distance field ----
//
// The trainer needs the distance to the exit on every decision. Rather than
// searching from the player each time, the distance from EVERY cell is solved
// once (Dijkstra from the exits over the reversed graph) and looked up; it is
// only re-solved when something that shapes the graph changes (a door opens,
// a wall is destroyed, a lift moves).
function navSig(g, nextNum) {
  const tw = g.tw, th = g.th;
  let s = `${g.level.name}|${nextNum}`;
  // Only the cells a solid covers matter (see navGraph), so animation frames
  // and sub-cell movement don't invalidate the field.
  for (const e of liftsOf(g)) if (e.links.length >= 3) s += liftRuns(e) ? ';L1' : ';L0'; // lifts waiting for a switch
  s += `;P${lockedPushers(g).length}`; // pushers still blowing
  for (const x of g.solids) {
    if (x.e && NOT_WALL_AI.has(x.e.ai)) continue;
    const soft = isSoft(x.e);
    if (soft && DOOR_AI.has(x.e.ai) && x.e.aistate !== 0) continue; // already opening
    s += `;${soft ? 'S' : ''}${Math.ceil((x.x0 - tw / 2) / tw)},${Math.floor(x.y0 / th)},${Math.floor((x.x1 - tw / 2) / tw)},${Math.ceil(x.y1 / th) - 1}`;
  }
  return s;
}

function buildField(g, nextNum) {
  const { N, grid, goals, expand, cellOf, locked } = navGraph(g, nextNum);
  const field = new Float64Array(N).fill(Infinity);
  field.grid = grid;
  field.locked = locked;
  // reversed edges as linked lists: head[to] -> edge -> next edge into `to`
  const head = new Int32Array(N).fill(-1);
  let eFrom = [], eCost = [], eNext = [];
  let cur = 0;
  const emit = (to, cost) => {
    eFrom.push(cur); eCost.push(cost); eNext.push(head[to]);
    head[to] = eFrom.length - 1;
  };
  for (cur = 0; cur < N; cur++) expand(cur, emit);
  eFrom = Int32Array.from(eFrom); eCost = Float32Array.from(eCost); eNext = Int32Array.from(eNext); // kept with the field
  // shortest way back along the reversed moves, from wherever `dist` starts finite
  const solve = (dist) => {
    const heap = makeHeap();
    for (let i = 0; i < N; i++) if (dist[i] !== Infinity) heap.push(dist[i], i);
    while (heap.size) {
      const to = heap.pop();
      if (heap.key > dist[to]) continue; // stale entry
      for (let e = head[to]; e !== -1; e = eNext[e]) {
        const from = eFrom[e];
        const nd = dist[to] + eCost[e];
        if (nd < dist[from]) { dist[from] = nd; heap.push(nd, from); }
      }
    }
  };
  for (const gl of goals) field[gl] = 0;
  solve(field);
  // "Reach" field, for when no walking route gets all the way to the exit
  // (the graph cannot model every puzzle): the cost of WALKING, by legal
  // moves only, to wherever is nearest the exit through open space. Every
  // cell starts at REACH x its open-space distance and the walking moves are
  // relaxed from there, so following it downhill never asks for something the
  // player cannot do, like rising through a closed trap door, which the
  // open-space distance alone happily does.
  const flood = buildFlood(g, nextNum);
  const reach = new Float64Array(N).fill(Infinity);
  for (let i = 0; i < N; i++) if (flood[i] !== Infinity) reach[i] = REACH * flood[i];
  solve(reach);
  for (let i = 0; i < N; i++) reach[i] /= REACH;
  reach.grid = grid;
  field.reach = reach;
  field.flood = flood;
  // walking distance to the nearest of some objects (used for switches)
  field.walkTo = (objects) => {
    const d = new Float64Array(N).fill(Infinity);
    for (const o of objects) { const c = cellOf(o); if (c !== null) d[c] = 0; }
    solve(d);
    d.grid = grid;
    return d;
  };
  return field;
}

// Second, looser field for wherever the walking graph finds no route (it
// can't model every lift, switch and puzzle): the distance through open space,
// ignoring gravity, around hard walls, through doors and breakable walls at a
// cost, and through teleporters. It still bends around the level's geometry,
// which a straight line to the exit does not.
function buildFlood(g, nextNum, targets = null) {
  const { W, H, N, grid, goals: walkGoals, cellOf } = navGraph(g, nextNum);
  const tw = g.tw, th = g.th;
  const flood = new Float64Array(N).fill(Infinity);
  const cellAt = (x, y) => {
    const c = Math.floor(x / tw), r = Math.floor(y / th);
    return c < 0 || c >= W || r < 0 || r >= H ? -1 : r * W + c;
  };
  // an object's feet sit on the cell boundary; its body is the cell above
  const bodyCell = (e) => {
    for (const dy of [8, 23, 38, -7]) { const i = cellAt(e.x, e.y - dy); if (i >= 0 && grid[i] !== 1) return i; }
    // set into the scenery: use the nearest spot it can be reached from
    const f = cellOf(e);
    return f !== null && f >= W ? f - W : -1;
  };
  const exits = g.entities.filter((e) => !e.dead && e.ai === 'next_level_ai');
  let goals = exits.filter((e) => nextNum === null || e.aistate === nextNum);
  if (!goals.length) goals = exits;
  const heap = makeHeap();
  // distance to the exit, or (with `targets`) to the nearest of those objects
  const seeds = (targets || goals).map(bodyCell);
  if (!targets) for (const f of walkGoals) if (f >= W) seeds.push(f - W); // where the walking graph stands to exit
  for (const i of seeds) if (i >= 0 && grid[i] !== 1 && flood[i] !== 0) { flood[i] = 0; heap.push(0, i); }
  // teleporters, reversed (arriving at the destination is reachable from the
  // pad), and lifts, which join their two stops in both directions
  const into = new Map();
  const link = (a, b, cost) => {
    if (a < 0 || b < 0 || a === b) return;
    if (!into.has(b)) into.set(b, []);
    into.get(b).push({ a, cost });
  };
  for (const e of g.entities) {
    if (e.dead) continue;
    if ((e.ai === 'tp2_ai' || e.ai === 'tpd_ai') && e.links[0]) link(bodyCell(e), bodyCell(e.links[0]), 4);
    else if (e.ai === 'platform_ai' && e.links.length >= 2 && liftRuns(e)) {
      const a = bodyCell(e.links[0]), b = bodyCell(e.links[1]);
      link(a, b, 8); link(b, a, 8);
    }
  }
  // ladders are climbed through the scenery (inLadder is rect-based)
  for (const l of g.ladders) {
    const x = (l.x0 + l.x1) / 2;
    const a = cellAt(x, l.y0 - 8), b = cellAt(x, l.y1 - 8);
    const cost = Math.abs(l.y1 - l.y0) / th * 1.4;
    link(a, b, cost); link(b, a, cost);
  }
  while (heap.size) {
    const cur = heap.pop();
    const d = flood[cur];
    if (heap.key > d) continue; // stale entry
    const c = cur % W, r = Math.floor(cur / W);
    const relax = (to, cost) => {
      if (grid[to] === 1) return;
      // stepping out of a soft cell backwards = going into it forwards
      const nd = d + cost + (grid[cur] === 2 && grid[to] !== 2 ? SOFT_COST : 0);
      if (nd < flood[to]) { flood[to] = nd; heap.push(nd, to); }
    };
    if (c > 0) relax(cur - 1, 1);
    if (c < W - 1) relax(cur + 1, 1);
    if (r > 0) relax(cur - W, 1);
    if (r < H - 1) relax(cur + W, 1);
    const tps = into.get(cur);
    if (tps) for (const t of tps) relax(t.a, t.cost);
  }
  return flood;
}

// The fields for a world state, most recently used first. A dozen are kept:
// runs start from different places (the level's start, each save station) and
// every destroyed wall or opened door is a new state, but the same states come
// round again and again, and solving one takes tens of milliseconds.
let fieldCache = null;
const fieldCaches = new Map();
function useFields(g, nextNum) {
  const sig = navSig(g, nextNum);
  if (fieldCache && fieldCache.sig === sig) return;
  fieldCache = fieldCaches.get(sig);
  if (fieldCache) fieldCaches.delete(sig); // re-inserted below as the newest
  else { fieldCache = { sig, field: buildField(g, nextNum), sw: new Map() }; navStats.builds++; }
  fieldCaches.set(sig, fieldCache);
  if (fieldCaches.size > 12) fieldCaches.delete(fieldCaches.keys().next().value); // each holds the level's whole move graph
}
// the lifts of the level as currently loaded (its objects are new on every load)
let liftCache = { level: null, list: [] };
const liftsOf = (g) => {
  if (liftCache.level !== g.level) liftCache = { level: g.level, list: g.entities.filter((e) => e.ai === 'platform_ai' && e.links.length >= 2) };
  return liftCache.list;
};
export const navStats = { builds: 0, lookups: 0 };

// Both fields for the current world state (diagnostics and tooling).
export function navFields(g, nextNum = null) {
  navDist(g, nextNum);
  return { field: fieldCache.field, flood: fieldCache.field.flood, reach: fieldCache.field.reach };
}

// Open-space distance to the exit (see buildFlood); Infinity only where the
// player is sealed off from it by hard walls.
export function floodDist(g, nextNum = null) {
  useFields(g, nextNum);
  return readFlood(g, fieldCache.field.flood);
}

function readFlood(g, flood) {
  const W = g.level.fgW, H = g.level.fgH, p = g.player;
  // The grid is coarse (a cell is judged by its centre), so in a passage
  // narrower than a cell the player's own column can read as wall: look in
  // the columns the body overlaps as well.
  for (const dx of [0, -6, 6]) {
    const c = Math.max(0, Math.min(W - 1, Math.floor((p.x + dx) / g.tw)));
    for (const dy of [8, 23, -7]) {
      const r = Math.max(0, Math.min(H - 1, Math.floor((p.y - dy) / g.th)));
      if (flood[r * W + c] !== Infinity) return flood[r * W + c];
    }
  }
  return Infinity;
}


// Distance from the player to the nearest "key" still to be dealt with: a
// switch that is off, or a creature whose death something is waiting on. Also
// which set of keys that was measured against (`key` changes when one is
// done). Levels are gated by things the exit compass knows nothing about (a
// lift that only runs once its switch is on, a door opened from another room,
// pushers that stop when a particular ant is dead), so "go and deal with the
// keys" is a second compass of its own.
export function switchDist(g) {
  // Always measured for the level's own next exit, whoever asks: the fields
  // are cached per (level state, exit), and asking with a different exit
  // would throw the cache away and rebuild it on every call.
  const nextNum = +(/(\d+)/.exec(g.level.name) || [0, 0])[1] + 1;
  useFields(g, nextNum);
  // (switches still off, and creatures a death sensor is waiting on)
  let off = keysToGo(g), key = '';
  for (const e of off) key += `${e.id},`;
  if (!off.length) off = null;
  if (!off) return { dist: Infinity, key: '', nearest: null, off: 0 };
  // one pair of fields per set of switches still off (kept: switches get pressed in the same few orders)
  let sw = fieldCache.sw.get(key);
  if (!sw) { sw = { walk: fieldCache.field.walkTo(off), flood: buildFlood(g, nextNum, off) }; fieldCache.sw.set(key, sw); if (fieldCache.sw.size > 8) fieldCache.sw.delete(fieldCache.sw.keys().next().value); }
  // by legal moves where a switch can be walked to; through open space otherwise
  let dist = readWalk(g, sw.walk);
  const walkable = dist !== Infinity;
  if (!walkable) dist = readFlood(g, sw.flood);
  // the nearest in a straight line, for the agent's view
  const p = g.player;
  let nearest = null, bd = Infinity;
  for (const e of off) { const d = Math.abs(e.x - p.x) + Math.abs(e.y - p.y); if (d < bd) { bd = d; nearest = e; } }
  return { dist, key: `${walkable ? 'w' : 'o'}${key}`, nearest, off: off.length };
}

// Offsets that rank the kinds of reading: a walking route to the exit beats an
// open-space one, which beats "still have switches to find".
export const OPEN = 1e6;
export const LOCKED = 2e6;

// The distance the trainer steers by: the walking route to the exit where
// there is one; otherwise (+OPEN) the "reach" distance, walking as near to the
// exit as legal moves allow; otherwise the plain open-space distance;
// otherwise Infinity (no reading here; callers keep their previous value
// rather than guess).
export function exitDist(g, nextNum = null) {
  const walk = navDist(g, nextNum);
  if (walk !== Infinity) return walk;
  // No walking route, and doors in the level that only switches will open:
  // the way forward is the switches. Scored as "switches still off" first,
  // then the distance to the nearest of them, so pressing one is a big step
  // and walking towards the next is steady progress.
  if (fieldCache.field.locked) {
    const sw = switchDist(g);
    if (sw.dist !== Infinity) return LOCKED + sw.off * 1000 + Math.min(sw.dist, 999);
    useFields(g, nextNum); // (switchDist may have moved the cache to its own exit number)
  }
  const reach = readWalk(g, fieldCache.field.reach);
  if (reach !== Infinity) return OPEN + reach;
  const open = floodDist(g, nextNum);
  return open === Infinity ? Infinity : OPEN + open;
}

// For people: "12" along the walking route, "~340" through open space.
export const fmtDist = (d) => (!isFinite(d) ? '?'
  : d >= LOCKED ? `${Math.floor((d - LOCKED) / 1000)} switch(es) to go, ${((d - LOCKED) % 1000).toFixed(0)} to the next`
  : d >= OPEN ? `~${(d - OPEN).toFixed(0)}` : d.toFixed(0));


// Path distance (in cells) from the player to the nearest exit that advances
// to level `nextNum`; Infinity when no route exists. Cheap after the first call.
export function navDist(g, nextNum = null) {
  useFields(g, nextNum);
  navStats.lookups++;
  return readWalk(g, fieldCache.field);
}

// Reads a walking-graph field at the player (on a lift, in a narrow passage,
// or in mid-air included).
function readWalk(g, field) {
  const W = g.level.fgW, H = g.level.fgH;
  // Riding a lift: the graph only knows its two stops, so in between the
  // distance is read off the ride itself, sliding evenly from one stop's value
  // to the other's. (Looked up by cell, a ride reads as hanging in mid-air and
  // the distance climbs most of the way up, which punishes taking the lift.)
  const p = g.player;
  for (const e of liftsOf(g)) {
    if (e.dead) continue;
    const deck = g.deckRect(e);
    // on the deck, or hopping just above it (a jump peaks 51 px up)
    if (!deck || !g.overDeck(deck) || p.y > deck.y0 + 3 || p.y < deck.y0 - 56) continue;
    const [a, b] = e.links;
    const at = (o) => {
      const c = Math.floor(o.x / g.tw), r = Math.floor(o.y / g.th);
      return c < 0 || c >= W || r < 0 || r >= H ? Infinity : field[r * W + c];
    };
    const fa = at(a), fb = at(b);
    const len2 = (b.x - a.x) ** 2 + (b.y - a.y) ** 2;
    if (fa === Infinity || fb === Infinity || !len2) break;
    const t = Math.max(0, Math.min(1, ((e.x - a.x) * (b.x - a.x) + (e.y - a.y) * (b.y - a.y)) / len2));
    return fa + (fb - fa) * t;
  }
  const cell = playerCell(g);
  if (field[cell] !== Infinity) return field[cell];
  const row = Math.floor(cell / W);
  // in a passage narrower than a cell, the feet can be in a column that reads as wall
  for (const dx of [-6, 6]) {
    const c = Math.max(0, Math.min(W - 1, Math.floor((p.x + dx) / g.tw)));
    if (field[row * W + c] !== Infinity) return field[row * W + c];
  }
  // Mid-jump or mid-fall the player's cell is open air, which the graph has
  // no moves out of. Score it as the spot they will come down on, so the
  // distance doesn't vanish on take-off and reappear on landing.
  const c = cell % W;
  for (let r = row + 1, n = 1; r < H && n <= MAX_FALL; r++, n++) {
    const idx = r * W + c;
    if (field[idx] !== Infinity) return field[idx] + n;
    if (field.grid[idx]) break;
  }
  return Infinity;
}

// ---- networks & PPO ----

export class Net {
  constructor(hidden = 96) {
    this.actor = new MLP(OBS_N, hidden, ACTS);
    this.critic = new MLP(OBS_N, hidden, 1);
  }
  forward(x) {
    const logits = this.actor.forward(x);
    const v = this.critic.forward(x);
    return { logits, v: v[0] };
  }
  // The twelve weight arrays, in a fixed order (parallel training averages
  // them across workers).
  weights() {
    const of = (m) => [m.d1.w, m.d1.b, m.d2.w, m.d2.b, m.do.w, m.do.b];
    return [...of(this.actor), ...of(this.critic)];
  }
  setWeights(list) { this.weights().forEach((w, i) => w.set(list[i])); }
  save() {
    // 6 significant digits: a float32 printed in full is ~20 characters, and
    // 200,000 of them alone come close to the browser's storage limit.
    const arr = (w) => Array.from(w, (v) => +v.toPrecision(6));
    const dump = (m) => ({
      w1: arr(m.d1.w), b1: arr(m.d1.b),
      w2: arr(m.d2.w), b2: arr(m.d2.b),
      wo: arr(m.do.w), bo: arr(m.do.b),
    });
    try {
      localStorage.setItem(PPO_KEY, JSON.stringify({ a: dump(this.actor), c: dump(this.critic) }));
    } catch { /* quota */ }
  }
  load() {
    try {
      const j = JSON.parse(localStorage.getItem(PPO_KEY));
      if (!j) return false;
      const fill = (m, d) => {
        m.d1.w.set(Float32Array.from(d.w1)); m.d1.b.set(Float32Array.from(d.b1));
        m.d2.w.set(Float32Array.from(d.w2)); m.d2.b.set(Float32Array.from(d.b2));
        m.do.w.set(Float32Array.from(d.wo)); m.do.b.set(Float32Array.from(d.bo));
      };
      fill(this.actor, j.a);
      fill(this.critic, j.c);
      return true;
    } catch { return false; }
  }
}

export function hasTrainedPolicy() {
  return !!localStorage.getItem(PPO_KEY);
}

// Best training run: the episode that got closest to the exit, stored as an
// action script so it can be replayed in the demo mode.
// A recorded stretch of play is a "piece": the state it started from (null =
// the level's own fresh start), the action chosen at each decision (one
// character each), and how many game steps it ran. Started the same way and
// fed the same actions, the game replays it exactly. Pieces chain: the next
// one starts from precisely the state the previous one ended in.
export const encodeActs = (acts) => String.fromCharCode(...acts.map((a) => 65 + a));
export const decodeActs = (str) => Array.from(str, (ch) => ch.charCodeAt(0) - 65);
const validRec = (j) => !!j && Array.isArray(j.pieces) && j.pieces.length > 0;

export function loadBestRun() {
  try {
    const j = JSON.parse(localStorage.getItem(BEST_KEY));
    return validRec(j) ? j : null;
  } catch { return null; }
}

// The campaign: every level passed so far, each as the chain of pieces that
// got through it, and the level the agent is on now ("frontier") with the
// state it enters that level in (weapons and ammo carried over). Training
// never goes back before the frontier, and the legs together are a complete
// playthrough up to it.
// Kept in memory as well as in localStorage: these records can outgrow the
// browser's storage limit, and then they live for the session only.
const big = new Map(); // key -> { raw, value }
function bigLoad(key, valid) {
  let raw = null;
  try { raw = localStorage.getItem(key); } catch { /* storage unavailable */ }
  const have = big.get(key);
  if (raw && (!have || raw !== have.raw)) {
    try { const v = JSON.parse(raw); if (valid(v)) { big.set(key, { raw, value: v }); return v; } } catch { /* keep what we have */ }
  }
  return have ? have.value : null;
}
function bigSave(key, value) {
  const raw = JSON.stringify(value);
  big.set(key, { raw, value });
  try { localStorage.setItem(key, raw); } catch { /* too big for storage: memory only */ }
}
function bigClear(key) {
  big.delete(key);
  try { localStorage.removeItem(key); } catch { /* ignore */ }
}

export function loadCampaign() {
  return bigLoad(CAMPAIGN_KEY, (c) => !!c && Array.isArray(c.legs)) || { frontier: 0, entry: null, legs: [] };
}
export function saveCampaign(c) { bigSave(CAMPAIGN_KEY, c); }
// Forgets all training in this browser: the policy, the levels passed, save
// stations, best runs and exploration counts (every abuse.ppo.* key, whatever
// its version).
export function forgetTraining() {
  big.clear();
  try {
    const keys = [];
    for (let i = 0; i < localStorage.length; i++) { const k = localStorage.key(i); if (k && k.startsWith('abuse.ppo.')) keys.push(k); }
    for (const k of keys) localStorage.removeItem(k);
  } catch { /* storage unavailable */ }
}

// Save stations the agent has used on the current level. A run may start from
// any of them (or from the level's start) and from nowhere else: they are the
// game's own respawn points, so the level is built to be finishable from each,
// which an arbitrary "furthest point so far" is not. Each holds the state the
// game was in and the chain of pieces that led there from the level's start.
export function loadStations(levelIdx) {
  const s = bigLoad(STATIONS_KEY, (v) => !!v && Array.isArray(v.list));
  return s && s.levelIdx === levelIdx ? s.list : [];
}
export function saveStations(levelIdx, list) { bigSave(STATIONS_KEY, { levelIdx, list }); }

// The recorded playthrough as a standalone file (web/data/ppo-demo.json): what
// "Full game demo" plays. Written by the headless trainer so the result of
// local training can be committed and shown by the page, wherever it runs.
export const DEMO_FORMAT = 'abuse-ppo-demo-1';
export function demoFile(campaign, rec) {
  const pieces = [];
  for (const leg of campaign.legs) for (const piece of leg.pieces) pieces.push({ ...piece, levelIdx: leg.level });
  if (rec && rec.levelIdx === campaign.frontier) for (const piece of rec.pieces) pieces.push({ ...piece, levelIdx: rec.levelIdx });
  // `dist`: how far from the current level's exit the recording gets (see exitDist), for comparing recordings
  return { format: DEMO_FORMAT, savedAt: new Date().toISOString(), passed: campaign.legs.map((l) => l.level), frontier: campaign.frontier, dist: rec && rec.levelIdx === campaign.frontier ? rec.dist : null, pieces };
}

// Everything recorded so far as one playthrough: the passed levels in order,
// then the best progress on the current one. Each item is a piece plus the
// level it is played on.
export function campaignPieces() {
  const out = [];
  const c = loadCampaign();
  for (const leg of c.legs) for (const piece of leg.pieces) out.push({ ...piece, levelIdx: leg.level });
  const rec = loadBestRun();
  if (rec && rec.levelIdx === c.frontier) for (const piece of rec.pieces) out.push({ ...piece, levelIdx: rec.levelIdx });
  return out;
}

export function hasBestRun() { return !!loadBestRun(); }

// History of the best runs (top 5 by distance), for watching progress.
export function loadBestHistory() {
  try {
    const j = JSON.parse(localStorage.getItem(HIST_KEY));
    return Array.isArray(j) ? j.filter(validRec) : [];
  } catch { return []; }
}

// Things that are destroyed by shooting them and can be in the way.
const BREAK_AI = new Set(['hwall_ai', 'big_wall_ai', 'block_ai']);

const actParts = (act) => ({
  move: (act % 3) - 1,
  jump: Math.floor(act / 3) % 2 === 1,
  down: Math.floor(act / 6) % 2 === 1,
  fire: Math.floor(act / 12) % 2 === 1,
});

export { actParts };
export { keysToGo }; // (for tools/search.mjs)

// Which weapon to hold is chosen for the agent, like its aim: it has no
// weapon keys of its own, and without this it would carry rockets through a
// whole level and never fire one.
//  - at something tough (a breakable wall, a gun turret, a juggernaut, a
//    boulder) it uses rockets, when it has them and the target is far enough
//    away not to be caught in the blast;
//  - otherwise the plasma gun if it has ammo, else the machine gun;
//  - a machine gun with no ammo (it still fires, slowly) gives way to
//    anything that has some.
const TOUGH_AI = new Set(['spray_gun_ai', 'track_ai', 'jug_ai', 'bolder_ai', 'rob1_ai']);
function chooseWeapon(g, target, wall) {
  const p = g.player;
  const has = (w) => p.owned.has(w) && (p.ammo[w] || 0) > 0;
  const aim = target || wall;
  const far = aim && Math.abs(aim.x - p.x) + Math.abs(aim.y - p.y) > 90;
  let want = has('PGUN') ? 'PGUN' : 'MGUN';
  if (far && has('ROCKET') && (wall || TOUGH_AI.has(target.ai))) want = 'ROCKET';
  else if (want === 'MGUN' && !has('MGUN')) {
    if (far && has('ROCKET')) want = 'ROCKET';
    else if (far && has('GRENADE')) want = 'GRENADE';
    else if (far && has('FIREBOMB')) want = 'FIREBOMB';
  }
  if (p.weapon !== want) g.selectWeapon(want);
}

// Applies an action through the same input fields the human/bot use.
export function applyAction(g, act, faceDir = 1) {
  const { move, jump, down, fire } = actParts(act);
  g.keys.clear();
  g.rightDown = false;
  if (move < 0) g.keys.add('ArrowLeft');
  else if (move > 0) g.keys.add('ArrowRight');
  // "Jump" is also "up": on a ladder it climbs. The game takes Up for both
  // (Up jumps unless the player is on a ladder), while Space always jumps,
  // which would knock a climber straight off again.
  if (jump) {
    const p = g.player;
    const onLadder = g.ladders.some((l) => p.x >= l.x0 - 5 && p.x <= l.x1 + 5 && p.y >= l.y0 && p.y <= l.y1);
    g.keys.add(onLadder ? 'ArrowUp' : 'Space');
  }
  // The action key calls a lift whenever the rider presses it at a stop, so
  // an agent that is still pressing it on arrival is sent straight back the
  // way it came. Once a lift has started, the key is ignored until a second
  // after the ride, which is long enough to step off.
  let rideLock = false;
  for (const e of g.entities) {
    if (e.ai !== 'platform_ai' || e.dead) continue;
    if (e.aistate !== 0) e.a.lastMove = g.tickCount || 0;
    if (e.a.lastMove === undefined || (g.tickCount || 0) - e.a.lastMove > 15) continue;
    const deck = g.deckRect(e);
    if (deck && g.overDeck(deck) && Math.abs(g.player.y - deck.y0) < 60) rideLock = true;
  }
  // The same goes for a two-way switch that is already on: pressing it again
  // turns it off and shuts whatever it opened. The agent leaves switches on.
  let switchLock = false;
  for (const e of g.entities) {
    if (e.ai !== 'switcher_ai' || e.aistate !== 2 || e.dead) continue;
    // a little wider than the switch's own reach (20 x 30): the player moves between decisions
    if (Math.abs(g.player.x - e.x) < 34 && Math.abs(g.player.y - e.y) < 44) { switchLock = true; break; }
  }
  if (down && !rideLock && !switchLock) g.keys.add('ArrowDown');
  // A power the player is carrying needs the power button held (cop.cpp
  // do_special_power). The agent has no button of its own for it: running
  // uses the fast power, and "jump" is the thrust of the fly power.
  if (g.player.power === 'FAST') g.rightDown = move !== 0;
  else if (g.player.power === 'FLY') g.rightDown = jump;
  // Aim at the nearest visible enemy that can actually be hit; otherwise
  // straight ahead. Dormant enemies (flyer_ai in aistate 0 is not targetable
  // and takes no damage, flyer.lsp) are skipped, or the aim locks onto them
  // and every shot is wasted. The window matches the weapons' effective range
  // so long-range fire gets rewarded.
  const p = g.player;
  let target = null, best = Infinity;
  for (const e of g.entities) {
    if (e.dead || e.hidden || !e.shootable || !ENEMY_AI.has(e.ai)) continue;
    const dx = Math.abs(e.x - p.x), dy = Math.abs(e.y - p.y);
    if (dx > 600 || dy > 400) continue;
    if (!g.sees(p.x, p.y - 15, e.x, e.y - 15)) continue;
    const d = dx + dy;
    if (d < best) { best = d; target = e; }
  }
  // No enemy to shoot: aim at the nearest breakable wall in view instead, so
  // the fire button can open the routes that are walled off ("shoot hidden
  // walls to destroy them"). Walls under the feet are aimed at too.
  let wall = null;
  if (!target) {
    best = Infinity;
    for (const e of g.entities) {
      if (e.dead || !e.shootable || !BREAK_AI.has(e.ai)) continue;
      if (e.ai !== 'block_ai' && e.links.length && e.links[0].aistate === 0) continue; // wired walls can't be shot down
      const r = g.rectOf(e);
      if (!r) continue;
      const wx = (r.x0 + r.x1) / 2, wy = (r.y0 + r.y1) / 2;
      const dx = Math.abs(wx - p.x), dy = Math.abs(wy - (p.y - 15));
      if (dx > 170 || dy > 120 || dx + dy >= best) continue; // within the machine gun's reach
      if (!g.sees(p.x, p.y - 15, wx, wy)) continue;
      best = dx + dy; wall = { x: wx, y: wy };
    }
  }
  if (target) g.mouse = { x: target.x - g.cam.x, y: target.y - 12 - g.cam.y };
  else if (wall) g.mouse = { x: wall.x - g.cam.x, y: wall.y - g.cam.y };
  else g.mouse = { x: p.x + faceDir * 120 - g.cam.x, y: p.y - 20 - g.cam.y };
  chooseWeapon(g, target, wall);
  g.mouseDown = fire;
}

export class PpoBot {
  constructor(opts = {}) {
    this.net = opts.net || new Net();
    if (!opts.net) this.net.load(); // standalone (demo) mode uses trained weights
    this.trainer = opts.trainer || null;
    this.act = 0;
    this.faceDir = 1;
    this.n = 0;
    this.progressSig = null;
    this.lastDist = Infinity;
  }
  step(g) {
    if (!g.level) return;
    const p = g.player;
    // demo stall-timeout progress marker (same semantics as the heuristic bot)
    if (g.demoTimeoutTicks) {
      const sig = Math.floor(p.x / 40) * 1000 + Math.floor(p.y / 40);
      if (sig !== this.progressSig) {
        this.progressSig = sig;
        g.demoStartTick = g.tickCount || 0;
      }
      if ((g.tickCount || 0) - (g.demoStartTick || 0) > g.demoTimeoutTicks) g.onDemoTimeout?.();
    }
    if (!this.trainer) {
      // evaluation mode: choose an action at 15 Hz and hold it in between
      this.n = (this.n + 1) % 4;
      if (this.n === 0) {
        // navigation distance to the zone that advances to the next level
        const lm = /level(\d+)/.exec(g.level?.name || '');
        this.lastDist = exitDist(g, lm ? +lm[1] + 1 : null);
        const obs = buildObs(g, this.lastDist);
        const { logits } = this.net.forward(obs);
        let act = 0;
        if (Math.random() < 0.02) act = Math.floor(Math.random() * ACTS);
        else for (let i = 1; i < ACTS; i++) if (logits[i] > logits[act]) act = i;
        this.act = act;
        const { move } = actParts(act);
        if (move !== 0) this.faceDir = move;
      }
    }
    // during training the trainer supplies this.act directly
    applyAction(g, this.act, this.faceDir);
  }
}

// ---- trainer ----

// Objects the action key (down) activates on touch: pressing down while
// touching one is a deliberate interaction the reward should credit.
// Save stations (restart_ai) are not here: using one is paid separately, once
// per station per run (see STATION), because a used station becomes a place
// later runs can start from.
const INTERACT_AI = new Set([
  'tp2_ai', 'tpd_ai', 'platform_ai', 'switcher_ai', 'strap_door_ai', 'sdoor_ai', 'next_level_ai',
]);

// Breakable walls (original: "Shoot hidden walls to destroy them", wall() explodes at hp<=0).
const WALL_AI = new Set(['hwall_ai', 'big_wall_ai']);

// ---- best-run state checkpointing: the next run starts from the exact state
// where the best run ended (player + entities + doors/switches), instead of
// re-simulating the action script from spawn (which drifts). ----
const SNAP_V = 2;
const SNAP_FIELDS = ['x', 'y', 'vx', 'vy', 'dir', 'state', 'frame', 'stateTime', 'aistate', 'aitype', 'hp', 'xvel', 'yvel', 'xacel', 'yacel', 'fade', 'shootable', 'hidden'];

function snapshotState(g) {
  const p = g.player;
  // Objects are matched by the id they were given at level load (their place
  // in the level file). The live list can't be used for that: it shrinks
  // whenever something dies and grows with every explosion and bullet cloud.
  const es = {};
  for (const e of g.entities) {
    if (e.dead || e.id === undefined) continue;
    const o = {};
    for (const k of SNAP_FIELDS) o[k] = e[k];
    o.links = e.links.filter((l) => l.id !== undefined).map((l) => l.id);
    es[e.id] = o;
  }
  return {
    v: SNAP_V,
    px: Math.round(p.x), py: Math.round(p.y), hp: p.hp,
    weapon: p.weapon, owned: [...(p.owned || [])],
    ammo: { ...(p.ammo || {}) }, // ammo is keyed by weapon name, not an array
    power: p.power || null,
    start: { ...g.startPos },
    es,
  };
}

// Call on a freshly loaded level.
function restoreState(g, st) {
  const p = g.player;
  p.x = st.px; p.y = st.py; p.vx = 0; p.vy = 0;
  p.hp = st.hp; p.dead = false; p.deadTime = 0;
  p.weapon = st.weapon || 'MGUN';
  p.owned = new Set(st.owned || []);
  p.ammo = { ...(st.ammo || {}) };
  p.power = st.power || null;
  if (st.start) g.startPos = { ...st.start };
  const byId = new Map();
  for (const e of g.entities) if (e.id !== undefined) byId.set(e.id, e);
  for (const e of g.entities) {
    const s = e.id === undefined ? null : st.es[e.id];
    if (!s) { e.dead = true; continue; } // it had died (or is a leftover effect)
    for (const k of SNAP_FIELDS) if (s[k] !== undefined) e[k] = s[k];
    if (!e.def.states.has(e.state)) e.state = 'stopped';
    e.px = e.x; e.py = e.y;
  }
  // links second: they can change in play (movers hand objects over, the dead are unlinked)
  for (const e of g.entities) {
    if (e.dead) continue;
    e.links = st.es[e.id].links.map((id) => byId.get(id)).filter((l) => l && !l.dead);
  }
  g.entities = g.entities.filter((e) => !e.dead);
  g.projs = [];
  g.refreshSolids();
  // the view goes with the player: left where the level starts, it would
  // glide across the level, with the wrong objects awake until it arrived
  g.cam.x = p.x - 160; g.cam.y = p.y - 114;
}

export { snapshotState, restoreState };

// Starts a run the one way every run starts — training episodes, on-screen
// replays and the "Replay best run" button alike — so a recorded run can be
// reproduced exactly: load the level, reset everything that outlives a level
// load, then (for runs resumed from a checkpoint) restore the saved state.
export async function startRun(g, levelFile, seed = null) {
  await g.start(levelFile);
  g.freshRun();
  if (!seed) return;
  restoreState(g, seed);
  // a checkpoint taken standing in an exit would end the run on its first step
  const onExit = g.entities.some((e) => !e.dead && e.ai === 'next_level_ai'
    && Math.abs(e.x - g.player.x) < 60 && Math.abs(e.y - g.player.y) < 60);
  if (onExit) g.respawn();
}

// Runs are ranked by the level they reached first, then by how close to that
// level's exit they got.
const levelOf = (rec, levels) => (rec ? (rec.levelIdx ?? levels.indexOf(rec.level)) : -1);
export const betterRun = (lvlA, distA, lvlB, distB) => lvlA > lvlB || (lvlA === lvlB && distA < distB);
// A new record has to beat the old one by a real margin on the same level.
// Without it, a run resumed from the best checkpoint "beats" the record on its
// very first step by a rounding error, and overwrites the real run with an
// empty one.
const beatsRecord = (lvl, dist, saved) => lvl > saved.lvl || (lvl === saved.lvl && dist < saved.dist - 0.5);


export class PpoTrainer {
  constructor(game, levelFiles, onStatus) {
    this.g = game;
    this.levels = levelFiles;
    this.onStatus = onStatus || (() => {});
    this.net = new Net();
    this.loaded = this.net.load();
    this.bot = new PpoBot({ net: this.net, trainer: this });
    this.running = false;
    this.resetting = true;
    this.paused = false;
    this.updating = null;
    this.lr = 3e-4;
    this.traj = { obs: [], act: [], rew: [], val: [], lp: [], done: [] };
    this.lastObs = null;
    this.lastAct = 0;
    this.lastVal = 0;
    this.lastLp = 0;
    this.lastDist = Infinity;
    this.prevDist = Infinity;
    this.noProgT = 0;
    this.lastX = 0;
    this.lastEnemyCount = 0;
    this.episodeSteps = 0;
    this.successes = 0;
    this.episodes = 0;
    this.updates = 0;
    this.rollouts = 0;
    this.levelIdx = 0;
    // per-episode action script + best progress (min path distance seen)
    this.recActs = [];
    this.recBestDist = Infinity;
    this.bestSnap = null;     // checkpoint taken at this episode's best point, if it set a record
    this.bestMark = Infinity;
    this.saved = { lvl: -1, dist: Infinity }; // the best run on record: level reached, distance left
    this.startIdx = 0;        // level this episode started on
    this.cleared = 0;         // 1 once this episode has passed its level
    this.epFrom = null;       // the save-station checkpoint this episode started from, if any
    this.stationHit = null;   // a save station used since the last decision: 'x,y'
    this.levelStartStep = 0;
    this.epSeeded = false;
    this.statusAt = 0;
    this.lastFrameAt = 0;
    this.epNoProg = 0;  // sim steps since last movement or best-distance improvement
    this.enemyHp = new Map(); // per-enemy hp snapshot for damage-based rewards
    this.prevPosX = null;
    this.prevPosY = null;
    this.lastWallCount = 0;
    this.visited = new Set(); // 40px cells visited this episode (exploration bonus)
    this.paid = new Set();    // one-off bonuses already paid this episode
    // how many runs have reached each 40 px cell of each level ("level:cell")
    this.visits = new Map();
    try { this.visits = new Map(JSON.parse(localStorage.getItem(VISITS_KEY) || '[]')); } catch { /* start empty */ }
    // Training runs unseen. After every `showEvery` policy updates the latest
    // attempt is replayed on screen (a run that passed the level is shown
    // once, when it happens), then training carries on.
    this.showEvery = 50;
    this.lastAttempt = null;  // { startIdx, seed, acts, steps, end, endIdx, cleared, how } of the latest finished episode
    this.winRun = null;       // a run that passed the level, until it has been shown
    this.replay = null;       // the replay in progress
    this.epSeedState = null;  // state this episode started from (null = the level's own start)
    this.showDue = false;
    this.lastShow = '';       // what was shown last, for the status line
    // next_level zones carry their destination level in aistate (original
    // people.lsp): only the zone loading THIS level's successor is a win.
    // An exit to a later level (the next one, or a secret exit that skips
    // ahead) passes this level; an exit leading back just ends the run.
    this.trainExit = (dest) => {
      if (this.resetting) return;
      this.g.transitioning = true; // the exit zone fires every tick while stood in
      if (dest > this.levelIdx) this.passLevel(dest);
      else this.endEpisode(0, true, 'took an exit leading back');
    };
  }

  async start() {
    if (this.running) return;
    this.running = true;
    const g = this.g;
    this.prevNext = g.nextLevel;
    g.audio.muted = true;      // training runs silently
    g.autoPause = true;
    g.demo = true;           // isolate real input while training
    g.speed = 1;
    g.bot = this.bot;
    g.demoTimeoutTicks = 0;
    g.nextLevel = this.trainExit;
    // told when the player uses a save station (replays and walkthroughs don't count)
    this.origCheckpoint = g.setCheckpoint;
    g.setCheckpoint = (x, y) => {
      this.origCheckpoint.call(g, x, y);
      if (!this.replay && !this.resetting) this.stationHit = `${x},${y}`;
    };
    g.onDemoStop = () => this.stop();
    console.log(`[ppo] training started (${this.loaded ? 'resumed' : 'fresh'})`);
    await this.reset();
    // Training itself is not drawn (the screen holds the last thing shown);
    // the first look comes as soon as the first attempt has finished.
    this.g.renderThrottle = HIDDEN;
    this.showDue = this.showEvery > 0;
    // 16ms ticks, each working to a time budget (see frame()). setInterval
    // also survives background-tab throttling far better than rAF.
    this.timer = setInterval(() => this.frame(), 16);
  }

  saveVisits() {
    try { localStorage.setItem(VISITS_KEY, JSON.stringify([...this.visits])); } catch { /* quota: counts stay in memory */ }
  }

  // ---- replaying a recorded attempt ----

  async beginReplay(att) {
    const g = this.g;
    this.showDue = false;
    this.resetting = true;
    this.lastObs = null;
    try { await startRun(g, this.levels[att.startIdx], att.seed); } catch { /* shown next time */ }
    if (!this.running && !this.paused) return; // stopped while the level loaded
    g.bot = this.bot;
    this.bot.faceDir = 1;
    this.bot.act = 0;
    const full = Math.min(att.steps, att.acts.length * 4);
    const total = att.stallAt == null ? full : Math.min(full, att.stallAt + 60 * 6);
    const rp = this.replay = { att, n: 0, total, full, acc: 0, exited: false, lvl: att.startIdx, loading: null };
    g.nextLevel = () => { g.transitioning = true; rp.exited = true; }; // a run ends at an exit
    // run the part that is not shown at full speed, unseen
    const skip = Math.max(0, total - (att === this.winRun ? REPLAY_WIN : REPLAY_TAIL));
    while (rp.n < skip) {
      if (rp.loading) await rp.loading;
      if (!this.replayStep()) break;
    }
    if (this.replay !== rp) return; // stopped meanwhile
    g.renderThrottle = 0; // draw every frame from here
    this.resetting = false;
    this.onStatus(this.replayLabel());
  }

  replayLabel() {
    const a = this.replay.att;
    const what = a.cleared ? `the run that passed level ${a.startIdx}` : 'the latest attempt';
    return `replaying ${what}: episode ${a.episode}, ${a.how}${a.stallAt != null ? ' (shown up to where it stopped getting anywhere)' : ''}`;
  }

  // One 60 Hz step of the replay, fed the recorded action for that decision.
  // Returns false once the recording is over.
  replayStep() {
    const g = this.g, rp = this.replay;
    if (g.player.dead || rp.exited || rp.n >= rp.total) return false;
    if (rp.n % 4 === 0) {
      const act = rp.att.acts[rp.n / 4];
      const { move } = actParts(act);
      if (move !== 0) this.bot.faceDir = move;
      this.bot.act = act;
    }
    rp.n++;
    g.update(1 / 60);
    return true;
  }

  endReplay() {
    const rp = this.replay;
    if (!rp) return;
    this.replay = null;
    const g = this.g, p = g.player, a = rp.att;
    g.renderThrottle = HIDDEN; // hold this last frame while training continues
    // The replay should finish exactly where the recorded run did.
    const off = Math.abs(p.x - a.end[0]) + Math.abs(p.y - a.end[1]);
    if (rp.total === rp.full && (off > 4 || rp.lvl !== a.endIdx) && !rp.exited) console.warn(`[ppo] replay of episode ${a.episode} ended on level ${rp.lvl}, ${Math.round(off)}px from where the recorded run did (level ${a.endIdx})`);
    this.lastShow = `shown: ${a.cleared ? `the run that passed level ${a.startIdx}` : 'latest attempt'}, episode ${a.episode}, ${a.how}`;
    this.onStatus(this.lastShow);
    this.reset();
  }

  stop() {
    if (this.timer) { clearInterval(this.timer); this.timer = null; }
    if (!this.running && !this.paused) return;
    this.running = false;
    this.paused = false;
    this.updating = null; // drop any partially-applied gradient rollout
    this.replay = null;
    this.showDue = false;
    this.g.renderThrottle = 0;
    this.g.audio.muted = false; // restore sound after training
    const g = this.g;
    g.autoPause = false;
    g.demo = false;
    g.bot = null;
    g.nextLevel = this.prevNext;
    if (this.origCheckpoint) { g.setCheckpoint = this.origCheckpoint; this.origCheckpoint = null; }
    g.onDemoStop = null;
    g.keys.clear();
    g.mouseDown = false;
    this.maybeSaveBest();
    this.net.save();
    this.saveVisits();
    console.log(`[ppo] stopped — updates ${this.updates}, rollouts ${this.rollouts}, levels cleared ${this.successes} in ${this.episodes} runs`);
    this.onStatus(`stopped — ${this.updates} updates, ${this.successes} levels cleared`);
  }

  // Pause/resume without tearing the trainer down (keeps weights + trajectory).
  pause() {
    if (!this.running) return;
    this.running = false;
    this.paused = true;
    if (this.timer) { clearInterval(this.timer); this.timer = null; }
    this.g.keys.clear();
    this.g.mouseDown = false;
    console.log('[ppo] paused');
    this.onStatus('paused');
  }

  resume() {
    if (this.running || !this.paused) return;
    this.running = true;
    this.paused = false;
    this.timer = setInterval(() => this.frame(), 16);
    console.log('[ppo] resumed');
    this.onStatus('resumed');
  }

  // Per-level bookkeeping, at the start of a run and again on each new level.
  enterLevel() {
    const g = this.g;
    this.noProgT = 0;
    this.prevDist = Infinity;
    this.lastDist = Infinity;
    this.recBestDist = Infinity;
    this.bestMark = Infinity;
    this.levelStartStep = this.episodeSteps;
    this.lastX = g.player.x;
    this.lastY = g.player.y;
    this.areaX = g.player.x; this.areaY = g.player.y; this.areaStep = this.episodeSteps;
    this.keyBaseline = true; // (see policyStep: death sensors already satisfied earn nothing)
    this.prevSw = Infinity; this.prevSwKey = null;
    this.prevHp = g.player.hp;
    this.prevOwned = g.player.owned.size;
    this.prevAmmo = Object.values(g.player.ammo).reduce((a, b) => a + b, 0);
    this.lastEnemyCount = g.entities.filter((e) => !e.dead && e.shootable && ENEMY_AI.has(e.ai)).length;
    this.epNoProg = 0;
    this.enemyHp = new Map();
    this.prevPosX = null;
    this.prevPosY = null;
    this.lastWallCount = g.entities.filter((e) => !e.dead && WALL_AI.has(e.ai)).length;
    this.visited = new Set();
    this.paid = new Set();
  }

  async reset() {
    this.resetting = true;
    this.episodeSteps = 0;
    this.lastObs = null;
    this.recActs = [];
    this.bestSnap = null;
    this.cleared = 0;
    // Every run starts on the frontier: the first level not yet passed. It
    // never goes back to an earlier one.
    const camp = loadCampaign();
    const F = Math.min(camp.frontier, this.levels.length - 1);
    let rec = loadBestRun();
    if (rec && levelOf(rec, this.levels) !== F) rec = null; // a record from a level since passed
    this.saved = rec ? { lvl: F, dist: rec.dist } : { lvl: F, dist: Infinity };
    // A run starts either the way the game enters the level (with the weapons
    // and ammo carried in) or at a save station the agent has already used on
    // it. Nowhere else: a save station is a spot the level can be finished
    // from, which the furthest point reached so far need not be. The newest
    // station gets extra turns, since that is where the frontier is.
    const stations = loadStations(F);
    let from = null;
    if (stations.length) {
      const roll = Math.random();
      if (roll < 0.4) from = stations[stations.length - 1];
      else from = [null, ...stations][Math.floor(Math.random() * (stations.length + 1))];
    }
    this.epSeeded = !!from;
    this.epFrom = from;
    this.epSeedState = from ? from.state : (camp.entry || null);
    this.stationHit = null;
    this.levelIdx = this.startIdx = F;
    try { await startRun(this.g, this.levels[F], this.epSeedState); } catch { /* retry next time */ }
    this.simSteps = 0; // game steps actually run this episode
    this.bot.faceDir = 1; // aim direction must not carry over from the previous run
    this.bot.act = 0;
    this.g.bot = this.bot;
    this.g.nextLevel = this.trainExit;
    this.enterLevel();
    if (from) this.paid.add(from.key); // no bonus for "using" the station the run starts at
    this.resetting = false;
  }

  // This run, as the chain of pieces from the level's start: what the save
  // station it started from already had, plus its own actions up to `nActs` /
  // `steps`.
  piecesSoFar(nActs, steps) {
    const own = { seed: this.epSeedState, acts: encodeActs(this.recActs.slice(0, nActs)), steps };
    return this.epFrom ? [...this.epFrom.pieces, own] : [own];
  }

  // The level's exit was reached. If this is the frontier level, it is passed
  // for good: the pieces that got through it join the campaign, the next
  // level is loaded (the player keeps their weapons and ammo, as in the game)
  // and its opening state becomes where every later run starts.
  async passLevel(dest) {
    const g = this.g;
    const from = this.levelIdx;
    const camp = loadCampaign();
    const pieces = this.piecesSoFar(this.recActs.length, this.simSteps + (this.inUpdate ? 1 : 0));
    this.cleared = 1;
    this.successes++;
    // score and record the run before the level under it changes
    this.endEpisode(30, true, `reached the exit to level ${dest}`, false);
    console.log(`[ppo] passed ${this.levels[from]} -> level ${dest} (episode ${this.episodes}, ${this.successes} in total)`);
    if (from === camp.frontier && dest < this.levels.length) {
      this.resetting = true;
      this.onStatus(`passed level ${from} — training moves on to level ${dest}`);
      try {
        await g.start(this.levels[dest]);
        // A concession: the machine gun is topped up to its starting 100 rounds
        // on entering a level. Otherwise one run that arrived empty-handed
        // would be how every later run starts this level.
        g.player.ammo.MGUN = Math.max(g.player.ammo.MGUN || 0, 100);
        saveCampaign({ frontier: dest, entry: snapshotState(g), legs: [...camp.legs, { level: from, dest, pieces }] });
        for (const k of [BEST_KEY, HIST_KEY, STATIONS_KEY]) bigClear(k);
        this.winRun = null; this.lastAttempt = null; // replays restart with the new level
        this.onCampaign?.();
      } catch (err) { console.error('[ppo] could not enter the next level:', err); }
      this.resetting = false;
    }
    this.nextEpisode();
  }

  endEpisode(termReward, done, how = null, next = true) {
    if (this.resetting) return;
    // Keep the attempt (where it started + the inputs it was given) so it can
    // be replayed on screen.
    if (this.recActs.length >= 4) {
      const p = this.g.player;
      const att = {
        startIdx: this.startIdx, seed: this.epSeedState, acts: this.recActs.slice(),
        steps: this.simSteps + (this.inUpdate ? 1 : 0), end: [Math.round(p.x), Math.round(p.y)],
        // a run that gave up spent its last minute getting nowhere: when
        // replayed, it is shown up to a few seconds past its last progress
        stallAt: !how && !p.dead ? Math.max(0, this.simSteps - this.epNoProg) : null,
        endIdx: this.levelIdx, cleared: this.cleared, episode: this.episodes + 1,
        how: `${how || (p.dead ? 'died' : 'gave up (no progress)')} on level ${this.levelIdx}`,
      };
      this.lastAttempt = att;
      this.onAttempt?.(att);
      // the furthest any attempt has got; the latest one among equals
      if (att.cleared > 0) this.winRun = att; // a run that passed the level: shown once
    }
    if (this.lastObs === null) { this.episodes++; if (next) this.nextEpisode(); return; } // ended right after a policy update: nothing to score
    this.traj.obs.push(this.lastObs);
    this.traj.act.push(this.lastAct);
    this.traj.rew.push(termReward);
    this.traj.val.push(this.lastVal);
    this.traj.lp.push(this.lastLp);
    this.traj.done.push(done ? 1 : 0);
    this.lastObs = null;
    this.episodes++;
    if (!this.cleared) this.maybeSaveBest(); // a run that passed its level is the campaign's, not a "best so far"
    if (this.traj.obs.length >= ROLLOUT && !this.updating) this.update();
    if (next) this.nextEpisode();
  }

  // Between two training episodes: if a look is due, play it now (so no
  // attempt is ever cut short for it), otherwise start the next episode.
  nextEpisode() {
    if (this.showDue && this.running) {
      // A new furthest run is shown once, when it happens; every other look
      // is the attempt that has just finished, so no two looks are the same.
      const att = this.winRun && !this.winRun.shown ? this.winRun : this.lastAttempt;
      if (att) { att.shown = true; this.beginReplay(att); return; }
    }
    this.reset();
  }

  // Persist the best point this episode reached, if it beats the run on
  // record: a later level, or closer to the exit of the same one. The record
  // carries everything needed to replay it (where the run started and its
  // inputs) and a checkpoint to resume training from.
  // Returns true when a new best run was saved.
  maybeSaveBest() {
    const snap = this.bestSnap;
    if (!snap || this.recActs.length < 8 || !isFinite(snap.dist)) return false;
    const prev = loadBestRun();
    if (prev && !beatsRecord(snap.lvl, snap.dist, { lvl: levelOf(prev, this.levels), dist: prev.dist })) return false;
    const rec = {
      level: this.levels[snap.lvl],
      levelIdx: snap.lvl,
      dist: Math.round(snap.dist * 10) / 10,
      pieces: this.piecesSoFar(snap.n, snap.steps), // from the level's start to the best point
      end: snap.end,     // where the best point was
      state: snap.state, // exact checkpoint of the best point, to resume from
      episodes: this.episodes,
      updates: this.updates,
      t: Date.now(),
    };
    try { localStorage.setItem(BEST_KEY, JSON.stringify(rec)); } catch { return false; }
    this.saved = { lvl: rec.levelIdx, dist: rec.dist };
    this.bestSnap = null;
    // Keep a small history of the top runs so progress is watchable.
    try {
      const hist = loadBestHistory();
      const dup = hist.findIndex((h) => h.level === rec.level && h.dist === rec.dist);
      if (dup === -1) {
        hist.push(rec);
        hist.sort((a, b) => (betterRun(levelOf(a, this.levels), a.dist, levelOf(b, this.levels), b.dist) ? -1 : 1));
        if (hist.length > 3) hist.length = 3;
        localStorage.setItem(HIST_KEY, JSON.stringify(hist));
      }
    } catch { /* history is best-effort */ }
    this.net.save(); // checkpoint weights with the best run: the demo uses this model
    console.log(`[ppo] best run saved: ${rec.level}, dist ${rec.dist}, ${rec.pieces.length} piece(s) (episode ${this.episodes})`);
    this.onStatus(`best run saved — level ${rec.levelIdx}, ${fmtDist(rec.dist)} from its exit`);
    return true;
  }

  frame() {
    if (!this.running) return;
    const now = performance.now();
    const gap = this.lastFrameAt ? now - this.lastFrameAt : 0;
    this.lastFrameAt = now;
    try {
      if (this.replay) {
        if (!this.resetting && !this.replay.loading) {
          this.replay.acc += Math.min(gap, 100) / (1000 / 60) * REPLAY_SPEED;
          while (this.replay && !this.replay.loading && this.replay.acc >= 1) { this.replay.acc--; if (!this.replayStep()) this.endReplay(); }
        }
      } else if (!this.resetting) {
        {
          // Work to a time budget per tick: ~40ms when visible, up to ~950ms
          // when the tab is throttled to 1Hz, so background tabs still chew
          // through full rollouts instead of idling. The sim waits while a
          // rollout is being learned from, so every rollout is collected by
          // one fixed policy (as PPO assumes).
          const deadline = now + (gap > 500 ? 950 : 40);
          while (this.running && !this.resetting && performance.now() < deadline) {
            if (this.updating) this.stepUpdateSlice(deadline);
            else for (let i = 0; i < 16 && !this.updating && !this.resetting; i++) this.step();
          }
        }
      }
      // live status twice a second so the run counters are always visible
      if (now - (this.statusAt || 0) > 500) {
        this.statusAt = now;
        if (this.replay) {
          if (!this.resetting && !this.replay.loading) this.onStatus(`${this.replayLabel()} · ${Math.floor(this.replay.n / 60)}s of ${Math.floor(this.replay.total / 60)}s`);
        }
        else {
          const next = this.showEvery <= 0 ? '' : this.showDue ? ' · replay when this attempt ends' : ` · next replay in ${this.showEvery - (this.updates % this.showEvery)} updates`;
          this.onStatus(`training (not shown) · run ${this.episodes + 1} on level ${this.levelIdx} · ${loadCampaign().legs.length} levels passed · upd ${this.updates}${next}${this.lastShow ? ` · ${this.lastShow}` : ''}`);
        }
      }
    } catch (err) {
      console.error('[ppo] training error:', err);
      this.onStatus('error: ' + err.message);
      this.stop();
      return;
    }
  }

  step() {
    const g = this.g;
    const p = g.player;
    if (!p || !g.level || this.resetting) return;       // a level reload is in flight
    if (p.dead) { this.endEpisode(-10, true); return; } // death ends the episode
    this.episodeSteps++;
    if (this.episodeSteps % 4 === 1) this.policyStep();
    // Progress-aware episode length: only cut after a long stall, never while
    // the agent is still moving or improving, so slow-but-steady runs that
    // actually reach the exit are never truncated.
    // epNoProg is cleared (in policyStep) by reaching a cell new to this run
    // or a new best distance. Merely moving about does not count: hopping on
    // the spot used to keep a run alive for its full ten minutes.
    this.epNoProg++;
    // the area it is in: re-centred whenever it gets AREA pixels away from the centre
    if (Math.abs(p.x - this.areaX) > AREA || Math.abs(p.y - this.areaY) > AREA) { this.areaX = p.x; this.areaY = p.y; this.areaStep = this.episodeSteps; }
    if (this.epNoProg > STALL_ROAMING || (this.epNoProg > STALL && this.episodeSteps - this.areaStep > STALL)) { this.endEpisode(-3, true); return; }
    if (this.episodeSteps - this.levelStartStep > LEVEL_CAP) { this.endEpisode(-3, true); return; }
    this.inUpdate = true; // an exit reached inside this update ends the episode mid-step
    g.update(1 / 60);
    this.inUpdate = false;
    this.simSteps++;
  }

  policyStep() {
    const g = this.g;
    const p = g.player;
    // The navigation distance is exact on every decision (cached field), so
    // the progress reward lands on the action that earned it.
    // The distance to the exit is exact on every decision (cached fields), so
    // the progress reward lands on the action that earned it. Where there is
    // no reading at all, the last one is kept: no progress, no penalty.
    let effDist = exitDist(g, this.levelIdx + 1);
    if (effDist === Infinity) effDist = this.prevDist;
    const dist = this.lastDist = effDist;
    const fallback = effDist >= OPEN;

    // reward for the transition that just completed
    let r = -0.02; // step cost
    if (this.prevDist !== Infinity && isFinite(this.prevDist)) {
      // crossing between the kinds of reading is not progress either way
      const tier = (d) => (d >= LOCKED ? 2 : d >= OPEN ? 1 : 0);
      const gain = tier(this.prevDist) === tier(effDist) ? this.prevDist - effDist : 0;
      // Progress pays and retreat costs the same amount, so stepping back and
      // forth over a cell boundary nets zero instead of farming reward.
      r += Math.max(-12, Math.min(12, gain * (effDist >= LOCKED ? COMPASS : fallback ? 3 : COMPASS)));
    }
    this.prevDist = effDist;
    const enemyCount = g.entities.filter((e) => !e.dead && e.shootable && ENEMY_AI.has(e.ai)).length;
    if (enemyCount < this.lastEnemyCount) r += 3 * (this.lastEnemyCount - enemyCount);
    this.lastEnemyCount = enemyCount;
    // Dense combat credit: reward each point of damage dealt since the last
    // decision, not just kills (a kill's remaining hp dies with the entity,
    // so it is not double-counted here). Breakable walls pay their own rate
    // and a bonus on destruction (wall puzzles).
    let dmg = 0, wallDmg = 0;
    const hpNow = new Map();
    for (const e of g.entities) {
      if (e.dead || (!ENEMY_AI.has(e.ai) && !WALL_AI.has(e.ai))) continue;
      const prev = this.enemyHp.get(e);
      if (prev !== undefined && e.hp < prev) {
        if (ENEMY_AI.has(e.ai)) dmg += prev - e.hp;
        else wallDmg += prev - e.hp;
      }
      hpNow.set(e, e.hp);
    }
    this.enemyHp = hpNow;
    if (dmg > 0) r += Math.min(1.2, dmg * 0.06);
    if (wallDmg > 0) r += Math.min(1.2, wallDmg * 0.06);
    const wallCount = g.entities.filter((e) => !e.dead && WALL_AI.has(e.ai)).length;
    if (wallCount < this.lastWallCount) r += 3 * (this.lastWallCount - wallCount);
    this.lastWallCount = wallCount;
    // The bonuses below are each paid ONCE per thing per episode. Paid on
    // every decision they out-earned finishing the level (a win is 30): the
    // agent could hold down on a lift, bounce between two teleporters or fire
    // at an enemy behind a wall for as long as the stall timer allowed.
    //
    // Teleportation: an instantaneous position jump means a teleport actually
    // happened — credit the first use of each teleporter.
    if (this.prevPosX !== null) {
      const jump = Math.abs(p.x - this.prevPosX) + Math.abs(p.y - this.prevPosY);
      const from = `tp${Math.floor(this.prevPosX / 40)},${Math.floor(this.prevPosY / 40)}`;
      if (jump > 100 && !this.paid.has(from)) { this.paid.add(from); r += 5; }
    }
    this.prevPosX = p.x;
    this.prevPosY = p.y;
    // Interaction credit: the previous decision pressed down while touching
    // something the action key works (teleporters pay most: their graph
    // distance is already tiny while standing on them, so the press itself
    // must carry the signal). What the press achieves — a door opening, a
    // lift ride — is then paid for by the distance to the exit dropping.
    {
      const { down } = actParts(this.lastAct);
      if (down) {
        for (const e of g.entities) {
          if (e.dead || !INTERACT_AI.has(e.ai) || this.paid.has(e) || !g.touchesPlayer(e)) continue;
          this.paid.add(e);
          r += (e.ai === 'tp2_ai' || e.ai === 'tpd_ai') ? 2 : 1;
          break;
        }
      }
    }
    // Pickups: health regained (hearts, save stations) and ammo or weapons
    // collected. A heart is only taken when the player is hurt, so this pays
    // for picking one up when it is needed, not for walking past it at full
    // health.
    {
      if (p.hp > this.prevHp) r += (p.hp - this.prevHp) * HEALTH;
      this.prevHp = p.hp;
      let ammo = 0;
      for (const w in p.ammo) ammo += p.ammo[w];
      if (p.owned.size > this.prevOwned) r += 3 * (p.owned.size - this.prevOwned);
      else if (ammo > this.prevAmmo) r += PICKUP;
      this.prevAmmo = ammo; this.prevOwned = p.owned.size;
    }
    // Save stations: using one pays once per station per run, and the first
    // time any run uses a station it becomes a place later runs can start
    // from. The checkpoint is taken here, between two game steps, so that it
    // is exactly "this run after simSteps steps".
    if (this.stationHit) {
      const key = this.stationHit;
      this.stationHit = null;
      if (!this.paid.has(key)) {
        this.paid.add(key);
        r += STATION;
        this.epNoProg = 0;
        // The checkpoint kept for a station is the one saved with the most
        // switches on. Levels are opened up by their switches, and a run that
        // presses one and then saves has banked that progress: later runs
        // start there with the doors it opened still open.
        const list = loadStations(this.levelIdx);
        const sw = keysDone(g);
        const old = list.find((st) => st.key === key);
        if (!p.dead && (!old || sw > (old.sw || 0))) {
          const st = { key, sw, x: Math.round(p.x), y: Math.round(p.y), state: snapshotState(g), pieces: this.piecesSoFar(this.recActs.length, this.simSteps) };
          saveStations(this.levelIdx, old ? list.map((x) => (x === old ? st : x)) : [...list, st]);
          console.log(`[ppo] save station ${key} on level ${this.levelIdx}: ${old ? `saved again with ${sw} switch(es) on (was ${old.sw || 0})` : `reached, runs can now start there (${list.length + 1} station(s))`}`);
          this.onStations?.();
        }
      }
    }
    // Death sensors that were already satisfied when the run began, or that
    // watch nothing, are not this run's doing.
    if (this.keyBaseline) {
      this.keyBaseline = false;
      for (const e of g.entities) if (e.ai === 'death_sen_ai' && (e.aistate !== 0 || !e.links.length)) this.paid.add(e);
    }
    // Switches: turning one on pays, once per switch per run, and so does
    // getting closer to the nearest one that is still off. A step where the
    // set of off switches changed is skipped: the distance then jumps to a
    // different switch, which is not movement.
    {
      for (const e of g.entities) {
        if (e.dead || e.aistate === 0 || this.paid.has(e) || !(SWITCH_AI.has(e.ai) || e.ai === 'death_sen_ai')) continue; // a switch turned on, or a watched creature killed
        this.paid.add(e);
        r += SWITCH_ON;
        this.epNoProg = 0; // the level has changed: worth staying alive for
      }
      const sw = switchDist(g);
      // (when the switches ARE the way forward, the main compass above already follows them)
      if (effDist < LOCKED && sw.key === this.prevSwKey && isFinite(sw.dist) && isFinite(this.prevSw)) {
        r += Math.max(-6, Math.min(6, (this.prevSw - sw.dist) * SWITCH_PULL));
      }
      this.prevSw = sw.dist; this.prevSwKey = sw.key;
    }
    // Trigger discipline with dense signal: firing while a hittable enemy is
    // in range AND in sight pays every decision, so sustained fire is rewarded
    // between hits. The aim is automatic, so this fire lands and the enemy
    // dies: the bonus is bounded by its health. Dormant, invulnerable enemies
    // and enemies behind walls don't pay.
    {
      const { fire } = actParts(this.lastAct);
      if (fire) {
        for (const e of g.entities) {
          if (e.dead || !e.shootable || !ENEMY_AI.has(e.ai)) continue;
          if (Math.abs(e.x - p.x) <= 600 && Math.abs(e.y - p.y) <= 400 && g.sees(p.x, p.y - 15, e.x, e.y - 15)) { r += 0.2; break; }
        }
      }
    }
    // Exploration, count-based: the first time this run enters a 40 px cell it
    // is paid EXPLORE / sqrt(1 + how many earlier runs reached that cell). A
    // cell nobody has reached pays in full; the well-trodden start of a level
    // pays next to nothing, so the pull is always outward, to the frontier.
    {
      const sig = Math.floor(p.x / 40) * 1000 + Math.floor(p.y / 40);
      if (!this.visited.has(sig)) {
        this.visited.add(sig);
        const key = `${this.levelIdx}:${sig}`;
        const n = this.visits.get(key) || 0;
        this.visits.set(key, n + 1);
        r += EXPLORE / Math.sqrt(1 + n);
        this.epNoProg = 0; // somewhere new: the run is still going places
      }
    }
    // Standing still (in x AND y: riding a lift is not standing still) for
    // two seconds starts to cost.
    if (Math.abs(p.x - this.lastX) > 2 || Math.abs(p.y - this.lastY) > 2) { this.noProgT = 0; this.lastX = p.x; this.lastY = p.y; }
    else this.noProgT++;
    if (this.noProgT > 30) r -= 0.1; // stagnation

    if (this.lastObs !== null) {
      this.traj.obs.push(this.lastObs);
      this.traj.act.push(this.lastAct);
      this.traj.rew.push(r);
      this.traj.val.push(this.lastVal);
      this.traj.lp.push(this.lastLp);
      this.traj.done.push(0);
    }

    // next action
    const obs = buildObs(g, dist);
    const { logits, v } = this.net.forward(obs);
    const pr = softmax(logits);
    const act = sampleCat(pr, Math.random());
    const lp = Math.log(pr[act] + 1e-9);
    const { move } = actParts(act);
    if (move !== 0) this.bot.faceDir = move;
    this.bot.act = act;
    this.bot.lastDist = dist;
    this.lastObs = obs;
    this.lastAct = act;
    this.lastVal = v;
    this.lastLp = lp;
    // record this episode's action script for the best-run replay
    this.recActs.push(act);
    if (isFinite(this.prevDist)) {
      const before = this.recBestDist;
      this.recBestDist = Math.min(this.recBestDist, this.prevDist);
      if (this.recBestDist < before) {
        this.epNoProg = 0; // improving: keep the run alive
        // A new all-time best: checkpoint the game right here, at the best
        // point. (Saving where the run later ENDS would restart future runs
        // from wherever it wandered back to, fell to or died.)
        if (beatsRecord(this.levelIdx, this.recBestDist, this.saved) && this.recBestDist < this.bestMark - 0.5 && !p.dead && p.hp > 0) {
          this.bestMark = this.recBestDist;
          this.bestSnap = { lvl: this.levelIdx, dist: this.recBestDist, state: snapshotState(g), end: [Math.round(p.x), Math.round(p.y)], n: this.recActs.length, steps: this.simSteps };
        }
      }
    }

    if (this.traj.obs.length >= ROLLOUT && !this.updating) this.update(v);
  }

  // bootV: the critic's value of the state the rollout was cut at, so a
  // rollout that ends mid-episode is not scored as if the episode ended there.
  update(bootV = 0) {
    const R = this.traj;
    const n = R.obs.length;
    if (n < 64) return;
    // GAE over the rollout (cheap), then hand the per-sample weight updates to
    // stepUpdateSlice() so the main thread is never blocked for seconds.
    const gamma = 0.995, lam = 0.95;
    const adv = new Float32Array(n);
    const ret = new Float32Array(n);
    let lastAdv = 0;
    for (let t = n - 1; t >= 0; t--) {
      const nextV = t === n - 1 ? bootV : R.val[t + 1];
      const notDone = 1 - R.done[t];
      const delta = R.rew[t] + gamma * nextV * notDone - R.val[t];
      lastAdv = delta + gamma * lam * notDone * lastAdv;
      adv[t] = lastAdv;
      ret[t] = lastAdv + R.val[t];
    }
    let m = 0, s2 = 0;
    for (const a of adv) m += a;
    m /= n;
    for (const a of adv) s2 += (a - m) * (a - m);
    const sd = Math.sqrt(s2 / n) + 1e-6;
    for (let i = 0; i < n; i++) adv[i] = (adv[i] - m) / sd;
    this.updating = {
      R, adv, ret, n,
      order: [...Array(n).keys()],
      epoch: 0, pos: 0, epochs: EPOCHS, shuffled: false,
    };
    this.traj = { obs: [], act: [], rew: [], val: [], lp: [], done: [] };
    this.lastObs = null;
  }

  // Minibatch PPO epochs over the finished rollout, a slice at a time: runs
  // until `deadline` (performance.now() ms) or until the update is complete.
  stepUpdateSlice(deadline) {
    const u = this.updating;
    if (!u) return;
    const clipEps = 0.2, entCoef = 0.01;
    const { actor, critic } = this.net;
    while (performance.now() < deadline) {
      if (u.pos === 0 && !u.shuffled) {
        for (let i = u.order.length - 1; i > 0; i--) {
          const j = (Math.random() * (i + 1)) | 0;
          [u.order[i], u.order[j]] = [u.order[j], u.order[i]];
        }
        u.shuffled = true;
      }
      const end = Math.min(u.n, u.pos + BATCH);
      const bs = end - u.pos;
      for (; u.pos < end; u.pos++) {
        const t = u.order[u.pos];
        const x = u.R.obs[t];
        const a = u.R.act[t];
        const { logits, v } = this.net.forward(x);
        const pr = softmax(logits);
        const lp = Math.log(pr[a] + 1e-9);
        const ratio = Math.exp(lp - u.R.lp[t]);
        // clipped surrogate: no gradient once the ratio has left the trust
        // region in the direction the advantage is pushing it
        const clipped = u.adv[t] >= 0 ? ratio > 1 + clipEps : ratio < 1 - clipEps;
        const gLp = clipped ? 0 : -ratio * u.adv[t];
        let H = 0;
        for (const q of pr) H -= q * Math.log(q + 1e-9);
        for (let i = 0; i < ACTS; i++) {
          _dOut[i] = gLp * ((i === a ? 1 : 0) - pr[i]) + entCoef * pr[i] * (Math.log(pr[i] + 1e-9) + H);
        }
        actor.backward(_dOut);
        _critD[0] = v - u.ret[t];
        critic.backward(_critD);
      }
      actor.applyGrads(this.lr, 1 / bs);
      critic.applyGrads(this.lr, 1 / bs);
      if (u.pos >= u.n) {
        u.pos = 0;
        u.epoch++;
        u.shuffled = false;
        if (u.epoch >= u.epochs) {
          const done = u.R;
          this.updating = null;
          this.updates++;
          this.rollouts++;
          if (this.showEvery > 0 && this.updates % this.showEvery === 0) this.showDue = true;
          this.maybeSaveBest(); // mid-episode progress counts too
          if (this.rollouts % 10 === 0) { this.net.save(); this.saveVisits(); }
          this.onUpdateDone?.();
          const avg = done.rew.reduce((a, b) => a + b, 0) / u.n;
          console.log(`[ppo] update ${this.updates} · levels cleared ${this.successes} in ${this.episodes} runs · avgR ${avg.toFixed(2)}`);
          return;
        }
      }
    }
  }
}
