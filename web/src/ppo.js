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
export const PPO_KEY = 'abuse.ppo.v4'; // v4: once-only bonuses, walls in view, compass on every decision
export const BEST_KEY = 'abuse.ppo.bestrun4'; // 4: runs span levels; records carry their start and are replayed exactly
export const HIST_KEY = 'abuse.ppo.besthistory4';

const ACTS = 24;            // move(-1..1) x jump x down x fire
const ROLLOUT = 1024;       // decisions collected per policy update
const BATCH = 32;           // samples per Adam step
const EPOCHS = 2;           // passes over each rollout
// Training stops while a walkthrough plays (there is one game instance), so
// they are kept short: a hopeless run ends after ~2 s of real time.
const SHOW_SPEED = 5;       // walkthroughs play at 5x real time
const SHOW_MAX = 15 * 60;   // ...for at most 60 game-seconds (in 15 Hz ticks)
const SHOW_STALL = 15 * 8;  // ...and end after 8 game-seconds without getting anywhere new
const REPLAY_SPEED = 6;     // recorded attempts are replayed at 6x real time
const REPLAY_TAIL = 60 * 45;   // the last 45 game-seconds of a failed attempt are shown (60 Hz steps)
const REPLAY_WIN = 60 * 150;   // ...and up to the last 150 game-seconds of one that reached the exit
const HIDDEN = 1e12;        // renderThrottle value that never draws
// Reward weights. Exploring is paid on a par with following the compass, so a
// compass that points the wrong way (or at a door it can't open) cannot pin
// the agent in place: somewhere it has rarely been is always worth going to.
const COMPASS = 4;          // per cell closer to the exit along the walking route (was 8)
const EXPLORE = 3;          // for a 40 px cell no run has reached before; falls as 1/sqrt(runs that have)
const STALL = 15 * 30 * 4;  // a run ends after 30 game-seconds with nowhere new and no progress (60 Hz steps)
const VISITS_KEY = 'abuse.ppo.visits1';
const WIN_C = 20, WIN_R = 12, CH = 4;
export const OBS_N = WIN_C * WIN_R * CH + 13;

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
      if (e && (e.ai === 'tp2_ai' || e.ai === 'tpd_ai' || e.ai === 'platform_ai' || e.ai === 'sdoor_ai' || e.ai === 'strap_door_ai' || e.ai === 'switcher_ai' || BREAK_AI.has(e.ai))) special = 1;
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
  obs[k++] = (isFinite(dist) ? Math.min(dist >= OPEN ? dist - OPEN : dist, 5000) : 5000) / 50; // distance to the exit, capped
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
  return obs;
}

// ---- Dijkstra distance to the nearest exit (shared navigation model) ----

const MAX_FALL = 80, MAX_JUMP = 8, MAX_CLIMB = 3;

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
// Obstacles the player can get through: doors (opened by a switch or sensor)
// and anything that is destroyed by shooting it (walls, bricks, gun turrets).
const SOFT_AI = new Set(['sdoor_ai', 'strap_door_ai', 'hwall_ai', 'big_wall_ai', 'block_ai', 'ff_ai']);
// Solid objects that are not obstacles to the route: lifts are modelled by
// their stops (and leaving the moving deck out keeps the graph fixed while one
// travels); exits and teleporters are things the player walks into.
const NOT_WALL_AI = new Set(['platform_ai', 'next_level_ai', 'tp2_ai', 'tpd_ai']);
const isSoft = (e) => !!e && (SOFT_AI.has(e.ai) || !!e.shootable);
const SOFT_COST = 15;

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
  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) if (g.tileSolid(c * tw + tw / 2, r * th + th / 2)) grid[r * W + c] = 1;
  for (const s of g.solids) {
    if (s.e && NOT_WALL_AI.has(s.e.ai)) continue;
    const soft = isSoft(s.e);
    if (soft && DOOR_AI.has(s.e.ai) && s.e.aistate !== 0) continue; // already opening
    const c0 = Math.max(0, Math.floor(s.x0 / tw) - 1), c1 = Math.min(W - 1, Math.floor(s.x1 / tw) + 1);
    const r0 = Math.max(0, Math.floor(s.y0 / th) - 1), r1 = Math.min(H - 1, Math.floor(s.y1 / th) + 1);
    for (let r = r0; r <= r1; r++) for (let c = c0; c <= c1; c++) {
      const x = c * tw + tw / 2;
      if (x < s.x0 || x > s.x1 || (r + 1) * th <= s.y0 || r * th >= s.y1) continue;
      const idx = r * W + c;
      if (!soft) grid[idx] = 1; else if (grid[idx] === 0) grid[idx] = 2;
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
    // Jump up onto a ledge beside the player: the jump peaks 51 px up
    // (3 rows of 15 px), so ledges 2 and 3 rows higher are in reach when
    // there is headroom above the take-off cell.
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
      if (rr < H && floorOK(nc, rr)) relax(nc, rr, rr - r + 0.5, 'drop');
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
  return { W, H, N, grid, goals, expand, cellOf, floorOK, bodyOK, jumpTo };
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
  for (const x of g.solids) {
    if (x.e && NOT_WALL_AI.has(x.e.ai)) continue;
    const soft = isSoft(x.e);
    if (soft && DOOR_AI.has(x.e.ai) && x.e.aistate !== 0) continue; // already opening
    s += `;${soft ? 'S' : ''}${Math.ceil((x.x0 - tw / 2) / tw)},${Math.floor(x.y0 / th)},${Math.floor((x.x1 - tw / 2) / tw)},${Math.ceil(x.y1 / th) - 1}`;
  }
  return s;
}

function buildField(g, nextNum) {
  const { N, grid, goals, expand } = navGraph(g, nextNum);
  const field = new Float64Array(N).fill(Infinity);
  field.grid = grid;
  if (!goals.length) return field;
  // reversed edges as linked lists: head[to] -> edge -> next edge into `to`
  const head = new Int32Array(N).fill(-1);
  const eFrom = [], eCost = [], eNext = [];
  let cur = 0;
  const emit = (to, cost) => {
    eFrom.push(cur); eCost.push(cost); eNext.push(head[to]);
    head[to] = eFrom.length - 1;
  };
  for (cur = 0; cur < N; cur++) expand(cur, emit);
  const heap = makeHeap();
  for (const gl of goals) if (field[gl] !== 0) { field[gl] = 0; heap.push(0, gl); }
  while (heap.size) {
    const to = heap.pop();
    if (heap.key > field[to]) continue; // stale entry
    for (let e = head[to]; e !== -1; e = eNext[e]) {
      const from = eFrom[e];
      const nd = field[to] + eCost[e];
      if (nd < field[from]) { field[from] = nd; heap.push(nd, from); }
    }
  }
  return field;
}

// Second, looser field for wherever the walking graph finds no route (it
// can't model every lift, switch and puzzle): the distance through open space,
// ignoring gravity, around hard walls, through doors and breakable walls at a
// cost, and through teleporters. It still bends around the level's geometry,
// which a straight line to the exit does not.
function buildFlood(g, nextNum) {
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
  const seeds = goals.map(bodyCell);
  for (const f of walkGoals) if (f >= W) seeds.push(f - W); // where the walking graph stands to exit
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
    else if (e.ai === 'platform_ai' && e.links.length >= 2) {
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

let fieldCache = null;
export const navStats = { builds: 0, lookups: 0 };

// Both fields for the current world state (diagnostics and tooling).
export function navFields(g, nextNum = null) {
  navDist(g, nextNum);
  return { field: fieldCache.field, flood: (fieldCache.flood ??= buildFlood(g, nextNum)) };
}

// Open-space distance to the exit (see buildFlood); Infinity only where the
// player is sealed off from it by hard walls.
export function floodDist(g, nextNum = null) {
  const sig = navSig(g, nextNum);
  if (!fieldCache || fieldCache.sig !== sig) { fieldCache = { sig, field: buildField(g, nextNum) }; navStats.builds++; }
  const flood = (fieldCache.flood ??= buildFlood(g, nextNum));
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

// Offset that ranks every open-space reading behind every walking-route one.
export const OPEN = 1e6;

// The distance the trainer steers by: the walking route where there is one,
// otherwise the open-space distance (+OPEN), otherwise Infinity (no reading
// here; callers keep their previous value rather than guess).
export function exitDist(g, nextNum = null) {
  const walk = navDist(g, nextNum);
  if (walk !== Infinity) return walk;
  const open = floodDist(g, nextNum);
  return open === Infinity ? Infinity : OPEN + open;
}

// For people: "12" along the walking route, "~340" through open space.
export const fmtDist = (d) => (!isFinite(d) ? '?' : d >= OPEN ? `~${(d - OPEN).toFixed(0)}` : d.toFixed(0));


// Path distance (in cells) from the player to the nearest exit that advances
// to level `nextNum`; Infinity when no route exists. Cheap after the first call.
export function navDist(g, nextNum = null) {
  const sig = navSig(g, nextNum);
  if (!fieldCache || fieldCache.sig !== sig) { fieldCache = { sig, field: buildField(g, nextNum) }; navStats.builds++; }
  navStats.lookups++;
  const { field } = fieldCache;
  const W = g.level.fgW, H = g.level.fgH;
  // Riding a lift: the graph only knows its two stops, so in between the
  // distance is read off the ride itself, sliding evenly from one stop's value
  // to the other's. (Looked up by cell, a ride reads as hanging in mid-air and
  // the distance climbs most of the way up, which punishes taking the lift.)
  const p = g.player;
  fieldCache.lifts ??= g.entities.filter((e) => e.ai === 'platform_ai' && e.links.length >= 2);
  for (const e of fieldCache.lifts) {
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
  save() {
    const dump = (m) => ({
      w1: Array.from(m.d1.w), b1: Array.from(m.d1.b),
      w2: Array.from(m.d2.w), b2: Array.from(m.d2.b),
      wo: Array.from(m.do.w), bo: Array.from(m.do.b),
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
export function loadBestRun() {
  try {
    const j = JSON.parse(localStorage.getItem(BEST_KEY));
    return j && Array.isArray(j.acts) && j.acts.length ? j : null;
  } catch { return null; }
}

export function hasBestRun() { return !!loadBestRun(); }

// History of the best runs (top 5 by distance), for watching progress.
export function loadBestHistory() {
  try {
    const j = JSON.parse(localStorage.getItem(HIST_KEY));
    return Array.isArray(j) ? j.filter((r) => r && Array.isArray(r.acts) && r.acts.length) : [];
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

// Applies an action through the same input fields the human/bot use.
export function applyAction(g, act, faceDir = 1) {
  const { move, jump, down, fire } = actParts(act);
  g.keys.clear();
  g.rightDown = false;
  if (move < 0) g.keys.add('ArrowLeft');
  else if (move > 0) g.keys.add('ArrowRight');
  if (jump) g.keys.add('Space');
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
  if (down && !rideLock) g.keys.add('ArrowDown');
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
// Save stations (restart_ai) are deliberately absent: a death ends the episode
// and the level is reloaded, so they do nothing for the agent, and paying for
// pressing down on one just teaches it to stand there.
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
    this.cleared = 0;         // levels cleared this episode
    this.levelStartStep = 0;
    // warm-start (frontier resume) state: disabled when the seed endpoint traps
    this.seedOk = true;
    this.seedStrike = 0;
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
    // Walkthroughs: training runs unseen; after every `showEvery` policy
    // updates the current policy plays one run from spawn on screen, picking
    // its best action each time (no exploration noise).
    this.showEvery = 50;
    // What is shown: 'last' = a replay of a recorded training attempt: the most
    // recent one that reached the exit if there has been one, otherwise the
    // most recent attempt, exploration noise and all; 'best' = a separate
    // walkthrough of the policy's best actions.
    this.showMode = 'last';
    this.lastAttempt = null;  // { startIdx, seed, acts, steps, end, endIdx, cleared, how } of the latest finished episode
    this.winRun = null;       // the attempt that got through the most levels (latest among equals)
    this.replay = null;       // the replay in progress
    this.epSeedState = null;  // checkpoint this episode started from (null = spawn)
    this.show = null;         // the walkthrough in progress
    this.showDue = false;
    this.lastShow = '';       // result of the last walkthrough, for the status line
    // next_level zones carry their destination level in aistate (original
    // people.lsp): only the zone loading THIS level's successor is a win.
    // An exit to a later level (the next one, or a secret exit that skips
    // ahead) clears this level and the run carries on there, as in the game.
    this.trainExit = (dest) => {
      if (this.resetting) return;
      this.g.transitioning = true; // the exit zone fires every tick while stood in
      if (dest > this.levelIdx && dest < this.levels.length) this.advance(dest);
      else if (dest >= this.levels.length) this.endEpisode(30, true, 'finished the last level');
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
    const total = Math.min(att.steps, att.acts.length * 4);
    const rp = this.replay = { att, n: 0, total, acc: 0, exited: false, lvl: att.startIdx, loading: null };
    // exits behave as they did in the recorded run: on to a later level, or the end
    g.nextLevel = (dest) => {
      if (rp.loading || rp.exited) return;
      g.transitioning = true;
      if (dest > rp.lvl && dest < this.levels.length) {
        rp.loading = g.start(this.levels[dest]).then(() => { rp.lvl = dest; rp.loading = null; }, () => { rp.exited = true; rp.loading = null; });
      } else rp.exited = true;
    };
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
    const what = a === this.winRun ? `the furthest run so far (cleared ${a.cleared} level${a.cleared > 1 ? 's' : ''})` : 'the latest attempt';
    return `replaying ${what}: episode ${a.episode}, ${a.how}, started on level ${a.startIdx}${a.seed ? ' from the best checkpoint' : ''}`;
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
    if ((off > 4 || rp.lvl !== a.endIdx) && !rp.exited) console.warn(`[ppo] replay of episode ${a.episode} ended on level ${rp.lvl}, ${Math.round(off)}px from where the recorded run did (level ${a.endIdx})`);
    this.lastShow = `shown: ${a === this.winRun ? `furthest run so far (cleared ${a.cleared})` : 'latest attempt'}, episode ${a.episode}, ${a.how}`;
    this.onStatus(this.lastShow);
    this.reset();
  }

  // ---- walkthroughs ----

  async beginShow() {
    const g = this.g;
    this.showDue = false;
    this.resetting = true;
    this.lastObs = null;
    this.levelIdx = 0; // walkthroughs play the first level
    try { await startRun(g, this.levels[0]); } catch { /* shown next time */ }
    if (!this.running && !this.paused) return; // stopped while the level loaded
    g.bot = this.bot;
    g.nextLevel = (dest) => this.endShow(dest > 0 ? `reached the exit to level ${dest}` : 'took an exit leading back');
    g.renderThrottle = 0; // draw every frame
    const d = exitDist(g, 1);
    this.show = { n: 0, acc: 0, startDist: d, bestDist: d, sig: null, lastNew: 0 };
    this.resetting = false;
    this.onStatus(`walkthrough after update ${this.updates}…`);
  }

  // One 60 Hz game step of the walkthrough; the policy's best action is
  // re-chosen every 4th step, as in training.
  showStep() {
    const g = this.g, p = g.player, sh = this.show;
    if (p.dead) { this.endShow('died'); return; }
    if (sh.n % 4 === 0) {
      const dist = exitDist(g, 1);
      if (dist < sh.bestDist) sh.bestDist = dist;
      const { logits } = this.net.forward(buildObs(g, dist));
      let act = 0;
      for (let i = 1; i < ACTS; i++) if (logits[i] > logits[act]) act = i;
      const { move } = actParts(act);
      if (move !== 0) this.bot.faceDir = move;
      this.bot.act = act;
      const sig = Math.floor(p.x / 40) * 1000 + Math.floor(p.y / 40);
      if (sig !== sh.sig) { sh.sig = sig; sh.lastNew = sh.n; }
    }
    sh.n++;
    g.update(1 / 60);
    if (!this.show) return; // reached an exit during the update
    if (sh.n - sh.lastNew > SHOW_STALL * 4) this.endShow('got stuck');
    else if (sh.n > SHOW_MAX * 4) this.endShow('ran out of time');
  }

  endShow(how) {
    const sh = this.show;
    if (!sh) return;
    this.show = null;
    const g = this.g;
    g.renderThrottle = HIDDEN; // hold this last frame while training continues
    const fmt = fmtDist;
    this.lastShow = `last walkthrough (update ${this.updates}): ${how} after ${Math.round(sh.n / 60)}s, distance to exit ${fmt(sh.startDist)} → ${fmt(sh.bestDist)}`;
    console.log(`[ppo] ${this.lastShow}`);
    this.onStatus(this.lastShow);
    this.reset();
  }

  stop() {
    if (this.timer) { clearInterval(this.timer); this.timer = null; }
    if (!this.running && !this.paused) return;
    this.running = false;
    this.paused = false;
    this.updating = null; // drop any partially-applied gradient rollout
    this.show = null;     // and any walkthrough
    this.replay = null;
    this.showDue = false;
    this.g.renderThrottle = 0;
    this.g.audio.muted = false; // restore sound after training
    const g = this.g;
    g.autoPause = false;
    g.demo = false;
    g.bot = null;
    g.nextLevel = this.prevNext;
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
    const rec = loadBestRun();
    this.saved = rec ? { lvl: levelOf(rec, this.levels), dist: rec.dist } : { lvl: -1, dist: Infinity };
    // Warm-start half of the episodes from the best run's checkpoint (which
    // may be on a later level), so training resumes from the frontier; the
    // rest start at the first level's spawn, so the policy keeps practising
    // the whole game. If checkpoint starts stop improving (the spot is a
    // trap), fall back to spawn starts until exploration finds a new best.
    const seeded = this.seedOk && !!rec && !!rec.state && rec.state.v === SNAP_V && this.saved.lvl >= 0
      && Math.random() < (this.seedStrike > 0 ? 0.25 : 0.5);
    this.epSeeded = seeded;
    this.epSeedState = seeded ? rec.state : null;
    this.levelIdx = this.startIdx = seeded ? this.saved.lvl : 0;
    try { await startRun(this.g, this.levels[this.levelIdx], this.epSeedState); } catch { /* retry next time */ }
    this.simSteps = 0; // game steps actually run this episode
    this.bot.faceDir = 1; // aim direction must not carry over from the previous run
    this.bot.act = 0;
    this.g.bot = this.bot;
    this.g.nextLevel = this.trainExit;
    this.enterLevel();
    this.resetting = false;
  }

  // The level's exit was reached: score it like the end of an episode (the
  // value of the next level is not this level's business), then carry the run
  // on into the level the exit leads to, weapons and ammo included.
  async advance(dest) {
    if (this.lastObs !== null) {
      this.traj.obs.push(this.lastObs);
      this.traj.act.push(this.lastAct);
      this.traj.rew.push(30);
      this.traj.val.push(this.lastVal);
      this.traj.lp.push(this.lastLp);
      this.traj.done.push(1);
      this.lastObs = null;
    }
    this.resetting = true;
    this.successes++;
    this.cleared++;
    console.log(`[ppo] cleared ${this.levels[this.levelIdx]} -> level ${dest} (episode ${this.episodes + 1}, ${this.successes} cleared in total)`);
    this.onStatus(`cleared level ${this.levelIdx} — on to level ${dest}`);
    try { await this.g.start(this.levels[dest]); } catch { this.resetting = false; this.endEpisode(0, true, 'next level failed to load'); return; }
    this.levelIdx = dest;
    this.enterLevel();
    if (this.traj.obs.length >= ROLLOUT && !this.updating) this.update();
    this.resetting = false;
  }

  endEpisode(termReward, done, how = null) {
    if (this.resetting) return;
    // Keep the attempt (where it started + the inputs it was given) so it can
    // be replayed on screen.
    if (this.recActs.length >= 4) {
      const p = this.g.player;
      const att = {
        startIdx: this.startIdx, seed: this.epSeedState, acts: this.recActs.slice(),
        steps: this.simSteps + (this.inUpdate ? 1 : 0), end: [Math.round(p.x), Math.round(p.y)],
        endIdx: this.levelIdx, cleared: this.cleared, episode: this.episodes + 1,
        how: `${how || (p.dead ? 'died' : 'gave up (no progress)')} on level ${this.levelIdx}`,
      };
      this.lastAttempt = att;
      // the furthest any attempt has got; the latest one among equals
      if (att.cleared > 0 && (!this.winRun || att.endIdx >= this.winRun.endIdx)) this.winRun = att;
    }
    if (this.lastObs === null) { this.episodes++; this.nextEpisode(); return; } // ended right after a policy update: nothing to score
    this.traj.obs.push(this.lastObs);
    this.traj.act.push(this.lastAct);
    this.traj.rew.push(termReward);
    this.traj.val.push(this.lastVal);
    this.traj.lp.push(this.lastLp);
    this.traj.done.push(done ? 1 : 0);
    this.lastObs = null;
    this.episodes++;
    const improved = this.maybeSaveBest();
    if (improved) { this.seedOk = true; this.seedStrike = 0; }
    else if (this.epSeeded) {
      // Started from the best checkpoint but made no headway: likely a trap.
      this.seedStrike++;
      if (this.seedStrike >= 3) {
        this.seedOk = false;
        console.log('[ppo] checkpoint starts not improving — exploring from spawn');
        this.onStatus('checkpoint stuck — exploring from spawn');
      }
    }
    if (this.traj.obs.length >= ROLLOUT && !this.updating) this.update();
    this.nextEpisode();
  }

  // Between two training episodes: if a look is due, play it now (so no
  // attempt is ever cut short for it), otherwise start the next episode.
  nextEpisode() {
    if (this.showDue && this.running) {
      if (this.showMode === 'best') { this.beginShow(); return; }
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
      start: { idx: this.startIdx, seed: this.epSeedState }, // where the run began
      acts: this.recActs.slice(0, snap.n), // its inputs up to the best point
      steps: snap.steps,                   // ...which is this many game steps in
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
        if (hist.length > 5) hist.length = 5;
        localStorage.setItem(HIST_KEY, JSON.stringify(hist));
      }
    } catch { /* history is best-effort */ }
    this.net.save(); // checkpoint weights with the best run: the demo uses this model
    console.log(`[ppo] best run saved: ${rec.level}, dist ${rec.dist}, ${rec.acts.length} actions (episode ${this.episodes})`);
    this.onStatus(`best run saved — level ${rec.levelIdx}, ${fmtDist(rec.dist)} from its exit`);
    return true;
  }

  frame() {
    if (!this.running) return;
    const now = performance.now();
    const gap = this.lastFrameAt ? now - this.lastFrameAt : 0;
    this.lastFrameAt = now;
    try {
      if (this.show) {
        // play the walkthrough at SHOW_SPEED x real time
        this.show.acc += Math.min(gap, 100) / (1000 / 60) * SHOW_SPEED;
        while (this.show && this.show.acc >= 1) { this.show.acc--; this.showStep(); }
      } else if (this.replay) {
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
        } else if (this.show) {
          this.onStatus(`walkthrough after update ${this.updates} · ${Math.floor(this.show.n / 60)}s · distance to exit ${fmtDist(this.show.bestDist)}`);
        }
        else {
          const what = this.showMode === 'best' ? 'walkthrough' : 'replay';
          const next = this.showEvery <= 0 ? '' : this.showDue ? ` · ${what} when this attempt ends` : ` · next ${what} in ${this.showEvery - (this.updates % this.showEvery)} updates`;
          this.onStatus(`training (not shown) · run ${this.episodes + 1} on level ${this.levelIdx} · levels cleared ${this.successes} · upd ${this.updates}${next}${this.lastShow ? ` · ${this.lastShow}` : ''}`);
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
    if (this.epNoProg > STALL) { this.endEpisode(-3, true); return; }
    if (this.episodeSteps - this.levelStartStep > 15 * 600 * 4) { this.endEpisode(-3, true); return; } // 600 sim-seconds per level
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
      // crossing between the two distance scales is not progress either way
      const gain = (this.prevDist >= OPEN) === fallback ? this.prevDist - effDist : 0;
      // Progress pays and retreat costs the same amount, so stepping back and
      // forth over a cell boundary nets zero instead of farming reward.
      r += Math.max(-12, Math.min(12, gain * (fallback ? 1.5 : COMPASS)));
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
          if (this.maybeSaveBest()) { this.seedOk = true; this.seedStrike = 0; } // mid-episode progress counts too
          if (this.rollouts % 10 === 0) { this.net.save(); this.saveVisits(); }
          const avg = done.rew.reduce((a, b) => a + b, 0) / u.n;
          console.log(`[ppo] update ${this.updates} · levels cleared ${this.successes} in ${this.episodes} runs · avgR ${avg.toFixed(2)}`);
          return;
        }
      }
    }
  }
}
