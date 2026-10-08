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

export const PPO_KEY = 'abuse.ppo.v3'; // v3: trained with the fixed compass (door-gap route)
export const BEST_KEY = 'abuse.ppo.bestrun2';
export const HIST_KEY = 'abuse.ppo.besthistory2';

const ACTS = 24;            // move(-1..1) x jump x down x fire
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
  applyGrads(lr) {
    for (let i = 0; i < this.w.length; i++) {
      const g = this.gw[i];
      this.mw[i] = 0.9 * this.mw[i] + 0.1 * g;
      this.vw[i] = 0.999 * this.vw[i] + 0.001 * g * g;
      this.w[i] -= lr * this.mw[i] / (Math.sqrt(this.vw[i]) + 1e-5);
      this.gw[i] = 0;
    }
    for (let i = 0; i < this.b.length; i++) {
      const g = this.gb[i];
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
  }
  forward(x) {
    this.x = x;
    for (let i = 0; i < this.hidden; i++) {
      let s = this.d1.b[i];
      const off = i * this.inN;
      for (let j = 0; j < this.inN; j++) s += this.d1.w[off + j] * x[j];
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
  // Backprop dE/d(out) into each layer's gradient arrays; x/h1/h2 cached.
  backward(dOut) {
    const h = this.hidden;
    const dh2 = new Float32Array(h);
    for (let o = 0; o < this.outN; o++) {
      const d = dOut[o];
      const off = o * h;
      for (let i = 0; i < h; i++) {
        dh2[i] += this.do.w[off + i] * d;
        this.do.gw[off + i] = d * this.h2[i];
      }
      this.do.gb[o] = d;
    }
    const dh1 = new Float32Array(h);
    for (let i = 0; i < h; i++) {
      dh2[i] *= 1 - this.h2[i] * this.h2[i];
      this.d2.gb[i] = dh2[i];
      const off = i * h;
      for (let j = 0; j < h; j++) {
        dh1[j] += this.d2.w[off + j] * dh2[i];
        this.d2.gw[off + j] = dh2[i] * this.h1[j];
      }
    }
    for (let i = 0; i < h; i++) {
      dh1[i] *= 1 - this.h1[i] * this.h1[i];
      this.d1.gb[i] = dh1[i];
      const off = i * this.inN;
      for (let j = 0; j < this.inN; j++) this.d1.gw[off + j] = dh1[i] * this.x[j];
    }
  }
  applyGrads(lr) {
    this.d1.applyGrads(lr);
    this.d2.applyGrads(lr);
    this.do.applyGrads(lr);
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
      obs[k++] = e && e.shootable && ENEMY_AI.has(e.ai) ? 1 : 0;
      obs[k++] = e && e.ai === 'next_level_ai' ? 1 : 0;
      let special = 0;
      if (e && (e.ai === 'tp2_ai' || e.ai === 'tpd_ai' || e.ai === 'platform_ai' || e.ai === 'sdoor_ai' || e.ai === 'strap_door_ai' || e.ai === 'switcher_ai' || e.ai === 'restart_ai')) special = 1;
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
    } else if (e.shootable && ENEMY_AI.has(e.ai)) {
      const d = Math.abs(e.x - p.x) + Math.abs(e.y - p.y);
      if (d < end) { end = d; enemy = e; }
    }
  }
  obs[k++] = exit ? (exit.x - p.x) / 100 : 0;
  obs[k++] = exit ? (exit.y - p.y) / 100 : 0;
  obs[k++] = (isFinite(dist) ? Math.min(dist, 5000) : 5000) / 50; // path distance, capped
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

const MAX_FALL = 80, MAX_JUMP = 8;

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
  return { dist: dbg.dist === 0xffffffff ? Infinity : dbg.dist, step: [cur % W, Math.floor(cur / W)], kind: entry ? entry.kind : '?' };
}

export function pathDist(g, nextNum = null) {
  const W = g.level.fgW, H = g.level.fgH, tw = g.tw, th = g.th;
  const N = W * H;
  const p = g.player;
  const pc = Math.max(0, Math.min(W - 1, Math.floor(p.x / tw)));
  const pr = Math.max(0, Math.min(H - 1, Math.floor(p.y / th)));
  const doorRects = [];
  for (const s of g.solids) {
    if (s.e && (s.e.ai === 'sdoor_ai' || s.e.ai === 'strap_door_ai') && s.e.aistate === 0) doorRects.push(s);
  }
  const inDoor = (c, r) => {
    const x = c * tw + tw / 2, y = r * th + th / 2;
    for (const s of doorRects) if (x >= s.x0 && x <= s.x1 && y >= s.y0 && y <= s.y1) return true;
    return false;
  };
  // Closed doors stay finite in the graph (so distance decreases as the agent
  // approaches them) but cost extra to route through; opening one drops the
  // distance sharply, which the PPO experiences as a progress burst.
  const doorCells = new Set();
  {
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) if (inDoor(c, r)) doorCells.add(r * W + c);
  }
  const inWallEntity = (c, r) => {
    const x = c * tw + tw / 2;
    const yTop = r * th, yBot = (r + 1) * th;
    for (const s of g.solids) {
      // Doors are handled separately via doorCells (passable at cost); their
      // rects must not block the gap drops they gate (e.g. level00's strap
      // door over the shelf chute).
      if (s.e && (s.e.ai === 'sdoor_ai' || s.e.ai === 'strap_door_ai')) continue;
      if (x >= s.x0 && x <= s.x1 && yBot > s.y0 && yTop < s.y1) return true;
    }
    return false;
  };
  const sol = (c, r) => {
    if (c < 0 || c >= W) return true;
    if (r < 0 || r >= H) return false;
    if (inWallEntity(c, r)) return true;
    return g.tileSolid(c * tw + tw / 2, r * th + th / 2);
  };
  const floorOK = (c, r) => r >= 0 && r < H && c >= 0 && c < W && sol(c, r);
  const bodyOK = (c, r) => {
    if (c < 0 || c >= W) return false;
    for (let rr = r - 1; rr >= r - 2; rr--) if (rr >= 0 && sol(c, rr)) return false;
    return true;
  };
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
  if (!goals.length) return Infinity;
  const goalSet = new Set(goals);
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
      const f = cellOf(e);
      const stops = [f, ...(e.links || []).map((l) => stopCell(l.x, l.y))];
      if (f === null) stops.unshift(stopCell(e.x, e.y));
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
  const dist = new Uint32Array(N).fill(0xffffffff);
  const prev = new Int32Array(N).fill(-1);
  const start = pr * W + pc;
  const dbg = { start, goals, expansions: 0, goalReached: [], heapPeak: 0, cells: [], first: [], prev, foundGoal: null, dist: null };
  dist[start] = 0;
  const heap = [start];
  const push = (idx, d) => {
    heap.push(idx);
    let i = heap.length - 1;
    while (i > 0) {
      const pi = (i - 1) >> 1;
      if (dist[heap[pi]] <= d) break;
      heap[i] = heap[pi];
      i = pi;
    }
    heap[i] = idx;
  };
  const pop = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length) {
      let i = 0;
      const lv = dist[last];
      for (;;) {
        let ci = i, cl = lv;
        const a = i * 2 + 1, b = a + 1;
        if (a < heap.length && dist[heap[a]] < cl) { cl = dist[heap[a]]; ci = a; }
        if (b < heap.length && dist[heap[b]] < cl) { cl = dist[heap[b]]; ci = b; }
        if (ci === i) break;
        heap[i] = heap[ci];
        i = ci;
      }
      heap[i] = last;
    }
    return top;
  };
  const relax = (from, c, r, cost, kind = '?') => {
    if (c < 0 || c >= W) return;
    const idx = r * W + c;
    if ((!floorOK(c, r) && !standCells.has(idx)) || !bodyOK(c, r)) return;
    let nd = dist[from] + cost;
    if (doorCells.has(idx)) nd += 15; // closed door: passable but expensive
    if (nd < dist[idx]) {
      dist[idx] = nd;
      prev[idx] = from;
      push(idx, nd);
      dbg.cells.push([c, r]);
      dbg.first.push({ to: [c, r], from: [from % W, Math.floor(from / W)], kind });
      if (goalSet.has(idx)) dbg.goalReached.push([idx, nd]);
    }
  };
  let found = Infinity;
  let expansions = 0;
  let foundGoal = null;
  while (heap.length && expansions++ < 200000) {
    const cur = pop();
    if (goalSet.has(cur)) { found = dist[cur]; foundGoal = cur; break; }
    const c = cur % W, r = Math.floor(cur / W);
    for (const s of [-1, 1]) {
      relax(cur, c + s, r, 1, 'walk');
      if (r > 0) relax(cur, c + s, r - 1, 1.5, 'stepup');
    }
    for (const s of [-1, 1]) {
      const nc = c + s;
      if (nc < 0 || nc >= W) continue;
      if (sol(nc, r - 1)) continue;
      let rr = r;
      for (; rr < Math.min(H, r + MAX_FALL); rr++) if (sol(nc, rr)) break;
      if (rr < H && sol(nc, rr)) {
        let open = true;
        for (let kk = r - 1; kk < rr; kk++) if (kk >= 0 && sol(nc, kk)) { open = false; break; }
        if (open) relax(cur, nc, rr, rr - r + 0.5, 'drop');
      }
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
          for (let rr = r - 6; rr <= r - 1; rr++) if (rr >= 0 && sol(cc, rr)) { clear = false; break; }
          if (!clear) break;
        }
        if (!clear) break;
        for (const dr of [-2, -1, 0, 1]) if (floorOK(nc, r + dr)) relax(cur, nc, r + dr, d + Math.abs(dr), 'jump');
      }
    }
    for (const l of g.ladders) {
      const c0 = Math.floor(l.x0 / tw), c1 = Math.floor(l.x1 / tw);
      if (c < c0 || c > c1) continue;
      const rTop = Math.floor(l.y0 / th), rBot = Math.floor(l.y1 / th);
      if (r < rTop || r > rBot) continue;
      for (let rr = rTop; rr <= rBot; rr++) {
        // Climbing is rect-based in the game (inLadder); tiles don't block it.
        if (rr === r) continue;
        const idx = rr * W + c;
        const nd = dist[cur] + Math.abs(rr - r) * 1.4;
        if (nd < dist[idx]) {
          dist[idx] = nd;
          prev[idx] = cur;
          push(idx, nd);
          dbg.cells.push([c, rr]);
          dbg.first.push({ to: [c, rr], from: [c, r], kind: 'ladder' });
        }
      }
    }
    for (const j of jumpTo.get(cur) || []) relax(cur, j.to % W, Math.floor(j.to / W), j.cost, 'ride');
  }
  dbg.expansions = expansions;
  dbg.reached = 0;
  dbg.rows = {};
  dbg.foundGoal = foundGoal;
  dbg.dist = found;
  for (let i = 0; i < N; i++) if (dist[i] !== 0xffffffff) { dbg.reached++; const rr = Math.floor(i / W); dbg.rows[rr] = (dbg.rows[rr] || 0) + 1; }
  lastPathDebug = dbg;
  return found;
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
  if (down) g.keys.add('ArrowDown');
  // Aim at the nearest visible enemy; otherwise straight ahead.
  const p = g.player;
  let target = null, best = Infinity;
  for (const e of g.entities) {
    if (e.dead || e.hidden || !e.shootable || !ENEMY_AI.has(e.ai)) continue;
    const dx = Math.abs(e.x - p.x), dy = Math.abs(e.y - p.y);
    if (dx > 400 || dy > 260) continue;
    if (!g.sees(p.x, p.y - 15, e.x, e.y - 15)) continue;
    const d = dx + dy;
    if (d < best) { best = d; target = e; }
  }
  if (target) g.mouse = { x: target.x - g.cam.x, y: target.y - 12 - g.cam.y };
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
        // refresh the navigation distance periodically (like the trainer does)
        this._dc = (this._dc || 0) + 1;
        if (this._dc % 24 === 0) {
          // target the zone that advances to the level after this one
          const lm = /level(\d+)/.exec(g.level?.name || '');
          this.lastDist = pathDist(g, lm ? +lm[1] + 1 : null);
        }
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

// Straight-line tile distance to the nearest exit; used as a fallback progress
// signal when the navigation graph can't reach any exit (e.g. sealed pockets).
// Teleporter-aware: standing near a teleporter whose destination is close to an
// exit counts as progress, so the proximity test keeps working with teleports.
function fallbackDist(g, nextNum = null) {
  const p = g.player;
  const tw = g.tw, th = g.th;
  const man = (x, y) => Math.abs(x - p.x) / tw + Math.abs(y - p.y) / th;
  const isGoal = (e) => !e.dead && e.ai === 'next_level_ai' && (nextNum === null || e.aistate === nextNum);
  let best = Infinity;
  let anyGoal = false;
  for (const e of g.entities) {
    if (!isGoal(e)) continue;
    anyGoal = true;
    best = Math.min(best, man(e.x, e.y));
  }
  for (const e of g.entities) {
    if (e.dead || (e.ai !== 'tp2_ai' && e.ai !== 'tpd_ai') || !e.links[0]) continue;
    const dest = e.links[0];
    for (const x of g.entities) {
      if (!isGoal(x)) continue;
      const viaTp = man(e.x, e.y) + 2 + (Math.abs(x.x - dest.x) + Math.abs(x.y - dest.y)) / tw;
      if (viaTp < best) best = viaTp;
    }
  }
  // Frontier term: the endpoint of each saved best run is where the real
  // compass worked, so moving toward it beats wall-hugging plateaus.
  try {
    const recs = [JSON.parse(localStorage.getItem(BEST_KEY)), ...JSON.parse(localStorage.getItem(HIST_KEY) || '[]')];
    for (const h of recs) {
      if (h && Array.isArray(h.end) && isFinite(h.dist)) {
        best = Math.min(best, man(h.end[0], h.end[1]) + h.dist);
      }
    }
  } catch { /* best-effort */ }
  // no zone matches the requested level: use any exit
  if (!anyGoal && nextNum !== null) return fallbackDist(g, null);
  return best;
}

// Objects the action key (down) activates on touch: pressing down while
// touching one is a deliberate interaction the reward should credit.
const INTERACT_AI = new Set([
  'tp2_ai', 'tpd_ai', 'platform_ai', 'switcher_ai', 'restart_ai', 'strap_door_ai', 'sdoor_ai', 'next_level_ai',
]);

// Breakable walls (original: "Shoot hidden walls to destroy them", wall() explodes at hp<=0).
const WALL_AI = new Set(['hwall_ai', 'big_wall_ai']);

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
    this.spb = 450; // constant sim steps per tick (~3x faster display; CPU-capped)
    this.lr = 3e-4;
    this.traj = { obs: [], act: [], rew: [], val: [], lp: [], done: [] };
    this.lastObs = null;
    this.lastAct = 0;
    this.lastVal = 0;
    this.lastLp = 0;
    this.lastDist = Infinity;
    this.lastDistAt = 0;
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
    // warm-start (frontier resume) state: disabled when the seed endpoint traps
    this.seedOk = true;
    this.seedStrike = 0;
    this.epSeeded = false;
    this.seedBaseDist = Infinity;
    this.seed = null;   // active chunked seed replay {acts, i, face, rec}
    this.statusAt = 0;
    this.maxRow = 0;    // deepest tile row reached this episode (descent bonus)
    this.lastFrameAt = 0;
    this.epNoProg = 0;  // sim steps since last movement or best-distance improvement
    this.epSig = null;  // 40px cell signature for the no-progress clock
    this.enemyHp = new Map(); // per-enemy hp snapshot for damage-based rewards
    this.prevPosX = null;
    this.prevPosY = null;
  }

  async start() {
    if (this.running) return;
    this.running = true;
    const g = this.g;
    this.prevNext = g.nextLevel;
    g.autoPause = true;
    g.demo = true;           // isolate real input while training
    g.speed = 1;
    g.bot = this.bot;
    g.demoTimeoutTicks = 0;
    // next_level zones carry their destination level in aistate (original
    // people.lsp): only the zone loading THIS level's successor is a win.
    g.nextLevel = (dest) => {
      if (dest === this.levelIdx + 1) this.endEpisode(30, true);
      else this.endEpisode(5, true); // branch exit: level ends, not the win
    };
    g.onDemoStop = () => this.stop();
    console.log(`[ppo] training started (${this.loaded ? 'resumed' : 'fresh'})`);
    await this.reset();
    // Sped-up display: render at ~4fps while training instead of 60fps, so
    // each shown frame advances ~40+ sim-seconds (the render itself is only
    // 0.11ms; this is purely to make the visible run look like a time-lapse).
    this.g.renderThrottle = 250;
    // 16ms ticks: 450 sim steps always run first (constant speed), then any
    // leftover budget goes to gradient slices. setInterval also survives
    // background-tab throttling far better than rAF.
    this.timer = setInterval(() => this.frame(), 16);
  }

  stop() {
    if (this.timer) { clearInterval(this.timer); this.timer = null; }
    if (!this.running && !this.paused) return;
    this.running = false;
    this.paused = false;
    this.updating = null; // drop any partially-applied gradient rollout
    this.seed = null;     // drop any in-progress warm-start replay
    this.g.renderThrottle = 0;
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
    console.log(`[ppo] stopped — updates ${this.updates}, rollouts ${this.rollouts}, wins ${this.successes}/${this.episodes}`);
    this.onStatus(`stopped — ${this.updates} updates, ${this.successes} wins`);
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

  async reset() {
    this.resetting = true;
    this.episodeSteps = 0;
    this.noProgT = 0;
    this.prevDist = Infinity;
    this.lastDist = Infinity;
    this.lastDistAt = 0;
    this.lastObs = null;
    this.recActs = [];
    this.recBestDist = Infinity;
    this.epSeeded = false;
    this.seedBaseDist = Infinity;
    try { await this.g.start(this.levels[this.levelIdx]); } catch { /* retry next time */ }
    // Warm-start most episodes from where the best run left off, so training
    // resumes from the frontier instead of always from spawn. If seeded
    // episodes stop improving (the endpoint is a trap), fall back to spawn
    // starts until exploration finds a new best.
    const rec = loadBestRun();
    this.seed = null;
    if (this.seedOk && rec && rec.level === this.levels[this.levelIdx] && rec.acts.length > 8
        && Math.random() < (this.seedStrike > 0 ? 0.35 : 0.75)) {
      // Chunked replay: seedStep() in frame() runs it across ticks so the UI
      // stays responsive and the status line can show progress. The best-run
      // history is replayed in chronological order (worst first) because each
      // script was recorded from the previous one's endpoint — chaining them
      // reconstructs the frontier faithfully instead of desyncing from spawn.
      const hist = loadBestHistory()
        .filter((h) => h.level === rec.level && !(h.dist === rec.dist && h.acts.length === rec.acts.length))
        .sort((a, b) => b.dist - a.dist); // worst first
      this.seed = { chain: [...hist, rec], ci: 0, i: 0, face: 1, rec };
    }
    if (!this.seed) {
      this.lastX = this.g.player.x;
      this.lastEnemyCount = this.g.entities.filter((e) => !e.dead && e.shootable && ENEMY_AI.has(e.ai)).length;
    }
    this.maxRow = Math.floor(this.g.player.y / this.g.th);
    this.epNoProg = 0;
    this.epSig = null;
    this.enemyHp = new Map();
    this.prevPosX = null;
    this.prevPosY = null;
    this.lastWallCount = this.g.entities.filter((e) => !e.dead && WALL_AI.has(e.ai)).length;
    this.resetting = false;
  }

  // Replays the best-run chain toward the frontier, a bounded chunk per tick.
  seedStep(budget) {
    const g = this.g;
    const s = this.seed;
    g.bot = null;                 // no policy decisions while seeding
    g.nextLevel = () => {};       // exiting during the seed is not a win
    let n = budget;
    while (n-- > 0 && !g.player.dead && s.ci < s.chain.length) {
      const seg = s.chain[s.ci];
      const a = seg.acts[Math.min(Math.floor(s.i / 4), seg.acts.length - 1)];
      const { move } = actParts(a);
      if (move !== 0) s.face = move;
      applyAction(g, a, s.face);
      g.update(1 / 60);
      s.i++;
      if (s.i >= seg.acts.length * 4) { s.i = 0; s.ci++; }
    }
    if (s.ci >= s.chain.length || g.player.dead) {
      const rec = s.rec;
      g.bot = this.bot;
      g.nextLevel = (dest) => {
        if (dest === this.levelIdx + 1) this.endEpisode(30, true);
        else this.endEpisode(5, true);
      };
      const onExit = g.entities.some((e) => !e.dead && e.ai === 'next_level_ai'
        && Math.abs(e.x - g.player.x) < 60 && Math.abs(e.y - g.player.y) < 60);
      if (onExit && !g.player.dead) g.respawn();
      this.epSeeded = true;
      this.seedBaseDist = rec.dist;
      this.lastX = g.player.x;
      this.lastEnemyCount = g.entities.filter((e) => !e.dead && e.shootable && ENEMY_AI.has(e.ai)).length;
      this.maxRow = Math.floor(g.player.y / g.th); // depth reached by the seed is baseline
      this.seed = null;
      console.log(`[ppo] seeded episode from best run end (${Math.round(g.player.x)}, ${Math.round(g.player.y)})`);
    }
  }

  endEpisode(termReward, done) {
    if (this.resetting || this.lastObs === null) return;
    this.traj.obs.push(this.lastObs);
    this.traj.act.push(this.lastAct);
    this.traj.rew.push(termReward);
    this.traj.val.push(this.lastVal);
    this.traj.lp.push(this.lastLp);
    this.traj.done.push(done ? 1 : 0);
    this.lastObs = null;
    this.episodes++;
    if (termReward >= 20) {
      this.successes++;
      console.log(`[ppo] win on ${this.levels[this.levelIdx]} (${this.successes} total, rollout ${this.rollouts})`);
      this.onStatus(`win! ${this.successes} (update ${this.updates})`);
    }
    const improved = this.maybeSaveBest();
    if (improved) { this.seedOk = true; this.seedStrike = 0; }
    else if (this.epSeeded) {
      // Seeded from the best endpoint but made no headway: likely a trap.
      this.seedStrike++;
      if (this.seedStrike >= 3) {
        this.seedOk = false;
        console.log('[ppo] seeded endpoint not improving — exploring from spawn');
        this.onStatus('seed stuck — exploring from spawn');
      }
    }
    if (this.traj.obs.length >= 1024) this.update();
    this.reset();
  }

  // Persist this episode's action script if it got closer to the exit than
  // any previous episode, so it can be replayed from the demo UI.
  // Returns true when a new best run was saved.
  maybeSaveBest() {
    if (this.recActs.length < 8 || !isFinite(this.recBestDist)) return false;
    const prev = loadBestRun();
    if (prev && prev.dist <= this.recBestDist) return false;
    const rec = {
      level: this.levels[this.levelIdx],
      dist: Math.round(this.recBestDist * 10) / 10,
      acts: this.recActs,
      end: [Math.round(this.g.player.x), Math.round(this.g.player.y)], // frontier for the fallback compass
      episodes: this.episodes,
      updates: this.updates,
      t: Date.now(),
    };
    try { localStorage.setItem(BEST_KEY, JSON.stringify(rec)); } catch { return false; }
    // Keep a small history of the top runs so progress is watchable.
    try {
      const hist = loadBestHistory();
      const dup = hist.findIndex((h) => h.level === rec.level && h.dist === rec.dist);
      if (dup === -1) {
        hist.push(rec);
        hist.sort((a, b) => a.dist - b.dist);
        if (hist.length > 5) hist.length = 5;
        localStorage.setItem(HIST_KEY, JSON.stringify(hist));
      }
    } catch { /* history is best-effort */ }
    this.net.save(); // checkpoint weights with the best run: the demo uses this model
    console.log(`[ppo] best run saved: ${rec.level}, dist ${rec.dist}, ${rec.acts.length} actions (episode ${this.episodes})`);
    this.onStatus(`best run saved — dist ${rec.dist} on ${rec.level}`);
    return true;
  }

  frame() {
    if (!this.running) return;
    const now = performance.now();
    const gap = this.lastFrameAt ? now - this.lastFrameAt : 0;
    this.lastFrameAt = now;
    try {
      if (!this.resetting) {
        if (this.seed) {
          this.seedStep(gap > 500 ? 3000 : 1200); // replay the warm-start chunked
        } else {
          for (let i = 0; i < this.spb; i++) this.step();
          if (this.updating) {
            // Fill the rest of the tick with gradient work: ~70ms when visible,
            // up to ~950ms when the tab is throttled to 1Hz, so background tabs
            // still chew through full rollouts instead of idling.
            const maxTick = gap > 500 ? 950 : 70;
            const samples = Math.max(24, Math.min(360, Math.floor((maxTick - (performance.now() - now)) / 0.5)));
            this.stepUpdateSlice(samples);
          }
        }
      }
      // live status twice a second so the run counters are always visible
      if (now - (this.statusAt || 0) > 500) {
        this.statusAt = now;
        if (this.seed) {
          const done = this.seed.chain.slice(0, this.seed.ci).reduce((a, s) => a + s.acts.length, 0)
            + Math.floor(this.seed.i / 4);
          const total = this.seed.chain.reduce((a, s) => a + s.acts.length, 0);
          this.onStatus(`seeding from best run… (${done}/${total} actions)`);
        }
        else this.onStatus(`ep ${this.episodes} · wins ${this.successes}/${this.episodes} · upd ${this.updates} · t ${Math.floor(this.episodeSteps / 60)}s`);
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
    if (!p || !g.level) return;
    if (p.dead) {
      if (this.lastObs === null) this.reset();          // died before first decision
      else this.endEpisode(-10, true);                  // death ends the episode
      return;
    }
    this.episodeSteps++;
    if (this.episodeSteps % 4 === 1) this.policyStep();
    // Progress-aware episode length: only cut after a long stall, never while
    // the agent is still moving or improving, so slow-but-steady runs that
    // actually reach the exit are never truncated.
    this.epNoProg++;
    const sig = Math.floor(p.x / 40) * 1000 + Math.floor(p.y / 40);
    if (sig !== this.epSig) { this.epSig = sig; this.epNoProg = 0; }
    if (this.epNoProg > 15 * 45 * 4) { this.endEpisode(-3, true); return; } // 45 sim-seconds stalled
    if (this.episodeSteps > 15 * 600 * 4) { this.endEpisode(-3, true); return; } // 600 sim-seconds absolute cap
    g.update(1 / 60);
  }

  policyStep() {
    const g = this.g;
    const p = g.player;
    // Refresh the navigation distance on a wall-clock budget (at most ~4x/sec)
    // instead of every N sim steps, so fast sim doesn't drown in pathfinding.
    const now = performance.now();
    if (!this.lastDistAt || now - this.lastDistAt > 250) {
      this.lastDist = pathDist(g, this.levelIdx + 1);
      this.lastDistAt = now;
    }
    const dist = this.lastDist;
    // If the graph can't see the next-level exit yet, fall back to a
    // teleport-aware straight-line metric so the reward gradient and the
    // best-run metric stay finite everywhere.
    const fallback = !isFinite(dist);
    const effDist = fallback ? 1e6 + fallbackDist(g, this.levelIdx + 1) : dist;

    // reward for the transition that just completed
    let r = -0.02; // step cost
    if (this.prevDist !== Infinity && isFinite(this.prevDist)) {
      const gain = this.prevDist - effDist;
      if (gain > 0) r += Math.min(12, gain * (fallback ? 1.5 : 8)); // BFS progress
      else if (gain < 0) r -= Math.min(3, -gain * (fallback ? 1 : 8)); // moved away
    }
    this.prevDist = effDist;
    const enemyCount = g.entities.filter((e) => !e.dead && e.shootable && ENEMY_AI.has(e.ai)).length;
    if (enemyCount < this.lastEnemyCount) r += 1.5 * (this.lastEnemyCount - enemyCount);
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
    if (dmg > 0) r += Math.min(0.8, dmg * 0.04);
    if (wallDmg > 0) r += Math.min(1.2, wallDmg * 0.06);
    const wallCount = g.entities.filter((e) => !e.dead && WALL_AI.has(e.ai)).length;
    if (wallCount < this.lastWallCount) r += 3 * (this.lastWallCount - wallCount);
    this.lastWallCount = wallCount;
    // Teleportation: an instantaneous position jump means a teleport actually
    // happened — credit the event strongly so using teleporters is clearly good.
    if (this.prevPosX !== null) {
      const jump = Math.abs(p.x - this.prevPosX) + Math.abs(p.y - this.prevPosY);
      if (jump > 100) r += 5;
    }
    this.prevPosX = p.x;
    this.prevPosY = p.y;
    // Interaction credit: the previous decision pressed down while touching a
    // teleporter/platform/switch — reward the action itself, not just the
    // state. Teleporters pay more: their graph distance is already tiny while
    // standing on them, so the press itself must carry the signal.
    {
      const { down } = actParts(this.lastAct);
      if (down) {
        for (const e of g.entities) {
          if (e.dead || !INTERACT_AI.has(e.ai)) continue;
          if (g.touchesPlayer(e)) {
            r += (e.ai === 'tp2_ai' || e.ai === 'tpd_ai') ? 2 : 1;
            break;
          }
        }
      }
    }
    // Descending into new depth is progress in this underground level: each
    // newly-reached deepest tile row pays a small one-time bonus, so dropping
    // down the chute once the way is open is clearly rewarded.
    const row = Math.floor(p.y / g.th);
    if (row > this.maxRow) {
      r += Math.min(0.5, (row - this.maxRow) * 0.05);
      this.maxRow = row;
    }
    if (Math.abs(p.x - this.lastX) > 2) { this.noProgT = 0; this.lastX = p.x; }
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
      if (this.recBestDist < before) this.epNoProg = 0; // improving: keep the run alive
    }

    if (this.traj.obs.length >= 1024 && !this.updating) this.update();
  }

  update() {
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
      const nextV = t === n - 1 ? 0 : R.val[t + 1];
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
      epoch: 0, pos: 0, epochs: 1, shuffled: false,
    };
    this.traj = { obs: [], act: [], rew: [], val: [], lp: [], done: [] };
    this.lastObs = null;
  }

  // Processes up to `budget` transition samples; returns when the slice budget
  // is spent or the whole rollout is done. Called once per frame while updating.
  stepUpdateSlice(budget = 128) {
    const u = this.updating;
    if (!u) return;
    const clipEps = 0.2, entCoef = 0.01;
    let left = budget;
    while (left-- > 0) {
      if (u.pos === 0) {
        if (!u.shuffled) {
          for (let i = u.order.length - 1; i > 0; i--) {
            const j = (Math.random() * (i + 1)) | 0;
            [u.order[i], u.order[j]] = [u.order[j], u.order[i]];
          }
          u.shuffled = true;
        }
      }
      const t = u.order[u.pos++];
      const x = u.R.obs[t];
      const a = u.R.act[t];
      const { logits, v } = this.net.forward(x);
      const pr = softmax(logits);
      const lp = Math.log(pr[a] + 1e-9);
      const ratio = Math.exp(lp - u.R.lp[t]);
      const clipped = ratio < 1 - clipEps || ratio > 1 + clipEps;
      const gLp = clipped ? 0 : -ratio * u.adv[t];
      let H = 0;
      for (const q of pr) H -= q * Math.log(q + 1e-9);
      for (let i = 0; i < ACTS; i++) {
        _dOut[i] = gLp * ((i === a ? 1 : 0) - pr[i]) + entCoef * pr[i] * (Math.log(pr[i] + 1e-9) + H);
      }
      this.net.actor.backward(_dOut);
      this.net.actor.applyGrads(this.lr);
      _critD[0] = v - u.ret[t];
      this.net.critic.backward(_critD);
      this.net.critic.applyGrads(this.lr);
      if (u.pos >= u.n) {
        u.pos = 0;
        u.epoch++;
        u.shuffled = false;
        if (u.epoch >= u.epochs) {
          const done = u.R;
          this.updating = null;
          this.updates++;
          this.rollouts++;
          if (this.maybeSaveBest()) { this.seedOk = true; this.seedStrike = 0; } // mid-episode progress counts too
          if (this.rollouts % 10 === 0) this.net.save();
          const avg = done.rew.reduce((a, b) => a + b, 0) / u.n;
          this.onStatus(`update ${this.updates} · wins ${this.successes}/${this.episodes} · avgR ${avg.toFixed(2)}`);
          return;
        }
      }
    }
  }
}
