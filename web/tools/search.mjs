// Finds a way through each level by search, not by learning.
//
// The game is deterministic and any moment of it can be saved and restored, so
// there is no need to learn a skill and hope it carries through a level: the
// search keeps every distinct situation it has reached ("cells": where the
// player is, and which switches, kills and walls are done), goes back to one
// of them, tries a short burst of actions from there, and keeps whatever new
// situations that turns up. It never replays the easy part of a level and
// never loses ground. (The idea is Go-Explore's.)
//
// What it finds is a recording in the form "Full game demo" already plays: a
// chain of pieces, each a saved state and the actions taken from it.
//
//   node web/tools/search.mjs                  carry on from where it stopped
//   node web/tools/search.mjs --fresh          start again at level 0
//   node web/tools/search.mjs --per-level 20   minutes to spend on a level before giving up (default 30)
//   node web/tools/search.mjs --levels 3       stop after passing this many more levels
//   node web/tools/search.mjs --seed 7         a different run of the dice
//   node web/tools/search.mjs --in-order       the whole game from level 0, weapons carried from level to level
//   node web/tools/search.mjs --parallel       every remaining level at once, each in its own process
//                                              (--jobs 6 at a time, --upto 21 the last level to try)
//
// Writes train-out/search.json (progress; resumed from) and
// web/data/ppo-demo.json (what "Full game demo" plays).
import { mkdir, readFile, writeFile, copyFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { deflateRawSync, inflateRawSync } from 'node:zlib';
import { makeGame, LEVELS } from './headless.mjs';

const arg = (name, dflt) => { const i = process.argv.indexOf(`--${name}`); return i >= 0 ? process.argv[i + 1] : dflt; };
const PER_LEVEL = +arg('per-level', 30) * 60e3;
const MAX_LEVELS = +arg('levels', 99);
const OUT = fileURLToPath(new URL('../../train-out/', import.meta.url));
const STATE_FILE = `${OUT}search.json`;
const DEMO_FILE = fileURLToPath(new URL('../data/ppo-demo.json', import.meta.url));

const g = await makeGame(LEVELS[0]);
const ppo = await import('../src/ppo.js');
g.audio.muted = true;

// ---- dice ----
let seed = (+arg('seed', 1)) >>> 0 || 1;
const rnd = () => { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };

// ---- the search's own settings ----
const BURST = 90;      // decisions tried from a cell (a decision is 4 game steps, 1/15 s): 6 seconds of play
const MAX_CELLS = +arg('max-cells', 30000); // cells a search keeps at most (see "Memory:" in searchLevel)
const LOOK = 3;        // decisions between looks at where the player has got to
const GRID_X = 36, GRID_Y = 30; // size of a cell, in pixels

// ---- saved states, stored as what differs from the level's start ----
const FIELDS = ['count', 'x', 'y', 'vx', 'vy', 'dir', 'state', 'frame', 'stateTime', 'aistate', 'aitype', 'hp', 'xvel', 'yvel', 'xacel', 'yacel', 'fade', 'shootable', 'hidden'];
function sameEnt(a, b) {
  for (const k of FIELDS) if (a[k] !== b[k]) return false;
  if (a.links.length !== b.links.length) return false;
  for (let i = 0; i < a.links.length; i++) if (a.links[i] !== b.links[i]) return false;
  return true;
}
// (and kept compressed, outside the JavaScript heap: a busy level has tens of
// thousands of cells, each with every object that has so much as changed its
// animation frame, which is more than the heap will hold as plain objects)
function pack(snap, base) {
  const es = {}, gone = [];
  for (const id in snap.es) if (!base.es[id] || !sameEnt(snap.es[id], base.es[id])) es[id] = snap.es[id];
  for (const id in base.es) if (!snap.es[id]) gone.push(id);
  return deflateRawSync(JSON.stringify({ ...snap, es, gone }), { level: 1 });
}
function unpack(buf, base) {
  const packed = JSON.parse(inflateRawSync(buf).toString());
  const es = { ...base.es };
  for (const id of packed.gone) delete es[id];
  Object.assign(es, packed.es);
  const { gone, ...rest } = packed; // eslint-disable-line no-unused-vars
  return { ...rest, es };
}

// ---- actions ----
// 24 of them: move (left, none, right) x jump x action key x fire, as the PPO agent's.
const ACT = (move, jump, down, fire) => (move + 1) + 3 * jump + 6 * down + 12 * fire;
const ENEMY_NEAR = 260;
let climb = 0; // set by randomAct: how long to hold a climb, in decisions
function enemyNear() {
  const p = g.player;
  for (const e of g.entities) {
    if (e.dead) continue;
    if (e.type === 'SWITCH_BALL' && e.aistate === 0 && Math.hypot(e.x - p.x, e.y - p.y) < ((p.ammo.ROCKET || 0) > 0 ? 420 : 180)) return true; // something to shoot, too
    if (e.shootable && !e.hidden && e.hp > 0 && Math.abs(e.x - p.x) < ENEMY_NEAR && Math.abs(e.y - p.y) < 180 && e.def.name !== 'DARNEL' && /ant|fly|gun|jug|bomb|rob|who|boss/i.test(e.ai || '')) return true;
  }
  return false;
}
// A random action, leaning the way a player does: mostly on the move, the
// action key now and then (it is what works switches, lifts, teleporters and
// exits), firing when there is something to fire at.
function randomAct(dirBias) {
  // On a ladder (or standing on the top of one) the way on is a long, steady
  // climb: straight down or straight up, held for seconds. Random play almost
  // never holds one key that long, so here it is offered as a move of its own.
  const p = g.player;
  if (g.ladders.some((l) => p.x >= l.x0 - 6 && p.x <= l.x1 + 6 && p.y >= l.y0 - 12 && p.y <= l.y1 + 4) && rnd() < 0.5) {
    climb = 6 + Math.floor(rnd() * 30);
    return rnd() < 0.5 ? ACT(0, 0, 1, 0) : ACT(0, 1, 0, 0);
  }
  // Riding a lift that is on the move: stand still and let it carry. Some
  // rides are long (level 8's run for 27 seconds), and random play walks off
  // the edge of a 60-pixel deck long before that.
  for (const e of g.entities) {
    if (e.ai !== 'platform_ai' || e.dead || e.aistate === 0) continue;
    const deck = g.deckRect(e);
    if (deck && p.x >= deck.x0 - 4 && p.x <= deck.x1 + 4 && Math.abs(p.y - deck.y0) < 10 && rnd() < 0.9) {
      climb = 10 + Math.floor(rnd() * 40);
      return ACT(0, 0, 0, enemyNear() && rnd() < 0.7 ? 1 : 0);
    }
  }
  const r = rnd();
  const move = r < 0.12 ? 0 : (rnd() < 0.5 + dirBias * 0.25 ? 1 : -1);
  const jump = rnd() < 0.3 ? 1 : 0;
  const down = rnd() < 0.18 ? 1 : 0;
  const fire = rnd() < (enemyNear() ? 0.7 : 0.12) ? 1 : 0;
  return ACT(move, jump, down, fire);
}

// ---- one level ----
let exitDest = null;
g.nextLevel = (n) => { exitDest = n; g.transitioning = true; };
g.onEndGame = () => { exitDest = LEVELS.length; }; // the last level's "exit" is the ending

// --exit <id>: only that exit counts (and the compass steers for it alone).
// Some levels have it fixed: level 0 has three exits to level 1, and the
// tutorial ends at the far one, at the bottom right.
const EXIT_FOR = { 0: 74 };
const EXIT_ID = arg('exit', null) !== null ? +arg('exit') : arg('only', null) !== null ? (EXIT_FOR[+arg('only')] ?? null) : null;
if (EXIT_ID !== null) ppo.exitChoice.ids = new Set([EXIT_ID]);
// levels whose exit only the jetpack reaches: level 16's, at the top of a shaft above its teleporter
const FLY_FOR = new Set([16]);
if (arg('only', null) !== null && FLY_FOR.has(+arg('only'))) ppo.exitChoice.needsFlight = true;
const atChosenExit = () => {
  if (EXIT_ID === null) return true;
  const e = g.entities.find((o) => o.id === EXIT_ID), p = g.player;
  return !!e && Math.abs(p.x - e.x) < 40 && Math.abs(p.y - e.y) < 60;
};
const wallsLeft = () => { let n = 0; for (const e of g.entities) if (!e.dead && (e.ai === 'hwall_ai' || e.ai === 'big_wall_ai' || e.ai === 'block_ai')) n++; return n; };
function cellKey() {
  const p = g.player;
  // Switches (and switch balls) are named one by one: which are on matters.
  // Everything else still to do (creatures to kill, sensors to trip) is
  // counted, not listed. Level 16 has groups of creatures that can be killed
  // in any order (the eight flyers by trap door 515); listed, every subset
  // killed was a state of its own, over a thousand of them, and the search's
  // time went in spreading itself across them.
  let keys = '', others = 0;
  for (const e of ppo.keysToGo(g)) { if (e.type === 'SWITCH_BALL' || /^switch/.test(e.ai || '')) keys += `${e.id}.`; else others++; }
  keys += `+${others}`;
  // A delay gate that is counting is progress too (a door that opens after
  // the player has stood by it for some seconds): how far each has got, in
  // steps of 20 ticks, so that waiting is not thrown away as "nothing new".
  let timers = '';
  for (const e of g.entities) if (e.a.count > 0 && e.ai === 'delay_ai') timers += `${e.id}:${Math.floor(e.a.count / 20)}.`;
  return `${Math.floor(p.x / GRID_X)},${Math.floor(p.y / GRID_Y)}|${keys}|${wallsLeft()}|${p.power || ''}|${[...p.owned].length}|${timers}`;
}
// Is arriving like this better than how the cell was reached before? Healthier, or as healthy and sooner.
const better = (hp, steps, old) => hp >= old.hp + 10 || (hp >= old.hp - 4 && steps < old.steps * 0.9);

async function searchLevel(lv, entry, deadline) {
  const next = lv + 1;
  await ppo.startRun(g, LEVELS[lv], entry);
  const level0 = ppo.snapshotState(g); // what saved states are stored against
  const cells = new Map();
  const list = [];
  // (The start itself is given no compass reading. Before the first tick the
  // level's force fields have not put up their beams, so the compass sees a
  // clear run to the exit that is not there, and nothing found afterwards
  // could ever look better than that.)
  const root = { key: cellKey(), snap: null, full: entry, parent: null, acts: '', pieceSteps: 0, steps: 0, hp: g.player.hp, dist: Infinity, chosen: 0, born: 0 };
  // the states of the level the cells fall into (everything in a cell's key but the place and the timers)
  const stateOf = (key) => key.split('|').slice(1, 5).join('|');
  const states = new Map();
  const noteState = (c) => {
    c.state = stateOf(c.key);
    const st = states.get(c.state) || states.set(c.state, { chosen: 0, best: Infinity }).get(c.state);
    if (c.dist < st.best) st.best = c.dist;
  };
  noteState(root);
  cells.set(root.key, root); list.push(root);
  let iter = 0, simSteps = 0, bestDist = root.dist, bestCell = root, lastPrint = Date.now(), deaths = 0, evicted = 0, lastBestDump = 0;
  const t0 = Date.now();

  const seedOf = (cell) => (cell.snap ? unpack(cell.snap, level0) : cell.full);

  // The recording that reaches a cell: the pieces down from the level's start.
  function chain(cell, lastActs = null, lastSteps = 0, from = null) {
    const pieces = [];
    if (lastActs !== null) pieces.push({ seed: seedOf(from), acts: lastActs, steps: lastSteps });
    for (let c = lastActs !== null ? from : cell; c && c.parent; c = c.parent) pieces.push({ seed: seedOf(c.parent), acts: c.acts, steps: c.pieceSteps });
    return pieces.reverse();
  }

  while (Date.now() < deadline) {
    iter++;
    // pick a cell: ones seldom tried, new ones, and ones furthest along the compass.
    // Half the time the pick is made within one state of the level (which
    // switches are on, what is dead), chosen evenly across the states: the
    // state with the fewest things left to do is not always the way on (a
    // toggle that shuts the route, a room that seals behind the player), and
    // left to the compass alone the search pours its time into it.
    let only = null, stateBest = bestDist;
    if (states.size > 1 && rnd() < 0.5) {
      let tot = 0;
      for (const st of states.values()) tot += 1 / Math.sqrt(1 + st.chosen);
      let pk = rnd() * tot;
      for (const [k, st] of states) { only = k; stateBest = st.best; if ((pk -= 1 / Math.sqrt(1 + st.chosen)) <= 0) break; }
      states.get(only).chosen++;
    }
    let total = 0;
    const w = new Float64Array(list.length);
    for (let i = 0; i < list.length; i++) {
      const c = list[i];
      if (only !== null && c.state !== only) continue;
      const gap = c.dist - stateBest;
      const lead = !isFinite(c.dist) ? 0.5 : gap <= 8 ? 10 : gap <= 60 ? 4 : gap <= 400 ? 1.5 : 1;
      const fresh = 1 + 3 * Math.exp(-(iter - c.born) / 300);
      w[i] = (lead * fresh) / Math.sqrt(1 + c.chosen);
      total += w[i];
    }
    let pick = rnd() * total, ci = 0;
    while (ci < list.length - 1 && (pick -= w[ci]) > 0) ci++;
    const cell = list[ci];
    cell.chosen++;

    await ppo.startRun(g, LEVELS[lv], seedOf(cell));
    exitDest = null;
    const acts = [];
    let face = 1, act = 0, hold = 0, cruise = null;
    const dirBias = rnd() < 0.5 ? (rnd() < 0.5 ? 1 : -1) : 0; // some bursts keep heading one way
    const burst = rnd() < 0.25 ? BURST * 3 : BURST; // now and then a long one: some things take patience
    for (let d = 0; d < burst; d++) {
      // With the jetpack, level flight is a move of its own: head one way and
      // thrust whenever below a chosen height. Random thrusting only ever
      // bobs up and down; getting across something at a fixed height (level
      // 5's arrow strips, which push along the floor and along the ceiling
      // and leave a gap between) needs the height held for a second or two.
      if (cruise && cruise.left-- > 0 && g.player.power === 'FLY') {
        const pl = g.player, t = cruise.to;
        if (t) {
          // flying to a switch: hold its height, steer over it, press the action key on alternate decisions
          const dx = t.x - pl.x, mv = Math.abs(dx) < 6 ? 0 : Math.sign(dx);
          const there = Math.abs(dx) < 16 && Math.abs(pl.y - t.y) < 26;
          act = ACT(mv, pl.y > t.y + 4 || pl.vy > 30 ? 1 : 0, there && cruise.left % 2 === 0 ? 1 : 0, enemyNear() && rnd() < 0.5 ? 1 : 0);
          if (t.aistate !== 0) cruise = null;
        } else act = ACT(cruise.dir, pl.y > cruise.y || pl.vy > 30 ? 1 : 0, 0, enemyNear() && rnd() < 0.5 ? 1 : 0);
      } else {
        cruise = null;
        if (hold-- <= 0) {
          climb = 0; act = randomAct(dirBias); hold = climb || Math.floor(rnd() * rnd() * 14);
          if (g.player.power === 'FLY' && rnd() < 0.3) {
            // A switch still to press that hangs out of reach overhead is
            // flown to (level 16's four, up the wall of a dead-end room, each
            // switching off one of the force fields across the way on); other
            // times the jetpack holds a chosen height while heading one way.
            const pl = g.player;
            const up = rnd() < 0.5 ? ppo.keysToGo(g).filter((e) => (e.ai === 'switcher_ai' || e.ai === 'switch_once_ai') && e.aistate === 0
              && Math.abs(e.x - pl.x) < 300 && e.y < pl.y - 35 && pl.y - e.y < 300) : [];
            if (up.length) cruise = { left: 30 + Math.floor(rnd() * 30), to: up[Math.floor(rnd() * up.length)] };
            else cruise = { left: 8 + Math.floor(rnd() * 40), dir: dirBias || (rnd() < 0.5 ? 1 : -1), y: g.player.y + 6 - rnd() * 45 };
          }
        }
      }
      acts.push(act);
      const mv = (act % 3) - 1;
      if (mv !== 0) face = mv;
      let s = 0;
      for (; s < 4 && exitDest === null && !g.player.dead; s++) { ppo.applyAction(g, act, face); g.update(1 / 60); simSteps++; }
      if (exitDest !== null) {
        if (exitDest === next && atChosenExit()) return { solved: true, pieces: chain(null, ppo.encodeActs(acts), (acts.length - 1) * 4 + s, cell), iter, simSteps, cells: list.length, ms: Date.now() - t0 };
        break; // an exit to somewhere else: not a way on
      }
      if (g.player.dead) { deaths++; break; }
      // Flown out over the top of the level (a start shaft open to the sky
      // and a jetpack will do it): nothing up there, and the compass, which
      // reads straight down through the ceiling, takes it for progress.
      if (g.player.y < -30) break;
      if ((d + 1) % LOOK === 0) {
        const p = g.player;
        // only where the player could be put down again: standing, climbing or flying
        if (!(p.ground || p.climbing || p.power === 'FLY')) continue;
        const key = cellKey();
        const steps = cell.steps + acts.length * 4;
        const old = cells.get(key);
        if (old && (old === root || !better(p.hp, steps, old))) continue;
        const dist = ppo.exitDist(g, next);
        const rec = { key, snap: pack(ppo.snapshotState(g), level0), parent: cell, acts: ppo.encodeActs(acts), pieceSteps: acts.length * 4, steps, hp: p.hp, dist: isFinite(dist) ? dist : (old ? old.dist : Infinity), chosen: old ? Math.floor(old.chosen / 2) : 0, born: iter, x: Math.round(p.x), y: Math.round(p.y) };
        cells.set(key, rec);
        noteState(rec);
        if (old) list[list.indexOf(old)] = rec; else list.push(rec);
        if (rec.dist < bestDist) {
          bestDist = rec.dist; bestCell = rec;
          // the furthest-on situation so far, as a saved state, for looking at by hand
          // (load it with ppo.startRun(g, LEVELS[lv], seed))
          if (Date.now() - lastBestDump > 30000) { lastBestDump = Date.now(); writeFile(`${OUT}best-level${String(lv).padStart(2, '0')}-${process.pid}.json`, JSON.stringify({ x: rec.x, y: rec.y, dist: ppo.fmtDist(rec.dist), seed: seedOf(rec) })).catch(() => {}); }
        }
      }
    }
    // Memory: every cell keeps a saved state, and a long search on a big
    // level (16) gathered 300,000-440,000 of them a process, which with
    // several processes side by side filled the machine's memory. Past
    // MAX_CELLS the least promising quarter is let go: furthest behind the
    // best of their state, most often tried already, oldest. (A cell let go
    // that is the parent of one kept stays in memory through it, as its
    // route back to the start needs it; the rest are freed.)
    if (list.length > MAX_CELLS) {
      const keep = new Set([root, bestCell]);
      const bestOf = new Map();
      for (const c of list) { const b = bestOf.get(c.state); if (!b || c.dist < b.dist) bestOf.set(c.state, c); }
      for (const c of bestOf.values()) keep.add(c);
      const worth = (c) => {
        const gap = c.dist - (states.get(c.state)?.best ?? c.dist);
        const lead = !isFinite(c.dist) ? 0.5 : gap <= 8 ? 10 : gap <= 60 ? 4 : gap <= 400 ? 1.5 : 1;
        return (lead * (1 + 3 * Math.exp(-(iter - c.born) / 300))) / Math.sqrt(1 + c.chosen);
      };
      const ranked = list.filter((c) => !keep.has(c)).sort((a, b) => worth(a) - worth(b));
      const drop = new Set(ranked.slice(0, Math.floor(list.length / 4)));
      for (const c of drop) if (cells.get(c.key) === c) cells.delete(c.key);
      list.splice(0, list.length, ...list.filter((c) => !drop.has(c)));
      const live = new Set(list.map((c) => c.state));
      for (const k of [...states.keys()]) if (!live.has(k)) states.delete(k);
      evicted += drop.size;
    }
    if (Date.now() - lastPrint > 15000) {
      lastPrint = Date.now();
      console.log(`  level ${String(lv).padStart(2, '0')}  ${((Date.now() - t0) / 60000).toFixed(1)} min  ${iter} bursts  ${list.length} cells${evicted ? ` (${evicted} let go)` : ''}  ${(process.memoryUsage().rss / 1e9).toFixed(2)} GB  ${Math.round(simSteps / ((Date.now() - t0) / 1000))} steps/s  nearest the exit: ${ppo.fmtDist(bestDist)} at (${bestCell.x ?? '-'},${bestCell.y ?? '-'})  deaths ${deaths}`);
    }
  }
  // out of time: a map of everywhere it stood in the furthest-on state of the
  // level, with what is still to be done marked, for working out what stops it
  {
    const state = (c) => c.state;
    const here = list.filter((c) => state(c) === state(bestCell));
    await ppo.startRun(g, LEVELS[lv], seedOf(bestCell));
    const W = g.level.fgW, H = g.level.fgH, rows = [];
    const mark = new Map();
    for (const c of here) if (c.x !== undefined) mark.set(`${Math.floor(c.x / g.tw)},${Math.floor((c.y - 1) / g.th)}`, 'o');
    const todo = ppo.keysToGo(g);
    for (const e of todo) mark.set(`${Math.floor(e.x / g.tw)},${Math.floor((e.y - 1) / g.th)}`, 'K');
    for (const e of g.entities) if (e.ai === 'next_level_ai') mark.set(`${Math.floor(e.x / g.tw)},${Math.floor((e.y - 1) / g.th)}`, e.aistate === next ? 'E' : 'x');
    mark.set(`${Math.floor(bestCell.x / g.tw)},${Math.floor((bestCell.y - 1) / g.th)}`, '@');
    for (let r = 0; r < H; r++) { let line = ''; for (let c = 0; c < W; c++) line += mark.get(`${c},${r}`) || (g.tileSolid(c * g.tw + 15, r * g.th + 7) ? '#' : ' '); rows.push(`${String(r * g.th).padStart(5)} ${line.replace(/\s+$/, '')}`); }
    // what the states differ by: how many distinct values each part of a state takes
    const parts = ['keys', 'walls left', 'power', 'weapons'].map((name, i) => `${name} ${new Set([...states.keys()].map((k) => k.split('|')[i])).size}`).join(', ');
    const text = `Level ${lv}: not passed. ${list.length} cells in ${states.size} states of the level (distinct ${parts}); the furthest-on state has ${here.length} cells.\n`
      + `Nearest the exit: ${ppo.fmtDist(bestDist)} at (${bestCell.x},${bestCell.y}), hp ${bestCell.hp}.\n`
      + `Still to do in that state (K on the map): ${todo.map((e) => `${e.def.name}@${Math.round(e.x)},${Math.round(e.y)}`).join('  ') || 'nothing'}\n`
      + `Player in that state: weapons ${[...g.player.owned].join(',')}, power ${g.player.power || 'none'}\n`
      + `Map: one character is ${g.tw} x ${g.th} px. # wall, o stood here, @ nearest the exit, K still to do, E the exit, x an exit to another level.\n\n${rows.join('\n')}\n`;
    await writeFile(`${OUT}search-stuck-level${String(lv).padStart(2, '0')}.txt`, text);
  }
  return { solved: false, pieces: chain(bestCell), dist: bestDist, at: [bestCell.x, bestCell.y], iter, simSteps, cells: list.length, ms: Date.now() - t0 };
}

// ---- the campaign ----
await mkdir(OUT, { recursive: true });
const LEVEL_DIR = `${OUT}levels/`;
await mkdir(LEVEL_DIR, { recursive: true });
const levelFile = (lv) => `${LEVEL_DIR}level${String(lv).padStart(2, '0')}.json`;
let state = { frontier: 0, entry: null, legs: [], partial: null };
if (!process.argv.includes('--fresh')) { try { state = JSON.parse(await readFile(STATE_FILE, 'utf8')); } catch { /* nothing to resume */ } }

// The demo: every level, in order. The levels passed one after another
// (health and weapons carried from each into the next), any searched on its
// own from a fresh start (train-out/levels/), and, for a level the search
// could not pass, its furthest attempt: played to where it got to, where the
// demo stops for a while under the words "AI gave up" (the last piece of such
// a level is marked gaveUp).
async function writeDemo() {
  const legs = new Map(state.legs.map((l) => [l.level, l.pieces]));
  const gaveUp = new Map();
  for (let lv = 0; lv < LEVELS.length; lv++) {
    if (legs.has(lv)) continue;
    try {
      const f = JSON.parse(await readFile(levelFile(lv), 'utf8'));
      if (f.solved) legs.set(lv, f.pieces); else if (f.pieces?.length) gaveUp.set(lv, f.pieces);
    } catch { /* not searched */ }
  }
  const levels = [...legs.keys()].sort((a, b) => a - b);
  const pieces = [];
  for (let lv = 0; lv < LEVELS.length; lv++) {
    const ps = legs.get(lv) || gaveUp.get(lv);
    if (!ps) continue;
    // each piece's saved state goes into the file as what differs from the
    // level as it starts (ppo.packSeed): whole, the file runs to 80 MB
    await ppo.startRun(g, LEVELS[lv], null);
    const fresh = ppo.snapshotState(g);
    ps.forEach((piece, i) => pieces.push({
      ...piece, seed: piece.seed ? ppo.packSeed(piece.seed, fresh) : null, levelIdx: lv,
      ...(gaveUp.has(lv) && i === ps.length - 1 ? { gaveUp: true } : {}),
    }));
  }
  let frontier = 0;
  while (legs.has(frontier)) frontier++;
  if (!pieces.length) return levels;
  // The file may hold the older PPO recording: kept beside the search's progress the first time it is replaced.
  try { const old = JSON.parse(await readFile(DEMO_FILE, 'utf8')); if (old.source !== 'search') await copyFile(DEMO_FILE, `${OUT}ppo-demo-before-search.json`); } catch { /* none */ }
  // the regression test's list of levels that have been won: added to, never taken from (tools/regress.mjs)
  try {
    const wonFile = fileURLToPath(new URL('../tests/won-levels.json', import.meta.url));
    const w = JSON.parse(await readFile(wonFile, 'utf8'));
    const all = [...new Set([...w.levels, ...levels])].sort((a, b) => a - b);
    if (all.length !== w.levels.length) await writeFile(wonFile, JSON.stringify({ ...w, levels: all }, null, 2));
  } catch { /* no list yet */ }
  await writeFile(DEMO_FILE, JSON.stringify({ format: ppo.DEMO_FORMAT, savedAt: new Date().toISOString(), source: 'search', passed: levels, gaveUp: [...gaveUp.keys()], frontier, dist: null, pieces }));
  return levels;
}
async function save() {
  await writeFile(STATE_FILE, JSON.stringify(state));
  await writeDemo();
}
const nn = (lv) => String(lv).padStart(2, '0');
const passedLine = (lv, r) => `level ${nn(lv)}: PASSED in ${(r.ms / 1000).toFixed(0)} s of searching (${r.iter} bursts, ${r.cells} cells); the route is ${(r.pieces.reduce((a, p) => a + p.steps, 0) / 60).toFixed(0)} s of play in ${r.pieces.length} pieces`;
const failedLine = (lv, r) => `level ${nn(lv)}: NOT passed in ${(r.ms / 60000).toFixed(1)} min (${r.iter} bursts, ${r.cells} cells). Nearest the exit: ${ppo.fmtDist(r.dist)} at (${r.at}).`;

// ---- one level on its own (what --parallel runs, one process per level) ----
// It starts with the weapons and ammunition the player finished the level
// before with, as the game carries them across (view.cpp: weapons belong to
// the player's view, not to the level). "The level before" is the nearest
// earlier level that has a winning run; its last piece is replayed to see
// what the player is holding at the exit. Health is full and the machine gun
// is topped up to 100, as for any level started on its own; the power in
// hand is not carried (people.lsp set_player_defaults).
//
// This is a snapshot of that run as it is now. If the earlier level is
// searched again later and ends with a different loadout, this level's
// recording still replays exactly (it carries its own starting state); only
// the hand-over in the demo no longer matches to the round.
//   --no-carry   start with the machine gun alone
async function routeOf(lv) {
  const leg = state.legs.find((l) => l.level === lv);
  if (leg) return leg.pieces;
  try { const f = JSON.parse(await readFile(levelFile(lv), 'utf8')); if (f.solved) return f.pieces; } catch { /* not searched */ }
  return null;
}
async function carryInto(lv) {
  if (process.argv.includes('--no-carry')) return { entry: null, from: null };
  for (let k = lv - 1; k >= 0; k--) {
    const pieces = await routeOf(k);
    if (!pieces || !pieces.length) continue;
    const last = pieces[pieces.length - 1];
    await ppo.startRun(g, LEVELS[k], last.seed || null);
    exitDest = null;
    const acts = ppo.decodeActs(last.acts), n = Math.min(last.steps, acts.length * 4);
    let face = 1, act = 0;
    for (let i = 0; i < n && exitDest === null && !g.player.dead; i++) {
      if (i % 4 === 0) { act = acts[i / 4]; const mv = (act % 3) - 1; if (mv !== 0) face = mv; }
      ppo.applyAction(g, act, face); g.update(1 / 60);
    }
    const owned = [...g.player.owned], ammo = { ...g.player.ammo };
    await ppo.startRun(g, LEVELS[lv], null); // the level as it starts: full health, no power in hand
    const p = g.player;
    p.owned = new Set(['MGUN', ...owned]);
    p.ammo = { ...ammo, MGUN: Math.max(ammo.MGUN || 0, 100) };
    p.weapon = 'MGUN';
    exitDest = null;
    return { entry: ppo.snapshotState(g), from: k, owned: [...p.owned], ammo: p.ammo };
  }
  return { entry: null, from: null };
}
// --repair (with --only): a won route that a change to the game has broken
// part way is mended rather than thrown away. Its pieces are replayed, each
// from its own saved start, as the regression does; the first piece that no
// longer ends where the next begins is kept, played as the game now plays it,
// and the search carries on from where it ends.
async function repairStart(lv) {
  let f = null;
  try { f = JSON.parse(await readFile(levelFile(lv), 'utf8')); } catch { /* none */ }
  if (!f?.pieces?.length) return null;
  const pieces = f.pieces;
  for (let k = 0; k < pieces.length; k++) {
    await ppo.startRun(g, LEVELS[lv], pieces[k].seed || null);
    exitDest = null;
    const acts = ppo.decodeActs(pieces[k].acts), n = Math.min(pieces[k].steps, acts.length * 4);
    let face = 1, act = 0;
    for (let i = 0; i < n && exitDest === null && !g.player.dead; i++) {
      if (i % 4 === 0) { act = acts[i / 4]; const mv = (act % 3) - 1; if (mv !== 0) face = mv; }
      ppo.applyAction(g, act, face); g.update(1 / 60);
    }
    const nx = pieces[k + 1];
    if (g.player.dead) return { prefix: pieces.slice(0, k), entry: pieces[k].seed, kept: k, of: pieces.length, carried: f.carried, carriedFrom: f.carriedFrom };
    if (!nx) return null; // it still wins as it is
    if (Math.round(g.player.x) !== nx.seed.px || Math.round(g.player.y) !== nx.seed.py) {
      return { prefix: pieces.slice(0, k + 1), entry: ppo.snapshotState(g), kept: k + 1, of: pieces.length, carried: f.carried, carriedFrom: f.carriedFrom };
    }
  }
  return null;
}
// --write-demo: rebuild the demo from the level files as they stand, and stop
if (process.argv.includes('--write-demo')) {
  const levels = await writeDemo();
  console.log(`demo: ${levels.length} levels won (${levels.join(', ')})`);
  process.exit(0);
}
if (process.argv.includes('--only') && process.argv.includes('--repair')) {
  const lv = +arg('only');
  const rep = await repairStart(lv);
  if (!rep) { console.log(`level ${nn(lv)}: nothing to repair`); process.exit(0); }
  console.log(`level ${nn(lv)}: keeping ${rep.kept} of ${rep.of} pieces of the old route; searching on from where piece ${rep.kept} now ends`);
  const r = await searchLevel(lv, rep.entry, Date.now() + PER_LEVEL);
  const out = arg('out', null) || (r.solved ? levelFile(lv) : levelFile(lv).replace(/\.json$/, '.failed.json'));
  const pieces = rep.prefix.concat(r.pieces);
  await writeFile(out, JSON.stringify({ level: lv, solved: r.solved, carried: rep.carried, carriedFrom: rep.carriedFrom, pieces, ms: r.ms, iter: r.iter, cells: r.cells, dist: r.dist ?? null, at: r.at ?? null, repaired: true }));
  console.log(r.solved ? passedLine(lv, { ...r, pieces }) : failedLine(lv, r));
  if (r.solved && !arg('out', null)) { const levels = await writeDemo(); console.log(`  demo: ${levels.length} levels won (${levels.join(', ')})`); }
  process.exit(0);
}
if (process.argv.includes('--only')) {
  const lv = +arg('only');
  const carry = await carryInto(lv);
  console.log(carry.entry ? `level ${nn(lv)}: starting with what level ${nn(carry.from)}'s run ends with: ${carry.owned.map((w) => `${w} ${carry.ammo[w] || 0}`).join(', ')}` : `level ${nn(lv)}: starting with the machine gun alone`);
  const r = await searchLevel(lv, carry.entry, Date.now() + PER_LEVEL);
  // --out: where to put the result (--in-order races several of these and picks the winner).
  // Without it, the level's own file, but a failed attempt never replaces a winning run.
  let out = arg('out', null);
  if (!out) {
    out = levelFile(lv);
    if (!r.solved) { try { if (JSON.parse(await readFile(out, 'utf8')).solved) out = out.replace(/\.json$/, '.failed.json'); } catch { /* nothing there */ } }
  }
  await writeFile(out, JSON.stringify({ level: lv, solved: r.solved, carried: !!carry.entry, carriedFrom: carry.from, pieces: r.pieces, ms: r.ms, iter: r.iter, cells: r.cells, dist: r.dist ?? null, at: r.at ?? null }));
  console.log(r.solved ? passedLine(lv, r) : failedLine(lv, r));
  process.exit(0);
}

// ---- the whole game in order ----
//   node web/tools/search.mjs --in-order [--from 0] [--upto 21] [--per-level 60] [--jobs 6]
// Plays through from the first level to the last, each level starting with
// the weapons and ammunition the one before ended with (see carryInto), so
// the demo is one continuous run. Each level is searched by several processes
// at once with different runs of the dice; the first to reach the exit wins
// and the rest are stopped. A level nobody passes in time keeps its earlier
// winning run if it has one, and the playthrough goes on to the next level.
if (process.argv.includes('--in-order')) {
  const { spawn } = await import('node:child_process');
  const { openSync } = await import('node:fs');
  const { rename, rm } = await import('node:fs/promises');
  const jobs = +arg('jobs', 6), from = +arg('from', 0), upto = +arg('upto', 21);
  const race = `${LEVEL_DIR}race${from ? `-${from}` : ''}/`; // (one per run, so two runs over different levels can go side by side)
  await mkdir(race, { recursive: true });
  // the old one-after-another campaign kept its runs in search.json: move them out to level files, the one place routes now live
  for (const leg of state.legs) { try { await readFile(levelFile(leg.level)); } catch { await writeFile(levelFile(leg.level), JSON.stringify({ level: leg.level, solved: true, carried: false, pieces: leg.pieces })); } }
  state = { frontier: 0, entry: null, legs: [], partial: null };
  await writeFile(STATE_FILE, JSON.stringify(state));
  const stamp = () => new Date().toISOString().slice(0, 16).replace('T', ' ');
  console.log(`${stamp()}  playing through in order from level ${from}: ${jobs} tries at a time, up to ${PER_LEVEL / 60000} min a level`);
  for (let lv = from; lv <= upto; lv++) {
    const results = [];
    let winner = null;
    await new Promise((resolve) => {
      let running = 0;
      const kids = [];
      for (let k = 0; k < jobs; k++) {
        const out = `${race}level${nn(lv)}-${k}.json`;
        const log = openSync(k === 0 ? `${LEVEL_DIR}level${nn(lv)}.log` : `${race}level${nn(lv)}-${k}.log`, 'w');
        const child = spawn(process.execPath, [fileURLToPath(import.meta.url), '--only', String(lv), '--out', out, '--per-level', String(PER_LEVEL / 60000), '--seed', String(seed + lv * 100 + k * 7)], { stdio: ['ignore', log, log] });
        kids.push(child); running++;
        child.on('exit', async () => {
          try { const f = JSON.parse(await readFile(out, 'utf8')); results.push(f); if (f.solved && !winner) { winner = f; for (const c of kids) if (c !== child) c.kill(); } } catch { /* stopped before it wrote anything */ }
          if (--running === 0) resolve();
        });
      }
    });
    let had = false;
    try { had = JSON.parse(await readFile(levelFile(lv), 'utf8')).solved; } catch { /* no earlier run */ }
    if (winner) {
      await writeFile(levelFile(lv), JSON.stringify(winner));
      console.log(`${stamp()}  ${passedLine(lv, winner)}${winner.carried ? ` (carrying from level ${nn(winner.carriedFrom)})` : ''}`);
    } else {
      const best = results.filter((f) => f.pieces?.length).sort((a, b) => (a.dist ?? Infinity) - (b.dist ?? Infinity))[0];
      if (!had && best) await writeFile(levelFile(lv), JSON.stringify(best));
      console.log(`${stamp()}  ${best ? failedLine(lv, best) : `level ${nn(lv)}: no result`}${had ? ' Its earlier winning run is kept.' : ''}`);
    }
    await rm(race, { recursive: true, force: true }); await mkdir(race, { recursive: true });
    const levels = await writeDemo();
    console.log(`${stamp()}    demo: ${levels.length} levels won (${levels.join(', ')})`);
  }
  console.log(`${stamp()}  finished the playthrough`);
  process.exit(0);
}

// ---- after the game's code has changed ----
//   node web/tools/search.mjs --recheck
// A route is only good for the game it was found in: any change to how the
// game plays can make it drift. This replays every route exactly as the demo
// does and throws away the ones that no longer join up and reach their exit,
// so that --parallel searches those levels again.
async function replays(lv, pieces) {
  for (let k = 0; k < pieces.length; k++) {
    const piece = pieces[k];
    await ppo.startRun(g, LEVELS[lv], piece.seed || null);
    exitDest = null;
    const acts = ppo.decodeActs(piece.acts), steps = Math.min(piece.steps, acts.length * 4);
    let face = 1, act = 0;
    for (let i = 0; i < steps && exitDest === null && !g.player.dead; i++) {
      if (i % 4 === 0) { act = acts[i / 4]; const mv = (act % 3) - 1; if (mv !== 0) face = mv; }
      ppo.applyAction(g, act, face); g.update(1 / 60);
    }
    if (g.player.dead) return false;
    const nx = pieces[k + 1];
    if (nx ? Math.round(g.player.x) !== nx.seed.px || Math.round(g.player.y) !== nx.seed.py : exitDest !== lv + 1) return false;
  }
  return true;
}
if (process.argv.includes('--recheck')) {
  // (dropped routes are moved aside, not deleted: if the change to the game is
  // taken back, they are good again; copy them back from train-out/levels/dropped/)
  const { rename } = await import('node:fs/promises');
  const aside = `${LEVEL_DIR}dropped/`;
  await mkdir(aside, { recursive: true });
  const kept = [], dropped = [];
  for (const leg of [...state.legs]) {
    if (await replays(leg.level, leg.pieces)) kept.push(leg.level);
    else {
      dropped.push(leg.level); state.legs = state.legs.filter((l) => l !== leg);
      await writeFile(`${aside}level${nn(leg.level)}.json`, JSON.stringify({ level: leg.level, solved: true, carried: false, pieces: leg.pieces }));
    }
  }
  for (let lv = 0; lv < LEVELS.length; lv++) {
    let f = null;
    try { f = JSON.parse(await readFile(levelFile(lv), 'utf8')); } catch { continue; }
    if (!f.solved) continue;
    if (await replays(lv, f.pieces)) kept.push(lv); else { dropped.push(lv); await rename(levelFile(lv), `${aside}level${nn(lv)}.json`); }
  }
  await save();
  console.log(`routes that still replay: levels ${kept.sort((a, b) => a - b).join(', ') || 'none'}\nroutes dropped (to be searched again): levels ${dropped.sort((a, b) => a - b).join(', ') || 'none'}`);
  process.exit(0);
}

// ---- many levels at once ----
//   node web/tools/search.mjs --parallel [--jobs 6] [--per-level 30] [--upto 21]
// Every level up to --upto (default 21, the last) that has no route yet is
// searched in a process of its own.
if (process.argv.includes('--parallel')) {
  const { spawn } = await import('node:child_process');
  const { openSync } = await import('node:fs');
  const jobs = +arg('jobs', 6), upto = +arg('upto', 21);
  const pick = arg('pick', null)?.split(',').map(Number) ?? null; // --pick 2,3,8: only these levels
  const todo = [];
  for (let lv = 0; lv <= upto; lv++) {
    let done = state.legs.some((l) => l.level === lv);
    if (!done) try { done = JSON.parse(await readFile(levelFile(lv), 'utf8')).solved; } catch { /* not searched */ }
    if (!done && (!pick || pick.includes(lv))) todo.push(lv);
  }
  console.log(`searching levels ${todo.join(', ')}: ${jobs} at a time, up to ${PER_LEVEL / 60000} min each`);
  let running = 0;
  await new Promise((resolve) => {
    const next = () => {
      if (!todo.length && !running) { resolve(); return; }
      while (running < jobs && todo.length) {
        const lv = todo.shift();
        running++;
        const log = openSync(`${LEVEL_DIR}level${nn(lv)}.log`, 'w');
        const child = spawn(process.execPath, [fileURLToPath(import.meta.url), '--only', String(lv), '--per-level', String(PER_LEVEL / 60000), '--seed', String(seed + lv)], { stdio: ['ignore', log, log] });
        child.on('exit', async (code) => {
          running--;
          let line = `level ${nn(lv)}: the search process stopped (exit code ${code}); see ${LEVEL_DIR}level${nn(lv)}.log`;
          try { const f = JSON.parse(await readFile(levelFile(lv), 'utf8')); line = f.solved ? passedLine(lv, f) : failedLine(lv, f); } catch { /* crashed before writing */ }
          console.log(line);
          const levels = await writeDemo();
          console.log(`  demo now has levels ${levels.join(', ')}`);
          next();
        });
      }
    };
    next();
  });
  const levels = await writeDemo();
  console.log(`\nlevels with a route: ${levels.join(', ') || 'none'}\nsaved ${DEMO_FILE}`);
  process.exit(0);
}

// ---- one level after another ----
let passed = 0;
while (state.frontier < LEVELS.length && passed < MAX_LEVELS) {
  const lv = state.frontier;
  console.log(`level ${nn(lv)}: searching (up to ${PER_LEVEL / 60000} min)`);
  const r = await searchLevel(lv, state.entry, Date.now() + PER_LEVEL);
  if (!r.solved) {
    state.partial = { levelIdx: lv, pieces: r.pieces, dist: r.dist };
    await save();
    console.log(failedLine(lv, r));
    break;
  }
  console.log(passedLine(lv, r));
  // enter the next level as the game would, carrying health and weapons over
  await g.start(LEVELS[lv + 1] ?? LEVELS[lv]);
  g.player.ammo.MGUN = Math.max(g.player.ammo.MGUN || 0, 100);
  state = { frontier: lv + 1, entry: lv + 1 < LEVELS.length ? ppo.snapshotState(g) : null, legs: [...state.legs, { level: lv, dest: lv + 1, pieces: r.pieces }], partial: null };
  await save();
  passed++;
}
console.log(`\nlevels passed: ${state.legs.map((l) => l.level).join(', ') || 'none'}; now on level ${state.frontier}\nsaved ${STATE_FILE}\nsaved ${DEMO_FILE}`);
process.exit(0);
