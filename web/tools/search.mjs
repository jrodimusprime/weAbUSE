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
//
// Writes train-out/search.json (progress; resumed from) and
// web/data/ppo-demo.json (what "Full game demo" plays).
import { mkdir, readFile, writeFile, copyFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
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
function pack(snap, base) {
  const es = {}, gone = [];
  for (const id in snap.es) if (!base.es[id] || !sameEnt(snap.es[id], base.es[id])) es[id] = snap.es[id];
  for (const id in base.es) if (!snap.es[id]) gone.push(id);
  return { ...snap, es, gone };
}
function unpack(packed, base) {
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
  for (const e of g.entities) if (!e.dead && e.shootable && !e.hidden && e.hp > 0 && Math.abs(e.x - p.x) < ENEMY_NEAR && Math.abs(e.y - p.y) < 180 && e.def.name !== 'DARNEL' && /ant|fly|gun|jug|bomb|rob|who/i.test(e.ai || '')) return true;
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

const wallsLeft = () => { let n = 0; for (const e of g.entities) if (!e.dead && (e.ai === 'hwall_ai' || e.ai === 'big_wall_ai' || e.ai === 'block_ai')) n++; return n; };
function cellKey() {
  const p = g.player;
  let keys = '';
  for (const e of ppo.keysToGo(g)) keys += `${e.id}.`;
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
  const root = { key: cellKey(), snap: null, full: entry, parent: null, acts: '', pieceSteps: 0, steps: 0, hp: g.player.hp, dist: ppo.exitDist(g, next), chosen: 0, born: 0 };
  cells.set(root.key, root); list.push(root);
  let iter = 0, simSteps = 0, bestDist = root.dist, bestCell = root, lastPrint = Date.now(), deaths = 0;
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
    // pick a cell: ones seldom tried, new ones, and ones furthest along the compass
    let total = 0;
    const w = new Float64Array(list.length);
    for (let i = 0; i < list.length; i++) {
      const c = list[i];
      const gap = c.dist - bestDist;
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
    let face = 1, act = 0, hold = 0;
    const dirBias = rnd() < 0.5 ? (rnd() < 0.5 ? 1 : -1) : 0; // some bursts keep heading one way
    const burst = rnd() < 0.25 ? BURST * 3 : BURST; // now and then a long one: some things take patience
    for (let d = 0; d < burst; d++) {
      if (hold-- <= 0) { climb = 0; act = randomAct(dirBias); hold = climb || Math.floor(rnd() * rnd() * 14); }
      acts.push(act);
      const mv = (act % 3) - 1;
      if (mv !== 0) face = mv;
      let s = 0;
      for (; s < 4 && exitDest === null && !g.player.dead; s++) { ppo.applyAction(g, act, face); g.update(1 / 60); simSteps++; }
      if (exitDest !== null) {
        if (exitDest === next) return { solved: true, pieces: chain(null, ppo.encodeActs(acts), (acts.length - 1) * 4 + s, cell), iter, simSteps, cells: list.length, ms: Date.now() - t0 };
        break; // an exit to somewhere else: not a way on
      }
      if (g.player.dead) { deaths++; break; }
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
        if (old) list[list.indexOf(old)] = rec; else list.push(rec);
        if (rec.dist < bestDist) { bestDist = rec.dist; bestCell = rec; }
      }
    }
    if (Date.now() - lastPrint > 15000) {
      lastPrint = Date.now();
      console.log(`  level ${String(lv).padStart(2, '0')}  ${((Date.now() - t0) / 60000).toFixed(1)} min  ${iter} bursts  ${list.length} cells  ${Math.round(simSteps / ((Date.now() - t0) / 1000))} steps/s  nearest the exit: ${ppo.fmtDist(bestDist)} at (${bestCell.x ?? '-'},${bestCell.y ?? '-'})  deaths ${deaths}`);
    }
  }
  // out of time: a map of everywhere it stood in the furthest-on state of the
  // level, with what is still to be done marked, for working out what stops it
  {
    const state = (c) => c.key.slice(c.key.indexOf('|'));
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
    const states = new Map(); for (const c of list) states.set(state(c), (states.get(state(c)) || 0) + 1);
    const text = `Level ${lv}: not passed. ${list.length} cells in ${states.size} states of the level; the furthest-on state has ${here.length} cells.\n`
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
let state = { frontier: 0, entry: null, legs: [], partial: null };
if (!process.argv.includes('--fresh')) { try { state = JSON.parse(await readFile(STATE_FILE, 'utf8')); } catch { /* nothing to resume */ } }

async function save() {
  await writeFile(STATE_FILE, JSON.stringify(state));
  const demo = ppo.demoFile({ frontier: state.frontier, legs: state.legs }, state.partial);
  demo.source = 'search';
  if (!demo.pieces.length) return;
  // The file may hold the older PPO recording: kept beside the search's progress the first time it is replaced.
  try { const old = JSON.parse(await readFile(DEMO_FILE, 'utf8')); if (old.source !== 'search') await copyFile(DEMO_FILE, `${OUT}ppo-demo-before-search.json`); } catch { /* none */ }
  await writeFile(DEMO_FILE, JSON.stringify(demo));
}

let passed = 0;
while (state.frontier < LEVELS.length && passed < MAX_LEVELS) {
  const lv = state.frontier;
  console.log(`level ${String(lv).padStart(2, '0')}: searching (up to ${PER_LEVEL / 60000} min)`);
  const r = await searchLevel(lv, state.entry, Date.now() + PER_LEVEL);
  if (!r.solved) {
    state.partial = { levelIdx: lv, pieces: r.pieces, dist: r.dist };
    await save();
    console.log(`level ${String(lv).padStart(2, '0')}: NOT passed in ${(r.ms / 60000).toFixed(1)} min (${r.iter} bursts, ${r.cells} cells). Nearest the exit: ${ppo.fmtDist(r.dist)} at (${r.at}).`);
    break;
  }
  const playSeconds = r.pieces.reduce((a, p) => a + p.steps, 0) / 60;
  console.log(`level ${String(lv).padStart(2, '0')}: PASSED in ${(r.ms / 1000).toFixed(0)} s of searching (${r.iter} bursts, ${r.cells} cells); the route is ${playSeconds.toFixed(0)} s of play in ${r.pieces.length} pieces`);
  // enter the next level as the game would, carrying health and weapons over
  await g.start(LEVELS[lv + 1] ?? LEVELS[lv]);
  g.player.ammo.MGUN = Math.max(g.player.ammo.MGUN || 0, 100);
  state = { frontier: lv + 1, entry: lv + 1 < LEVELS.length ? ppo.snapshotState(g) : null, legs: [...state.legs, { level: lv, dest: lv + 1, pieces: r.pieces }], partial: null };
  await save();
  passed++;
}
console.log(`\nlevels passed: ${state.legs.map((l) => l.level).join(', ') || 'none'}; now on level ${state.frontier}\nsaved ${STATE_FILE}\nsaved ${DEMO_FILE}`);
process.exit(0);
