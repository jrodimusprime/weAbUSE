// Checks every puzzle in every level by working it, headlessly, in the real
// game code: nothing here knows how a door or a lift is "supposed" to behave
// beyond what the level file wires up.
//
//   node web/tools/audit.mjs                 all 22 levels
//   node web/tools/audit.mjs 5 12            only these levels
//   node web/tools/audit.mjs --verbose       also list everything that passed
//   node web/tools/audit.mjs --behaviours    the checklist: every behaviour the
//                                            levels use, where the original
//                                            defines it, and whether it is ported
//
// What is checked, per level:
//   chains   everything a trigger controls (door, trap door, force field,
//            wired wall, lift with an enable switch, pushers, movers, and the
//            creatures a switch or sensor lets loose). The object's trigger
//            link is followed back through the logic gates to its sources:
//            switches are pressed, sensors stood in, and whatever a death
//            sensor watches is removed. The object must then change.
//   lifts    standing at the far stop calls the lift; pressing the action key
//            on its deck is carried to the other stop.
//   teleporters  pressing the action key in one delivers the player to its
//            destination.
//   exits    pressing the action key at one leaves the level.
//
// The player is placed by hand (and cannot die), so this proves each puzzle
// works once reached, not that it can be reached.
import { readFile, readdir } from 'node:fs/promises';
import { makeGame, LEVELS } from './headless.mjs';

const args = process.argv.slice(2);
const verbose = args.includes('--verbose');
const only = args.filter((a) => /^\d+$/.test(a)).map(Number);

const GATE_AI = new Set(['and_ai', 'or_ai', 'xor_ai', 'not_ai', 'delay_ai', 'pulse_ai', 'indicator_ai']);
const SWITCH_AI = new Set(['switcher_ai', 'switch_once_ai', 'switch_delay_ai']);
const SOURCE_AI = new Set([...SWITCH_AI, 'sensor_ai', 'death_sen_ai']);
// Which link is the one an object listens to (platform.lsp: the third, after its two stops).
const triggerLink = (e) => (e.ai === 'platform_ai' ? e.links[2] : e.links[0]);
// Puzzle pieces, as opposed to creatures and scenery that a trigger merely wakes.
// Behaviours whose first link is not something they listen to (the original's code never reads its state).
const DEAF_AI = new Set(['end_game_ai', 'hp_up', 'tp2_ai', 'health_power_ai', 'fly_power_ai', 'fast_power_ai', 'sneaky_power_ai', 'next_level_ai', 'latter_ai', 'marker_ai']);
const PUZZLE_AI = new Set(['sdoor_ai', 'strap_door_ai', 'ff_ai', 'hwall_ai', 'big_wall_ai', 'platform_ai', 'pusher_ai', 'switch_mover_ai', 'mover_ai', 'tp_door_ai', 'lightin_ai']);

const g = await makeGame(LEVELS[0]);
const { behaviors } = await import('../src/behaviors.js');
const p = () => g.player;
const ent = (id) => g.entities.find((e) => e.id === id) || null;
const name = (e) => `${e.def.name}@${Math.round(e.x)},${Math.round(e.y)}`;
let exitHit = null;

async function load(i) {
  await g.start(LEVELS[i]);
  g.god = true;
  exitHit = null;
  g.nextLevel = (n) => { exitHit = n ?? true; };
  g.onEndGame = () => { exitHit = 'the ending'; };
}

// Runs the game with the player held at (x, y), camera on them.
function hold(x, y, ticks, keys = []) {
  const pl = p();
  for (let i = 0; i < ticks * 4; i++) {
    pl.x = x; pl.y = y; pl.vx = 0; pl.vy = 0; pl.hp = 100; pl.dead = false;
    g.cam.x = x - 160; g.cam.y = y - 100;
    g.keys.clear(); for (const k of keys) g.keys.add(k);
    g.rightDown = false; g.mouseDown = false;
    g.update(1 / 60);
  }
}
// Runs the game with the player left to stand, ride and fall.
function free(ticks, keys = []) {
  const pl = p();
  for (let i = 0; i < ticks * 4; i++) {
    pl.hp = 100; pl.dead = false;
    g.keys.clear(); for (const k of keys) g.keys.add(k);
    g.rightDown = false; g.mouseDown = false;
    g.update(1 / 60);
  }
}

// What an object looks like to the player: gone, or its state and place. Two
// kinds show being switched on only as activity, so that is sampled.
function sig(id) {
  const e = ent(id);
  if (!e || e.dead) return 'gone';
  let s = `${e.aistate}|${e.state}|${Math.round(e.x)},${Math.round(e.y)}`;
  if (e.ai === 'ff_ai') s += `|beam:${!!e.a.beam}`;
  if (e.ai === 'pusher_ai') s += `|on:${e.links.length ? e.links[0].aistate !== 0 : true}`;
  return s;
}

// The sources behind a trigger link: followed back through the gates.
function sources(e, seen = new Set(), out = []) {
  if (!e || seen.has(e)) return out;
  seen.add(e);
  if (GATE_AI.has(e.ai)) for (const l of e.links) sources(l, seen, out);
  else out.push(e);
  return out;
}

// How long a signal takes to get through the gates behind a trigger link (delay gates hold it back).
function delayOf(e, seen = new Set()) {
  if (!e || seen.has(e) || !GATE_AI.has(e.ai)) return 0;
  seen.add(e);
  return (e.ai === 'delay_ai' ? e.lv?.delay_time ?? 1 : 1) + Math.max(0, ...e.links.map((l) => delayOf(l, seen)));
}
// Whether an object runs at all with the player at (x, y): game.cpp's rule, the
// view widened by a quarter each way plus the object's own range.
function awakeFrom(e, x, y) {
  const [rx, ry] = e.def.range, cx = x - 160, cy = y - 100;
  return e.x + rx >= cx - 80 && e.x - rx <= cx + 400 && e.y + ry >= cy - 50 && e.y - ry <= cy + 250;
}

// Works a source the way the player would. Returns a description, or null if it cannot be worked.
function work(id) {
  const s = ent(id);
  if (!s) return 'already gone';
  if (SWITCH_AI.has(s.ai)) {
    const was = s.aistate;
    if (was !== 0 && s.ai !== 'switcher_ai') return 'already on';
    hold(s.x, s.y, 3, ['ArrowDown']);
    hold(s.x, s.y, 3);
    // (a two-way switch that starts on is turned off: that is the change its puzzle is waiting for)
    return !!ent(id)?.aistate !== !!was ? (was ? 'switched off' : 'pressed') : 'PRESSING IT DID NOTHING';
  }
  if (s.ai === 'sensor_ai') {
    let on = s.aistate !== 0;
    for (let t = 0; t < 4 && !on; t++) { hold(s.x, s.y, 1); on = !!ent(id)?.aistate; }
    return on ? 'stood in' : 'STANDING IN IT DID NOTHING';
  }
  if (s.ai === 'death_sen_ai') {
    for (const l of [...s.links]) l.dead = true; // (killed outright: the kill itself is not what is under test)
    hold(s.x, s.y, 3);
    return ent(id)?.aistate ? 'emptied' : 'EMPTYING IT DID NOTHING';
  }
  return null;
}

// ---- the checklist of behaviours ----
if (args.includes('--behaviours')) {
  const root = new URL('../../abuse-0.8/', import.meta.url);
  const texts = [];
  for (const dir of ['data/lisp/', 'src/']) {
    let files = [];
    try { files = await readdir(new URL(dir, root)); } catch { /* the original source is not checked in */ }
    for (const f of files) if (/\.(lsp|cpp)$/.test(f)) texts.push([dir + f, await readFile(new URL(dir + f, root), 'latin1')]);
  }
  const where = (ai) => {
    const hits = [];
    for (const [f, t] of texts) if (new RegExp(f.endsWith('.lsp') ? `\\(defun ${ai}\\b` : `^void \\*${ai}\\(`, 'm').test(t)) hits.push(f);
    return hits.join(', ') || '?';
  };
  const use = new Map();
  for (let i = 0; i < LEVELS.length; i++) {
    await load(i);
    for (const o of g.level.objects) {
      const ai = g.assets.defs.get(o.type)?.funs.get('ai_fun');
      if (!ai) continue;
      const u = use.get(ai) || use.set(ai, { n: 0, levels: new Set() }).get(ai);
      u.n++; u.levels.add(i);
    }
  }
  console.log('behaviour                objects  levels  ported  puzzle  original');
  for (const [ai, u] of [...use].sort((a, b) => a[0].localeCompare(b[0]))) {
    console.log(`${ai.padEnd(24)} ${String(u.n).padStart(7)}  ${String(u.levels.size).padStart(6)}  ${(behaviors[ai] ? 'yes' : 'NO').padEnd(6)}  ${(PUZZLE_AI.has(ai) || GATE_AI.has(ai) || SOURCE_AI.has(ai) || ['tp2_ai', 'next_level_ai'].includes(ai) ? 'yes' : '').padEnd(6)}  ${where(ai)}`);
  }
  process.exit(0);
}

// ---- the audit ----
const totals = { pass: 0, fail: 0, warn: 0 };
const failures = [];
for (let lv = 0; lv < LEVELS.length; lv++) {
  if (only.length && !only.includes(lv)) continue;
  await load(lv);
  const rows = []; // [kind, verdict, text]
  const add = (kind, verdict, text) => { rows.push([kind, verdict, text]); totals[verdict]++; if (verdict !== 'pass') failures.push(`level ${String(lv).padStart(2, '0')}  ${verdict.toUpperCase()}  ${kind}: ${text}`); };

  // what there is to test, from the level as it starts
  const chains = [], lifts = [], teles = [], exits = [];
  for (const e of g.entities) {
    if (e.ai === 'platform_ai' && e.links.length >= 2) lifts.push(e.id);
    if (e.ai === 'tp2_ai' && e.links.length) teles.push(e.id);
    if (e.ai === 'next_level_ai' || e.ai === 'end_game_ai') exits.push(e.id);
    if (GATE_AI.has(e.ai) || SOURCE_AI.has(e.ai) || DEAF_AI.has(e.ai)) continue;
    const t = triggerLink(e);
    if (!t) continue;
    const src = sources(t);
    if (!src.length || !src.every((s) => SOURCE_AI.has(s.ai))) continue; // its link is not a trigger (a destination, something carried, ...)
    // (things that stay on are worked first; a sensor only holds while the player stands in it)
    // and of the sensors, first the ones that stay on longest after the player walks away (xacel is that range)
    src.sort((a, b) => (a.ai === 'sensor_ai') - (b.ai === 'sensor_ai') || b.xacel - a.xacel);
    chains.push({ id: e.id, puzzle: PUZZLE_AI.has(e.ai), src: src.map((s) => s.id), delay: delayOf(t), name: name(e), names: new Map(src.map((s) => [s.id, name(s)])) });
  }

  // Works the given sources of a chain in a fresh copy of the level; says how the object changed, if it did.
  const tryChain = async (c, src) => {
    await load(lv);
    const e0 = ent(c.id);
    const at = [e0.x, e0.y - 60];
    const start = sig(c.id);
    hold(at[0], at[1], 20);
    const near = sig(c.id);
    let changed = null;
    const notes = [];
    for (const sid of src) {
      const how = work(sid);
      notes.push(how);
      if (how && /[A-Z]{4}/.test(how)) continue;
      // wait out the gates: beside the object, or in the sensor for one that only holds while stood in
      const s = ent(sid), stay = s && s.ai === 'sensor_ai' ? [s.x, s.y] : at;
      for (let t = 0; t < c.delay + 20 && sig(c.id) === near; t += 5) hold(stay[0], stay[1], 5);
      if (sig(c.id) !== near) { changed = c.delay > 5 ? `after about ${c.delay} ticks of gate delay` : 'at once'; break; }
      hold(at[0], at[1], 40);
      if (sig(c.id) !== near) { changed = 'on coming back to it'; break; }
    }
    // several sensors at once: stand where they overlap, if they do
    const sens = src.map(ent).filter((s) => s && s.ai === 'sensor_ai');
    if (!changed && sens.length > 1) {
      const x0 = Math.max(...sens.map((s) => s.x - s.xvel)), x1 = Math.min(...sens.map((s) => s.x + s.xvel));
      const y0 = Math.max(...sens.map((s) => s.y - s.yvel)), y1 = Math.min(...sens.map((s) => s.y + s.yvel));
      if (x1 - x0 > 2 && y1 - y0 > 2) {
        for (let t = 0; t < c.delay + 20 && sig(c.id) === near; t += 5) hold((x0 + x1) / 2, (y0 + y1) / 2, 5);
        if (sig(c.id) !== near) changed = 'standing where its sensors overlap';
      }
    }
    return { changed, notes, onApproach: near !== start, broken: notes.find((n) => n && /[A-Z]{4}/.test(n)) };
  };

  for (const c of chains) {
    const names = c.names;
    const label = `${c.name} <- ${c.src.map((id) => names.get(id)).join(' + ')}`;
    const kind = c.puzzle ? 'chain' : 'wake';
    let r = await tryChain(c, c.src);
    // A combination lock: some sources must be worked and others left alone (a NOT gate in the wiring).
    if (!r.changed && !r.broken && c.src.length > 1 && c.src.length <= 4) {
      for (let m = 1; m < (1 << c.src.length) - 1 && !r.changed; m++) {
        const sub = c.src.filter((_, k) => m & (1 << k));
        const t = await tryChain(c, sub);
        if (t.changed) r = { ...t, changed: `${t.changed}, working only ${sub.map((id) => names.get(id)).join(' + ')}` };
      }
    }
    if (r.broken) add(kind, 'fail', `${label}: ${r.broken.toLowerCase()}`);
    else if (r.changed) add(kind, 'pass', `${label} (${r.changed})`);
    else if (r.onApproach && c.src.every((id) => /^SENSOR/.test(names.get(id)))) add(kind, 'pass', `${label} (as the player walks up: the sensor is beside it)`);
    else if (r.onApproach) add(kind, c.puzzle ? 'warn' : 'pass', `${label}: changes when the player comes near, but not when its trigger is worked`);
    else add(kind, c.puzzle ? 'fail' : 'warn', `${label}: nothing happens (${r.notes.join(', ')})`);
  }

  for (const id of lifts) {
    // called from the far stop
    await load(lv);
    let e = ent(id);
    if (!e) { add('lift', 'warn', `lift ${id} is not there when the level starts`); continue; }
    const label = name(e);
    const enable = () => {
      const l = ent(id).links[2];
      if (!l || l.aistate !== 0) return true;
      for (const s of sources(l)) work(s.id);
      for (let t = 0; t < delayOf(l) + 20 && ent(id).links[2].aistate === 0; t += 5) free(5);
      return ent(id).links[2].aistate !== 0;
    };
    if (!enable()) { add('lift', 'warn', `${label}: needs ${name(e.links[2])} on first, which could not be worked here`); continue; }
    e = ent(id);
    const far = e.links[e.aitype], home = [e.x, e.y];
    if (!far) { add('lift', 'fail', `${label}: no stop to travel to`); continue; }
    if (far.ai === 'sensor_ai' && !awakeFrom(e, far.x, far.y)) {
      add('lift', 'pass', `${label}: too far from ${name(far)} to be called from there (the same in the original: it is asleep off screen)`);
    } else if (far.ai === 'sensor_ai') {
      // (a lift whose two sensors overlap shuttles for as long as the player stands there: arriving once is enough)
      let there = false;
      for (let t = 0; t < 120 && !there; t++) { hold(far.x, far.y, 1); e = ent(id); there = Math.abs(e.x - far.x) < 2 && Math.abs(e.y - far.y) < 2 && (e.aistate === 0 || t > 0); }
      const went = there && (Math.abs(far.x - home[0]) > 2 || Math.abs(far.y - home[1]) > 2 || e.aitype !== ent(id).aitype || true);
      add('lift', went ? 'pass' : 'fail', `${label}: ${went ? 'comes when called from' : 'does NOT come when called from'} ${name(far)}`);
    }
    // ridden with the action key
    await load(lv);
    e = ent(id);
    enable();
    for (let t = 0; t < 150 && ent(id).aistate !== 0; t++) free(1); // (enabling it may send it off at once: let it settle)
    e = ent(id);
    const from = [e.x, e.y];
    const deck = g.deckRect(e);
    if (!deck) { add('lift', 'warn', `${label}: no deck to stand on`); continue; }
    const pl = p();
    pl.x = (deck.x0 + deck.x1) / 2; pl.y = deck.y0; pl.vx = 0; pl.vy = 0; g.cam.x = pl.x - 160; g.cam.y = pl.y - 100;
    free(2);
    free(1, ['ArrowDown']);
    let started = ent(id).aistate !== 0;
    for (let t = 0; t < 150 && ent(id).aistate !== 0; t++) free(1);
    e = ent(id);
    const moved = started && (Math.abs(e.x - from[0]) > 2 || Math.abs(e.y - from[1]) > 2);
    const d2 = g.deckRect(e);
    const rode = pl.x >= d2.x0 - 10 && pl.x <= d2.x1 + 10 && Math.abs(pl.y - d2.y0) < 14;
    add('lift', moved && rode ? 'pass' : 'fail', `${label}: action key on the deck ${!moved ? 'does NOT start it' : rode ? 'carries the player to the other stop' : `starts it but the player is left behind (lift ${Math.round(e.x)},${Math.round(e.y)}, player ${Math.round(pl.x)},${Math.round(pl.y)})`}`);
  }

  for (const id of teles) {
    await load(lv);
    const e = ent(id), dest = e.links[0];
    hold(e.x, e.y, 2, ['ArrowDown']);
    free(20);
    const pl = p();
    const ok = Math.abs(pl.x - dest.x) < 60 && Math.abs(pl.y - dest.y) < 120;
    add('teleporter', ok ? 'pass' : 'fail', `${name(e)} -> ${name(dest)}${ok ? '' : `: player ended at ${Math.round(pl.x)},${Math.round(pl.y)}`}`);
  }

  if (!exits.length) add('exit', 'fail', 'the level has no exit');
  for (const id of exits) {
    await load(lv);
    const e = ent(id);
    if (e.ai === 'end_game_ai') {
      // the last level ends when the switch behind this is on and its animation has run
      for (const s of sources(e.links[0])) work(s.id);
      for (let t = 0; t < 200 && exitHit === null; t += 5) hold(e.x, e.y, 5);
      add('exit', exitHit !== null ? 'pass' : 'fail', `${name(e)}${exitHit !== null ? ' -> the ending' : ': working its switch does not end the game'}`);
      continue;
    }
    hold(e.x, e.y, 3, ['ArrowDown']);
    add('exit', exitHit !== null ? 'pass' : 'fail', `${name(e)}${exitHit !== null ? ` -> level ${exitHit}` : ': action key does not leave the level'}`);
  }

  const count = (v) => rows.filter((r) => r[1] === v).length;
  const kinds = ['chain', 'wake', 'lift', 'teleporter', 'exit'].map((k) => `${rows.filter((r) => r[0] === k).length} ${k}`).join(', ');
  console.log(`level ${String(lv).padStart(2, '0')}: ${String(count('pass')).padStart(3)} pass, ${String(count('fail')).padStart(2)} fail, ${String(count('warn')).padStart(2)} warn   (${kinds})`);
  if (verbose) for (const [k, v, t] of rows) if (v === 'pass') console.log(`    ok    ${k}: ${t}`);
}

console.log(`\n${totals.pass} passed, ${totals.fail} failed, ${totals.warn} warnings`);
if (failures.length) console.log('\n' + failures.join('\n'));
process.exit(totals.fail ? 1 : 0);
