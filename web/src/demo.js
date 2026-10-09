// Demo sweep: an automated 2x-speed playthrough of every level, starting at 0.
// Test harness only — the original has no autopilot (see bot.js notes).
//
// Esc (or the Demo button) stops the sweep; per-level results are logged to the
// console and a summary table is printed when level 21 finishes.

import { Bot } from './bot.js';
import { PpoBot, hasTrainedPolicy, applyAction, actParts, loadBestRun, startRun, fmtDist } from './ppo.js';

export const LEVELS = Array.from({ length: 22 }, (_, i) => `level${String(i).padStart(2, '0')}.spe`);

// Stall budget: the bot resets game.demoStartTick whenever it moves to a new
// 40px cell, so a level only force-advances after this long with NO progress.
const TIMEOUT_TICKS = 15 * 90; // 90 sim-seconds of no progress (45 real at 2x)

let active = null;

export function toggleDemo(game, btn, select) {
  if (active) { stopDemo(false); return; }

  active = {
    game, btn, select,
    idx: 0,
    results: [],
    oldOnLevel: game.onLevel,
    oldNextLevel: game.nextLevel,
    btnBaseText: btn.textContent,
  };

  select.disabled = true;
  game.demo = true;
  game.speed = 2;
  game.keys.clear();
  game.mouseDown = false;
  game.rightDown = false;
  game.bot = hasTrainedPolicy() ? new PpoBot() : new Bot();
  game.demoTimeoutTicks = TIMEOUT_TICKS;

  const finishAndNext = (result) => {
    finishLevel(result);
    active.idx++;
    advance();
  };
  game.nextLevel = () => {
    if (game.transitioning) return;
    game.transitioning = true;
    finishAndNext('completed');
  };
  game.onDemoTimeout = () => {
    if (game.transitioning) return;
    game.transitioning = true;
    finishAndNext('timeout');
  };
  game.onDemoStop = () => stopDemo(false);
  game.onLevel = (name) => {
    active?.oldOnLevel?.(name);
    if (!active) return;
    active.levelStartTick = game.tickCount || 0;
    game.demoStartTick = game.tickCount || 0;
    game.toast(`Demo (2x): ${name}`);
    console.log(`[demo] ${name} started`);
  };

  console.log('[demo] sweep started at 2x from level00 (Esc or Demo button stops)');
  btn.textContent = 'Demo: running (Esc stops)';
  btn.classList.add('on');
  advance();
}

function finishLevel(result) {
  const a = active;
  if (!a) return;
  const ticks = (a.game.tickCount || 0) - (a.levelStartTick || 0);
  const sim = Math.round(ticks / 15);
  a.results.push({ level: a.idx, result, simSeconds: sim });
  console.log(`[demo] level${String(a.idx).padStart(2, '0')} ${result} after ${sim}s sim`);
}

function advance() {
  const a = active;
  if (!a) return;
  if (a.idx > 21) { stopDemo(true); return; }
  a.game.start(LEVELS[a.idx]);
}

export function stopDemo(finished = false) {
  const a = active;
  if (!a) return;
  const g = a.game;
  g.demo = false;
  g.hold = false;
  g.speed = 1;
  g.bot = null;
  g.demoTimeoutTicks = 0;
  // The bot drives game.keys/mouseDown; drop its last inputs so the player
  // doesn't keep walking in the bot's final direction.
  g.keys.clear();
  g.mouseDown = false;
  g.rightDown = false;
  g.nextLevel = a.oldNextLevel;
  g.onLevel = a.oldOnLevel;
  g.onDemoStop = null;
  g.onDemoTimeout = null;
  a.select.disabled = false;
  if (finished) {
    console.log('[demo] sweep finished');
    console.table(a.results);
    g.toast('Demo sweep finished — see console');
  } else {
    console.log('[demo] sweep stopped');
    g.toast('Demo stopped');
  }
  a.btn.textContent = a.btnBaseText || 'Demo: play 0-21 at 2x';
  a.btn.classList.remove('on');
  active = null;
}

// ---- best-run replay: plays back a run saved by the PPO trainer ----
//
// A saved run is where it started (level, and a checkpoint if it resumed from
// one) plus the inputs it was given. Started the same way and fed the same
// inputs, the game plays out identically, across level changes too.

class ReplayBot {
  constructor(rec) { this.acts = rec.acts; this.steps = Math.min(rec.steps ?? rec.acts.length * 4, rec.acts.length * 4); this.i = 0; this.faceDir = 1; this.ended = false; }
  step(g) {
    if (!g.level || this.ended) return;
    if (g.player.dead || this.i >= this.steps) { this.ended = true; g.onDemoTimeout?.(); return; }
    if (this.i % 4 === 0) {
      this.act = this.acts[this.i / 4];
      const { move } = actParts(this.act);
      if (move !== 0) this.faceDir = move;
    }
    applyAction(g, this.act, this.faceDir);
    this.i++;
  }
}

export function replayBestRun(game, btn, select, recOverride = null) {
  if (active) { stopDemo(false); return; }
  const rec = recOverride || loadBestRun();
  if (!rec) { game.toast('No PPO best run saved yet — train first'); return; }

  active = {
    game, btn, select,
    oldOnLevel: game.onLevel,
    oldNextLevel: game.nextLevel,
    btnBaseText: btn.textContent,
  };
  const mine = active;
  const startIdx = rec.start?.idx ?? Math.max(0, LEVELS.indexOf(rec.level));
  let lvl = startIdx;
  select.disabled = true;
  game.demo = true;
  game.speed = 2;
  game.demoTimeoutTicks = 0;
  const finish = (text) => {
    if (active !== mine) return;
    game.hold = false;
    stopDemo(false);
    game.toast(text);
    console.log(`[replay] ${text}`);
  };
  // exits behave as they did in the recorded run: on to a later level, or the end
  game.nextLevel = (dest) => {
    if (game.transitioning) return;
    game.transitioning = true;
    if (dest > lvl && dest < LEVELS.length) {
      game.hold = true;
      game.start(LEVELS[dest]).then(() => { lvl = dest; if (active === mine) game.hold = false; }, () => finish('Replay stopped: level failed to load'));
    } else finish('Replay finished — took the last exit of the run');
  };
  game.onDemoTimeout = () => finish(`Replay finished on level ${lvl}`);
  game.onDemoStop = () => { game.hold = false; stopDemo(false); };
  game.onLevel = (name) => {
    mine.oldOnLevel?.(name);
    if (active === mine) game.toast(`Replay (2x): ${name}`);
  };
  console.log(`[replay] best run: started on level ${startIdx}${rec.start?.seed ? ' from a checkpoint' : ''}, reached ${rec.level} (${fmtDist(rec.dist)} from its exit), ${rec.acts.length} actions`);
  btn.textContent = 'Replay: running (Esc stops)';
  btn.classList.add('on');
  // no steps until the run has been set up exactly as it was recorded
  game.hold = true;
  game.bot = null;
  startRun(game, LEVELS[startIdx], rec.start?.seed || null).then(() => {
    if (active !== mine) return;
    game.bot = new ReplayBot(rec);
    game.hold = false;
  }, () => finish('Replay stopped: level failed to load'));
}
