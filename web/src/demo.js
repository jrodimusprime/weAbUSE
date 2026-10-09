// Demo sweep: an automated 2x-speed playthrough of every level, starting at 0.
// Test harness only — the original has no autopilot (see bot.js notes).
//
// Esc (or the Demo button) stops the sweep; per-level results are logged to the
// console and a summary table is printed when level 21 finishes.

import { Bot } from './bot.js';
import { PpoBot, hasTrainedPolicy, applyAction, actParts, loadBestRun, startRun, decodeActs, campaignPieces } from './ppo.js';

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

// ---- replays of what the PPO trainer recorded ----
//
// A recording is a list of pieces. Each piece is the state it started from
// (or the level's own fresh start), the action taken at each decision and how
// many game steps it ran; started the same way and fed the same actions, the
// game plays it out identically. Consecutive pieces join up exactly, so a list
// of them is one continuous playthrough, across level changes too.

class ReplayBot {
  constructor(piece) { this.acts = decodeActs(piece.acts); this.steps = Math.min(piece.steps, this.acts.length * 4); this.i = 0; this.faceDir = 1; this.act = 0; this.done = null; }
  step(g) {
    if (!g.level || this.done) return;
    if (g.player.dead) { this.done = 'died'; return; }
    if (this.i >= this.steps) { this.done = 'end'; return; }
    if (this.i % 4 === 0) {
      this.act = this.acts[this.i / 4];
      const { move } = actParts(this.act);
      if (move !== 0) this.faceDir = move;
    }
    applyAction(g, this.act, this.faceDir);
    this.i++;
  }
}

function playPieces(game, btn, select, pieces, title) {
  if (active) { stopDemo(false); return; }
  if (!pieces.length) { game.toast('Nothing recorded yet — train first'); return; }
  active = {
    game, btn, select,
    oldOnLevel: game.onLevel,
    oldNextLevel: game.nextLevel,
    btnBaseText: btn.textContent,
  };
  const mine = active;
  select.disabled = true;
  game.demo = true;
  game.speed = 2;
  game.demoTimeoutTicks = 0;
  game.onDemoStop = () => stopDemo(false);
  game.onLevel = (name) => { mine.oldOnLevel?.(name); if (active === mine) game.toast(`${title} (2x): ${name}`); };
  let k = -1, exited = false;
  const finish = (text) => {
    if (active !== mine) return;
    stopDemo(false);
    game.toast(text);
    console.log(`[replay] ${text}`);
  };
  // An exit ends the piece it is in; the next piece starts the next level.
  game.nextLevel = () => { game.transitioning = true; exited = true; };
  const next = () => {
    if (active !== mine) return;
    if (++k >= pieces.length) { finish(`${title} finished on level ${pieces[pieces.length - 1].levelIdx}`); return; }
    const piece = pieces[k];
    exited = false;
    game.hold = true; // no steps until the piece is set up exactly as it was recorded
    game.bot = null;
    startRun(game, LEVELS[piece.levelIdx], piece.seed || null).then(() => {
      if (active !== mine) return;
      const bot = new ReplayBot(piece);
      // the bot is stepped inside each game update; move on once it has run out
      game.bot = { step: (g) => { bot.step(g); if (bot.done === 'died') finish(`${title} stopped: the run died on level ${piece.levelIdx}`); else if (bot.done || exited) { game.hold = true; game.bot = null; queueMicrotask(next); } } };
      game.hold = false;
    }, () => finish(`${title} stopped: level failed to load`));
  };
  console.log(`[replay] ${title}: ${pieces.length} piece(s), levels ${[...new Set(pieces.map((p) => p.levelIdx))].join(' -> ')}`);
  btn.textContent = 'Replay: running (Esc stops)';
  btn.classList.add('on');
  next();
}

// The best run on the current level: from the level's start to its best point.
export function replayBestRun(game, btn, select, recOverride = null) {
  const rec = recOverride || loadBestRun();
  if (!active && !rec) { game.toast('No PPO best run saved yet — train first'); return; }
  playPieces(game, btn, select, rec ? rec.pieces.map((p) => ({ ...p, levelIdx: rec.levelIdx })) : [], 'Best run');
}

// Everything recorded so far as one playthrough: every level passed, in
// order, then the best progress on the level the agent is on now.
export function replayCampaign(game, btn, select) {
  playPieces(game, btn, select, active ? [] : campaignPieces(), 'Full game');
}
