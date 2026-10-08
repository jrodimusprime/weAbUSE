// Demo sweep: an automated 2x-speed playthrough of every level, starting at 0.
// Test harness only — the original has no autopilot (see bot.js notes).
//
// Esc (or the Demo button) stops the sweep; per-level results are logged to the
// console and a summary table is printed when level 21 finishes.

import { Bot } from './bot.js';
import { PpoBot, hasTrainedPolicy, applyAction, actParts, loadBestRun } from './ppo.js';

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

// ---- best-run replay: plays back the PPO trainer's saved best action script ----

class ReplayBot {
  constructor(acts) { this.acts = acts; this.i = 0; this.faceDir = 1; this.ended = false; }
  step(g) {
    if (!g.level || this.ended) return;
    if (g.player.dead) { this.ended = true; g.onDemoTimeout?.(); return; } // death desyncs the script
    const k = Math.min(Math.floor(this.i / 4), this.acts.length - 1);
    const act = this.acts[k];
    const { move } = actParts(act);
    if (move !== 0) this.faceDir = move;
    applyAction(g, act, this.faceDir);
    this.i++;
    if (k === this.acts.length - 1 && this.i > this.acts.length * 4 + 16) {
      this.ended = true;
      g.onDemoTimeout?.(); // script exhausted without reaching the exit
    }
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
  select.disabled = true;
  game.demo = true;
  game.speed = 2;
  game.keys.clear();
  game.mouseDown = false;
  game.rightDown = false;
  game.bot = new ReplayBot(rec.acts);
  game.demoTimeoutTicks = 0;
  game.nextLevel = () => {
    if (game.transitioning) return;
    game.transitioning = true;
    stopDemo(false);
    game.toast('Replay finished — reached the exit!');
    console.log(`[replay] reached the exit on ${rec.level}`);
  };
  game.onDemoTimeout = () => {
    if (game.transitioning) return;
    game.transitioning = true;
    stopDemo(false);
    game.toast('Replay finished (script ended or death)');
  };
  game.onDemoStop = () => stopDemo(false);
  game.onLevel = (name) => {
    active?.oldOnLevel?.(name);
    if (!active) return;
    game.toast(`Replay (2x): ${name}`);
    console.log(`[replay] ${name} — best-run script (dist ${rec.dist}, ${rec.acts.length} actions)`);
  };
  console.log(`[replay] best run on ${rec.level} (dist ${rec.dist}, ${rec.acts.length} actions)`);
  btn.textContent = 'Replay: running (Esc stops)';
  btn.classList.add('on');
  game.start(rec.level);
}
