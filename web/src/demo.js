// Demo sweep: an automated 2x-speed playthrough of every level, starting at 0.
// Test harness only — the original has no autopilot (see bot.js notes).
//
// Esc (or the Demo button) stops the sweep; per-level results are logged to the
// console and a summary table is printed when level 21 finishes.

import { Bot } from './bot.js';

export const LEVELS = Array.from({ length: 22 }, (_, i) => `level${String(i).padStart(2, '0')}.spe`);

const TIMEOUT_TICKS = 15 * 90; // 90 sim-seconds (45 real at 2x) per level

let active = null;

export function toggleDemo(game, btn, select) {
  if (active) { stopDemo(false); return; }

  active = {
    game, btn, select,
    idx: 0,
    results: [],
    oldOnLevel: game.onLevel,
    oldNextLevel: game.nextLevel,
  };

  select.disabled = true;
  game.demo = true;
  game.speed = 2;
  game.bot = new Bot();
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
  const ticks = (a.game.tickCount || 0) - (a.game.demoStartTick || 0);
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
  a.btn.textContent = 'Demo: play 0-21 at 2x';
  a.btn.classList.remove('on');
  active = null;
}
