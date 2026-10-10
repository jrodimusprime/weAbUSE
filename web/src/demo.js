// Demo: plays back a recorded playthrough.
//
// A recording is a list of pieces. Each piece is the state it started from
// (or the level's own fresh start), the action taken at each decision and how
// many game steps it ran; started the same way and fed the same actions, the
// game plays it out identically. Consecutive pieces join up exactly, so a list
// of them is one continuous playthrough, across level changes too.
//
// The recording comes from data/ppo-demo.json, committed with the site: the
// route tools/search.mjs found through each level. (The PPO trainer writes
// the same format; without the file, whatever this browser trained is played.)

import { applyAction, actParts, startRun, decodeActs, campaignPieces, DEMO_FORMAT } from './ppo.js';
import { DATA } from './assets.js';

export const LEVELS = Array.from({ length: 22 }, (_, i) => `level${String(i).padStart(2, '0')}.spe`);

let active = null;

export function stopDemo() {
  const a = active;
  if (!a) return;
  const g = a.game;
  g.demo = false;
  g.hold = false;
  g.speed = 1;
  g.bot = null;
  // The bot drives game.keys/mouseDown; drop its last inputs so the player
  // doesn't keep walking in the bot's final direction.
  g.keys.clear();
  g.mouseDown = false;
  g.rightDown = false;
  g.nextLevel = a.oldNextLevel;
  g.onEndGame = a.oldOnEndGame;
  g.onLevel = a.oldOnLevel;
  g.onDemoStop = null;
  a.select.disabled = false;
  a.btn.textContent = a.btnBaseText;
  a.btn.classList.remove('on');
  active = null;
}

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

function playPieces(game, btn, select, pieces) {
  active = {
    game, btn, select,
    oldOnLevel: game.onLevel,
    oldNextLevel: game.nextLevel,
    btnBaseText: btn.textContent,
  };
  const mine = active;
  select.disabled = true;
  game.demo = true; // the keyboard is ignored; Esc stops
  game.speed = 2;
  game.demoTimeoutTicks = 0;
  game.onDemoStop = () => stopDemo();
  game.onLevel = (name) => { mine.oldOnLevel?.(name); if (active === mine) game.toast(`Demo (2x): ${name}`); };
  let k = -1, exited = false;
  const finish = (text) => {
    if (active !== mine) return;
    stopDemo();
    game.toast(text);
    console.log(`[demo] ${text}`);
  };
  // An exit ends the piece it is in; the next piece starts the next level.
  game.nextLevel = () => { game.transitioning = true; exited = true; };
  // (the ending, reached in a recording, ends its piece like an exit; the real ending is shown when the demo stops)
  mine.oldOnEndGame = game.onEndGame;
  game.onEndGame = () => { exited = true; };
  const next = () => {
    if (active !== mine) return;
    if (++k >= pieces.length) { finish(`Demo finished on level ${pieces[pieces.length - 1].levelIdx}: that is as far as the recording goes`); return; }
    const piece = pieces[k];
    exited = false;
    game.hold = true; // no steps until the piece is set up exactly as it was recorded
    game.bot = null;
    startRun(game, LEVELS[piece.levelIdx], piece.seed || null).then(() => {
      if (active !== mine) return;
      const bot = new ReplayBot(piece);
      // the bot is stepped inside each game update; move on once it has run out
      game.bot = { step: (g) => { bot.step(g); if (bot.done === 'died') finish(`Demo stopped: the run died on level ${piece.levelIdx}`); else if (bot.done || exited) { game.hold = true; game.bot = null; queueMicrotask(next); } } };
      game.hold = false;
    }, () => finish('Demo stopped: level failed to load'));
  };
  console.log(`[demo] ${pieces.length} piece(s), levels ${[...new Set(pieces.map((p) => p.levelIdx))].join(' -> ')}`);
  btn.textContent = 'Demo: running (Esc stops)';
  btn.classList.add('on');
  next();
}

// The committed recording, or null if there is none.
export async function loadDemoFile() {
  try {
    const r = await fetch(`${DATA}ppo-demo.json`, { cache: 'no-store' });
    if (!r.ok) return null;
    const d = await r.json();
    return d && d.format === DEMO_FORMAT && Array.isArray(d.pieces) && d.pieces.length ? d : null;
  } catch { return null; }
}

// Starts the demo, or stops it if it is running.
export async function toggleFullGameDemo(game, btn, select) {
  if (active) { stopDemo(); return; }
  const file = await loadDemoFile();
  const pieces = file ? file.pieces : campaignPieces();
  if (!pieces.length) { game.toast('No demo recording yet'); return; }
  if (active) return; // started twice while the file loaded
  playPieces(game, btn, select, pieces);
}
