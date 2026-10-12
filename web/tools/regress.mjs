// The regression test: every level that has ever been won must still be won
// by its recorded run.
//
//   node web/tools/regress.mjs           all of them (about half a minute)
//   node web/tools/regress.mjs 3 8       only these levels
//
// The winning runs are the ones in web/data/ppo-demo.json (what the Demo
// button plays). web/tests/won-levels.json lists the levels that have been
// won; tools/search.mjs adds to it and nothing takes a level off it. For each
// level on the list this replays its recording in the real game code, exactly
// as the demo does, and checks that every piece ends where the next begins
// and that the last one leaves by the level's exit (for the last level: that
// it reaches the ending).
//
// A failure means the game no longer plays as it did when the run was
// recorded. If that was not intended, it is a bug. If it was (a fix that
// brings the port closer to the original changes how things move), the level
// has to be searched again before the change is pushed:
//   node web/tools/search.mjs --recheck && node web/tools/search.mjs --parallel
// Exit code 0 when everything passes, 1 otherwise.
import { readFile } from 'node:fs/promises';
import { makeGame, LEVELS } from './headless.mjs';

const only = process.argv.slice(2).filter((a) => /^\d+$/.test(a)).map(Number);
const here = (p) => new URL(p, import.meta.url);
const won = JSON.parse(await readFile(here('../tests/won-levels.json'), 'utf8')).levels;
const demo = JSON.parse(await readFile(here('../data/ppo-demo.json'), 'utf8'));

const g = await makeGame(LEVELS[0]);
const ppo = await import('../src/ppo.js');
g.audio.muted = true;
let exit = null;
g.nextLevel = (n) => { exit = n; g.transitioning = true; };
g.onEndGame = () => { exit = LEVELS.length; };

const t0 = Date.now();
let failed = 0;
for (const lv of won) {
  if (only.length && !only.includes(lv)) continue;
  const pieces = demo.pieces.filter((p) => p.levelIdx === lv);
  const name = `level ${String(lv).padStart(2, '0')}`;
  if (!pieces.length || !demo.passed.includes(lv)) { failed++; console.log(`FAIL  ${name}: it has been won before, but there is no winning run recorded for it now`); continue; }
  let why = null, steps = 0;
  for (let k = 0; k < pieces.length && !why; k++) {
    const piece = pieces[k];
    await ppo.startRun(g, LEVELS[lv], piece.seed || null);
    exit = null;
    const acts = ppo.decodeActs(piece.acts), n = Math.min(piece.steps, acts.length * 4);
    let face = 1, act = 0;
    for (let i = 0; i < n && exit === null && !g.player.dead; i++) {
      if (i % 4 === 0) { act = acts[i / 4]; const mv = (act % 3) - 1; if (mv !== 0) face = mv; }
      ppo.applyAction(g, act, face); g.update(1 / 60); steps++;
    }
    const at = `${Math.round(g.player.x)},${Math.round(g.player.y)}`;
    const nx = pieces[k + 1];
    if (g.player.dead) why = `the player dies in piece ${k + 1} of ${pieces.length}, at ${at}`;
    else if (nx && (Math.round(g.player.x) !== nx.seed.px || Math.round(g.player.y) !== nx.seed.py)) why = `piece ${k + 1} of ${pieces.length} ends at ${at}, not at ${nx.seed.px},${nx.seed.py} where the next begins`;
    else if (!nx && exit !== lv + 1) why = `the last piece ends at ${at} without ${lv + 1 === LEVELS.length ? 'reaching the ending' : `taking the exit to level ${lv + 1}`}`;
  }
  if (why) { failed++; console.log(`FAIL  ${name}: ${why}`); } else console.log(`ok    ${name}  (${pieces.length} pieces, ${Math.round(steps / 60)} s of play)`);
}
const n = only.length ? won.filter((l) => only.includes(l)).length : won.length;
console.log(`\n${n - failed} of ${n} winning runs still win  (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
process.exit(failed ? 1 : 0);
