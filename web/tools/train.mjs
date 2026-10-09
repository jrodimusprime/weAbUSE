// Trains the PPO agent without a browser, on several CPU cores at once, and
// reports how far it gets and where its runs end.
//
//   node web/tools/train.mjs [--minutes 30] [--workers 6] [--out train-out] [--fresh]
//
// Each worker is the same trainer the page runs (same game code, rewards and
// learning), playing its own copy of the game. After every policy update the
// workers' networks are averaged and handed back, so they learn as one. They
// also share the campaign (the levels passed so far: as soon as one worker
// passes a level, all of them move on to the next), the save stations reached
// on the current level (the places runs may start from), the best run there,
// and the exploration counts.
//
// Results:
//   web/data/ppo-demo.json  the recorded playthrough (every level passed, then
//                           the best progress on the current one). The page's
//                           "Full game demo" button plays it; commit it to
//                           publish the demo with the site.
//   <out>/ppo.json          everything needed to carry on training: resumed
//                           from automatically (--fresh starts over), or loaded
//                           into the page with "Load training file".
//   <out>/report.txt        how far it got and where its runs end.
//
//   node web/tools/train.mjs --export    rewrites web/data/ppo-demo.json from
//                                        <out>/ppo.json without training.
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { availableParallelism } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { makeGame, LEVELS } from './headless.mjs';

const ppo = await import('../src/ppo.js');
const { PPO_KEY, BEST_KEY, HIST_KEY, VISITS_KEY, CAMPAIGN_KEY, STATIONS_KEY } = ppo;
const levelOfRec = (rec) => (rec ? (rec.levelIdx ?? LEVELS.indexOf(rec.level)) : -1);
const DEMO_FILE = fileURLToPath(new URL('../data/ppo-demo.json', import.meta.url));

// The demo recording, from a saved store (the campaign plus the best run on the current level).
async function writeDemo(store) {
  const campaign = store[CAMPAIGN_KEY] ? JSON.parse(store[CAMPAIGN_KEY]) : { frontier: 0, entry: null, legs: [] };
  const rec = store[BEST_KEY] ? JSON.parse(store[BEST_KEY]) : null;
  const demo = ppo.demoFile(campaign, rec);
  if (!demo.pieces.length) return null; // nothing recorded yet: leave any existing demo alone
  await writeFile(DEMO_FILE, JSON.stringify(demo));
  return demo;
}

if (isMainThread) await main(); else await worker();

// ---------------------------------------------------------------- worker

async function worker() {
  const { id, seedStore } = workerData;
  for (const [k, v] of Object.entries(seedStore)) if (v != null) localStorage.setItem(k, v);
  const g = await makeGame();
  console.log = () => {}; // the trainer's own progress lines; the main thread reports instead
  const tr = new ppo.PpoTrainer(g, LEVELS, () => {});
  tr.showEvery = 0; // nothing is shown
  let round = { decisions: 0, attempts: [], cleared: 0 };
  let synced = new Map(tr.visits); // exploration counts as of the last sync
  let waiting = false, stop = false, lastBest = localStorage.getItem(BEST_KEY), passed = false;
  tr.onCampaign = () => { passed = true; };
  let newStation = false;
  tr.onStations = () => { newStation = true; };

  const policyStep = tr.policyStep.bind(tr);
  tr.policyStep = () => { round.decisions++; policyStep(); };
  tr.onAttempt = (a) => round.attempts.push({ start: a.startIdx, lvl: a.endIdx, x: a.end[0], y: a.end[1], how: a.how.replace(/ on level \d+$/, ''), cleared: a.cleared, secs: Math.round(a.steps / 60) });

  // after each policy update: report, then wait for the averaged network
  tr.onUpdateDone = () => {
    const visits = [];
    for (const [k, n] of tr.visits) { const d = n - (synced.get(k) || 0); if (d > 0) visits.push([k, d]); }
    const best = localStorage.getItem(BEST_KEY);
    const weights = tr.net.weights().map((w) => w.slice());
    parentPort.postMessage({ type: 'update', id, weights, round, visits, best: best !== lastBest ? best : null, campaign: passed ? localStorage.getItem(CAMPAIGN_KEY) : null, stations: newStation ? localStorage.getItem(STATIONS_KEY) : null }, weights.map((w) => w.buffer));
    lastBest = best; passed = false; newStation = false;
    round = { decisions: 0, attempts: [], cleared: 0 };
    waiting = true;
  };
  parentPort.on('message', (m) => {
    if (m.type === 'stop') { stop = true; return; }
    if (m.type !== 'sync') return;
    tr.net.setWeights(m.weights);
    for (const [k, n] of m.visits) { tr.visits.set(k, n); synced.set(k, n); }
    if (m.campaign) {
      // a level was passed (by this worker or another): everyone moves on
      ppo.saveCampaign(JSON.parse(m.campaign));
      localStorage.removeItem(BEST_KEY); localStorage.removeItem(HIST_KEY);
      ppo.saveStations(JSON.parse(m.campaign).frontier, []);
      lastBest = null; passed = false; newStation = false;
    }
    if (m.stations) { const st = JSON.parse(m.stations); ppo.saveStations(st.levelIdx, st.list); }
    if (m.best) { localStorage.setItem(BEST_KEY, m.best); lastBest = m.best; }
    waiting = false;
  });

  await tr.start();
  clearInterval(tr.timer); // no 16 ms ticks: this loop runs the trainer flat out
  while (!stop) {
    if (!waiting) { tr.lastFrameAt = performance.now() - 1000; tr.frame(); } // a long gap = the big work budget
    if (!tr.running) throw new Error('trainer stopped itself (see the error above)');
    await new Promise((r) => setImmediate(r)); // let level loads and messages through
  }
  process.exit(0);
}

// ---------------------------------------------------------------- main

async function main() {
  const arg = (name, dflt) => { const i = process.argv.indexOf(`--${name}`); return i < 0 ? dflt : process.argv[i + 1]; };
  const minutes = +arg('minutes', 30);
  const nWorkers = Math.max(1, +arg('workers', Math.max(1, Math.min(6, availableParallelism() - 2))));
  const outDir = path.resolve(arg('out', 'train-out'));
  const fresh = process.argv.includes('--fresh');
  await mkdir(outDir, { recursive: true });
  const outFile = path.join(outDir, 'ppo.json');
  if (process.argv.includes('--export')) {
    const demo = await writeDemo(JSON.parse(await readFile(outFile, 'utf8')).store);
    console.log(demo ? `wrote ${DEMO_FILE}: levels passed ${demo.passed.join(', ') || 'none'}, now on level ${demo.frontier}, ${demo.pieces.length} piece(s)` : 'nothing recorded in that training file');
    return;
  }

  // resume
  let seedStore = {};
  if (!fresh) { try { seedStore = JSON.parse(await readFile(outFile, 'utf8')).store || {}; } catch { /* nothing to resume */ } }
  for (const [k, v] of Object.entries(seedStore)) if (v != null) localStorage.setItem(k, v);
  const net = new ppo.Net();
  const resumed = net.load();
  if (!resumed) { net.save(); seedStore[PPO_KEY] = localStorage.getItem(PPO_KEY); } // every worker starts from the same network
  let best = localStorage.getItem(BEST_KEY);
  let hist = ppo.loadBestHistory();
  let campaign = localStorage.getItem(CAMPAIGN_KEY);
  const frontier = () => (campaign ? JSON.parse(campaign).frontier : 0);
  let stations = (() => { try { const st = JSON.parse(localStorage.getItem(STATIONS_KEY)); return st && st.levelIdx === frontier() ? st.list : []; } catch { return []; } })();
  const visits = new Map(JSON.parse(localStorage.getItem(VISITS_KEY) || '[]'));

  const t0 = performance.now();
  const total = { decisions: 0, runs: 0, cleared: 0, rounds: 0 };
  const ends = new Map();       // "level|cellx|celly|how" -> count: where runs end
  const reached = new Map();    // level -> runs that got there

  console.log(`training on ${nWorkers} workers for ${minutes} min (${resumed ? 'resuming' : 'fresh'}) -> ${outFile}`);
  const workers = [];
  let pending = new Map(); // worker id -> its update, until all have reported
  let stopping = false;

  const save = async () => {
    net.save();
    const store = { [PPO_KEY]: localStorage.getItem(PPO_KEY), [BEST_KEY]: best, [HIST_KEY]: JSON.stringify(hist), [VISITS_KEY]: JSON.stringify([...visits]), [CAMPAIGN_KEY]: campaign, [STATIONS_KEY]: JSON.stringify({ levelIdx: frontier(), list: stations }) };
    await writeFile(outFile, JSON.stringify({ format: 'abuse-ppo-train-1', savedAt: new Date().toISOString(), store }));
    await writeDemo(store);
    await writeFile(path.join(outDir, 'report.txt'), report());
  };

  const report = () => {
    const mins = (performance.now() - t0) / 60000;
    const rec = best ? JSON.parse(best) : null;
    const lines = [];
    lines.push(`PPO training report — ${new Date().toISOString()}`);
    lines.push(`${mins.toFixed(1)} min on ${nWorkers} workers · ${total.decisions.toLocaleString()} decisions (${Math.round(total.decisions / (mins * 60))}/s, ${(total.decisions / 15 / 3600).toFixed(1)} game-hours) · ${total.rounds} policy updates · ${total.runs} runs`);
    const camp = campaign ? JSON.parse(campaign) : { frontier: 0, legs: [] };
    lines.push(camp.legs.length
      ? `Levels passed: ${camp.legs.map((l) => `${l.level}→${l.dest} (${Math.round(l.pieces.reduce((a, p) => a + p.steps, 0) / 60)}s of play)`).join(', ')}. Now on level ${camp.frontier}.`
      : 'Levels passed: none yet. Still on level 0.');
    lines.push(stations.length ? `Save stations reached on level ${camp.frontier} (runs start from these or the level's start): ${stations.map((st) => `(${st.key}, ${st.sw || 0} switch${st.sw === 1 ? '' : 'es'} on)`).join(' ')}` : `Save stations reached on level ${camp.frontier}: none yet, so every run starts at the level's start.`);
    lines.push(rec ? `Best on level ${camp.frontier}: ${ppo.fmtDist(rec.dist)} from its exit, at (${rec.end}).` : `Best on level ${camp.frontier}: nothing recorded yet.`);
    lines.push('');
    lines.push('Runs played on each level:');
    for (const lvl of [...reached.keys()].sort((a, b) => a - b)) lines.push(`  level ${String(lvl).padStart(2)}: ${reached.get(lvl)}`);
    lines.push('');
    lines.push('Where runs end (120 px cells; the busiest spots are where it is blocked):');
    const byLevel = new Map();
    for (const [k, n] of ends) { const [lvl, cx, cy, how] = k.split('|'); if (!byLevel.has(lvl)) byLevel.set(lvl, []); byLevel.get(lvl).push({ x: cx * 120 + 60, y: cy * 120 + 60, how, n }); }
    for (const lvl of [...byLevel.keys()].sort((a, b) => a - b)) {
      const list = byLevel.get(lvl).sort((a, b) => b.n - a.n);
      const sum = list.reduce((a, e) => a + e.n, 0);
      lines.push(`  level ${lvl} (${sum} runs ended here):`);
      for (const e of list.slice(0, 8)) lines.push(`    ${String(Math.round(100 * e.n / sum)).padStart(3)}%  around x=${e.x}, y=${e.y}  (${e.how})`);
    }
    return lines.join('\n') + '\n';
  };

  const finishRound = () => {
    const ups = [...pending.values()];
    pending = new Map();
    // one network: the average of the workers'
    const avg = ups[0].weights.map((w) => new Float32Array(w.length));
    for (const u of ups) u.weights.forEach((w, i) => { const a = avg[i]; for (let j = 0; j < w.length; j++) a[j] += w[j]; });
    for (const a of avg) for (let j = 0; j < a.length; j++) a[j] /= ups.length;
    net.setWeights(avg);
    // shared exploration counts: only the cells that changed go back out
    const changed = new Map();
    let newBest = null, sendCampaign = false;
    // a level passed: the furthest destination wins if two workers got through at once
    for (const u of ups) {
      if (!u.campaign) continue;
      sendCampaign = true; // also puts right a worker whose pass came second
      const c = JSON.parse(u.campaign);
      if (c.frontier > frontier()) {
        campaign = u.campaign; best = null; hist = []; stations = [];
        console.log(`  ** passed level ${c.legs[c.legs.length - 1].level} -> now on level ${c.frontier} (${((performance.now() - t0) / 60000).toFixed(1)} min)`);
      }
    }
    // save stations: the first worker to use one adds it for everybody
    let sendStations = false;
    for (const u of ups) {
      if (!u.stations) continue;
      const st = JSON.parse(u.stations);
      if (st.levelIdx !== frontier()) continue;
      for (const one of st.list) {
        const i = stations.findIndex((x) => x.key === one.key);
        if (i < 0) { stations.push(one); sendStations = true; console.log(`  ** save station (${one.key}) reached on level ${frontier()} (${((performance.now() - t0) / 60000).toFixed(1)} min)`); }
        else if ((one.sw || 0) > (stations[i].sw || 0)) { stations[i] = one; sendStations = true; console.log(`  ** save station (${one.key}) saved again with ${one.sw} switch(es) on (${((performance.now() - t0) / 60000).toFixed(1)} min)`); }
      }
    }
    for (const u of ups) {
      total.decisions += u.round.decisions;
      for (const [k, d] of u.visits) { visits.set(k, (visits.get(k) || 0) + d); changed.set(k, visits.get(k)); }
      for (const a of u.round.attempts) {
        total.runs++;
        total.cleared += a.cleared;
        reached.set(a.lvl, (reached.get(a.lvl) || 0) + 1);
        const key = `${a.lvl}|${Math.floor(a.x / 120)}|${Math.floor(a.y / 120)}|${a.how}`;
        ends.set(key, (ends.get(key) || 0) + 1);
      }
      if (u.best && levelOfRec(JSON.parse(u.best)) === frontier()) { // not a record from a level since passed
        const rec = JSON.parse(u.best), cur = best ? JSON.parse(best) : null;
        if (!cur || ppo.betterRun(levelOfRec(rec), rec.dist, levelOfRec(cur), cur.dist)) { best = u.best; newBest = u.best; }
        if (!hist.some((h) => h.level === rec.level && h.dist === rec.dist)) {
          hist.push(rec);
          hist.sort((a, b) => (ppo.betterRun(levelOfRec(a), a.dist, levelOfRec(b), b.dist) ? -1 : 1));
          hist.length = Math.min(hist.length, 3);
        }
      }
    }
    total.rounds++;
    const msg = { type: 'sync', weights: avg, visits: [...changed], best: sendCampaign ? best : newBest, campaign: sendCampaign ? campaign : null, stations: sendStations ? JSON.stringify({ levelIdx: frontier(), list: stations }) : null };
    for (const w of workers) w.postMessage(msg);
  };

  for (let id = 0; id < nWorkers; id++) {
    const w = new Worker(new URL(import.meta.url), { workerData: { id, seedStore } });
    w.on('message', (m) => { if (m.type !== 'update' || stopping) return; pending.set(m.id, m); if (pending.size === nWorkers) finishRound(); });
    w.on('error', (e) => { console.error(`worker ${id} failed:`, e); process.exitCode = 1; stopping = true; });
    workers.push(w);
  }

  let lastLine = 0, lastSave = performance.now();
  await new Promise((resolve) => {
    const iv = setInterval(async () => {
      const now = performance.now(), mins = (now - t0) / 60000;
      if (now - lastLine > 30000) {
        lastLine = now;
        const rec = best ? JSON.parse(best) : null;
        console.log(`${mins.toFixed(1).padStart(5)} min · ${Math.round(total.decisions / Math.max(1, (now - t0) / 1000))} decisions/s · ${total.runs} runs · on level ${frontier()} · best there: ${rec ? `${ppo.fmtDist(rec.dist)} to exit` : 'none yet'}`);
      }
      if (now - lastSave > 120000) { lastSave = now; await save(); }
      if (mins >= minutes || stopping) { clearInterval(iv); resolve(); }
    }, 1000);
    process.on('SIGINT', () => { console.log('\nstopping…'); stopping = true; });
  });
  stopping = true;
  for (const w of workers) w.postMessage({ type: 'stop' });
  await save();
  console.log('\n' + report());
  console.log(`saved ${outFile} (resume from it, or "Load training file" in the page)\nsaved ${DEMO_FILE} (what "Full game demo" plays: commit it to publish)`);
  for (const w of workers) await w.terminate();
  process.exit(process.exitCode || 0);
}
