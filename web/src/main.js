import { Game } from './game.js';
import { WEAPON_ORDER } from './weapons.js';
import { readPalette, T } from './spec.js';
import { LEVELS, toggleFullGameDemo, loadDemoFile, stopDemo } from './demo.js';
import { PpoTrainer, loadCampaign, saveCampaign, saveStations, forgetTraining, CAMPAIGN_KEY, STATIONS_KEY } from './ppo.js';

const select = document.getElementById('level');
for (const l of LEVELS) select.add(new Option(l, l));

const godBtn = document.getElementById('god');
const game = new Game(document.getElementById('c'), document.getElementById('hud'));
window.game = game;

// PPO trainer: trains a policy in the page. Training runs unseen; every so
// many policy updates the latest attempt is replayed, then training carries on.
const ppoBtn = document.getElementById('ppo');
const ppoStatus = document.getElementById('ppoStatus');
const pauseBtn = document.getElementById('pause');
const fullGameBtn = document.getElementById('fullGame');
const forgetBtn = document.getElementById('forget');
const showEverySel = document.getElementById('showEvery');
let ppoTrainer = null;
showEverySel.addEventListener('change', () => { if (ppoTrainer) ppoTrainer.showEvery = +showEverySel.value; showEverySel.blur(); });
const trainingActive = () => !!(ppoTrainer?.running || ppoTrainer?.paused);
const stopTraining = () => { if (trainingActive()) ppoTrainer.stop(); ppoBtn.textContent = 'Train PPO'; refreshPauseBtn(); };
const refreshPauseBtn = () => {
  pauseBtn.disabled = !trainingActive();
  pauseBtn.textContent = ppoTrainer?.paused ? 'Resume' : 'Pause';
};
const startTraining = () => {
  stopDemo();
  ppoTrainer = new PpoTrainer(game, LEVELS, (s) => {
    ppoStatus.textContent = s;
    refreshPauseBtn();
    if (!trainingActive()) ppoBtn.textContent = 'Train PPO';
  });
  ppoTrainer.showEvery = +showEverySel.value;
  ppoBtn.textContent = 'Stop training';
  ppoTrainer.start();
  refreshPauseBtn();
};
refreshPauseBtn();
ppoBtn.addEventListener('click', () => {
  if (trainingActive()) stopTraining(); else startTraining();
  ppoBtn.blur();
});
pauseBtn.addEventListener('click', () => {
  if (!trainingActive()) return;
  if (ppoTrainer.paused) ppoTrainer.resume();
  else ppoTrainer.pause();
  refreshPauseBtn();
});

// Demo: the recorded playthrough committed with the site (data/ppo-demo.json,
// found by tools/search.mjs), every level passed so far, in order.
const describeDemo = async () => {
  const file = await loadDemoFile();
  const passed = file ? file.passed.length : loadCampaign().legs.length;
  const on = file ? file.frontier : loadCampaign().frontier;
  void on;
  const levels = file ? file.passed : loadCampaign().legs.map((l) => l.level);
  if (!fullGameBtn.classList.contains('on')) {
    fullGameBtn.textContent = !passed ? 'Demo' : passed === 1 ? `Demo (level ${levels[0]})` : `Demo (levels ${levels[0]} to ${levels[levels.length - 1]})`;
  }
};
describeDemo();
fullGameBtn.addEventListener('click', async () => {
  stopTraining();
  fullGameBtn.blur();
  await toggleFullGameDemo(game, fullGameBtn, select);
});

// Forget training: wipes what this browser has learned and recorded, so the
// next "Train PPO" starts from nothing, on level 0. The committed demo
// recording is a file in the repository and is not touched.
forgetBtn.addEventListener('click', () => {
  forgetBtn.blur();
  if (!confirm('Forget all PPO training stored in this browser (policy, levels passed, save stations, exploration)?')) return;
  stopTraining();
  stopDemo();
  ppoTrainer = null;
  forgetTraining();
  ppoStatus.textContent = 'training forgotten — the next run starts from scratch on level 0';
  describeDemo();
});

// Results of headless training (node web/tools/train.mjs writes train-out/ppo.json):
// the trained policy, the levels passed, save stations and exploration counts.
// Loading one replaces what this browser has stored, so training can carry on
// here from where the headless run left off.
const loadTrainBtn = document.getElementById('loadTrain');
const loadTrainFile = document.getElementById('loadTrainFile');
loadTrainBtn.addEventListener('click', () => { loadTrainFile.click(); loadTrainBtn.blur(); });
loadTrainFile.addEventListener('change', async () => {
  const file = loadTrainFile.files[0];
  loadTrainFile.value = '';
  if (!file) return;
  try {
    const data = JSON.parse(await file.text());
    if (data.format !== 'abuse-ppo-train-1' || !data.store) throw new Error('not a training file');
    stopTraining();
    forgetTraining();
    for (const [key, value] of Object.entries(data.store)) {
      if (!key.startsWith('abuse.ppo.') || value == null) continue;
      // these can be too big for browser storage: they are then kept for this session only
      if (key === CAMPAIGN_KEY) { saveCampaign(JSON.parse(value)); continue; }
      if (key === STATIONS_KEY) { const st = JSON.parse(value); saveStations(st.levelIdx, st.list); continue; }
      try { localStorage.setItem(key, value); } catch { /* over quota: skip */ }
    }
    ppoStatus.textContent = `loaded training file (saved ${data.savedAt || 'unknown'})`;
    describeDemo();
  } catch (err) {
    ppoStatus.textContent = `could not load training file: ${err.message}`;
  }
});

// On mobile, tapping the canvas acts as the mouse (aim + fire) and the buttons
// below the screen drive movement, use, special power and the lights.
const isMobile =
  /Android|iPhone|iPad|iPod|Windows Phone|webOS|BlackBerry|Opera Mini|IEMobile|Mobile/i.test(navigator.userAgent) ||
  (navigator.maxTouchPoints > 0 && matchMedia('(any-pointer: coarse)').matches);

if (isMobile) {
  document.body.classList.add('touch');
  const canvas = document.getElementById('c');
  const aim = (e) => {
    const b = canvas.getBoundingClientRect();
    game.mouse = {
      x: ((e.clientX - b.left) / b.width) * game.viewW,
      y: ((e.clientY - b.top) / b.height) * game.viewH,
    };
  };
  canvas.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    aim(e);
    game.mouseDown = true;
    try { canvas.setPointerCapture(e.pointerId); } catch { /* synthetic pointer */ }
  });
  canvas.addEventListener('pointermove', (e) => { if (e.buttons) aim(e); });
  canvas.addEventListener('pointerup', () => { game.mouseDown = false; });
  canvas.addEventListener('pointercancel', () => { game.mouseDown = false; });
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());

  for (const btn of document.querySelectorAll('#touch button')) {
    const code = btn.dataset.key;
    const act = btn.dataset.act;
    const press = (e) => {
      e.preventDefault();
      btn.classList.add('active');
      if (code) game.keys.add(code);
      else if (act === 'special') game.rightDown = true;
      else if (act === 'lights') game.lightsOn = !game.lightsOn;
    };
    const release = (e) => {
      e.preventDefault();
      btn.classList.remove('active');
      if (code) game.keys.delete(code);
      else if (act === 'special') game.rightDown = false;
    };
    btn.addEventListener('pointerdown', press);
    btn.addEventListener('pointerup', release);
    btn.addEventListener('pointercancel', release);
    btn.addEventListener('pointerleave', release);
  }

  // Virtual joystick: drag straight from one direction to another without
  // lifting (and hold diagonals). The knob follows the finger inside the base.
  const stick = document.getElementById('stick');
  const knob = stick.querySelector('.knob');
  const DIRS = ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'];
  const setDirs = (keys) => DIRS.forEach((k) => (keys.includes(k) ? game.keys.add(k) : game.keys.delete(k)));
  let stickPointer = null;
  const stickMove = (e) => {
    const r = stick.getBoundingClientRect();
    const max = r.width / 2;
    let dx = e.clientX - (r.left + r.width / 2);
    let dy = e.clientY - (r.top + r.height / 2);
    const len = Math.hypot(dx, dy);
    if (len > max) { dx = (dx / len) * max; dy = (dy / len) * max; }
    knob.style.transform = `translate(${dx.toFixed(1)}px, ${dy.toFixed(1)}px)`;
    const dz = max * 0.34;
    const keys = [];
    if (dx < -dz) keys.push('ArrowLeft'); else if (dx > dz) keys.push('ArrowRight');
    if (dy < -dz) keys.push('ArrowUp'); else if (dy > dz) keys.push('ArrowDown');
    setDirs(keys);
  };
  const stickEnd = () => { stickPointer = null; knob.style.transform = 'translate(0, 0)'; setDirs([]); };
  stick.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    stickPointer = e.pointerId;
    try { stick.setPointerCapture(e.pointerId); } catch { /* synthetic pointer */ }
    stickMove(e);
  });
  stick.addEventListener('pointermove', (e) => { if (e.pointerId === stickPointer) stickMove(e); });
  stick.addEventListener('pointerup', (e) => { if (e.pointerId === stickPointer) { e.preventDefault(); stickEnd(); } });
  stick.addEventListener('pointercancel', (e) => { if (e.pointerId === stickPointer) stickEnd(); });
}

game.onGod = (on) => {
  godBtn.textContent = `God mode: ${on ? 'on' : 'off'}`;
  godBtn.classList.toggle('on', on);
};
game.onLevel = (name) => { select.value = name; };
// (help text is drawn by the game itself, in its own font)
const coordsBtn = document.getElementById('coords');
coordsBtn.addEventListener('click', () => {
  game.showCoords = !game.showCoords;
  coordsBtn.textContent = game.showCoords ? 'Hide coordinates' : 'Show coordinates';
  coordsBtn.classList.toggle('on', game.showCoords);
  coordsBtn.blur();
});
godBtn.addEventListener('click', () => { game.setGod(!game.god); godBtn.blur(); });

await game.init();

// Training is manual: press "Train PPO" to start (it runs in the background,
// even with the tab hidden). The Demo button uses the best saved model.

// Bottom status bar, drawn from the original artwork (art/statbar.spe): the
// sbar background, bright/dim weapon icons, numpad windows, digit images and
// the selection highlight, laid out exactly like the original status_bar.
await game.assets.preloadDef({ file: 'art/statbar.spe' });
const statbar = game.assets.specCache?.get('art/statbar.spe');
const statbarPal = statbar?.ofType(T.PALETTE).length ? readPalette(statbar, statbar.ofType(T.PALETTE)[0]) : game.assets.palette;
const sbImage = (name) => {
  const img = game.assets.sprite('art/statbar.spe', name);
  if (!img?.pix) return null;
  const cv = document.createElement('canvas');
  cv.width = img.w; cv.height = img.h;
  const ctx = cv.getContext('2d');
  const data = ctx.createImageData(img.w, img.h);
  const pal = statbarPal || new Uint8Array(256 * 4);
  for (let i = 0; i < img.w * img.h; i++) {
    const idx = img.pix[i], p = idx * 4;
    data.data[i * 4] = pal[p]; data.data[i * 4 + 1] = pal[p + 1]; data.data[i * 4 + 2] = pal[p + 2];
    data.data[i * 4 + 3] = idx === 0 ? 0 : 255;
  }
  ctx.putImageData(data, 0, 0);
  return cv;
};
const art = {
  sbar: sbImage('sbar'), select: sbImage('sbar_select'), numpad: sbImage('sbar_numpad'),
  bnum: Array.from({ length: 30 }, (_, i) => sbImage(`bnum${String(i).padStart(2, '0')}`)),
  bright: WEAPON_ORDER.map((_, i) => sbImage(`bweap000${i + 1}.pcx`)),
  dim: WEAPON_ORDER.map((_, i) => sbImage(`dweap000${i + 1}.pcx`)),
};

// Layout constants from status_bar::redraw (non-scaled render).
const BAR_W = 320, BAR_H = 32, WX = 40, WA = 34, NUM_Y = 21, SEL_OFF = 4;
const panel = document.getElementById('weapons');
panel.width = BAR_W; panel.height = BAR_H;
const pctx = panel.getContext('2d');
pctx.imageSmoothingEnabled = false;
let hover = -1;

function drawNum(x, y, num, base) {
  const b = art.bnum[base];
  if (!b) return;
  let n = Math.max(0, Math.min(999, Math.floor(num)));
  const h = Math.floor(n / 100); n -= h * 100;
  const t = Math.floor(n / 10), o = n - t * 10;
  pctx.drawImage(art.bnum[base + h], x, y);
  pctx.drawImage(art.bnum[base + t], x + b.width, y);
  pctx.drawImage(art.bnum[base + o], x + 2 * b.width, y);
}

function drawBar() {
  const p = game.player;
  if (!p || !art.sbar) return;
  pctx.clearRect(0, 0, BAR_W, BAR_H);
  pctx.drawImage(art.sbar, 0, 0);
  drawNum(17, 11, Math.ceil(p.hp), 0);
  for (let i = 0; i < WEAPON_ORDER.length; i++) {
    const w = WEAPON_ORDER[i];
    if (!p.owned.has(w)) continue;
    const xOn = WX + i * WA, current = w === p.weapon;
    pctx.drawImage(current ? art.bright[i] : art.dim[i], xOn, 0);
    pctx.drawImage(art.numpad, xOn - 2, NUM_Y);
    drawNum(52 + i * WA, 25, game.god ? 999 : (p.ammo[w] || 0), current ? 20 : 10);
    if (i === hover) pctx.drawImage(art.select, xOn + SEL_OFF, 0);
  }
}
let lastBarSig = '';
(function refreshBar() {
  const p = game.player;
  if (p) {
    const sig = WEAPON_ORDER.map((w) => `${p.owned.has(w) ? 1 : 0}${w === p.weapon ? 1 : 0}${p.ammo[w] || 0}`).join('|')
      + `|${Math.ceil(p.hp)}|${game.god ? 1 : 0}|${hover}`;
    if (sig !== lastBarSig) { lastBarSig = sig; drawBar(); }
  }
  requestAnimationFrame(refreshBar);
})();

const barSlotAt = (clientX) => {
  const r = panel.getBoundingClientRect();
  if (!r.width) return -1;
  const x = ((clientX - r.left) / r.width) * BAR_W;
  const slot = Math.floor((x - WX) / WA);
  return slot >= 0 && slot < WEAPON_ORDER.length && game.player?.owned.has(WEAPON_ORDER[slot]) ? slot : -1;
};
panel.addEventListener('mousemove', (e) => { hover = barSlotAt(e.clientX); });
panel.addEventListener('mouseleave', () => { hover = -1; });
panel.addEventListener('click', (e) => { const slot = barSlotAt(e.clientX); if (slot >= 0) game.selectWeapon(WEAPON_ORDER[slot]); });

select.addEventListener('change', () => { select.blur(); game.start(select.value); });
await game.start(LEVELS[0]);
