import { Game } from './game.js';
import { WEAPON_ORDER } from './weapons.js';
import { readPalette, T } from './spec.js';

const LEVELS = Array.from({ length: 22 }, (_, i) => `level${String(i).padStart(2, '0')}.spe`);

const select = document.getElementById('level');
for (const l of LEVELS) select.add(new Option(l, l));

const godBtn = document.getElementById('god');
const game = new Game(document.getElementById('c'), document.getElementById('hud'));
window.game = game;

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
}

game.onGod = (on) => {
  godBtn.textContent = `God mode: ${on ? 'on' : 'off'}`;
  godBtn.classList.toggle('on', on);
};
game.onLevel = (name) => { select.value = name; };
const hint = document.getElementById('hint');
game.onHelp = (text) => { hint.textContent = text; };
godBtn.addEventListener('click', () => { game.setGod(!game.god); godBtn.blur(); });

await game.init();

// Bottom weapon panel, like the original status bar: a bright icon for the
// current weapon, dim icons for the others, and the ammo count under each gun.
await game.assets.preloadDef({ file: 'art/statbar.spe' });
const statbar = game.assets.specCache?.get('art/statbar.spe');
const statbarPal = statbar?.ofType(T.PALETTE).length ? readPalette(statbar, statbar.ofType(T.PALETTE)[0]) : game.assets.palette;
const iconURL = (name) => {
  const img = game.assets.sprite('art/statbar.spe', name);
  if (!img?.pix) return '';
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
  return cv.toDataURL();
};
const slots = WEAPON_ORDER.map((weapon, i) => ({
  weapon,
  bright: iconURL(`bweap000${i + 1}.pcx`),
  dim: iconURL(`dweap000${i + 1}.pcx`),
}));
const panel = document.getElementById('weapons');
for (const slot of slots) {
  const el = document.createElement('button');
  el.type = 'button';
  el.className = 'wslot';
  el.title = slot.weapon;
  const img = document.createElement('img');
  img.alt = slot.weapon;
  img.src = slot.dim;
  const ammo = document.createElement('span');
  ammo.className = 'ammo';
  el.append(img, ammo);
  el.addEventListener('click', () => { game.selectWeapon(slot.weapon); el.blur(); });
  panel.appendChild(el);
}
let lastWeaponSig = '';
(function refreshWeapons() {
  const p = game.player;
  if (p) {
    const sig = WEAPON_ORDER.map((w) => `${p.owned.has(w) ? 1 : 0}${w === p.weapon ? 1 : 0}${p.ammo[w] || 0}`).join('|') + (game.god ? 'G' : '');
    if (sig !== lastWeaponSig) {
      lastWeaponSig = sig;
      [...panel.children].forEach((el, i) => {
        const w = WEAPON_ORDER[i];
        const owned = p.owned.has(w);
        el.classList.toggle('owned', owned);
        el.classList.toggle('current', w === p.weapon);
        el.querySelector('img').src = owned && w === p.weapon ? slots[i].bright : slots[i].dim;
        el.querySelector('.ammo').textContent = owned ? (game.god ? '∞' : (p.ammo[w] || 0)) : '';
      });
    }
  }
  requestAnimationFrame(refreshWeapons);
})();

select.addEventListener('change', () => { select.blur(); game.start(select.value); });
await game.start(LEVELS[0]);
