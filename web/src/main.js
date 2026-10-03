import { Game } from './game.js';

const LEVELS = Array.from({ length: 22 }, (_, i) => `level${String(i).padStart(2, '0')}.spe`);

const select = document.getElementById('level');
for (const l of LEVELS) select.add(new Option(l, l));

const godBtn = document.getElementById('god');
const game = new Game(document.getElementById('c'), document.getElementById('hud'));
window.game = game;

game.onGod = (on) => {
  godBtn.textContent = `God mode: ${on ? 'on' : 'off'}`;
  godBtn.classList.toggle('on', on);
};
game.onLevel = (name) => { select.value = name; };
godBtn.addEventListener('click', () => { game.setGod(!game.god); godBtn.blur(); });

await game.init();
select.addEventListener('change', () => { select.blur(); game.start(select.value); });
await game.start(LEVELS[0]);
