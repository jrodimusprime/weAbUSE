// Quick inspection of a .spe level: tile id histogram and floor tile info.
import { loadLevel } from '../src/level.js';

const file = process.argv[2] || 'level01.spe';
const lv = await loadLevel(file);
console.log('level', lv.name, 'fg', lv.fgW, lv.fgH, 'bg', lv.bgW, lv.bgH);
const hist = new Map();
for (const id of lv.fgmap) hist.set(id, (hist.get(id) || 0) + 1);
const top = [...hist.entries()].sort((a, b) => b[1] - a[1]).slice(0, 30);
console.log('top fg tile ids:', top);
console.log('objects:', lv.objects.length);
const types = new Map();
for (const o of lv.objects) types.set(o.type, (types.get(o.type) || 0) + 1);
console.log('object types:', [...types.entries()].sort((a, b) => b[1] - a[1]));

