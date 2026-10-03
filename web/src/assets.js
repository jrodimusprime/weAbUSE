// Loads the game's Lisp data files and exposes tiles, palette and character definitions.
import { Interp, sym, read } from './lisp.js';
import { loadSpec, readPalette, readBackTile, readForeTile, readCharacter, T } from './spec.js';

export const DATA = new URL('../data/', import.meta.url).href;

const LOAD_RE = /\(\s*load\s+"([^"]+)"/g;

async function prefetch(path, texts) {
  if (texts.has(path)) return;
  texts.set(path, null);
  try {
    const r = await fetch(DATA + path);
    if (!r.ok) return;
    const text = await r.text();
    texts.set(path, text);
    const deps = [...text.matchAll(LOAD_RE)].map((m) => m[1]);
    await Promise.all(deps.map((d) => prefetch(d, texts)));
  } catch { /* missing optional addon */ }
}

export class Assets {
  constructor() {
    this.defs = new Map();      // character name -> definition
    this.tileFiles = [];
    this.paletteFile = 'art/back/backgrnd.spe';
    this.fore = new Map();      // tile number -> {spec, entry, img}
    this.back = new Map();
    this.sprites = new Map();   // "file|name" -> image
    this.palette = null;
  }

  async init() {
    const texts = new Map();
    await prefetch('abuse.lsp', texts);
    const self = this;
    const interp = new Interp({
      defChar(it, args, env) { return self.defChar(it, args, env); },
      load_tiles(files) { self.tileFiles.push(...files.filter((f) => typeof f === 'string')); return sym('T'); },
      load_palette(a) { if (typeof a[0] === 'string') self.paletteFile = a[0]; return sym('T'); },
      load(a) {
        const text = texts.get(a[0]);
        if (!text) return null;
        this.evalAll(read(text));
        return sym('T');
      },
    });
    interp.evalAll(read(texts.get('abuse.lsp')));
    await this.registerTiles();
    const pal = await loadSpec(DATA + this.paletteFile);
    this.palette = readPalette(pal, pal.ofType(T.PALETTE)[0]);
  }

  defChar(it, args, env) {
    const name = args[0].name;
    const def = { name, file: null, states: new Map(), funs: new Map(), flags: new Map(), abilities: new Map() };
    for (const clause of args.slice(1)) {
      if (!Array.isArray(clause) || !clause.length) continue;
      const head = clause[0].name;
      if (head === 'states') {
        const file = it.eval(clause[1], env);
        def.file = typeof file === 'string' ? file : null;
        for (const st of clause.slice(2)) {
          if (!Array.isArray(st) || !st[0].name) continue;
          const frames = it.eval(st[1], env);
          const list = typeof frames === 'string' ? [frames] : Array.isArray(frames) ? frames.filter((f) => typeof f === 'string') : [];
          if (list.length) def.states.set(st[0].name, list);
        }
      } else if (head === 'funs') {
        for (const f of clause.slice(1)) if (Array.isArray(f) && f[1]?.name) def.funs.set(f[0].name, f[1].name);
      } else if (head === 'flags') {
        for (const f of clause.slice(1)) if (Array.isArray(f)) def.flags.set(f[0].name, f[1]?.name ?? f[1]);
      } else if (head === 'abilities') {
        for (const f of clause.slice(1)) if (Array.isArray(f) && typeof f[1] === 'number') def.abilities.set(f[0].name, f[1]);
      }
    }
    this.defs.set(name, def);
    return null;
  }

  async registerTiles() {
    const specs = await Promise.all(this.tileFiles.map((f) => loadSpec(DATA + f).catch(() => null)));
    specs.forEach((spec) => {
      if (!spec) return;
      for (const e of spec.entries) {
        const n = parseInt(e.name, 10);
        if (Number.isNaN(n)) continue;
        if (e.type === T.FORETILE) this.fore.set(n, { spec, entry: e, img: null });
        else if (e.type === T.BACKTILE) this.back.set(n, { spec, entry: e, img: null });
      }
    });
  }

  foreTile(n) {
    const t = this.fore.get(n);
    if (!t) return null;
    return (t.img ??= readForeTile(t.spec, t.entry));
  }

  backTile(n) {
    const t = this.back.get(n);
    if (!t) return null;
    return (t.img ??= readBackTile(t.spec, t.entry));
  }

  async loadSpecFile(path) { return loadSpec(DATA + path); }

  // Sprite image by art file + entry name; the file must have been loaded with loadSpecFile.
  sprite(file, name) {
    name = name.replace(/\+$/, '');
    const key = `${file}|${name}`;
    if (this.sprites.has(key)) return this.sprites.get(key);
    const spec = this.specCache?.get(file);
    const e = spec?.find(name);
    const img = e ? readCharacter(spec, e) : null;
    this.sprites.set(key, img);
    return img;
  }

  async preloadDef(def) {
    if (!def?.file) return;
    this.specCache ??= new Map();
    if (!this.specCache.has(def.file)) {
      try { this.specCache.set(def.file, await loadSpec(DATA + def.file)); } catch { /* ignore */ }
    }
  }
}
