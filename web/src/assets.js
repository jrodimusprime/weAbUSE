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
    await this.loadTints();
  }

  // Colour tints (ant.lsp ant_tints, guns.lsp gun_tints): each is a palette
  // file, turned into a table that sends every colour to the closest one in
  // the game palette (items.cpp char_tint, palette::find_closest). Indexed by
  // the object's aitype; null is normal_tint.
  async loadTints() {
    const P = this.palette;
    const closest = (r, g, b) => {
      let c = 0, d = 0x100000;
      for (let i = 0; i < 256; i++) {
        const nd = (r - P[i * 4]) ** 2 + (g - P[i * 4 + 1]) ** 2 + (b - P[i * 4 + 2]) ** 2;
        if (nd < d) { c = i; d = nd; }
      }
      return c;
    };
    const table = async (file) => {
      if (!file) return null;
      try {
        const spec = await loadSpec(DATA + file);
        const pal = readPalette(spec, spec.ofType(T.PALETTE)[0]);
        const t = new Uint8Array(256);
        for (let i = 0; i < 256; i++) t[i] = closest(pal[i * 4], pal[i * 4 + 1], pal[i * 4 + 2]);
        return t;
      } catch { return null; }
    };
    const A = 'art/tints/ant/', G = 'art/tints/guns/';
    const ant = [`${A}green`, `${A}blue`, `${A}brown`, `${A}egg`, `${A}yellow`, `${A}mustard`, `${A}orange`, `${A}gray`, `${G}green`, `${A}darkblue`];
    const gun = [null, `${G}orange`, `${G}green`, `${G}redish`, `${G}blue`];
    this.tints = {
      ant_draw: await Promise.all(ant.map((f) => table(`${f}.spe`))),
      gun_draw: await Promise.all(gun.map((f) => table(f && `${f}.spe`))),
    };
    this.tints.ant_draw[0] = null; // ant_draw: aitype 0 is drawn plain
  }

  // The tint table an object is drawn with, if any (ant_draw / gun_draw).
  tintFor(e) {
    const list = this.tints?.[e.def.funs.get('draw_fun')];
    return (list && list[e.aitype]) || null;
  }

  // `img` recoloured through a tint table (TransImage::PutRemap); transparent pixels stay transparent.
  tinted(img, table) {
    img.tintCache ??= new Map();
    let out = img.tintCache.get(table);
    if (!out) {
      const pix = new Uint8Array(img.pix.length);
      for (let i = 0; i < pix.length; i++) pix[i] = img.pix[i] ? table[img.pix[i]] : 0;
      out = { ...img, pix, tintCache: null, atlasEpoch: undefined };
      img.tintCache.set(table, out);
    }
    return out;
  }

  defChar(it, args, env) {
    const name = args[0].name;
    const def = { name, file: null, states: new Map(), funs: new Map(), flags: new Map(), abilities: new Map(), range: [0, 0] };
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
      } else if (head === 'range') {
        def.range = [Number(clause[1]) || 0, Number(clause[2]) || 0];
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
