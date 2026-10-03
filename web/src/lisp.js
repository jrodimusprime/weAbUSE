// Minimal Lisp reader/evaluator, used only to read the game's data definitions
// (def_char, load_tiles, ...). Game logic itself is implemented natively in JS.

export class Sym {
  constructor(name) { this.name = name; }
}
const symtab = new Map();
export function sym(name) {
  if (!symtab.has(name)) symtab.set(name, new Sym(name));
  return symtab.get(name);
}
const isNil = (x) => x == null || (Array.isArray(x) && x.length === 0);
const QUOTE = sym('quote'), QQ = sym('quasiquote'), UQ = sym('unquote'), UQS = sym('unquote-splicing');

export function read(src) {
  let i = 0;
  const n = src.length;
  const ws = () => {
    for (;;) {
      while (i < n && /\s/.test(src[i])) i++;
      if (src[i] === ';') { while (i < n && src[i] !== '\n') i++; continue; }
      if (src[i] === '/' && src[i + 1] === '*') {
        const j = src.indexOf('*/', i + 2);
        i = j < 0 ? n : j + 2;
        continue;
      }
      return;
    }
  };
  const form = () => {
    ws();
    const c = src[i];
    if (c === '(') {
      i++;
      const list = [];
      for (;;) {
        ws();
        if (i >= n) return list;
        if (src[i] === ')') { i++; return list; }
        list.push(form());
      }
    }
    if (c === ')') { i++; return null; }
    if (c === "'") { i++; return [QUOTE, form()]; }
    if (c === '`') { i++; return [QQ, form()]; }
    if (c === ',') {
      i++;
      if (src[i] === '@') { i++; return [UQS, form()]; }
      return [UQ, form()];
    }
    if (c === '"') {
      let s = '';
      i++;
      while (i < n && src[i] !== '"') {
        if (src[i] === '\\') { i++; s += src[i] === 'n' ? '\n' : src[i]; } else s += src[i];
        i++;
      }
      i++;
      return s;
    }
    let j = i;
    while (j < n && !/[\s()"';]/.test(src[j])) j++;
    const tok = src.slice(i, j);
    i = Math.max(j, i + 1);
    if (/^[+-]?\d+$/.test(tok)) return parseInt(tok, 10);
    if (/^0x[0-9a-f]+$/i.test(tok)) return parseInt(tok, 16);
    return sym(tok);
  };
  const out = [];
  for (;;) {
    ws();
    if (i >= n) break;
    out.push(form());
  }
  return out;
}

class Env {
  constructor(parent) { this.vars = new Map(); this.parent = parent; }
  lookup(name) {
    for (let e = this; e; e = e.parent) if (e.vars.has(name)) return e;
    return null;
  }
}

class Lambda {
  constructor(params, body, env) { this.params = params; this.body = body; this.env = env; }
}

export class Interp {
  constructor(hooks) {
    this.global = new Env(null);
    this.funcs = new Map();
    this.hooks = hooks;
    this.dyn = [];
    this.global.vars.set('nil', null);
    this.global.vars.set('T', sym('T'));
    this.global.vars.set('t', sym('T'));
    this.errors = 0;
  }

  evalAll(forms) {
    for (const f of forms) {
      try { this.eval(f, this.global); } catch (e) { this.errors++; if (this.errors < 20) console.debug('lisp:', e.message); }
    }
  }

  eval(x, env) {
    if (x instanceof Sym) {
      const e = env.lookup(x.name);
      if (e) return e.vars.get(x.name);
      for (let i = this.dyn.length - 1; i >= 0; i--) {
        const d = this.dyn[i].lookup(x.name);
        if (d) return d.vars.get(x.name);
      }
      return x; // unbound symbols evaluate to themselves (state names etc.)
    }
    if (!Array.isArray(x)) return x;
    if (x.length === 0) return null;
    const head = x[0];
    if (!(head instanceof Sym)) return null;
    const name = head.name;
    switch (name) {
      case 'quote': return x[1];
      case 'quasiquote': return this.qq(x[1], env);
      case 'if': return !isNil(this.eval(x[1], env)) ? this.eval(x[2], env) : this.body(x.slice(3), env);
      case 'cond':
        for (const clause of x.slice(1)) {
          const t = this.eval(clause[0], env);
          if (!isNil(t)) return clause.length > 1 ? this.body(clause.slice(1), env) : t;
        }
        return null;
      case 'and': { let r = sym('T'); for (const a of x.slice(1)) { r = this.eval(a, env); if (isNil(r)) return null; } return r; }
      case 'or': { for (const a of x.slice(1)) { const r = this.eval(a, env); if (!isNil(r)) return r; } return null; }
      case 'progn': return this.body(x.slice(1), env);
      case 'setq': case 'setf': {
        let v = null;
        for (let k = 1; k + 1 < x.length; k += 2) {
          v = this.eval(x[k + 1], env);
          const target = x[k];
          if (target instanceof Sym) (env.lookup(target.name) || this.global).vars.set(target.name, v);
        }
        return v;
      }
      case 'let': case 'let*': {
        const inner = new Env(env);
        for (const b of x[1] || []) {
          if (Array.isArray(b)) inner.vars.set(b[0].name, this.eval(b[1], name === 'let*' ? inner : env));
          else if (b instanceof Sym) inner.vars.set(b.name, null);
        }
        return this.body(x.slice(2), inner);
      }
      case 'defun': this.funcs.set(x[1].name, new Lambda(x[2] || [], x.slice(3), this.global)); return x[1];
      case 'defmacro': return x[1];
      case 'lambda': return new Lambda(x[1] || [], x.slice(2), env);
      case 'function': return x[1] instanceof Sym ? this.funcs.get(x[1].name) ?? x[1] : this.eval(x[1], env);
      case 'def_char': return this.hooks.defChar(this, x.slice(1), env);
      default: break;
    }
    const args = x.slice(1).map((a) => this.eval(a, env));
    const user = this.funcs.get(name);
    if (user) return this.apply(user, args);
    const b = builtins[name];
    if (b) return b.call(this, args);
    const hook = this.hooks[name];
    if (hook) return hook.call(this, args);
    return null;
  }

  body(forms, env) {
    let r = null;
    for (const f of forms) r = this.eval(f, env);
    return r;
  }

  apply(fn, args) {
    const env = new Env(fn.env);
    const params = Array.isArray(fn.params) ? fn.params : [];
    params.forEach((p, i) => { if (p instanceof Sym) env.vars.set(p.name, args[i] ?? null); });
    this.dyn.push(env);
    try { return this.body(fn.body, env); } finally { this.dyn.pop(); }
  }

  qq(x, env) {
    if (!Array.isArray(x)) return x;
    if (x[0] === UQ) return this.eval(x[1], env);
    const out = [];
    for (const item of x) {
      if (Array.isArray(item) && item[0] === UQS) {
        const v = this.eval(item[1], env);
        if (Array.isArray(v)) out.push(...v);
      } else out.push(this.qq(item, env));
    }
    return out;
  }
}

const num = (v) => (typeof v === 'number' ? v : 0);
const builtins = {
  '+': (a) => a.reduce((s, v) => s + num(v), 0),
  '-': (a) => (a.length === 1 ? -num(a[0]) : a.slice(1).reduce((s, v) => s - num(v), num(a[0]))),
  '*': (a) => a.reduce((s, v) => s * num(v), 1),
  '/': (a) => Math.trunc(num(a[0]) / (num(a[1]) || 1)),
  '<': (a) => (num(a[0]) < num(a[1]) ? sym('T') : null),
  '>': (a) => (num(a[0]) > num(a[1]) ? sym('T') : null),
  '<=': (a) => (num(a[0]) <= num(a[1]) ? sym('T') : null),
  '>=': (a) => (num(a[0]) >= num(a[1]) ? sym('T') : null),
  '=': (a) => (a[0] === a[1] ? sym('T') : null),
  eq: (a) => (a[0] === a[1] || (isNil(a[0]) && isNil(a[1])) ? sym('T') : null),
  not: (a) => (isNil(a[0]) ? sym('T') : null),
  null: (a) => (isNil(a[0]) ? sym('T') : null),
  list: (a) => a,
  cons: (a) => [a[0], ...(Array.isArray(a[1]) ? a[1] : [])],
  car: (a) => (Array.isArray(a[0]) ? a[0][0] ?? null : null),
  cdr: (a) => (Array.isArray(a[0]) ? a[0].slice(1) : null),
  cadr: (a) => (Array.isArray(a[0]) ? a[0][1] ?? null : null),
  caar: (a) => (Array.isArray(a[0]) && Array.isArray(a[0][0]) ? a[0][0][0] ?? null : null),
  length: (a) => (Array.isArray(a[0]) ? a[0].length : typeof a[0] === 'string' ? a[0].length : 0),
  nth: (a) => (Array.isArray(a[1]) ? a[1][a[0]] ?? null : null),
  append: (a) => a.flatMap((l) => (Array.isArray(l) ? l : [])),
  reverse: (a) => (Array.isArray(a[0]) ? [...a[0]].reverse() : null),
  concatenate: (a) => a.slice(1).map((v) => (v instanceof Sym ? v.name : String(v ?? ''))).join(''),
  digstr: (a) => String(num(a[0])).padStart(num(a[1]), '0'),
  eval(a) { return this.eval(a[0], this.dyn[this.dyn.length - 1] ?? this.global); },
  list_fn: (a) => a,
};
