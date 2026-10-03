// A live game object: one def_char instance in the level.

export class Entity {
  constructor(def, o = {}) {
    this.def = def;
    this.type = def.name;
    this.x = o.x ?? 0;
    this.y = o.y ?? 0;
    this.px = this.x;
    this.py = this.y;
    this.vx = 0;
    this.vy = 0;
    this.dir = o.dir || 1;
    this.state = o.stateName && def.states.has(o.stateName) ? o.stateName : 'stopped';
    this.frame = o.frame || 0;
    this.stateTime = 0;
    this.aistate = o.aistate || 0;
    this.aitype = o.aitype || 0;
    this.hp = def.abilities.get('start_hp') ?? (o.hp > 0 ? o.hp : 0);
    this.xvel = o.xvel || 0;
    this.yvel = o.yvel || 0;
    this.xacel = o.xacel || 0;
    this.yacel = o.yacel || 0;
    this.links = [];
    this.ai = def.funs.get('ai_fun');
    this.dead = false;
    this.hidden = false;
    this.a = {}; // behaviour scratch state
  }

  setState(s) {
    if (this.state !== s && this.def.states.has(s)) { this.state = s; this.frame = 0; this.stateTime = 0; }
  }

  frames() { return this.def.states.get(this.state) || this.def.states.get('stopped'); }

  // Advances the animation; false once it wraps past the last frame.
  nextPicture() {
    const f = this.frames();
    if (!f) return false;
    this.frame++;
    if (this.frame >= f.length) { this.frame = 0; return false; }
    return true;
  }
}
