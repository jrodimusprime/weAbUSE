// Sound effects from the original WAV files via WebAudio, attenuated and panned by distance.
import { DATA } from './assets.js';

const FILES = {
  mgun: 'shotgn31', plasma: 'plasma03', throw: 'throw01', rocket: 'rocket02', explode: 'explod02',
  enemyshot: 'plasma02', antdie: 'adie02', antscare: 'alien01', antland: 'aland01', antslash: 'aslash01',
  health: 'health01', ammo: 'ammo01', spring: 'spring03', swish: 'swish01', switch: 'switch01', tick: 'timerfst',
  hwall: 'crmble01', crumble: 'crmble01', teleport: 'telept01', die: 'pldeth02',
};

export class Audio {
  constructor() {
    this.ctx = null;
    this.buffers = new Map();
    const unlock = () => {
      this.ctx ??= new AudioContext();
      if (this.ctx.state === 'suspended') this.ctx.resume();
    };
    for (const ev of ['keydown', 'mousedown']) addEventListener(ev, unlock);
  }

  async load(name) {
    if (!this.buffers.has(name)) {
      this.buffers.set(name, fetch(`${DATA}sfx/${FILES[name] ?? name}.wav`)
        .then((r) => r.arrayBuffer())
        .then((b) => this.ctx.decodeAudioData(b))
        .catch(() => null));
    }
    return this.buffers.get(name);
  }

  // `at` is the source position relative to the listener, or null for non-positional sounds.
  async play(name, at, volume = 1) {
    if (!this.ctx || this.ctx.state !== 'running') return;
    const buf = await this.load(name);
    if (!buf) return;
    const dist = at ? Math.hypot(at.dx, at.dy) : 0;
    const vol = Math.max(0, 1 - dist / 450);
    if (vol <= 0) return;
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    const gain = this.ctx.createGain();
    gain.gain.value = vol * 0.6 * volume;
    const pan = this.ctx.createStereoPanner();
    pan.pan.value = at ? Math.max(-1, Math.min(1, at.dx / 300)) : 0;
    src.connect(gain).connect(pan).connect(this.ctx.destination);
    src.start();
  }
}
