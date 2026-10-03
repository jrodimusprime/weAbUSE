// WebGL renderer: 8-bit paletted images live in one index atlas; the fragment
// shader looks each index up in a 256-entry palette texture.

const VERT = `
attribute vec2 a_pos;
attribute vec2 a_uv;
attribute float a_opaque;
uniform vec2 u_view;
uniform vec2 u_atlas;
varying vec2 v_uv;
varying float v_opaque;
void main() {
  vec2 p = a_pos / u_view * 2.0 - 1.0;
  gl_Position = vec4(p.x, -p.y, 0.0, 1.0);
  v_uv = a_uv / u_atlas;
  v_opaque = a_opaque;
}`;

const FRAG = `
precision highp float;
uniform sampler2D u_tex;
uniform sampler2D u_pal;
uniform float u_bright;
varying vec2 v_uv;
varying float v_opaque;
void main() {
  float idx = floor(texture2D(u_tex, v_uv).r * 255.0 + 0.5);
  if (idx < 0.5 && v_opaque < 0.5) discard;
  vec3 c = texture2D(u_pal, vec2((idx + 0.5) / 256.0, 0.5)).rgb;
  gl_FragColor = vec4(c * u_bright, 1.0);
}`;

const ATLAS = 4096;
const MAX_QUADS = 4096;

export class Renderer {
  constructor(canvas, width, height) {
    this.w = width;
    this.h = height;
    canvas.width = width;
    canvas.height = height;
    const gl = canvas.getContext('webgl', { alpha: false, antialias: false });
    if (!gl) throw new Error('WebGL is not available');
    this.gl = gl;

    this.prog = this.link(VERT, FRAG);
    gl.useProgram(this.prog);
    this.loc = {
      pos: gl.getAttribLocation(this.prog, 'a_pos'),
      uv: gl.getAttribLocation(this.prog, 'a_uv'),
      op: gl.getAttribLocation(this.prog, 'a_opaque'),
    };
    gl.uniform2f(gl.getUniformLocation(this.prog, 'u_view'), width, height);
    gl.uniform2f(gl.getUniformLocation(this.prog, 'u_atlas'), ATLAS, ATLAS);
    gl.uniform1i(gl.getUniformLocation(this.prog, 'u_tex'), 0);
    gl.uniform1i(gl.getUniformLocation(this.prog, 'u_pal'), 1);
    this.brightLoc = gl.getUniformLocation(this.prog, 'u_bright');
    gl.uniform1f(this.brightLoc, 1);

    gl.activeTexture(gl.TEXTURE0);
    this.atlas = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.atlas);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, ATLAS, ATLAS, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, null);
    this.nearest();

    gl.activeTexture(gl.TEXTURE1);
    this.palTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.palTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(1024));
    this.nearest();

    this.buf = gl.createBuffer();
    this.data = new Float32Array(MAX_QUADS * 6 * 5);
    this.count = 0;
    this.resetAtlas();
  }

  nearest() {
    const gl = this.gl;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  link(vs, fs) {
    const gl = this.gl;
    const mk = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, mk(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }

  setPalette(rgba) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE1);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 256, 1, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
  }

  setBrightness(b) { this.flush(); this.gl.uniform1f(this.brightLoc, b); }

  // Forget packed images; callers must re-request them via draw().
  resetAtlas() {
    this.shelfX = 0; this.shelfY = 0; this.shelfH = 0;
    this.epoch = (this.epoch || 0) + 1;
  }

  place(img) {
    if (img.atlasEpoch === this.epoch) return;
    if (this.shelfX + img.w > ATLAS) { this.shelfX = 0; this.shelfY += this.shelfH; this.shelfH = 0; }
    if (this.shelfY + img.h > ATLAS) throw new Error('atlas full');
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, this.shelfX, this.shelfY, img.w, img.h, gl.LUMINANCE, gl.UNSIGNED_BYTE, img.pix);
    img.ax = this.shelfX; img.ay = this.shelfY; img.atlasEpoch = this.epoch;
    this.shelfX += img.w;
    this.shelfH = Math.max(this.shelfH, img.h);
  }

  begin() {
    const gl = this.gl;
    gl.viewport(0, 0, this.w, this.h);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    this.count = 0;
  }

  draw(img, x, y, { flip = false, opaque = false, w = img.w, h = img.h } = {}) {
    if (x >= this.w || y >= this.h || x + w <= 0 || y + h <= 0) return;
    this.place(img);
    if (this.count >= MAX_QUADS) this.flush();
    const u0 = flip ? img.ax + img.w : img.ax;
    const u1 = flip ? img.ax : img.ax + img.w;
    const v0 = img.ay, v1 = img.ay + img.h;
    const x1 = x + w, y1 = y + h, o = opaque ? 1 : 0;
    const d = this.data;
    let i = this.count * 30;
    const v = [x, y, u0, v0, o, x1, y, u1, v0, o, x, y1, u0, v1, o, x1, y, u1, v0, o, x1, y1, u1, v1, o, x, y1, u0, v1, o];
    for (let k = 0; k < 30; k++) d[i++] = v[k];
    this.count++;
  }

  // Solid rectangle in palette colour `idx`, drawn from a cached 1x1 image.
  rect(x, y, w, h, idx) {
    this.swatches ??= new Map();
    let s = this.swatches.get(idx);
    if (!s) { s = { w: 1, h: 1, pix: new Uint8Array([idx]) }; this.swatches.set(idx, s); }
    this.draw(s, x, y, { w, h, opaque: true });
  }

  flush() {
    if (!this.count) return;
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
    gl.bufferData(gl.ARRAY_BUFFER, this.data.subarray(0, this.count * 30), gl.DYNAMIC_DRAW);
    const stride = 20;
    gl.enableVertexAttribArray(this.loc.pos);
    gl.vertexAttribPointer(this.loc.pos, 2, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(this.loc.uv);
    gl.vertexAttribPointer(this.loc.uv, 2, gl.FLOAT, false, stride, 8);
    gl.enableVertexAttribArray(this.loc.op);
    gl.vertexAttribPointer(this.loc.op, 1, gl.FLOAT, false, stride, 16);
    gl.drawArrays(gl.TRIANGLES, 0, this.count * 6);
    this.count = 0;
  }
}
