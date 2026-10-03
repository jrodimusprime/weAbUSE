// Parses an Abuse level (.spe) into tile maps and an object list.
import { loadSpec } from './spec.js';
import { DATA } from './assets.js';

const OBJECT_COLUMNS = [
  'type', 'state', 'x', 'y', 'aistate', 'cur_frame', 'direction', 'fade_count', 'xvel', 'yvel', 'active', 'flags',
  'hp', 'aitype', 'xacel', 'yacel',
];

// "object_descripitions" holds [u16 count][len-prefixed type names].
function readTypeNames(spec, count) {
  const e = spec.find('object_descripitions');
  let p = e.offset + 2;
  const out = [];
  for (let i = 0; i < count; i++) {
    const len = spec.u8[p++];
    out.push(String.fromCharCode(...spec.u8.subarray(p, p + len)).replace(/\0+$/, ''));
    p += len;
  }
  return out;
}

// Per-type variable names, then one list of 32-bit values per object, in name order.
function readLvars(spec, typeCount, count, typeIdx) {
  const out = Array.from({ length: count }, () => ({}));
  const d = spec.find('describe_lvars');
  const l = spec.find('lvars');
  if (!d || !l) return out;
  const names = [];
  let p = d.offset;
  for (let i = 0; i < typeCount; i++) {
    const n = spec.dv.getUint16(p, true);
    p += 2;
    const list = [];
    for (let j = 0; j < n; j++) {
      const len = spec.u8[p++];
      list.push(String.fromCharCode(...spec.u8.subarray(p, p + len)).replace(/\0+$/, ''));
      p += len;
    }
    names.push(list);
  }
  p = l.offset;
  for (let i = 0; i < count; i++) {
    const n = spec.dv.getUint16(p, true);
    p += 2;
    for (let k = 0; k < n; k++) {
      if (spec.u8[p++] !== 2) return out; // tag 2 = 32-bit value
      const v = spec.dv.getInt32(p, true);
      p += 4;
      const name = names[typeIdx[i]]?.[k];
      if (name) out[i][name] = v;
    }
  }
  return out;
}

// Light sources: [u32 count][u32 min level] then 25-byte records.
function readLights(spec) {
  const e = spec.find('lights');
  if (!e) return [];
  const n = spec.dv.getUint32(e.offset, true);
  const out = [];
  for (let i = 0; i < n; i++) {
    const p = e.offset + 8 + i * 25;
    const u = (k) => spec.dv.getInt32(p + k * 4, true);
    out.push({ x: u(0), y: u(1), xs: u(2), ys: u(3), inner: u(4), outer: u(5), type: spec.u8[p + 24] });
  }
  return out;
}

// Ambient-light areas: tag byte, count, then 11 u32 per area.
function readAreas(spec) {
  const e = spec.find('area_list.v1');
  if (!e || spec.u8[e.offset] !== 2) return [];
  const n = spec.dv.getUint32(e.offset + 1, true);
  const out = [];
  for (let i = 0; i < n; i++) {
    const p = e.offset + 5 + i * 44;
    const u = (k) => spec.dv.getInt32(p + k * 4, true);
    out.push({
      x: u(0), y: u(1), w: u(2), h: u(3), active: u(4), ambient: u(5),
      panX: u(6), panY: u(7), ambientSpeed: u(8), panXSpeed: u(9), panYSpeed: u(10),
    });
  }
  return out;
}

function readStateNames(spec, typeCount) {
  const e = spec.find('describe_states');
  const out = [];
  if (!e) return out;
  let p = e.offset;
  for (let i = 0; i < typeCount; i++) {
    const n = spec.dv.getUint16(p, true);
    p += 2;
    const names = [];
    for (let j = 0; j < n; j++) {
      const len = spec.u8[p++];
      names.push(String.fromCharCode(...spec.u8.subarray(p, p + len)).replace(/\0+$/, ''));
      p += len;
    }
    out.push(names);
  }
  return out;
}

// Object properties are stored column by column: a width tag byte followed by one value per object.
function readColumn(spec, name, count, signed) {
  const e = spec.find(name);
  if (!e) return new Array(count).fill(0);
  const width = (e.size - 1) / count;
  let p = e.offset + 1;
  const out = new Array(count);
  for (let i = 0; i < count; i++, p += width) {
    if (width === 1) out[i] = signed ? spec.dv.getInt8(p) : spec.u8[p];
    else if (width === 2) out[i] = signed ? spec.dv.getInt16(p, true) : spec.dv.getUint16(p, true);
    else out[i] = signed ? spec.dv.getInt32(p, true) : spec.dv.getUint32(p, true);
  }
  return out;
}

export async function loadLevel(file) {
  const spec = await loadSpec(`${DATA}levels/${file}`);

  const fg = spec.find('fgmap');
  const fgW = spec.dv.getUint32(fg.offset, true);
  const fgH = spec.dv.getUint32(fg.offset + 4, true);
  const fgmap = new Uint16Array(fgW * fgH);
  for (let i = 0; i < fgmap.length; i++) fgmap[i] = spec.dv.getUint16(fg.offset + 8 + i * 2, true) & 0x3fff;

  let bgW = 0, bgH = 0, bgmap = new Uint16Array(0);
  const bg = spec.find('bgmap');
  if (bg) {
    bgW = spec.dv.getUint32(bg.offset, true);
    bgH = spec.dv.getUint32(bg.offset + 4, true);
    bgmap = new Uint16Array(bgW * bgH);
    for (let i = 0; i < bgmap.length; i++) bgmap[i] = spec.dv.getUint16(bg.offset + 8 + i * 2, true);
  }

  // Background parallax: view * mul / div, stored as [tag, xmul, xdiv, ymul, ydiv].
  let bgRate = { xmul: 1, xdiv: 8, ymul: 1, ydiv: 8 };
  const rate = spec.find('bg_scroll_rate');
  if (rate) {
    const o = rate.offset + 1;
    bgRate = {
      xmul: spec.dv.getUint32(o, true), xdiv: spec.dv.getUint32(o + 4, true) || 1,
      ymul: spec.dv.getUint32(o + 8, true), ydiv: spec.dv.getUint32(o + 12, true) || 1,
    };
  }

  const objList = spec.find('object_list');
  const count = objList ? spec.dv.getUint32(objList.offset, true) : 0;
  const descr = spec.find('object_descripitions');
  const typeCount = descr ? spec.dv.getUint16(descr.offset, true) : 0;
  const typeNames = descr ? readTypeNames(spec, typeCount) : [];
  const stateNames = readStateNames(spec, typeCount);

  const cols = {};
  for (const c of OBJECT_COLUMNS) cols[c] = readColumn(spec, c, count, c !== 'type' && c !== 'state');

  const objects = [];
  const lvars = readLvars(spec, typeCount, count, cols.type);
  for (let i = 0; i < count; i++) {
    const t = cols.type[i];
    const type = typeNames[t];
    const stateName = stateNames[t]?.[cols.state[i]];
    objects.push({
      type, stateName, x: cols.x[i], y: cols.y[i], frame: cols.cur_frame[i], dir: cols.direction[i] || 1,
      fade: cols.fade_count[i], aistate: cols.aistate[i], hp: cols.hp[i], aitype: cols.aitype[i],
      xvel: cols.xvel[i], yvel: cols.yvel[i], xacel: cols.xacel[i], yacel: cols.yacel[i], links: [], lights: [], lv: lvars[i],
    });
  }

  // Links are 1-based object numbers; negative targets refer to players and are ignored.
  const linkEntry = spec.find('object_links');
  if (linkEntry) {
    const n = spec.dv.getUint32(linkEntry.offset + 1, true);
    for (let i = 0; i < n; i++) {
      const a = spec.dv.getInt32(linkEntry.offset + 5 + i * 8, true);
      const b = spec.dv.getInt32(linkEntry.offset + 9 + i * 8, true);
      if (a > 0 && b > 0 && objects[a - 1] && objects[b - 1]) objects[a - 1].links.push(b - 1);
    }
  }

  const lights = readLights(spec);
  const areas = readAreas(spec);
  const lightLinkEntry = spec.find('light_links');
  if (lightLinkEntry && spec.u8[lightLinkEntry.offset] === 2) {
    const n = spec.dv.getUint32(lightLinkEntry.offset + 1, true);
    for (let i = 0; i < n; i++) {
      const o = spec.dv.getUint32(lightLinkEntry.offset + 5 + i * 8, true);
      const l = spec.dv.getUint32(lightLinkEntry.offset + 9 + i * 8, true);
      if (objects[o - 1] && lights[l - 1]) objects[o - 1].lights.push(l - 1);
    }
  }

  return { name: file, fgW, fgH, fgmap, bgW, bgH, bgmap, bgRate, objects, lights, areas };
}
