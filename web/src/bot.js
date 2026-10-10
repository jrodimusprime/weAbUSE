// Test-harness autopilot for "demo mode".
//
// NOTE: the original Abuse has no autopilot player (demo.cpp only records and
// replays real input packets, and -nodelay just removes frame pacing). This
// module is a port-only test aid: it synthesises the same keyboard/mouse input
// a human would produce, through the existing input fields on Game. It changes
// no gameplay, physics or AI code paths.
//
// Navigation is a Dijkstra search over coarse tile-grid cells. Edges model the
// things the player can actually do: walk one tile, auto-step up one tile,
// walk off edges and fall, jump across gaps (up to 8 tiles), climb ladders and
// use teleporters. The bot follows the path one waypoint at a time:
//   - walk/jump/fall toward the waypoint
//   - climb when the path runs up/down a ladder column
//   - press the action key near switches, teleporters, save stations and
//     platforms (pulsed, so latch-based objects see press+release)
//   - shoot the nearest enemy in line of sight; when stuck against a solid,
//     shoot that solid if it is destructible (hidden walls / blocks)
// If no path exists (switch-gated doors, platforms, springs) it falls back to
// reactive wandering until the map opens up and a path appears.

const ENEMY_AI = new Set([
  'ant_ai', 'flyer_ai', 'track_ai', 'spray_gun_ai', 'jug_ai', 'boss_ai',
  'crack_ai', 'lightin_ai', 'bolder_ai', 'air_mine_ai', 'mine_ai',
]);

const ACTION_AI = new Set([
  'switcher_ai', 'switch_once_ai', 'switch_delay_ai', 'tpd_ai', 'tp2_ai',
  'restart_ai', 'platform_ai', 'next_level_ai',
]);

const HALF_W = 6;  // player half width, matches game.js
const BODY_H = 29; // player collision height, matches game.js

const MAX_FALL = 80; // deepest walk-off fall to model (no fall damage)
const MAX_JUMP = 8;  // furthest gap a jump crosses

export class Bot {
  constructor() {
    this.t = 0;          // sim seconds
    this.lastT = 0;      // last progress check time
    this.stuckT = 0;     // seconds without closing in on the target
    this.climbStuckT = 0;// same, while climbing
    this.lastDxn = null; // |dx| to the target at the last progress check
    this.walkDir = 1;    // latched walk direction (avoids ±2px jitter)
    this.recover = false;// walking back to a missed path cell
    this.avoidT = 0;     // seconds to wander before re-aiming
    this.dir = 1;        // wander direction
    this.pulseT = 0;     // action-key hold timer
    this.saveCd = 0;     // don't camp on save stations
    this.climbUp = true;
    this.lastJumpT = -9; // jump throttle
    this.tpCd = 0;       // teleporter cooldown (anti ping-pong)
    this.seen = null;    // demo stall-timeout: visited cells
    this.seenKey = null; // level the seen set belongs to
    this.doorWaitT = 0;  // time standing on a closed door
    this.doorLatchT = 0; // freeze timer while a gated chute/door is shot open
    // pathfinding state
    this.path = null;    // array of {c, r, isGoal} cells to the exit
    this.pathAt = 0;     // sim time the path was computed
    this.pathIdx = 0;    // next unreached path index (hysteresis)
    this.dist = null;    // distToGoal map (Uint32Array) or null
    this.nav = null;     // current waypoint {x, y, r, isGoal}
    this.pathGoal = null;// the exit entity the current path leads to
  }

  step(g) {
    const p = g.player;
    const dt = 1 / 60;
    this.t += dt;
    this.saveCd = Math.max(0, this.saveCd - dt);
    this.tpCd = Math.max(0, this.tpCd - dt);

    // Test-mode safety net: keep the run moving without the usual death loop,
    // and keep the weapon fed so long levels never run dry. Always use the
    // machine gun (straight line) — lobbed weapons arc and can't break walls.
    if (p.hp < 80) p.hp = p.maxhp;
    if (p.owned.has('MGUN') && p.weapon !== 'MGUN') p.weapon = 'MGUN';
    if ((p.ammo[p.weapon] || 0) < 10) p.ammo[p.weapon] = 100;

    g.keys.clear();
    g.mouseDown = false;
    if (p.dead || !g.level) return;

    // Demo sweep timeout hook (demo.js arms these fields while running).
    // The deadline is a stall budget: entering NEW territory resets the clock
    // (pacing back and forth does not), so long levels get as long as they
    // need while genuinely stuck bots still advance.
    if (g.demoTimeoutTicks) {
      if (this.seenKey !== g.level.name) { this.seen = new Set(); this.seenKey = g.level.name; }
      const sig = Math.floor(p.x / 40) * 1000 + Math.floor(p.y / 40);
      if (!this.seen.has(sig)) {
        this.seen.add(sig);
        g.demoStartTick = g.tickCount || 0;
      }
      if ((g.tickCount || 0) - (g.demoStartTick || 0) > g.demoTimeoutTicks) g.onDemoTimeout?.();
    }

    // Level exit to head for: prefer the exit the current path leads to,
    // otherwise the nearest one (levels can have several).
    const exits = g.entities.filter((e) => !e.dead && e.ai === 'next_level_ai');
    let goal = this.pathGoal && !this.pathGoal.dead ? this.pathGoal : null;
    if (!goal) {
      let best = Infinity;
      for (const e of exits) {
        const d = Math.abs(e.x - p.x) + Math.abs(e.y - p.y);
        if (d < best) { best = d; goal = e; }
      }
    }
    const gd = goal ? Math.abs(goal.x - p.x) + Math.abs(goal.y - p.y) : Infinity;
    const touchingGoal = !!goal && g.touchesPlayer(goal);
    if (!goal) return;

    // ---- pathfinding: advance along the current path; recompute rarely ----
    // (Recomputing from scratch every second makes the bot oscillate between
    // equal-cost alternative routes. Instead we walk the path forward and only
    // recompute when it is exhausted, when stuck, or when there is none.)
    const pc = Math.floor(p.x / g.tw), pr = Math.floor(p.y / g.th);
    const cellReached = (cell) => Math.abs((cell.c + 0.5) * g.tw - p.x) < 20 && Math.abs(cell.r - pr) <= 1;
    if (!this.path) {
      this.pathIdx = 0;
      if (this.t - this.pathAt > 1.2) this.refreshPath(g, pc, pr, goal);
    } else {
      let i = this.pathIdx;
      while (i < this.path.length && cellReached(this.path[i])) i++;
      this.pathIdx = i;
      if (i >= this.path.length || this.stuckT > 2) {
        this.pathIdx = 0;
        this.refreshPath(g, pc, pr, goal);
      }
    }
    if (this.path && this.path.length) {
      // Hysteresis: once the next cell is missed (skipped, e.g. the open
      // chute was passed), keep aiming back at it until it is actually
      // reached — otherwise the waypoint flip-flops and the bot paces.
      const next = this.path[this.pathIdx];
      if (next) {
        const missed = Math.abs((next.c + 0.5) * g.tw - p.x) > 60;
        if (!this.recover && missed) this.recover = true;
        if (this.recover && cellReached(next)) this.recover = false;
      }
      const look = this.recover ? this.pathIdx : Math.min(this.pathIdx + 3, this.path.length - 1);
      const cell = this.path[look];
      this.nav = {
        x: (cell.c + 0.5) * g.tw,
        y: cell.r * g.th,
        r: cell.r,
        isGoal: cell.isGoal,
      };
    } else {
      this.nav = null;
    }

    // Navigation target: the BFS waypoint, or the exit itself as a fallback.
    const tx = this.nav ? this.nav.x : goal ? goal.x : p.x;
    const ty = this.nav ? this.nav.y : goal ? goal.y : p.y;
    const tr = this.nav ? this.nav.r : goal ? Math.floor(goal.y / g.th) : pr;
    const dxn = tx - p.x, dyn = ty - p.y;
    // Without a path, aim for the exit directly; if it is far below, descend
    // rather than jump around on the current floor.
    const descentMode = !this.nav && !!goal && goal.y - p.y > 150;
    const atTarget = !!this.nav && this.nav.isGoal && !!goal
      && Math.abs(goal.x - p.x) < 36 && Math.abs(goal.y - p.y) < 44;

    // Stuck detection: no horizontal progress towards the target for a while.
    // (Vertical-only jitter — hopping against a wall — must count as stuck,
    // so only |dx| improvement resets the timer.)
    if (!atTarget && this.avoidT <= 0) {
      if (this.t - this.lastT >= 0.5) {
        const improved = this.lastDxn !== null && Math.abs(dxn) < this.lastDxn - 4;
        if (!improved) {
          this.stuckT += 0.5;
          if (p.climbing) this.climbStuckT += 0.5;
        } else {
          this.stuckT = 0;
          this.climbStuckT = 0;
        }
        this.lastDxn = Math.abs(dxn);
        this.lastT = this.t;
      }
    } else {
      this.stuckT = 0;
      this.climbStuckT = 0;
      this.lastDxn = Math.abs(dxn);
    }
    this.avoidT = Math.max(0, this.avoidT - dt);

    // Aim the wander direction at the target unless recovering from a flip.
    // In descent mode keep a persistent direction instead: re-centering on the
    // goal's column while it is far below makes the bot pace in place.
    if (this.avoidT <= 0 && Math.abs(dxn) > 6 && !descentMode) this.dir = Math.sign(dxn);

    // Chute awareness: standing on a closed trap/switch door means the way
    // down is gated (usually by enemies that must die). Hold position and
    // keep shooting instead of dashing over the door. (Applies even with a
    // path: the path can still run along the floor on top of a closed door.)
    let onClosedDoor = false;
    for (const s of g.solids) {
      const e = s.e;
      if (!e || (e.ai !== 'strap_door_ai' && e.ai !== 'sdoor_ai') || e.aistate !== 0) continue;
      if (p.x >= s.x0 && p.x <= s.x1 && p.y >= s.y0 - 4 && p.y <= s.y0 + 34) { onClosedDoor = true; break; }
    }
    // Latch immediately on contact: at 2x speed the bot crosses a 30px door in
    // ~0.1s, so any accumulation threshold is never reached.
    if (onClosedDoor) this.doorLatchT = 8;
    if (this.doorLatchT > 0) this.doorLatchT -= dt;
    const hold = this.doorLatchT > 0;

    // ---- aim & fire ----
    // While holding on a gated door, reach further out for the enemies whose
    // death opens it.
    const rangeX = hold ? 560 : 330, rangeY = hold ? 360 : 210;
    let target = null, best = Infinity;
    for (const e of g.entities) {
      if (e.dead || e.hidden || !e.shootable || !ENEMY_AI.has(e.ai)) continue;
      const dx = Math.abs(e.x - p.x), dy = Math.abs(e.y - p.y);
      if (dx > rangeX || dy > rangeY) continue;
      if (!g.sees(p.x, p.y - 15, e.x, e.y - 15)) continue;
      const d = dx + dy;
      if (d < best) { best = d; target = e; }
    }
    let wallTarget = null;
    if (!target && this.stuckT > 1.5) {
      // Stuck against destructible scenery (hidden walls, blocks): shoot it.
      // Aim at whatever blocks the way toward the waypoint, not just ahead.
      const wallDir = this.path ? Math.sign(tx - p.x) : this.dir;
      for (const s of g.solids) {
        const e = s.e;
        if (!e || e.dead || !e.shootable) continue;
        const cx = (s.x0 + s.x1) / 2, cy = (s.y0 + s.y1) / 2;
        if (Math.abs(cx - p.x) > 200 || Math.abs(cy - p.y) > 120) continue;
        if (wallDir && Math.sign(cx - p.x) !== wallDir) continue;
        if (!g.sees(p.x, p.y - 15, cx, cy)) continue;
        target = e;
        wallTarget = e;
        break;
      }
    }

    // Wander the other way when stuck — but never before a blocking wall is
    // destroyed. Allowed with a path too: a long stall means the way is
    // blocked by something the path can't see (e.g. a closed door).
    if (!wallTarget && this.stuckT > 8 && this.avoidT <= 0) {
      this.dir = -this.dir;
      this.avoidT = 8;
      this.stuckT = 0;
    }

    // Stuck without a path: head for a spring (auto-launch) if one is near.
    if (!this.nav && this.stuckT > 2) {
      let spring = null, sd = Infinity;
      for (const e of g.entities) {
        if (e.dead || e.ai !== 'spring_ai') continue;
        const d = Math.abs(e.x - p.x) + Math.abs(e.y - p.y);
        if (d < 400 && d < sd) { sd = d; spring = e; }
      }
      if (spring) this.dir = Math.sign(spring.x - p.x);
    }

    // ---- action pulse: switches, teleporters, save stations, platforms ----
    let wantAction = touchingGoal;
    if (!wantAction && !p.climbing && !p.ladderExit) {
      for (const e of g.entities) {
        if (e.dead || !ACTION_AI.has(e.ai)) continue;
        const ai = e.ai;
        const dx = Math.abs(e.x - p.x), dy = Math.abs(e.y - p.y);
        // Teleporters and platforms need real contact; switches only nearness.
        if (ai === 'tpd_ai' || ai === 'tp2_ai') {
          if (!g.touchesPlayer(e)) continue;
          // With a path, only press when the next path cell is this pad —
          // that prevents paired pads from ping-ponging the bot between them.
          let helps = false;
          if (this.path && this.pathIdx < this.path.length) {
            const src = this.cellOf(g, e);
            const nxt = this.path[this.pathIdx];
            const W = g.level.fgW;
            helps = src !== null && src % W === nxt.c && Math.floor(src / W) === nxt.r;
          } else if (!this.path) {
            // Exploring: press on a cooldown, or when stuck.
            helps = this.stuckT > 2 || this.tpCd <= 0;
          } else if (this.dist) {
            const dcell = this.cellOf(g, e.links[0]);
            const cur = this.cellOf(g, null, p.x, p.y);
            helps = dcell !== null && cur !== null && this.dist[dcell] + 3 < this.dist[cur];
          } else if (goal) {
            const d = e.links[0];
            const nxt = d ? Math.abs(goal.x - d.x) + Math.abs(goal.y - d.y) : Infinity;
            helps = nxt < gd - 60;
          }
          if (!helps) continue;
          this.tpCd = 6;
        } else if (ai === 'platform_ai') {
          if (!(e.aistate === 0 && g.touchesPlayer(e) && goal && Math.abs(goal.y - p.y) > 80)) continue;
          // Only summon while the platform rests at its boarding end; this
          // stops the bot from re-pressing at the destination and riding back.
          // Exception: when wedged against a wall (neither side walkable),
          // press anyway — the elevator descending is the only way out.
          const ends = e.links.map((l) => l.y).filter((v) => isFinite(v));
          const wedged = g.boxHits(p.x + 10, p.y, HALF_W, BODY_H, null)
            && g.boxHits(p.x - 10, p.y, HALF_W, BODY_H, null);
          if (ends.length && Math.abs(e.y - Math.max(...ends)) > 40 && !wedged) continue;
        } else if (ai === 'restart_ai') {
          if (dx > 26 || dy > 36) continue;
          if (this.saveCd > 0) continue;
          this.saveCd = 5; // press once, then move on
        } else {
          if (dx > 24 || dy > 34) continue;
          if (e.aistate !== 0) continue; // switches: press only while off
        }
        wantAction = true;
        break;
      }
    }
    if (wantAction) this.pulseT = 0.3;
    if (this.pulseT > 0) this.pulseT -= dt;

    // ---- ladders ----
    let ladder = null;
    if (!p.climbing && !p.ladderExit) {
      for (const l of g.ladders) {
        if (p.x >= l.x0 - 40 && p.x <= l.x1 + 40 && p.y >= l.y0 - 15 && p.y <= l.y1 + 15) { ladder = l; break; }
      }
    }
    const needClimb = !!ladder && (this.stuckT > 1.5
      || (Math.abs(tx - (ladder.x0 + ladder.x1) / 2) < 90 && Math.abs(ty - p.y) > 60));

    if (p.climbing || needClimb) {
      if (!p.climbing) {
        const cx = (ladder.x0 + ladder.x1) / 2;
        this.climbUp = tr < pr - 1 || ty < p.y - 12;
        if (Math.abs(p.x - cx) > 3) g.keys.add(p.x < cx ? 'ArrowRight' : 'ArrowLeft');
        else g.keys.add(this.climbUp ? 'ArrowUp' : 'ArrowDown');
      } else {
        const l = g.ladders.find((x) => p.x >= x.x0 - 5 && p.x <= x.x1 + 5 && p.y >= x.y0 && p.y <= x.y1);
        if (this.climbStuckT > 2) {
          // Deadlocked on a ladder: hop off and try the other way.
          g.keys.add('Space');
        } else if (this.climbUp) {
          if (!p.ladderExit && l) g.keys.add('ArrowUp'); // climb-off happens at the top
        } else if (!this.climbUp) {
          if (l && l.y1 - p.y > 6) g.keys.add('ArrowDown');
          else g.keys.add('Space'); // reached the bottom: hop off
        }
      }
    } else if (atTarget || !wantAction) {
      // ---- walk / jump ----
      // Follow the wander direction while recovering from a flip or while
      // descending; otherwise aim at the waypoint/exit column, latching the
      // direction so the bot doesn't jitter around the target column.
      let dir;
      if (!this.nav || this.avoidT > 0 || descentMode) dir = this.dir;
      else if (Math.abs(dxn) > 2) { dir = Math.sign(dxn); this.walkDir = dir; }
      else dir = this.walkDir;
      this.dir = dir;
      // Open chute: if the path's next cell is a drop landing directly below,
      // release the movement key over the gap. Holding it would carry the bot
      // across (the auto-step pops it back up the far side).
      const nextCell = this.path && this.pathIdx < this.path.length ? this.path[this.pathIdx] : null;
      const overDrop = !!nextCell && nextCell.r > pr + 1
        && Math.abs((nextCell.c + 0.5) * g.tw - p.x) < 14;
      if (!hold && !overDrop) g.keys.add(dir > 0 ? 'ArrowRight' : 'ArrowLeft');
      const ahead = p.x + dir * 26;
      const groundHere = g.solidAt(p.x, p.y + 4);
      const groundAhead = g.solidAt(ahead, p.y + 4) || g.solidAt(ahead + dir * 6, p.y + 4);
      const gap = p.ground && groundHere && !groundAhead;
      const wallAhead = g.boxHits(p.x + dir * 10, p.y, HALF_W, BODY_H, null);
      const goalUp = dyn < -14 && Math.abs(dxn) < 150;
      // Walk off step-downs and deep gaps (no jump — low ceilings bounce the
      // jump back); only jump same-level gaps or rises toward the waypoint.
      const walkOff = gap && (tr > pr || descentMode);
      // Don't jump while shooting a wall — keep the gun level with the target.
      // While following a path, never wall-jump (it can hop over the intended
      // drop); only cross gaps and climb to higher waypoints.
      if (p.ground && !hold && !overDrop && !wallTarget && !walkOff
        && ((gap && !descentMode) || goalUp
          || (!this.path && wallAhead && this.stuckT > 1 && this.t - this.lastJumpT > 2))) {
        g.keys.add('Space');
        this.lastJumpT = this.t;
      }
    }

    // Keep the action key down while the pulse is active and not climbing.
    if (this.pulseT > 0.1 && !p.climbing && !needClimb) g.keys.add('ArrowDown');

    // ---- apply aim ----
    if (target) {
      g.mouse = { x: target.x - g.cam.x, y: target.y - 12 - g.cam.y };
      g.mouseDown = true;
    } else {
      g.mouse = { x: p.x + this.dir * 120 - g.cam.x, y: p.y - 20 - g.cam.y };
    }
  }

  // ---- tile-grid pathfinding ----

  // The walkable cell nearest an entity position or raw coordinates, or null.
  // Searches a few columns either side because markers (sensors, teleporter
  // pads, platform ends) often sit inside a shaft or just off a ledge.
  cellOf(g, e, x, y) {
    const tw = g.tw, th = g.th;
    const cx = Math.floor((e ? e.x : x) / tw), cy = Math.floor((e ? e.y : y) / th);
    let best = null, bd = Infinity;
    for (let c = cx - 4; c <= cx + 4; c++) {
      for (let r = cy - 8; r <= cy + 8; r++) {
        if (r < 0 || r >= g.level.fgH) continue;
        if (this.floorOK(g, c, r) && this.bodyOK(g, c, r)) {
          const d = Math.abs(c - cx) * 2 + Math.abs(r - cy);
          if (d < bd) { bd = d; best = r * g.level.fgW + c; }
        }
      }
    }
    return best;
  }

  floorOK(g, c, r) {
    if (c < 0 || c >= g.level.fgW || r < 0 || r >= g.level.fgH) return false;
    return g.tileSolid(c * g.tw + g.tw / 2, r * g.th + g.th / 2);
  }

  bodyOK(g, c, r) {
    if (c < 0 || c >= g.level.fgW) return false;
    for (let rr = r - 1; rr >= r - 2; rr--) {
      if (rr >= 0 && g.tileSolid(c * g.tw + g.tw / 2, rr * g.th + g.th / 2)) return false;
    }
    return true;
  }

  refreshPath(g, pc, pr, goal) {
    this.path = null;
    this.dist = null;
    this.pathAt = this.t;
    this.pathGoal = null;
    if (!goal) return;
    const W = g.level.fgW, H = g.level.fgH, tw = g.tw, th = g.th;
    const N = W * H;
    const dist = new Uint32Array(N).fill(0xffffffff);
    const prev = new Int32Array(N).fill(-1);

    // Closed switch doors are solid in practice but invisible to the tile map;
    // treat them as walls so paths don't route through gated doors.
    const doorRects = [];
    for (const s of g.solids) {
      if (s.e && (s.e.ai === 'sdoor_ai' || s.e.ai === 'strap_door_ai') && s.e.aistate === 0) doorRects.push(s);
    }
    const inDoor = (c, r) => {
      const x = c * tw + tw / 2, y = r * th + th / 2;
      for (const s of doorRects) if (x >= s.x0 && x <= s.x1 && y >= s.y0 && y <= s.y1) return true;
      return false;
    };

    const sol = (c, r) => {
      if (c < 0 || c >= W) return true;
      if (r < 0 || r >= H) return false;
      if (inDoor(c, r)) return true;
      return g.tileSolid(c * tw + tw / 2, r * th + th / 2);
    };
    const floorOK = (c, r) => r >= 0 && r < H && c >= 0 && c < W && sol(c, r);
    const bodyOK = (c, r) => {
      if (c < 0 || c >= W) return false;
      for (let rr = r - 1; rr >= r - 2; rr--) if (rr >= 0 && sol(c, rr)) return false;
      return true;
    };

    // Start cell: the player's current tile cell, accepted unconditionally.
    // (The player can legally stand on mask-shaped/sloped tiles where the
    // coarse centre-sample says "not walkable", e.g. the level 0 spawn pocket.)
    // Outgoing edges are still validated, so the search leaves it correctly.
    const start = pr * W + pc;

    // Goal cells for every exit; the search stops at the closest reachable one.
    const goalCells = new Map(); // cell -> entity
    for (const e of g.entities) {
      if (e.dead || e.ai !== 'next_level_ai') continue;
      const cell = this.cellOf(g, e);
      if (cell !== null && !goalCells.has(cell)) goalCells.set(cell, e);
    }
    if (!goalCells.size) return;

    // Precomputed special edges: teleporters and platforms.
    const jumpTo = new Map(); // from cell -> [{to, cost}]
    // Cells covered by a closed switch door: the tiles may look open, but the
    // solid door entity blocks falling through until its switch is pressed.
    const doorCells = new Set();
    for (const e of g.entities) {
      if (e.dead) continue;
      if (e.ai === 'strap_door_ai' || e.ai === 'sdoor_ai') {
        const sw = e.links[0];
        if (sw && sw.aistate === 0) {
          doorCells.add(Math.floor(e.y / th) * W + Math.floor(e.x / tw));
        }
      }
    }
    const addJump = (from, to, cost) => {
      if (from === null || to === null || from === to) return;
      if (!jumpTo.has(from)) jumpTo.set(from, []);
      jumpTo.get(from).push({ to, cost });
    };
    for (const e of g.entities) {
      if (e.dead) continue;
      if (e.ai === 'tp2_ai' || e.ai === 'tpd_ai') {
        addJump(this.cellOf(g, e), e.links[0] ? this.cellOf(g, e.links[0]) : null, 4);
      } else if (e.ai === 'platform_ai') {
        const f = this.cellOf(g, e);
        for (const l of e.links) addJump(f, this.cellOf(g, l), 8);
      } else if (e.ai === 'spring_ai') {
        // Springs launch the player ~9 tiles up (spring_ai: vy = -15*15 px/tick).
        const f = this.cellOf(g, e);
        if (f !== null) {
          const c = f % W, r = Math.floor(f / W);
          addJump(f, r >= 9 ? (r - 9) * W + c : null, 6);
        }
      }
    }

    dist[start] = 0;

    // Binary heap for Dijkstra.
    const heap = [start];
    const push = (idx, d) => {
      heap.push(idx);
      let i = heap.length - 1;
      while (i > 0) {
        const pi = (i - 1) >> 1;
        if (dist[heap[pi]] <= d) break;
        heap[i] = heap[pi];
        i = pi;
      }
      heap[i] = idx;
    };
    const pop = () => {
      const top = heap[0];
      const last = heap.pop();
      if (heap.length) {
        let i = 0;
        const lv = dist[last];
        for (;;) {
          let ci = i, cl = lv;
          const a = i * 2 + 1, b = a + 1;
          if (a < heap.length && dist[heap[a]] < cl) { cl = dist[heap[a]]; ci = a; }
          if (b < heap.length && dist[heap[b]] < cl) { cl = dist[heap[b]]; ci = b; }
          if (ci === i) break;
          heap[i] = heap[ci];
          i = ci;
        }
        heap[i] = last;
      }
      return top;
    };
    const relax = (from, c, r, cost) => {
      if (!floorOK(c, r) || !bodyOK(c, r)) return;
      const idx = r * W + c;
      const nd = dist[from] + cost;
      if (nd < dist[idx]) {
        dist[idx] = nd;
        prev[idx] = from;
        push(idx, nd);
      }
    };

    let foundCell = -1;
    let expansions = 0;
    while (heap.length && expansions++ < 200000) {
      const cur = pop();
      if (goalCells.has(cur)) { foundCell = cur; break; }
      const c = cur % W, r = Math.floor(cur / W);
      // walk sideways: flat, or one tile up (auto-step)
      for (const s of [-1, 1]) {
        relax(cur, c + s, r, 1);
        if (r > 0) relax(cur, c + s, r - 1, 1.5);
      }
      // walk off an edge and fall
      for (const s of [-1, 1]) {
        const nc = c + s;
        if (nc < 0 || nc >= W) continue;
        if (sol(nc, r - 1)) continue; // floor at or above us: not a drop
        let rr = r;
        for (; rr < Math.min(H, r + MAX_FALL); rr++) {
          if (sol(nc, rr)) break;
        }
        if (rr < H && sol(nc, rr)) {
          // rows r-1..rr-1 must be open for the falling body
          let open = true;
          for (let k = r - 1; k < rr; k++) {
            if ((k >= 0 && sol(nc, k)) || doorCells.has(k * W + nc)) { open = false; break; }
          }
          if (open) relax(cur, nc, rr, rr - r + 0.5);
        }
      }
      // jump across gaps and over low walls (apex ~9 tiles clears ~5)
      for (const s of [-1, 1]) {
        const n1 = c + s;
        if (n1 < 0 || n1 >= W) continue;
        if (floorOK(n1, r) && bodyOK(n1, r)) continue; // can just walk there
        for (let d = 2; d <= MAX_JUMP; d++) {
          const nc = c + s * d;
          if (nc < 0 || nc >= W) break;
          // the arc must clear anything taller than a low wall
          let clear = true;
          for (let i = 1; i < d; i++) {
            const cc = c + s * i;
            for (let rr = r - 6; rr <= r - 1; rr++) {
              if (rr >= 0 && sol(cc, rr)) { clear = false; break; }
            }
            if (!clear) break;
          }
          if (!clear) break;
          for (const dr of [-2, -1, 0, 1]) {
            if (floorOK(nc, r + dr)) relax(cur, nc, r + dr, d + Math.abs(dr));
          }
        }
      }
      // ladders: any cell within the ladder column connects vertically;
      // climbing needs no floor, just clear body space around the ladder.
      for (const l of g.ladders) {
        const c0 = Math.floor(l.x0 / tw), c1 = Math.floor(l.x1 / tw);
        if (c < c0 || c > c1) continue;
        const rTop = Math.floor(l.y0 / th), rBot = Math.floor(l.y1 / th);
        if (r < rTop || r > rBot) continue;
        for (let rr = rTop; rr <= rBot; rr++) {
          if (rr === r || !bodyOK(c, rr)) continue;
          const idx = rr * W + c;
          const nd = dist[cur] + Math.abs(rr - r) * 1.4;
          if (nd < dist[idx]) {
            dist[idx] = nd;
            prev[idx] = cur;
            push(idx, nd);
          }
        }
      }
      // teleporters and platforms: press-action shortcuts
      for (const j of jumpTo.get(cur) || []) {
        relax(cur, j.to % W, Math.floor(j.to / W), j.cost);
      }
    }

    if (foundCell < 0) return;
    // Reconstruct the path.
    const cells = [];
    let cur = foundCell;
    let hops = 0;
    while (cur !== start && hops++ < N) {
      const c = cur % W, r = Math.floor(cur / W);
      cells.push({ c, r, isGoal: cur === foundCell });
      cur = prev[cur];
    }
    cells.reverse();
    this.path = cells;
    this.dist = dist;
    this.pathAt = this.t;
    this.pathGoal = goalCells.get(foundCell) || goal;
  }
}

