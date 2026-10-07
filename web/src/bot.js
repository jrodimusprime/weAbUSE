// Test-harness autopilot for "demo mode".
//
// NOTE: the original Abuse has no autopilot player (demo.cpp only records and
// replays real input packets, and -nodelay just removes frame pacing). This
// module is a port-only test aid: it synthesises the same keyboard/mouse input
// a human would produce, through the existing input fields on Game. It changes
// no gameplay, physics or AI code paths.
//
// Strategy (all constants in px, game is 320x200 at 15 Hz logic):
//   - head for the nearest NEXT_LEVEL object (level exit)
//   - jump gaps and steps, wander back the other way when stuck
//   - climb ladders when they are near the exit's column or when stuck
//   - press the action key near switches, teleporters, save stations and
//     platforms (pulsed, so latch-based objects see press+release)
//   - shoot the nearest enemy in line of sight; when stuck against a solid,
//     shoot that solid if it is destructible (hidden walls / blocks)

const ENEMY_AI = new Set([
  'ant_ai', 'flyer_ai', 'track_ai', 'spray_gun_ai', 'jug_ai',
  'crack_ai', 'lightin_ai', 'bolder_ai', 'air_mine_ai', 'mine_ai',
]);

const ACTION_AI = new Set([
  'switcher_ai', 'switch_once_ai', 'switch_delay_ai', 'tpd_ai', 'tp2_ai',
  'restart_ai', 'platform_ai', 'next_level_ai',
]);

const HALF_W = 6;  // player half width, matches game.js
const BODY_H = 29; // player collision height, matches game.js

export class Bot {
  constructor() {
    this.t = 0;          // sim seconds
    this.lastT = 0;      // last progress check time
    this.lastGd = null;  // goal distance at the last progress check
    this.stuckT = 0;     // seconds without closing in on the exit
    this.climbStuckT = 0;// same, while climbing
    this.avoidT = 0;     // seconds to wander before re-aiming at the exit
    this.dir = 1;        // wander direction
    this.pulseT = 0;     // action-key hold timer
    this.saveCd = 0;     // don't camp on save stations
    this.climbUp = true;
  }

  step(g) {
    const p = g.player;
    const dt = 1 / 60;
    this.t += dt;
    this.saveCd = Math.max(0, this.saveCd - dt);

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
    if (g.demoTimeoutTicks && (g.tickCount || 0) - (g.demoStartTick || 0) > g.demoTimeoutTicks) g.onDemoTimeout?.();

    // Level exit to head for (levels can have several; pick the nearest).
    let goal = null, gd = Infinity;
    for (const e of g.entities) {
      if (e.dead || e.ai !== 'next_level_ai') continue;
      const d = Math.abs(e.x - p.x) + Math.abs(e.y - p.y);
      if (d < gd) { gd = d; goal = e; }
    }
    const atGoal = !!goal && Math.abs(goal.x - p.x) < 36 && Math.abs(goal.y - p.y) < 44;
    const touchingGoal = !!goal && g.touchesPlayer(goal);

    // Stuck detection: no progress towards the exit for a while.
    // (Measured against the goal distance, so jumping in place against a wall
    // still counts as stuck — unlike a pure movement check.)
    if (!atGoal && this.avoidT <= 0) {
      if (this.t - this.lastT >= 0.5) {
        const improved = this.lastGd !== null && gd < this.lastGd - 4;
        if (!improved) {
          this.stuckT += 0.5;
          if (p.climbing) this.climbStuckT += 0.5;
        } else {
          this.stuckT = 0;
          this.climbStuckT = 0;
        }
        this.lastGd = gd;
        this.lastT = this.t;
      }
    } else {
      this.stuckT = 0;
      this.climbStuckT = 0;
      this.lastGd = gd;
    }
    this.avoidT = Math.max(0, this.avoidT - dt);

    // Aim the wander direction at the exit unless recovering from a flip.
    if (this.avoidT <= 0 && goal && Math.abs(goal.x - p.x) > 6) this.dir = Math.sign(goal.x - p.x);

    // ---- aim & fire ----
    let target = null, best = Infinity;
    for (const e of g.entities) {
      if (e.dead || e.hidden || !e.shootable || !ENEMY_AI.has(e.ai)) continue;
      const dx = Math.abs(e.x - p.x), dy = Math.abs(e.y - p.y);
      if (dx > 330 || dy > 210) continue;
      if (!g.sees(p.x, p.y - 15, e.x, e.y - 15)) continue;
      const d = dx + dy;
      if (d < best) { best = d; target = e; }
    }
    let wallTarget = null;
    if (!target && this.stuckT > 1.5) {
      // Stuck against destructible scenery (hidden walls, blocks): shoot it.
      for (const s of g.solids) {
        const e = s.e;
        if (!e || e.dead || !e.shootable) continue;
        const cx = (s.x0 + s.x1) / 2, cy = (s.y0 + s.y1) / 2;
        if (Math.abs(cx - p.x) > 200 || Math.abs(cy - p.y) > 120) continue;
        if (Math.sign(cx - p.x) !== this.dir) continue;
        if (!g.sees(p.x, p.y - 15, cx, cy)) continue;
        target = e;
        wallTarget = e;
        break;
      }
    }

    // Wander the other way when stuck — but finish destroying a wall first.
    if (!wallTarget && this.stuckT > 4 && this.avoidT <= 0) {
      this.dir = -this.dir;
      this.avoidT = 6;
      this.stuckT = 0;
    }

    // ---- action pulse: switches, teleporters, save stations, platforms ----
    let wantAction = touchingGoal;
    if (!wantAction && !p.climbing && !p.ladderExit) {
      for (const e of g.entities) {
        if (e.dead || !ACTION_AI.has(e.ai)) continue;
        if (Math.abs(e.x - p.x) > 22 || Math.abs(e.y - p.y) > 30) continue;
        const ai = e.ai;
        if (ai === 'tpd_ai' || ai === 'tp2_ai') {
          const d = e.links[0];
          const nxt = d && goal ? Math.abs(goal.x - d.x) + Math.abs(goal.y - d.y) : Infinity;
          if (!(nxt < gd - 60 || this.stuckT > 2)) continue; // only when it helps
        } else if (ai === 'restart_ai') {
          if (this.saveCd > 0) continue;
          this.saveCd = 5; // press once, then move on
        } else if (ai === 'platform_ai') {
          if (!(e.aistate === 0 && g.touchesPlayer(e) && goal && Math.abs(goal.y - p.y) > 80)) continue;
        } else if (e.aistate !== 0) {
          continue; // switches: press only while off
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
      || (goal && Math.abs(goal.x - (ladder.x0 + ladder.x1) / 2) < 90 && Math.abs(goal.y - p.y) > 60));

    if (p.climbing || needClimb) {
      if (!p.climbing) {
        const cx = (ladder.x0 + ladder.x1) / 2;
        this.climbUp = goal ? goal.y < p.y - 12 : true;
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
    } else if (atGoal || !wantAction) {
      // ---- walk / jump (towards the exit when we are on it) ----
      const dir = atGoal && Math.abs(goal.x - p.x) > 2 ? Math.sign(goal.x - p.x) : this.dir;
      g.keys.add(dir > 0 ? 'ArrowRight' : 'ArrowLeft');
      const ahead = p.x + dir * 26;
      const groundHere = g.solidAt(p.x, p.y + 4);
      const groundAhead = g.solidAt(ahead, p.y + 4) || g.solidAt(ahead + dir * 6, p.y + 4);
      const gap = p.ground && groundHere && !groundAhead;
      const wallAhead = g.boxHits(p.x + dir * 10, p.y, HALF_W, BODY_H, null);
      const goalUp = atGoal && goal && goal.y < p.y - 14;
      // Don't jump while shooting a wall — keep the gun level with the target.
      if (p.ground && !wallTarget && (gap || (wallAhead && this.stuckT > 0.3) || this.stuckT > 1.4 || goalUp)) g.keys.add('Space');
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
}
