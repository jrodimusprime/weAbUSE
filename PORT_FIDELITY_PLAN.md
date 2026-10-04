# WEaBuse — original-fidelity audit plan

Goal: make `web/` a direct port of Abuse 0.8. Per the working rules, every
change below must be checked against the original source first and cite
file + symbol in the commit message. If something cannot be matched to the
original, stop and ask.

Source references:

- Engine (C++): `abuse-0.8/src/**` (git-ignored — grep with includeIgnoredFiles)
- Game logic (Lisp): `abuse-0.8/data/lisp/**`, `abuse-0.8/data/addon/**`
- Art/spec: `.spe` files under `abuse-0.8/data/**`

Port: `web/src/*.js` (plain ES modules). Serve `web/` and hard-reload after edits.

---

## Done (this session)

- **Machine gun bullet** (`SHOTGUN_BULLET`, guns.lsp `fire_object` type 10 +
  cop.cpp `sgun_ai`):
  - speed 15 + creator xvel/2, accelerated 6/5 every tick along a fixed angle,
    lifetime 6, 6 moves total (~179 px range vs ~105 px before);
  - damage 5 (was 3);
  - wall hit → `EXPLODE5`; hurtable hit → `EXPLODE3` + 5 damage + push of
    `(cos(angle)*10, sin(angle)*10)` (cop.cpp `sgun_ai`);
  - colour verified exact: `find_rgb 255 0 0` → palette 86, `find_rgb 150 0 0`
    → palette 88, read from the real `color_filter` table in `backgrnd.spe`
    (clisp.cpp case 172 → `ColorFilter::Lookup`);
  - no spread (the original has none for the player).
- **Muzzle position** (`cop.cpp` `player_fire_weapon` + `small_fire_off` /
  `large_fire_off`): the muzzle is now the per-frame table offset from the
  character, with the +4 x shift when the body faces left. `small_fire_off`
  for all weapons except ROCKET/DFRIS which use `large_fire_off`.
- **Aim frame + fire angle** (`cop.cpp` `top_ai`): the upper-body frame is
  picked by angular distance to the pointer, pivoted at the gun grip
  (`fire_off[12], fire_off[1]`); the fire angle (`point_angle`) is measured
  from the barrel tip to the pointer (or the frame's own angle when the
  pointer is within 45/40 px of the muzzle).
- **Body direction** no longer follows the mouse; it follows movement input,
  like `game_object::mover` (objects.cpp). The gun still aims at the mouse.
- **Fire delays** are the original `fire_delay1` tick counts
  (cop.cpp ufuns): MGUN 3, PLASMA 2, GRENADE/FIREBOMB/ROCKET/DFRIS 6,
  LSABER 1 (× 1/15 s).

## In progress / next

1. **Grenades** (guns.lsp type 2): original is `set_course angle 20` plus the
   creator's full velocity, with gravity from the GRENADE object def and the
   grenade's own ai/fuse. Port uses speed 13, gravity 2, life 40 — replace
   with the original values (check GRENADE def in common.lsp + its ai).
2. **Plasma** (guns.lsp type 4 `PLASMAGUN_BULLET`): near-instant
   (`set_course angle 200`), resolves the first hit immediately, damage 10,
   EXPLODE5 on wall / EXPLODE3 + push on victim. Port flies at 34 px/tick
   with damage 9.
3. **Rocket** (guns.lsp type 3): starts at speed 5, accelerates to max 14
   (player-fired), homes onto a linked target if `can_see`, frame chosen by
   `set_frame_angle`. Port has constant speed and no homing.
4. **Death frisbee** (guns.lsp type 6 `DFRIS_BULLET` + `dfris_ai`): speed 25,
   boomerang return behaviour. Port treats it as a plasma bolt.
5. **Light saber** (guns.lsp type 7 `LSABER_BULLET`): `set_course angle 45`,
   immediate first-hit damage 30 with push, plus the `(tick&7)-8` angle wiggle
   from `lsaber_ufun` (cop.cpp). Port treats it as plasma.
6. **Firebomb** (guns.lsp type 5): `set_course angle 20` + creator velocity,
   fire radius from the FIREBOMB def/ai.
7. **Unactive shields**: `sgun_ai` pops EXPLODE5 without damage when the hit
   object has `CFLAG_UNACTIVE_SHIELD` and aistate 0. The port does not model
   shield entities yet — add when shields appear (level Lisp).
8. **First-move timing**: guns.lsp calls `(sgun_ai)` once at fire time, so the
   bullet's first move is instant. The port moves on the next tick (same total
   moves, 1 tick late) — acceptable, or replicate the immediate call.

## Audit checklist (top to bottom)

### Player physics — cop.cpp (`player_move`, `do_ai` case 9, abilities in people.lsp)
- [ ] run accel/top speed (`start_accel`, `run_top_speed` abilities), FAST/SNEAKY/HEALTH powers
- [ ] jump (`jump_yvel`?), gravity, max fall; wall kick off ceilings (cop.cpp ceiling squash)
- [ ] climbing: climb area flag, ladder up/down rates
- [ ] fly power: rise/fall rates, cloud effect, gravity handling
- [ ] death/respawn: `do_damage` on player, respawn point, invulnerability window
- [ ] `just_fired` bright tint (people.lsp `bright_tint`) — port uses setBright(1.8) on the player only; compare the actual tint palette

### Collision / movement — objects.cpp (`mover`, `try_move`, `bmove`), collide.cpp
- [ ] try_move checks &1..&7 (port has "moving edge only" vertical checks; verify against all cases)
- [ ] all_boundary_setback, foreground_intersect (muzzle line-of-sight pullback)
- [ ] platforms (one-way), ladders, `blocked` flags, crush damage
- [ ] bmove used by bullets/DFRIS/LSABER (excludes linked creator, sub-pixel)

### Enemies — ant.cpp/ant.lsp, flyer.lsp, people.lsp, guns.lsp (`spray_gun_ai`)
- [ ] ant states, fire/pounce/dodge probabilities and ranges (done), damage values, jump velocities
- [ ] roof ants (ANT_CEIL_SHOOT etc.)
- [ ] turret/spray gun fire types (type 1 yellow bullet colours), aim, sensor look
- [ ] flyers (flyer.lsp), juggers, guns/hidden-wall behaviours, AFT fires
- [ ] `do_damage` hurt functions (flinch, sound, team check / no friendly fire)

### Weapons firing — cop.cpp ufuns + guns.lsp
- [ ] per-weapon fire delays (done), ammo counters, switch weapon top part
- [ ] items 2/4/3/6/7/5 from the list above
- [ ] strait rocket (type 9) if used anywhere

### Explosions / damage — explo.lsp (`do_explo`, `do_small_explo`, `hurt_radius`), objects.cpp `do_damage`
- [ ] hurt_radius falloff `(r-d)*m/r` (done for walls), max_damage variant, exclusions (creator)
- [ ] do_explo light (`EXP_LIGHT`), sound, frame_panic skip
- [ ] grenade/rocket explosion radii + damage per Lisp calls

### Level loading — level.cpp, loader2.cpp, level Lisp (addon/aliens)
- [ ] all object types spawn (HIDDEN_WALL*, sensors, gates, doors, teleports, save stations)
- [ ] tile solidity table (techno.spe damage column) — verify each tile's solidity matches
- [ ] level start position, checkpoints, area light definitions (applyArea)
- [ ] difficulty (hardness.lsp) hp multipliers

### Status bar / UI — statbar.cpp, sbar
- [x] original statbar.spe layout, scale, hover/click select, ammo counters
- [ ] numpad / weapon digit graphics if any remain, HP warning flash

### Rendering — view.cpp, objects.cpp (`draw`), light.cpp
- [ ] palette effects: bright_tint / dark_tint / predator / sneaky, per-object
- [ ] lighting remap tables (light.cpp `remap`), headlight, EXP_LIGHT, lamp objects
- [ ] transparency/transp effects, gamma (gamma.cpp)
- [ ] camera pan (director.cpp `pan`), camera clamp

### Sound — sound.cpp, sfx
- [ ] positional volume/attenuation, per-sound volumes (127 scale), looped sounds (flyer)

### Save/load, demo, network
- [ ] loadgame.cpp / checkpoints / saved games (probably out of scope for the web port — ask if needed)
- [ ] demo playback (out of scope unless requested)

## Verification per item

1. Read the cited source; copy exact numbers.
2. Test in the browser (http://localhost:8000, hard reload) with `window.game`.
3. Screenshot + pixel-check any visual change (Rule 3).
4. Commit with the source citation in the message.
