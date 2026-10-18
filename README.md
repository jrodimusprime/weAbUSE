# Abuse (WebGL)

A native WebGL port of **Abuse**, the dark 2D side-scrolling run-and-gun by Crack dot Com.
It is not an emulator or a WebAssembly build: the engine is re-implemented in plain JavaScript
and reads the original game data files directly.

**Play:** https://jrodimusprime.github.io/weAbUSE/

## Run locally

```sh
cd web
python3 -m http.server 8000
# open http://localhost:8000/
```

Any static file server works. The game fetches about 22 MB of art, levels and sound on first load.

## Controls

| Action | Keys |
| --- | --- |
| Move | Arrow keys / A, D |
| Jump | Space, Up, W or Z |
| Use switch, elevator, door, save station | Down, S or E |
| Aim / fire | Mouse / left click, F or Ctrl |
| Weapon | 1-7, Insert / Right Ctrl, or mouse wheel |
| Special power (once picked up) | Right mouse button |
| Lights on/off | L |
| God mode | G or the button (survives death, runs at speed-power pace, infinite ammo, counts would-be deaths per level) |

On mobile, tapping the screen aims and fires, and on-screen buttons below the canvas provide move (▲ ▼ ◀ ▶), **Use**, **Special** power and **Lights**.

Below the screen the original status bar (`art/statbar.spe`) is drawn — health, each owned gun's icon (bright for the current weapon) and its ammo; click a gun to switch.

## Status

Working: all 22 levels, tiles and parallax, per-pixel lighting with ambient areas and dimmer switches,
the player, seven weapons, ants, flyers, gun turrets, juggernauts, cleaner robots, bombs, mines, boulders, doors,
switches, logic gates, sensors, elevators, ladders, springs, teleporters, force fields, save stations,
training hints with the original voice-overs, speed and flight power-ups, the final boss, the ending, and sound effects.

Not done yet: music, saving and menus. Known bugs and fidelity work are listed under [TODO](#todo).

## Jared rules

This port copies the original Abuse 0.8 wherever it can. The exceptions below
are deliberate: places where I don't like how the original works, so the port
does something else on purpose. They are not bugs, and they should not be
"fixed" back to the original.

- **Level 5, the sealed room with the jetpack (bottom right).** In the
  original, picking up the heart in that room shuts its door at (2130,1394)
  for good, and the only way out is to take the jetpack and fly back up the
  shaft you fell down. Here, once you have picked up the jetpack, that door
  opens again and stays open.

## How the demo levels were passed

The Demo button plays a recorded playthrough of all 22 levels, in order, from
level 0 to the ending. Nobody played it. The routes were found by a search
program, `web/tools/search.mjs`, which runs the real game code in Node.js
without a browser, much faster than real time. This section explains how it
works and what had to be taught to it along the way.

### Why a search and not a trained AI

The first attempt trained a neural network with reinforcement learning (PPO).
It needed hours per level and still stalled at the first puzzle it did not
understand. Two facts about the game make something simpler possible:

- **The game is deterministic.** The same inputs from the same moment always
  give the same result: there is no randomness the program does not control.
- **Any moment can be saved and restored exactly**, including every object's
  state, timers and the random number generator.

So nothing has to *learn* to play well. A program can try things, keep
whatever got somewhere new, and go back to it. That method is known as
Go-Explore.

### The loop

1. **An archive of situations.** The search keeps every distinct situation it
   has reached. Two moments count as the same situation (a "cell") when all of
   these match:
   - the player is in the same 36 by 30 pixel square of the level;
   - the same "keys" are still to be dealt with (switches not yet pressed,
     creatures not yet killed; see the compass below);
   - the same number of breakable walls is left;
   - the player holds the same special power (jetpack, speed) and the same
     number of weapons;
   - any delay gate that is counting down has got as far (so standing by a
     door that opens after a few seconds counts as progress).
2. **Pick a cell, restore it, play a burst.** A burst is 90 decisions, about
   six seconds of play (a quarter of bursts are three times longer). Each
   decision is a move, jump, use and fire combination held for 4 game steps.
3. **Look every 3 decisions.** If the player is standing, climbing or flying
   (somewhere it could be put down again) and the situation is new, it is
   saved with the actions that led there. A situation already in the archive
   is replaced only if this way of reaching it is healthier, or as healthy and
   quicker.
4. **Dying ends the burst** and saves nothing.
5. **Repeat** until a burst walks into the level's exit.

It never replays the easy part of a level, and it cannot lose ground: once it
has stood past an obstacle, it can start from there for ever.

### What a burst plays

Bursts are random but shaped like a player's input: moving most of the time
(sometimes holding one direction for the whole burst), jumping about a third
of the time, the use key now and then, and firing mostly when an enemy is
near. Aiming is automatic: at the nearest enemy it can see, else a switch ball
not yet set off (from as far as its weapons with ammunition carry: about 180
pixels for the machine gun, more for rockets and plasma), else a breakable
wall. The weapon is chosen for the target (rockets for far or tough ones).

Some things random input almost never does, so they are moves of their own:

- **Climbing a ladder**: straight up or down, held for seconds.
- **Riding a lift**: standing still on the deck while it moves (some rides
  last half a minute).
- **Flying with the jetpack**: holding a chosen height while heading one way,
  to get across things like level 5's arrow strips.
- **Flying to a switch out of reach**: steering up to a switch that hangs
  overhead and pressing use there (level 16's four switches on a wall).

Two input rules stop it undoing its own work: it does not press a two-way
switch that is already on, and it does not press use on a lift it has just
started.

### How it chooses where to explore from: the compass

Each saved situation gets a reading, its distance from finishing the level.
Cells are picked favouring those tried least, those found recently, and those
with the best reading. Half of the picks are instead spread evenly across the
level's states (which keys are done), so that one dead end, such as a room
that seals behind the player, cannot swallow the whole run.

The reading is worked out on a map of the level's tiles:

- **Walking distance to the exit**, by shortest path over moves the player can
  make: walk, step up, jump onto ledges up to four tiles high, drop, ladders,
  lifts, springs and teleporters. Doors, breakable walls and force fields are
  passable at a cost, so the distance still falls as the player nears them.
- **If no walking route exists** because doors are locked, the reading is the
  number of keys left, then the distance to the nearest one. Pressing a switch
  is a big step forward; walking towards the next is steady progress.

**Keys** are what has to be dealt with before the way opens. The compass reads
the level's own wiring (switches, sensors, logic gates and death sensors, as
placed in the level file) to decide them:

- A switch that is off is a key. A switch that the level starts with on and
  that holds force fields or pushers up is a key to turn *off* (level 10).
- A creature counts only if its death opens a way: if its death sensor drives,
  through its gates, a door, wall, force field, pusher, lift, object mover,
  teleporter or exit. Level 11's chain of 31 hidden ants, each letting out the
  next and opening nothing, was otherwise a hunt for every ant.
- A creature that cannot be reached yet is replaced by what wakes it. Level
  15's turrets only become targetable after switch 11 is on. Level 16's roof
  ants over trap door 515 drop only when their flyers are shot, and its funnel
  flyers come out only when the heart above them is taken.
- A hidden wall wired to a switch is locked like a door (it cannot be shot
  until what it is wired to comes on), and the sensors its gates wait on
  become places to go.

And keys are made for at the places they can be dealt with, not where the
object sits:

- a **switch ball** at the places it can be seen from within weapon range,
  standing, at the top of a jump, or from a lift's stops (level 15's third
  ball hangs at the end of a slanting slit);
- a **sensor** at the places where the player's feet, with a jump, come inside
  its box (level 15's sits on top of a turret);
- a **teleporter or exit stored inside the rock** where its object mover will
  put it (level 10's door out of the juggernaut room).

### What comes out

Following a winning situation back through the archive gives a chain of
pieces, each one "restore this saved state and play these actions". That chain
is the recording, stored in `web/data/ppo-demo.json` (about 1,200 pieces for
the whole game, kept small by storing each saved state as its difference from
the level's start, compressed). Playing a piece back reproduces it exactly.

### Playing the game in order

The demo is one playthrough. `--in-order` searches level 0, then starts level
1 with the weapons and ammunition level 0's route ended with, and so on (level
16 cannot be won with the machine gun alone). Several searches race on each
level with different random seeds; the first to win is kept and the others are
stopped. Each search keeps at most `--max-cells` saved situations (30,000 by
default, about 1 GB of memory), letting the least promising go.

### When the game changes

A recording belongs to one version of the game. A fix, even one that brings
the port closer to the original, can make old routes drift by a pixel, and
the regression test then fails. `--repair` mends such a route instead of
starting over: the pieces that still replay are kept, the first that no longer
meets the next is played as the game now plays it, and the search carries on
from where that ends. Level 11 was repaired this way after the turret fix.

### Running it

```sh
node web/tools/search.mjs --in-order --jobs 2            # the whole game from level 0, weapons carried
node web/tools/search.mjs --in-order --from 16 --upto 16 # just some levels, in order
node web/tools/search.mjs --only 11 --repair             # mend a route a game change has broken part way
node web/tools/search.mjs --write-demo                   # rebuild the demo from train-out/levels/
node web/tools/regress.mjs                               # check every won level still replays to its exit
```

Progress goes to `train-out/` (one log per level, and for a level that is not
passed, a map of where the search went and what it still had to do).

### What it found out about the port

Most levels fall in seconds to a few minutes; level 16 took about two hours.
Where a level would not fall, asking why usually found a bug in the port, not
in the search: a timed switch that reset too soon, a spring that was too weak,
teleporter pads you fell through, a juggernaut that never walked, and turrets
whose whole picture, not their outline, blocked a passage on level 15. Lava
that seemed to do no damage turned out to be the player's whole hurt reaction
missing (the flinch that stops you for a moment, the red flash, the cry of
pain): lava's 3 a time on easy went unnoticed among level 0's hearts.

(The file name `ppo-demo.json`, and `web/src/ppo.js`, which now holds the
compass, are left over from the PPO attempt. Its controls are still in the
page, hidden.)

## TODO

Bugs:

- **Elevators can clip into the player and lock them in place.**
- **The digger machines do no damage.**

Closer to the original (found while checking level 15; each changes how most
levels play, so every level's recorded route has to be searched again, or
repaired, after it):

- **Shut guns take no damage.** In the original a turret or spray gun takes
  no damage while it is closed (`weapons.lsp` `guner_damage`). Here the blast
  of level 15's floor walls destroys three turrets that should survive.
- **A blast spares the gun that fired it.** The original rocket or grenade
  explosion leaves its shooter alone (`explo.lsp` `do_explo`, which passes the
  shot's first link to `hurt_radius`). Here turrets blow themselves up with
  their own rockets.
- **No sound when a switch ball is hit.** The original `sball_damage` plays
  none. (Removing it changes the random number sequence, so it also breaks
  recordings.)
- **Every solid object blocks by its outline.** The original stops the player
  at a blocking object's boundary (`level.cpp` `all_boundary_setback`), not its
  whole picture. Turrets, lifts, boulders and the cleaner robot do this here;
  the rest still use the picture.

## Tests

Every level that has been completed by a recorded run is a regression test:

```sh
node web/tools/regress.mjs
```

It replays each winning run (the ones the Demo button plays) in the real game
code and fails if any no longer reaches its exit. It is a project rule that
this runs after every commit and passes before anything is pushed; enable the
hook that does it with `git config core.hooksPath .githooks`.

`node web/tools/audit.mjs` works every wired puzzle, lift, teleporter and exit
in all 22 levels.

## How it works

- `web/src/spec.js` reads the `SPEC1.0` archives and decodes images, tiles and sprites.
- `web/src/lisp.js` is a small Lisp reader used only to read the game's `def_char` data
  (sprite names per state, abilities, ranges). Game behaviour is not run from Lisp.
- `web/src/behaviors.js` and `web/src/weapons.js` are native ports of the object AI and weapons.
- `web/src/renderer.js` draws 8-bit paletted sprites through one index atlas and a palette lookup
  in a fragment shader, so the original palette is kept.

## Credits

This project exists because of the people who made Abuse and kept it alive. Thank you.

- **Crack dot Com** for Abuse itself, and for releasing its source code into the public domain.
  Dave Taylor and Jonathan Clark created it. It is a remarkable game: fast, tense, atmospheric,
  with a built-in level editor and a Lisp scripting layer.
- **Bobby Prince** for the sound effects and music that give the game its mood.
- **Sam Hocevar**, maintainer of Abuse-SDL, **Anthony Kruize** (SDL port), **Jochen Schleu**
  (music playback), **Joris Beugnies** (OpenGL support) and the other Abuse-SDL contributors
  listed in its `AUTHORS` file.
- **Justin Cassidy** and the fRaBs community for the free levels and art, and **Bungie** for the
  Mac version, all merged into the Abuse 0.8 data this port loads.

## License

- Abuse's original source code is public domain (Crack dot Com). The Abuse-SDL parts are
  licensed separately: `src/sdlport` under GPL 2+ (Anthony Kruize) and `src/lol` under the WTFPL
  (Sam Hocevar). Neither is part of this repository.
- The game data in `web/data` (art, levels, Lisp scripts) is the Abuse 0.8 data. Its copyright is
  Crack dot Com and others (see above) and it was released into the public domain, though the
  Abuse-SDL project notes that the licensing terms for the data are still being sorted out.
- The sound effects in `web/data/sfx` are copyright Bobby Prince, redistributed with his permission
  and unmodified. Please keep them unmodified.
- The JavaScript in `web/src` is new code written for this port. It has no license file yet.

Abuse-SDL, from which the data comes: http://abuse.zoy.org/

This is an unofficial fan port and is not affiliated with or endorsed by Crack dot Com or any
of the people named above.
