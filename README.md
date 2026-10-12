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
the player, seven weapons, ants, flyers, gun turrets, juggernauts, cleaner robots, bombs, mines, lava, boulders, doors,
switches, logic gates, sensors, elevators, ladders, springs, teleporters, force fields, save stations,
training hints with the original voice-overs, speed and flight power-ups, the final boss, the ending, and sound effects.

Not done yet: music, saving and menus.

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

The Demo button plays a recorded playthrough. Nobody played it: the routes
were found by a search program, `web/tools/search.mjs`, running the real game
code without a browser.

**The idea.** The game is deterministic, and any moment of it can be saved and
restored exactly. So there is no need to train something to play well and hope
it gets through a level. The search explores instead (the method is known as
Go-Explore):

1. It keeps an archive of every distinct situation it has reached. Two moments
   count as the same situation if the player is in the same small square of the
   level, the same switches are on, the same watched creatures are dead, the
   same breakable walls are left, and the player holds the same power and
   weapons.
2. It picks one situation from the archive, restores the game to it, and plays
   a burst of about six seconds from there.
3. Anything new it reaches is saved to the archive. Dying saves nothing.
4. It repeats until the player takes the level's exit.

It never replays the easy part of a level, and it cannot lose ground: once it
has stood past an obstacle it can start from there for ever.

**What it plays.** Bursts are random, but shaped like a player's input: mostly
on the move, the action key now and then, firing when something is in range.
Aiming is automatic. A few things random play cannot do are offered as moves of
their own: climbing a ladder steadily, standing still on a moving lift, and
flying level with the jetpack.

**How it chooses where to explore from.** It favours situations tried least
often, recent ones, and ones nearer the exit by a "compass": the walking
distance to the exit, or, while doors are still locked, the number of switches
and watched creatures left to deal with. Half its picks are spread evenly
across the states of the level, so that a dead end cannot absorb the whole run.

**What comes out.** Following a winning situation back through the archive
gives a chain of pieces, each one "start from this saved state and play these
actions". That chain is the recording, stored in `web/data/ppo-demo.json`.
Because the game is deterministic, playing a piece back reproduces it exactly.
For a level the search could not pass, the demo plays its furthest attempt and
then shows "AI gave up".

**Running it.**

```sh
node web/tools/search.mjs --parallel   # search every level that has no route, several at once
node web/tools/search.mjs --recheck    # after changing the game: drop the routes that no longer replay
node web/tools/regress.mjs             # check that every level won before is still won
```

**What it is good and bad at.** Most levels fall in seconds to a few minutes.
It struggles where a level needs a long exact sequence with nowhere to stand in
between, many switches in a particular order, or a target out of weapon range.
It is also a strict test of the port: several times a level it could not pass
turned out to be a bug in the port (a switch that reset too soon, a spring that
was too weak, teleporter pads you fell through), found by asking why it was
stuck.

**A recording belongs to one version of the game.** Any change to how the game
plays, including a fix that brings it closer to the original, can make old
recordings drift, and those levels then have to be searched again. That is
what the regression test below is for.

(The file name `ppo-demo.json`, and `web/src/ppo.js`, are left over from an
earlier attempt that trained a neural network with PPO. It passed a few levels
slowly and was replaced by the search. Its controls are still in the page,
hidden.)

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
