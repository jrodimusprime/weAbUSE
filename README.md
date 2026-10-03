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

Below the screen a weapon panel shows every gun with its ammo, highlighting the current one; click an owned weapon to switch.

## Status

Working: all 22 levels, tiles and parallax, per-pixel lighting with ambient areas and dimmer switches,
the player, seven weapons, ants, flyers, gun turrets, juggernauts, bombs, mines, lava, boulders, doors,
switches, logic gates, sensors, elevators, ladders, springs, teleporters, force fields, save stations,
training hints with the original voice-overs, speed and flight power-ups, and sound effects.

Not done yet: robots (`ROB1`), the boss, music, saving and menus.

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
