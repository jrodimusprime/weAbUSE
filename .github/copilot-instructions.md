# WEaBuse — working rules

## RULE 1 (highest priority): always check the original Abuse source first
This project is a faithful re-implementation of **Abuse 0.8**. Before changing,
adding or "fixing" any gameplay, rendering, physics, animation, sound or UI
behaviour, **read the original source and copy its behaviour**:

- Engine (C++): `abuse-0.8/src/**` — e.g. `objects.cpp`, `cop.cpp`, `level.cpp`,
  `items.cpp`, `collide.cpp`, `statbar.cpp`, `game.cpp`.
  NOTE: `abuse-0.8/` is git-ignored, so search it with `includeIgnoredFiles: true`.
- Game logic (Lisp): `abuse-0.8/data/lisp/**` and `abuse-0.8/data/addon/**`
  — `people.lsp`, `ant.lsp`, `guns.lsp`, `weapons.lsp`, `doors.lsp`, `common.lsp`.
- Art/spec: `.spe` files under `abuse-0.8/data/**` (and `web/data/**`).

Rules of thumb:
- Cite the source file + symbol (e.g. `cop.cpp player_move`) in the commit message
  or reply for every behavioural change.
- Prefer the original's **exact numbers**: offsets, radii, frame rates, probabilities.
- If the original and the current port disagree, the original wins unless the user
  explicitly says otherwise.
- If the original has no such feature, say so — do not invent behaviour.

## RULE 2: ask before violating Rule 1
If a change cannot or should not be matched to the original source (time/scope,
renderer limitations, user explicitly asking for something different, or the
original genuinely lacks the feature), **stop and ask the user first**. Do not
silently diverge.

## RULE 3: verify UI/visual changes with a screenshot
After any visual change, take a screenshot (full page, not just an element) and
confirm it renders correctly before reporting done. Check element bounding boxes
for unexpected sizes.

## RULE 4 (key bindings): never clash with game inputs
Never add keyboard shortcuts that overlap the game's existing controls:
arrows/WASD (move), Space/Up (jump), Down/E (use/action), 1-7 or wheel
(weapons), mouse buttons (fire/special), L (lights), G (god). New hotkeys must
use unused keys (e.g. F-keys) or be button-only. A clash reads as a gameplay
bug (e.g. pressing D started the demo while moving right).

## RULE 5 (regression): a level that has been won stays won
Every level that has been completed by a recorded run is a regression test.
The runs are the ones in `web/data/ppo-demo.json`; the levels are listed in
`web/tests/won-levels.json`, which is only ever added to.

- **After every commit, run `node web/tools/regress.mjs`** (about half a
  minute). It replays each winning run in the real game code and fails if any
  no longer reaches its exit. The post-commit hook in `.githooks/` does this
  by itself once enabled: `git config core.hooksPath .githooks`.
- A failure means the game no longer plays as it did when the run was
  recorded. If that was not intended, it is a bug: fix it.
- If it was intended (a fidelity fix changes how things move, so old
  recordings drift), search the affected levels again before pushing:
  `node web/tools/search.mjs --recheck && node web/tools/search.mjs --parallel`.
- **Do not push with the regression failing**, and never take a level off the
  won list to make it pass.

## Project notes
- Web port lives in `web/` (plain ES modules, no build step).
- Serve with `cd web && python3 -m http.server 8000` → http://localhost:8000/
- `web/index.html` loads every script through an import map with a fresh
  `?v=` stamp, so a plain reload always runs the current code. A new file in
  `web/src/` must be added to the `FILES` list there.
- Tools (all run the real game headlessly, `node web/tools/<name>.mjs`):
  `regress` (Rule 5), `audit` (works every wired puzzle, lift, teleporter and
  exit), `search` (finds and records a route through each level).
