#!/bin/sh
# Keeps searching for routes through the levels that have none, for hours,
# unattended.
#
#   sh web/tools/search-loop.sh [hours]        (default 8)
#
# Each pass searches every level without a winning run, several at once, with
# a different run of the dice from the pass before; a level gets up to an hour
# per pass. After every pass the demo file is rebuilt and the regression test
# run. Nothing is committed or pushed.
#
# What to read afterwards (all under train-out/):
#   search-status.txt   the current state, rewritten after every pass
#   search-long.log     one line for every level passed or given up on, with the time
#   levels/levelNN.log  each level's own progress
# To publish what it found:
#   node web/tools/regress.mjs && git add web/data/ppo-demo.json web/tests/won-levels.json && git commit -m "Demo: more levels" && git push
cd "$(dirname "$0")/../.." || exit 1
HOURS=${1:-8}
END=$(( $(date +%s) + HOURS * 3600 ))
LOG=train-out/search-long.log
STATUS=train-out/search-status.txt
mkdir -p train-out
say() { echo "$(date '+%Y-%m-%d %H:%M')  $*" >> "$LOG"; }

status() {
  {
    echo "Search status, written $(date '+%Y-%m-%d %H:%M')  ($1)"
    echo
    node -e "const d=JSON.parse(require('fs').readFileSync('web/data/ppo-demo.json','utf8'));const all=[...Array(22).keys()];console.log('Levels with a winning run ('+d.passed.length+' of 22): '+d.passed.join(', '));console.log('Levels the search has not passed: '+all.filter(l=>!d.passed.includes(l)).join(', '))"
    echo
    echo "Regression test (every level won before must still be won):"
    node web/tools/regress.mjs 2>&1 | grep -E "^FAIL|winning runs still win"
    echo
    echo "Results so far, newest last:"
    grep -E "PASSED|NOT passed" "$LOG" | tail -40
  } > "$STATUS" 2>&1
}

say "started: up to $HOURS hours"
# a search already running is left to finish first, so two do not work the same levels
# (checked twice, half a minute apart: one that runs in passes pauses briefly between them)
while pgrep -f "tools/search.mjs" > /dev/null || { sleep 30; pgrep -f "tools/search.mjs" > /dev/null; }; do sleep 20; done

SEED=101
PASS=0
while [ "$(date +%s)" -lt "$END" ]; do
  PASS=$((PASS + 1))
  LEFT=$(( (END - $(date +%s)) / 60 ))
  PER=60; [ "$LEFT" -lt "$PER" ] && PER=$LEFT
  [ "$PER" -lt 5 ] && break
  say "pass $PASS: seed $SEED, up to $PER minutes a level"
  node web/tools/search.mjs --parallel --jobs 6 --per-level "$PER" --seed "$SEED" > train-out/search-parallel.log 2>&1
  grep -E "PASSED|NOT passed|stopped" train-out/search-parallel.log | while read -r line; do say "$line"; done
  status "after pass $PASS"
  if grep -q "^searching levels : " train-out/search-parallel.log; then say "every level has a winning run: nothing left to search"; break; fi
  SEED=$((SEED + 37))
done
status "finished"
say "finished after $PASS pass(es)"
