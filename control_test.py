"""
Phase 1b: prove we can CONTROL the game.

This is the real-world replacement for step(action) from the grid world.
Instead of updating a Python variable, we send real arrow-key presses that
the browser game receives -- exactly how a human plays.

HOW TO USE:
  1. Open the game and start a race.
  2. Run this script:  python control_test.py
  3. You get a 5-second countdown. During it, CLICK THE GAME so it has
     keyboard focus (keys go to whatever window is focused).
  4. Watch your soigneur move in a square pattern.

If nothing moves: the game window probably wasn't focused, OR pydirectinput's
keystrokes aren't reaching this browser. In that case try the pynput fallback
noted at the bottom of this file.
"""

import time

import pydirectinput

# pydirectinput adds a 0.1s pause after each call by default; make it snappy.
pydirectinput.PAUSE = 0.0

MOVES = ["up", "right", "down", "left"]  # one lap around a square


def main():
    print("Click the GAME window now. Sending keys in:")
    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    print("Go! Watch the game.")

    for lap in range(3):
        for key in MOVES:
            print(f"  press {key}")
            pydirectinput.press(key)   # one tap = keydown + keyup
            time.sleep(0.5)

    print("Done. Did the soigneur trace a square?")


if __name__ == "__main__":
    main()


# --- pynput fallback ------------------------------------------------------
# If pydirectinput doesn't move the game, install pynput (pip install pynput)
# and replace the press() calls above with:
#
#   from pynput.keyboard import Key, Controller
#   kb = Controller()
#   kb.press(Key.up); time.sleep(0.1); kb.release(Key.up)
#
# For continuous movement (holding a direction) you'll want press/release with
# a gap rather than a single tap -- we'll switch to that once we know which
# library your browser responds to.
