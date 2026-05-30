"""
Measure the TRUE screen-capture speed, with zero display overhead.

live_capture.py reports a low FPS mostly because it draws two scaled image
panels every frame. The training pipeline does none of that -- it just grabs
a frame and shrinks it to 84x84. This script times exactly that bare path so
we know our real frame-rate ceiling.

Run it WITH the game running, so the measurement reflects real conditions.
Edit REGION to match your calibrated values from live_capture.py.
"""

import time

import cv2
import mss
import numpy as np

REGION = {"top": 210, "left": 3900, "width": 370, "height": 230}
OUT_SIZE = 84
SECONDS = 5


def main():
    with mss.MSS() as sct:
        for _ in range(10):                 # warm up (first grabs are slow)
            np.array(sct.grab(REGION))

        n = 0
        t0 = time.time()
        while time.time() - t0 < SECONDS:
            raw = np.array(sct.grab(REGION))                 # capture
            gray = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2GRAY)
            cv2.resize(gray, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
            n += 1
        dt = time.time() - t0

    print(f"Captured {n} frames in {dt:.1f}s = {n / dt:.1f} FPS")
    print(f"(region {REGION['width']}x{REGION['height']}, output {OUT_SIZE}x{OUT_SIZE})")


if __name__ == "__main__":
    main()
