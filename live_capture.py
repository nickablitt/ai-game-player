"""
Phase 1a: prove we can SEE the game.

This is the real-world replacement for render() from the grid world.
Instead of drawing the board ourselves, we photograph a region of your
screen where the Feed Zone game is running.

It shows two panels side by side:
  - LEFT : the raw pixels we captured
  - RIGHT: the 84x84 grayscale frame the AI would actually receive
           (small + gray on purpose -- less data = faster learning)

CALIBRATION (do this first):
  1. Open the game in your browser and start a race.
  2. Run this script:  python live_capture.py
  3. Use the keys below to drag/resize the capture box until the LEFT
     panel frames ONLY the game playfield (no browser chrome, no score
     bar yet -- we'll capture the score separately in Phase 2).
  4. Press P to print the final REGION values, then paste them back into
     the REGION dict below so they're saved.

KEYS (in this window):
  Arrow keys        move the capture box
  Shift + arrows    resize the capture box
  P                 print current REGION to the terminal
  ESC               quit
"""

import os
import time

# Where the PREVIEW WINDOW opens, as an absolute (x, y) in your combined
# multi-monitor desktop. Set this to a point on the screen you want it on --
# usually the monitor WITHOUT the game. Must be set before pygame loads.
os.environ["SDL_VIDEO_WINDOW_POS"] = "50,50"

import cv2
import mss
import numpy as np
import pygame

# The screen rectangle to capture, in pixels. EDIT after calibrating.
REGION = {"top": 210, "left": 3900, "width": 370, "height": 230}
OUT_SIZE = 84  # the AI sees an 84x84 grayscale image

pygame.init()
WIN_W, WIN_H = 820, 520
screen = pygame.display.set_mode((WIN_W, WIN_H))
pygame.display.set_caption("Phase 1: what the AI sees")
font = pygame.font.SysFont("consolas", 16)
small = pygame.font.SysFont("consolas", 13)
clock = pygame.time.Clock()


def grab(sct):
    raw = np.array(sct.grab(REGION))          # (H, W, 4), BGRA
    bgr = raw[:, :, :3]
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small_gray = cv2.resize(gray, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
    return rgb, small_gray


def to_surface(arr):
    """arr is (H, W, 3); pygame wants (W, H, 3)."""
    return pygame.surfarray.make_surface(np.ascontiguousarray(np.transpose(arr, (1, 0, 2))))


def main():
    move = 10
    with mss.MSS() as sct:
        running = True
        frames, t0, fps = 0, time.time(), 0.0
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    shift = pygame.key.get_mods() & pygame.KMOD_SHIFT
                    if e.key == pygame.K_ESCAPE:
                        running = False
                    elif e.key == pygame.K_p:
                        print("REGION =", REGION)
                    elif e.key == pygame.K_LEFT:
                        if shift: REGION["width"] = max(20, REGION["width"] - move)
                        else: REGION["left"] -= move
                    elif e.key == pygame.K_RIGHT:
                        if shift: REGION["width"] += move
                        else: REGION["left"] += move
                    elif e.key == pygame.K_UP:
                        if shift: REGION["height"] = max(20, REGION["height"] - move)
                        else: REGION["top"] -= move
                    elif e.key == pygame.K_DOWN:
                        if shift: REGION["height"] += move
                        else: REGION["top"] += move

            rgb, gray = grab(sct)

            screen.fill((20, 20, 24))
            screen.blit(font.render(f"FPS {fps:4.1f}   capturing {REGION['width']}x{REGION['height']} "
                                    f"at ({REGION['left']},{REGION['top']})", True, (235, 235, 235)), (10, 10))
            screen.blit(small.render("arrows move   shift+arrows resize   P print   ESC quit",
                                     True, (160, 160, 160)), (10, 34))

            # LEFT: raw capture, scaled to fit a 480x430 box, aspect preserved.
            h, w = rgb.shape[:2]
            scale = min(480 / w, 430 / h)
            raw_scaled = pygame.transform.smoothscale(to_surface(rgb), (int(w * scale), int(h * scale)))
            screen.blit(raw_scaled, (10, 70))
            screen.blit(small.render("RAW CAPTURE", True, (160, 160, 160)), (10, 54))

            # RIGHT: the 84x84 the AI sees, upscaled to 252x252.
            gray_rgb = np.stack([gray, gray, gray], axis=-1)
            gray_scaled = pygame.transform.scale(to_surface(gray_rgb), (252, 252))
            screen.blit(gray_scaled, (540, 70))
            pygame.draw.rect(screen, (120, 120, 130), (540, 70, 252, 252), 1)
            screen.blit(small.render("WHAT THE AI SEES (84x84)", True, (160, 160, 160)), (540, 54))

            pygame.display.flip()

            frames += 1
            if time.time() - t0 >= 0.5:
                fps = frames / (time.time() - t0)
                frames, t0 = 0, time.time()
            clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
