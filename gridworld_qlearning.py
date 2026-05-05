"""
Watch a Q-learning agent learn to navigate a grid world.

The "AI" here is a Q-table: for every (x, y) cell on the grid and every
action (up/down/left/right), it stores a number — its current best guess
of the long-term reward for taking that action from that cell.

What you'll see on screen:
  - 4 colored triangles per cell = Q-values for the 4 actions
    green = "this action looks good", red = "this action looks bad"
  - The number in the corner = best Q-value for that cell
  - Blue circle = the agent
  - Green cell = goal (+10 reward)
  - Red cells = obstacles (-5 reward, ends episode)

Controls:
  SPACE      pause/resume
  +  /  -    faster / slower
  ESC        quit
"""

import random
import numpy as np
import pygame

# --- Environment ----------------------------------------------------------

GRID_W, GRID_H = 6, 6
START = (0, 0)
GOAL = (5, 5)
OBSTACLES = [(1, 2), (2, 2), (3, 2), (3, 3), (1, 4), (4,0)]

# (dx, dy) for up, down, left, right. y grows downward (screen coords).
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]


def step(state, action_idx):
    """Apply an action. Returns (next_state, reward, done)."""
    dx, dy = ACTIONS[action_idx]
    nx, ny = state[0] + dx, state[1] + dy
    if not (0 <= nx < GRID_W and 0 <= ny < GRID_H):
        return state, -1.0, False  # bumped a wall, stay put
    new_state = (nx, ny)
    if new_state in OBSTACLES:
        return new_state, -5.0, True
    if new_state == GOAL:
        return new_state, 10.0, True
    return new_state, -0.1, False  # small step penalty -> prefers shorter paths


# --- Q-learning -----------------------------------------------------------

Q = np.zeros((GRID_W, GRID_H, 4))  # the "brain"

ALPHA = 0.1          # learning rate: how much new info overwrites old
GAMMA = 0.95         # discount: how much we care about future reward
EPSILON = 1.0        # exploration rate: chance of trying a random action
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.05


def choose_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 3)        # explore
    return int(np.argmax(Q[state[0], state[1]]))  # exploit best known


def update_q(state, action, reward, new_state, done):
    best_next = 0.0 if done else float(np.max(Q[new_state[0], new_state[1]]))
    target = reward + GAMMA * best_next
    Q[state[0], state[1], action] += ALPHA * (target - Q[state[0], state[1], action])


# --- Visualization --------------------------------------------------------

CELL = 80
HEADER = 60
W = GRID_W * CELL
H = GRID_H * CELL + HEADER

pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Q-learning: watch it learn")
font = pygame.font.SysFont("consolas", 14)
big_font = pygame.font.SysFont("consolas", 18, bold=True)
clock = pygame.time.Clock()


def value_color(v, vmin, vmax):
    if vmax - vmin < 1e-9:
        t = 0.5
    else:
        t = (v - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    r = int(255 * (1 - t))
    g = int(255 * t)
    return (r, g, 60)


def draw(state, episode, total_reward, epsilon):
    screen.fill((25, 25, 28))

    txt = f"Episode {episode}   reward {total_reward:+6.1f}   epsilon {epsilon:.2f}"
    screen.blit(big_font.render(txt, True, (255, 255, 255)), (10, 8))
    screen.blit(font.render("SPACE pause   +/- speed   ESC quit",
                            True, (170, 170, 170)), (10, 34))

    vmin, vmax = float(Q.min()), float(Q.max())

    for x in range(GRID_W):
        for y in range(GRID_H):
            cx = x * CELL
            cy = y * CELL + HEADER

            if (x, y) in OBSTACLES:
                pygame.draw.rect(screen, (130, 30, 30), (cx, cy, CELL, CELL))
            elif (x, y) == GOAL:
                pygame.draw.rect(screen, (30, 130, 50), (cx, cy, CELL, CELL))
            else:
                pygame.draw.rect(screen, (55, 55, 60), (cx, cy, CELL, CELL))
                mx, my = cx + CELL // 2, cy + CELL // 2
                triangles = [
                    [(cx, cy), (cx + CELL, cy), (mx, my)],                      # up
                    [(cx, cy + CELL), (cx + CELL, cy + CELL), (mx, my)],        # down
                    [(cx, cy), (cx, cy + CELL), (mx, my)],                      # left
                    [(cx + CELL, cy), (cx + CELL, cy + CELL), (mx, my)],        # right
                ]
                for i, pts in enumerate(triangles):
                    pygame.draw.polygon(screen, value_color(Q[x, y, i], vmin, vmax), pts)
                    pygame.draw.polygon(screen, (35, 35, 38), pts, 1)
                best = float(Q[x, y].max())
                lbl = font.render(f"{best:+.1f}", True, (0, 0, 0))
                screen.blit(lbl, (cx + 4, cy + 4))

            pygame.draw.rect(screen, (95, 95, 100), (cx, cy, CELL, CELL), 1)

    ax = state[0] * CELL + CELL // 2
    ay = state[1] * CELL + HEADER + CELL // 2
    pygame.draw.circle(screen, (80, 180, 255), (ax, ay), CELL // 4)
    pygame.draw.circle(screen, (255, 255, 255), (ax, ay), CELL // 4, 2)

    pygame.display.flip()


# --- Main loop ------------------------------------------------------------

def main():
    global EPSILON
    paused = False
    fps = 20
    episode = 0

    while True:
        episode += 1
        state = START
        total_reward = 0.0
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit(); return
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    if event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        fps = min(120, fps + 10)
                    if event.key == pygame.K_MINUS:
                        fps = max(2, fps - 5)

            if paused:
                draw(state, episode, total_reward, EPSILON)
                clock.tick(30)
                continue

            action = choose_action(state, EPSILON)
            new_state, reward, done = step(state, action)
            update_q(state, action, reward, new_state, done)
            state = new_state
            total_reward += reward

            draw(state, episode, total_reward, EPSILON)
            clock.tick(fps)

        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)


if __name__ == "__main__":
    main()
