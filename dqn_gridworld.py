"""
DQN (Deep Q-Network) on the same grid world.

What's different from gridworld_qlearning.py:
  - The "brain" is a tiny neural net instead of a 6x6x4 table.
  - Given a one-hot-encoded position, the net outputs 4 Q-values.
  - Two standard DQN tricks are added so the net actually learns:
      1. Replay buffer  -- train on random past experiences, not just the
         latest one. Breaks correlation between consecutive samples.
      2. Target network -- a frozen copy of the net used to compute
         training targets. Updated every TARGET_SYNC steps. Without this,
         the target moves with the prediction and training oscillates.

Why bother for a 6x6 grid? You don't need a NN here -- a Q-table is
strictly better for this size. The point is the *machinery*: this same
file will scale to states the size of a screenshot. A Q-table can't.
"""

import random
from collections import deque

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

# --- Environment (identical to the tabular version) ----------------------

GRID_W, GRID_H = 6, 6
START = (0, 0)
GOAL = (5, 5)
OBSTACLES = [(1, 2), (2, 2), (3, 2), (3, 3), (1, 4)]
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
N_ACTIONS = 4
N_STATES = GRID_W * GRID_H


def step(state, action_idx):
    dx, dy = ACTIONS[action_idx]
    nx, ny = state[0] + dx, state[1] + dy
    if not (0 <= nx < GRID_W and 0 <= ny < GRID_H):
        return state, -1.0, False
    new_state = (nx, ny)
    if new_state in OBSTACLES:
        return new_state, -5.0, True
    if new_state == GOAL:
        return new_state, 10.0, True
    return new_state, -0.1, False


def encode(state):
    """Turn (x, y) into a 36-dim one-hot vector the network can read."""
    v = np.zeros(N_STATES, dtype=np.float32)
    v[state[0] * GRID_H + state[1]] = 1.0
    return v


# --- The neural net ------------------------------------------------------

class QNet(nn.Module):
    """36-dim input -> 64 -> 64 -> 4 Q-values out. About 7000 parameters."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_STATES, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, N_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cpu")
qnet = QNet().to(device)
target_net = QNet().to(device)
target_net.load_state_dict(qnet.state_dict())
target_net.eval()
optimizer = optim.Adam(qnet.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()


# --- Hyperparameters -----------------------------------------------------

GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.05
BATCH_SIZE = 64        # how many past samples to train on per step
BUFFER_SIZE = 5000     # max stored experiences
MIN_BUFFER = 200       # don't start training until buffer has this many
TARGET_SYNC = 100      # copy weights qnet -> target_net every N train steps

replay = deque(maxlen=BUFFER_SIZE)
train_step = 0


def choose_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, N_ACTIONS - 1)
    with torch.no_grad():
        x = torch.from_numpy(encode(state)).unsqueeze(0).to(device)
        return int(qnet(x).argmax(dim=1).item())


def train_one_batch():
    """One gradient step on a random batch from replay. Returns loss or None."""
    global train_step
    if len(replay) < MIN_BUFFER:
        return None

    batch = random.sample(replay, BATCH_SIZE)
    s, a, r, s2, d = zip(*batch)
    s = torch.from_numpy(np.stack(s)).to(device)
    s2 = torch.from_numpy(np.stack(s2)).to(device)
    a = torch.tensor(a, dtype=torch.long).to(device)
    r = torch.tensor(r, dtype=torch.float32).to(device)
    d = torch.tensor(d, dtype=torch.float32).to(device)

    # Q-value the net currently predicts for the action we actually took.
    q_pred = qnet(s).gather(1, a.unsqueeze(1)).squeeze(1)

    # Target = immediate reward + discounted best future value (from the
    # *target* net, not the live one). Zero out future term if episode ended.
    with torch.no_grad():
        q_next = target_net(s2).max(dim=1).values
        q_target = r + GAMMA * q_next * (1.0 - d)

    loss = loss_fn(q_pred, q_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_step += 1
    if train_step % TARGET_SYNC == 0:
        target_net.load_state_dict(qnet.state_dict())
    return float(loss.item())


# --- Visualization -------------------------------------------------------

CELL = 80
HEADER = 60
W = GRID_W * CELL
H = GRID_H * CELL + HEADER

pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("DQN: watch the neural net learn")
font = pygame.font.SysFont("consolas", 14)
big_font = pygame.font.SysFont("consolas", 18, bold=True)
clock = pygame.time.Clock()


def all_q_values():
    """Run the net once for every cell so we can paint the grid."""
    states = np.stack([encode((x, y)) for x in range(GRID_W) for y in range(GRID_H)])
    with torch.no_grad():
        q = qnet(torch.from_numpy(states).to(device)).cpu().numpy()
    return q.reshape(GRID_W, GRID_H, N_ACTIONS)


def value_color(v, vmin, vmax):
    if vmax - vmin < 1e-9:
        t = 0.5
    else:
        t = (v - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    return (int(255 * (1 - t)), int(255 * t), 60)


def draw(state, episode, total_reward, epsilon, last_loss):
    screen.fill((25, 25, 28))

    loss_str = f"loss {last_loss:.3f}" if last_loss is not None else f"buf {len(replay)}/{MIN_BUFFER}"
    txt = f"Ep {episode}  rew {total_reward:+6.1f}  eps {epsilon:.2f}  {loss_str}"
    screen.blit(big_font.render(txt, True, (255, 255, 255)), (10, 8))
    screen.blit(font.render("SPACE pause   +/- speed   ESC quit",
                            True, (170, 170, 170)), (10, 34))

    Q = all_q_values()
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
                    [(cx, cy), (cx + CELL, cy), (mx, my)],
                    [(cx, cy + CELL), (cx + CELL, cy + CELL), (mx, my)],
                    [(cx, cy), (cx, cy + CELL), (mx, my)],
                    [(cx + CELL, cy), (cx + CELL, cy + CELL), (mx, my)],
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


# --- Main loop -----------------------------------------------------------

def main():
    global EPSILON
    paused = False
    fps = 20
    episode = 0
    last_loss = None

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
                draw(state, episode, total_reward, EPSILON, last_loss)
                clock.tick(30)
                continue

            action = choose_action(state, EPSILON)
            new_state, reward, done = step(state, action)
            replay.append((encode(state), action, reward, encode(new_state), float(done)))
            loss = train_one_batch()
            if loss is not None:
                last_loss = loss
            state = new_state
            total_reward += reward

            draw(state, episode, total_reward, EPSILON, last_loss)
            clock.tick(fps)

        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)


if __name__ == "__main__":
    main()
