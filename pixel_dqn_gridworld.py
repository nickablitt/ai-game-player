"""
Pixel-input DQN on the same grid world -- the bridge to real games.

The big change from dqn_gridworld.py:
  - Before, the agent got a one-hot vector saying exactly which cell it
    was in. That's cheating: in a real game nobody hands you your (x, y).
  - Now the agent gets a small IMAGE of the whole board (a 24x24 grayscale
    picture) and NOTHING ELSE. It has to learn, from pixels alone, where
    it is, where the goal is, and which way to move.

To read an image you need a different kind of network: a CNN
(convolutional neural net). Conv layers slide small filters across the
image and learn to detect local patterns (edges, the bright agent blob,
the goal blob). That's the same architecture that played Atari from
pixels in 2015 -- and the same idea you'll point at a feed-zone screenshot.

Everything else (replay buffer, target net, Bellman target, training
loop) is IDENTICAL to dqn_gridworld.py. Only the input and the first few
network layers changed.

Heads-up: learning from pixels is MUCH slower than from a one-hot vector.
Expect a few hundred episodes before the value map looks sensible. Press
'+' to speed up the simulation. This sluggishness is the honest taste of
what real games demand.
"""

import random
from collections import deque

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

# --- Environment (identical to the other versions) -----------------------

GRID_W, GRID_H = 6, 6
START = (0, 0)
GOAL = (5, 5)
OBSTACLES = [(1, 2), (2, 2), (3, 2), (3, 3), (1, 4), (4, 0)]
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
N_ACTIONS = 4


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


# --- State -> image ------------------------------------------------------

CELL_PX = 4                       # each grid cell becomes a 4x4 block of pixels
IMG_H = GRID_H * CELL_PX          # 24
IMG_W = GRID_W * CELL_PX          # 24

# Brightness codes. The net must learn what each shade means.
SHADE_EMPTY = 0.15
SHADE_OBSTACLE = 0.40
SHADE_GOAL = 0.70
SHADE_AGENT = 1.00


def render(state):
    """Draw the board as a (1, IMG_H, IMG_W) grayscale image in [0, 1].
    The leading 1 is the 'channel' dimension a CNN expects."""
    img = np.full((IMG_H, IMG_W), SHADE_EMPTY, dtype=np.float32)
    for (ox, oy) in OBSTACLES:                       # numpy is [row=y, col=x]
        img[oy*CELL_PX:(oy+1)*CELL_PX, ox*CELL_PX:(ox+1)*CELL_PX] = SHADE_OBSTACLE
    gx, gy = GOAL
    img[gy*CELL_PX:(gy+1)*CELL_PX, gx*CELL_PX:(gx+1)*CELL_PX] = SHADE_GOAL
    ax, ay = state
    img[ay*CELL_PX:(ay+1)*CELL_PX, ax*CELL_PX:(ax+1)*CELL_PX] = SHADE_AGENT
    return img[np.newaxis, :, :]


# --- The convolutional network -------------------------------------------

class ConvQNet(nn.Module):
    """image (1x24x24) -> conv -> conv -> flatten -> dense -> 4 Q-values."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 24 -> 12
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 12 -> 6
            nn.ReLU(),
        )
        # Figure out the flattened size by running a dummy image through conv,
        # so we never have to compute it by hand.
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, 1, IMG_H, IMG_W)).flatten(1).shape[1]
        self.head = nn.Sequential(
            nn.Linear(n_flat, 128),
            nn.ReLU(),
            nn.Linear(128, N_ACTIONS),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)          # keep batch dim, flatten C,H,W into one vector
        return self.head(x)


device = torch.device("cpu")
qnet = ConvQNet().to(device)
target_net = ConvQNet().to(device)
target_net.load_state_dict(qnet.state_dict())
target_net.eval()
optimizer = optim.Adam(qnet.parameters(), lr=5e-4)
loss_fn = nn.MSELoss()

n_params = sum(p.numel() for p in qnet.parameters())
print(f"ConvQNet has {n_params:,} parameters. Input image is {IMG_H}x{IMG_W}.")


# --- Hyperparameters -----------------------------------------------------

GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.995      # decay slower than before: pixels need more exploration
EPSILON_MIN = 0.05
BATCH_SIZE = 64
BUFFER_SIZE = 10000
MIN_BUFFER = 500
TARGET_SYNC = 200

replay = deque(maxlen=BUFFER_SIZE)
train_step = 0


def choose_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, N_ACTIONS - 1)
    with torch.no_grad():
        x = torch.from_numpy(render(state)).unsqueeze(0).to(device)  # (1,1,H,W)
        return int(qnet(x).argmax(dim=1).item())


def train_one_batch():
    global train_step
    if len(replay) < MIN_BUFFER:
        return None

    batch = random.sample(replay, BATCH_SIZE)
    s, a, r, s2, d = zip(*batch)
    s = torch.from_numpy(np.stack(s)).to(device)     # (B,1,H,W)
    s2 = torch.from_numpy(np.stack(s2)).to(device)
    a = torch.tensor(a, dtype=torch.long).to(device)
    r = torch.tensor(r, dtype=torch.float32).to(device)
    d = torch.tensor(d, dtype=torch.float32).to(device)

    q_pred = qnet(s).gather(1, a.unsqueeze(1)).squeeze(1)
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
pygame.display.set_caption("Pixel DQN: the CNN learns from images")
font = pygame.font.SysFont("consolas", 14)
small_font = pygame.font.SysFont("consolas", 11)
big_font = pygame.font.SysFont("consolas", 18, bold=True)
clock = pygame.time.Clock()


def all_q_values():
    """Counterfactual value map: for each cell, render an image with the
    agent placed there and ask the net what it would do. Lets us paint the
    same triangle view as before, now driven entirely by the CNN's reading
    of pictures."""
    imgs = np.stack([render((x, y)) for x in range(GRID_W) for y in range(GRID_H)])
    with torch.no_grad():
        q = qnet(torch.from_numpy(imgs).to(device)).cpu().numpy()
    return q.reshape(GRID_W, GRID_H, N_ACTIONS)


def value_color(v, vmin, vmax):
    t = 0.5 if vmax - vmin < 1e-9 else (v - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    return (int(255 * (1 - t)), int(255 * t), 60)


def draw_input_preview(state):
    """Show the actual 24x24 image the CNN receives, scaled up, top-right."""
    img = render(state)[0]                                  # (H, W) in [0,1]
    rgb = (np.stack([img]*3, axis=-1) * 255).astype(np.uint8)
    surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    scale = 2
    surf = pygame.transform.scale(surf, (IMG_W*scale, IMG_H*scale))
    x0 = W - IMG_W*scale - 8
    screen.blit(surf, (x0, 6))
    pygame.draw.rect(screen, (120, 120, 130), (x0, 6, IMG_W*scale, IMG_H*scale), 1)
    screen.blit(small_font.render("CNN input", True, (160, 160, 160)), (x0, 6 + IMG_H*scale + 1))


def draw(state, episode, total_reward, epsilon, last_loss):
    screen.fill((25, 25, 28))

    loss_str = f"loss {last_loss:.3f}" if last_loss is not None else f"buf {len(replay)}/{MIN_BUFFER}"
    txt = f"Ep {episode}  rew {total_reward:+6.1f}  eps {epsilon:.2f}  {loss_str}"
    screen.blit(big_font.render(txt, True, (255, 255, 255)), (10, 8))
    screen.blit(font.render("SPACE pause   +/- speed   ESC quit", True, (170, 170, 170)), (10, 34))

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
                screen.blit(font.render(f"{best:+.1f}", True, (0, 0, 0)), (cx + 4, cy + 4))
            pygame.draw.rect(screen, (95, 95, 100), (cx, cy, CELL, CELL), 1)

    ax = state[0] * CELL + CELL // 2
    ay = state[1] * CELL + HEADER + CELL // 2
    pygame.draw.circle(screen, (80, 180, 255), (ax, ay), CELL // 4)
    pygame.draw.circle(screen, (255, 255, 255), (ax, ay), CELL // 4, 2)

    draw_input_preview(state)
    pygame.display.flip()


# --- Main loop -----------------------------------------------------------

def main():
    global EPSILON
    paused = False
    fps = 30
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
                        fps = min(240, fps + 20)
                    if event.key == pygame.K_MINUS:
                        fps = max(2, fps - 10)

            if paused:
                draw(state, episode, total_reward, EPSILON, last_loss)
                clock.tick(30)
                continue

            action = choose_action(state, EPSILON)
            new_state, reward, done = step(state, action)
            replay.append((render(state), action, reward, render(new_state), float(done)))
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
