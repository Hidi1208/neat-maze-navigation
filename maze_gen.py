"""
maze_gen.py
Generates 101 procedural maze environments.
Mazes 1-100 are used for training.
Maze 101 is the unseen test environment.

Run this FIRST before train.py.
"""

import numpy as np
import os
import cv2

WIDTH  = 1000
HEIGHT = 600

START      = (50, 300)
GOAL_RECT  = (870, 220, 100, 160)   # x, y, w, h
GOAL_COLOR = (60, 30, 100)           # dark red in BGR

# ── Directories ───────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MAZE_DIR  = os.path.join(BASE_DIR, "mazes")
os.makedirs(MAZE_DIR, exist_ok=True)


def generate_maze(seed: int) -> np.ndarray:
    """
    Generate a solvable maze as a boolean wall mask (True = wall).
    Each maze has:
    - Border walls (10px thick)
    - 2-4 random vertical walls, each with 1-2 navigable gaps
    - Clear start area (left) and goal area (right)
    """
    rng  = np.random.RandomState(seed)
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)

    # Border walls
    mask[:10,  :]  = True
    mask[-10:, :]  = True
    mask[:, :10]   = True
    mask[:, -10:]  = True

    # Start notch — small opening on the left border
    mask[260:340, :10] = False

    # Clear goal area from right border
    gx, gy, gw, gh = GOAL_RECT
    mask[gy:gy+gh, gx:gx+gw+10] = False

    # Number of vertical walls
    n_walls = rng.randint(2, 5)

    # Spread walls across the width, avoiding start and goal regions
    usable_x     = np.linspace(150, 820, n_walls + 2, dtype=int)[1:-1]
    wall_thickness = rng.randint(20, 40)

    for wx in usable_x:
        wx = int(wx) + rng.randint(-30, 30)
        wx = max(150, min(wx, 800))

        # Full vertical wall
        mask[10:HEIGHT-10, wx:wx+wall_thickness] = True

        # Cut 1 or 2 navigable gaps
        n_gaps = rng.randint(1, 3)
        for _ in range(n_gaps):
            gap_h = rng.randint(90, 160)
            gap_y = rng.randint(15, HEIGHT - gap_h - 15)
            mask[gap_y:gap_y+gap_h, wx:wx+wall_thickness] = False

    return mask


def mask_to_image(mask: np.ndarray, maze_idx: int) -> np.ndarray:
    """Convert wall mask to a coloured BGR image (like the Canva mazes)."""
    img = np.full((HEIGHT, WIDTH, 3), fill_value=(20, 20, 20), dtype=np.uint8)

    # Navigable space — teal
    img[~mask] = (180, 210, 110)   # BGR teal-ish

    # Walls — dark
    img[mask]  = (20, 20, 20)

    # Goal — dark red
    gx, gy, gw, gh = GOAL_RECT
    img[gy:gy+gh, gx:gx+gw] = GOAL_COLOR

    # Start marker
    sx, sy = START
    cv2.circle(img, (sx, sy), 12, (60, 200, 60), -1)

    # Labels
    cv2.putText(img, f"MAZE {maze_idx:03d}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    cv2.putText(img, "START", (15, sy + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(img, "GOAL",  (gx + 15, gy + gh//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    return img


def generate_all(n_mazes: int = 101):
    print(f"Generating {n_mazes} mazes...")
    for i in range(1, n_mazes + 1):
        mask = generate_maze(seed=i)
        np.save(os.path.join(MAZE_DIR, f"maze_{i:03d}.npy"), mask)

        img = mask_to_image(mask, i)
        cv2.imwrite(os.path.join(MAZE_DIR, f"maze_{i:03d}.png"), img)

        if i % 10 == 0:
            print(f"  {i}/{n_mazes} mazes generated")

    print(f"✓ All mazes saved to: {MAZE_DIR}/")
    print(f"  maze_001.npy ... maze_100.npy  →  training")
    print(f"  maze_101.npy                   →  test (unseen)")


if __name__ == "__main__":
    generate_all()