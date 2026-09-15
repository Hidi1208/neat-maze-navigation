"""
test.py
Tests the aggregated genome on maze 101 — never seen during training.
Runs a full pygame visualisation and saves all metrics for the paper.

Run order:
    1. maze_gen.py
    2. train.py
    3. aggregate.py
    4. test.py    ← you are here
    5. visualise.py
"""

import pygame
import neat
import math
import os
import pickle
import json
import csv
import time
import numpy as np
import cv2

# ── Constants ─────────────────────────────────
WIDTH  = 1000
HEIGHT = 600

START       = (50, 300)
GOAL_CENTER = (920, 300)
GOAL_RECT   = (870, 220, 100, 160)   # x, y, w, h

RAY_ANGLES = [-70, -40, -15, 0, 15, 40, 70]
RAY_LEN    = 150
RAY_STEP   = 4

MAX_FRAMES  = 600   # longer for test — give it full time

# ── Colours ───────────────────────────────────
DRONE_COL = (60,  140, 255)
LIDAR_COL = (255, 120, 120)
TRAIL_COL = (180, 180, 180)
GOAL_COL  = (60,  220,  80)
WALL_COL  = (20,   20,  20)
NAV_COL   = (110, 210, 180)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MAZE_DIR    = os.path.join(BASE_DIR, "mazes")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

TEST_MAZE_IDX = 101


# ── Pygame init ───────────────────────────────
pygame.init()
WIN   = pygame.display.set_mode((WIDTH, HEIGHT))
CLOCK = pygame.time.Clock()
pygame.display.set_caption("NEAT Drone — Generalisation Test (Maze 101)")


# ── Load maze ─────────────────────────────────
def load_maze(idx: int):
    mask_path = os.path.join(MAZE_DIR, f"maze_{idx:03d}.npy")
    img_path  = os.path.join(MAZE_DIR, f"maze_{idx:03d}.png")

    wall_mask = np.load(mask_path)
    img_cv    = cv2.imread(img_path)
    img_rgb   = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    surface   = pygame.surfarray.make_surface(np.transpose(img_rgb, (1, 0, 2)))

    return wall_mask, surface


# ── LIDAR ─────────────────────────────────────
def lidar_scan(x, y, angle, wall_mask):
    readings = []
    for offset in RAY_ANGLES:
        ray_angle = angle + offset
        cos_a = math.cos(math.radians(ray_angle))
        sin_a = math.sin(math.radians(ray_angle))
        dist  = RAY_LEN
        for d in range(0, RAY_LEN, RAY_STEP):
            px = int(x + cos_a * d)
            py = int(y + sin_a * d)
            if not (0 <= px < WIDTH and 0 <= py < HEIGHT):
                dist = d; break
            if wall_mask[py, px]:
                dist = d; break
        readings.append((ray_angle, dist / RAY_LEN))
    return readings


# ── Drone ─────────────────────────────────────
class Drone:
    def __init__(self):
        self.x        = float(START[0])
        self.y        = float(START[1])
        self.angle    = 0.0
        self.vel      = 0.0
        self.turn_vel = 0.0
        self.alive    = True
        self.trail    = []

    def step(self, steer, wall_mask):
        self.turn_vel += steer * 0.4
        self.turn_vel *= 0.85
        self.angle    += self.turn_vel
        self.vel      += 0.08
        self.vel      *= 0.97

        nx = self.x + math.cos(math.radians(self.angle)) * self.vel
        ny = self.y + math.sin(math.radians(self.angle)) * self.vel

        hit = False
        for dx in (-6, 0, 6):
            for dy in (-6, 0, 6):
                px = int(nx + dx)
                py = int(ny + dy)
                if not (0 <= px < WIDTH and 0 <= py < HEIGHT):
                    hit = True; break
                if wall_mask[py, px]:
                    hit = True; break
            if hit:
                break

        if hit:
            self.alive = False
            return

        self.x, self.y = nx, ny
        self.trail.append((self.x, self.y))
        if len(self.trail) > 100:
            self.trail.pop(0)

    def draw(self, scan):
        # LIDAR rays
        for ang, dist in scan:
            px = int(self.x + math.cos(math.radians(ang)) * dist * RAY_LEN)
            py = int(self.y + math.sin(math.radians(ang)) * dist * RAY_LEN)
            pygame.draw.line(WIN, LIDAR_COL, (int(self.x), int(self.y)), (px, py), 1)

        # Trail
        for i, t in enumerate(self.trail):
            alpha = int(80 * i / len(self.trail))
            pygame.draw.circle(WIN, (alpha, alpha+80, alpha+160),
                               (int(t[0]), int(t[1])), 2)

        # Drone body
        surf = pygame.Surface((24, 24), pygame.SRCALPHA)
        pygame.draw.circle(surf, DRONE_COL,  (12, 12), 7)
        pygame.draw.circle(surf, (30,30,30), (4,  4),  3)
        pygame.draw.circle(surf, (30,30,30), (20, 4),  3)
        pygame.draw.circle(surf, (30,30,30), (4,  20), 3)
        pygame.draw.circle(surf, (30,30,30), (20, 20), 3)
        rot  = pygame.transform.rotate(surf, -self.angle)
        rect = rot.get_rect(center=(int(self.x), int(self.y)))
        WIN.blit(rot, rect)


# ── HUD ───────────────────────────────────────
def draw_hud(font, frame, min_dist, reached, cause):
    hud = pygame.Surface((320, 180), pygame.SRCALPHA)
    hud.fill((0, 0, 0, 170))
    WIN.blit(hud, (10, 10))

    goal_col = (60, 220, 80) if reached else (255, 80, 80)

    lines = [
        (f"NEAT Drone — Maze {TEST_MAZE_IDX} (UNSEEN)", (220, 220, 220)),
        (f"Method : Genome Aggregation (100 mazes)",    (180, 180, 180)),
        (f"Frame  : {frame} / {MAX_FRAMES}",            (255, 255, 255)),
        (f"Min dist to goal : {min_dist:.0f} px",       (255, 255, 255)),
        (f"Goal   : {'REACHED ✓' if reached else 'NOT YET'}",  goal_col),
        (f"Status : {cause}",                           (200, 200, 200)),
    ]
    for i, (text, col) in enumerate(lines):
        WIN.blit(font.render(text, True, col), (20, 18 + i * 26))


# ── Main test ─────────────────────────────────
def run_test(genome, config):
    wall_mask, maze_surf = load_maze(TEST_MAZE_IDX)
    net   = neat.nn.FeedForwardNetwork.create(genome, config)
    drone = Drone()
    font  = pygame.font.SysFont("arial", 20)

    gx, gy, gw, gh = GOAL_RECT
    min_dist   = float("inf")
    reached    = False
    cause      = "timeout"
    frame      = 0
    t_start    = time.time()

    while drone.alive and frame < MAX_FRAMES:
        frame += 1
        CLOCK.tick(60)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); return None

        # Draw maze
        WIN.blit(maze_surf, (0, 0))

        # Goal highlight
        pygame.draw.rect(WIN, (60, 220, 80),
                         pygame.Rect(gx, gy, gw, gh), 3, border_radius=6)

        # Scan + decide
        scan       = lidar_scan(drone.x, drone.y, drone.angle, wall_mask)
        lidar_vals = [d for _, d in scan]

        dx_n = (GOAL_CENTER[0] - drone.x) / WIDTH
        dy_n = (GOAL_CENTER[1] - drone.y) / HEIGHT
        inputs = lidar_vals + [dx_n, dy_n, drone.angle/180.0, drone.vel/5.0]

        desired    = math.degrees(math.atan2(
            GOAL_CENTER[1] - drone.y, GOAL_CENTER[0] - drone.x))
        angle_diff = (desired - drone.angle + 180) % 360 - 180
        base_steer = angle_diff * 0.025
        neat_out   = net.activate(inputs)[0]
        steer      = max(-1.0, min(1.0, base_steer + neat_out * 2.0))

        drone.step(steer, wall_mask)
        drone.draw(scan)

        # Metrics
        dist     = math.hypot(GOAL_CENTER[0] - drone.x,
                              GOAL_CENTER[1] - drone.y)
        min_dist = min(min_dist, dist)

        if gx <= drone.x <= gx + gw and gy <= drone.y <= gy + gh:
            reached = True
            cause   = "goal_reached"
            drone.alive = False

        if not drone.alive and not reached:
            cause = "wall_collision"

        draw_hud(font, frame, min_dist, reached, cause)
        pygame.display.update()

    elapsed = time.time() - t_start

    return {
        "maze_idx"          : TEST_MAZE_IDX,
        "method"            : "genome_aggregation",
        "n_training_mazes"  : 100,
        "goal_reached"      : reached,
        "frames_survived"   : frame,
        "max_frames"        : MAX_FRAMES,
        "min_dist_to_goal_px": round(min_dist, 2),
        "cause_of_end"      : cause,
        "test_duration_s"   : round(elapsed, 2),
        "genome_n_nodes"    : len(genome.nodes),
        "genome_n_connections": len(genome.connections),
    }


# ── Results screen ────────────────────────────
def show_results_screen(result):
    font_big   = pygame.font.SysFont("arial", 36)
    font_med   = pygame.font.SysFont("arial", 22)
    font_small = pygame.font.SysFont("arial", 18)
    waiting    = True

    while waiting:
        WIN.fill((12, 12, 20))
        WIN.blit(font_big.render("Generalisation Test — Results", True, (255,255,255)),
                 (WIDTH//2 - 240, 28))

        y = 100
        WIN.blit(font_med.render("── Maze 101 (Unseen Test Environment) ──",
                                 True, (255, 200, 60)), (60, y)); y += 45

        reached_col = (60, 220, 80) if result["goal_reached"] else (255, 80, 80)
        rows = [
            ("Method",              "Genome Aggregation (100 winners averaged)",   (180,180,180)),
            ("Goal Reached",        "YES ✓" if result["goal_reached"] else "NO ✗", reached_col),
            ("Frames Survived",     f"{result['frames_survived']} / {result['max_frames']}",  (200,200,200)),
            ("Closest to Goal",     f"{result['min_dist_to_goal_px']} px",          (200,200,200)),
            ("Ended By",            result["cause_of_end"],                         (200,200,200)),
            ("Aggregated Nodes",    str(result["genome_n_nodes"]),                  (200,200,200)),
            ("Aggregated Conns",    str(result["genome_n_connections"]),             (200,200,200)),
        ]
        for label, val, col in rows:
            WIN.blit(font_small.render(f"{label:<28}: {val}", True, col), (80, y))
            y += 30

        WIN.blit(font_small.render("Results saved to results/ folder",
                                   True, (100,100,100)), (60, y + 20))
        WIN.blit(font_small.render("Press ESC to exit", True, (70,70,70)),
                 (WIDTH//2 - 80, HEIGHT - 36))

        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                waiting = False


# ── Save results ──────────────────────────────
def save_results(result):
    # JSON
    json_path = os.path.join(RESULTS_DIR, "test_result.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"✓ JSON result : {json_path}")

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "test_result.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)
    print(f"✓ CSV result  : {csv_path}")

    # Plain text for paper
    txt_path = os.path.join(RESULTS_DIR, "test_result.txt")
    with open(txt_path, "w") as f:
        f.write("NEAT DRONE — GENERALISATION TEST RESULT\n")
        f.write("=" * 50 + "\n\n")
        f.write("Research Question:\n")
        f.write("  Can a genome aggregated from 100 independent NEAT\n")
        f.write("  training runs navigate an unseen maze environment?\n\n")
        f.write("Method: Genome Aggregation\n")
        f.write(f"  Training mazes   : 100 (mazes 1-100)\n")
        f.write(f"  Test maze        : maze 101 (never seen during training)\n")
        f.write(f"  Aggregation      : Weight averaging (majority threshold 30%)\n\n")
        f.write("Result:\n")
        f.write(f"  Goal reached     : {'YES' if result['goal_reached'] else 'NO'}\n")
        f.write(f"  Frames survived  : {result['frames_survived']} / {result['max_frames']}\n")
        f.write(f"  Closest to goal  : {result['min_dist_to_goal_px']} px\n")
        f.write(f"  Ended by         : {result['cause_of_end']}\n\n")
        f.write("Aggregated Genome Topology:\n")
        f.write(f"  Nodes       : {result['genome_n_nodes']}\n")
        f.write(f"  Connections : {result['genome_n_connections']}\n")
    print(f"✓ TXT summary : {txt_path}")


# ── Entry point ───────────────────────────────
def run():
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config-feedforward.txt")
    genome_path = os.path.join(base_dir, "aggregated_genome.pkl")

    if not os.path.exists(genome_path):
        print("ERROR: aggregated_genome.pkl not found.")
        print("Run aggregate.py first.")
        pygame.quit(); return

    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    print(f"✓ Aggregated genome loaded")
    print(f"  Nodes: {len(genome.nodes)}  |  Connections: {len(genome.connections)}")
    print(f"\nRunning test on maze {TEST_MAZE_IDX} (unseen)...")

    result = run_test(genome, config)

    if result:
        print("\n── Test Results ─────────────────────────────")
        for k, v in result.items():
            print(f"  {k:<28}: {v}")

        save_results(result)
        show_results_screen(result)

    pygame.quit()
    print("\nNext step: python visualise.py")


if __name__ == "__main__":
    run()