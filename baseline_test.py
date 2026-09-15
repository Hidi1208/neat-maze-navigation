"""
baseline_test.py  —  Method B: Test single population genome on maze 101 (unseen)
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

WIDTH, HEIGHT = 1000, 600
START         = (50, 300)
GOAL_CENTER   = (920, 300)
GOAL_RECT     = (870, 220, 100, 160)
MAX_FRAMES    = 600
TEST_MAZE_IDX = 101

RAY_ANGLES = [-70, -40, -15, 0, 15, 40, 70]
RAY_LEN    = 150
RAY_STEP   = 4

DRONE_COL = (255, 140, 60)
LIDAR_COL = (120, 255, 120)
TRAIL_COL = (180, 180, 180)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MAZE_DIR    = os.path.join(BASE_DIR, "mazes")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

pygame.init()
WIN   = pygame.display.set_mode((WIDTH, HEIGHT))
CLOCK = pygame.time.Clock()
pygame.display.set_caption(f"Method B — Baseline Single Population — Maze {TEST_MAZE_IDX}")


def load_maze(idx):
    mask    = np.load(os.path.join(MAZE_DIR, f"maze_{idx:03d}.npy"))
    img_cv  = cv2.imread(os.path.join(MAZE_DIR, f"maze_{idx:03d}.png"))
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    surf    = pygame.surfarray.make_surface(np.transpose(img_rgb, (1,0,2)))
    return mask, surf


def lidar_scan(x, y, angle, wall_mask):
    readings = []
    for offset in RAY_ANGLES:
        ra    = angle + offset
        cos_a = math.cos(math.radians(ra))
        sin_a = math.sin(math.radians(ra))
        dist  = RAY_LEN
        for d in range(0, RAY_LEN, RAY_STEP):
            px = int(x + cos_a * d)
            py = int(y + sin_a * d)
            if not (0 <= px < WIDTH and 0 <= py < HEIGHT):
                dist = d; break
            if wall_mask[py, px]:
                dist = d; break
        readings.append((ra, dist / RAY_LEN))
    return readings


class Drone:
    def __init__(self):
        self.x, self.y = float(START[0]), float(START[1])
        self.angle = self.vel = self.turn_vel = 0.0
        self.alive = True
        self.trail = []

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
                px, py = int(nx+dx), int(ny+dy)
                if not (0 <= px < WIDTH and 0 <= py < HEIGHT):
                    hit = True; break
                if wall_mask[py, px]:
                    hit = True; break
            if hit:
                break
        if hit:
            self.alive = False; return
        self.x, self.y = nx, ny
        self.trail.append((self.x, self.y))
        if len(self.trail) > 120:
            self.trail.pop(0)

    def draw(self, scan):
        for ang, dist in scan:
            ex = int(self.x + math.cos(math.radians(ang)) * dist * RAY_LEN)
            ey = int(self.y + math.sin(math.radians(ang)) * dist * RAY_LEN)
            pygame.draw.line(WIN, LIDAR_COL, (int(self.x), int(self.y)), (ex, ey), 1)
        for t in self.trail:
            pygame.draw.circle(WIN, TRAIL_COL, (int(t[0]), int(t[1])), 2)
        surf = pygame.Surface((24, 24), pygame.SRCALPHA)
        pygame.draw.circle(surf, DRONE_COL,  (12,12), 7)
        pygame.draw.circle(surf, (30,30,30), (4, 4),  3)
        pygame.draw.circle(surf, (30,30,30), (20,4),  3)
        pygame.draw.circle(surf, (30,30,30), (4, 20), 3)
        pygame.draw.circle(surf, (30,30,30), (20,20), 3)
        rot = pygame.transform.rotate(surf, -self.angle)
        WIN.blit(rot, rot.get_rect(center=(int(self.x), int(self.y))))


def run_test(genome, config):
    wall_mask, maze_surf = load_maze(TEST_MAZE_IDX)
    net   = neat.nn.FeedForwardNetwork.create(genome, config)
    drone = Drone()
    font  = pygame.font.SysFont("arial", 20)

    gx, gy, gw, gh = GOAL_RECT
    min_dist = float("inf")
    reached  = False
    cause    = "timeout"
    frame    = 0
    t0       = time.time()

    while drone.alive and frame < MAX_FRAMES:
        frame += 1
        CLOCK.tick(60)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); return None

        WIN.blit(maze_surf, (0, 0))
        pygame.draw.rect(WIN, (60,220,80), pygame.Rect(gx,gy,gw,gh), 3, border_radius=4)

        scan       = lidar_scan(drone.x, drone.y, drone.angle, wall_mask)
        lidar_vals = [d for _, d in scan]
        dx_n = (GOAL_CENTER[0]-drone.x)/WIDTH
        dy_n = (GOAL_CENTER[1]-drone.y)/HEIGHT
        inputs = lidar_vals + [dx_n, dy_n, drone.angle/180.0, drone.vel/5.0]

        desired    = math.degrees(math.atan2(GOAL_CENTER[1]-drone.y, GOAL_CENTER[0]-drone.x))
        angle_diff = (desired - drone.angle + 180) % 360 - 180
        steer      = max(-1.0, min(1.0, angle_diff*0.025 + net.activate(inputs)[0]*2.0))

        drone.step(steer, wall_mask)
        drone.draw(scan)

        dist     = math.hypot(GOAL_CENTER[0]-drone.x, GOAL_CENTER[1]-drone.y)
        min_dist = min(min_dist, dist)

        if gx <= drone.x <= gx+gw and gy <= drone.y <= gy+gh:
            reached = True; cause = "goal_reached"; drone.alive = False
        if not drone.alive and not reached:
            cause = "wall_collision"

        hud = pygame.Surface((340, 155), pygame.SRCALPHA)
        hud.fill((0,0,0,165))
        WIN.blit(hud, (10, 10))
        goal_col = (60,220,80) if reached else (255,80,80)
        lines = [
            (f"Method B — Single Population (Baseline)",    (220,220,220)),
            (f"Trained on all 100 mazes simultaneously",    (160,160,160)),
            (f"Frame    : {frame}/{MAX_FRAMES}",            (255,255,255)),
            (f"Min dist : {min_dist:.0f} px",               (255,255,255)),
            (f"Goal     : {'REACHED ✓' if reached else 'NOT YET'}", goal_col),
            (f"Status   : {cause}",                         (200,200,200)),
        ]
        for i, (txt, col) in enumerate(lines):
            WIN.blit(font.render(txt, True, col), (20, 18+i*23))

        pygame.display.update()

    result = {
        "method"               : "single_population_multi_env",
        "test_maze"            : TEST_MAZE_IDX,
        "n_training_mazes"     : 100,
        "goal_reached"         : reached,
        "frames_survived"      : frame,
        "max_frames"           : MAX_FRAMES,
        "min_dist_to_goal_px"  : round(min_dist, 2),
        "cause_of_end"         : cause,
        "test_duration_s"      : round(time.time()-t0, 2),
        "genome_n_nodes"       : len(genome.nodes),
        "genome_n_connections" : len(genome.connections),
    }

    with open(os.path.join(RESULTS_DIR,"test_result_B.json"),"w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(RESULTS_DIR,"test_result_B.csv"),"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=result.keys())
        w.writeheader(); w.writerow(result)

    print("\n── Method B Results ─────────────────────────")
    for k,v in result.items():
        print(f"  {k:<28}: {v}")
    print(f"\n✓ Saved to results/test_result_B.*")
    return result


def show_results(result):
    font_big   = pygame.font.SysFont("arial", 34)
    font_small = pygame.font.SysFont("arial", 20)
    waiting    = True
    while waiting:
        WIN.fill((12,12,20))
        WIN.blit(font_big.render("Method B — Single Population Baseline", True, (255,255,255)),
                 (WIDTH//2-290, 30))
        y = 100
        reached_col = (60,220,80) if result["goal_reached"] else (255,80,80)
        rows = [
            ("Test maze (unseen)", f"Maze {result['test_maze']}",                  (180,180,180)),
            ("Goal reached",       "YES ✓" if result["goal_reached"] else "NO ✗",  reached_col),
            ("Frames survived",    f"{result['frames_survived']} / {result['max_frames']}", (200,200,200)),
            ("Closest to goal",    f"{result['min_dist_to_goal_px']} px",           (200,200,200)),
            ("Ended by",           result["cause_of_end"],                          (200,200,200)),
            ("Genome nodes",       str(result["genome_n_nodes"]),                   (200,200,200)),
            ("Genome connections", str(result["genome_n_connections"]),             (200,200,200)),
        ]
        for label, val, col in rows:
            WIN.blit(font_small.render(f"{label:<28}: {val}", True, col), (80, y))
            y += 32
        WIN.blit(font_small.render("Press ESC to exit", True, (70,70,70)),
                 (WIDTH//2-80, HEIGHT-36))
        pygame.display.update()
        for e in pygame.event.get():
            if e.type == pygame.QUIT: waiting = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE: waiting = False


def run():
    config_path = os.path.join(BASE_DIR, "config-feedforward.txt")
    genome_path = os.path.join(BASE_DIR, "baseline_genome.pkl")

    if not os.path.exists(genome_path):
        print("ERROR: baseline_genome.pkl not found. Run baseline_train.py first.")
        pygame.quit(); return

    config = neat.config.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path
    )
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    print(f"✓ Baseline genome: nodes={len(genome.nodes)} conns={len(genome.connections)}")
    result = run_test(genome, config)
    if result:
        show_results(result)
    pygame.quit()
    print("Next: python visualise.py")


if __name__ == "__main__":
    run()
