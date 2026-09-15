"""
train.py
Trains NEAT independently on each of the 100 training mazes.
For each maze, one winner genome is saved to genomes/.

Run order:
    1. maze_gen.py
    2. train.py   ← you are here
    3. aggregate.py
    4. test.py
    5. visualise.py
"""

import neat
import math
import os
import pickle
import csv
import time
import numpy as np

# ── Constants ─────────────────────────────────
WIDTH  = 1000
HEIGHT = 600

START       = (50, 300)
GOAL_CENTER = (920, 300)
GOAL_RECT   = (870, 220, 100, 160)   # x, y, w, h

MAX_FRAMES    = 600     # frames per drone per generation
N_GENERATIONS = 50      # generations per maze
N_MAZES       = 100     # mazes to train on

RAY_ANGLES = [-70, -40, -15, 0, 15, 40, 70]
RAY_LEN    = 150
RAY_STEP   = 4          # step size — larger = faster, less precise

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MAZE_DIR    = os.path.join(BASE_DIR, "mazes")
GENOME_DIR  = os.path.join(BASE_DIR, "genomes")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(GENOME_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── LIDAR (headless, numpy-based) ─────────────
def lidar_scan(x: float, y: float, angle: float,
               wall_mask: np.ndarray) -> list:
    readings = []
    for offset in RAY_ANGLES:
        ray_angle = angle + offset
        cos_a = math.cos(math.radians(ray_angle))
        sin_a = math.sin(math.radians(ray_angle))
        dist  = RAY_LEN

        for d in range(0, RAY_LEN, RAY_STEP):
            px = int(x + cos_a * d)
            py = int(y + sin_a * d)
            if px < 0 or px >= WIDTH or py < 0 or py >= HEIGHT:
                dist = d; break
            if wall_mask[py, px]:
                dist = d; break

        readings.append(dist / RAY_LEN)
    return readings   # 7 normalised values


# ── Drone simulation (headless) ────────────────
class Drone:
    def __init__(self):
        self.x        = float(START[0])
        self.y        = float(START[1])
        self.angle    = 0.0
        self.vel      = 0.0
        self.turn_vel = 0.0
        self.alive    = True
        self.last_dist = math.hypot(GOAL_CENTER[0] - self.x,
                                    GOAL_CENTER[1] - self.y)
        self.visited  = {}   # cell -> visit count  (anti-circling)

    def step(self, steer: float, wall_mask: np.ndarray):
        self.turn_vel += steer * 0.4
        self.turn_vel *= 0.85
        self.angle    += self.turn_vel
        self.vel      += 0.08
        self.vel      *= 0.97

        nx = self.x + math.cos(math.radians(self.angle)) * self.vel
        ny = self.y + math.sin(math.radians(self.angle)) * self.vel

        # Collision — check a small bounding box
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

        # Anti-circling tracker
        cell = (int(self.x) // 60, int(self.y) // 60)
        self.visited[cell] = self.visited.get(cell, 0) + 1


# ── Fitness evaluation ─────────────────────────
def run_genome(net, wall_mask: np.ndarray) -> float:
    drone   = Drone()
    fitness = 0.0
    gx_rect, gy_rect, gw_rect, gh_rect = GOAL_RECT

    for _ in range(MAX_FRAMES):
        if not drone.alive:
            break

        scan = lidar_scan(drone.x, drone.y, drone.angle, wall_mask)

        # Normalised goal direction
        dx = (GOAL_CENTER[0] - drone.x) / WIDTH
        dy = (GOAL_CENTER[1] - drone.y) / HEIGHT

        inputs = scan + [dx, dy, drone.angle / 180.0, drone.vel / 5.0]
        # 7 LIDAR + dx + dy + angle + vel = 11 inputs

        # Hybrid control
        desired    = math.degrees(math.atan2(
            GOAL_CENTER[1] - drone.y, GOAL_CENTER[0] - drone.x))
        angle_diff = (desired - drone.angle + 180) % 360 - 180
        base_steer = angle_diff * 0.025
        neat_out   = net.activate(inputs)[0]
        steer      = max(-1.0, min(1.0, base_steer + neat_out * 2.0))

        drone.step(steer, wall_mask)

        # ── Fitness components ─────────────────

        # 1. Progress toward goal
        curr_dist      = math.hypot(GOAL_CENTER[0] - drone.x,
                                    GOAL_CENTER[1] - drone.y)
        fitness       += (drone.last_dist - curr_dist) * 3.0
        drone.last_dist = curr_dist
        if curr_dist < 200:
            fitness += 50
        if curr_dist < 100:
            fitness += 100
        if curr_dist < 120:
            fitness -= 2.0    

        # 2. Anti-circling penalty
        cell = (int(drone.x) // 60, int(drone.y) // 60)
        if drone.visited.get(cell, 0) > 12:
            fitness -= 6.0

        # 3. Wall proximity penalty (light)
        for s in scan:
            if s < 0.15:
                fitness -= 1.0

        # 4. Smooth movement
        fitness -= abs(drone.turn_vel) * 0.2

        # 5. Goal reached
        if (gx_rect <= drone.x <= gx_rect + gw_rect and
                gy_rect <= drone.y <= gy_rect + gh_rect):
            fitness += 5000.0
            break

    return fitness


# ── NEAT evaluation function ───────────────────
CURRENT_WALL_MASK = None   # set before each maze run

def eval_genomes(genomes, config):
    global CURRENT_WALL_MASK
    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = run_genome(net, CURRENT_WALL_MASK)


# ── Train on one maze ──────────────────────────
def train_one_maze(maze_idx: int, config) -> tuple:
    """
    Trains NEAT on maze_idx.
    Returns (winner_genome, best_fitness, training_time_s).
    """
    global CURRENT_WALL_MASK

    mask_path = os.path.join(MAZE_DIR, f"maze_{maze_idx:03d}.npy")
    CURRENT_WALL_MASK = np.load(mask_path)

    p = neat.Population(config)
    # Suppress per-generation output for cleanliness
    # p.add_reporter(neat.StdOutReporter(False))

    t0     = time.time()
    winner = p.run(eval_genomes, N_GENERATIONS)
    elapsed = time.time() - t0

    return winner, winner.fitness, elapsed


# ── Main training loop ─────────────────────────
def run():
    config_path = os.path.join(BASE_DIR, "config-feedforward.txt")
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    log_path = os.path.join(RESULTS_DIR, "training_log.csv")
    log_rows = []

    print("=" * 60)
    print("  NEAT Drone — Training on 100 Mazes")
    print("=" * 60)
    print(f"  Generations per maze : {N_GENERATIONS}")
    print(f"  Population size      : check config")
    print(f"  Max frames/episode   : {MAX_FRAMES}")
    print(f"  Total mazes          : {N_MAZES}")
    print("=" * 60)

    total_t0 = time.time()

    for maze_idx in range(1, N_MAZES + 1):
        genome_path = os.path.join(GENOME_DIR, f"genome_{maze_idx:03d}.pkl")

        # Skip if already trained (allows resuming)
        if os.path.exists(genome_path):
            print(f"  [{maze_idx:03d}/{N_MAZES}] Already trained — skipping")
            continue

        winner, best_fit, elapsed = train_one_maze(maze_idx, config)

        # Save genome
        with open(genome_path, "wb") as f:
            pickle.dump(winner, f)

        log_rows.append({
            "maze_idx"    : maze_idx,
            "best_fitness": round(best_fit, 4),
            "time_s"      : round(elapsed, 2),
            "n_nodes"     : len(winner.nodes),
            "n_connections": len(winner.connections),
        })

        elapsed_total = time.time() - total_t0
        remaining     = (elapsed_total / maze_idx) * (N_MAZES - maze_idx)

        print(f"  [{maze_idx:03d}/{N_MAZES}] fit={best_fit:8.1f} "
              f"| {elapsed:.1f}s | ETA {remaining/60:.1f}min "
              f"| nodes={len(winner.nodes)} conns={len(winner.connections)}")

        # Write log incrementally so you can check progress
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)

    total_elapsed = time.time() - total_t0
    print("=" * 60)
    print(f"✓ Training complete in {total_elapsed/60:.1f} minutes")
    print(f"✓ {N_MAZES} genomes saved to: {GENOME_DIR}/")
    print(f"✓ Training log saved to: {log_path}")
    print("\nNext step: python aggregate.py")


if __name__ == "__main__":
    run()