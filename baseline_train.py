"""
baseline_train.py  —  Method B: Multi-Environment Single Population
Trains ONE NEAT population where every genome is evaluated on ALL 100
training mazes each generation. Fitness = average across all mazes.
Saves the single best genome as baseline_genome.pkl.

This is the baseline to compare against Method A (genome aggregation).

Run order:
    1. maze_gen.py
    2. train.py  +  aggregate.py  (Method A)
    3. baseline_train.py          ← you are here
    4. baseline_test.py
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
GOAL_RECT   = (870, 220, 100, 160)

MAX_FRAMES    = 1000
N_GENERATIONS = 50
N_MAZES       = 100

RAY_ANGLES = [-70, -40, -15, 0, 15, 40, 70]
RAY_LEN    = 150
RAY_STEP   = 4

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MAZE_DIR    = os.path.join(BASE_DIR, "mazes")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── LIDAR ─────────────────────────────────────
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
        readings.append(dist / RAY_LEN)
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
        self.last_dist = math.hypot(GOAL_CENTER[0]-START[0],
                                    GOAL_CENTER[1]-START[1])
        self.visited  = {}

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
            self.alive = False
            return

        self.x, self.y = nx, ny
        cell = (int(self.x)//60, int(self.y)//60)
        self.visited[cell] = self.visited.get(cell, 0) + 1


# ── Fitness on one maze ────────────────────────
def run_genome_on_maze(net, wall_mask):
    drone   = Drone()
    fitness = 0.0
    gx, gy, gw, gh = GOAL_RECT

    for _ in range(MAX_FRAMES):
        if not drone.alive:
            break

        scan   = lidar_scan(drone.x, drone.y, drone.angle, wall_mask)
        dx_n   = (GOAL_CENTER[0] - drone.x) / WIDTH
        dy_n   = (GOAL_CENTER[1] - drone.y) / HEIGHT
        inputs = scan + [dx_n, dy_n, drone.angle/180.0, drone.vel/5.0]

        desired    = math.degrees(math.atan2(
            GOAL_CENTER[1]-drone.y, GOAL_CENTER[0]-drone.x))
        angle_diff = (desired - drone.angle + 180) % 360 - 180
        steer      = max(-1.0, min(1.0,
                         angle_diff*0.025 + net.activate(inputs)[0]*2.0))

        drone.step(steer, wall_mask)

        curr_dist       = math.hypot(GOAL_CENTER[0]-drone.x,
                                     GOAL_CENTER[1]-drone.y)
        fitness        += (drone.last_dist - curr_dist) * 3.0
        drone.last_dist = curr_dist

        cell = (int(drone.x)//60, int(drone.y)//60)
        if drone.visited.get(cell, 0) > 12:
            fitness -= 5.0

        for s in scan:
            if s < 0.15:
                fitness -= 1.0

        fitness -= abs(drone.turn_vel) * 0.4

        if gx <= drone.x <= gx+gw and gy <= drone.y <= gy+gh:
            fitness += 5000.0
            break

    return fitness


# ── NEAT eval — averaged across all mazes ─────
WALL_MASKS = []
GEN        = 0
gen_log    = []

def eval_genomes(genomes, config):
    global GEN
    GEN += 1

    for _, genome in genomes:
        net           = neat.nn.FeedForwardNetwork.create(genome, config)
        total_fitness = 0.0

        for mask in WALL_MASKS:
            total_fitness += run_genome_on_maze(net, mask)

        genome.fitness = total_fitness / len(WALL_MASKS)

    fitnesses = [g.fitness for _, g in genomes]
    gen_log.append((GEN, max(fitnesses), sum(fitnesses)/len(fitnesses)))

    print(f"  Gen {GEN:02d}/{N_GENERATIONS} | "
          f"best={max(fitnesses):8.1f} | avg={sum(fitnesses)/len(fitnesses):8.1f}")


# ── Main ──────────────────────────────────────
def run():
    global WALL_MASKS

    print("=" * 60)
    print("  Method B — Multi-Environment Single Population (Baseline)")
    print(f"  Mazes: 1–{N_MAZES}  |  Gens: {N_GENERATIONS}  |  Frames: {MAX_FRAMES}")
    print("=" * 60)

    print(f"Loading {N_MAZES} training mazes...")
    WALL_MASKS = []
    for i in range(1, N_MAZES + 1):
        mask = np.load(os.path.join(MAZE_DIR, f"maze_{i:03d}.npy"))
        WALL_MASKS.append(mask)
    print(f"✓ {len(WALL_MASKS)} mazes loaded\n")

    config = neat.config.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        os.path.join(BASE_DIR, "config-feedforward.txt")
    )

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(False))
    p.add_reporter(neat.StatisticsReporter())

    t0     = time.time()
    winner = p.run(eval_genomes, N_GENERATIONS)
    elapsed = time.time() - t0

    print(f"\n✓ Training complete in {elapsed/60:.1f} minutes")
    print(f"  Winner fitness : {winner.fitness:.2f}")
    print(f"  Nodes          : {len(winner.nodes)}")
    print(f"  Connections    : {len(winner.connections)}")

    genome_path = os.path.join(BASE_DIR, "baseline_genome.pkl")
    with open(genome_path, "wb") as f:
        pickle.dump(winner, f)
    print(f"✓ Genome saved  : {genome_path}")

    log_path = os.path.join(RESULTS_DIR, "baseline_training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_fitness", "avg_fitness"])
        writer.writerows(gen_log)
    print(f"✓ Log saved     : {log_path}")

    print("\nNext: python baseline_test.py")


if __name__ == "__main__":
    run()
