"""
visualise.py
Generates all paper-ready charts and a full paper summary.

Run AFTER test.py.

Outputs (all in results/):
  fitness_distribution.png   — histogram of best fitness across 100 mazes
  fitness_per_maze.png       — bar chart of best fitness per maze
  topology_stats.png         — node/connection count distribution
  aggregation_weights.png    — weight diversity across genomes
  test_result_card.png       — clean single-card summary for paper
  paper_summary.txt          — full plain-English summary for paper writing
"""

import os
import csv
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
GENOME_DIR  = os.path.join(BASE_DIR, "genomes")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MAZE_DIR    = os.path.join(BASE_DIR, "mazes")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor" : "#0d1117",
    "axes.facecolor"   : "#161b22",
    "axes.edgecolor"   : "#30363d",
    "axes.labelcolor"  : "#c9d1d9",
    "xtick.color"      : "#8b949e",
    "ytick.color"      : "#8b949e",
    "text.color"       : "#c9d1d9",
    "grid.color"       : "#21262d",
    "grid.linestyle"   : "--",
    "grid.alpha"       : 0.6,
    "font.family"      : "DejaVu Sans",
    "axes.titlecolor"  : "#f0f6fc",
})

BLUE   = "#58a6ff"
YELLOW = "#d29922"
GREEN  = "#3fb950"
RED    = "#f85149"
PURPLE = "#bc8cff"


# ── Load training log ─────────────────────────
def load_training_log():
    path = os.path.join(RESULTS_DIR, "training_log.csv")
    mazes, fitnesses, times, n_nodes, n_conns = [], [], [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mazes.append(int(row["maze_idx"]))
            fitnesses.append(float(row["best_fitness"]))
            times.append(float(row["time_s"]))
            n_nodes.append(int(row["n_nodes"]))
            n_conns.append(int(row["n_connections"]))
    return mazes, fitnesses, times, n_nodes, n_conns


# ── Load test result ──────────────────────────
def load_test_result():
    path = os.path.join(RESULTS_DIR, "test_result.json")
    with open(path) as f:
        return json.load(f)


# ── Load aggregation stats ────────────────────
def load_agg_stats():
    path = os.path.join(RESULTS_DIR, "aggregation_stats.json")
    with open(path) as f:
        return json.load(f)


def save_fig(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {name}")


# ── Chart 1: Fitness distribution ─────────────
def plot_fitness_distribution(fitnesses):
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.suptitle("Best Fitness Distribution Across 100 Training Mazes",
                 fontsize=13, fontweight="bold", color="#f0f6fc")

    ax.hist(fitnesses, bins=20, color=BLUE, edgecolor="#0d1117", alpha=0.85)
    ax.axvline(np.mean(fitnesses), color=YELLOW, linewidth=2,
               linestyle="--", label=f"Mean = {np.mean(fitnesses):.1f}")
    ax.axvline(np.median(fitnesses), color=GREEN, linewidth=2,
               linestyle=":",  label=f"Median = {np.median(fitnesses):.1f}")

    ax.set_xlabel("Best Fitness (per maze)")
    ax.set_ylabel("Number of Mazes")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
    ax.grid(True, axis="y")

    save_fig(fig, "fitness_distribution.png")


# ── Chart 2: Fitness per maze (bar) ───────────
def plot_fitness_per_maze(mazes, fitnesses):
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.suptitle("Best Fitness per Training Maze",
                 fontsize=13, fontweight="bold", color="#f0f6fc")

    colors = [GREEN if f > 0 else RED for f in fitnesses]
    ax.bar(mazes, fitnesses, color=colors, alpha=0.8, width=0.8)
    ax.axhline(np.mean(fitnesses), color=YELLOW, linewidth=1.5,
               linestyle="--", label=f"Mean = {np.mean(fitnesses):.1f}")

    ax.set_xlabel("Maze Index (1–100)")
    ax.set_ylabel("Best Fitness")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
    ax.grid(True, axis="y")

    save_fig(fig, "fitness_per_maze.png")


# ── Chart 3: Topology stats ───────────────────
def plot_topology(n_nodes, n_conns):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Winner Genome Topology Across 100 Mazes",
                 fontsize=13, fontweight="bold", color="#f0f6fc")

    ax1.hist(n_nodes, bins=range(min(n_nodes), max(n_nodes)+2),
             color=PURPLE, edgecolor="#0d1117", alpha=0.85)
    ax1.set_title("Node Count Distribution", color="#f0f6fc")
    ax1.set_xlabel("Number of Nodes")
    ax1.set_ylabel("Frequency")
    ax1.grid(True, axis="y")

    ax2.hist(n_conns, bins=15, color=BLUE, edgecolor="#0d1117", alpha=0.85)
    ax2.set_title("Connection Count Distribution", color="#f0f6fc")
    ax2.set_xlabel("Number of Connections")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, axis="y")

    plt.tight_layout()
    save_fig(fig, "topology_stats.png")


# ── Chart 4: Test result card ─────────────────
def plot_test_card(result, agg_stats):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    fig.patch.set_facecolor("#0d1117")

    reached = result["goal_reached"]
    title_col = GREEN if reached else RED

    ax.text(0.5, 0.95, "Generalisation Test — Maze 101 (Unseen)",
            ha="center", va="top", fontsize=14, fontweight="bold",
            color="#f0f6fc", transform=ax.transAxes)

    ax.text(0.5, 0.82,
            "GOAL REACHED ✓" if reached else "GOAL NOT REACHED ✗",
            ha="center", va="top", fontsize=18, fontweight="bold",
            color=title_col, transform=ax.transAxes)

    rows = [
        ("Method",              "Genome Aggregation (100 winners averaged)"),
        ("Training mazes",      "100 independent NEAT runs (mazes 1–100)"),
        ("Test maze",           "Maze 101 (never seen during training)"),
        ("Frames survived",     f"{result['frames_survived']} / {result['max_frames']}"),
        ("Closest to goal",     f"{result['min_dist_to_goal_px']} px"),
        ("Ended by",            result["cause_of_end"].replace("_", " ").title()),
        ("Aggregated nodes",    str(result["genome_n_nodes"])),
        ("Aggregated conns",    str(result["genome_n_connections"])),
        ("Connections included",
            f"{agg_stats['connections_included']} / {agg_stats['total_unique_connections']}"),
        ("Mean weight std dev", f"{agg_stats['weight_std_mean']:.4f}"),
    ]

    y = 0.68
    for label, val in rows:
        ax.text(0.08, y, f"{label}:", ha="left", fontsize=10,
                color="#8b949e", transform=ax.transAxes)
        ax.text(0.55, y, val, ha="left", fontsize=10,
                color="#c9d1d9", transform=ax.transAxes)
        y -= 0.06

    save_fig(fig, "test_result_card.png")


# ── Paper summary text ─────────────────────────
def write_paper_summary(mazes, fitnesses, result, agg_stats):
    path = os.path.join(RESULTS_DIR, "paper_summary.txt")

    with open(path, "w") as f:
        f.write("NEAT DRONE GENERALISATION — FULL PAPER SUMMARY\n")
        f.write("=" * 55 + "\n\n")

        f.write("ABSTRACT RECAP\n")
        f.write("-" * 40 + "\n")
        f.write("NEAT is independently applied to 100 simulated maze puzzles.\n")
        f.write("The best-performing genome from each run is extracted.\n")
        f.write("These 100 genomes are averaged to form a single generalised controller.\n")
        f.write("The controller is then tested on an unseen 101st maze.\n\n")

        f.write("ENVIRONMENT\n")
        f.write("-" * 40 + "\n")
        f.write("Simulator    : Custom Python/Pygame drone simulator\n")
        f.write("Maze size    : 1000 x 600 pixels\n")
        f.write("Obstacles    : Procedurally generated vertical walls with gaps\n")
        f.write("Sensors      : 7-ray LIDAR (ray scattering) at angles "
                "[-70, -40, -15, 0, 15, 40, 70] degrees\n")
        f.write("Control      : Hybrid (goal-directed autopilot + NEAT correction)\n")
        f.write("Inputs       : 7 LIDAR readings + goal dx + goal dy + angle + velocity = 11\n")
        f.write("Outputs      : 1 (steering correction)\n\n")

        f.write("TRAINING\n")
        f.write("-" * 40 + "\n")
        f.write(f"Training mazes     : {len(mazes)} (procedurally generated, seeds 1-100)\n")
        f.write(f"NEAT runs          : {len(mazes)} (one independent run per maze)\n")
        f.write(f"Best fitness mean  : {np.mean(fitnesses):.2f}\n")
        f.write(f"Best fitness std   : {np.std(fitnesses):.2f}\n")
        f.write(f"Best fitness min   : {np.min(fitnesses):.2f}\n")
        f.write(f"Best fitness max   : {np.max(fitnesses):.2f}\n")
        f.write(f"Best fitness median: {np.median(fitnesses):.2f}\n\n")

        f.write("GENOME AGGREGATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Genomes aggregated          : {agg_stats['n_genomes']}\n")
        f.write(f"Majority threshold          : {agg_stats['majority_threshold_pct']:.0f}% "
                f"({int(agg_stats['n_genomes'] * agg_stats['majority_threshold_pct']/100)} genomes)\n")
        f.write(f"Unique connections found    : {agg_stats['total_unique_connections']}\n")
        f.write(f"Connections included        : {agg_stats['connections_included']}\n")
        f.write(f"Connections excluded        : {agg_stats['connections_excluded']}\n")
        f.write(f"Unique nodes found          : {agg_stats['total_unique_nodes']}\n")
        f.write(f"Nodes included              : {agg_stats['nodes_included']}\n")
        f.write(f"Mean weight std dev         : {agg_stats['weight_std_mean']:.4f}\n")
        f.write("  (measures disagreement between genomes — lower = more consensus)\n\n")

        f.write("GENERALISATION TEST — MAZE 101 (UNSEEN)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Test maze              : Maze 101 (never seen during training)\n")
        f.write(f"Goal reached           : {'YES' if result['goal_reached'] else 'NO'}\n")
        f.write(f"Frames survived        : {result['frames_survived']} / {result['max_frames']}\n")
        f.write(f"Closest to goal        : {result['min_dist_to_goal_px']} px\n")
        f.write(f"Cause of termination   : {result['cause_of_end']}\n")
        f.write(f"Aggregated genome nodes: {result['genome_n_nodes']}\n")
        f.write(f"Aggregated genome conns: {result['genome_n_connections']}\n\n")

        f.write("INTERPRETATION\n")
        f.write("-" * 40 + "\n")
        if result["goal_reached"]:
            f.write("The aggregated genome successfully navigated the unseen maze.\n")
            f.write("This supports the hypothesis that genome aggregation can produce\n")
            f.write("a generalised controller from environment-specific winners.\n")
        else:
            f.write("The aggregated genome did not reach the goal in the unseen maze.\n")
            f.write(f"However, it survived {result['frames_survived']} frames and reached\n")
            f.write(f"within {result['min_dist_to_goal_px']} px of the goal.\n")
            f.write("This indicates partial generalisation. Further training iterations\n")
            f.write("or a lower majority threshold may improve cross-environment transfer.\n")

        f.write("\n")
        f.write("CHARTS GENERATED (all in results/)\n")
        f.write("-" * 40 + "\n")
        f.write("  fitness_distribution.png  — histogram of fitness across 100 mazes\n")
        f.write("  fitness_per_maze.png      — bar chart, fitness per maze\n")
        f.write("  topology_stats.png        — node/connection distributions\n")
        f.write("  test_result_card.png      — clean summary card for paper\n")

    print(f"  ✓ paper_summary.txt")
    return path


# ── Entry point ───────────────────────────────
def run():
    print("Generating paper charts and summary...\n")

    missing = []
    for name in ["training_log.csv", "test_result.json", "aggregation_stats.json"]:
        if not os.path.exists(os.path.join(RESULTS_DIR, name)):
            missing.append(name)

    if missing:
        print(f"ERROR: Missing result files: {missing}")
        print("Run the pipeline in order: maze_gen → train → aggregate → test → visualise")
        return

    mazes, fitnesses, times, n_nodes, n_conns = load_training_log()
    result    = load_test_result()
    agg_stats = load_agg_stats()

    plot_fitness_distribution(fitnesses)
    plot_fitness_per_maze(mazes, fitnesses)
    plot_topology(n_nodes, n_conns)
    plot_test_card(result, agg_stats)
    write_paper_summary(mazes, fitnesses, result, agg_stats)

    print(f"\n✓ All outputs saved to: {RESULTS_DIR}/")
    print("  Share the entire results/ folder with your co-author.")


if __name__ == "__main__":
    run()