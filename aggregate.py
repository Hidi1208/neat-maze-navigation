"""
aggregate.py
Loads the 100 winner genomes and aggregates them into a single
generalised controller using weight averaging.

Algorithm:
  For every unique connection key (in_node, out_node) found across
  all 100 genomes:
    - Collect weights from every genome that has that connection
    - Include the connection if present in >= MAJORITY_THRESHOLD genomes
    - Set weight = mean of collected weights

  Same logic applies to node biases.

Run order:
    1. maze_gen.py
    2. train.py
    3. aggregate.py   ← you are here
    4. test.py
    5. visualise.py
"""

import os
import pickle
import copy
import csv
import json
import numpy as np

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
GENOME_DIR  = os.path.join(BASE_DIR, "genomes")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

N_MAZES             = 100
MAJORITY_THRESHOLD  = 0.3   # connection must appear in >=30% of genomes to be included


def load_genomes() -> list:
    genomes = []
    missing = []
    for i in range(1, N_MAZES + 1):
        path = os.path.join(GENOME_DIR, f"genome_{i:03d}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                genomes.append(pickle.load(f))
        else:
            missing.append(i)

    if missing:
        print(f"WARNING: Missing genomes for mazes: {missing}")

    print(f"✓ Loaded {len(genomes)} genomes")
    return genomes


def aggregate_genomes(genomes: list):
    """
    Aggregate N genomes into one via connection weight averaging.
    Returns the aggregated genome.
    """
    n = len(genomes)
    threshold = int(n * MAJORITY_THRESHOLD)

    # ── Collect connection weights ─────────────
    conn_weights = {}    # (in_key, out_key) -> [weights]
    conn_enabled = {}    # (in_key, out_key) -> [enabled bools]

    for genome in genomes:
        for conn_key, conn in genome.connections.items():
            if conn_key not in conn_weights:
                conn_weights[conn_key] = []
                conn_enabled[conn_key] = []
            conn_weights[conn_key].append(conn.weight)
            conn_enabled[conn_key].append(conn.enabled)

    # ── Collect node biases ────────────────────
    node_biases = {}    # node_key -> [biases]

    for genome in genomes:
        for node_key, node in genome.nodes.items():
            if node_key not in node_biases:
                node_biases[node_key] = []
            node_biases[node_key].append(node.bias)

    # ── Build aggregated genome ────────────────
    # Start from a deep copy of the first genome as the base structure
    agg = copy.deepcopy(genomes[0])

    # Clear connections — rebuild from majority vote
    agg.connections = {}

    # Statistics for paper
    stats = {
        "n_genomes"             : n,
        "majority_threshold_pct": MAJORITY_THRESHOLD * 100,
        "total_unique_connections": len(conn_weights),
        "connections_included"  : 0,
        "connections_excluded"  : 0,
        "total_unique_nodes"    : len(node_biases),
        "nodes_included"        : 0,
        "weight_std_mean"       : 0.0,   # mean of per-connection weight std
        "fitness_stats"         : {},
    }

    weight_stds = []

    for conn_key, weights in conn_weights.items():
        if len(weights) >= threshold:
            # Find a genome that has this connection to copy its structure
            for g in genomes:
                if conn_key in g.connections:
                    new_conn      = copy.deepcopy(g.connections[conn_key])
                    new_conn.weight  = float(np.mean(weights))
                    # Enable if majority say enabled
                    new_conn.enabled = (sum(conn_enabled[conn_key]) >= len(conn_enabled[conn_key]) / 2)
                    agg.connections[conn_key] = new_conn
                    weight_stds.append(float(np.std(weights)))
                    stats["connections_included"] += 1
                    break
        else:
            stats["connections_excluded"] += 1

    # Update node biases for nodes already in base genome
    agg.nodes = {}

    for node_key, biases in node_biases.items():
        if len(biases) >= threshold:
            # Get node from any genome that has it
            for g in genomes:
                if node_key in g.nodes:
                    new_node      = copy.deepcopy(g.nodes[node_key])
                    new_node.bias = float(np.mean(biases))
                    agg.nodes[node_key] = new_node
                    stats["nodes_included"] += 1
                    break

    stats["weight_std_mean"] = float(np.mean(weight_stds)) if weight_stds else 0.0

    # Fitness stats across training
    fitnesses = [g.fitness for g in genomes if g.fitness is not None]
    stats["fitness_stats"] = {
        "mean"   : round(float(np.mean(fitnesses)), 4),
        "std"    : round(float(np.std(fitnesses)),  4),
        "min"    : round(float(np.min(fitnesses)),  4),
        "max"    : round(float(np.max(fitnesses)),  4),
        "median" : round(float(np.median(fitnesses)), 4),
    }

    return agg, stats


def run():
    print("=" * 60)
    print("  Genome Aggregation")
    print("=" * 60)

    genomes = load_genomes()

    if len(genomes) == 0:
        print("ERROR: No genomes found. Run train.py first.")
        return

    agg_genome, stats = aggregate_genomes(genomes)

    # ── Save aggregated genome ─────────────────
    agg_path = os.path.join(BASE_DIR, "aggregated_genome.pkl")
    with open(agg_path, "wb") as f:
        pickle.dump(agg_genome, f)
    print(f"✓ Aggregated genome saved: {agg_path}")

    # ── Save aggregation stats ─────────────────
    stats_path = os.path.join(RESULTS_DIR, "aggregation_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Aggregation stats saved: {stats_path}")

    # ── Save human-readable summary ────────────
    summary_path = os.path.join(RESULTS_DIR, "aggregation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("GENOME AGGREGATION SUMMARY\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Genomes aggregated     : {stats['n_genomes']}\n")
        f.write(f"Majority threshold     : {stats['majority_threshold_pct']:.0f}% "
                f"({int(stats['n_genomes'] * MAJORITY_THRESHOLD)} genomes)\n\n")
        f.write("CONNECTION STATISTICS\n")
        f.write(f"  Unique connections found  : {stats['total_unique_connections']}\n")
        f.write(f"  Connections included       : {stats['connections_included']}\n")
        f.write(f"  Connections excluded       : {stats['connections_excluded']}\n")
        f.write(f"  Mean weight std dev        : {stats['weight_std_mean']:.4f}\n")
        f.write(f"  (higher std = more disagreement between genomes)\n\n")
        f.write("NODE STATISTICS\n")
        f.write(f"  Unique nodes found         : {stats['total_unique_nodes']}\n")
        f.write(f"  Nodes included             : {stats['nodes_included']}\n\n")
        f.write("TRAINING FITNESS (across 100 mazes)\n")
        fs = stats["fitness_stats"]
        f.write(f"  Mean best fitness  : {fs['mean']}\n")
        f.write(f"  Std dev            : {fs['std']}\n")
        f.write(f"  Min best fitness   : {fs['min']}\n")
        f.write(f"  Max best fitness   : {fs['max']}\n")
        f.write(f"  Median best fitness: {fs['median']}\n")

    print(f"✓ Summary saved: {summary_path}")

    # Print to console too
    print("\n── Aggregation Results ──────────────────────")
    print(f"  Connections included : {stats['connections_included']} "
          f"/ {stats['total_unique_connections']}")
    print(f"  Nodes included       : {stats['nodes_included']}")
    print(f"  Mean weight std      : {stats['weight_std_mean']:.4f}")
    print(f"  Training fitness — mean: {stats['fitness_stats']['mean']}, "
          f"max: {stats['fitness_stats']['max']}")
    print("\nNext step: python test.py")


if __name__ == "__main__":
    run()