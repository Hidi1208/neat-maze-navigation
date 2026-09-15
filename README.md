# Cross-Maze Generalisation in Autonomous Drone Navigation Using NEAT

This project investigates whether a controller assembled from multiple independent NEAT (NeuroEvolution of Augmenting Topologies) evolutions can navigate an unseen maze environment without further training. Two methods are compared:

- **Method A (Genome Aggregation):** NEAT is run independently on 100 procedurally generated mazes. The champion genome from each run is extracted, and all 100 are combined into a single controller by averaging connection weights aligned via NEAT innovation numbers, subject to a 30% majority presence threshold.

- **Method B (Single Population Baseline):** A single NEAT population is trained simultaneously across all 100 mazes, with fitness averaged across all environments per generation.

Both controllers are tested zero-shot on maze 101 — an environment never used during any training procedure.

## Results Summary

| Metric | Method A (Aggregation) | Method B (Single Pop.) |
|---|---|---|
| Goal reached | No | No |
| Frames survived | 338 / 600 | 268 / 600 |
| Closest to goal | 331.06 px | 463.96 px |
| Cause of end | Wall collision | Wall collision |

Method A outperformed Method B on both metrics, suggesting genome aggregation preserves directional navigation better than joint training. Full analysis including failure mode discussion is in the accompanying paper.

## Run Order

```bash
pip install pygame neat-python numpy opencv-python matplotlib
```

```bash
python maze_gen.py          # Generate 101 mazes (~5 seconds)
python train.py             # Method A: 100 independent NEAT runs
python aggregate.py         # Average 100 champion genomes
python test.py              # Test aggregated genome on maze 101

python baseline_train.py    # Method B: single population on all 100 mazes
python baseline_test.py     # Test baseline genome on maze 101

python visualise.py         # Generate all paper charts (saved to results/)
```

## Project Structure

```
├── maze_gen.py              # Procedural maze generator (101 mazes)
├── train.py                 # Method A: independent training per maze
├── aggregate.py             # Genome averaging with majority threshold
├── test.py                  # Method A: visual test on unseen maze
├── baseline_train.py        # Method B: single population multi-env training
├── baseline_test.py         # Method B: visual test on unseen maze
├── visualise.py             # Paper-ready chart generation
├── config-feedforward.txt   # NEAT hyperparameters
└── results/                 # All metrics, charts, and paper summary
```

## Environment

- **Simulator:** Custom Python/Pygame 2D drone simulator
- **Maze size:** 1000 × 600 pixels, procedurally generated vertical walls with gaps
- **Sensors:** 7-ray LIDAR at [-70°, -40°, -15°, 0°, +15°, +40°, +70°]
- **Control:** Hybrid autopilot (goal-seeking) + NEAT neural network (obstacle avoidance)
- **NN Architecture:** 11 inputs → evolved hidden layer → 1 output (steering correction)
