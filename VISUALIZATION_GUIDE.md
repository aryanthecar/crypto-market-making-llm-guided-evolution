# LLM-GE Results Visualization Guide

After completing an LLM-Guided Evolution run, you can visualize your results using the `visualize_results.py` script.

## Quick Start

### Basic Usage

```bash
python visualize_results.py <checkpoint_dir>
```

This will create visualizations in a `visualizations/` directory.

### Examples

**For Titanic (2 objectives: FP, FN):**
```bash
python visualize_results.py titanic_test --output_dir titanic_viz --fitness_weights -1.0,-1.0
```

**For MarketMaking (1 objective):**
```bash
python visualize_results.py marketMaking_test --output_dir mm_viz --fitness_weights 1.0
```

**Plot specific generations:**
```bash
python visualize_results.py checkpoints --generations 0,5,10,20
```

## What Gets Visualized

The script generates several types of visualizations:

### 1. **Fitness Evolution** (`fitness_evolution.png`)
   - Shows how fitness values change over generations
   - Includes best, average, and standard deviation
   - For multi-objective: separate plots for each objective

### 2. **Pareto Front Plots** (`pareto_gen_X.png`)
   - For multi-objective optimization (2+ objectives)
   - Shows all individuals and highlights Pareto-optimal solutions
   - For single objective: shows fitness distribution histogram

### 3. **Population Statistics** (`population_stats.png`)
   - Population size over generations
   - Success rate (percentage of individuals with valid fitness)

### 4. **Ancestry Information**
   - Text output showing the genealogy of top-performing genes
   - Shows mutation types and parent-child relationships

## Command Line Arguments

- `checkpoint_dir`: **Required**. Directory containing checkpoint files (e.g., `titanic_test`, `marketMaking_test`)
- `--output_dir`: Output directory for visualizations (default: `visualizations`)
- `--generations`: Comma-separated list of generations to plot (e.g., `0,5,10,20`). If not specified, plots all generations.
- `--fitness_weights`: Comma-separated fitness weights (e.g., `-1.0,-1.0` for minimization, `1.0` for maximization)

## Understanding the Results

### Checkpoint Files
Checkpoints are saved during evolution in the format: `checkpoint_gen_X.pkl`

Each checkpoint contains:
- `GLOBAL_DATA`: Fitness values and status for each gene
- `population`: Current population of individuals
- `hof`: Hall of Fame (Pareto front)
- `GLOBAL_DATA_ANCESTRY`: Genealogy information

### Fitness Values
- **Titanic**: 2 objectives `(FP, FN)` - both minimized
- **MarketMaking**: 1 objective (combined fitness score) - maximized
- Invalid fitness values (infinite or placeholder) are automatically filtered out

## Tips

1. **For multi-objective problems**: The Pareto front shows the trade-off between objectives. Points on the front are non-dominated solutions.

2. **For single-objective problems**: Focus on the fitness evolution plot to see improvement over generations.

3. **Check population stats**: If success rate is low, many individuals may be failing evaluation.

4. **Ancestry tree**: Helps understand which mutations/operations led to the best solutions.

## Troubleshooting

**No checkpoint files found:**
- Make sure you're pointing to the correct directory (the one specified in `run_improved.py` as the checkpoint folder)

**No valid fitness data:**
- Check that evaluations completed successfully
- Look for `results/{gene_id}_results.txt` files in your SOTA directory

**Plots look empty:**
- Verify that fitness values are being saved correctly
- Check that `FITNESS_WEIGHTS` in your constants file matches the number of objectives

