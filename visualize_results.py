#!/usr/bin/env python3
"""
Visualization script for LLM-Guided Evolution results.

This script loads checkpoint files from LLM-GE runs and creates various visualizations:
- Pareto front plots (for multi-objective optimization)
- Fitness evolution over generations
- Population statistics
- Ancestry trees

Usage:
    python visualize_results.py <checkpoint_dir> [--output_dir <output_dir>] [--generations <gen1,gen2,...>]
    
Example:
    python visualize_results.py checkpoints --output_dir visualizations --generations 0,5,10,20
"""

import os
import sys
import pickle
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import defaultdict
import math

# DEAP imports for recreating classes
from deap import base, creator, tools

def extract_generation(filename):
    """Extract generation number from checkpoint filename."""
    real_filename = os.path.split(filename)[1]
    return int(real_filename.split('_')[2].split('.')[0])

def setup_deap_classes(num_objectives=None, fitness_weights=None):
    """Setup DEAP classes needed for loading checkpoints."""
    # Clear existing classes to avoid conflicts
    try:
        if hasattr(creator, "FitnessMulti"):
            del creator.FitnessMulti
    except:
        pass
    try:
        if hasattr(creator, "Individual"):
            del creator.Individual
    except:
        pass
    
    # Determine number of objectives and weights
    if fitness_weights is not None:
        num_objectives = len(fitness_weights)
        weights = fitness_weights
    elif num_objectives is not None:
        # Default to minimization for all objectives
        weights = tuple([-1.0] * num_objectives)
    else:
        # Default: assume 2 objectives (common case)
        num_objectives = 2
        weights = (-1.0, -1.0)
    
    # Create fitness class
    try:
        creator.create("FitnessMulti", base.Fitness, weights=weights)
    except:
        # Class might already exist with different weights, try to recreate
        if hasattr(creator, "FitnessMulti"):
            del creator.FitnessMulti
        creator.create("FitnessMulti", base.Fitness, weights=weights)
    
    # Create individual class
    try:
        creator.create("Individual", list, fitness=creator.FitnessMulti, file_id=None)
    except:
        # Class might already exist, try to recreate
        if hasattr(creator, "Individual"):
            del creator.Individual
        creator.create("Individual", list, fitness=creator.FitnessMulti, file_id=None)
    
    return num_objectives, weights

def peek_checkpoint_structure(checkpoint_path):
    """Peek at checkpoint to determine number of objectives by trying common configurations."""
    # Try common configurations
    common_configs = [
        (1, (1.0,)),           # Single objective (maximize)
        (1, (-1.0,)),          # Single objective (minimize)
        (2, (-1.0, -1.0)),     # Two objectives (both minimize) - most common
        (2, (1.0, 1.0)),       # Two objectives (both maximize)
    ]
    
    for num_obj, weights in common_configs:
        try:
            setup_deap_classes(num_obj, weights)
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict) and 'GLOBAL_DATA' in data:
                    global_data = data['GLOBAL_DATA']
                    # Find first valid fitness to verify structure
                    for gene_id, gene_data in global_data.items():
                        if isinstance(gene_data, dict) and 'fitness' in gene_data:
                            fitness = gene_data['fitness']
                            if fitness is not None:
                                if isinstance(fitness, (tuple, list)):
                                    if len(fitness) == num_obj:
                                        return num_obj
                                elif num_obj == 1:
                                    return 1
        except:
            continue
    
    return None

def load_checkpoint_safe(checkpoint_path):
    """Safely load checkpoint by extracting only GLOBAL_DATA without DEAP objects."""
    # First, ensure DEAP classes exist (with dummy values if needed)
    # This helps with the unpickling process
    try:
        if not hasattr(creator, 'FitnessMulti'):
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
    except:
        pass
    
    try:
        if not hasattr(creator, 'Individual'):
            creator.create("Individual", list, fitness=creator.FitnessMulti, file_id=None)
    except:
        pass
    
    # Monkey-patch creator to handle meta_create calls
    if not hasattr(creator, 'meta_create'):
        def meta_create(name, base, *args, **kwargs):
            # This is what DEAP uses internally - accept variable args
            class_dict = dict(base.__dict__)
            class_dict.update(kwargs)
            # Handle any additional args
            if args:
                # Sometimes meta_create is called with additional positional args
                pass
            return type(name, (base,), class_dict)
        creator.meta_create = meta_create
    
    class SafeUnpickler(pickle.Unpickler):
        """Custom unpickler that handles missing DEAP classes."""
        def find_class(self, module, name):
            # For DEAP creator classes, try to get them or create dummies
            if module == 'deap.creator':
                if name == 'FitnessMulti':
                    if hasattr(creator, 'FitnessMulti'):
                        return creator.FitnessMulti
                    # Create a dummy
                    class DummyFitness:
                        def __init__(self, *args, **kwargs):
                            self.values = None
                    return DummyFitness
                elif name == 'Individual':
                    if hasattr(creator, 'Individual'):
                        return creator.Individual
                    # Create a dummy
                    class DummyIndividual(list):
                        def __init__(self, *args, **kwargs):
                            if args:
                                super().__init__(args[0] if isinstance(args[0], list) else args)
                            else:
                                super().__init__()
                            self.fitness = type('Fitness', (), {'values': None})()
                    return DummyIndividual
            # For everything else, use default behavior
            try:
                return super().find_class(module, name)
            except (AttributeError, ModuleNotFoundError):
                # If class not found, return a generic object
                return object
    
    try:
        with open(checkpoint_path, 'rb') as file:
            # Try with safe unpickler first
            unpickler = SafeUnpickler(file)
            checkpoint_data = unpickler.load()
            # Ensure we have the required keys
            if not isinstance(checkpoint_data, dict):
                checkpoint_data = {}
            checkpoint_data.setdefault('GLOBAL_DATA', {})
            checkpoint_data.setdefault('GLOBAL_DATA_HIST', {})
            checkpoint_data.setdefault('GLOBAL_DATA_ANCESTRY', {})
            checkpoint_data.setdefault('population', [])
            checkpoint_data.setdefault('hof', None)
            return checkpoint_data
    except Exception as e:
        # If safe unpickler fails, try normal loading
        try:
            with open(checkpoint_path, 'rb') as file:
                checkpoint_data = pickle.load(file)
                return checkpoint_data
        except Exception as e2:
            print(f"Warning: Could not load checkpoint {checkpoint_path}: {e2}")
            # Return minimal structure
            return {
                'GLOBAL_DATA': {},
                'GLOBAL_DATA_HIST': {},
                'GLOBAL_DATA_ANCESTRY': {},
                'population': [],
                'hof': None
            }

def load_checkpoint(checkpoint_path, num_objectives=None, fitness_weights=None, results_dir=None):
    """Load a checkpoint file and return its data."""
    # Always use safe loading first to avoid DEAP class issues
    checkpoint_data = load_checkpoint_safe(checkpoint_path)
    
    # If we got GLOBAL_DATA with data, we're good (that's all we need for visualization)
    if checkpoint_data.get('GLOBAL_DATA') and len(checkpoint_data.get('GLOBAL_DATA', {})) > 0:
        return checkpoint_data
    
    # If safe loading didn't work, try full loading with DEAP classes as fallback
    setup_deap_classes(num_objectives, fitness_weights)
    try:
        with open(checkpoint_path, 'rb') as file:
            checkpoint_data = pickle.load(file)
        return checkpoint_data
    except (AttributeError, TypeError) as e:
        # If DEAP class error, try loading from results files as fallback
        if 'meta_create' in str(e) or 'creator' in str(e).lower():
            if results_dir and os.path.exists(results_dir):
                print(f"  Loading fitness data from results directory: {results_dir}")
                global_data = load_fitness_from_results(results_dir, num_objectives or 2)
                checkpoint_data['GLOBAL_DATA'] = global_data
            return checkpoint_data
        raise

def get_all_checkpoints(checkpoint_dir):
    """Get all checkpoint files sorted by generation."""
    pattern = os.path.join(checkpoint_dir, 'checkpoint_gen_*.pkl')
    checkpoint_files = glob.glob(pattern)
    checkpoint_files = sorted(checkpoint_files, key=extract_generation)
    return checkpoint_files

def load_fitness_from_results(results_dir, num_objectives=2):
    """Load fitness data directly from results files as fallback."""
    global_data = {}
    results_pattern = os.path.join(results_dir, '*_results.txt')
    result_files = glob.glob(results_pattern)
    
    for result_file in result_files:
        try:
            # Extract gene_id from filename
            gene_id = os.path.basename(result_file).replace('_results.txt', '')
            
            # Read fitness values
            with open(result_file, 'r') as f:
                content = f.read().strip()
                values = [float(v.strip()) for v in content.split(',')]
                
                # Take only the number of objectives we need
                fitness = tuple(values[:num_objectives])
                
                # Create GLOBAL_DATA entry
                global_data[gene_id] = {
                    'fitness': fitness,
                    'status': 'completed',
                    'sub_flag': True
                }
        except Exception as e:
            # Skip files that can't be read
            continue
    
    return global_data

def find_results_directory(checkpoint_dir):
    """Try to find the results directory based on checkpoint location."""
    # Check common locations
    possible_paths = [
        os.path.join(os.path.dirname(checkpoint_dir), 'sota', 'Titanic', 'results'),
        os.path.join(os.path.dirname(checkpoint_dir), 'sota', 'MarketMaking', 'results'),
        os.path.join(checkpoint_dir, '..', 'sota', 'Titanic', 'results'),
        os.path.join(checkpoint_dir, '..', 'sota', 'MarketMaking', 'results'),
        os.path.join(os.path.dirname(os.path.dirname(checkpoint_dir)), 'sota', 'Titanic', 'results'),
        os.path.join(os.path.dirname(os.path.dirname(checkpoint_dir)), 'sota', 'MarketMaking', 'results'),
    ]
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path) and os.path.isdir(abs_path):
            result_files = glob.glob(os.path.join(abs_path, '*_results.txt'))
            if len(result_files) > 0:
                return abs_path
    
    return None

def extract_fitness_values(global_data):
    """Extract fitness values from GLOBAL_DATA, filtering out invalid ones."""
    fitness_data = []
    gene_ids = []
    
    for gene_id, data in global_data.items():
        if 'fitness' in data and data['fitness'] is not None:
            fitness = data['fitness']
            # Check if fitness is valid (not infinite or placeholder)
            if isinstance(fitness, (tuple, list)):
                if all(math.isfinite(f) for f in fitness):
                    # Check if it's not a placeholder (very large negative number)
                    if not all(abs(f) > 1e10 for f in fitness):
                        fitness_data.append(fitness)
                        gene_ids.append(gene_id)
            elif math.isfinite(fitness):
                fitness_data.append((fitness,))
                gene_ids.append(gene_id)
    
    return fitness_data, gene_ids

def plot_pareto_front(checkpoint_data, generation, output_path, fitness_weights=None, num_objectives=None):
    """Plot Pareto front for multi-objective optimization."""
    global_data = checkpoint_data.get('GLOBAL_DATA', {})
    fitness_data, gene_ids = extract_fitness_values(global_data)
    
    if len(fitness_data) == 0:
        print(f"Warning: No valid fitness data found for generation {generation}")
        return
    
    # Determine number of objectives
    num_objectives = len(fitness_data[0])
    
    if num_objectives == 1:
        # Single objective - plot histogram or line plot
        fitness_values = [f[0] for f in fitness_data]
        plt.figure(figsize=(10, 6))
        plt.hist(fitness_values, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Fitness Value')
        plt.ylabel('Frequency')
        plt.title(f'Fitness Distribution - Generation {generation}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    elif num_objectives == 2:
        # Two objectives - plot Pareto front
        obj1_values = [f[0] for f in fitness_data]
        obj2_values = [f[1] for f in fitness_data]
        
        # Apply fitness weights if provided
        if fitness_weights:
            if fitness_weights[0] < 0:
                obj1_values = [-v for v in obj1_values]  # Flip for minimization
            if fitness_weights[1] < 0:
                obj2_values = [-v for v in obj2_values]  # Flip for minimization
        
        # Find Pareto front
        pareto_indices = find_pareto_front(obj1_values, obj2_values, minimize=(True, True))
        pareto_obj1 = [obj1_values[i] for i in pareto_indices]
        pareto_obj2 = [obj2_values[i] for i in pareto_indices]
        
        # Sort for plotting
        sorted_indices = np.argsort(pareto_obj1)
        pareto_obj1 = [pareto_obj1[i] for i in sorted_indices]
        pareto_obj2 = [pareto_obj2[i] for i in sorted_indices]
        
        plt.figure(figsize=(10, 8))
        plt.scatter(obj1_values, obj2_values, alpha=0.6, s=50, label='All Individuals', color='blue')
        plt.plot(pareto_obj1, pareto_obj2, 'r--', linewidth=2, label='Pareto Front', alpha=0.8)
        plt.scatter(pareto_obj1, pareto_obj2, color='red', s=100, marker='*', 
                   label='Pareto Optimal', zorder=5)
        
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title(f'Pareto Front - Generation {generation}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    else:
        # More than 2 objectives - create pairwise plots
        n_pairs = num_objectives * (num_objectives - 1) // 2
        cols = min(3, n_pairs)
        rows = (n_pairs + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if n_pairs == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        pair_idx = 0
        for i in range(num_objectives):
            for j in range(i+1, num_objectives):
                obj_i = [f[i] for f in fitness_data]
                obj_j = [f[j] for f in fitness_data]
                
                axes[pair_idx].scatter(obj_i, obj_j, alpha=0.6, s=50)
                axes[pair_idx].set_xlabel(f'Objective {i+1}')
                axes[pair_idx].set_ylabel(f'Objective {j+1}')
                axes[pair_idx].set_title(f'Objectives {i+1} vs {j+1}')
                axes[pair_idx].grid(True, alpha=0.3)
                pair_idx += 1
        
        # Hide unused subplots
        for idx in range(pair_idx, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Multi-Objective Fitness - Generation {generation}', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

def find_pareto_front(obj1_values, obj2_values, minimize=(True, True)):
    """Find indices of Pareto-optimal solutions."""
    pareto_indices = []
    n = len(obj1_values)
    
    for i in range(n):
        is_pareto = True
        for j in range(n):
            if i == j:
                continue
            # Check if j dominates i
            if minimize[0] and minimize[1]:
                if obj1_values[j] <= obj1_values[i] and obj2_values[j] <= obj2_values[i] and \
                   (obj1_values[j] < obj1_values[i] or obj2_values[j] < obj2_values[i]):
                    is_pareto = False
                    break
            elif not minimize[0] and not minimize[1]:
                if obj1_values[j] >= obj1_values[i] and obj2_values[j] >= obj2_values[i] and \
                   (obj1_values[j] > obj1_values[i] or obj2_values[j] > obj2_values[i]):
                    is_pareto = False
                    break
            # Mixed cases can be added if needed
        
        if is_pareto:
            pareto_indices.append(i)
    
    return pareto_indices

def plot_evolution(checkpoint_files, output_dir, fitness_weights=None, results_dir=None, num_objectives=2):
    """Plot fitness evolution across generations."""
    generations = []
    all_fitness_data = []
    best_fitness = []
    avg_fitness = []
    std_fitness = []
    
    for checkpoint_file in checkpoint_files:
        gen = extract_generation(checkpoint_file)
        checkpoint_data = load_checkpoint(checkpoint_file, num_objectives, fitness_weights, results_dir)
        global_data = checkpoint_data.get('GLOBAL_DATA', {})
        fitness_data, _ = extract_fitness_values(global_data)
        
        if len(fitness_data) == 0:
            continue
        
        generations.append(gen)
        all_fitness_data.append(fitness_data)
        
        # Calculate statistics (for single objective or first objective)
        if len(fitness_data[0]) == 1:
            fitness_values = [f[0] for f in fitness_data]
            best_fitness.append(max(fitness_values) if fitness_weights and fitness_weights[0] > 0 
                               else min(fitness_values))
            avg_fitness.append(np.mean(fitness_values))
            std_fitness.append(np.std(fitness_values))
        else:
            # For multi-objective, use first objective
            fitness_values = [f[0] for f in fitness_data]
            best_fitness.append(max(fitness_values) if fitness_weights and fitness_weights[0] > 0 
                               else min(fitness_values))
            avg_fitness.append(np.mean(fitness_values))
            std_fitness.append(np.std(fitness_values))
    
    if len(generations) == 0:
        print("No valid data found for evolution plot")
        return
    
    # Plot evolution
    num_objectives = len(all_fitness_data[0][0]) if all_fitness_data else 1
    
    if num_objectives == 1:
        plt.figure(figsize=(12, 6))
        plt.plot(generations, best_fitness, 'o-', label='Best Fitness', linewidth=2, markersize=8)
        plt.plot(generations, avg_fitness, 's-', label='Average Fitness', linewidth=2, markersize=6)
        plt.fill_between(generations, 
                         [a - s for a, s in zip(avg_fitness, std_fitness)],
                         [a + s for a, s in zip(avg_fitness, std_fitness)],
                         alpha=0.3, label='±1 Std Dev')
        plt.xlabel('Generation')
        plt.ylabel('Fitness Value')
        plt.title('Fitness Evolution Over Generations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fitness_evolution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        # Plot each objective separately
        fig, axes = plt.subplots(1, num_objectives, figsize=(6*num_objectives, 6))
        if num_objectives == 1:
            axes = [axes]
        
        for obj_idx in range(num_objectives):
            obj_best = []
            obj_avg = []
            obj_std = []
            
            for fitness_data in all_fitness_data:
                obj_values = [f[obj_idx] for f in fitness_data]
                obj_best.append(max(obj_values) if fitness_weights and fitness_weights[obj_idx] > 0 
                              else min(obj_values))
                obj_avg.append(np.mean(obj_values))
                obj_std.append(np.std(obj_values))
            
            axes[obj_idx].plot(generations, obj_best, 'o-', label='Best', linewidth=2, markersize=8)
            axes[obj_idx].plot(generations, obj_avg, 's-', label='Average', linewidth=2, markersize=6)
            axes[obj_idx].fill_between(generations,
                                      [a - s for a, s in zip(obj_avg, obj_std)],
                                      [a + s for a, s in zip(obj_avg, obj_std)],
                                      alpha=0.3, label='±1 Std Dev')
            axes[obj_idx].set_xlabel('Generation')
            axes[obj_idx].set_ylabel(f'Objective {obj_idx+1}')
            axes[obj_idx].set_title(f'Objective {obj_idx+1} Evolution')
            axes[obj_idx].legend()
            axes[obj_idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fitness_evolution.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_population_stats(checkpoint_files, output_dir, num_objectives=None, fitness_weights=None, results_dir=None):
    """Plot population size and success rate over generations."""
    generations = []
    population_sizes = []
    success_rates = []
    
    for checkpoint_file in checkpoint_files:
        gen = extract_generation(checkpoint_file)
        checkpoint_data = load_checkpoint(checkpoint_file, num_objectives, fitness_weights, results_dir)
        global_data = checkpoint_data.get('GLOBAL_DATA', {})
        population = checkpoint_data.get('population', [])
        
        generations.append(gen)
        population_sizes.append(len(population))
        
        # Calculate success rate (individuals with valid fitness)
        valid_count = 0
        for gene_id in [ind[0] for ind in population]:
            if gene_id in global_data:
                fitness = global_data[gene_id].get('fitness')
                if fitness is not None:
                    if isinstance(fitness, (tuple, list)):
                        if all(math.isfinite(f) for f in fitness):
                            if not all(abs(f) > 1e10 for f in fitness):
                                valid_count += 1
                    elif math.isfinite(fitness):
                        valid_count += 1
        
        success_rate = valid_count / len(population) if len(population) > 0 else 0
        success_rates.append(success_rate)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(generations, population_sizes, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Population Size')
    ax1.set_title('Population Size Over Generations')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(generations, success_rates, 's-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Success Rate Over Generations')
    ax2.set_ylim([0, 1.1])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'population_stats.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_ancestry_tree(checkpoint_data, output_path, max_depth=5, max_nodes=50):
    """Plot ancestry tree for a given generation."""
    ancestry = checkpoint_data.get('GLOBAL_DATA_ANCESTRY', {})
    global_data = checkpoint_data.get('GLOBAL_DATA', {})
    
    if len(ancestry) == 0:
        print("No ancestry data found")
        return
    
    # Build tree structure
    def get_ancestry_path(gene_id, depth=0):
        if depth > max_depth or gene_id not in ancestry:
            return []
        path = ancestry[gene_id].get('GENES', [])
        return path[:max_depth+1]
    
    # Get all valid genes with fitness
    valid_genes = []
    for gene_id, data in global_data.items():
        if 'fitness' in data and data['fitness'] is not None:
            fitness = data['fitness']
            if isinstance(fitness, (tuple, list)):
                if all(math.isfinite(f) for f in fitness):
                    if not all(abs(f) > 1e10 for f in fitness):
                        valid_genes.append(gene_id)
            elif math.isfinite(fitness):
                valid_genes.append(gene_id)
    
    if len(valid_genes) == 0:
        print("No valid genes found for ancestry tree")
        return
    
    # Limit number of nodes for visualization
    if len(valid_genes) > max_nodes:
        # Select top performers
        fitness_scores = []
        for gene_id in valid_genes:
            fitness = global_data[gene_id]['fitness']
            if isinstance(fitness, (tuple, list)):
                score = sum(fitness)  # Simple sum for ranking
            else:
                score = fitness
            fitness_scores.append((gene_id, score))
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        valid_genes = [g[0] for g in fitness_scores[:max_nodes]]
    
    # Create a simple text-based tree (can be enhanced with graphviz)
    print(f"\nAncestry Tree (showing top {len(valid_genes)} genes):")
    print("=" * 80)
    for gene_id in valid_genes[:20]:  # Show first 20
        if gene_id in ancestry:
            genes = ancestry[gene_id].get('GENES', [])
            mutations = ancestry[gene_id].get('MUTATE_TYPE', [])
            fitness = global_data[gene_id].get('fitness', 'N/A')
            print(f"\nGene: {gene_id[:20]}...")
            print(f"  Fitness: {fitness}")
            print(f"  Ancestry: {' -> '.join([g[:10] + '...' if len(g) > 10 else g for g in genes[-5:]])}")
            print(f"  Operations: {' -> '.join(mutations[-5:])}")

def main():
    parser = argparse.ArgumentParser(description='Visualize LLM-GE results')
    parser.add_argument('checkpoint_dir', type=str, help='Directory containing checkpoint files')
    parser.add_argument('--output_dir', type=str, default='visualizations', 
                       help='Output directory for visualizations (default: visualizations)')
    parser.add_argument('--generations', type=str, default=None,
                       help='Comma-separated list of generations to plot (e.g., 0,5,10). If not specified, plots all.')
    parser.add_argument('--fitness_weights', type=str, default=None,
                       help='Comma-separated fitness weights (e.g., -1.0,-1.0) for determining min/max')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all checkpoints
    checkpoint_files = get_all_checkpoints(args.checkpoint_dir)
    
    if len(checkpoint_files) == 0:
        print(f"Error: No checkpoint files found in {args.checkpoint_dir}")
        sys.exit(1)
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    # Parse fitness weights if provided
    fitness_weights = None
    if args.fitness_weights:
        fitness_weights = tuple([float(w.strip()) for w in args.fitness_weights.split(',')])
    
    # Parse generations to plot
    generations_to_plot = None
    if args.generations:
        generations_to_plot = [int(g.strip()) for g in args.generations.split(',')]
    
    # Initialize num_objectives
    num_objectives = None
    
    # Try to find results directory as fallback
    results_dir = find_results_directory(args.checkpoint_dir)
    if results_dir:
        print(f"Found results directory: {results_dir}")
        # Try to determine num_objectives from results files
        sample_files = glob.glob(os.path.join(results_dir, '*_results.txt'))[:5]
        if sample_files:
            try:
                with open(sample_files[0], 'r') as f:
                    content = f.read().strip()
                    num_values = len(content.split(','))
                    num_objectives = num_values
                    print(f"Detected {num_objectives} objective(s) from results files")
            except:
                pass
    
    # Determine number of objectives from first checkpoint
    if num_objectives is None:
        if fitness_weights is not None:
            num_objectives = len(fitness_weights)
            print(f"Using {num_objectives} objective(s) from fitness_weights argument")
        else:
            # Try to determine from first checkpoint by peeking
            print("Determining number of objectives from checkpoint data...")
            num_objectives = peek_checkpoint_structure(checkpoint_files[0])
            if num_objectives is None:
                # Fallback: use default
                print("Using default: 2 objectives")
                num_objectives = 2
            else:
                print(f"Detected {num_objectives} objective(s) from checkpoint data")
    
    # Plot evolution
    print("Generating evolution plot...")
    plot_evolution(checkpoint_files, args.output_dir, fitness_weights, results_dir, num_objectives)
    
    # Plot population statistics
    print("Generating population statistics...")
    plot_population_stats(checkpoint_files, args.output_dir, num_objectives, fitness_weights, results_dir)
    
    # Plot Pareto fronts for specified generations
    print("Generating Pareto front plots...")
    for checkpoint_file in checkpoint_files:
        gen = extract_generation(checkpoint_file)
        if generations_to_plot is None or gen in generations_to_plot:
            checkpoint_data = load_checkpoint(checkpoint_file, num_objectives, fitness_weights, results_dir)
            output_path = os.path.join(args.output_dir, f'pareto_gen_{gen}.png')
            plot_pareto_front(checkpoint_data, gen, output_path, fitness_weights, num_objectives)
            print(f"  Generated plot for generation {gen}")
    
    # Plot ancestry tree for the last generation
    print("Generating ancestry information...")
    if len(checkpoint_files) > 0:
        last_checkpoint = checkpoint_files[-1]
        checkpoint_data = load_checkpoint(last_checkpoint, num_objectives, fitness_weights, results_dir)
        gen = extract_generation(last_checkpoint)
        plot_ancestry_tree(checkpoint_data, os.path.join(args.output_dir, f'ancestry_gen_{gen}.txt'))
    
    print(f"\nVisualizations saved to: {args.output_dir}")
    print("Generated files:")
    for file in os.listdir(args.output_dir):
        if file.endswith('.png'):
            print(f"  - {file}")

if __name__ == '__main__':
    main()

