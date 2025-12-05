# Market Making - LLM Guided Evolution

This directory contains the market making implementation using LLM Guided Evolution. This guide will walk you through setting up and running the evolution process for market making strategies.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Data Preparation](#data-preparation)
- [Running the Evolution](#running-the-evolution)
- [Understanding the Output](#understanding-the-output)
- [Troubleshooting](#troubleshooting)

---

## Overview

LLM Guided Evolution combines Large Language Models (LLMs) with genetic algorithms to evolve and optimize market making strategies. The framework uses:

- **Evolution of Thought (EoT)**: Enables LLMs to receive result-driven feedback and make informed improvements
- **Genetic Algorithms**: Maintains diversity while evolving strategies
- **Character Role Play (CRP)**: Enhances creativity and feasibility of generated ideas

---

## Prerequisites

### System Requirements

- **Python**: 3.12 or higher
- **Operating System**: Linux (for SLURM) or macOS (for local development)
- **GPU**: Recommended for faster training (CUDA-compatible or Apple Silicon)
- **Memory**: At least 16GB RAM recommended

### Required Accounts

- **Google Gemini API Key** (if using Gemini LLM): [Get API Key](https://ai.google.dev/gemini-api/docs/api-key)
- **Hugging Face Account** (if using remote inference): [Sign up](https://huggingface.co/join)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/aryanthecar/crypto-market-making-llm-guided-evolution.git
cd crypto-market-making-llm-guided-evolution
git checkout hft
```

### 2. Install Dependencies

Dependencies are managed through `pyproject.toml`. Install using one of the following methods:

**Using pip:**
```bash
pip install .
```

**Using uv (recommended for faster installation):**
```bash
uv pip install -e .
```

**Using conda (for isolated environments):**
```bash
conda create -n llm_guided_env python=3.12
conda activate llm_guided_env
pip install .
```

### 3. Verify Installation

```bash
python -c "import torch; import deap; print('Installation successful!')"
```

---

## Configuration

### 1. Set Up Environment Variables

Create a `.env` file in the root directory or export environment variables:

```bash
# For Gemini API
export GEMINI_API_KEY="your-api-key-here"

# Optional: For Hugging Face
export HF_TOKEN="your-huggingface-token-here"
```

### 2. Configure Constants

Edit `src/cfg/constants.py` to customize the evolution parameters:

#### Key Configuration Parameters

```python
# Root directory - Update this to your project path
ROOT_DIR = "/path/to/crypto-market-making-llm-guided-evolution"

# Data path - Location of your market data
DATA_PATH = "./market_data"

# SOTA root - Location of seed model
SOTA_ROOT = os.path.join(ROOT_DIR, 'sota/marketmaking')

# Seed network - Initial model architecture
SEED_NETWORK = os.path.join(SOTA_ROOT, "network.py")

# Execution mode
LOCAL = True  # Set to False for SLURM cluster execution
MACOS = False  # Set to True if running on macOS

# Device selection (auto-detected)
# Options: 'cuda', 'mps' (Apple Silicon), 'cpu'
DEVICE = 'cuda'  # or 'mps' or 'cpu'

# LLM Model selection
LLM_MODEL = 'gemini'  # Options: 'gemini', 'mixtral', 'llama3'
```

#### Evolution Parameters

```python
# Fitness weights (1.0 = maximize, -1.0 = minimize)
FITNESS_WEIGHTS = (1.0, -1.0)  # Example: (profit, risk)

# Number of elite individuals for EoT
NUM_EOT_ELITES = 10

# Evolution probabilities
PROB_EOT = 0.25  # Probability of Evolution of Thought operation
PROB_QC = 0.0    # Probability of quality control checks

# Population settings
start_population_size = 32  # Initial population size
population_size = 8         # Population size per generation
num_generations = 30        # Total number of generations

# Genetic algorithm parameters
crossover_probability = 0.35  # Probability of crossover
mutation_probability = 0.8    # Probability of mutation
num_elites = 44              # Number of elite individuals
hof_size = 100               # Hall of Fame size
```

#### SLURM Configuration (if using cluster)

```python
# GPU requirements for SLURM
LLM_GPU = 'A100-40GB|A100-80GB|H100|V100-16GB|V100-32GB|RTX6000|A40|L40S'

# Job submission settings
QC_CHECK_BOOL = False
INFERENCE_SUBMISSION = True
```

### 3. Prepare Seed Network

Create your initial market making model in `sota/marketmaking/network.py`. This serves as the seed architecture that will be evolved.

Example structure:
```python
import torch
import torch.nn as nn

class MarketMakingNetwork(nn.Module):
    def __init__(self, input_features, hidden_size, output_size):
        super().__init__()
        # Define your initial architecture here
        self.fc1 = nn.Linear(input_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Define forward pass
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

---

## Data Preparation

### 1. Market Data Structure

Organize your market data in the following structure:

```
market_data/
├── train/
│   ├── features/
│   └── labels/
├── val/
│   ├── features/
│   └── labels/
└── test/
    ├── features/
    └── labels/
```

### 2. Data Format

Ensure your data is in a format compatible with PyTorch DataLoader:
- Features: Numerical arrays (numpy or torch tensors)
- Labels: Target values for training

### 3. Update Data Path

Update `DATA_PATH` in `src/cfg/constants.py` to point to your data directory.

---

## Running the Evolution

### Local Execution

#### 1. Basic Run

```bash
python run_improved.py experiment_name
```

This will:
- Create an initial population from the seed network
- Run evolution for the specified number of generations
- Save checkpoints and results in the `experiment_name` directory

#### 2. Resume from Checkpoint

```bash
python run_improved.py experiment_name
```

If the directory exists, the script will automatically resume from the last checkpoint.

#### 3. Using the Shell Script (Local)

```bash
bash run.sh
```

Note: Modify `run.sh` to remove SLURM directives if running locally.

### SLURM Cluster Execution

#### 1. Submit Job

```bash
sbatch run.sh
```

#### 2. Monitor Job

```bash
# Check job status
squeue -u $USER

# View output
tail -f slurm-<job_id>.out
```

#### 3. Cancel Job (if needed)

```bash
scancel <job_id>
```

### Execution Modes

The framework supports two execution modes:

1. **Local Mode** (`LOCAL = True`):
   - Runs all operations on the local machine
   - Suitable for development and small experiments
   - Uses `bash` for job execution

2. **SLURM Mode** (`LOCAL = False`):
   - Distributes jobs across a SLURM cluster
   - Suitable for large-scale evolution
   - Uses `sbatch` for job submission

---

## Understanding the Output

### Directory Structure

After running, you'll find the following structure:

```
experiment_name/
├── 0/                    # Generation 0
│   ├── <gene_id>.sh      # Mutation scripts
│   └── ...
├── 1/                    # Generation 1
│   └── ...
├── checkpoint.pkl        # Evolution checkpoint
├── hall_of_fame.pkl      # Best individuals
└── results/              # Evaluation results
    ├── fitness_scores.csv
    └── model_performances.json
```

### Key Files

- **`checkpoint.pkl`**: Contains the entire evolution state (population, fitness, ancestry)
- **`hall_of_fame.pkl`**: Stores the best-performing individuals across all generations
- **`network_<gene_id>.py`**: Evolved model architectures in `sota/marketmaking/models/`

### Monitoring Progress

The script prints progress information including:
- Population statistics
- Fitness scores
- Generation progress
- LLM operation status

Example output:
```
═══════════════════════════════════════════════════════════════
                    STARTING GENERATION: 0
═══════════════════════════════════════════════════════════════
Population: 8 individuals
Best fitness: (0.85, -0.12)
Average fitness: (0.72, -0.15)
```

---

## Advanced Usage

### Custom Templates

Create custom mutation templates in `templates/`:

1. **Fixed Prompts**: `templates/FixedPrompts/`
2. **Evolution of Thought**: `templates/EoT/`
3. **Crossover**: `templates/CrossOver/`

### Quality Control

Enable quality control checks:

```python
QC_CHECK_BOOL = True
PROB_QC = 0.1  # 10% of mutations will be quality-checked
```

### Custom Fitness Function

Modify the fitness evaluation in the training script to match your market making objectives (e.g., Sharpe ratio, profit factor, maximum drawdown).

### Multi-Objective Optimization

The framework supports multi-objective optimization. Adjust `FITNESS_WEIGHTS` to optimize multiple objectives simultaneously:

```python
FITNESS_WEIGHTS = (1.0, -1.0, 1.0)  # Maximize profit, minimize risk, maximize Sharpe
```

---

## Troubleshooting

### Common Issues

#### 1. API Key Not Found

**Error**: `GEMINI_API_KEY not found`

**Solution**:
```bash
export GEMINI_API_KEY="your-api-key"
# Or add to .env file
```

#### 2. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
- Reduce batch size in training script
- Use smaller population size
- Enable gradient checkpointing
- Use CPU mode: `DEVICE = 'cpu'`

#### 3. Module Not Found

**Error**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Ensure you're in the project root
cd crypto-market-making-llm-guided-evolution
# Install in development mode
pip install -e .
```

#### 4. SLURM Job Fails

**Error**: Job submission fails or jobs hang

**Solution**:
- Check SLURM configuration in `constants.py`
- Verify GPU availability: `sinfo -o "%G"`
- Check job logs: `cat slurm-<job_id>.out`
- Ensure conda environment is accessible on compute nodes

#### 5. LLM Response Errors

**Error**: Invalid code generated by LLM

**Solution**:
- Enable quality control: `QC_CHECK_BOOL = True`
- Adjust temperature: Lower values (0.1-0.3) for more conservative mutations
- Check LLM API quotas and limits
- Review generated code in `sota/marketmaking/models/`

### Debug Mode

Enable verbose logging:

```python
# In constants.py or run_improved.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Getting Help

- Check existing issues: [GitHub Issues](https://github.com/aryanthecar/crypto-market-making-llm-guided-evolution/issues)
- Review the main README: `README.md`
- Consult the paper: `assets/paper/LLM_Guided_Evolution___The_Automation_of_Models_Advancing_Models.pdf`

---

## Best Practices

1. **Start Small**: Begin with a small population (8-16) and few generations (5-10) to test your setup
2. **Monitor Resources**: Keep an eye on GPU memory and API usage
3. **Save Checkpoints**: The framework auto-saves, but ensure sufficient disk space
4. **Version Control**: Commit your seed network and configuration before running evolution
5. **Experiment Tracking**: Document your hyperparameters and results for reproducibility
6. **Incremental Evolution**: Start with a well-performing seed model for better results

---

## Example Workflow

```bash
# 1. Setup
git checkout hft
pip install .

# 2. Configure
export GEMINI_API_KEY="your-key"
# Edit src/cfg/constants.py

# 3. Prepare data
# Organize market data in DATA_PATH

# 4. Create seed network
# Create sota/marketmaking/network.py

# 5. Run evolution
python run_improved.py market_making_experiment_1

# 6. Monitor progress
tail -f market_making_experiment_1/0/*.out

# 7. Analyze results
# Check checkpoint.pkl and hall_of_fame.pkl
```

---

## References

- **Main Paper**: [LLM Guided Evolution - The Automation of Models Advancing Models](./assets/paper/LLM_Guided_Evolution___The_Automation_of_Models_Advancing_Models.pdf)
- **ExquisiteNetV2**: Reference implementation in `sota/ExquisiteNetV2/`
- **DeepMind AlphaEvolve**: This framework inspired DeepMind's AlphaEvolve (see main README)

---

## License

See the main repository LICENSE file for details.

---

## Contributing

Contributions are welcome! Please follow the contribution guidelines in the main repository.

---

**Last Updated**: 2025-01-16
