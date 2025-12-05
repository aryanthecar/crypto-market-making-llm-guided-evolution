# Market Making – HFT Backtesting & Seed Strategy Evaluation

This directory contains the **High-Frequency Trading (HFT) backtesting pipeline** and the **seed trading strategies** used by the LLM-Guided Evolution system.  
The goal of this module is to evaluate market-making strategies (seeds or evolved variants) using a realistic, order-book-driven backtest and produce standardized fitness scores for the genetic algorithm.

---

## Folder Structure

```
MarketMaking/
│
├── eval.py                 # Main entry point for running HFT backtests
│
├── seeds/                  # Built-in seed strategies used as starting genes
│   ├── seedAS.py
│   ├── seedCJR.py
│   ├── seedGLTP.py         # Default model
│   ├── seedHOST.py
│   └── seedOBI.py
│
├── models/                 # Evolved model variants (created by LLM-GE)
│   └── llmge_models/       # Generated during evolution
│
└── dataExploration/
    └── BTCUSDT_explore.ipynb   # Notebook for analyzing BTC/USDT orderbook data
```

---

## Running an HFT Backtest

`eval.py` is the main script used to evaluate a strategy.  
It loads data, runs the strategy in an HFT simulation, and outputs both metrics and a fitness score.

### **Run the default seed (GLTP):**

```bash
python eval.py
```

---

### **Run a specific seed model:**

Each seed corresponds to a Python file in the `seeds/` folder.

Example:

```bash
python eval.py --model seeds.seedHOST
```

Other valid options:

```bash
--model seeds.seedAS
--model seeds.seedCJR
--model seeds.seedGLTP
--model seeds.seedOBI
```

---

### **Run an evolved variant (from `models/` folder):**

The genetic algorithm creates models like:

```
models/llmge_models/
    model_12345.py
    model_67890.py
```

Run them with:

```bash
python eval.py --model model_12345 --variant_dir models/llmge_models
```

---

## Backtest Configuration

Inside `eval.py`, you can configure:

- **Data files**  
- **Trading fees (maker/taker)**  
- **Tick size & lot size**  
- **Initial USD balance**  
- **Latency & exchange model**  
- **Fitness score weighting (ROI vs Sharpe)**  

Example (from eval.py):

```python
DATA_FILES = ['data/btcusdt_20200201.npz']
TICK_SIZE = 0.1
LOT_SIZE = 0.001
MAKER_FEE = -0.00005
TAKER_FEE = 0.0007
INITIAL_BALANCE = 10000.0
```

---

## Output & Saved Results

Every evaluation automatically creates:

```
trained/<gene_id>/
    <gene_id>_stats.txt      # Detailed performance metrics
    <gene_id>_fitness.txt    # Single numeric fitness value (GA uses this)

results/<gene_id>_results.txt   # Fitness value for easy loading
```

Metrics include:

- ROI  
- Sharpe & Sortino ratios  
- Max drawdown  
- # of trades  
- Return/MDD  
- Return/trade  
- Final equity  

---

## Multi-Seed Configuration for LLM-Guided Evolution

The LLM-Guided Evolution system supports using multiple seed strategies simultaneously to create a more diverse initial population.

### **Enabling Multi-Seed Mode**

Configure in `src/cfg/constants_marketMaking.py`:

```python
USE_MULTIPLE_SEEDS = True
INDIVIDUALS_PER_SEED = 3  # Create 3 individuals from each seed
```

### **How It Works**

1. **Seed Discovery**: The system automatically finds all files matching `seed*.py` in the `seeds/` directory
2. **Population Creation**: Creates `INDIVIDUALS_PER_SEED` individuals from each seed file
3. **Initial Population Size**: `(number of seed files) × INDIVIDUALS_PER_SEED`

**Example:**
- 4 seed files (seedAS, seedCJR, seedGLTP, seedOBI)
- `INDIVIDUALS_PER_SEED = 3`
- Initial population: 4 × 3 = **12 individuals**

### **Population Size Parameters**

**Important Notes:**
- `start_population_size`: **Only used when `USE_MULTIPLE_SEEDS = False`**
- `population_size`: Controls how many individuals are selected each generation (always active)
- `INDIVIDUALS_PER_SEED`: **Only used when `USE_MULTIPLE_SEEDS = True`**

**Example Configuration:**
```python
USE_MULTIPLE_SEEDS = True
INDIVIDUALS_PER_SEED = 3
population_size = 12
num_elites = 4

# With 4 seed files:
# - Initial population: 4 × 3 = 12 individuals
# - Each generation: Selects 12 individuals (population_size)
# - After adding elites: Final population = 12 + 4 = 16 individuals
```

### **Benefits of Multi-Seed Mode**

- **Diversity**: Start evolution from multiple different strategies simultaneously
- **Exploration**: Explore different regions of the solution space in parallel
- **Robustness**: Reduces dependency on a single initial seed
- **Faster Convergence**: Multiple starting points can lead to better solutions faster

### **Switching Between Modes**

- **Single seed mode**: Set `USE_MULTIPLE_SEEDS = False` and specify `start_population_size`
- **Multi-seed mode**: Set `USE_MULTIPLE_SEEDS = True` and ensure seed files exist in `seeds/` directory

---

## Adding a New Seed Model

To create a new market-making strategy:

1. Add a file inside `seeds/`, e.g.:

```
seeds/seedMyStrategy.py
```

2. Implement a `Model` class:

```python
class Model:
    def run(self, backtest, recorder):
        # Your trading logic
        ...
```

3. Run it:

```bash
python eval.py --model seeds.seedMyStrategy
```

4. **For Multi-Seed Evolution**: The new seed will automatically be included when `USE_MULTIPLE_SEEDS = True`

---

## 📓 Data Exploration

The `dataExploration/` folder contains:

```
BTCUSDT_explore.ipynb
```

Use this notebook to:

- Inspect order book depth  
- Validate data quality  
- Visualize spreads, trades, and liquidity  
- Understand market structure before designing strategies  

---

## How This Fits Into LLM-Guided Evolution

1. **Seed models** in `seeds/` are used as **base strategies** for evolution
2. **Multi-seed mode** (optional) creates diverse initial populations from multiple seeds
3. The evolutionary system **mutates & crosses** seeds to create new variants in `models/llmge_models/`
4. `eval.py` computes a **fitness score** combining:
   - ROI  
   - Sharpe Ratio  
5. The GA selects the best-performing variants for the next generation

This README provides the required documentation for anyone to:
- Run seeds  
- Test evolved models  
- Add new strategies  
- Configure multi-seed evolution
- Reproduce results  

---

## 🛠 Dependencies

Install project dependencies:

```bash
pip install -r requirements.txt
```

Some modules (e.g., `hftbacktest`) may require local installation depending on the environment.

---

## Troubleshooting

**Data file not found:**  
Make sure `.npz` files are stored in `sota/MarketMaking/data/` or update `DATA_FILES` accordingly.

**Import error for seeds or models:**  
Check that the `--model` flag and directory names match your folder structure.

**Recorder buffer overflow:**  
Increase:
```python
recorder = Recorder(1, 5_000_000)
```

**Multi-seed not working:**
- Verify `USE_MULTIPLE_SEEDS = True` in `constants_marketMaking.py`
- Ensure seed files are named `seed*.py` and located in `seeds/` directory
- Check that `INDIVIDUALS_PER_SEED` is set appropriately

---

## Maintainers

Document maintained by  
**EMADE VIP – Crypto Team**  
Georgia Tech, 2025

---

