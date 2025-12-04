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
models/
    model_12345.py
    model_67890.py
```

Run them with:

```bash
python eval.py --model model_12345 --variant_dir models
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

1. Seed models in `seeds/` are used as **base strategies**.  
2. The evolutionary system mutates & crosses them to create new variants in `models/`.  
3. `eval.py` computes a **fitness score** combining:
   - ROI  
   - Sharpe Ratio  
4. The GA selects the best-performing variants for the next generation.

This README provides the required documentation for anyone to:
- Run seeds  
- Test evolved models  
- Add new strategies  
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

---

## Maintainers

Document maintained by  
**EMADE VIP – Crypto Team**  
Georgia Tech, 2025

---

