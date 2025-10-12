import argparse
import importlib
import os
import sys
from pathlib import Path as p
from os.path import join as pj

import numpy as np
import random
import logging

from hftbacktest import BacktestAsset, ROIVectorMarketDepthBacktest, Recorder
from hftbacktest.stats import LinearAssetRecord

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS - Configure your backtest parameters here
# ============================================================================

# Data files (7 days of Bitcoin data)
DATA_FILES = [
    'data/btcusdt_20200201.npz',
]

# Initial snapshot (optional but recommended for continuity)
# Set to None if you don't have one or your data files are complete
INITIAL_SNAPSHOT = None # 'data/btcusdt_20240808_eod.npz'

# Trading parameters
TICK_SIZE = 0.1          # Minimum price increment for Bitcoin
LOT_SIZE = 0.001         # Minimum trading quantity (0.001 BTC)
MAKER_FEE = -0.00005     # Maker fee (-0.005% rebate on Binance)
TAKER_FEE = 0.0007       # Taker fee (0.07% on Binance)
INITIAL_BALANCE = 10000.0  # Starting capital in USD

# Fitness calculation weights
ROI_WEIGHT = 0.4         # Weight for ROI component (40%)
SHARPE_WEIGHT = 0.6      # Weight for Sharpe ratio component (60%)

# ============================================================================


def create_save_dir(save_root):
    """Create an experiment directory with auto-incrementing number."""
    if not p(save_root).exists():
        p(save_root).mkdir(exist_ok=True, parents=True)    

    n = []
    for exp_dir in p(save_root).iterdir():
        if exp_dir.is_dir():
            exp_name = exp_dir.name
            i = -1
            while exp_name[i].isdigit():
                i -= 1
            i += 1
            if i != 0:
                n.append(int(exp_name[i:]))
    
    if len(n) == 0:
        save_dir = pj(save_root, "exp1")
    else:
        save_dir = f"{save_root}/exp{sorted(n)[-1] + 1}"
    
    p(save_dir).mkdir(exist_ok=True, parents=True)
    return save_dir


def calculate_fitness(stats, roi_weight=ROI_WEIGHT, sharpe_weight=SHARPE_WEIGHT):
    """
    Calculate fitness score combining ROI and Sharpe ratio.
    
    Args:
        stats: Stats object from LinearAssetRecord
        roi_weight: Weight for ROI component
        sharpe_weight: Weight for Sharpe ratio component
    
    Returns:
        fitness: Combined fitness score
        metrics: Dictionary of individual metrics
    """
    # Extract statistics from the Polars DataFrame
    # stats.entire is a Polars DataFrame with columns: 
    # start, end, SR (Sharpe), Sortino, Return, MaxDrawdown, Daily, 
    # NumberOfTrades, DailyTurnover, ReturnOverMDD, ReturnOverTrade, MaxPositionValue
    entire_stats = stats.summary()
    print(type(entire_stats))
    # # Get the first row (should only be one row for entire period stats)
    row = entire_stats.row(0, named=True)
    print(row)
    # Extract metrics
    # Return is already in percentage form from the stats

    roi = row['Return']  # This is the total return in percentage
    sharpe = row['SR']  # Sharpe Ratio
    max_drawdown = abs(row['MaxDrawdown'])  # Make positive for display
    sortino = row['Sortino']
    total_trades = row['DailyNumberOfTrades']
    return_over_mdd = row['ReturnOverMDD']
    return_over_trade = row['ReturnOverTrade']
    max_position_value = row['MaxPositionValue']
    
    # Normalize metrics for combining
    # ROI: typically ranges from -100% to +100%, normalize to 0-1 scale
    # We use a sigmoid-like transformation: tanh((roi/100) * scale_factor)
    roi_normalized = np.tanh(roi / 50.0)  # Scale so ±50% ROI → ±0.76
    
    # Sharpe: typically ranges from -3 to +3 for good strategies
    # Normalize similarly
    sharpe_normalized = np.tanh(sharpe / 2.0)  # Scale so ±2 Sharpe → ±0.76
    
    # Ensure non-negative (shift to 0-1 range)
    roi_normalized = (roi_normalized + 1) / 2
    sharpe_normalized = (sharpe_normalized + 1) / 2
    
    # Calculate weighted fitness
    fitness = roi_weight * roi_normalized + sharpe_weight * sharpe_normalized
    
    # Calculate equity values
    initial_equity = INITIAL_BALANCE
    final_equity = initial_equity * (1 + roi / 100.0)
    
    metrics = {
        'fitness': fitness,
        'roi': roi,
        'sharpe': sharpe,
        'sortino': sortino,
        'roi_normalized': roi_normalized,
        'sharpe_normalized': sharpe_normalized,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'return_over_mdd': return_over_mdd,
        'return_over_trade': return_over_trade,
        'max_position_value': max_position_value,
        'final_equity': final_equity,
        'initial_equity': initial_equity
    }
    
    return fitness, metrics


def get_args():
    parser = argparse.ArgumentParser(description='HFT Backtest Evaluation')
    parser.add_argument('--model', type=str, default="seeds.seedGLTP", 
                        help="Model module name (without .py)")
    parser.add_argument('--save_dir', type=str, default="trained", 
                        help="Directory where results will be saved")
    parser.add_argument('--random_seed', type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument('--variant_dir', type=str, default='models', 
                        help="Directory containing model variants")
    
    return parser.parse_args()


def main():
    logger.info("Starting HFT backtest evaluation")
    script_directory = p(__file__).parent.resolve()
    os.chdir(script_directory)
    
    args = get_args()
    
    # Set random seeds
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Import model dynamically
    sys.path.append(args.variant_dir)
    model_module = importlib.import_module(args.model)
    
    # Extract gene/model ID
    try:
        gene_id = args.model.split('model_')[1]
    except:
        gene_id = 'seed'
    
    save_dir = create_save_dir(f'{args.save_dir}/{gene_id}')
    logger.info(f"Results will be saved to: {save_dir}")
    
    # Validate data files exist
    logger.info("Validating data files...")
    for data_file in DATA_FILES:
        if not p(data_file).exists():
            logger.error(f"Data file not found: {data_file}")
            raise FileNotFoundError(f"Data file not found: {data_file}")
    
    if INITIAL_SNAPSHOT and not p(INITIAL_SNAPSHOT).exists():
        logger.warning(f"Initial snapshot not found: {INITIAL_SNAPSHOT}")
        logger.warning("Proceeding without initial snapshot...")
        initial_snapshot = None
    else:
        initial_snapshot = INITIAL_SNAPSHOT
    
    # Configure backtest asset
    logger.info("Configuring backtest asset...")
    asset_config = (
        BacktestAsset()
        .data(DATA_FILES)
        .linear_asset(1.0)  # 1x leverage for spot or 1x futures
        .constant_latency(10_000_000, 10_000_000)  # 10ms constant latency (feed, order)
        .risk_adverse_queue_model()  # Conservative queue position model
        .no_partial_fill_exchange()  # No partial fills
        .trading_value_fee_model(MAKER_FEE, TAKER_FEE)
        .tick_size(TICK_SIZE)
        .lot_size(LOT_SIZE)
        .last_trades_capacity(1000) 
        .roi_lb(0.0)  # Lower bound of price range of interest
        .roi_ub(100000.0)  # Upper bound (adjust for BTC price range)
    )
    
    # Add initial snapshot if available
    if initial_snapshot:
        asset_config = asset_config.initial_snapshot(initial_snapshot)
        logger.info(f"Using initial snapshot: {initial_snapshot}")
    else:
        logger.info("No initial snapshot provided - building order book from data files")
    
    asset = asset_config
    
    logger.info("Creating backtest instance...")
    hbt = ROIVectorMarketDepthBacktest([asset])
    # hbt.set_equity(0, INITIAL_BALANCE)
    
    # Create recorder with sufficient buffer
    # Buffer size should accommodate: recording_frequency * duration / interval
    # For 7 days at 100ms intervals: ~6 million records
    recorder = Recorder(1, 5_000_000)
    
    
    logger.info("Initializing model...")
    model = model_module.Model()
    
    logger.info("Running backtest...")
    logger.info(f"  - Data files: {len(DATA_FILES)} files")
    logger.info(f"  - Tick size: {TICK_SIZE}")
    logger.info(f"  - Lot size: {LOT_SIZE}")
    logger.info(f"  - Maker fee: {MAKER_FEE * 100:.3f}%")
    logger.info(f"  - Taker fee: {TAKER_FEE * 100:.3f}%")
    logger.info(f"  - Initial balance: ${INITIAL_BALANCE:,.2f}")
    
    # Run the strategy
    model.run(hbt, recorder.recorder)
    
    # Close backtest
    logger.info("Backtest complete, processing results...")
    hbt.close()

    logger.info("Congrats your model has passed the sanity check!")
    # sys.exit(0)


    # Get recorded data and calculate statistics
    recorded_data = recorder.get(0)  # Get data for asset 0
    
    # Create stats object
    # book_size is the position size used for calculating various metrics
    stats = LinearAssetRecord(recorded_data).stats(book_size=INITIAL_BALANCE)
    stats1 = LinearAssetRecord(recorder.get(0)).stats(book_size=INITIAL_BALANCE)
    # sys.exit(0)
    # Calculate fitness
    fitness, metrics = calculate_fitness(stats)
    
    # Replace the logger.info section with:
    logger.info("="*80)
    logger.info("BACKTEST RESULTS")
    logger.info("="*80)
    logger.info(f"Gene ID: {gene_id}")
    logger.info(f"Fitness Score: {fitness:.6f}")
    logger.info(f"ROI: {metrics['roi']:.2f}%")
    logger.info(f"Sharpe Ratio: {metrics['sharpe']:.4f}")
    logger.info(f"Sortino Ratio: {metrics['sortino']:.4f}")
    logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    logger.info(f"Total Trades: {metrics['total_trades']}")
    logger.info(f"Return/MDD: {metrics['return_over_mdd']:.4f}")
    logger.info(f"Return/Trade: {metrics['return_over_trade']:.4f}")
    logger.info(f"Max Position Value: ${metrics['max_position_value']:.2f}")
    logger.info(f"Final Equity: ${metrics['final_equity']:.2f}")
    logger.info("="*80)
    
    # Save detailed statistics
    stats_filename = pj(save_dir, f'{gene_id}_stats.txt')
    with open(stats_filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"ROI: {metrics['roi']:.2f}%\n")
        f.write(f"Sharpe Ratio: {metrics['sharpe']:.4f}\n")
        f.write(f"Sortino Ratio: {metrics['sortino']:.4f}\n")
        f.write(f"Max Drawdown: {metrics['max_drawdown']:.2f}%\n")
        f.write(f"Return/MaxDrawdown: {metrics['return_over_mdd']:.4f}\n")
        f.write(f"Return/Trade: {metrics['return_over_trade']:.4f}\n")
        f.write(f"Total Trades: {metrics['total_trades']}\n")
        f.write(f"Max Position Value: ${metrics['max_position_value']:.2f}\n")
        f.write(f"Initial Equity: ${metrics['initial_equity']:.2f}\n")
        f.write(f"Final Equity: ${metrics['final_equity']:.2f}\n")
        f.write("="*80 + "\n")
    
    logger.info(f"Detailed statistics saved to: {stats_filename}")
    
    # Save fitness score in simplified format (for genetic algorithm)
    fitness_filename = pj(save_dir, f'{gene_id}_fitness.txt')
    with open(fitness_filename, 'w') as f:
        f.write(f"{fitness:.8f}")
    
    logger.info(f"Fitness score saved to: {fitness_filename}")
    logger.info("="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)


    results_text = f"{fitness:.8f}"

    """
    This line formats the results into comma-separated string
    fp_count: Number of false positives.
    fn_count: Number of false negatives.
    """

    # Define Output Filename
    """
    Defines an Output Filename
    gene_id is a unique identifier for the experiment or model.
    """
    filename = os.path.abspath(f'results/{gene_id}_results.txt')
    
    dir_path = os.path.dirname(filename)

    # Create the directory, ignore error if it already exists
    os.makedirs(dir_path, exist_ok=True)

    """
    os.path.dirname(filename) extracts the directory path from the filename.
    os.makedirs(dir_path, exist_ok=True) creates the directory if it does not already exist. 
    The exist_ok=True parameter ensures that no error is raised if the directory already exists.
    """

    # Open the file in write mode and write the text
    with open(filename, 'w') as file:
        file.write(results_text)
    """
    This block opens the specified file in write mode ('w').
    If the file does not exist, it will be created.
    The results_text string is written to the file.
    """

    print(f"results have been written to {filename}")

    print('='*120);print('job done');print('='*120)

    
    return fitness, metrics


if __name__ == '__main__':
    try:
        fitness, metrics = main()
        sys.exit(0)  # Success
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)  # Failure