import os
import sys
import numpy as np
import torch
import platform

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(ROOT_DIR, "sota/MarketMaking/data")
SOTA_ROOT = os.path.join(ROOT_DIR, 'sota/MarketMaking')
SEED_NETWORK = os.path.join(SOTA_ROOT, 'seeds/seedOBI.py')
MODEL = "model"

# Path to local LLM model path used by server.py for LLM operations
MODEL_PATH = "/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.3-70B-Instruct/"
VARIANT_DIR = os.path.join(SOTA_ROOT, "models/llmge_models") 
TRAIN_FILE = os.path.join(SOTA_ROOT, "eval.py") 

# TODO: Adding this here, I think it's supposed to parse from the command line
OUTPUT_DIR = "marketMaking_test"
PORT=8137

CLUSTER = "pace-ice"
LLM_MODEL = 'gemini' # change to llama3.3 for local development
ENVIRONMENT_DIR = os.path.join(ROOT_DIR, ".venv")
SLURM_CONFIG_DIR = os.path.join(ROOT_DIR, "slurm-config/")
LOCAL_LLM = False # change to True for local development
HOSTNAME_DIR = os.path.join(ROOT_DIR, "hostname.log")
# HOSTNAME_DIR = "/home/hice1/athakkar37/VIP/V2/crypto-market-making-llm-guided-evolution/hostname.log"
QC_CHECK_BOOL = False
HUGGING_FACE_BOOL = False
INFERENCE_SUBMISSION = True # should I be setting this to False
CUF_TIMEOUT = 20000

LOCAL = False
if LOCAL:
	RUN_COMMAND = 'bash'
	DELAYED_CHECK = False
else: 
	RUN_COMMAND = 'sbatch'
	DELAYED_CHECK = True
MACOS = platform.system() == "Darwin"
RUNLINE_AMP = ''
if torch.mps.is_available():
	DEVICE = 'mps'
	MACOS = True
	RUNLINE_AMP = "-amp"
elif torch.cuda.is_available():
	DEVICE = 'cuda'
else:
	DEVICE = 'cpu'
# ExquisiteNetV2
# RUNLINE_TMP = f"-data {DATA_PATH} -end_lr 0.001 -seed 21 -val_r 0.2 {RUNLINE_AMP} -epoch 200"
# EVAL_RUNLINE = 'python {} -bs 216 -network "models.llmge_models.{MODEL}_{}" {}
# Point-Transformers

# resolves to {MODEL}_{gene_id}
RUNLINE_TMP = '{}_{}'
EVAL_RUNLINE = "uv run python {} --model {} --variant_dir {VARIANT_DIR}"
"""
Evolution Constants/Params
"""
FITNESS_WEIGHTS = (1.0,)  # Single fitness value since eval.py combines ROI and Sharpe internally
INVALID_FITNESS_MAX = tuple([float(x*np.inf*-1) for x in FITNESS_WEIGHTS])
PLACEHOLDER_FITNESS = tuple([int(x*9999999999*-1) for x in FITNESS_WEIGHTS])
NUM_EOT_ELITES = 2
GENERATION = 0
PROB_QC = 0.0
PROB_EOT = 0.25
num_generations = 3 # Number of generations
start_population_size = 5  # Starting population size
# start_population_size = 144   # Size of the population 124=72
#population_size = 44 # with cx_prob (0.25) and mute_prob (0.7) you get about %50 successful turnover
population_size = 5 # with cx_prob (0.25) and mute_prob (0.7) you get about %50 successful turnover
crossover_probability = 0.35  # Probability of mating two individuals
mutation_probability = 0.8 # Probability of mutating an individual
num_elites = 2
hof_size = 4

DNA_TXT = """
⠀⠀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⣿⡇⠀⠀⠀⠀⠀⠀⠀⢀⣠⣤⣶⣶⠶⣶⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⣀⣹⣟⣛⣛⣻⣿⣿⣿⡾⠟⢉⣴⠟⢁⣴⠋⣹⣷⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠈⠛⠛⣿⠉⢉⣩⠵⠚⠁⢀⡴⠛⠁⣠⠞⠁⣰⠏⠸⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⢻⣷⠋⠁⠀⢀⡴⠋⠀⢀⡴⠋⠀⣼⠃⠀⡼⢿⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⢻⣆⣠⡴⠋⠀⠀⣠⠟⠀⢀⡾⠁⠀⡼⠁⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠻⣯⡀⠀⢀⡼⠃⠀⢠⡟⠀⢀⡾⠁⢀⣾⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠙⠻⣶⣟⡀⠀⣰⠏⠀⢀⡾⠁⠀⣼⢹⣿⣀⣤⣤⣴⠶⢿⡿⠛⢛⣷⢶⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠻⠿⠶⠶⠾⠷⠶⠿⠛⢻⣟⠉⣥⠟⠁⣠⠟⠀⢠⠞⠁⣄⡿⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣿⠞⠁⢀⡴⠋⠀⣴⠋⠀⣰⠟⠀⣤⡾⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⡄⢠⠞⠁⢀⡾⠁⢀⡼⠃⢀⡴⠋⠀⢸⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣷⠋⠀⣰⠏⠀⣠⠟⠀⣰⠟⠁⢀⡴⠛⣿⠀⠀⣀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⣧⡼⠃⢀⡼⠋⢠⡞⠁⣠⣞⣋⣤⣶⣿⡟⠛⣿⠛⠛⣻⠟⠷⢶⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠻⣦⣾⣤⣴⣯⡶⠾⠟⠛⠉⠉⠉⣿⡇⢠⡏⠀⣰⠏⠀⢀⣼⠋⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⡾⠀⢰⠏⠀⢠⡞⠁⠀⣠⠞⢻⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣷⠇⢠⠏⠀⣰⠋⠀⣠⠞⠁⠀⢀⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡟⢠⠟⢀⡼⠁⣠⠞⠁⣀⣴⢾⣿⣤⣿⣦⣄⣀⡀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⡟⣠⠏⣠⠞⣁⣴⣾⣿⣿⣿⣿⣿⣿⡏⢹⡏⠛⠳⣦⣄⡀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⢷⣾⣷⠿⠿⠛⠉⠀⠀⠈⠳⣬⣿⡟⣾⠁⠀⣼⠃⠉⠻⠆
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿⣧⡏⠀⣼⠃⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⠁⡼⠁⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣟⡼⠁⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡿⠁⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢙⣃⠀⠀
"""
