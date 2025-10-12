#!/bin/bash
#SBATCH --job-name=llm_opt
#SBATCH -t 8:00:00
#SBATCH --mem 16G
#SBATCH -c 4
#SBATCH -N 1
echo "launching LLM Guided Evolution"
hostname
module load uv

export SERVER_HOSTNAME=$(hostname)
uv run python run_improved.py marketMaking_test
