#!/bin/bash
# Slurm job for Northeastern's Explorer cluster (adapt partition/GRES elsewhere).
#
#   sbatch bench/explorer.sbatch
#
#SBATCH --job-name=specsim-bench
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.log

set -euo pipefail
WORK=${SPECSIM_WORK:-/scratch/$USER/specsim}
cd "$WORK"
module load python/3.13.5
source env/bin/activate
export HF_HOME=${HF_HOME:-/scratch/$USER/hf}
export HF_HUB_OFFLINE=1
mkdir -p results

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python bench/spec_bench.py \
  --draft Qwen/Qwen2.5-0.5B-Instruct \
  --target Qwen/Qwen2.5-7B-Instruct \
  --out results/qwen2.5-0.5b_7b_${SLURM_JOB_ID}.json \
  --hf-reference
