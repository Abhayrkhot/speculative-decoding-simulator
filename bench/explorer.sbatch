#!/bin/bash
# Slurm job for Northeastern's Explorer cluster (adapt partition/GRES elsewhere).
#
#   sbatch bench/explorer.sbatch
#   TARGET=Qwen/Qwen2.5-32B-Instruct TAG=0.5b_32b GAMMAS=1,2,3,4,6,8 sbatch bench/explorer.sbatch
#
# BENCH_FLAGS adds options, e.g. BENCH_FLAGS="--cuda-graphs".
# VENV selects the virtualenv (torch must be built for a CUDA version the
# node's driver supports; Explorer's driver 545 needs cu12x wheels).
#
#SBATCH --job-name=specsim-bench
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:55:00
#SBATCH --output=%x-%j.log

set -euo pipefail
WORK=${SPECSIM_WORK:-/scratch/$USER/specsim}
DRAFT=${DRAFT:-Qwen/Qwen2.5-0.5B-Instruct}
TARGET=${TARGET:-Qwen/Qwen2.5-7B-Instruct}
TAG=${TAG:-0.5b_7b}
GAMMAS=${GAMMAS:-1,2,3,4,5,6,8}
BENCH_FLAGS=${BENCH_FLAGS:---hf-reference}

cd "$WORK"
module load python/3.13.5
source "${VENV:-env}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch/$USER/hf}
export HF_HUB_OFFLINE=1
mkdir -p results

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# shellcheck disable=SC2086  # BENCH_FLAGS is intentionally word-split
python bench/spec_bench.py \
  --draft "$DRAFT" \
  --target "$TARGET" \
  --out "results/qwen2.5-${TAG}_${SLURM_JOB_ID}.json" \
  --gammas "$GAMMAS" \
  $BENCH_FLAGS
