#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=9G
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:8.6|GPU_CC:8.9|GPU_CC:9.0"
#SBATCH --time=00:30:00
#SBATCH --output=/storage/slurm/zyuef/logs/bgs-memprobe-%j.out
#SBATCH --error=/storage/slurm/zyuef/logs/bgs-memprobe-%j.err
set -euo pipefail
: "${SCENE:=room}"
: "${OUT_DIR:=runs/joint-dual-room375k}"
: "${FACTOR:=1}"
echo "job : mem_probe scene=${SCENE} out=${OUT_DIR} factor=${FACTOR}"
echo "node : $(hostname)"

export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
export PATH="$HOME/storage/user/zyuef/micromamba/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate main

cd "$HOME/budgetgs-m0b"
python mem_probe.py --scene-name "${SCENE}" --out-dir "${OUT_DIR}" --factor "${FACTOR}"
echo "done: $(date --iso-8601=seconds)"
