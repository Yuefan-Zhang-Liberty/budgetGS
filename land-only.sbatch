#!/bin/bash
# BudgetGS landing-only rerun: finish the REAL landing for a saved joint ckpt
# without retraining (streamed per-view evaluation, val images only).
# Scene via env SCENE, out dir via env OUT_DIR, factor via env FACTOR.
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:8.6|GPU_CC:8.9|GPU_CC:9.0"
#SBATCH --time=01:00:00
#SBATCH --output=/storage/slurm/zyuef/logs/bgs-land-%j.out
#SBATCH --error=/storage/slurm/zyuef/logs/bgs-land-%j.err

set -euo pipefail
: "${SCENE:?SCENE must be set}"
: "${SCENE_NAME:=room}"
: "${OUT_DIR:=runs/joint-dual-room375k-f1}"
: "${FACTOR:=1}"
echo "job      : ${SLURM_JOB_ID} (${SLURM_JOB_NAME}) scene=${SCENE} out_dir=${OUT_DIR} factor=${FACTOR}"
echo "node     : $(hostname)"
echo "gpus     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "started  : $(date --iso-8601=seconds)"

export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
export PATH="$HOME/storage/user/zyuef/micromamba/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate main

cd "$HOME/budgetgs-m0b"
python land_only.py \
    --scene "$HOME/storage/user/zyuef/datasets/360_v2/${SCENE}" \
    --scene-name "${SCENE_NAME}" \
    --out-dir "${OUT_DIR}" \
    --factor "${FACTOR}" \
    --budget-mb 10 \
    --sh-degree 3

echo "finished : $(date --iso-8601=seconds)"
