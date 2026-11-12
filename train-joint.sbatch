#!/bin/bash
# BudgetGS joint training (compress during training) on one Mip-NeRF360 scene.
# Two-phase: 0-15k standard densify/no-quant; 15k-30k differentiable learnable
# per-attribute quantization + pruning to a target Gaussian count + MCMC
# regularizers/noise; then a REAL zlib landing with PSNR/SSIM/LPIPS.
# Scene via env SCENE, target Gaussian count via env TARGET_N.
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:8.6|GPU_CC:8.9|GPU_CC:9.0"
#SBATCH --time=08:00:00
#SBATCH --output=/storage/slurm/zyuef/logs/bgs-joint-%j.out
#SBATCH --error=/storage/slurm/zyuef/logs/bgs-joint-%j.err

set -euo pipefail
: "${SCENE:?SCENE must be set}"
: "${TARGET_N:?TARGET_N must be set}"
: "${LAMBDA_MODE:=dual}"
: "${LAMBDA_ETA:=1e-3}"
: "${OUT_DIR:=runs/joint}"
: "${FACTOR:=2}"
: "${N_HOLD_STEP:=}"
echo "job      : ${SLURM_JOB_ID} (${SLURM_JOB_NAME}) scene=${SCENE} target_n=${TARGET_N} n_hold=${N_HOLD_STEP:-none} lambda_mode=${LAMBDA_MODE} eta=${LAMBDA_ETA} out_dir=${OUT_DIR} factor=${FACTOR}"
echo "node     : $(hostname)"
echo "gpus     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "started  : $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
export PATH="$HOME/storage/user/zyuef/micromamba/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate main
echo "nvcc     : $(which nvcc)"

cd "$HOME/budgetgs-m0b"
EXTRA_ARGS=""
if [ -n "${N_HOLD_STEP}" ]; then EXTRA_ARGS="--n-hold-step ${N_HOLD_STEP}"; fi
python train_joint.py \
    --scene "$HOME/storage/user/zyuef/datasets/360_v2/${SCENE}" \
    --scene-name "${SCENE}" \
    --out-dir "${OUT_DIR}" \
    --steps 30000 \
    --warmup 15000 \
    --target-n "${TARGET_N}" \
    --budget-mb 10 \
    --prune-every 250 \
    --sh-degree 3 \
    --seed 0 \
    --factor "${FACTOR}" \
    --lambda-mode "${LAMBDA_MODE}" \
    --lambda-eta "${LAMBDA_ETA}" \
    ${EXTRA_ARGS}

echo "finished : $(date --iso-8601=seconds)"
