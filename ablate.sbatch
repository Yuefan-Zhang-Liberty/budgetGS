#!/bin/bash
# BudgetGS A/D/C forward ablation: joint training (compress during training).
#   base     -> ABLATE_FLAGS="--ablate-a --ablate-d --ablate-c"
#   +A       -> ABLATE_FLAGS="--ablate-d --ablate-c"
#   +A+D     -> ABLATE_FLAGS="--ablate-c"
#   +A+D+C   -> ABLATE_FLAGS=""           (full method, code defaults)
# Scene via SCENE; output dir via OUT_DIR; optional STEPS/WARMUP for smoke tests.
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:8.6|GPU_CC:8.9|GPU_CC:9.0"
#SBATCH --time=08:00:00
#SBATCH --output=/storage/slurm/zyuef/logs/bgs-ablate-%j.out
#SBATCH --error=/storage/slurm/zyuef/logs/bgs-ablate-%j.err

set -euo pipefail
: "${SCENE:?SCENE must be set}"
: "${OUT_DIR:?OUT_DIR must be set}"
: "${TARGET_N:=375000}"
: "${FACTOR:=2}"
: "${STEPS:=30000}"
: "${WARMUP:=15000}"
: "${ABLATE_FLAGS:=}"

echo "job      : ${SLURM_JOB_ID} (${SLURM_JOB_NAME}) scene=${SCENE} out_dir=${OUT_DIR} target_n=${TARGET_N} factor=${FACTOR} steps=${STEPS} warmup=${WARMUP}"
echo "ablate   : '${ABLATE_FLAGS}'"
echo "node     : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
export PATH="$HOME/storage/user/zyuef/micromamba/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate main
echo "python   : $(which python)"

cd "$HOME/budgetgs-m0b/experiments"
export PYTHONPATH="$HOME/budgetgs-m0b:${PYTHONPATH:-}"

python train_joint.py \
    --scene "$HOME/storage/user/zyuef/datasets/360_v2/${SCENE}" \
    --scene-name "${SCENE}" \
    --out-dir "${OUT_DIR}" \
    --steps "${STEPS}" \
    --warmup "${WARMUP}" \
    --target-n "${TARGET_N}" \
    --budget-mb 10 \
    --prune-every 250 \
    --sh-degree 3 \
    --seed 0 \
    --factor "${FACTOR}" \
    ${ABLATE_FLAGS}

echo "finished : $(date --iso-8601=seconds)"
