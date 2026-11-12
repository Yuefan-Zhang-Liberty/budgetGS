#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=9G
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:8.6|GPU_CC:8.9|GPU_CC:9.0"
#SBATCH --time=00:30:00
#SBATCH --output=/storage/slurm/zyuef/logs/bgs-render-%j.out
#SBATCH --error=/storage/slurm/zyuef/logs/bgs-render-%j.err
set -euo pipefail

# P1-3: render GT / ref / landed comparison tiles for a trained ckpt.
# env: SCENE (room|bonsai), RENDER_OUT (dir holding joint-<scene>.pt+result.json), FACTOR
: "${SCENE:?}"
: "${RENDER_OUT:=runs/joint}"
: "${FACTOR:=2}"

echo "job      : ${SLURM_JOB_ID} scene=${SCENE} render_out=${RENDER_OUT} factor=${FACTOR}"

export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
export PATH="$HOME/storage/user/zyuef/micromamba/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate main

cd "$HOME/budgetgs-m0b"
python render_compare.py \
    --scene-name "${SCENE}" \
    --out-dir "${RENDER_OUT}" \
    --factor "${FACTOR}" \
    --sh-degree 3
echo "finished : $(date --iso-8601=seconds)"
