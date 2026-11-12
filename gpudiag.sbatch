#!/bin/bash
#SBATCH --nodes=1
#SBATCH --nodelist=node8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --time=00:03:00
#SBATCH --output=/storage/slurm/zyuef/logs/gpudiag-%j.out
#SBATCH --error=/storage/slurm/zyuef/logs/gpudiag-%j.err
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "=== /dev/nvidia* visible inside job ==="
ls -la /dev/nvidia* 2>&1
echo "=== nvidia-smi full ==="
nvidia-smi 2>&1 | head -25
