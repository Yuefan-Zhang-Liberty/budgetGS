#!/bin/bash
# BudgetGS CUDA 13.3b toolchain env (devcube2 workstation, RTX 4090)
# Source before any gsplat JIT build / training run:
#   source slurm/env_cuda133b.sh
export PATH="$HOME/storage/user/zyuef/micromamba/bin:$HOME/.local/share/mamba/envs/main/bin:$PATH"
C="$HOME/.local/share/mamba/envs/cuda133b"
export CUDA_PATH="$C/targets/x86_64-linux"
export CUDA_HOME="$CUDA_PATH"
export TORCH_CUDA_ARCH_LIST=8.9
export LD_LIBRARY_PATH="$CUDA_PATH/lib:$C/lib:${LD_LIBRARY_PATH:-}"
