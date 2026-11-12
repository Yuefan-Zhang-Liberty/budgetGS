#!/bin/bash
set +e
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
export PATH="$HOME/storage/user/zyuef/micromamba/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate main
echo "WHICH_NVCC=$(which nvcc)"
echo "CUDA_HOME=$CUDA_HOME"
python -c "import torch.utils.cpp_extension as c; print('torch_CUDA_HOME=', c.CUDA_HOME)"
echo "--- nvcc version ---"
nvcc --version 2>&1 | tail -3
echo "--- gsplat build cache ---"
find "$HOME/.cache" -maxdepth 4 -iname '*gsplat*' -o -iname '*fully_fused*' 2>/dev/null | head
