#!/bin/bash
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
export PATH="$HOME/storage/user/zyuef/micromamba/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate main
cd "$HOME/budgetgs-m0b"
echo "=== py_compile ==="
python -m py_compile directionA_offline.py && echo COMPILE_OK
echo "=== import chain ==="
python -c "import directionA_offline as m; print('IMPORT_OK'); from budgetgs.modules.quantize import ATTRIBUTE_GROUPS; print('ATTRIBUTE_GROUPS:', ATTRIBUTE_GROUPS); from budgetgs.evaluation.landing import _metrics, _render_views, _splats_tensors; print('LANDING_HELPERS_OK'); from directionB_scene import groups_with_shn, write_mixed_container, dequant_mixed, real_roundtrip; print('DIRB_HELPERS_OK'); print('groups3:', groups_with_shn(3))"
