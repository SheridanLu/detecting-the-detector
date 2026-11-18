#!/bin/bash
set -e
export UV_CACHE_DIR=/data/uv_cache TMPDIR=/data/tmp UV_PYTHON_INSTALL_DIR=/data/uv_python
PY=$(ls -d /data/uv_python/cpython-3.11*/bin/python3.11 | head -1)
V=/data/int4096_unified_v4/venv
/usr/local/bin/uv venv --python $PY $V
/usr/local/bin/uv pip install --python $V/bin/python --index-url https://download.pytorch.org/whl/cu128 "torch==2.9.*"
/usr/local/bin/uv pip install --python $V/bin/python "transformers==5.22.2" peft accelerate safetensors "huggingface_hub" datasets scikit-learn pandas pyarrow scipy statsmodels matplotlib
$V/bin/python -c "import torch,transformers,peft,sklearn;print(\"VENV_OK torch\",torch.__version__,\"cuda\",torch.version.cuda,\"transformers\",transformers.__version__,\"peft\",peft.__version__,\"sklearn\",sklearn.__version__)"
