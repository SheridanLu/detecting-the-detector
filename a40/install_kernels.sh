#!/bin/bash
export UV_CACHE_DIR=/data/uv_cache TMPDIR=/data/tmp CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH
V=/data/int4096_unified_v4/venv/bin/python
/usr/local/bin/uv pip install --python $V setuptools wheel ninja packaging && /usr/local/bin/uv pip install --python $V flash-linear-attention && echo FLA_OK
# causal-conv1d: try prebuilt wheel first, else build for sm86 only
TORCH_CUDA_ARCH_LIST="8.6" CAUSAL_CONV1D_FORCE_BUILD=FALSE MAX_JOBS=16 /usr/local/bin/uv pip install --python $V --no-build-isolation causal-conv1d && echo CC1D_OK
$V -c "import fla, causal_conv1d; print('KERNELS_IMPORT_OK', fla.__version__, causal_conv1d.__version__)"
echo KERNELS_DONE
