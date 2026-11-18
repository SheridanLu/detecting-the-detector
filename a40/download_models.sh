#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=0
M=/data/int4096_unified_v4/models; mkdir -p $M
V=/data/int4096_unified_v4/venv/bin/python
for repo in FacebookAI/roberta-base tiiuae/falcon-7b tiiuae/falcon-7b-instruct; do
  $V - "$repo" "$M" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, M = sys.argv[1], sys.argv[2]
p = snapshot_download(repo, local_dir=f"{M}/{repo.split('/')[-1]}", max_workers=8,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "merges.txt", "vocab.json"])
print("MODEL_OK", repo, p, flush=True)
PY
done
echo MODELS_DONE
