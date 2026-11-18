#!/bin/bash
# Resumable, stall-proof download of falcon-7b / falcon-7b-instruct from hf-mirror (size-verified).
M=/data/int4096_unified_v4/models
for repo in falcon-7b falcon-7b-instruct; do
  mkdir -p $M/$repo
  curl -sS "https://hf-mirror.com/api/models/tiiuae/$repo/tree/main" > $M/$repo/.tree.json
  python3 -c "
import json
for f in json.load(open('$M/$repo/.tree.json')):
    p=f['path']
    if f.get('type')=='file' and (p.endswith(('.json','.safetensors','.txt','.model')) or p.startswith('tokenizer')) and not p.startswith('pytorch_model'):
        print(p, f.get('lfs',{}).get('size', f.get('size')))
" > $M/$repo/.files.txt
  while read p size; do
    for try in $(seq 1 60); do
      have=$(stat -c %s "$M/$repo/$p" 2>/dev/null || echo 0)
      [ "$have" = "$size" ] && break
      curl -sSL -C - --speed-limit 200000 --speed-time 30 --retry 3 -o "$M/$repo/$p" "https://hf-mirror.com/tiiuae/$repo/resolve/main/$p"
    done
    have=$(stat -c %s "$M/$repo/$p" 2>/dev/null || echo 0)
    echo "FILE $repo/$p have=$have want=$size $([ "$have" = "$size" ] && echo OK || echo SHORT)"
  done < $M/$repo/.files.txt
done
echo FALCON_DONE
