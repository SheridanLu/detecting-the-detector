#!/bin/bash
# int4096 unified_v4 data download (server side). Each step logs; failures do not stop other steps.
D=/data/int4096_unified_v4/data
mkdir -p $D/{liang,bea,ellipse,daigt,repos}
export HF_ENDPOINT=https://hf-mirror.com
cd $D/liang && [ -d ChatGPT-Detector-Bias ] || git clone -q https://github.com/Weixin-Liang/ChatGPT-Detector-Bias && (cd ChatGPT-Detector-Bias && echo "LIANG_COMMIT $(git rev-parse HEAD)")
cd $D/bea && for f in wi+locness_v2.1.bea19.tar.gz fce_v2.1.bea19.tar.gz; do [ -s "$f" ] || curl -sSL --retry 3 -o "$f" "https://www.cl.cam.ac.uk/research/nl/bea2019st/data/$f"; echo "BEA $f $(stat -c %s $f) $(sha256sum $f | cut -c1-64)"; done
for spec in "ellipse tcapelle/feedback-prize-english-language-learning-fluency 9a7a03c8b0ee2401b7e33d97f8689ae94c9ccaf7" "daigt ramensoft/daigt_v4 16457627f531a21640a2f6c723dadaf818f94578"; do
  set -- $spec; dir=$1; repo=$2; rev=$3
  files=$(curl -sS "$HF_ENDPOINT/api/datasets/$repo/tree/$rev?recursive=true" | python3 -c "import json,sys;print(\"\n\".join(x[\"path\"] for x in json.load(sys.stdin) if x.get(\"type\")==\"file\"))")
  echo "HF $repo files: $(echo $files | tr '\n' ' ')"
  for p in $files; do mkdir -p "$D/$dir/$(dirname $p)"; [ -s "$D/$dir/$p" ] || curl -sSL --retry 3 -o "$D/$dir/$p" "$HF_ENDPOINT/datasets/$repo/resolve/$rev/$p"; echo "HF_FILE $dir/$p $(stat -c %s $D/$dir/$p) $(sha256sum $D/$dir/$p | cut -c1-64)"; done
done
cd $D/repos && for r in ahans30/Binoculars baoguangsheng/fast-detect-gpt; do n=$(basename $r); [ -d $n ] || git clone -q https://github.com/$r; (cd $n && echo "REPO $r $(git rev-parse HEAD) $(git log -1 --format=%cs)"); done
echo DOWNLOAD_DONE
