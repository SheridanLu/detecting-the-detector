# Compare dequantized tensors against an original bf16 shard (Qwen3.8-27B, partial Sept download).
import json, torch
from safetensors import safe_open
ORIG = "/data/qwen38_bf16/model/model-00007-of-00018.safetensors"
DST = "/data/int4096_unified_v4/qwen27b_bf16_from_fp8"
wm = json.load(open(f"{DST}/model.safetensors.index.json"))["weight_map"]
fp8_src = json.load(open("/data/qwen38fn/model/model.safetensors.index.json"))["weight_map"]
LABEL = {True: "FP8->bf16", False: "kept as-is"}
rows = []
with safe_open(ORIG, "pt") as fo:
    keys = list(fo.keys())
    common = [k for k in keys if k in wm]
    print(f"orig shard keys={len(keys)} common_with_dequant={len(common)}")
    if not common:
        print("sample orig keys:", keys[:5]); print("sample dequant keys:", list(wm)[:5])
    for k in common:
        a = fo.get_tensor(k).float()
        with safe_open(f"{DST}/{wm[k]}", "pt") as fd:
            b = fd.get_tensor(k).float()
        if a.shape != b.shape:
            print("SHAPE MISMATCH", k, tuple(a.shape), tuple(b.shape)); continue
        was_fp8 = (k + "_scale_inv") in fp8_src
        rel = ((a - b).norm() / a.norm()).item()
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        rows.append((was_fp8, rel, cos, k))
for was_fp8 in (True, False):
    r = [x for x in rows if x[0] == was_fp8]
    if not r:
        continue
    print(f"{LABEL[was_fp8]}: n={len(r)} max_rel_err={max(x[1] for x in r):.4f} "
          f"mean_rel_err={sum(x[1] for x in r)/len(r):.4f} min_cos={min(x[2] for x in r):.6f}")
for x in sorted(rows, key=lambda x: -x[1])[:3]:
    print("worst:", f"fp8={x[0]} rel={x[1]:.4f} cos={x[2]:.6f}", x[3])
bad = [x for x in rows if x[0] and (x[1] > 0.10 or x[2] < 0.99)]
n_fp8 = sum(1 for x in rows if x[0])
print("CHECK", "PASS" if n_fp8 > 0 and not bad else "FAIL", f"(fp8_compared={n_fp8}, bad={len(bad)})")
