# Dequantize Qwen3.8-27B FP8 (e4m3, 128x128 block scales, weight_scale_inv) -> bf16 safetensors.
# W_bf16 = W_fp8.float() * block-expanded(weight_scale_inv). Scale tensors are dropped.
import json, os, shutil, time, torch
from safetensors import safe_open
from safetensors.torch import save_file
torch.set_num_threads(16)
SRC = "/data/qwen38fn/model"
DST = "/data/int4096_unified_v4/qwen27b_bf16_from_fp8"
B = 128
os.makedirs(DST, exist_ok=True)
idx = json.load(open(f"{SRC}/model.safetensors.index.json"))
wm = idx["weight_map"]
files = sorted(set(wm.values()))
new_map = {}
t0 = time.time()
n_deq = 0
for fi, fn in enumerate(files):
    out_path = f"{DST}/{fn}"
    with safe_open(f"{SRC}/{fn}", "pt") as f:
        keys = list(f.keys())
        tensors = {}
        for k in keys:
            if k.endswith("_scale_inv"):
                continue
            t = f.get_tensor(k)
            sk = k + "_scale_inv"
            if t.dtype == torch.float8_e4m3fn:
                if sk in keys:
                    s = f.get_tensor(sk)
                else:
                    sf = wm.get(sk)
                    assert sf is not None, f"missing scale for {k}"
                    with safe_open(f"{SRC}/{sf}", "pt") as g:
                        s = g.get_tensor(sk)
                o, i = t.shape
                se = s.float().repeat_interleave(B, 0)[:o].repeat_interleave(B, 1)[:, :i]
                t = (t.float() * se).to(torch.bfloat16)
                n_deq += 1
            tensors[k] = t.contiguous()
            new_map[k] = fn
    if not os.path.exists(out_path):
        save_file(tensors, out_path, metadata={"format": "pt"})
    print(f"[{fi+1}/{len(files)}] {fn} tensors={len(tensors)} elapsed={time.time()-t0:.0f}s", flush=True)
json.dump({"metadata": {"note": "dequantized from /data/qwen38fn/model (fp8 e4m3, block 128) by int4096 unified_v4"}, "weight_map": new_map},
          open(f"{DST}/model.safetensors.index.json", "w"), indent=1)
cfg = json.load(open(f"{SRC}/config.json"))
cfg.pop("quantization_config", None)
cfg["torch_dtype"] = "bfloat16"
if "text_config" in cfg: cfg["text_config"]["torch_dtype"] = "bfloat16"
json.dump(cfg, open(f"{DST}/config.json", "w"), indent=2)
for fn in os.listdir(SRC):
    if fn.endswith(".safetensors") or fn in ("config.json", "model.safetensors.index.json") or fn.startswith("."):
        continue
    p = f"{SRC}/{fn}"
    if os.path.isfile(p): shutil.copy2(p, f"{DST}/{fn}")
print(f"DEQUANT_DONE files={len(files)} fp8_tensors_dequantized={n_deq} total_tensors={len(new_map)} secs={time.time()-t0:.0f}", flush=True)
