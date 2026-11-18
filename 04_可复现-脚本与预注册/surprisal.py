# Per-token surprisal (negative log-probability) of each essay under falcon-7b, the model behind Binoculars / Fast-DetectGPT.
# CPU only. Output: per-token character spans + nll (nats). A BOS token (eos id) is prepended so the first word has a context.
import json, sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(48)
M = "/data/int4096_unified_v4/models/falcon-7b"
texts = json.load(open(sys.argv[1])); t0 = time.time()
tok = AutoTokenizer.from_pretrained(M); model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16); model.eval()
print("loaded", round(time.time() - t0), "s", flush=True)
out = {}
for name, t in texts.items():
    enc = tok(t, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=512)
    ids = enc.input_ids; offs = enc.offset_mapping[0].tolist()
    inp = torch.cat([torch.tensor([[tok.eos_token_id]]), ids], 1)
    with torch.no_grad():
        lp = torch.log_softmax(model(inp).logits[0, :-1].float(), -1).gather(1, ids[0].unsqueeze(1)).squeeze(1)
    out[name] = {"text": t, "tokens": [{"s": a, "e": b, "nll": -float(l)} for (a, b), l in zip(offs, lp)]}
    mean = -lp.mean().item(); print(f"{name}: {ids.shape[1]} tokens, mean nll {mean:.2f}, perplexity {2.718281828 ** mean:.1f}, {round(time.time() - t0)} s", flush=True)
json.dump(out, open(sys.argv[2], "w"))
print("SURPRISAL_DONE", flush=True)
