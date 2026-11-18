# falcon-7b per-essay perplexity for every TOEFL and Hewlett essay (CPU), to check the pattern over all essays, not just one pair.
import json, sys, time, torch, numpy as np, pathlib
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(48)
M = "/data/int4096_unified_v4/models/falcon-7b"; L = pathlib.Path("/data/int4096_unified_v4/data/liang_local/Data_and_Results/Human_Data")
tok = AutoTokenizer.from_pretrained(M); model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).eval(); t0 = time.time()
res = {}
for corpus in ("CollegeEssay_real_70", "CS224N_real_145"):
    docs = [d["document"] for d in json.loads((L / corpus / "data.json").read_text(encoding="utf-8"))]; gz = json.loads((L / corpus / "GPTZero.json").read_text(encoding="utf-8"))
    rows = []
    for i, t in enumerate(docs):
        ids = tok(t, return_tensors="pt", truncation=True, max_length=512).input_ids
        inp = torch.cat([torch.tensor([[tok.eos_token_id]]), ids], 1)
        with torch.no_grad():
            lp = torch.log_softmax(model(inp).logits[0, :-1].float(), -1).gather(1, ids[0].unsqueeze(1)).squeeze(1)
        rows.append(dict(i=i, words=len(t.split()), has_at="@" in t, falcon_ppl=float(np.exp(-lp.mean().item())), gptzero_ppl=gz[i]["perplexity"], gptzero_score=gz[i]["score"]))
    res[corpus] = rows; print(corpus, len(rows), round(time.time() - t0), "s", flush=True)
json.dump(res, open("/data/int4096_unified_v4/data/all_ppl_native.json", "w"))
import statistics as st
for c, rows in res.items():
    print(c, 'n', len(rows), '| essays with @ placeholder:', sum(r['has_at'] for r in rows), '| words median', st.median([r['words'] for r in rows]), '| falcon ppl median %.1f' % st.median([r['falcon_ppl'] for r in rows]), '| GPTZero ppl median %.1f' % st.median([r['gptzero_ppl'] for r in rows]))

