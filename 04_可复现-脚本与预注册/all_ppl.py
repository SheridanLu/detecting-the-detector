# falcon-7b per-essay perplexity for every TOEFL and Hewlett essay (CPU), to check the pattern over all essays, not just one pair.
import json, sys, time, torch, numpy as np, pathlib
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(48)
M = "/data/int4096_unified_v4/models/falcon-7b"; L = pathlib.Path("/data/int4096_unified_v4/data/liang_local/Data_and_Results/Human_Data")
tok = AutoTokenizer.from_pretrained(M); model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).eval(); t0 = time.time()
res = {}
for corpus in ("TOEFL_real_91", "HewlettStudentEssay_real_88", "TOEFL_gpt4polished_91"):
    docs = [d["document"] for d in json.loads((L / corpus / "data.json").read_text(encoding="utf-8"))]; gz = json.loads((L / corpus / "GPTZero.json").read_text(encoding="utf-8"))
    rows = []
    for i, t in enumerate(docs):
        ids = tok(t, return_tensors="pt", truncation=True, max_length=512).input_ids
        inp = torch.cat([torch.tensor([[tok.eos_token_id]]), ids], 1)
        with torch.no_grad():
            lp = torch.log_softmax(model(inp).logits[0, :-1].float(), -1).gather(1, ids[0].unsqueeze(1)).squeeze(1)
        rows.append(dict(i=i, words=len(t.split()), has_at="@" in t, falcon_ppl=float(np.exp(-lp.mean().item())), gptzero_ppl=gz[i]["perplexity"], gptzero_score=gz[i]["score"]))
    res[corpus] = rows; print(corpus, len(rows), round(time.time() - t0), "s", flush=True)
json.dump(res, open("/data/int4096_unified_v4/data/all_ppl.json", "w"))
m = lambda rows, k: float(np.median([r[k] for r in rows]))
T, H, P = res["TOEFL_real_91"], res["HewlettStudentEssay_real_88"], res["TOEFL_gpt4polished_91"]
Hn = [r for r in H if not r["has_at"]]
print(f"falcon perplexity median: TOEFL {m(T,'falcon_ppl'):.1f} | Hewlett {m(H,'falcon_ppl'):.1f} | Hewlett without @ placeholders (n={len(Hn)}) {m(Hn,'falcon_ppl'):.1f} | TOEFL polished {m(P,'falcon_ppl'):.1f}")
print(f"words median: TOEFL {m(T,'words'):.0f} | Hewlett {m(H,'words'):.0f}")
print(f"GPTZero perplexity median: TOEFL {m(T,'gptzero_ppl'):.1f} | Hewlett {m(H,'gptzero_ppl'):.1f}")
print("corr(falcon_ppl, gptzero_ppl) over all essays:", round(float(np.corrcoef([r['falcon_ppl'] for r in T + H], [r['gptzero_ppl'] for r in T + H])[0, 1]), 2))
print("share of TOEFL essays with falcon ppl below the Hewlett median:", round(float(np.mean([r['falcon_ppl'] < m(H, 'falcon_ppl') for r in T])), 2))
