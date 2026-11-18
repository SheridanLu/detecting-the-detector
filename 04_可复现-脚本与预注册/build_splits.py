# int4096 unified_v4 — build exams, pools, validation and training manifests (server side).
# Gate: zero overlap (normalized full text or first 200 chars) between any train/val text and any exam text.
import json, re, hashlib, random, pathlib, collections
import pandas as pd

ROOT = pathlib.Path("/data/int4096_unified_v4")
D = ROOT / "data"
OUT = ROOT / "splits"
OUT.mkdir(exist_ok=True)
LIANG = D / "liang_local/Data_and_Results"

def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip()

def keys_of(t):
    n = norm(t)
    return n, n[:200]

rows = []  # exam rows

# ---- E1 Liang 925 ----
HUMAN = ["TOEFL_real_91", "HewlettStudentEssay_real_88", "CollegeEssay_real_70", "CS224N_real_145"]
AI = ["CS224N_gpt3_145", "CS224N_gpt3PromptEng_145", "CollegeEssay_gpt3_31", "CollegeEssay_gpt3PromptEng_31"]
EDITED = ["TOEFL_gpt4polished_91", "HewlettStudentEssay_GPTsimplify_88"]
for c in HUMAN + AI + EDITED:
    sub = "Human_Data" if (LIANG / "Human_Data" / c).exists() else "GPT_Data"
    docs = json.loads((LIANG / sub / c / "data.json").read_text(encoding="utf-8"))
    role = "human" if c in HUMAN else ("ai" if c in AI else "edited")
    for i, d in enumerate(docs):
        rows.append(dict(exam="E1", corpus=c, essay_id=f"{c}#{i}", role=role, text=d["document"]))
n_e1 = collections.Counter((r["corpus"]) for r in rows)
assert sum(n_e1.values()) == 925, n_e1

# ---- PERSUADE (E2 + training pools) ----
p = pd.read_csv(D / "persuade/persuade_essays.csv", dtype=str)
p["ell"] = p.ell_status.fillna("").str.strip()
ell_df = pd.read_parquet(D / "ellipse/data/train-00000-of-00001.parquet")
ellipse_k200 = set(ell_df.full_text.map(lambda s: keys_of(s)[1]))
p["k200"] = p.full_text.map(lambda s: keys_of(s)[1])
p["in_ellipse"] = p.k200.isin(ellipse_k200)

rng = random.Random(0)
ell_avail = p[(p.ell == "Yes") & (~p.in_ellipse)].copy()
non_ell = p[(p.ell == "No") & (~p.in_ellipse)].copy()
# E2: prompt-stratified 792 ELL (proportional), 800 non-ELL matched to the same prompt distribution
ell_avail = ell_avail.sample(frac=1, random_state=0)
target_ell = 792
props = ell_avail.prompt_name.value_counts(normalize=True)
alloc = (props * target_ell).round().astype(int)
diff = target_ell - alloc.sum()
alloc.iloc[0] += diff
e2_ell = pd.concat([ell_avail[ell_avail.prompt_name == pn].head(k) for pn, k in alloc.items()])
alloc_nat = (props * 800).round().astype(int); alloc_nat.iloc[0] += 800 - alloc_nat.sum()
non_ell_sh = non_ell.sample(frac=1, random_state=0)
e2_nat = pd.concat([non_ell_sh[non_ell_sh.prompt_name == pn].head(k) for pn, k in alloc_nat.items()])
for _, r in e2_ell.iterrows():
    rows.append(dict(exam="E2", corpus="PERSUADE_ELL", essay_id=f"P#{r.essay_id_comp}", role="human", text=r.full_text, prompt=r.prompt_name, grade=r.grade_level, score=r.holistic_essay_score))
for _, r in e2_nat.iterrows():
    rows.append(dict(exam="E2", corpus="PERSUADE_nonELL", essay_id=f"P#{r.essay_id_comp}", role="human", text=r.full_text, prompt=r.prompt_name, grade=r.grade_level, score=r.holistic_essay_score))

# ---- E3 W&I + LOCNESS ----
wi = D / "bea/wi+locness/json"
for lvl in ["A", "B", "C", "N"]:
    for part in ["train", "dev"]:
        f = wi / f"{lvl}.{part}.json"
        if not f.exists():
            continue
        for i, line in enumerate(open(f, encoding="utf-8")):
            j = json.loads(line)
            corpus = "LOCNESS_native" if lvl == "N" else f"WI_{lvl}"
            rows.append(dict(exam="E3", corpus=corpus, essay_id=f"{lvl}.{part}#{j.get('id', i)}", role="human", text=j["text"], level=lvl, cefr=j.get("cefr")))

# ---- E4 FCE ----
for part in ["train", "dev", "test"]:
    for i, line in enumerate(open(D / f"bea/fce/json/fce.{part}.json", encoding="utf-8")):
        j = json.loads(line)
        rows.append(dict(exam="E4", corpus="FCE", essay_id=f"fce.{part}#{j.get('id', i)}", role="human", text=j["text"], score=j.get("answer-s"), script_score=j.get("script-s"), l1=j.get("l1"), age=j.get("age")))

# ---- ELLIPSE split: 1000 train / rest = E5 ----
ell_ids = list(range(len(ell_df)))
random.Random(0).shuffle(ell_ids)
ell_train_idx = set(ell_ids[:1010])  # 1010 candidates; >=1000 survive exam dedup
for i, r in ell_df.iterrows():
    if i in ell_train_idx:
        continue
    rows.append(dict(exam="E5", corpus="ELLIPSE_test", essay_id=f"ELL#{r.text_id}", role="human", text=r.full_text,
                     score=float(pd.to_numeric(r[["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]]).mean())))

# ---- daigt AI: E6 test 400 / val 300 / train pool ----
dg = pd.read_csv(D / "daigt/daigt_magic_generations.csv")
assert (dg.label == 1).all()
dg["n"] = dg.text.map(norm)
dg = dg.drop_duplicates("n").sample(frac=1, random_state=0).reset_index(drop=True)
for i, r in dg.iloc[:400].iterrows():
    rows.append(dict(exam="E6", corpus="daigt_llama_AI", essay_id=f"DG#{i}", role="ai", text=r.text, prompt=r.prompt_name))
ai_val = dg.iloc[400:550]
ai_pool = dg.iloc[550:]
hv = pd.read_json(D / "hcvar/hc_var.jsonl", lines=True)
hv = hv[(hv.domain == "writing") & (hv.label == 1)].copy()
hv["n"] = hv.text.map(norm)
hv = hv.drop_duplicates("n").sample(frac=1, random_state=0).reset_index(drop=True)
for i, r in hv.iloc[:400].iterrows():
    rows.append(dict(exam="E6", corpus="HCVar_ChatGPT_AI", essay_id=f"HV#{i}", role="ai", text=r.text))
hv_val = hv.iloc[400:550]
hv_pool = hv.iloc[550:]
ag = pd.concat([pd.read_csv(D / f"argugpt/machine-{x}.csv") for x in ["train", "dev", "test"]])
ag = ag[ag.exam_type == "toefl"]
for r in ag.itertuples():
    rows.append(dict(exam="E7", corpus="ArguGPT_TOEFL_AI", essay_id=f"AG#{r.id}", role="ai", text=r.text, prompt=str(r.prompt_id), level=str(r.model)))

exams = pd.DataFrame(rows)
exam_full = set(); exam_200 = set()
for t in exams.text:
    a, b = keys_of(t); exam_full.add(a); exam_200.add(b)
# all ELLIPSE texts are also off-limits for non-ELLIPSE pools (ELLIPSE train is its own pool)
def clean(df, col):
    k = df[col].map(keys_of)
    bad = k.map(lambda x: x[0] in exam_full or x[1] in exam_200)
    return df[~bad].copy(), int(bad.sum())

removed = {}
e2_ids = set(e2_ell.essay_id_comp) | set(e2_nat.essay_id_comp)
nat_pool = non_ell[~non_ell.essay_id_comp.isin(e2_ids)].drop_duplicates("k200")
nat_pool, removed["native_pool_vs_exams"] = clean(nat_pool, "full_text")
pell_pool = ell_avail[~ell_avail.essay_id_comp.isin(set(e2_ell.essay_id_comp))].drop_duplicates("k200")
pell_pool, removed["persuade_ell_pool_vs_exams"] = clean(pell_pool, "full_text")
ell_train = ell_df.iloc[sorted(ell_train_idx)].copy()
ell_train, removed["ellipse_train_vs_exams"] = clean(ell_train, "full_text")
ai_val, removed["ai_val_vs_exams"] = clean(ai_val, "text")
ai_pool, removed["ai_pool_vs_exams"] = clean(ai_pool, "text")
hv_val, removed["hcvar_val_vs_exams"] = clean(hv_val, "text")
hv_pool, removed["hcvar_pool_vs_exams"] = clean(hv_pool, "text")

nat_pool = nat_pool.sample(frac=1, random_state=0).reset_index(drop=True)
nat_val = nat_pool.iloc[:300]; nat_pool = nat_pool.iloc[300:]

def rec(text, label, source, sid):
    return dict(id=sid, text=text, label=label, source=source)

pool_rows = []
pool_rows += [rec(r.full_text, 0, "persuade_nonELL", f"P#{r.essay_id_comp}") for r in nat_pool.itertuples()]
pool_rows += [rec(r.full_text, 0, "persuade_ELL", f"P#{r.essay_id_comp}") for r in pell_pool.itertuples()]
pool_rows += [rec(r.full_text, 0, "ellipse", f"ELL#{r.text_id}") for r in ell_train.itertuples()]
pool_rows += [rec(r.text, 1, "daigt_ai", f"DGP#{i}") for i, r in zip(ai_pool.index, ai_pool.itertuples())]
pool_rows += [rec(r.text, 1, "hcvar_ai", f"HVP#{i}") for i, r in zip(hv_pool.index, hv_pool.itertuples())]
pool = pd.DataFrame(pool_rows)
val = pd.DataFrame([rec(r.full_text, 0, "persuade_nonELL", f"P#{r.essay_id_comp}") for r in nat_val.itertuples()] +
                   [rec(r.text, 1, "daigt_ai", f"DGV#{i}") for i, r in zip(ai_val.index, ai_val.itertuples())] +
                   [rec(r.text, 1, "hcvar_ai", f"HVV#{i}") for i, r in zip(hv_val.index, hv_val.itertuples())])

# ---- manifests: 5 seeds x 2 arms x sizes ----
SIZES = [1000, 2000, 4000]
nat_ids = list(pool[pool.source == "persuade_nonELL"].id)
ai_ids = list(pool[pool.source == "daigt_ai"].id)
hv_ids = list(pool[pool.source == "hcvar_ai"].id)
ellp_ids = list(pool[pool.source == "ellipse"].id)
pell_ids = list(pool[pool.source == "persuade_ELL"].id)
assert len(nat_ids) >= 4000 and len(ai_ids) >= 2000 and len(hv_ids) >= 2000, (len(nat_ids), len(ai_ids), len(hv_ids))
assert len(ellp_ids) >= 1000 and len(pell_ids) >= 1000, (len(ellp_ids), len(pell_ids))
manifests = {}
for s in range(5):
    r = random.Random(1000 + s)
    nat = r.sample(nat_ids, 4000); dg_s = r.sample(ai_ids, 2000); hv_s = r.sample(hv_ids, 2000)
    ellp = r.sample(ellp_ids, 1000); pell = r.sample(pell_ids, 1000)
    for n in SIZES:
        hA = nat[:n]
        hB = nat[:n // 2] + ellp[:n // 4] + pell[:n // 4]
        a = dg_s[:n // 2] + hv_s[:n // 2]
        manifests[f"seed{s}_A_n{n}"] = {"human": hA, "ai": a}
        manifests[f"seed{s}_B_n{n}"] = {"human": hB, "ai": a}

# ---- final overlap gate ----
used = set()
for m in manifests.values():
    used.update(m["human"]); used.update(m["ai"])
id2text = dict(zip(pool.id, pool.text))
overlap = 0
for i in used:
    a, b = keys_of(id2text[i])
    overlap += (a in exam_full or b in exam_200)
for t in val.text:
    a, b = keys_of(t)
    overlap += (a in exam_full or b in exam_200)

for c in ["score", "script_score", "grade", "cefr", "l1", "age", "level", "prompt"]:
    if c in exams.columns:
        exams[c] = exams[c].map(lambda v: None if v is None or (isinstance(v, float) and v != v) else str(v))
exams.to_parquet(OUT / "exams.parquet", index=False)
pool.to_parquet(OUT / "pool.parquet", index=False)
val.to_parquet(OUT / "val.parquet", index=False)
json.dump(manifests, open(OUT / "manifests.json", "w"))
summary = {
    "exam_counts": exams.groupby(["exam", "corpus"]).size().to_dict(),
    "pool_counts": pool.source.value_counts().to_dict(),
    "val_counts": val.source.value_counts().to_dict(),
    "removed_vs_exams": removed,
    "persuade_ell_total": int((p.ell == "Yes").sum()), "persuade_ell_in_ellipse": int(((p.ell == "Yes") & p.in_ellipse).sum()),
    "daigt_after_dedup": int(len(dg)),
    "n_manifests": len(manifests), "final_overlap_train_val_vs_exams": int(overlap),
}
summary["exam_counts"] = {f"{k[0]}|{k[1]}": int(v) for k, v in summary["exam_counts"].items()}
for f in ["exams.parquet", "pool.parquet", "val.parquet", "manifests.json"]:
    summary[f"sha256_{f}"] = hashlib.sha256((OUT / f).read_bytes()).hexdigest()
json.dump(summary, open(OUT / "split_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
print("SPLIT_GATE", "PASS" if overlap == 0 else "FAIL")
