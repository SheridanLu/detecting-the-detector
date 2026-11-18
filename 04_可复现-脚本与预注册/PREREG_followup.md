# INT4096 unified_v4 follow-up (classical ML only) — design and primary endpoints, frozen BEFORE running

Why: the first run showed that self-trained detectors flag *any* essay unlike their training human data
(native CollegeEssay 80-90%, native CS224N ~100% for LR/RoBERTa), so "TOEFL vs Hewlett" confounded
non-nativeness with distribution shift, and arm B's non-natives (US school ELL) do not resemble TOEFL exam essays.

Arms (LR = TF-IDF 1-2gram + LogisticRegression, identical to the first run; 5 seeds; n_human = n_AI = 4,000):
- A (existing)  human = 4,000 native PERSUADE
- B (existing)  human = 2,000 native + 1,000 ELLIPSE + 1,000 PERSUADE-ELL
- C (new)       human = 2,000 native (same first 2,000 as A) + 2,000 exam-style non-native essays
                (random half of FCE and of each W&I level A/B/C; the other half stays as exam)
- D (new, control for "more diverse human sources") human = 2,000 native + 2,000 HC-Var human `writing` essays
AI side identical in all arms (same 2,000 daigt + 2,000 HC-Var ChatGPT per seed).
Evaluation for ALL arms excludes the FCE/W&I essays used to train arm C.

Native references (evaluation only): Hewlett (in-distribution for training), and native out-of-distribution:
CollegeEssay_real_70, CS224N_real_145, LOCNESS_native (50).

Primary endpoints (conclusions only from these):
P1. TOEFL_real_91 flag rate at each run's own 90%-TPR threshold (validation AI): arm C vs arm A, paired per seed
    (exact McNemar + bootstrap CI of C - A).
P2. Specific excess = TOEFL flag rate minus the mean flag rate over the three native-OOD corpora, same threshold:
    arm C vs arm A. Positive excess = TOEFL flagged more than other unfamiliar native writing.
P3. Detection must not collapse: AUC of ArguGPT_TOEFL_AI vs TOEFL_real_91 and of Liang AI 352 vs Hewlett,
    arm C vs arm A (a drop > 0.05 is reported as a failure of the intervention).
Attribution rule: an effect is credited to "exam-style non-native data" only if arm C beats BOTH arm A and arm D on P1.
Everything else is descriptive.

Disclosure: arm A/B results, the domain-shift diagnosis and the native-OOD flag rates were viewed before this file was written.
