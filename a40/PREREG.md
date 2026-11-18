# INT4096 unified_v4 — primary endpoints, frozen BEFORE the GPU run (RoBERTa / Qwen / open-source)

Frozen at: (see file mtime and sha256 recorded in logs/prereg_sha256.txt)
Design: arm A (human = native PERSUADE non-ELL only) vs arm B (human = half native + half non-native: ELLIPSE + PERSUADE ELL); same size, same AI texts, same settings. No arm C (owner decision 2026-11-17).

Primary endpoints (conclusions are drawn only from these):
1. TOEFL_real_91 false-flag rate at the threshold giving 90% TPR on the shared validation AI set: arm A vs arm B, paired per seed (McNemar + essay bootstrap 95% CI of B - A).
2. Bias AUC TOEFL_real_91 vs HewlettStudentEssay_real_88 (0.5 = no bias), arm A vs arm B, and vs the 7 commercial detectors.
3. Detection AUC ArguGPT_TOEFL_AI (1,635 AI essays on TOEFL prompts) vs TOEFL_real_91: arm B must not be clearly worse than arm A.

Everything else (E2-E6 rates and AUCs, learning curves, proficiency curves, threshold 0.5 rates) is secondary/descriptive.
Disclosure: classical-ML (LR) results were computed and viewed before these endpoints were written down.
