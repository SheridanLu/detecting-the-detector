# INT4096 follow-up on GPU models (RoBERTa, Qwen 27B): arms C and D — frozen BEFORE running
Same design, arms, native references, endpoints P1-P3 and attribution rule as PREREG_followup.md (sha256 in logs/prereg_followup_sha256.txt).
Training sets for C/D are the exact texts used by the classical-ML arms (followup/trainsets/, verified: retraining LR from the dump reproduces the saved predictions with max |diff| = 0.0).
Seeds: RoBERTa 5, Qwen 27B 2. Arms A/B are the existing runs (n=4000).
Amendments decided in advance (because they were observed in the first run):
1. Qwen's 90%-TPR threshold was degenerate (both arms flagged 1.1% of TOEFL) -> for Qwen, P1/P2 are reported at threshold 0.5 as the primary form, with AUC-based measures (bias AUC vs native-OOD, vs Hewlett) alongside; the 90%-TPR version is reported but not used for conclusions.
2. Qwen has only 2 seeds and varied strongly between seeds in the first run -> its effects are labelled "indicative" unless both seeds agree in sign and each is significant.
3. RoBERTa/Qwen co-location: RoBERTa runs are memory-capped and may fail with OOM; a failed run is re-run after Qwen finishes, never dropped silently.
