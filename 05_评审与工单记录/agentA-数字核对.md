# Agent A 数字核对表（2026-11-17）

审计 agent 因只读约束未落盘，主会话代存。证据列均由 agent 引用，关键行已由主会话抽查复核（rq1_model_metrics.csv、roberta_5seeds.json、qwen38_summary.json）。

| 页码 | Deck 数字 | 判定 | 证据 |
|---|---|---|---|
| P2 | 61.2% vs ~0% | 找不到（文献值） | results/ 全部 CSV grep 61.2 无数据命中；仅出现在 deck 自身文本 PACKAGE/slides/SLIDES_OUTLINE.md:32 与 PACKAGE/README.md:16。本地最近值：7 检测器 TOEFL FP 均值 61.85%（results/rq1_fp_rate_corpus_by_detector.csv 行 TOEFL_real_91） |
| P6 | 925 records | 一致 | corpus_summary.csv 10 语料 n_docs 合计 925 |
| P6 | 394 human / 179 GPT-edited / 352 AI | 一致 | corpus_summary.csv 分类合计 |
| P6 | 91 TOEFL + 88 Hewlett | 一致 | corpus_summary.csv |
| P6 | 阈值 0.5 | 一致 | sensitivity_fp_by_threshold.csv 首组 0.5；notebook:22 |
| P7 | TOEFL 61.9% vs Hewlett 5.4% | 一致 | rq1_fp_rate_corpus_by_detector.csv 复算 61.85% / 5.36% |
| P8 | LR AUC 0.890 / Recall 0.531 / F1 0.584 | 一致 | rq1_model_metrics.csv 行 LogisticRegression (primary)：0.88996 / 0.53084 / 0.58424（主会话复核✓） |
| P8 | RF AUC 0.903 / Recall 0.196 | 一致 | 同文件 RandomForestClassifier：0.90261 / 0.19604（主会话复核✓） |
| P8 | majority baseline | 一致 | 同文件 accuracy=0.8160 |
| P8 | 2,468 records | 一致 | notebook:812「RQ1 rows built: 2468 (expected 394*7 - 290)」。注意口径：(essay×detector) 行数，≠ P6 的 925 essays |
| P8 | TF-IDF + 8 语言特征 | 一致 | notebook:781；handcrafted_features.csv 表头 8 特征列 |
| P8 | essay-grouped 5-fold CV | 一致 | notebook:969 StratifiedGroupKFold(n_splits=5) |
| P9 | 81 对 | 一致 | rq4_paired_deltas.csv 全 7 行 n_pairs=81 |
| P9 | 7/7 分数下降 | 一致 | 同文件 7 个检测器 mean_delta 全负（-0.285 至 -0.588） |
| P9 | GPTZero 33/41 | 一致 | rq4_flip_analysis.csv GPTZero 行：n_real_flagged=41, n_unfooled=33 |
| P11 | RoBERTa 5 seeds，84% vs 0% | 一致 | roberta_5seeds.json：5 seeds，mean_fpr_toefl=0.8396，mean_fpr_native=0.0（主会话复核✓；另有 mean_fpr_polished=1.0，与 P11 图第三根柱对应，但"误报"术语见 agentC H4） |
| P12 | 非母语 FP 0 / 母语 35.2% | 一致 | qwen38_summary.json lora_finetuned：toefl_human_fpr=0.0，hewlett_human_fpr=0.35227（主会话复核✓） |
| P12 | 3,911 unseen 96% / 两阶段 72% | 一致 | 同文件 ellipse_audit：n=3911，ai_rate_fpr=0.9596；twostage.ellipse_3911_fpr=0.7213（主会话复核✓） |
| P13 | kappa 最高 0.87 | 一致 | stats_kappa_matrix.csv 最大非对角值 0.8677（Crossplag–HFOpenAI） |
| P17 | 0–13% vs 52–76% | 一致 | rq1_fp_rate_corpus_by_detector.csv：母语真人语料 FP 范围 0.0–12.5%；TOEFL_real_91 范围 51.65–75.82% |

## 不一致项
无。

## 找不到项
P2 61.2%——任何结果文件中无此数据值；若声称是 Liang et al. 文献值则合理，但本地文件无法背书。最接近的本地值是 61.85%。
