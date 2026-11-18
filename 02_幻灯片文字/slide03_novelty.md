# 第 3 页:Data & design —— 写清楚"我们的新东西"(英文稿 + 讲稿)

规格书原文:"If a dataset has been analyzed by many people, you may consider choosing something else, **unless you have novel insights.**"
Liang et al. (2023) 的数据被很多人分析过,所以这一页要主动把"新"摆出来。数字都已在服务器上核对过(split_summary.json、exams.parquet)。

## 幻灯片文字(英文,直接放进 Canva)

**Title:** Data & design: what we added to a well-known benchmark

**Left column — "Starting point"**
- Liang et al. (2023): 925 essays, scores from 7 commercial detectors (2023)
- Widely analysed, so we keep it as the shared test and add what it lacks

**Right column — "What is new"**
1. **7 new data sources** (PERSUADE 2.0, ELLIPSE, daigt_v4, HC-Var, W&I + LOCNESS, FCE, ArguGPT)
   → from 925 to about **10,000 held-out test essays**, from US school essays to learner exams
2. **Detectors we trained ourselves, three families on identical data**
   logistic regression · RoBERTa · Qwen 27B + LoRA
3. **A controlled A / B / C / D design**: only one thing changes, what the human essays are
   A native only · B + school-age non-native · C + exam-style non-native · D + more native types (control)
4. **A new finding**: self-trained detectors flag *unfamiliar* writing, native or not (distribution shift), so "bias" must be tested against native writing of the same unfamiliar kind
5. **A test that was missing**: Binoculars' paper compared ESL essays with their grammar-corrected versions; we compare **non-native with native** essays

**Bottom strip — "How we kept it honest"**
- Train / validation / test: **0 overlapping essays** (checked by full text and first 200 characters)
- All texts cleaned the same way (an earlier format shortcut was found and removed)
- Primary endpoints written down **before** the GPU runs

## 讲稿(约 1.5 分钟)
"Liang and colleagues released this benchmark in 2023 and many people have analysed it, so we did not want to just repeat it. We kept it as our common test, and added seven new data sources: school essays, learner essays from exams, and AI essays from several generators. That takes us from 925 essays to about ten thousand held-out test essays.

We also trained detectors ourselves, in three sizes, on exactly the same data. The design is controlled: arm A learns from native essays only, arm B adds school-age non-native essays, arm C adds exam-style non-native essays, and arm D adds more kinds of native essays as a control. Only the human essays change, so any difference comes from them.

This produced a finding we did not expect, which I will show in a moment: our detectors flag any writing they have not seen, including native writers. Finally, we checked that no essay appears in both training and test."

## 数字出处(被问到时用)
| 说法 | 数字 | 出处 |
|---|---|---|
| 测试作文 | 925 → 10,171 | exams.parquet 13,672 篇,去掉臂 C 训练用的那一半 FCE 和 W&I 后 10,171(E1 925, E2 1,592, E3 1,700, E4 618, E5 2,901, E6 800, E7 1,635) |
| 7 个新数据来源 | PERSUADE 2.0, ELLIPSE, daigt_v4, HC-Var, W&I + LOCNESS, FCE, ArguGPT | 其中 W&I 和 LOCNESS 是同一次发布,算一个来源 |
| 零重叠 | final_overlap_train_val_vs_exams = 0 | split_summary.json |
| PERSUADE 与 ELLIPSE 有 452 篇是同一批作文 | 452 / 2,244 | 已从训练池里去重 |
| Binoculars 论文 | EssayForum 的 ESL 作文原文 vs 语法修正版,准确率均 99.67% | arXiv 2401.12070 §5(主会话已读原文) |

## 注意
- "about 10,000" 是去掉臂 C 训练那一半以后的数;如果只说 13,672,要同时说明其中约 3,500 篇用于训练臂 C。
- 第 3 页不要写"我们的检测器更好",只写设计和新增内容。
