# figures_1117 — 汇报用图(16:9, 1920x1080 PNG)

| 文件 | 放在哪一页 | 一句话 |
|---|---|---|
| fig01_hook.png | 第1页 开场 | 一篇真实 TOEFL 作文被 7 个商用检测器全判 AI,对照一篇母语作文 0/7(说明是挑出来的例子) |
| fig02_commercial.png | 第5页 | 商用检测器:TOEFL 误判 52-76%,抓 AI 的 AUC 仅 0.44-0.60 |
| fig03_opensource.png | 第6页 | 开源检测器误判和检出都好得多,但偏见 AUC 并没有比最好的商用更低 |
| fig04_shift.png | 第13页 | 自训检测器(臂A)对没见过的母语作文也大量误判 |
| fig05_trend.png | 第14页(最重要,高潮) | A→B→C 三个模型 TOEFL 误判一路下降,D 对照不降 |
| fig06_scoreboard.png | 第9页 | 逻辑回归/RoBERTa/Qwen 记分牌 |
| fig07_notfixed.png | 第10页 | 臂C 也改善了别的作文,但 CS224N 仍被判 AI |
| fig08_learning_backup.png | 备用页 | 数据量翻 4 倍几乎没用 |
| fig09_length_backup.png | 备用页 | 作文长度只解释开源检测器偏见的一小部分 |
| fig10_prf1.png | 第9页前后(评估页) | **Accuracy / Precision / Recall / F1**,以及 TOEFL 与 Hewlett 两组各自的 F1(评分表要求的指标) |
| fig11_prf1_full.png | 备用页 / 报告附录 | 同上,每个检测器、每个臂,均值±SD |
| fig12_hotspots.png | **新增一页:哪些词/句子惹的祸**(放在商用/开源检测器两页之后) | 一篇被判 AI 的 TOEFL 作文 vs 一篇母语作文,按"语言模型能不能预测这个词"染色,每句标困惑度 |
| fig13_polish.png | 备用页或紧跟上一页 | 同一篇 TOEFL 作文 GPT-4 润色前后:7/7 → 0/7,并写明改了哪几个词 |
| fig14_pipeline.png | **第4页(过程)** | 实验流程图:收集 → 清洗 → 去重 → 切分 → 训练 → 评估,关键数字标在上面 |
| fig15_pitfalls.png | **第13页(过程)** | 我们发现并改正的四个错误,每个都带"之前→之后"的数字 |
| fig16_models.png | **第5页(方法)** | 我们怎么用 ML / DL / LoRA:三个模型各自怎么读作文、训练了什么、为什么选、成本 |
| fig17_settings.png | 备用页 | 所有设置、设置的来源、"没有调优"及原因 |
| fig18_ml_method.png | **第7页(ML怎么做)** | 逻辑回归:流程图(哪部分训练)+ 训练了什么 / 为什么选 / 做不到什么 |
| fig19_ml_results.png | **第8页(ML结果)** | 逻辑回归四种训练数据下的误判(柱)+ 精确率/召回/F1(表) |
| fig20_dl_method.png | **第9页(DL怎么做)** | RoBERTa:流程图 + 训练了什么 / 为什么选 / 做不到什么 |
| fig21_dl_results.png | **第10页(DL结果)** | RoBERTa 同样的结果页 |
| fig22_lora_method.png | **第11页(LoRA怎么做)** | LoRA:冻结的 27B 模型 + 小附件(只训练 0.48%)+ 三格说明 |
| fig23_lora_results.png | **第12页(LoRA结果)** | Qwen+LoRA 同样的结果页(只有 2 个种子) |

- `tables/`:每张图对应的数据表(CSV),图上每个数都能在这里查到。
- `data/`:从服务器 `/data/int4096_unified_v4/metrics/` 拉回的原始指标。
- `make_figures.py`:重画命令 `python make_figures.py <本目录> <Liang Data_and_Results 目录> [hook commercial opensource shift trend scoreboard notfixed learning length]`。
- 口径:全部统一阈值 0.5(开源检测器用官方阈值);逻辑回归/RoBERTa 5 个 seed,Qwen 2 个 seed;FCE/W&I 的训练半份已从考卷里剔除。
- 颜色固定:蓝=逻辑回归,橙=RoBERTa,绿=Qwen,紫=开源检测器,红=TOEFL,灰=商用/对照。

- `slide03_novelty.md`:第 3 页"我们的新东西"的英文稿、讲稿和数字出处。
- P/R/F1 口径:Liang 测试集 394 篇真人(TOEFL 91 / Hewlett 88 / CollegeEssay 70 / CS224N 145)+ 352 篇 AI,AI 为正类,阈值 0.5;"F1 on TOEFL/Hewlett"用平衡精确率 P=TPR/(TPR+FPR_组),即把 AI 和该组真人当作一样多。

- 第12页的母语作文用的是大学申请文(无占位符),**不用 Hewlett**:Hewlett 88 篇里有 85 篇含 @CAPS1 之类的匿名占位符,会把困惑度推高,对比不公平。
- 第12页只是一对作文的示例;整体证据是右边 GPTZero 自己报告的困惑度中位数(TOEFL 42 对母语大学申请文 86)和 falcon-7b 的 89%。
