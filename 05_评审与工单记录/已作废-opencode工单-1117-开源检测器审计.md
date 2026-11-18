# opencode 工单 · 1117 · 开源 SOTA 检测器审计(Binoculars + Fast-DetectGPT)

项目根目录:`~/Documents/课程-2026秋/INT4096-AI专题/小组项目-DetectingTheDetector/`
签发:Claude 主会话 2026-11-17。验收由主会话执行,不采信自报。
**执行顺序:重灾区演示 → 本单 → Qwen v2 重训。**本单与 Qwen v2 不要同时占用同一块 GPU。

## 目标(一句话)
用两个 2024 年顶会的开源零样本检测器(**只打分,不训练**),给非母语与母语的人写作文打分,看它们是否同样把非母语作文误判为 AI。回应两个质疑:"2023 年的商用分数太旧"和"你们自己训的检测器本来就差"。

## 红线
1. 主线审计、`results/` 已有文件、`data/`、主 notebook 一律不改。新代码写 `colab/OPENSOURCE_AUDIT_CELL.py`,结果写 `results/opensource_audit/`(新建)。
2. **阈值只用各仓库/论文给的官方阈值,不准在我们的数据上调阈值。**另报告 AUC 作为不依赖阈值的补充。
3. 不训练、不微调任何模型。
4. 长任务 nohup 后台 + 短轮询,遵守 `colab/RECOVERY_SOP.md`。
5. 不确定就停下写进报告,不要猜。

## 阶段 0:摸底(结果写进报告第 1 节)
1. 核实仓库存在且可用,记录 URL、commit hash、许可证、Star 数(现查,不要凭印象):
   - Binoculars:预期 `github.com/ahans30/Binoculars`(论文 Hans et al., ICML 2024)
   - Fast-DetectGPT:预期 `github.com/baoguangsheng/fast-detect-gpt`(论文 Bao et al., ICLR 2024)
   以上地址未经本次核实,以实测为准。
2. 记录各自使用的底座模型、默认 max tokens、**官方阈值**(写明原文出处:README 行或论文页码)。
3. **Binoculars 论文 ESL 部分(主会话 1117 已读 arXiv:2401.12070 原文核实)**:Hans et al., ICML 2024(PMLR 235),第 5 节 "False-positive rates on non-native speakers' writing"。他们用的是 **EssayForum** 的 ESL 作文,比较的是**原文 vs 语法修正版**,报告两者准确率都是 99.67%(Figure 7);**没有用 Liang 的 TOEFL 数据,也没有拿非母语和母语作文做对比**。结论段原话:"this analysis is only a first step … does not absolve developers … from carefully verifying the impact on their systems"。你只需补:EssayForum 篇数、Figure 7 用的阈值。我们这次的测试正好补上它没做的"非母语 vs 母语"对比。
4. 估算显存与时长;Colab 当前 GPU 不够就停下报告。

## 阶段 1:数据(全部只测)
- Liang 数据集(仓库 `Weixin-Liang/ChatGPT-Detector-Bias`,路径 `Data_and_Results/`,见 RECOVERY_SOP):
  - 人写:`TOEFL_real_91`、`HewlettStudentEssay_real_88`,以及其余人写语料(合计应为 394 篇,逐个列出)
  - AI:全部 AI 生成语料(合计应为 352 篇,逐个列出)
  - `TOEFL_gpt4polished_91`:**单列,不得计入 FPR**
- ELLIPSE 3,911:HF 镜像 `tcapelle/feedback-prize-english-language-learning-fluency`(v1 已用过),全部只测。
- 篇数与上面预期不符就停下报告。

## 阶段 2:打分
- 两个检测器各给上面所有文本打分,保存逐篇原始分数(CSV:corpus, essay_id, n_tokens, raw_score, flagged_official)。
- 所有文本统一截断长度(取两者都支持的值,写明);记录每个语料的 token 数中位数。
- 先用 10 篇试跑并核对方向(分数越高/越低代表 AI,按官方定义),写进报告再全量跑。

## 阶段 3:统计
每个检测器 × 每个语料:
- 人写语料:按官方阈值的**误判率(FPR)** + Wilson 95% 置信区间;
- AI 语料:**检出率(TPR)** + 95% CI;
- `TOEFL_gpt4polished_91`:"润色后被判 AI 比例",单列;
- **AUC**:Liang 人写 394 vs AI 352;另算 TOEFL 91 vs AI 352、Hewlett 88 vs AI 352 两个子 AUC;
- 关键对比:TOEFL FPR − Hewlett FPR、ELLIPSE FPR − Hewlett FPR(带 CI);
- 加一张图:各语料 FPR 柱状图(带误差线),英文,16:9,投影可读。

## 阶段 4:用新检测器检验我们的经典 ML 风险模型(跨检测器外部验证)
前提:重灾区工单的阶段 1 复现闸门已通过(LR 复现 AUC 0.88996±0.005)。未通过则跳过本阶段并说明。**本阶段不受重灾区工单阶段 1.5 闸门结果约束,无论过不过都要跑**——它本身就是另一种语料内检验。
背景:主会话 1117 发现全语料 LR 的 AUC(0.890)与"只按语料打分"的基线(≈0.888)几乎相同,模型可能主要在认 TOEFL 话题词。ELLIPSE 题目与 TOEFL 不同,话题词捷径在这里用不上,所以本阶段能检验模型有没有学到写法本身。
1. 准备**两个**模型,都在原 Liang 人写 394 篇 × 7 商用检测器数据上全量训练、不调参:
   - (a) notebook 原模型:TF-IDF 1–2gram + 8 个手工特征;
   - (b) 仅 8 个手工特征的 LR。
   给 TOEFL 91、Hewlett 88、ELLIPSE 3,911 每篇人写作文分别输出两套"被冤枉风险分"。(检测器编号不是特征,同一篇作文只有一个分。)
2. 以 Binoculars、Fast-DetectGPT 各自按官方阈值的"是否误判"为真值,(a)(b) 分别计算风险分的 ROC-AUC(带 bootstrap 95% 区间)、在 0.5 阈值下的 Recall/Precision/F1,以及多数类基线。**重点报告 ELLIPSE 3,911 篇**(训练时从未见过,检测器也不同,而且是单一语料)。
3. 加一个**纯长度基线**:只用作文词数做风险分,同样算 AUC。若 (a)(b) 都不比长度基线好,要明确写出来。
3. 某检测器在某语料上误判篇数 < 20 时,不计算该格 AUC,写"正例太少"。
4. 结果进 REPORT.md 单独一节,标题"External validation of the false-flag risk model"。

## 交付(写到 `results/opensource_audit/`)
- `scores_binoculars.csv`、`scores_fastdetectgpt.csv`、`summary.json`、图 PNG、运行日志。
- `REPORT.md`:
  1. 阶段 0 结论(仓库信息、官方阈值出处、Binoculars 论文 ESL 部分原文摘录);
  2. 篇数核对;
  3. 统计表(两个检测器并排);
  4. GPU 型号、耗时、截断长度;
  5. 产物 sha256 与可复制的重跑命令;
  6. 所有数字标"请独立复核",写明出自哪次运行;
  7. 局限:语料来源/年龄段不同(TOEFL 成人考生、Hewlett 美国中学生、ELLIPSE 8–12 年级 ELL),差异不能全归因于母语背景;零样本检测器结果依赖底座模型。
