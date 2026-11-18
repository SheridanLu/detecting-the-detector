# opencode 工单 · 1117 · Qwen v2 重训(修"训练集当测试集"+测加入非母语数据能否修偏)

项目根目录:`~/Documents/课程-2026秋/INT4096-AI专题/小组项目-DetectingTheDetector/`
签发:Claude 主会话 2026-11-17。验收由主会话执行,不采信自报。
**优先级低于《opencode工单-1117-真实作文重灾区演示》,那份先跑。**

## 为什么要重训(背景,必读)
v1(`colab/REPAIR_CELL.py`、`MEGA_CELL.py`)训练集 = `TOEFL_real_91`(label 0)+ GPT-3 176(label 1),
随后又在 `TOEFL_real_91` 上算 `toefl_human_fpr = 0.0`——**训练集当测试集,这个 0% 无效**。
另外 v1 没留任何 AI 测试集,算不出检出率/AUC,无法与基线比较。
v2 要同时解决这两点,并回答一个新问题:**训练时加入非母语学生作文,偏见能不能修掉?**

## 红线
1. 主线审计(7 商用检测器、2023 冻结分数)一行不动;`results/` 已有文件不改不删(含 `qwen38_summary.json`,v1 结果留档)。
2. v2 代码写新文件 `colab/QWEN_V2_CELL.py`,不改 v1 文件;结果写 `results/qwen_v2/`(新建)。
3. **TOEFL_real_91、HewlettStudentEssay_real_88、TOEFL_gpt4polished_91 一律不进训练**,只做测试。
4. 训练集与测试集按文本去重核对(完全相同文本 + 前 200 字符相同都算),重叠数必须为 0,写进报告。
5. 长任务一律 nohup 后台 + 短轮询,遵守 `colab/RECOVERY_SOP.md`。
6. 不确定就停下写进报告,不要猜;数据源拿不到就停,不要换成别的数据凑。

## 阶段 0:数据源摸底(先做,结果写报告第 1 节)
1. **ELLIPSE**:v1 用的是 HF 镜像 `tcapelle/feedback-prize-english-language-learning-fluency`(3,911 篇)。确认可下载、字段、篇数。
2. **AI 文本 + 母语人写作文(同为学生议论文体裁)**:找一个**不需要 Kaggle 登录**就能下的来源(HF 镜像优先),候选方向:Kaggle "LLM - Detect AI Generated Text" 相关的 DAIGT 数据集的 HF 镜像、PERSUADE 2.0 人写作文。**以上名称均未核实**,逐个实测可下载性、许可证、篇数、标签含义后再选。
   - 若只能走 Kaggle API:停下,报告需要业主提供 Kaggle token,不要自己找 token。
3. 记录每个数据源的 URL、版本/commit、许可证、下载后的 sha256。

## 阶段 1:切分(固定 seed=0,切分结果存成 id 列表文件)
- **ELLIPSE 3,911**:随机抽 1,000 篇作 `ell_train`,其余 2,911 篇作 `ell_test`(只测)。
- **母语人写**(阶段 0 选定的来源):抽 1,000 篇 `nat_train` + 300 篇 `nat_test`。
- **AI 文本**:抽 1,600 篇 `ai_train` + 400 篇 `ai_test`;Liang 的 GPT-3 176 篇全部进 `ai_train`(与 v1 口径衔接)。
- 篇数不够就按比例缩,写明实际篇数。

## 阶段 2:训练两个臂(同一套超参,同 seed)
- **臂 A(不含非母语)**:`nat_train` 1,000 + `ai_train`(label 1)。人写/AI 两类大致平衡。
- **臂 B(加入非母语)**:臂 A 全部 + `ell_train` 1,000(label 0);AI 侧补到与人写侧大致平衡(从 AI 池补,补的篇数写明)。
- 模型与 v1 一致:`Qwen/Qwen3.8-27B` + LoRA(SEQ_CLS,r=16,alpha=32,dropout 0.05,all-linear),bf16,max_length 512,1 epoch,lr 1e-4(v1 是 3 epoch,数据量大了改 1 epoch,写明)。
- 时间预算:两臂合计目标 ≤ 2 小时 GPU。若估算超时,**先跑臂 B**,臂 A 用剩余时间;只跑成一个就如实报告。

## 阶段 3:评估(两臂各跑一遍,阈值 0.5)
对每个测试集报告 n、被判 AI 的比例,并保存逐篇概率:
| 测试集 | 含义 |
|---|---|
| `TOEFL_real_91` | 非母语人写 → 误判率(FPR) |
| `HewlettStudentEssay_real_88` | 母语人写 → FPR |
| `ell_test` 2,911 | 非母语学生人写 → FPR |
| `nat_test` 300 | 母语人写 → FPR |
| `ai_test` 400 | AI → 检出率(TPR) |
| `TOEFL_gpt4polished_91` | **单列为"润色版被判 AI 比例",不得叫 FPR** |

另算:人写全体 vs `ai_test` 的 ROC-AUC、F1(阈值 0.5);非母语 FPR − 母语 FPR 的差值;**基线**(全判"人写"和随机)的同口径指标,用于对比。
单 seed,报告里注明"未做多 seed,方差未知"。

## 交付(写到 `results/qwen_v2/`)
- `split_ids/`(各切分 id 列表)、`arm_A.json`、`arm_B.json`(汇总 + 逐篇概率)、训练日志。
- `REPORT.md`:
  1. 阶段 0 数据源结论;
  2. 切分篇数 + 训练/测试去重核对结果(必须为 0);
  3. 上面评估表,两臂并排 + 基线;
  4. 实际 GPU 型号、耗时、超参;
  5. 产物 sha256;
  6. 所有数字标"请独立复核",写明出自哪次运行;
  7. 局限:单 seed;ELLIPSE 为 8–12 年级,与 TOEFL 成人考生不同;AI 文本来源与 GPT-4 时代差异。
