# Agent B — 深度学习禁令合规审查（2026-11-17）

审查范围：INT4096 小组项目 DetectingTheDetector 是否违反 spec 的深度学习（DL）禁令。
核对文件：作业说明 docx 两份、PACKAGE/COMPLIANCE_CHECK.md、PROJECT_DECISIONS.md、PROJECT_PLAN_v2_1112.md、PACKAGE/slides/SLIDES_OUTLINE.md。

---

## 1. Spec 原文引用（逐字）

来源：`~/Documents/课程-2026秋/INT4096-AI专题/小组项目-DetectingTheDetector/20261008-作业说明-INT4096小组项目要求.docx` 及 `20261015-INT4096-小组项目规格书-B1-LP-01.docx`（两份经 `textutil -convert txt` 转出后**逐字一致**，Methodology 节同句）：

> "Detail the machine learning model(s) you plan to utilize. You could use other models not included in the lecture notes. **Since this course is about classical machine learning, please do not use deep learning in this project.** You will have the opportunity to use it in future projects."

### 禁令范围解读

- **措辞级别**："please do not use" 是祈使式禁止。虽有 "please"，但结合后句 "You will have the opportunity to use it in **future** projects"，语义是明确的本项目禁令 + 未来补偿，**不是建议**。
- **豁免条款**：**无**。全文（218 行转出文本）没有"除非经导师同意/可申请豁免"之类条款。唯一的软出口是紧邻的 "You **could** use other models not included in the lecture notes"——它放行的是"非授课清单的**其他**模型"，但随后一句立即把 DL 单独划出，所以该句**不构成 DL 豁免**。
- **范围**："in this project" 限定本项目全部交付物（报告、slides、notebook 均属 project 范围）。

### 评分细则权重原文（两份 docx 的 Evaluation Criteria 表一致）

| 维度 | 权重 |
|---|---|
| Introduction and theoretical background | 20% |
| Data searching preparation | 10% |
| Programming Skill | 20% |
| Model Evaluation, discussion & conclusion | 20% |
| Clarity and Organization of the Report | 10% |
| Clarity and Presentation of the Results | 20% |

与 `PACKAGE/COMPLIANCE_CHECK.md:19` 所记 "20/10/20/20/10/20" 一致。注意：六个维度**没有任何"超纲加分/creativity bonus"项**，"老师口头说 DL 加分"在 rubric 层面无对应载体。

---

## 2. 项目自己的合规论证原文

来源：`PACKAGE/COMPLIANCE_CHECK.md:30-32`（"深度学习口径"节）：

> "规格书明文禁止 DL;老师口头允许 LLM/DL 且加分。汇报措辞:主线用经典 ML(合规),RoBERTa/Qwen 定位为 "audit extension",挂靠 W7 transfer learning 语言。主线审计结果一行不动。"

配套的 Q&A 预案（`PACKAGE/slides/SLIDES_OUTLINE.md:96`）：

> "**Why no deep learning?** — Spec says classical ML; our RoBERTa/Qwen runs are bonus audits showing even DL doesn't escape the bias. (挂靠 W7 transfer learning 话术)"

即项目的论证是：**主线合规（经典 ML）+ DL 挂靠为 stress-test 扩展**。论证结构成立的前提是"老师口头允许"为真。

---

## 3. 教师批准证据：**无（书面）**

- `PROJECT_DECISIONS.md:3-8`（2026-11-12）记录的是**口头**批准："## 2026-11-12 · LLM 微调扩展线（**老师口头批准**：LLM/深度学习可以做，做了加分）…… 老师明确说过：LLM 和深度学习都可以做，且做了加分。" 该条自身标注"（用户确认，勿再质疑）"——即来源是转述，无原始凭证。
- `PROJECT_PLAN_v2_1112.md:10-11`（同日）明确自认："**'老师允许 DL/LLM 加分'目前无任何书面痕迹** → 书面确认到手前，交付物按禁令执行。" Phase 0 步骤 0.1 要求发 Moodle 消息书面确认并"截图存 `./evidence/`"（`PROJECT_PLAN_v2_1112.md:24`）。
- **实证核查**：在 `小组项目-DetectingTheDetector/` 目录下 `find . -iname "*evidence*" -o -iname "*confirm*"` **无任何命中**，`evidence/` 目录不存在。即截至今天（11-17），0.1 的书面确认在仓库内**没有落盘证据**。
- 附加红旗：`PROJECT_PLAN_v2_1112.md:25`（步骤 0.2）提到存在一份来历不明的自编辑文件 `02-任务书-深度学习已允许.docx`，官方原件无此修订——曾有人改文件制造"已允许"假象，组内已计划澄清。

**结论：口头批准仅有转述记录，书面批准证据：无。** 且 `PROJECT_DECISIONS.md`（"勿再质疑"）与 `PROJECT_PLAN_v2_1112.md`（"无书面痕迹，按禁令执行"）两份同日文档互相矛盾，计划书一方的闸门规则（Phase 2 闸门：书面确认没到手，DL 结果"只跑不交付"）才是合规上更保守的既定纪律。

---

## 4. 裁定：**灰色地带，偏违规**（无书面批准的条件下，按 spec 字面即违规）

**依据**：

1. 禁令是明文祈使句，无豁免条款，"future projects" 一句堵死了"本项目可以用"的解释空间。
2. RoBERTa 微调与 Qwen+LoRA 微调属教科书式 deep learning，无解释余地。
3. 项目内部纪律（`PROJECT_PLAN_v2_1112.md` Phase 2 闸门）本就规定书面确认没到手则"正式交付物不含 DL"——slides P9/P10（outline 版；即最终 deck 里的 P11/P12）把 "Attempt 2: We trained our own detector (RoBERTa)"、"Attempt 3: Fine-tuned LLM (Qwen + LoRA)" 作为正片 Attempt 链条交付，**已越过自己设的闸门**。
4. 缓解因素（为何不是纯违规）：主线 RQ1/RQ2 及 Finding 1–3 全部是经典 ML（逻辑回归、阈值/校准/集成修复），不依赖 DL；DL 部分定位为 RQ3 stress-test，删掉不影响主结论成立。

### 建议的口播口径

现有讲法（"经典 ML 主线 + DL stress-test"）**部分站得住，但单靠它不够**——它解决的是"主线是否合规"，解决不了"交付物里出现 DL"这一字面违规。补强口径：

1. **主动披露，不藏**：讲到 P9/P10 前加一句书面式声明："The spec asks for classical ML, so our core findings rest entirely on classical models. For RQ3's stress-test we *additionally* fine-tuned RoBERTa and an LLM (post-lecture W7 material) to check whether deep learning escapes the bias — it does not."
2. **把 DL 讲成"对结论的加固"而非"方法"**：强调删掉这两页结论不变（"0/7 修不好"→"微调后的新检测器照样带偏误"），证明 DL 不是评分维度里的承重墙。
3. **口头确认留痕**： slides 冻结前（11-18 18:00 前）让 M4 在 Moodle 发消息请老师书面确认（即 0.1 步骤），并在 slides 该两页脚注写 "extension confirmed with course instructor (in progress)"；拿到回复前，Q&A 遇到质疑就退到预案口径（`SLIDES_OUTLINE.md:96`）。
4. **降级预案**（若书面确认始终没拿到）：把 RoBERTa/Qwen 两页降为 backup slides，正片只留一句 "we also stress-tested with fine-tuned models; same bias"——既保字面合规，又保留被问到时的弹药。

---

## 引用文件清单

| 文件 | 用途 |
|---|---|
| `小组项目-DetectingTheDetector/20261008-作业说明-INT4096小组项目要求.docx` | 禁令原文 + 评分权重（转出文本第 15 行、55–155 行） |
| `小组项目-DetectingTheDetector/20261015-INT4096-小组项目规格书-B1-LP-01.docx` | 同上，逐字一致 |
| `小组项目-DetectingTheDetector/PACKAGE/COMPLIANCE_CHECK.md` | 项目合规论证（30–32 行）、rubric 权重（19 行） |
| `小组项目-DetectingTheDetector/PROJECT_DECISIONS.md` | 口头批准记录（3–8 行） |
| `小组项目-DetectingTheDetector/PROJECT_PLAN_v2_1112.md` | "无书面痕迹"自认（10–11 行）、闸门规则（46 行）、0.1/0.2 步骤（24–25 行） |
| `小组项目-DetectingTheDetector/PACKAGE/slides/SLIDES_OUTLINE.md` | DL 在 slides 中的定位与 Q&A 预案（54–60、96 行） |
