# Agent C — 严格评委挑刺清单（2026-11-17）

评审对象：18 页 INT4096 小组 deck（80dpi PNG，page-01 至 page-18）
范围说明：已剔除已知问题清单（P5 AUIT、P13 重复段、P9 残留图注、P2/P7 数字不一致、P8 缺 LR takeaway、P6 缺 split、P18 Liang 重复、P3 11% 遮挡）。以下全部为本次逐页看图新发现，只记图内实际可见内容。

---

## High

**H1. P5 · Stage 3 卡片末行 — "retained detectors" 应为 "retrained detectors"**
- 位置：Stage 3 卡片正文 "Evaluate threshold tuning, calibration, ensembles, and retained detectors (RoBERTa / Qwen)."
- 问题：意思性拼写错误。"retained（保留的）"与"retrained（重新训练的）"在方法论上完全两回事；框架页出现这个词，评委直接质疑作者对自家方案的理解。与已知 AUIT 拼写错误是两处独立错误。
- 修法：改为 "retrained detectors"。

**H2. P9 · Figure 5 标题引用了从未定义的 "RQ4"**
- 位置：图内标题 "Figure 5 — RQ4: pairing quality and paired detector-score deltas"
- 问题：P4 只定义了 RQ1–RQ3；P5 明确把 editing 划入 RQ2（Stage 2 "Test Predictability & Editing (RQ2)"）。图注却写 RQ4——RQ 编号体系前后矛盾，且说明图是从别的版本 deck 里复用来的没改干净。
- 修法：图注改 "RQ2 (editing)" 并重出图。

**H3. RQ3 承诺的 "ensembles" 全篇没有任何结果页回答**
- 位置：P4 RQ3 "…by thresholds, ensembles, or even training our own detector?"；P5 Stage 3 "threshold tuning, calibration, ensembles, and retrained detectors"
- 问题：P10 讲了 threshold tuning 与 calibration，P11/P12 讲了自训检测器，但 ensembles 从头到尾没有一页、一句话的结果。评委按 P4 逐项对答案时会发现承诺落空（"P4 的 RQ 是否都被回答"这一项上最明显的缺口）。
- 修法：要么补一张 ensembles 结果页，要么在 P10/P12 口播明确 "ensembles tried, no improvement / dropped"，并在 P4 或 P5 删除该词。

**H4. P11 · 纵轴 "False-positive rate" 下画了 "GPT-4 polished = 100%" 柱**
- 位置：P11 柱状图第三根柱
- 问题：GPT-4 polished 文本是被 AI 改写过的文本，检测器把它判为 AI 不是 false positive——把它画在"误报率"轴下且标 100%，属于评估指标术语错误，会被当成"模型评估不正确"直接扣分。前两根柱（native 0%、TOEFL 84%，均为纯人写文本）才是真 FP。
- 修法：把第三根柱单独命名（如 "Flag rate on GPT-4-polished"）或移出该图另述。

**H5. P12 · 右下文字块里 96% / 72% / 3,911 三个关键数字无图无锚点**
- 位置："BUT THE ERRORS MOVED" 段："On 3,911 unseen non-native essays from another corpus it reaches 96%; a two-stage training run lands at 72%."
- 问题：(a) 96% 和 72% 是什么的百分比（FP rate？哪个模型？Qwen 还是 two-stage Qwen？）完全靠口播猜；(b) "another corpus" 不给名字，违反"数据描述与划分"的评分点；(c) 这是全页信息密度最高、字号最小的一块，却承载着与 0%/35.2% 同级的关键证据。既是清晰性问题也是时间风险。
- 修法：拆成两行小字各带主语（"FP on <corpus名> non-native: 96% (Qwen) → 72% (two-stage)"），口播只讲一句。

---

## Medium

**M1. P2 · "~0%" native 基线与自家 P17 "0–13%" 自相矛盾**
- 位置：P2 右侧红色大数字 "61.2% vs. ~0%" vs P17 "0–13% Hewlett · n = 88"
- 问题：与已知的 61.2/61.9 不一致是两回事：P2 声称 native 误报率约 0%，但 P7 热图 Hewlett 行最高 12%、P17 自己写 0–13%。评委同时看到三页会问"native 到底是不是 0"。
- 修法：P2 改为 "61.2% vs. 0–13%"，或注明 "~0%（Liang 原文口径）, own audit: up to 13%"。

**M2. P17 · "0–13%" 与 P7 热图 Hewlett 行最大值 12 对不上**
- 位置：P17 左侧大数字 vs P7 左热图 HewlettStudentEssay_real_88 行（最大格 12）
- 问题：同一统计量两页两个上限值（若 P7 格子是四舍五入可以解释，但 deck 内没有解释，评委只看数字差异）。
- 修法：两页统一为一个数（建议都用 unrounded 后的 13%，或 P7 保留一位小数）。

**M3. P3 · "2025 follow-up audits" 无出处**
- 位置：右侧 "Literature & Empirical Context" 面板第二段
- 问题：面板开头刚立了 Liang/Weber-Wulff 两块引文牌子，紧接着的"2025 follow-up audits confirm bias persists (25% vs 11% on GPTZero)"却没有任何引用。背景准确性评分点上，这个最"新"的证据反而是悬空的。
- 修法：补一条具体引用（作者+年份）或删掉 2025 句。

**M4. P13 · kappa 热图没说算在什么数据上；"lower-perplexity" 论断无支撑**
- 位置：左热图整体；右上 "THE BOUNDARY" 段
- 问题：(a) kappa 是在哪个 corpus、哪个阈值下的 binary flag 上算的，页面无任何说明，不可复现；(b) "Non-native writing is naturally lower-perplexity" 直接断言了 7 个商用检测器的工作原理是 perplexity-based，而这恰是 Liang et al. 的解释假说而非定论——评委（尤其懂 NLP 的）会追打。与已知的 P13 重复段是两处问题。
- 修法：图下加一行 "Pairwise Cohen's κ on binary flags (threshold 0.5), all corpora pooled"；BOUNDARY 段加 "consistent with the perplexity-gap explanation (Liang et al., 2023)"。

**M5. P16 · 自曝"阈值在选择它的同一份数据上评估"，且该页叙事位置断裂**
- 位置：第 1 条 "Thresholds were selected and evaluated on the same data."；页脚 "PART 04 / REPAIR EXPERIMENTS"
- 问题：(a) 这是 in-sample 评估的方法论软肋，虽然诚实但页面上没给任何补救说明（哪怕一句 "we acknowledge this limitation"），评委必问；(b) 修复实验的内容被拆在 P10（Part 3?）和 P16（Part 4）两处，中间隔着 Recommendations 和 Challenges，叙事被打断——P14 排在 P16 前不合逻辑。
- 修法：P16 挪到 P11 之后、P14 之前；该句后补半句 "so the parity gains are optimistic estimates."

**M6. P11 · "220 essays per run" 来源不明，测试集规模缺失**
- 位置：右侧 "WHAT WE DID"
- 问题：220 从哪来（P6 的 179？394 的子集？）没说；"tested on essays it had never seen" 没给测试集 n 和构成。训练/测试划分数据描述缺失，和已知 P6 缺 split 是不同位置的新缺口。
- 修法：改为 "Fine-tuned RoBERTa on 220 essays (<构成一句话>), tested on <n> held-out essays (<构成>)"。

**M7. P9 · "7 of 7 detectors gave lower scores on average" 无显著性检验支撑**
- 位置：右侧两段结论文字
- 问题：箱线图里 Sapling 的中位数几乎贴 0、多个人检测器的 IQR 跨越 0，"on average" 的结论全靠点估计；没有 paired t-test/Wilcoxon，"7 of 7" 会被追问 direction 是否稳定于显著性。属于"结论超前于证据"。
- 修法：图注加一行配对检验结果（如 Wilcoxon signed-rank p 值），或口播明确承认只报均值。

**M8. P6 · 394 篇 human originals 与 179 篇 GPT-edited 的对应关系没交代**
- 位置：右侧三条数据条 + "91 TOEFL and 88 Hewlett originals"
- 问题：91+88=179 恰好等于 GPT-edited 条数，说明只有 TOEFL+Hewlett 有编辑版本，但页面上 394 和 179 的关系要听众自己算；评委容易误以为数据描述不清。与已知"缺 train/test split"是不同的问题。
- 修法：179 条后加注 "(edited counterparts of the 91 TOEFL + 88 Hewlett originals)"。

---

## Low

**L1. P7/P8/P9 · 标题序号 "1.TOEFL / 2.Text / 3.GPT-4" 缺空格** — 三页一致的排版小错。改为 "1. " 即可。

**L2. P7 · "[n/a = detector file absent]" 是内部黑话** — 图题里的括号注释讲的是"文件缺失"而非"该检测器不适用该 corpus"，评委看不懂且会问数据完整性；行标签 CollegeEssay__gpt3_31 等内部文件名同样晦涩。改为 "n/a = score file not provided in the benchmark"。

**L3. P1 · 主标题框 "DETECTING THE DETECTOR:" 冒号悬空** — 冒号后面没有内容（副标题在框外另一行），首屏像个截断。去掉冒号。

**L4. P18 · 参考文献排序与拼接问题** — (a) 列表未按字母序：Cohen 排在 Liang 之后；(b) Sadasivan 条目 "arXiv:https://doi.org/..." 前缀拼接错误，是引用工具直出的痕迹。

**L5. P9 · 81 对匹配 vs 91 篇 TOEFL originals，10 篇去向未说明** — 口播备一句即可（"9 篇无合格匹配"）。

**L6. P9 · 图 (a) 图例 "low-confidence mark = 0.2" 术语未定义** — 红虚线什么含义（低于此相似度视为低置信匹配？）现场没人能看懂。

---

## 投影可读性（80dpi 已可见的衰减）

- **P7 双热图**：格子数字和 45° 旋转的检测器名在 80dpi 已偏小，后排基本读不到具体数值，只剩色块——口播必须点名 52→76 和 12 这几个数，不能指望观众看图。
- **P9 (b) 箱线图**：纵轴标题 "paired score delta (polished − real)" 字号全场最小，投影后不可读；而这正是本页唯一解释"负值=变好"的元素。
- **P8 Figure 4**：特征名（type_token_ratio 等）为小号等宽字，80dpi 下勉强可辨，投影必糊；好在本页有结论句兜底。

## 时间风险（20 分钟 / 18 页 ≈ 67 秒每页）

- **P7**：双热图 + 平均数 + 阈值说明，讲透需 2.5 分钟。建议口播只保留"TOEFL 行 vs 其他三行"一个对比。
- **P9**：两个子图 + 3 组数字（81 对、7/7、33/41）+ 已知残留图注，信息过载。建议砍掉 (a) 匹配质量直方图的现场讲解，只留一句带过。
- **P10**：三标准 × 三方法柱状图 + 两段文字框，Platt "6 通过 equalized opportunity 却 0 通过 calibration" 的反直觉点需要 1.5 分钟以上，且与 P16 内容重叠——两页合并口播。
- **P12**：见 H5，文字块数字过密，务必口播简化为"错误没有消失，只是转移"。

## RQ 对答案总账（评委视角）

| RQ | 承诺位置 | 回答位置 | 状态 |
|----|---------|---------|------|
| RQ1 | P4/P5 | P7 | ✅ 完整 |
| RQ2 (predictability) | P4/P5 | P8 | ✅ 完整 |
| RQ2 (editing) | P5 | P9 | ✅ 但图注错标 RQ4（H2） |
| RQ3 thresholds/calibration | P4/P5 | P10、P16 | ✅（含 in-sample 软肋 M5） |
| RQ3 ensembles | P4/P5 | **无** | ❌ 缺口（H3） |
| RQ3 自训检测器 | P4/P5 | P11、P12 | ✅ 但指标术语存疑（H4） |
