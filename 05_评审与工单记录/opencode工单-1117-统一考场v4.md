# opencode 工单 · 1117 · 统一考场 v4(一晚上跑完的完整版)

项目根目录(本机):`~/Documents/课程-2026秋/INT4096-AI专题/小组项目-DetectingTheDetector/`
签发:Claude 主会话 2026-11-17。验收由主会话执行,不采信自报。

**本单取代**(这些文件已改名加"已作废-"前缀,不要再执行):统一考场 v3、开源检测器审计、Qwen v2 重训、QWEN_V2_CELL 退回单。`colab/QWEN_V2_CELL.py` 作废。
**仍然有效、另行执行**:`opencode工单-1117-真实作文重灾区演示.md`(本机 CPU,与本单互不依赖;本单 §8.3 会复用它的阶段 1 复现结果,没有的话自己复现)。

---

## 0. 一句话目标
在**同一套考卷**上,比较三类检测器对非母语作文的冤枉程度,同时确认它们还能抓住 AI:
- 起点:7 个商用检测器(2023 年的现成分数);
- 现成开源:Binoculars、Fast-DetectGPT(2024 顶会,只打分不训练);
- 我们自己训的:经典 ML、RoBERTa、Qwen 27B,每种分"臂 A / 臂 B"两份训练数据(只差人写部分有没有非母语)。

## 1. 红线
1. 主线审计、notebook、`results/` 已有文件、`data/` 一律不改。代码放本机 `unified_v4/` 并同步到服务器 `/data/int4096_unified_v4/`;结果写回本机 `results/unified_v4/`。
2. **考卷里的任何文本都不得进训练或验证**(§4 去重闸门)。
3. 不调超参。唯一允许"选"的是 §7 的阈值,只能在验证集上选,绝不能在考卷上选。开源检测器只用官方阈值。
4. 远端杀进程只用显式 PID,禁止 `pkill -f` / `pgrep -f`(会杀掉自己的 ssh shell);存活判据用 pidfile + `kill -0`。
5. 任何数据源拿不到、篇数对不上、字段含义不确定:**跳过该项并写进报告**,不要换别的数据凑。
6. **执行者自身不能依赖要停掉的服务**(§9 第 0 步)。

## 2. 考卷(全部只测)
| # | 考卷 | 内容 | 主要用途 |
|---|---|---|---|
| E1 | **Liang 925**(本机已有 `data/zenodo/Weixin-Liang-ChatGPT-Detector-Bias-8cc05ff/Data_and_Results`) | `TOEFL_real_91`(非母语)、`HewlettStudentEssay_real_88`(母语)、`CollegeEssay_real_70`、`CS224N_real_145`;AI 352 篇;改写文本 179 篇 | **与起点直接比较**。TOEFL−Hewlett 是**决定性指标**。CollegeEssay/CS224N 单列,不并入"母语"(作者母语身份不确定)。改写文本单列,不叫误判 |
| E2 | **PERSUADE 2.0 同题考卷**(§3.1) | 同题、同年级的 ELL vs 非 ELL 学生作文,各最多 1,000 篇 | 排除"题目不同"干扰的最干净对比 |
| E3 | **W&I + LOCNESS**(BEA-2019,§3.2) | W&I:全球非母语学习者作文,带 CEFR 等级 A/B/C;LOCNESS:英美大学母语学生作文 | 与训练数据**完全不同源**的第二组"非母语 vs 母语";按 CEFR 等级画熟练度曲线 |
| E4 | **FCE v2.1**(BEA-2019,§3.2) | 1,244 份剑桥 FCE 考试答卷(非母语,2000 年代,带分数) | 不同源、ChatGPT 之前写的非母语文本;按分数画熟练度曲线 |
| E5 | **ELLIPSE 测试 2,911** | 8–12 年级 ELL 作文,带 6 项熟练度分数 | 大样本误判率(**次要**:与臂 B 训练数据同源);熟练度曲线 |
| E6 | **现代 AI 作文** | daigt 留出 AI 400 篇 + GLM 现场生成 300 篇(§3.3) | 检查"还能不能抓住**现在的** AI",不只是 2023 年的 GPT-3 |

**解读优先级写死**:E1 的 TOEFL−Hewlett 第一;E2、E3 第二(独立复核);E4、E5 第三。不允许只挑好看的考卷讲。

## 3. 数据来源与核实(阶段 0,结果写进报告第 1 节)
每个来源都记录 URL、版本/commit/revision、许可证、下载文件 sha256、实际篇数。

### 3.1 PERSUADE 2.0
- `github.com/scrosseye/persuade_corpus_2.0`。README(主会话 1117 已读)写明:25,000+ 篇 6–12 年级议论文,15 个题目,每位作者有人口学信息,CC BY 4.0;CSV 在 README 的 Google Drive 公开链接(测试集 zip 密码 `persuade_test`,README 原文公开)。
- **ELL 列的列名和取值未核实**。没有 ELL 列或下载需要登录:跳过 E2,人写训练池退回 daigt 人写部分,并写明。
- 有 ELL 列时:非 ELL 作文作为**人写训练池**;按题目分层留出 ELL ≤1,000 + 非 ELL 1,000 作为 E2(ELL 不够就全部留出)。

### 3.2 W&I+LOCNESS v2.1 与 FCE v2.1
- 页面 `https://www.cl.cam.ac.uk/research/nl/bea2019st/`(主会话 1117 已核实):两者均"immediately downloadable",文件 `data/wi+locness_v2.1.bea19.tar.gz`、`data/fce_v2.1.bea19.tar.gz`;许可证仅限非商业用途(课程作业可用,报告里注明)。
- **必须用学习者的原始文本(纠错前、未分词)**。先看压缩包里有没有含原文的 JSON;如果只有 M2 格式(已分词、标点前有空格),**不要自己拼接还原**,跳过该语料并写明——分词后的文本会让检测器打分失真。
- W&I 按 CEFR 等级(A/B/C)标注;LOCNESS 是母语。记录 W&I 各等级篇数、LOCNESS 篇数。**LOCNESS 篇数可能很少**,少于 50 篇就只报数不算差距。
- 只用 train/dev 里公开带原文的部分;test 部分没有标注的跳过。

### 3.3 现代 AI 作文(**停服之前**生成,要用到 GLM)
- 用 .138 现役 GLM(前门 1234,本机 OpenCode 配置里的 `mochu-glm` key),给 PERSUADE 的 15 个题目各生成 20 篇约 400 词的议论文,共 300 篇。temperature 0.7,关闭思考链或只取正文。prompt 模板、模型名、参数全部记录。
- 生成作文只用于 E6 考卷,不进训练。
- GLM 是在线生产服务:**每次最多 2 路并发**,不要压满。

### 3.4 其余
- ELLIPSE:HF `tcapelle/feedback-prize-english-language-learning-fluency`,revision `9a7a03c8b0ee2401b7e33d97f8689ae94c9ccaf7`,3,911 篇;seed=0 固定抽 1,000 篇训练,其余 2,911 篇为 E5。
- daigt:HF `ramensoft/daigt_v4`,revision `16457627f531a21640a2f6c723dadaf818f94578`,文件 `daigt_magic_generations.csv`(sha256 `b0ab2d7a…d3e5`)。已核实全部为 AI(见 §5)。
- PERSUADE 2.0:主会话 1117 已在本机下载(Google Drive 在服务器上不通),合并成每篇一行的 `persuade_essays.csv`(25,996 篇,sha256 `85647e74…d0df`),上传到 `/data/int4096_unified_v4/data/persuade/`;`ell_status` 列已核实存在。
- W&I+LOCNESS、FCE:已下载并核实 `json/` 下有纠错前原文(`text` 字段)。W&I:A 1,430 / B 1,100 / C 770(train+dev);LOCNESS:**仅 50 篇**(N.dev);FCE:约 2,469 份答卷,带 `answer-s` 分数、`l1` 母语、`age`。
- Binoculars:`github.com/ahans30/Binoculars`;Fast-DetectGPT:`github.com/baoguangsheng/fast-detect-gpt`。地址未经本次核实。记录 commit、底座模型、**官方阈值**及出处(README 行或论文页码)。
- Binoculars 论文(arXiv 2401.12070,ICML 2024)§5 已由主会话核实:它测的是 EssayForum ESL 作文的**原文 vs 语法修正版**(准确率都是 99.67%),**没有做非母语 vs 母语对比,也没有用 Liang 的数据**。报告里照此描述,不要扩大或缩小。

## 4. 去重闸门(切分前做)
文本规范化(连续空白压成一个、去首尾空白)。所有训练池、验证池中与**任何考卷文本(E1–E6 全部)以及 ELLIPSE 全部 3,911 篇**全文相同或前 200 字符相同的条目,一律剔除并记录剔除数。PERSUADE 与 ELLIPSE 可能有重复作文,E2 里与 ELLIPSE 重复的也要剔除。切分完成后再查一遍"训练/验证 vs 考卷"的重叠,**必须为 0**,否则停。

## 5. 训练与验证数据(1117 主会话修订,已按实测数据改)
**实测事实**(主会话 1117 在服务器上核实):
- daigt_v4(ramensoft)= **8,306 篇,全部是 AI**(label 全为 1,来源 `persuade_finetuned_llamas`,模型 llama,15 个 PERSUADE 题目)。**没有人写作文**,不能当人写池用。
- PERSUADE 2.0 合计 25,996 篇:ELL=Yes 2,244、ELL=No 22,451、其余缺失。**其中 452 篇 ELL 作文与 ELLIPSE 是同一批文章**(前 200 字符规范化后相同);去掉后,PERSUADE ELL 可用 1,792 篇(14 个题目)。

**池子**:人写-母语 = PERSUADE ELL=No(去重后);人写-非母语 = ELLIPSE 训练份 + PERSUADE ELL(不在 ELLIPSE 中、不在 E2 中);AI = daigt。ELL 缺失的一律不用。
**E2 同题考卷**:PERSUADE ELL(不含与 ELLIPSE 重复者)按题目分层抽 792 篇 + 同题目分布的非 ELL 800 篇。剩下 1,000 篇 ELL 进训练池。
**E5**:ELLIPSE 用 seed=0 抽 1,000 篇进训练池,其余 2,911 篇为考卷(与 PERSUADE 重复的 452 篇若落在考卷里照常保留;它们不会出现在训练里,因为 PERSUADE ELL 训练池已排除与 ELLIPSE 重复者)。

- **验证集**(两臂、所有 seed 共用):PERSUADE 非 ELL 300 + AI 300,只用来定 §7 阈值 ②,并报告同分布误判率。
- **主训练集**(每个 seed 从剩余池子重新抽,非母语 2,000 篇固定为全部可用):

| | 人写 4,000 | AI 4,000(daigt) |
|---|---|---|
| 臂 A | PERSUADE 非 ELL 4,000 | 本 seed 抽的 4,000 |
| 臂 B | 臂 A 那 4,000 篇中的**前 2,000** + ELLIPSE 训练 1,000 + PERSUADE ELL 训练 1,000 | **与臂 A 完全相同** |

- **学习曲线**(只对经典 ML 和 RoBERTa,5 个 seed):在同样的设计下,人写/AI 各取 1,000、2,000、4,000(即主训练集的嵌套子集,臂 B 中非母语始终占人写的一半),检验结论会不会随数据量改变。Qwen 只跑 4,000 规模。
- Liang 的 AI 作文、TOEFL、Hewlett 等**一律不进训练**。所有切分存成列表文件并带 sha256。

**27B 权重**:不下载。主会话已把 `/data/qwen38fn/model`(Qwen3.8-27B FP8,e4m3,128×128 块缩放)还原为 bf16,存于 `/data/int4096_unified_v4/qwen27b_bf16_from_fp8`(91 秒,52GB,1,199 个张量);与 9 月残留的原版 bf16 分片 `model-00007-of-00018` 逐张量对比:17 个 FP8 还原张量相对误差 2.66%,13 个未量化张量误差 0,检查 PASS(日志 `logs/check_dequant.log`)。报告局限里注明"权重经 FP8 量化再还原,约 2.7% 权重误差"。
**训练环境**:`/data/int4096_unified_v4/venv`(torch 2.9.1+cu128、transformers 5.22.2、peft 0.21.3、scikit-learn 1.9.1)。

## 6. 模型
| 类别 | 模型 | 设置(固定不调) | seed 数 | 设备 |
|---|---|---|---|---|
| 经典 ML | TF-IDF + 逻辑回归 | 同 notebook:`TfidfVectorizer(ngram_range=(1,2), max_features=20000, min_df=2)` + `LogisticRegression(max_iter=2000)` | 5 | CPU,不需停服 |
| DL | `roberta-base` 全参微调 | lr 2e-5、batch 16、max_len 512、2 epoch(原 4 epoch/220 篇;数据约 18 倍,改 2 epoch,写明)、`DataCollatorWithPadding` | 5 | 每卡 ≤2 个任务 |
| 大模型 | `Qwen/Qwen3.8-27B` + LoRA(SEQ_CLS) | 与 v1 相同:r=16、alpha=32、dropout 0.05、all-linear、bf16、max_len 512、lr 1e-4、batch 8 × grad_accum 2、gradient checkpointing;1 epoch | 2 | 每任务 2 张卡(`device_map="auto"`,每卡留 2GB) |
| 开源检测器 | Binoculars、Fast-DetectGPT | 官方默认底座和官方阈值,只打分;统一截断长度(写明) | — | 每个 1–2 张卡 |

## 7. 评估与指标
- 每个"检测器(× 臂 × seed)"对 E1–E6 和验证集**逐篇**输出 AI 分数,存 CSV(`exam, corpus, essay_id, score`;有熟练度/等级的一并带上)。
- **主比较口径:两个不依赖阈值的 AUC**(1117 主会话修订;与起点比较以此为准):
  - **偏见 AUC** = P(随机一篇 TOEFL 作文的 AI 分数 > 随机一篇 Hewlett 作文的分数)。0.5 = 无偏见,越接近 1 越偏向冤枉非母语。同样算 E2(ELL vs 非 ELL)、E3(W&I vs LOCNESS)。
  - **检出 AUC** = 人写(TOEFL + Hewlett)vs Liang AI 352,另报 vs E6。
  - 两者都带作文级 bootstrap 95% 区间。**7 个商用检测器逐个用现有分数计算这两个 AUC**(不只报平均),作为表中起点行。目标表述:偏见 AUC 比商用检测器更接近 0.5,且检出 AUC 不明显更低。
- **阈值 ① 固定 0.5**(开源检测器用官方阈值)。
- **阈值 ② 同检出率**(**误判率的主口径**;AUC 仍是总主口径):在验证集 300 篇 AI 上把阈值定到恰好抓住 90% AI,再去考卷算误判率。防止某个模型只是变"松"才显得公平。商用检测器没有验证集,只报阈值 ①。开源检测器两种都报(② 用同一份验证集)。
- 每张考卷:被判 AI 比例 + Wilson 95% 区间;多 seed 报均值 ± 标准差。
- 差距:E1 TOEFL−Hewlett(**主**)、E2 ELL−非ELL、E3 W&I−LOCNESS、E5 ELLIPSE−Hewlett。
- 检出率:E1 的 AI 352(老 AI)和 E6(现代 AI)分开报。
- AUC:E1 人写(TOEFL + Hewlett)vs E1 AI 352;另报 vs E6。
- **显著性**:同一模型的臂 A 与臂 B 在**同一批 91 篇 TOEFL**上配对比较:每个 seed 做 McNemar 检验,同时对作文 bootstrap 1,000 次求"臂 B − 臂 A 误判率"的 95% 区间。区间跨过 0 只能写"差别不显著"。E2、E3 同样做。
- **熟练度曲线**:E3(CEFR A/B/C)、E4(FCE 分数分 3 档)、E5(ELLIPSE 总分分 3–5 档),每个检测器画"熟练度档位 → 误判率"折线(带区间)。
- **基线行**:商用 7 个(用现有 `results/` 分数,只有 E1)、全判人写、随机猜。

## 8. 报告产物(本机 `results/unified_v4/`)
1. `summary_threshold05.csv`、`summary_tpr90.csv`:一行一个检测器(起点 → 开源 → 我们的 3×2),列 = 各考卷误判率、两种 AI 检出率、各差距、AUC。另出一张 16:9 英文表格图 `summary_table.png`,可直接放 slides。
2. `proficiency_curves.png`:熟练度曲线,16:9 英文。
3. **风险模型外部验证**(原开源审计工单阶段 4,并入本单,CPU):用 notebook 同款风险模型 (a) TF-IDF+8 特征、(b) 仅 8 特征、(c) 仅词数,在 Liang 人写 394 × 7 商用检测器上全量训练、不调参;给 E5 ELLIPSE 2,911 篇打"被冤枉风险分",以 Binoculars、Fast-DetectGPT 按官方阈值的误判为真值,算 AUC(bootstrap 区间)。某检测器在 E5 误判数 < 20 就写"正例太少"。(a)(b) 都不比 (c) 好要明说。
4. `REPORT.md`:第 0 步自检结论;各数据源核实结果(含被跳过的及原因);去重剔除数与最终重叠数;两张总表;**按 §2 的优先级顺序解读**;显著性结果;熟练度曲线说明;风险模型外部验证;GPU、耗时、版本、可复制的重跑命令;所有数字标"请独立复核"并注明出自哪次运行。
5. **局限**(必须写):各考卷的作者年龄段、文体、题目不同(TOEFL 成人考生;PERSUADE/ELLIPSE 中学生;W&I 含书信和故事;FCE 是考试答卷),差距不能全归因于母语背景;若 E2 跳过,"母语"人写池只能称"以母语为主";ELLIPSE 训练与测试同源;Qwen 只有 2 个 seed;开源检测器结果依赖其底座模型。

## 9. A40 执行流程(111.48.160.138,8×A40 46GB,业主已选择"暂停生产全上 A40")
**第 0 步 · 执行者自检(最先做)**:确认 opencode 当前使用的模型**不是** .138 上的 GLM(provider `mochu-glm`,前门 1234)。是的话先切换到其他 provider;否则一停服,执行者自己就断了。

**第 1 步 · 不停服就能做的准备(生产照常)**
- 一切放 `/data/int4096_unified_v4/`,不占根盘。新建 venv,版本写进报告。HF 直连被墙,统一 `export HF_ENDPOINT=https://hf-mirror.com`。该机入口约 12MB/s,27B 权重约 54GB:**最先开始下载**;下载前先在 `/data` 查有没有现成的 Qwen3.8-27B bf16 权重(该机曾跑过),有就核对完整性后复用。Binoculars/Fast-DetectGPT 的底座权重同样提前下好。
- 下载并核实全部数据源(§3),完成去重和切分(§4、§5)。
- **生成 E6 现代 AI 作文**(§3.3,必须在停服前做)。
- 经典 ML 全部在 CPU 上跑完,风险模型外部验证的 (c) 词数基线也可以先算。
- RoBERTa、Qwen、开源检测器的代码先用 10 篇样本走通数据管线(不把 27B 加载到 GPU)。

**第 2 步 · 停服前快照**(存到 `/data/int4096_unified_v4/prod_snapshot/`)
- `bash /data/glm_dflash2/sglang_serve.sh status`;运行中 SGLang 进程的完整命令行与环境变量(从 `/proc/<pid>/cmdline`、`/proc/<pid>/environ` **现读**);`crontab -l` 全文备份;`nvidia-smi` 全文;所有 GPU 进程的 PID 与所属程序。
- 发现不属于 GLM 的 GPU 进程(别人的任务):不要动,停下报告。

**第 3 步 · 停服**
- 先在 crontab 里注释掉 GLM 看门狗和监测项(`/data/glm_test/monitor/sg_watchdog.py`、`sglang_monitor.py` 等)。**不注释的话,它会在停服后自动把 GLM 拉起来,和训练抢卡。**
- `bash /data/glm_dflash2/sglang_serve.sh stop`;确认 8 张卡显存都已释放。记录停服时间。

**第 4 步 · GPU 任务(三波,全部 nohup 后台 + pidfile + 轮询)**
- 第 1 波:RoBERTa 10 个任务(每卡 ≤2 个)+ Binoculars + Fast-DetectGPT 打分,填满 8 张卡。
- 第 2 波:Qwen 4 个任务(2 臂 × 2 seed),每个 2 张卡,8 张卡全用,并行。
- **时间闸门**:Qwen 跑满 20 步后,按实测速度估算训练 + 考卷推理的结束时间。若**预计总停服时间超过 8 小时**,停下汇报,等主会话决定(选项:只留 seed 0,或另找时间)。
- 任一任务失败:记录日志,其余任务继续;不要为了重跑而延长停服超过闸门。

**第 5 步 · 恢复生产(训练结束或任何原因中止,都必须执行)**
- 按第 2 步快照里**现读到的**命令原样启动(参考:1115 的启动方式是 `EXTRA_ARGS="--enable-hierarchical-cache --hicache-ratio 3 --enable-metrics" bash /data/glm_dflash2/sglang_serve.sh start`,以快照为准)。
- READY 后经前门 1234 发一个真实生成请求,确认正常输出;用备份恢复 crontab,`crontab -l` 与备份 diff 必须为空。
- 记录恢复时间与总停服时长,写入 `results/unified_v4/prod_downtime.md`(含快照、恢复后验证输出、crontab diff)。
