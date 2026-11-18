# opencode 工单 · 1117 · 统一考场 v3(经典 ML / RoBERTa / Qwen 27B × 臂 A/B,A40 整机)

项目根目录(本机):`~/Documents/课程-2026秋/INT4096-AI专题/小组项目-DetectingTheDetector/`
签发:Claude 主会话 2026-11-17。验收由主会话执行,不采信自报。
**本单取代**:`opencode工单-1117-Qwen-v2重训.md` 和 `退回修改-QWEN_V2_CELL-1117.md`。`colab/QWEN_V2_CELL.py` 作废,不再使用。
**不受影响、照常执行**:`opencode工单-1117-真实作文重灾区演示.md`、`opencode工单-1117-开源检测器审计.md`。

---

## 0. 一句话目标
三种模型(经典 ML、RoBERTa、Qwen 27B),每种用两份只差一个变量的训练数据(臂 A 人写部分不含非母语;臂 B 人写部分一半换成非母语),全部在**同一个考场**上考试,跟起始基线(7 个商用检测器)放进同一张表比较。

## 1. 红线
1. 主线审计、notebook、`results/` 已有文件、`data/` 一律不改。新代码放 `unified_v3/`(本机项目根下新建,并同步到服务器 `/data/int4096_unified_v3/`),结果写回本机 `results/unified_v3/`。
2. **考场里的任何文本都不得进训练或验证**(见 §3 去重闸门)。
3. 不调超参;唯一允许"选"的是 §6 的阈值,而且只能用验证集选,绝不能用考场选。
4. 远端杀进程只用显式 PID,禁止 `pkill -f` / `pgrep -f`(会杀掉自己的 ssh shell);存活判据用 pidfile + `kill -0`。
5. 不确定就停下写进报告,不要猜。
6. **执行者自身不能依赖要停掉的服务**:见 §7 第 0 步。

## 2. 考场(只测,所有模型同一套)
| 考场 | 内容 | 算什么 |
|---|---|---|
| 主考场 · Liang 925 | `TOEFL_real_91`(非母语) | 误判率,**决定性指标** |
| | `HewlettStudentEssay_real_88`(母语) | 误判率,**主对照** |
| | `CollegeEssay_real_70`、`CS224N_real_145` | 误判率,**单列**,不并入"母语"(作者母语身份不确定) |
| | AI 352 篇(`CS224N_gpt3_145`、`CS224N_gpt3PromptEng_145`、`CollegeEssay_gpt3_31`、`CollegeEssay_gpt3PromptEng_31`) | 检出率 |
| | `TOEFL_gpt4polished_91`、`HewlettStudentEssay_GPTsimplify_88` | "改写后被判 AI 比例",单列,**不叫误判** |
| 附加考场 · ELLIPSE 测试 2,911 | 非母语学生作文 | 大样本误判率,**次要指标**(与臂 B 训练数据同源同题,改善可能只是"见过同类题") |
| **同题考场 · PERSUADE 2.0 留出**(见 §3.1) | 同一批题目、同年级的 ELL 学生作文 vs 非 ELL 学生作文 | 误判率差距。**题目和年级都相同**,排除了"不同语料题目不同"这个干扰 |

数据来源:Liang 仓库 `Weixin-Liang/ChatGPT-Detector-Bias` 的 `Data_and_Results/`(本机已有 `data/zenodo/Weixin-Liang-ChatGPT-Detector-Bias-8cc05ff/Data_and_Results`);ELLIPSE 为 HF `tcapelle/feedback-prize-english-language-learning-fluency`(revision `9a7a03c8b0ee2401b7e33d97f8689ae94c9ccaf7`,3,911 篇)。各语料篇数与上表不符就停。

## 3. 训练与验证数据

### 3.1 PERSUADE 2.0(带 ELL 身份标签,1117 新增)
- 来源:`github.com/scrosseye/persuade_corpus_2.0`,README 写明 25,000+ 篇 6–12 年级议论文、15 个题目、每位作者有人口学信息,CC BY 4.0;CSV 放在 README 里的 Google Drive 公开链接(训练集 / 测试集,测试集 zip 密码 `persuade_test`,README 原文公开)。主会话 1117 只读过 README,**ELL 标签的列名和取值未核实**。
- 阶段 0 必须先确认:有没有 ELL 状态列、列名、取值、ELL 与非 ELL 各多少篇。**没有 ELL 列、或下载需要登录,就跳过 §3.1 全部内容**,退回原方案(人写池用 daigt 的人写部分),并在报告里写明。
- 有 ELL 列时:
  1. **人写训练池改用 PERSUADE 2.0 的非 ELL 作文**(不再用 daigt 人写部分),"母语"从"以母语为主"变成"非 ELL"。daigt 只用来取 AI 作文。
  2. **同题考场**:按题目分层,留出 ELL 作文最多 1,000 篇 + 非 ELL 作文 1,000 篇,只测不训(ELL 不够 1,000 就全部留出,写明实际篇数)。
  3. PERSUADE 与 ELLIPSE 可能有重复作文:同题考场里与 ELLIPSE 3,911 篇重复的条目一律剔除;非 ELL 训练池同样要和全部考场去重。
池子:
- **PERSUADE 人写** 与 **AI 作文**:HF `ramensoft/daigt_v4`(revision `16457627f531a21640a2f6c723dadaf818f94578`)。先打印每个 label 各 2 篇的前 300 字符,人工确认 0=人写、1=AI,写进日志后再继续。
- **ELLIPSE 训练**:ELLIPSE 3,911 篇用 seed=0 固定抽 1,000 篇,其余 2,911 篇即考场。

**去重闸门(切分前做)**:文本规范化(连续空白压成一个空格、去首尾空白)后,把 daigt 两个池子里与**全部考场文本 + 全部 ELLIPSE 3,911 篇**全文相同或前 200 字符相同的条目剔除,记录剔除数。切分后再查一遍"训练/验证 vs 考场"的重叠,必须为 0,否则停。

**验证集(两臂共用同一份,所有 seed 共用)**:从过滤后的池子固定抽 PERSUADE 人写 300 + AI 300。只用于 §6 选阈值和报告"同分布误判率"。

**训练集(每个 seed 从剩余池子重新抽)**:
| | 人写 2,000 | AI 2,000 |
|---|---|---|
| 臂 A | PERSUADE 2,000 | 本 seed 抽的 2,000 |
| 臂 B | 臂 A 那 2,000 篇 PERSUADE 中的**前 1,000** + ELLIPSE 训练 1,000 | **与臂 A 完全相同**的 2,000 |

两臂规模、比例、AI 部分完全相同,唯一差别是人写那一半里有没有非母语。ELLIPSE 训练只有 1,000 篇,所以各 seed 共用这 1,000 篇,报告里写明。
所有切分存成 id/文本列表文件,带 sha256。

## 4. 三种模型(超参固定,不调)
| 层级 | 模型 | 设置 | seed 数 |
|---|---|---|---|
| 经典 ML | TF-IDF + 逻辑回归 | 与 notebook 相同:`TfidfVectorizer(ngram_range=(1,2), max_features=20000, min_df=2)` + `LogisticRegression(max_iter=2000)`,CPU | 5 |
| DL | `roberta-base` 全参微调 | lr 2e-5、batch 16、max_len 512、**2 epoch**(v0 为 4 epoch/220 篇;数据量约 18 倍,改 2 epoch,写明)、`DataCollatorWithPadding` | 5 |
| 大模型 | `Qwen/Qwen3.8-27B` + LoRA(SEQ_CLS) | 与 v1 相同:r=16、alpha=32、dropout 0.05、all-linear、bf16、max_len 512、lr 1e-4、batch 8 × grad_accum 2、gradient checkpointing;**1 epoch** | 2 |

Qwen 27B bf16 约 54GB,单张 A40(46GB)放不下:每个 Qwen 任务占 **2 张卡**(`device_map="auto"` + `max_memory` 每卡留 2GB 余量)。

## 5. 评估
每个"模型 × 臂 × seed"对 §2 全部考场和验证集逐篇输出 AI 概率,存 CSV(`corpus, essay_id, prob`)。

## 6. 指标(两种阈值都报)
- **阈值 ① 固定 0.5**。
- **阈值 ② 同检出率**:在验证集 300 篇 AI 上,把阈值定在"恰好抓住 90% AI"的位置。所有模型都按这个标准定阈值,再去考场算误判率。**这是主比较口径**,防止某个模型只是变"松"了才显得公平。
- 每个考场:被判 AI 的比例 + Wilson 95% 区间;多 seed 报均值 ± 标准差。
- 差距:TOEFL − Hewlett(**主指标**)、ELLIPSE − Hewlett、PERSUADE 同题 ELL − 非 ELL(若 §3.1 可用)。
- **显著性(因为 TOEFL 91 / Hewlett 88 样本小,必须做)**:同一个模型的臂 A 与臂 B 在**同一批 91 篇 TOEFL**上做配对比较:每个 seed 做 McNemar 检验,同时对作文做 bootstrap(1,000 次)求"臂 B − 臂 A 误判率"的 95% 区间。区间跨过 0 就要写"差别不显著",不准写成"改善"。
- AUC:Liang 人写(TOEFL + Hewlett)vs AI 352。
- **基线行**(同一张表最上面):
  - 7 个商用检测器:用现有 `results/` 里的分数,按 0.5 阈值算同一组考场指标(含 AI 352 的检出率)。它们没有验证集,所以只报阈值 ①,表里写明。
  - 全判人写、随机猜。
- 开源检测器(Binoculars、Fast-DetectGPT)的结果来自另一份工单,汇总表里留两行占位。

## 7. A40 执行流程(服务器 111.48.160.138,8×A40 46GB)
**第 0 步 · 执行者自检(必须最先做)**:确认 opencode 当前使用的模型**不是** .138 上的 GLM(本机 OpenCode 配置里名为 `mochu-glm` 的 provider,入口 glm-front:1234)。如果是,先切换到其他 provider 再继续;否则一停服务,执行者自己就断了。写进报告。

**第 1 步 · 不停服就能做的准备(生产照常运行)**
- 环境、权重、数据一律放 `/data/int4096_unified_v3/`,根盘不放东西。新建 venv,版本写进报告。
- HF 直连被墙,统一用 `export HF_ENDPOINT=https://hf-mirror.com`。该机入口带宽实测约 12MB/s,27B 权重约 54GB,预计 1 小时以上,**最先开始下载**。下载前先在 `/data` 下查有没有现成的 Qwen3.8-27B bf16 权重(该机曾跑过 27B BF16),有的话核对完整性后直接用。
- 经典 ML 5 个 seed × 2 臂全部在 CPU 上跑完(不需要停服)。
- RoBERTa、Qwen 的代码先用 10 篇样本在 CPU 或 meta 设备上走通数据管线(不加载 27B 权重到 GPU)。

**第 2 步 · 停服前快照**(全部存到 `/data/int4096_unified_v3/prod_snapshot/`)
- `bash /data/glm_dflash2/sglang_serve.sh status` 的输出;运行中 SGLang 进程的完整启动命令行与环境变量(从 `/proc/<pid>/cmdline`、`/proc/<pid>/environ` 现读,不要照抄旧记录);
- `crontab -l` 全文备份;`nvidia-smi` 全文;所有占用 GPU 的进程 PID 与所属程序。
- 若发现**不属于 GLM 的** GPU 进程(别人的任务),不要动它,停下报告。

**第 3 步 · 停服**
- 先在 crontab 里注释掉 GLM 看门狗与监测项(`/data/glm_test/monitor/sg_watchdog.py`、`sglang_monitor.py` 等)。**不注释的话,它会在停服后自动把 GLM 拉起来,和训练抢卡。**
- `bash /data/glm_dflash2/sglang_serve.sh stop`;复核 8 张卡显存都已释放,存活检查按显式 PID。
- 记录停服时间。

**第 4 步 · 训练(两波)**
- 第 1 波:RoBERTa 10 个任务(2 臂 × 5 seed),每张卡同时最多 2 个,8 张卡并行。
- 第 2 波:Qwen 4 个任务(2 臂 × 2 seed),每个占 2 张卡,8 张卡全部用上,并行。
- **时间闸门**:Qwen 跑满 20 步后,按实测 step 耗时估算结束时间。若**预计总停服时间超过 6 小时**,停下汇报,等主会话决定(可选方案:只保留 seed 0,或改天再跑)。
- 所有长任务 nohup 后台运行,写 pidfile,轮询日志。

**第 5 步 · 恢复生产(训练结束或任何异常中止都必须执行)**
- 按第 2 步快照里**现读到的**启动命令原样启动(参考:1115 时的启动方式是 `EXTRA_ARGS="--enable-hierarchical-cache --hicache-ratio 3 --enable-metrics" bash /data/glm_dflash2/sglang_serve.sh start`,以快照为准)。
- 等 READY 后,经前门 1234 发一个真实生成请求,确认有正常输出;再用 `crontab` 备份恢复看门狗和监测项,`crontab -l` 与备份做 diff,结果必须为空。
- 记录恢复时间、总停服时长。

## 8. 交付
本机 `results/unified_v3/`:
- `splits/`(全部切分 + sha256)、`dedup_report.json`、`preds/`(逐篇概率)、`thresholds.json`、`summary.csv`(总表)、`summary_table.png`(16:9 英文表格图,可直接放 slides)、各任务日志;
- `prod_downtime.md`:停服快照、停服与恢复时间、恢复后的验证输出、crontab diff;
- `REPORT.md`:
  1. 第 0 步自检结论;数据源核对(篇数、label 确认、revision、sha256);
  2. 去重:剔除数,以及切分后重叠数(必须为 0);
  3. **总表**:基线行 + 3 模型 × 2 臂,阈值 ① 和 ② 各一张,列 = TOEFL / Hewlett / CollegeEssay / CS224N / ELLIPSE 测试的误判率、AI 352 检出率、改写文本被判 AI 比例、TOEFL−Hewlett 差距、AUC;
  4. **按固定顺序解读**:先看 TOEFL−Hewlett 差距在臂 B 是否小于臂 A,且 AI 检出率没有明显下降;再看 ELLIPSE。不允许只挑好看的数字讲;
  5. GPU 型号、每个任务耗时、版本、可复制的重跑命令;
  6. 所有数字标"请独立复核",注明出自哪次运行;
  7. 局限:PERSUADE 人写中也有 ELL 学生(daigt 不带人口学字段),"母语"只能称"以母语为主";ELLIPSE 训练与测试同源;TOEFL 是成人考生而 PERSUADE/ELLIPSE 是中学生;Qwen 只有 2 个 seed。
