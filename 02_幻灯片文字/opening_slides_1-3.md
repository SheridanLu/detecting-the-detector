# 开场三页:先问题,再故事,再"为什么选这个题"(英文稿 + 讲稿)

整套变成 14 页(多了标题页)。开场目标:让听的人在 3 分半里明白 ①这件事发生在真实学生身上 ②为什么轮到我们来做 ③我们要回答什么。

---
## 第 1 页 标题页(20 秒)

**Title:** Detecting the Detector
**Subtitle:** Would an AI-text detector believe you wrote your own essay?
**Small line:** Auditing AI-text detectors for bias against non-native English writers · INT4096 Group 08

**讲稿(一句话):** "We start with one question: if you write your own essay in English, will an AI detector believe you?"

(可选互动:先问一句 "Has anyone here ever worried that a detector might flag their own writing?",停 2 秒再翻页。)

---
## 第 2 页 开场故事(1 分钟)  → 图:fig01_hook.png

整页放 fig01。图上标题已经是 "A student wrote this essay. Seven AI detectors said: AI."

**讲稿:** "This essay was written by a student. It is a TOEFL essay, 107 words about why they want to study engineering. We ran it through seven commercial AI detectors. All seven said: AI. Below it is an essay by a native English writer, with spelling mistakes and all. None of the seven flagged it. I picked this pair to make a point, not at random: in this benchmark, 19 of 91 TOEFL essays were flagged by all seven detectors, and 72 of 88 native essays by none. So what is going on, and who pays for it?"

---
## 第 3 页 Why we chose this topic(2 分钟)

**Title:** Why this topic: a detector's score can decide a student's fate, and it may not treat everyone alike

**左栏 — "Why it matters" (四个理由,每条一行)**
1. **High stakes in education.** A detector score can start an academic-misconduct case against a student. [team: confirm / add an example you know]
2. **It affects people like us.** Non-native English writers study and write in English every day. [team: confirm, e.g. "many of us write in English as a second language at EdUHK"]
3. **The evidence is old and incomplete.** Liang et al. (2023) found detectors flag non-native writing far more; on their archived scores we reproduce **61.9%** of TOEFL essays flagged vs **5.4%** of native essays. But that was 2023, commercial tools only, and nobody asked whether the cause is *non-native* writing or just *unfamiliar* writing.
4. **It fits this course.** Machine learning that supports decisions about students must be tested for bias, and model results "should not replace the professional judgement of teachers" (assignment brief).

**右栏 — "Our three questions"**
- **RQ1** Do detectors flag non-native exam essays more than native ones, and are newer open-source detectors better?
- **RQ2** Is that bias about the writer's background, or about writing the detector has not seen?
- **RQ3** Can training data fix it, and does a bigger model help?

**讲稿(约 2 分钟):** "We chose this topic for four reasons. First, the stakes are real: a detector score can start a misconduct case against a student. Second, it affects people like us, [team to confirm: students who write in English as a second language]. Third, the evidence we have is old and incomplete: Liang and colleagues showed in 2023 that commercial detectors flag non-native writing far more often, and we reproduce 62 percent versus 5 percent on their scores. But nobody had asked whether that is because the writer is non-native, or simply because the writing is unfamiliar to the detector. And fourth, this is exactly what our course teaches: if machine learning supports decisions about students, we must test it for bias, and its results should never replace a teacher's judgement. So we asked three questions: is there bias, what causes it, and can training data fix it?"

---
## 注意(别说过头)
- 第 1 条"一个分数能引起学术不端调查":幻灯片上写 *can*,不要写 "universities rely on"。旧稿 P3 那句 "Universities increasingly rely on detector scores as proof" **没有出处,要么补引用,要么删**。
- 第 3 条的 61.9% / 5.4% 是**我们用 Liang 公开的分数复算的**(商用检测器 2023 年的现成分数),不是 Liang 论文里的原数;论文里写的是 61.22%。口播说 "we reproduce" 即可。
- Stowe et al.(ACL 2026,arXiv:2512.09292)用约 4.17 万篇学生作文测了 16 个检测器的偏见。引用前请对着原文核实一次,再决定放在第 3 页还是参考文献里。
- 第 2、3 条括号里的 [team: confirm] 是**必须由你们填的真实理由**。我不知道你们当时选题的真实原因,不要照抄我的推测。
