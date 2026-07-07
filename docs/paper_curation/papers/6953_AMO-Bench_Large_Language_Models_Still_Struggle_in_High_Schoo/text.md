# arXiv:2510.26768v1[cs.CL]30Oct2025

## AMO-Bench: Large Language Models Still Struggle in High School Math Competitions

Shengnan An∗♢, Xunliang Cai♢, Xuezhi Cao∗♢, Xiaoyu Li♢, Yehao Lin♢, Junlin Liu†♣, Xinxuan Lv♢, Dan Ma♢, Xuanlin Wang†♡, Ziwen Wang♢, Shuang Zhou♢ (Alphabetical order by last name)

♢Meituan ♣University of Chinese Academy of Sciences ♡Harbin Institute of Technology

### ABSTRACT

We present AMO-Bench, an Advanced Mathematical reasoning benchmark with Olympiad level or even higher difficulty, comprising 50 human-crafted problems. Existing benchmarks have widely leveraged high school math competitions for evaluating mathematical reasoning capabilities of large language models (LLMs). However, many existing math competitions are becoming less effective for assessing top-tier LLMs due to performance saturation (e.g., AIME24/25). To address this, AMO-Bench introduces more rigorous challenges by ensuring all 50 problems are (1) cross-validated by experts to meet at least the International Mathematical Olympiad (IMO) difficulty standards, and (2) entirely original problems to prevent potential performance leakages from data memorization. Moreover, each problem in AMO-Bench requires only a final answer rather than a proof, enabling automatic and robust grading for evaluation. Experimental results across 26 LLMs on AMO-Bench show that even the best-performing model achieves only 52.4% accuracy on AMO-Bench, with most LLMs scoring below 40%. Beyond these poor performances, our further analysis reveals a promising scaling trend with increasing test-time compute on AMO-Bench. These results highlight the significant room for improving the mathematical reasoning in current LLMs. We release AMOBench to facilitate further research into advancing the reasoning abilities of language models.

#### Code, Dataset, and Leaderboard: amo-bench.github.io

|Models|99.2 99.6 98.8 99.2|
|---|---|
|LongCat-Flash-Thinking<br><br>GLM-4.5<br><br>Qwen3-235B-A22B-Thinking-2507<br><br>Gemini-2.5-Pro|90.6<br><br>93.3<br><br>89.3<br><br>95.4 92.5<br><br>93.9<br><br>89.2<br><br>90.7<br><br>98.0 94.6 93.9<br><br>92.0|
|DeepSeek-V3.1-Thinking<br><br>GPT-5-Thinking (High)|83.7<br><br>85.5 83.8<br><br>87.9 84.8|

100

90

80.4

80

79.3

76.3

Accuracy(%)

70

60

52.4

50

47.8

47.6

43.6

40

38.7

36.8

30

20

AMO-Bench(Ours) HMMT25 AIME25 AIME24 MATH500

Figure 1: Performance of top-tier reasoning models on AMO-Bench as well as existing competition-level math benchmarks. Except for the results on AMO-Bench, all other results are sourced from Meituan LongCat Team [2025a].

∗ Correspondence to: {anshengnan, caoxuezhi}@meituan.com. † Work done during the internship at Meituan.

### 1 Introduction

Recent advances in large language models (LLMs) have demonstrated significant improvements in reasoning capabilities [OpenAI, 2024, Gemini Team, 2025, OpenAI, 2025, Anthropic, 2025, xAI, 2025, Yang et al., 2025, Guo et al., 2025, DeepSeek-AI, 2025, Meituan LongCat Team, 2025b, GLM-4.5 Team, 2025, ByteDance Seed, 2025, Tencent Hunyuan Team, 2025, Kimi Team, 2025, Meituan LongCat Team, 2025a]. To track this rapid progress, mathematical problem solving has become a critical metric for evaluation, as it inherently demands complex and multi-step reasoning processes to arrive at correct answers. As a result, many current benchmarks utilize problems from high school mathematics competitions (e.g., HMMT and AIME) to assess the reasoning abilities of LLMs [Balunovi´c et al., 2025, He et al.,

- 2024, Gao et al., 2024, Fang et al., 2025]. Recent results indicate that state-of-the-art models are achieving remarkable performances on these benchmarks, with some even surpassing 90% accuracy on competitions like AIME24/25.

However, these impressive results also expose an emerging challenge: many existing mathematics benchmarks are approaching performance saturation and are becoming less effective for assessing further advancements in reasoning capabilities. On the one hand, as LLMs gradually approach or even surpass human-level capabilities in mathematics, some math competitions are becoming less challenging for top-tier models [OpenAI, 2025, DeepSeek-AI, 2025, Yang et al., 2025, Meituan LongCat Team, 2025a]. On the other hand, most current benchmarks are derived from previous competitions, raising concerns about potential data memorization and performance leakage [Sun et al.,

- 2025, Balunovi´c et al., 2025]. While recent efforts have incorporated problems from more difficult and newly held contests such as the International Mathematical Olympiad (IMO), these questions tend to be proof-based and require manual verification by experts [Balunovi´c et al., 2025, Petrov et al., 2025]. This reliance on expert review hinders the implementation of automated scoring processes, leading to inefficiency and inconsistency in large-scale evaluations and result reproductions.

To address these limitations, we present AMO-Bench, an advanced mathematical reasoning benchmark consisting of 50 novel and extremely challenging problems. The core features of AMO-Bench are as follows:

- • Original problems. To prevent performance leaks from existing resources as much as possible, all problems in AMO-Bench are newly crafted by human experts. Moreover, we conduct a secondary verification to ensure that there are no highly similar problems in existing competitions or online resources.
- • Guaranteed difficulty. Each problem has undergone rigorous cross-validation by multiple experts to ensure it meets at least the difficulty standards of IMO. We also incorporate an LLM-based difficulty filtering stage to exclude questions that do not present sufficient challenge to current reasoning models.
- • Final-answer based grading. Each problem in AMO-Bench requires a final answer rather than a full proof, enabling efficient automatic grading. For each problem, we employ a parser-based or LLM-based grading method according to its answer type, balancing the grading cost and generalizability.
- • Human-annotated reasoning paths. In addition to the final answer, each problem also includes a detailed reasoning path written by human experts. These additional annotations enhance solution transparency and could support further explorations on AMO-Bench, such as prompt engineering and error analysis.

Experimental results across various LLMs demonstrate that contemporary LLMs still struggle with the significant challenges presented by AMO-Bench. Among 26 evaluated models, the state-of-the-art accuracy on AMO-Bench is only 52.4%, achieved by GPT-5-Thinking (High), with most models scoring below 40%. Figure 1 illustrates the performance of several leading models on AMO-Bench as well as the comparison with other mathematical benchmarks. Beyond their limited final performances on AMO-Bench, LLMs consume substantially more output tokens in AMO-Bench compared to existing evaluation datasets. For example, GPT-5-Thinking (High) generates an average of approximately 37K output tokens for AMO-Bench, whereas it produces only about 7K and 6K tokens for AIME25 and AIME24, respectively. This exceptionally high token consumption further underscores the difficulty of AMO-Bench for current LLMs. Despite the poor performances of current LLMs, our analysis also reveals considerable potential for further improvements. Notably, top-tier models achieve pass@32 rates exceeding 70%, suggesting they possess the initial capability to solve these challenging problems even if they do not consistently identify the correct reasoning path at present. Furthermore, we show that the model performances exhibit a near-linear growth trend relative to the logarithm of output length, indicating continued benefits from test-time scaling. These analyses suggest substantial opportunities remain to enhance reasoning capabilities in future generations of language models.

The data and evaluation code of AMO-Bench are publicly available at amo-bench.github.io. We hope this novel and challenging benchmark will facilitate further research into advancing the reasoning abilities of language models.

- Example 1:

Problem: Let 𝑥1, 𝑥2, …, 𝑥2024 be positive real numbers such that 𝑥𝑘 + 𝑥𝑚 ≥ 𝑘𝑚 …

Solution: From the assumption: for any positive

integer 1 ≤ 𝑘 ≤ 1012, we have … Final Answer: 1382935444

- Example 2:

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Data Correctness

Exist Competitions

[Figure 5]

Manually Review

Parser-Based

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

MO Syllabus Validation

Model

Web Search

LLM-Based

Problem: Find all positive integers n such that

Performance

Human Experts

for any: 𝑎𝑛≥ 𝑎𝑛−1 ≥ … ≥ 𝑎2 ≥ 𝑎1 > 0 … Solution: For 𝑛 = 1 , it clearly holds. For 𝑛 =

2, we have 𝑎2𝑎1 = 1, so 𝑎1 ∙ 𝑎22 = 𝑎2 … Final Answer: {1, 2, 3}

Originality Review

Difficulty Review

Grading Method

Quality Review

Data Creation

Figure 2: The construction and grading pipeline of AMO-Bench.

### 2 AMO-Bench

In this section, we first introduce the construction process of AMO-Bench (Section 2.1) and present the basic statistics of this dataset (Section 2.2). Then, we elaborate on the grading methodology designed for AMO-Bench (Section 2.3).

- Figure 2 briefly illustrate the construction and grading pipeline of AMO-Bench.

#### 2.1 Construction Pipeline

To ensure the high standards of quality, originality, and difficulty level in our dataset, we have built up a comprehensive multi-stage construction pipeline that covers the entire process from question creation to final inclusion. This pipeline comprises four major stages: data creation, quality review, originality review, and difficulty review.

Data creation. All problems are independently designed by mathematics experts from top universities and educational institutions. These experts have extensive backgrounds in high school mathematics competitions, either having won MO-level mathematics competition awards or possessing experience in competition problem design. Beyond the final answer, each problem author must provide a detailed step-by-step solution. These annotated solutions will be utilized in the subsequent quality review stage and will also aid in assessing the overall difficulty of AMO-Bench (see Section 2.2 for details).

Quality review. Each candidate problem undergoes blind review by at least three experts to assess its quality. This quality review stage focuses primarily on two aspects:

- • Whether the problem statement and solution are semantically unambiguous and logically correct.
- • Whether the mathematical knowledge required for the problem is within the scope typically covered in MO-level competitions such as IMO.

Originality review. The originality review stage aims to ensure that these newly created problems are not mere rewrites of publicly available materials, but demonstrate genuine originality. To this end, we assess the originality of each problem through the following methods:

- • Compare it against problems in existing datasets (e.g., AIME24/25) with 10-gram matching.
- • Conduct web searches to identify any similar online content.

Additionally, during the quality review stage, experts are also required to indicate whether they have encountered highly similar questions in past competitions.

Difficulty review. To ensure that AMO-Bench presents a sufficient challenge to state-of-the-art LLMs, we implement a difficulty review stage to filter out problems lacking adequate complexity (even if they may be suitable for some MO-level competitions, e.g., the first 10 questions in AIME). Specifically, each selected problem must satisfy the following two criteria:

- • The problem must meet or exceed the IMO difficulty standards, as verified by the human expert.
- • We employed multiple advanced reasoning models (such as GPT, DeepSeek, and Gemini series models) for preliminary evaluation, requiring that at least two such models fail to correctly and consistently solve the problem3.

3For each model, our preliminary evaluation involves three samples. If all three samples are correct, the model is deemed capable of consistently solving the problem.

[Figure 10]

[Figure 11]

100

MATH500

93%

AIME2024

[Figure 12]

AMO-Bench

Geometry 10%

80

SolutionLengthRatio(%)

Functions & Sequences

60

26%

Number Theory 18%

47%

[Figure 13]

[Figure 14]

40%

40%

40

Algebraic Equations

26%

& Inequalities 22%

20%

Combinatorics 24%

20

12%

10%

7%

3%

2%

0% 0% 0%

0%

0

<512 512-1,024 1,024-2,048 2,048-4,096 >4,096

(a) Distribution of problem categories.

(b) Comparison of solution lengths.

- Figure 3: Basic statistics of AMO-Bench. (a) The distribution of problem categories in AMO-Bench. (b) The distribution of human-annotated solutions in AMO-Bench as well as the comparison with MATH500 and AIME24.

#### 2.2 Dataset Statistics

Problem categories. Referring several official competition syllabus, we categorize the 50 problems of AMO-Bench into the following five primary categories: Algebraic Equations & Inequalities (11/50), Functions & Sequences (13/50), Geometry (5/50), Number Theory (9/50), and Combinatorics (12/50). Figure 3a show the overall distribution of problem categories in AMO-Bench.

Length distribution of human-annotated solutions. Since the problems in our AMO-Bench are equipped with manually annotated solutions, we can preliminarily analyze the reasoning complexity of these problems from the view of solution length. We measure solution length in terms of token count4. Additionally, we compare the distribution of solution lengths with those from AIME245 and MATH5006. Figure 3b illustrates the solution length distributions across these benchmarks. It reveals that solutions in AMO-Bench exhibit significantly higher lengths, indicating that problems in this benchmark are inherently more challenging and require more complex reasoning to arrive at the final answer. We conduct a further analysis of the model solution lengths in Section 3.2.

#### 2.3 Grading Method

For evaluating answers generated by LLMs, prior work has primarily utilized two approaches: parser-based grading and LLM-based grading. Parser-based grading offers high efficiency and accuracy when the model’s response can be successfully parsed; however, its applicability is limited to simple answer formats such as numerical values or sets, making it challenging to assess more complex answers. In contrast, LLM-based grading provides greater flexibility across diverse answer types but may be less efficient and does not consistently guarantee accuracy.

To fully leverage the strengths of both grading methods, AMO-Bench employs different grading approaches based on the specific answer type for each problem. Specifically, problems in AMO-Bench are divided into four main answer types: numerical answers (e.g., Example 1), set answers (e.g., Example 2), variable-expression answers (e.g.,

- Example 3 which requires providing the general formula for an arithmetic sequence), and descriptive answers (e.g.,
- Example 4 which involves comprehensively considering multiple scenarios). The prompt templates for used for grading are contained in Appendix A.

#### Example 1: Problem with Numerical Answer

Question: Let x1,x2,··· ,x2024 be positive real numbers such that xk + xm ≥ km for any 1 ≤ k < m ≤ 2024. Find the minimum value of x1 + x2 + ··· + x2024. Answer:

|1382935444|
|---|

4We use the tokenizer of DeepSeek-V3.1 model to count tokens in solutions.

- 5https://huggingface.co/datasets/HuggingFaceH4/aime_2024.
- 6https://huggingface.co/datasets/HuggingFaceH4/MATH-500.

#### Example 2: Problem with Set Answer

Question: Find all positive integers n such that for any: an ≥ an−1 ≥ an−2 ≥ ······a2 ≥ a1 > 0, satisfying

n

n

n

akk ≥ 1 holds. Answer:

ak , the inequality

1

ak =

k=1

k=1

k=1

|{1,2,3}|
|---|

#### Example 3: Problem with Variable-Expression Answer

Question: The sequence {an}∞n=1 consists of positive terms, with a1 = 7, a2 = 2, and satisfies the recurrence relation

8a4n+2 = 3 + 4an+1 + an (n ∈ N∗). Find the general term formula for this sequence. Answer:

|(2 + √3)2<br><br>2−n<br><br>+ (2 −<br><br>√3)2<br><br>2−n<br><br>2|
|---|

#### Example 4: Problem with Descriptive Answer

Question: Let n be an integer with n > 2. Real numbers a1,a2,...,an satisfy

n

n

k |ak| = 4n.

ak = 2n,

k=1

k=1

Find the minimum value of a21 + a22 + ··· + a2n. Answer: For n = 3, the minimum of a21 + a22 + a23 is 12.

6n2 5

For n ≥ 4, the minimum of a21 + a22 + ··· + a2n is

.

For problems requiring numerical, set, or variable-expression answers (39 out of 50), we employ the parser-based grading. The evaluated LLMs are instructed to format their final responses as \boxed{<answer>}. We then utilize the tools provided by math-verify7 to parse these answers and verify the equivalence with the ground truth. Moreover, if the model answer containing decimal values, we require an accuracy of at least four decimal places. For variableexpression answers, we assign multiple sets of values to the variables in the expression, then verify whether the values of the generated expression match that of the ground-truth expression. We also manually review the parsing results during the preliminary evaluation and adjust the post-processing algorithms.

For problems requiring descriptive answers (11 out of 50), we use LLM-based grading with o4-mini (Low) serving as the grading model. To ensure robust assessment, majority voting is performed across five independent grading samples for each response. Additionally, during preliminary evaluation, we manually verify the correctness of LLM-based grades for all descriptive answers and revise answer descriptions where needed to enhance grading accuracy.

Grading accuracy. Prior to conducting the large-scale evaluation, we performed a manual quality check to ensure the reliability of the designed grading method. This assessment included 1,000 responses generated by 10 different LLMs. The results indicate that the grading accuracy reached 99.2%, providing strong validation for the effectiveness of the grading method on AMO-Bench.

### 3 Experiments

###### In this section, we present the experimental results on AMO-Bench. We first describe the experimental setup (Section 3.1), followed by a discussion of the main results and analysis (Section 3.2).

7https://github.com/huggingface/Math-Verify.

52.4

Proprietary Models

50

Open Source Models Reasoning Models Non-Reasoning Models

47.8 47.6 47.3

| |
|---|

43.6

| |
|---|

40.2

40

38.7

36.8

34.8 34.3

32.3

AVG@32(%)

30.0

30

28.8

25.9

20

18.2 18.1 17.6

14.6

13.1

10.9 10.6

9.8

10

7.5

5.2

4.1

1.5

0

Qwen3-235B-A22B-Thinking-2507GPT-5-Thinking(High)DeepSeek-V3.1-ThinkingGPT-5-Thinking(Medium)LongCat-Flash-Thinkingo4-mini(High)Gemini-2.5-ProQwen3-Next-80B-ThinkingGLM-4.5DeepSeek-R1-0528o3-mini(High)o4-mini(Medium)Qwen3-Max-InstructGPT-5-Thinking(Low)Qwen3-Next-80B-InstructGemini-2.5-FlashClaude-Sonnet-4.5LongCat-Flasho3-mini(Medium)DeepSeek-R1Claude-Opus-4DeepSeek-V3.1DeepSeek-V3-0324Kimi-K2 GPT-4o-20241120GPT-4.1

Figure 4: The AVG@32 performance of various LLMs on AMO-Bench.

#### 3.1 Experimental Setup

Models. To conduct a comprehensive and representative evaluation on AMO-Bench, we select a diverse set of leading LLMs, encompassing both open-source models and proprietary models. Specifically, the evaluation includes top-tier models provided by OpenAI [OpenAI, 2025], Gemini [Gemini Team, 2025], Anthropic [Anthropic, 2025], DeepSeek [Guo et al., 2025], Qwen [Yang et al., 2025], GLM [GLM-4.5 Team, 2025], Moonshot [Kimi Team, 2025], and LongCat [Meituan LongCat Team, 2025a]. In addition to evaluating reasoning models that have been specifically enhanced for long-term thinking tasks, we also incorporated several powerful non-reasoning models to demonstrate their potential in tackling complex reasoning challenges.

Sampling settings. We set the temperature of sampling to 1.0 for reasoning models and 0.7 for non-reasoning models. For all evaluated models, we use top-k=50 and top-p=0.95 during sampling. We configure the maximum context/output length to the highest allowable limit for each model during inference. This avoids underestimating the reasoning capabilities of the model due to restrictions on the token budget. To ensure the stability of the final evaluation results, we sampled the results from each model 32 times and reported the average performance of these 32 results as the final metric (denoted as AVG@32). Appendix B illustrates the fluctuation of the average result across different sampling times. It demonstrates that when sampling 32 times, the average model performance exhibits a relatively small fluctuation and rarely appears to reverse the model ranking order.

#### 3.2 Results and Analysis

Main results. Figure 4 presents the AVG@32 performance of various leading LLMs, categorized by proprietary/opensource status and reasoning/non-reasoning properties8. Overall, all these models still struggle with the significant challenges presented by AMO-Bench. Even the highest performing model GPT-5-Thinking (High) reaches just 52.4%, while most others score below 40%. This indicates substantial room for improvement in complex reasoning abilities across all current language models. Moreover, both proprietary and open-source reasoning models occupy top ranks in the leaderboard, indicating that recent open-source advancements are closing the gap with leading commercial models. The best-performing open-source model is only about 5% lower than the top proprietary result. Besides reasoning models, some non-reasoning models demonstrate a performance exceeding expectations, such as Qwen3-Max-Instruct

8To facilitate easier reproduction and utilization of AMO-Bench, you can take a fast try on the AMO-Bench-P subset, which includes only the 39 parser-based grading problems from AMO-Bench. Appendix C presents the AVG@32 performance of LLMs on AMO-Bench-P.

| | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |GPT|-5-Th|inking|(Hig|h)| | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |Dee|pSee|k-V3.|1-Thin|king| |Qwe|n3-2|35B-T|hinki|ng|
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |Lo|ngCa|t-Flas|h-Thin|
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | |o4-m|ini (|High)|Gem|ini-2|.5-Pro| | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |G|LM-4|.5| | |
| | | | | | | | | | | | | | | | | | |Qwe|n3-N|ext-Th|inkin|g| | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | |o3-m|ini (H|igh)| | | | | | |Dee|pSeek|-R1-|0528| |
| | | | |Qw|en3-M|ax-In|struc|t| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | |Qwe|n3-N|ext-In|struc|t|Cla|ude-S|onne|t-4.5| |Gem|ini-2|.5-Fla|sh| | | | | | | | | |
| | | | | | |ongC|at-Fl|ash| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | |Clau|de-O|pus-4|De|epSe|ek-R|1| | | | | | | | | | | | | | | | |
|eep|Seek|-V3.1| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | |K|imi-K2| | | | | | | | | | | | | | | | | | | | | |
|See|k-V3|-0324| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | |G|PT-4.|1| | | | | | | | | | | | | | | | | | | | | | |
| |GP|T-4o-2|0241|120| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |

50

king

40

AVG@32(%)

30

20

D

10

Deep

0

0 10000 20000 30000 40000 50000

Average Output Tokens

Figure 5: The AVG@32 performance of LLMs vs. the average model output length.

and LongCat-Flash. These non-reasoning models even outperforms several reasoning models such as o3-mini (Medium), indicating their significant potential in tackling complex reasoning tasks.

Comparison of reasoning efficiency. Figure 5 shows the average output length and the AVG@32 performance of each model. Overall, it demonstrates a clear trend that higher-performing models tend to require more output tokens. The first-tier models that reach higher than 40% AVG@32 scores utilize more than 35K completion tokens. Even among non-reasoning models, those with superior performance are distinguished by their ability to process more tokens, sometimes reaching levels comparable to reasoning models. Additionally, when examining models within the same series, there are notable improvements in reasoning efficiency over time. For example, o4-mini (High) outperforms

- o3-mini (High) at similar or slightly increased token counts. Likewise, DeepSeek-V3.1-Thinking shows significant gains compared to DeepSeek-R1-0528 with even significantly less output tokens.

Beyond the main results outlined above, we also provide further analysis and insights based on the AMO-Bench experimental findings.

The model output length could indicate the reasoning challenge of the benchmark. Section 2.2 provides a pre-analysis of benchmark difficulty based on annotated solution lengths. Here, we offer a post-hoc analysis of benchmark difficulty based on the relationship between model performance and model output length. Figure 6 clearly demonstrates that the average output length of each model increases as the reasoning benchmark becomes more challenging. Specifically, across six models, benchmarks with higher accuracy scores (such as MAH500 and AIME24) correspond to shorter average outputs, while those with lower scores (like AMO-Bench) require significantly longer responses. This suggests that harder benchmarks demand more elaborate reasoning steps or explanations from the models, resulting in increased token usage. These results demonstrate that the model output length could be an indicator

- of reasoning challenge in the benchmark.

Performance on AMO-Bench still benefits from test-time scaling. The reasoning efficiency results discussed above indicate a correlation between model performance and output length. Here, we conduct a more rigorous analysis by directly controlling the reasoning effort for the same model. As shown in the Figure 7, all three models (GPT-5, o4-mini, and o3-mini) exhibit a near-linear growth trend in AVG@32 as the logarithm of average output length increases. Such a trend is highly aligned with earlier experimental observations from existing benchmarks such as MATH500

MAH500

MAH500

MAH500

100

100

LongCat-Flash-Thinking

GLM-4.5

Qwen3-235B-A22B-Thinking-2507

AIME24

AIME24

AIME24

90

AIME25

AIME25

AIME25

90

90

HMMT25

80

HMMT25

HMMT25

Accuracy(%)

Accuracy(%)

Accuracy(%)

80

80

70

BeyondAIME

BeyondAIME

BeyondAIME

70

70

60

60

60

50

50

40

50

AMO-Bench

AMO-Bench

AMO-Bench

10000 20000 30000 40000 50000

5000 10000 15000 20000 25000 30000 35000 40000 45000

10000 20000 30000 40000

Average Output tokens

Average Output tokens

Average Output tokens

MAH500

MAH500

MAH500

100

100

100

Gemini-2.5-Pro

DeepSeek-V3.1-Thinking

GPT-5-Thinking (High)

AIME24

AIME25 AIME24

AIME24

AIME25

90

90

AIME25

90

HMMT25

HMMT25

80

HMMT25

Accuracy(%)

Accuracy(%)

Accuracy(%)

80

80

70

BeyondAIME

BeyondAIME

70

BeyondAIME

70

60

60

50

60

50

AMO-Bench

AMO-Bench

AMO-Bench

40

5000 10000 15000 20000 25000 30000

5000 10000 15000 20000 25000 30000 35000

5000 10000 15000 20000 25000 30000 35000

Average Output tokens

Average Output tokens

Average Output tokens

Figure 6: The relationship between accuracy and average output length on different math benchmarks.

GPT-5 (High) Model

o4-mini (High) Model

o3-mini (High) Model

40

GPT-5

o4-mini

o3-mini

30

50

GPT-5 (Medium)

35

o4-mini (Medium)

25

45

30

AVG@32(%)

AVG@32(%)

AVG@32(%)

20

40

25

o3-mini (Medium)

15

35

20

15

10

30

GPT-5 (Low)

o4-mini (Low)

o3-mini (Low)

10

5

25

8,192 16,384 32,768

2048 4,096 8,192 16,384 32,768

2048 4,096 8,192 16,384 32,768

Logarithm of Average Output Length

Logarithm of Average Output Length

Logarithm of Average Output Length

Figure 7: The model performance and output length under different reasoning effort settings.

and AIME24 [Muennighoff et al., 2025]. This indicates that further increasing the inference budget will further drive improvements on AMO-Bench.

Top-tier models demonstrate promising potential for improvement on AMO-Bench. Existing work reveals that the pass@k performance of the model can reflect its inherent potential to achieve further improvement through reinforcement learning. Inspired by this, we illustrate the pass@k of evaluated models to indicate their inner potential. As shown in Figure 8, the pass@k metric exhibits rapid growth as k increases from 1 to 8, followed by a sustained but gradual improvement as k continues to rise. Notably, the top-tier reasoning models achieve over 70% performance on the pass@32 metric. These results highlight the significant room for improvement in the reasoning capabilities of LLMs.

### 4 Related Work

Evaluating LLMs on mathematical problem solving has been a critical aspect for assessing advancements in reasoning capabilities. Early datasets such as GSM8K [Cobbe et al., 2021] and MATH [Hendrycks et al., 2021] provided

| | | | | | | | | | | | | | | | | | | | | | | | |83|.5| | | | | | |86|.0| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | |80|.9| | | | | | |80|.3| | | | | | |82|.0| |
| | | | | | | | |73<br><br>76|.6<br><br>.6| | | | | | |78 74 72|.0 .9 .4| | | | | | |76 73|.8 .4| | | | | | |78 74|.0 .0| |
| | | | | | | | |71 70|.3 .0| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |65|.1| | | | | | |68|.0| |
| | | | | | | | | | | | | | | | |61|.0| | | | | | | | | | | | | | | | | |
| | | | | | | | |53|.8| | | | | | | | | | | | | | | | | | | | | | |54|.0| |
| |52|.4| | | | | | | | | | | | | | | | | | | | | |49|.6| | | | | | | | | |
| |4747 43|.6.8 .6| | | | | | | | | | | | | |43|.9| | | | | | | | | | | | | | |44|.0| |
| | | | | | | | | | | | | | | | | | | | | | | | |40|.5| | | | | | | | | |
| | | | | | | | |35|.0| | | | | | |35|.3| | | | | | | | | | | | | | | | | |
| |28|.8| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | |27|.1| | | | | | | | | | | | | | | | | | | | | | |26|.0| |
| | | | | | | | | | | | | | | | |19|.3| | | | | | |22|.9| | | | | | | | | |
| |14|.6| | | | | |14|.3| | | | | | | | | | | | | | | | | | | | | | | | | |
| |9.|8| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |4.|1| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

80

60

Pass@K(%)

40

20

0 5 10 15 20 25 30

K

GPT-5-Thinking (High)

Qwen3-235B-Thinking

Qwen3-Max-Instruct

DeepSeek-V3.1

DeepSeek-V3.1-Thinking

LongCat-Flash-Thinking

LongCat-Flash

GPT-4.1

Figure 8: The the pass@k trend of various LLMs with increasing k.

initial explorations to evaluate these abilities. However, model performance on these benchmarks has quickly reached saturation. To further advance the study of mathematical proficiency in LLMs, recent work has shifted toward more challenging benchmarks.

In terms of increasing difficulty, two primary lines of work have emerged. One line focuses on Mathematical Olympiad (MO)-level problems, which rely on a specific range of math knowledge and require complex and intuitive reasoning skills. For instance, Omni-MATH [Gao et al., 2024] introduces a multi-subject evaluation suite designed to rigorously test mathematical reasoning and generalization; OlympiadBench [He et al., 2024] focuses on evaluating the bilingual and multi-modal reasoning abilities with Olympid-level challenges; OlymMATH [Sun et al., 2025] collects MO-level problems from printed publications and evaluates mathematical reasoning by offering problems of two difficulty levels; MathOdyssey [Fang et al., 2025] broadens the scope to include more complex tasks, with a particular focus on long-range and compositional reasoning; BeyondAIME [ByteDance-Seed, 2025] collects problems similar in style to AIME with increased difficulty and expanded data scale; MathArena [Balunovi´c et al., 2025] rapidly tracks model performance in newly held MO-level competitions and explores evaluation paradigms for proof-based competitions such as the IMO and USAMO. Our proposed AMO-Bench also falls within this category and it stands as one of the most challenging benchmarks at the time of writing.

The other line of work focuses on problems derived from graduate-level examinations or advanced mathematical research. For instance, RealMath [Zhang et al., 2025] provides a comprehensive evaluation of LLMs in real-world mathematical tasks, assessing their reasoning capabilities across a diverse range of research-level content; FrontierMath [Glazer et al., 2024] covers computationally intensive problems and abstract questions across most branches of mathematics, highlighting the significant gap between LLMs and the prowess of the mathematical community; HARDMath2 [Roggeveen et al., 2025] focuses on approximation-based mathematical problems, particularly those commonly encountered in applied sciences and engineering; HLE [Phan et al., 2025] constructs a final closed-ended academic benchmark spanning multiple subjects, evaluating reasoning capabilities on human frontier knowledge. Beside requiring the reasoning abilities, these datasets also challenge models by demanding extensive and deep mathematical knowledge.

### 5 Conclusion

We introduce AMO-Bench, an advanced mathematical reasoning benchmark featuring problems at the level of mathematical Olympiads or higher. The benchmark consists of 50 human-crafted questions designed to rigorously assess advanced mathematical reasoning. Compared with existing benchmarks, AMO-Bench offers more challenging assessments by ensuring that all 50 problems are entirely original and meet or exceed IMO difficulty standards. Each problem in AMO-Bench requires only a final answer rather than a full proof, enabling automatic and robust grading for evaluation purposes. Experimental results across various LLMs demonstrate that contemporary LLMs still struggle with the significant challenges presented by AMO-Bench. Despite these low performances, our further analysis underscore substantial opportunities for advancing mathematical reasoning capabilities in current LLMs.

### Acknowledgments

We thank Zijian Zhang, Jun Kuang, Yiyang Li, Siyu Ren, Zongyu Wang, Yaoming Zhu, Ziyi Zhao, Linsen Guo, Yuhuai Wei, Cunguang Wang, Jiaming Wang and Mengjie Cao for their insightful suggestions regarding the construction and analysis of AMO-Bench. We are grateful to Wei Wang, Wenjie Shi, Jiaqi Zhang, Xiangyu Xi, Xiangzhou Huang, Rongxiang Weng, and Jingang Wang for the valuable discussions and insights on model performance. We also appreciate the engineering support provided by Yunke Zhao and Dengchang Zhao, and open-source assistance from Qi Li, Peng Wang and Xiangyang Ji.

### References

Meituan LongCat Team. Longcat-flash-thinking technical report, 2025a. URL https://arxiv.org/abs/2509.18883. OpenAI. Openai o1 system card, 2024. URL https://arxiv.org/abs/2412.16720. Gemini Team. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next

generation agentic capabilities, 2025. URL https://arxiv.org/abs/2507.06261. OpenAI. Gpt-5 system card, 2025. URL https://cdn.openai.com/gpt-5-system-card.pdf. Anthropic. System card: Claude opus 4 and claude sonnet 4, 2025. URL https://www-cdn.anthropic.com/4263b

940cabb546aa0e3283f35b686f4f3b2ff47.pdf. xAI. Grok 4 model card, 2025. URL https://data.x.ai/2025-08-20-grok-4-model-card.pdf.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081): 633–638, 2025.

DeepSeek-AI. Deepseek-v3 technical report, 2025. URL https://arxiv.org/abs/2412.19437. Meituan LongCat Team. Longcat-flash technical report, 2025b. URL https://arxiv.org/abs/2509.01322. GLM-4.5 Team. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models, 2025. URL https://arxiv.org/

abs/2508.06471. ByteDance Seed. Seed1.5-thinking: Advancing superb reasoning models with reinforcement learning, 2025. URL https://arxiv.org/abs/2504.13914. Tencent Hunyuan Team. Hunyuan-turbos: Advancing large language models through mamba-transformer synergy and

adaptive chain-of-thought, 2025. URL https://arxiv.org/abs/2505.15431. Kimi Team. Kimi k2: Open agentic intelligence, 2025. URL https://arxiv.org/abs/2507.20534. Mislav Balunovi´c, Jasper Dekoninck, Ivo Petrov, Nikola Jovanovi´c, and Martin Vechev. Matharena: Evaluating llms on

uncontaminated math competitions. arXiv preprint arXiv:2505.23281, 2025.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3828–3850, 2024.

Bofei Gao, Feifan Song, Zhe Yang, Zefan Cai, Yibo Miao, Qingxiu Dong, Lei Li, Chenghao Ma, Liang Chen, Runxin Xu, et al. Omni-math: A universal olympiad level mathematic benchmark for large language models. In The Thirteenth International Conference on Learning Representations, 2024.

Meng Fang, Xiangpeng Wan, Fei Lu, Fei Xing, and Kai Zou. Mathodyssey: Benchmarking mathematical problemsolving skills in large language models using odyssey math data. Scientific Data, 12(1):1392, 2025.

Haoxiang Sun, Yingqian Min, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, Zheng Liu, Zhongyuan Wang, and Ji-Rong Wen. Challenging the boundaries of reasoning: An olympiad-level math benchmark for large language models. arXiv preprint arXiv:2503.21380, 2025.

Ivo Petrov, Jasper Dekoninck, Lyuben Baltadzhiev, Maria Drencheva, Kristian Minchev, Mislav Balunovi´c, Nikola Jovanovi´c, and Martin Vechev. Proof or bluff? evaluating llms on 2025 usa math olympiad, 2025. URL https: //arxiv.org/abs/2503.21934.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candes, and Tatsunori Hashimoto. s1: Simple test-time scaling. In Workshop on Reasoning and Planning for Large Language Models, 2025.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021.

ByteDance-Seed. Beyondaime: Advancing math reasoning evaluation beyond high school olympiads, 2025. URL https://huggingface.co/datasets/ByteDance-Seed/BeyondAIME.

Jie Zhang, Cezara Petrui, Kristina Nikoli´c, and Florian Tramèr. Realmath: A continuous benchmark for evaluating language models on research-level mathematics. arXiv preprint arXiv:2505.12575, 2025.

Elliot Glazer, Ege Erdil, Tamay Besiroglu, Diego Chicharro, Evan Chen, Alex Gunning, Caroline Falkman Olsson, Jean-Stanislas Denain, Anson Ho, Emily de Oliveira Santos, et al. Frontiermath: A benchmark for evaluating advanced mathematical reasoning in ai. arXiv preprint arXiv:2411.04872, 2024.

James V Roggeveen, Erik Y Wang, Will Flintoft, Peter Donets, Lucy S Nathwani, Nickholas Gutierrez, David Ettel, Anton Marius Graf, Siddharth Dandavate, Arjun Nageswaran, et al. Hardmath2: A benchmark for applied mathematics built by students as part of a graduate class. arXiv preprint arXiv:2505.11774, 2025.

Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Chen Bo Calvin Zhang, Mohamed Shaaban, John Ling, Sean Shi, et al. Humanity’s last exam. arXiv preprint arXiv:2501.14249, 2025.

### A Prompt Templates

Query prompt template. In order to guide LLMs in generating answers in a parser-readable format, we use the following prompt template guide the model generation. There are mainly three requirements in the instruction: the answer prefix (i.e., ### The final answer is:), the LaTeX box environment (i.e., \boxed{}), and the precision requirement.

Example 5: Query Prompt Template

... After solving the above problem, please output your final answer in the following format: ### The final answer is: $\boxed{<your answer>}$ Example: ### The final answer is: $\boxed{123}$ The final answer should be given as precisely as possible (using LaTeX symbols such as \sqrt, \frac, \pi, etc.). If the final answer involves a decimal approximation, it must be accurate to at least four decimal places.

Grading prompt template. We employ the LLM-based grading using o4-mini (Low) as the grading model, and use the following grading prompt to verify the equivalence between the LLM output and the reference answer.

Example 6: Grading Prompt Template

For the following math problem, we have the reference answer and the student’s answer. Determine whether the student’s answer is equivalent to the reference answer. If equivalent, output "Correct". If not equivalent, output "Incorrect".

### Problem

... ### Reference Answer

... ### Student Answer

... Now, please provide your judgment. Please strictly follow the format below to summarize your conclusion at the end of your judgment: ### Conclusion: Correct/Incorrect If the answer involves a decimal approximation, it must be accurate to at least four decimal places.

### B Analysis of AVG@k

Figure 9 illustrates the fluctuation of the average performance across different sampling times. It shows that as the sampling time grows, the models’ performance become more stable. When sampling 32 times, it rarely appears the reverse-order phenomenon.

### C Performance on AMO-Bench-P Subset

To facilitate easier reproduction and use of AMO-Bench, you can utilize the AMO-Bench-P subset, which includes only the 39 parser-based grading problems from AMO-Bench. Table 1 presents the AVG@32 performance of LLMs on AMO-Bench-P. In general, performance on AMO-Bench-P tends to be slightly higher than on the full AMO-Bench, as problems requiring complex descriptive answers are inherently more challenging than those with simple-format answers.

60

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |58|.0| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | |56|.0| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | |53|.3| |53|.0<br><br>53|.7 53|.2 52|.9|52|.7| | | | | | | | | | | | | | | | | | | | | | |
| | | | |52|.5 52|.4| | | |52|.6|52|.3 52|.6 52|.1 52|.5 52|.1 52|.0 51|.7 52|.0 51|.7 51|.9 52|.2 52|.2 52|.0 52|.0 52|.2 51|.9 52|.1 52|.2 52|.3 52|.5 52|.4| |
| |50|.0 50|.0| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | |48|.0 48|.0 48|.0 47 46|.7 47<br><br>.7|.4 47|.5 47|.3<br><br>48 46|.0 48 .4 46|.5<br><br>47<br><br>.7|.7 48|.0 48|.0 47|.5 47|.0 46|.7 46|.8 47|.1 47|.1 47|.3 47|.4 47|.1 47|.2 47|.2 47|.4 47|.5 47|.4 47|.2 47|.3 47|.4 47|.6| |
| | | | |46|.0 46|.0|46|.0 46|.0 46|.0| |46|.0<br><br>45|.2 45|.0 44|.8 45|.1 45|.3 45|.1| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |44|.6 44|.2 44|.2 44|.0 44|.2 44|.5 44|.5 44|.2 43|.9 43|.7 43|.6 43|.5 43|.7 43|.6| |
| |42|.0 42 41|.0<br><br>42<br><br>.0|.7| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |40|.0|40|.0| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | |38|.5<br><br>36|.8 37|.0 36|.9| | | | | | | | | |36|.8 37|.1 36|.7 36|.9 36|.9 37|.0 37|.0 37|.0 36|.8 36|.9 36|.9 36|.9 36|.9 36|.9 36|.8| |
| | | | | | | | |35|.5| |35|.6 36|.0 35|.8 35|.9 35|.9 36|.0 36|.0| | | | | | | | | | | | | | | | |
| | | | | | | | | |34|.9 35|.0| | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |28|.7 28|.9 29|.0 29|.0 28|.8|28|.8 28|.8 28|.8| |
| | |27|.0 27|.3|26|.8| |27|.5 27|.1<br><br>28|.4 28|.5 28|.5 28|.2<br><br>27|.6 27|.9 27|.8 28|.0<br><br>27|.4<br><br>28|.1 28|.0 28|.1 28|.4 28|.5| | | | |28|.6| | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |26|.0| |26|.5|26|.7 26|.6| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |20|.0| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | |16|.0| | |14|.7| | |14|.8 14|.7 15|.0 15|.2 15|.0 15|.3 15|.6 15|.5 15|.2 15|.3 15|.2 15|.3 15|.2 15|.0 15|.0 14|.9 14|.8 14|.7 14|.7| | | | | |
| | | |14|.0|13|.6|14|.3 14|.3 14|.4| | | | | | | | | | | | | | | | | | |14|.5 14|.5 14|.6 14|.6| |
| |12|.0<br><br>13|.0<br><br>11|13<br><br>.3|.0|11|.0 11|.4| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | |10|.5 10|.4| |10|.5<br><br>9.|8 10|.0 10|.4 10|.2 10|.0 9.|9 10|.3 10|.4 10|.4 10|.2 10|.3 10|.3 10|.2 10|.1 10|.1 10|.2 10|.2 10|.2 10|.1 10|.1 10|.0 9.|9 9.|9 9.|8| |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |6.|0 6.|0| |4.|8 5.|3 5.|4 5.|5 5.|3 5.|4 5.|3 5.|0 4.|8| | | | | | | | | | | | | | | | | | | | |
| | | |4.|0 4.|0| | | | | | | | |4.|6 4.|4 4.|1 4.|4 4.|4 4.|4 4.|3 4.|3 4.|3 4.|2 4.|2 4.|0 4.|1 4.|1 4.|1 4.|0 4.|1 4.|0 4.|1| |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

50

40

###### AVG@K(%)

30

20

10

0 5 10 15 20 25 30

K

GPT-5-Thinking (High)

LongCat-Flash-Thinking

Qwen3-Max-Instruct

DeepSeek-V3.1

DeepSeek-V3.1-Thinking

GLM-4.5

LongCat-Flash

GPT-4.1

Figure 9: The AVG@k trend of various LLMs with increasing k.

Table 1: The AVG@32 performance of LLMs on the AMO-Bench and AMO-Bench-P, the latter of which contains only 39 parser-based grading problems.

Model AMO-Bench AMO-Bench-P GPT-5-Thinking (High) 52.4 54.8

Qwen3-235B-A22B-Thinking-2507 47.8 56.2 DeepSeek-V3.1-Thinking 47.6 53.0 LongCat-Flash-Thinking 43.6 45.3

o4-mini (High) 40.2 43.8 Gemini-2.5-Pro 38.7 41.7

GLM-4.5 36.8 41.0 Qwen3-Next-80B-Thinking 34.8 37.4 DeepSeek-R1-0528 34.3 37.1 o3-mini (High) 32.3 34.0

Qwen3-Max-Instruct 28.8 30.9 Qwen3-Next-80B-Instruct 18.2 17.8

Gemini-2.5-Flash 18.1 18.0 Claude-Sonnet-4.5 17.6 18.1

LongCat-Flash 14.6 14.9

DeepSeek-R1 10.9 11.7 Claude-Opus-4 10.6 11.4 DeepSeek-V3.1 9.8 9.6

Kimi-K2 7.5 8.4 DeepSeek-V3-0324 5.2 5.4

GPT-4.1 4.1 4.8 GPT-4o-20241120 1.5 1.9

