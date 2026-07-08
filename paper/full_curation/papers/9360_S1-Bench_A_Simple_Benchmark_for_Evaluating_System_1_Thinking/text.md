# arXiv:2504.10368v4[cs.CL]1May2026

## Exploring the System 1 Thinking Capability of Large Reasoning Models

Wenyuan Zhang1,2∗ , Shuaiyi Nie1,2∗ , Xinghua Zhang3 , Zefeng Zhang1,2 , Tingwen Liu1,2† 1Institute of Information Engineering, Chinese Academy of Sciences 2School of Cyber Security, University of Chinese Academy of Sciences 3Tongyi Lab, Alibaba Group {zhangwenyuan, nieshuaiyi, zhangzefeng, liutingwen}@iie.ac.cn, zhangxinghua.zxh@alibaba-inc.com

### Abstract

This paper explores the system 1 thinking capability of Large Reasoning Models (LRMs), the intuitive ability to respond efficiently with minimal token usage. While existing LRMs rely on long-chain reasoning and excel at complex tasks, their system 1 thinking ability remains largely underexplored. This capability is essential as it reflects models’ difficulty awareness and reasoning efficiency, both critical for real-world applications. We propose S1-Bench, a multi-domain, multilingual benchmark comprising model-simple system 1 questions. Our investigation of 28 LRMs reveals under-accuracy and inefficiency on system 1 problems. We find existing efficient reasoning methods either generalize poorly to simple questions or sacrifice performance for efficiency. Further exploration uncovers LRMs’ early difficulty awareness accompanied by lower confidence, and shows that problem difficulty is implicitly encoded in hidden states1.

### 1 Introduction

Recent advances in Large Reasoning Models (LRMs), notably OpenAI’s o1/o3 [OpenAI, 2024] and the DeepSeek-R1 [Guo et al., 2025] series, have propelled the development of Large Language Models (LLMs). Unlike traditional LLMs that exhibit intuitive, heuristic system 1 thinking, LRMs demonstrate deliberate and analytical system 2 reasoning [Qu et al., 2025a; Li et al., 2025b] by explicitly generating external chain-ofthought [Wei et al., 2022] before producing final answers.

While most prior research focuses on enhancing LRMs’ reasoning capabilities to achieve superior performance on complex tasks requiring system 2 [Yang et al., 2025a; Li et al., 2025a; Yeo et al., 2025], exploration of their system 1 thinking abilities remains limited. Specifically, system 1 thinking refers to the human-like intuitive ability [Kahneman, 2011] to answer questions rapidly with minimal tokens usage. This capability reflects a model’s perception of task difficulty as well as its proficiency in efficiently responding to questions it recognizes as simple [Booch et al., 2021]. In real-world scenarios, where

∗ denotes equal contribution. † denotes corresponding author. 1The code can be found in https://github.com/WYRipple/S1 Bench.

Cross Realistic MultiDomain Scenarios lingual Acc.

Benchmark

AIME ✗ ✓ ✗ 6.67 GPQA ✓ ✓ ✗ 24.94 Olympiad ✓ ✓ ✓ 27.94 AMC ✗ ✓ ✗ 31.88 MATH500 ✗ ✓ ✗ 58.30 MMLU ✓ ✓ ✗ 66.27

GSM8K ✗ ✓ ✗ 87.45 ASDIV ✗ ✓ ✗ 97.51

GSM8K-zero ✗ ✗ ✗ 77.98 RoR-Bench ✗ ✗ ✗ 14.24

S1-Bench (ours) ✓ ✓ ✓ 100.00

Table 1: Characteristics of S1-Bench, with ‘Acc.’ representing the average accuracy of four 7-9B LLMs. See Appendix A.1 for more details.

the vast majority of user queries are relatively straightforward for models, ensuring accurate and efficient system 1 thinking is particularly crucial [Yang et al., 2024].

In this work, we aim to explore the system 1 thinking capabilities of LRMs, yet a suitable benchmark remains absent. Some studies attempt to evaluate LRMs using questions that humans consider simple (e.g., GSM8K-zero and RoR [Chiang and Lee, 2024; Yan et al., 2025a]), yet these questions remain challenging for models. Other works utilize single-domain benchmarks, such as basic mathematics (GSM8K and ASDIV [Cobbe et al., 2021; Miao et al., 2021]), but lack domain diversity. Other research on LRM efficiency primarily targets challenging problems, such as AIME. As shown in Table 1, the top six benchmarks which are most commonly used in these work remain difficult for small LLMs, making it unsuitable for exploring system 1 thinking capabilities.

To complement the exploration foundation, we introduce the System 1 Thinking Capability Benchmark (S1-Bench), which measures the performance of LRMs across various simple tasks that commonly encountered in real-world applications. S1-Bench has the following three characteristics: (1) Simple. The questions can be easily answered by LLMs. As shown in Table 1, LLMs with 7-9B parameters can robustly provide correct answers across different temperatures. (2) Diverse. It encompasses four major categories and 28 subcategories in two languages. (3) Natural. The questions are clear, without any misleading elements or ambiguities, ensuring they can be answered intuitively.

|[Figure 1]<br><br>[Figure 2]<br><br>Discriminators<br><br>[Figure 3]<br><br>[Figure 4]<br><br>Annotators|
|---|

question 2 question 1 ... ... ...

... ... ...

[Figure 5]

[Figure 6]

category: reasoning question subcategory: numerical reasoning question: What is 7 minus 7? answer: 0

[Figure 7]

[Figure 8]

Subcategories

Generators questions answers

question 3

...

...

category: knowledge question subcategory: biographical facts question: What was the gender of chemist Marie Curie? answer: Female

Yes, reduce difficulty

No

Iterative time < 4

Delete

answers

category: instruction following

Annotators

discussion

subcategory: counting constraints question: Output the number of English letters in the word "cat". answer: 3

t = 0.0, 0.2, 0.4 Sample 10 times

No

[Figure 9]

[Figure 10]

category: analysis question

Yes

responses

questions

[Figure 11]

[Figure 12]

S1-Bench

ALL Correct?

subcategory: language classification question: Is the word "hello" English or Japanese? answer: English

[Figure 13]

[Figure 14]

questions-answers (retain/modify/discard )

Evaluator

Validators

Figure 1: Construction workflow for S1-Bench and an illustrative example from each major category.

We conduct extensive exploration on S1-Bench across 28 LRMs, yielding the following key findings: (1) Underaccuracy. Despite employing deep reasoning, several LRMs exhibit under-accuracy on simple questions, particularly in knowledge-based and instruction-following tasks, suggesting that long-chain reasoning may introduce catastrophic forgetting. (2) Inefficiency: Current LRMs lack system 1 capabilities across all categories of questions, with average output lengths 13.8× longer than small LLMs on S1-Bench. (3) Limitations of efficient reasoning methods: Existing efficient reasoning methods either suffer from limited generalization, failing to effectively compress length on S1-Bench, or fail to balance performance and efficiency, leading to significant drops in accuracy. (4) Early awareness of problem difficulty: We observe that LRMs often exhibit explicit difficulty-related comments prior to formal reasoning, indicating that models possess an early awareness of problem difficulty. Moreover, even when identifying a question as simple, LRMs still generate redundant responses with higher average token entropy, indicating stronger exploratory. These reveal a gap between models’ difficulty awareness and reasoning behavior. Furthermore, our representation-level analysis shows that question difficulty is implicitly encoded in model hidden states, as a simple single-layer MLP can capture difficulty-related signals.

Our contributions can be summarized as follows:

- • We propose S1-Bench, a benchmark dedicated to evaluating LRMs’ system 1 thinking capabilities, and a semiautomated construction workflow for system 1 questions.
- • We reveal LRMs’ under-accuracy and inefficiency in system 1 thinking, and highlight the inadequacy of current efficient reasoning methods, informing future research.
- • We explore LRMs’ early difficulty awareness, revealing correlations between explicit responses and model confidence, and disentanglement of difficulty in hidden states.

### 2 S1-Bench

We introduce S1-Bench, a bilingual, multi-domain benchmark designed to evaluate system 1 thinking capability of LRM on extremely simple questions. These questions are easily solvable by traditional LLMs. S1-Bench, which covers both English and Chinese, is organized into four major categories: reasoning (RSN), knowledge (KNO), instruction following (IF) and analysis (ANA), representing major dimensions commonly

employed in LLM capability evaluation [Zheng et al., 2023; Chang et al., 2024]. This section begins with how simplicity is ensured, then the detailed construction workflow for S1Bench, and concludes with an overview of the dataset statistics. Figure 1 shows the construction workflow and an illustrative example per category.

#### 2.1 How to Ensure Simplicity?

We ensure questions are simple and suitable for system 1 thinking through the following two aspects.

A Priori Simplicity Constraints

We begin by generating question–answer pairs through collaboration between humans and LLMs. Each pair is required to satisfy both the general and the category-specific simplicity criteria. The general simplicity criteria requires that: (1) Questions must be naturally and clearly expressed, unambiguous, and free of intentional traps. (2) Answers must be unique or easily falsifiable (e.g., providing a three-letter English word).

The category-specific simplicity criteria are as follows. RSN: Limited to problems solvable with minimal reasoning or intuition. KNO: Restricted to common knowledge with unique, verifiable answers from sources like Wikipedia. IF: Involve straightforward instructions without strict formatting requirements. ANA: Limited to questions whose answers can be directly inferred from the prompt, such as binary classification. These constraints ensure all questions remain straightforward for human respondents.

A Posteriori Simplicity Verification

Due to the biases existing between language models and humans [Gallegos et al., 2024], questions that are simple for humans may be difficult for LLMs. Therefore, we introduce additional posteriori verification to ensure that questions are simple enough to be correctly and robustly answered by smaller LLMs from different families.

#### 2.2 Construction Workflow

Subcategory Preparation. To ensure diversity, we refer to the subcategories included in existing benchmarks (e.g., MMLU, IFEval, and GSM8K) and evaluation surveys [Chang et al., 2024] to select, merge, or design subcategories for S1-bench, ensuring that each meets the simplicity requirements. The definition and example question for each subcategory can be found in Appendix A.2.

Model t=0.0 t=0.2 t=0.4 Tokens

Gemma2-9B 100.00 100.00 100.00 38.77 Llama3.1-8B 100.00 100.00 100.00 42.00 Mistral-8B 100.00 100.00 100.00 44.38 Qwen2.5-7B 100.00 100.00 100.00 42.81

DeepSeek-v3 100.00 100.00 100.00 79.53 Llama3.3-70B 100.00 99.76 99.76 53.71 Qwen2.5-14B 99.74 99.76 99.76 40.00 Qwen2.5-32B 99.98 99.98 99.98 43.17 Qwen2.5-72B 100.00 100.00 100.00 44.61 Qwen3-32B (w/o think) 100.00 100.00 100.00 103.30 Qwen3-14B (w/o think) 100.00 100.00 100.00 86.35 Qwen3-8B (w/o think) 100.00 100.00 99.76 90.54 Qwen3-1.7B (w/o think) 98.10 97.16 95.73 114.32

Table 2: Average accuracy (acc@k) and response token count of different LLMs, each sampled 10 times at three temperature settings.

Implementation of A Priori Simplicity. First, we use two data generators2 to create 100 initial bilingual questionanswer pairs for each candidate subcategory. The data generation prompt explicitly incorporates the subcategory definitions, along with both the general and category-specific simplicity criteria, while also aiming to ensure diversity in the generated questions. Second, these question–answer pairs are then independently evaluated by three annotators and two quality discriminators3 according to the general and category-specific simplicity criteria, resulting in five evaluation outcomes per pair. The three annotators are experienced graduate students familiar with LLMs and well-acquainted with the goals of S1-Bench. Finally, based on these evaluation outcomes, three annotators discuss and collectively decide whether to retain, modify, or discard each question (Details in Appendix A.4).

Implementation of A Posteriori Simplicity. First, each question obtained from the previous stage is input into the small LLM validators4 with 7˜9 B parameters. For each question, we sample 10 answers at three different temperature settings (0, 0.2, and 0.4), resulting in a total of 30 responses per question. These responses are then individually evaluated for correctness using GPT-4o. Second, if all 30 sampled responses are correct, the question is accepted into S1-Bench. Otherwise, the question is returned to the generators, where a difficulty-reduction prompt is applied to simplify it. The simplified questions then undergoes the same subsequent process. Finally, questions fail to meet the full-accuracy criterion (i.e., 30 out of 30 correct) after three rounds of difficulty reduction are excluded from the workflow.

The final S1-Bench comprises questions validated by both human a priori constraints and LLM a posteriori verification. All prompts are in Appendix A.3.

#### 2.3 Benchmark Statistics

S1-Bench comprises 422 question-answer pairs across four major categories and 28 subcategories, balanced with 220 English and 202 Chinese questions, with questions averaging 14.46 tokens. To ensure that the a posteriori verification process does not introduce simplicity only tailored to the small

- 2We select Claude-3.7-Sonnet and Qwen2.5-72B-Instruct.
- 3We select GPT-4o and DeepSeek-V3-241226.
- 4We select four small LLMs: Qwen2.5-7B, Llama3.1-8B, Mistral-8B, and

Gemma2-9B. The full model IDs are detailed in Appendix Table 12.

Model ID size pass@1↑ acc@k↑ f-acc↑ Tokens ↓ Validator LLMs 7-9B 100.00 100.00 – 42.00

close-source LRMs

claude-sonnet-4 – 100.00 100.00 100.00 166.32 claude-3.7-sonnet – 99.95 99.76 100.00 178.65 gemini-2.5-flash – 98.48 98.01 100.00 309.03 o4-mini – 99.53 99.53 100.00 129.99 Hunyuan-T1 – 99.91 99.53 100.00 542.31

open-source LRMs QwQ-32B 32B 100.00 100.00 100.00 720.10 Qwen3-A22B 235B 99.91 99.76 100.00 701.65 Qwen3-A3B 30B 99.95 99.76 100.00 638.40 Qwen3-32B 32B 99.91 99.53 99.91 668.69 Qwen3-14B 14B 99.95 99.76 99.95 582.99 Qwen3-8B 8B 99.95 99.76 99.95 657.76 Qwen3-1.7B 1.7B 99.34 97.39 99.81 595.90 DS-R1 671B 100.00 100.00 100.00 646.40 DS-R1-70B 70B 99.38 96.92 99.91 453.81 DS-R1-32B 32B 99.72 98.82 100.00 429.91 DS-R1-14B 14B 99.57 97.87 100.00 475.46 DS-R1-8B 8B 97.39 97.16 99.53 452.11 DS-R1-7B 7B 95.21 85.78 99.24 454.55 DS-R1-1.5B 1.5B 81.47 54.50 97.58 489.54 Nemotron-49B 49B 99.15 97.39 100.00 362.54 Nemotron-8B 8B 79.81 59.00 84.31 372.57 L-R1-32B 32B 94.74 79.62 95.07 1095.36 L-R1-32B-DS 32B 99.57 98.10 99.81 524.12 L-R1-14B-DS 14B 99.05 95.97 99.19 693.19 L-R1-7B-DS 7B 94.64 83.65 99.67 496.47 s1.1-32B 32B 99.48 98.10 99.53 998.00 s1.1-14B 14B 97.25 91.94 97.39 839.86 s1.1-7B 7B 88.58 63.98 88.96 711.49

Table 3: Accuracy results on S1-Bench, sorted by model family. Bold teal denotes the best performance, teal the second best, bold burgundy the worst, and burgundy the second worst.

LLM validator, we evaluate S1-Bench on five additional LLMs and on Qwen3 Family with reasoning modes disabled. As shown in Table 2, even the 1.7B model achieves over 98% accuracy at temperature 0.

### 3 Experiment Setting

#### 3.1 Baseline Models and Configurations

We evaluated 28 different LRMs, which are explicitly trained to first respond with a thinking process, and then generate a final answer. The evaluated LRMs include both open-source models—such as DeepSeek [Guo et al., 2025], Qwen [Yang et al., 2025a], Nemotron [Singhal et al., 2025], Light-R1 [Wen et al., 2025], and s1.1 [Muennighoff et al., 2025]—and closedsource models including claude-sonnet-4-20250514, claude-37-sonnet-20250219, gemini-2.5-flash-preview-09-25, o4-mini, and Hunyuan-T1 [Tencent, 2025], covering parameter scales from 1.5B to 671B.5 For each model, we adopt a temperature of t = 0.6, top-p = 0.95, and a sampling size of k = 5, with the maximum generation length set to 10,000 tokens. As a supplementary exploration, greedy sampling results with temperature t=0 are also provided in the Appendix C.1.

#### 3.2 Evaluation Metrics

Format Metrics. We measure format accuracy (f-acc) as the proportion of responses that follow the required output format, averaged over five runs. A response is considered valid if an

5Model details are provided in Appendix Table 13.

Model pass@1 RSN KNO IF ANA DS-R1-1.5B 81.5 86.3 75.2 76.8 84.4 Nemotron-8B 79.8 84.8 71.3 76.2 83.7 s1.1-7B 88.6 93.1 83.7 82.9 91.4 Average 83.3 88.1 76.7 78.6 86.5

- Table 4: Pass@1 performance across different question types on S1Bench. Only models with pass@1 below 90% on S1-Bench are reported.

end-of-thinking marker (e.g., </think>) correctly separates the reasoning process from a non-empty final answer.

Efficiency Metrics. We report the average response length in tokens (Tokens), excluding samples with endless thinking, with all token counts computed using the Qwen2.5 tokenizer to ensure fair comparison.

Accuracy Metrics. We calculate accuracy metrics only for responses with correct formatting. Following the LLM-asjudge setting [Zheng et al., 2023; Zhang et al., 2023], we use GPT-4o as the evaluator to assess the final answer part. We use two metrics: pass@1 and acc@k. pass@1 follows the definition in DeepSeek-R1 [Guo et al., 2025], computing the average accuracy over k samples, while acc@k is the percentage of questions where all k answers are correct. Both metrics use k=5. Notably, f-acc represents the upper bound for pass@1 and acc@k under the formatting requirement. All details regarding formatting and evaluation can be found in the Appendix B.

### 4 Main Experiment

In this section, we conduct a comprehensive evaluation of LRMs on S1-Bench from three perspectives: accuracy (Table 3), efficiency (Table 5), and the impact of RL-based efficient reasoning algorithms (Table 6).

#### 4.1 Accuracy Analysis

We analyze the accuracy-related results, and as shown in Table 3, we obtain the following observations:

Although the vast majority of larger-scale or closedsource models can maintain the same level of accuracy as small LLM-based validators (often exceeding 99% accuracy), some smaller LRMs suffer from severe performance degradation. For example, DS-R1-1.5B and Nemotron-8B achieve acc@k values only slightly above 50%. This behavior directly contradicts the original goal of LRMs: scaling thinking length can actually undermine accuracy on System 1 problems. Moreover, many LRMs exhibit limited robustness, with pronounced gaps between pass@1 and acc@k consistently observed across models, including DS-R1-1.5B (↓26.97%), L-R1-32B (↓15.12%), DS-R1-7B (↓9.43%), LR1-7B-DS (↓10.99%), Nemotron-8B (↓20.81%), and s1.1-7B (↓24.60%). Finally, certain models suffer from output format misalignment, as evidenced by format correctness below 90% for Nemotron-8B and s1.1-7B.

Accuracy degradation is particularly pronounced on instruction-following and knowledge-based questions. We conduct an question type analysis on models whose pass@1 falls below 90%. As shown in Table 4, a clear performance gap emerges across categories: instruction-following

FA Correct & TP Correct

FA Incorrect & TP mentions Correct Answer

FA Correct & TP Incorrect

FA Incorrect & TP omits Correct Answer

100%

Percentage(%)

80%

Qwen3-A22BQwen3-A3BQwQ-32BQwen3-32BQwen3-14BQwen3-8BQwen3-1.7BDS-R1DS-R1-70BDS-R1-32BDS-R1-14BDS-R1-8BDS-R1-7BDS-R1-1.5BL-R1-32BL-R1-32B-DSL-R1-14B-DSL-R1-7B-DSs1.1-32Bs1.1-14Bs1.1-7BNemo-49BNemo-8B

Figure 2: Distribution of the thinking process across four categories. FA and TP refer to Final Answer and Thinking Process, respectively. Green bars indicate cases where the final answer is correct, while red bars indicate cases where it is incorrect.

and knowledge questions reach average accuracies of only 76.7% and 78.6%, respectively, versus 88.1% on reasoning problems. This pattern suggests that long-chain reasoning does not consistently improve model behavior and may instead weaken instruction-following ability and factual reliability.

We further analyze the error cases and types in the responses. Specifically, we use DeepSeek-v3 to categorize LRM responses into four cases and compute their proportions. (1) Final answer correct; thinking process entirely accurate. (2) Final answer correct; thinking process contains intermediate errors. (3) Final answer incorrect; correct answer mentioned in thinking process. (4) Final answer incorrect; correct answer never mentioned in thinking process. The classification details are in Appendix C; results are shown in Figure 2. Key findings include: (1) Lower-accuracy LRMs tend to produce less reliable reasoning chains; even when they arrive at the correct final answer, their intermediate steps often contain errors (light green). LRMs with high accuracy (e.g., DS-R1) show almost no flawed reasoning steps, whereas those with lower accuracy (e.g., DS-R1-1.5B) often generate incorrect intermediate conclusions, further indicating that they lack robust reasoning ability. (2) Although LRMs sometimes mention the correct answer during reasoning, they may deviate and ultimately produce incorrect final answers (light red). We present representative error cases in the Appendix D. For example, in one case, the LRM initially arrived at the correct answer but undermined it through excessive verification. In another case, the LRM directly denied the correct answer that appeared during reasoning.

#### 4.2 Efficiency Analysis

As shown in Table 3, we present the average response tokens for each problem category in S1-Bench sampled five times (2.11k sample points per model). We observe that: (1) all LRMs exhibit overthinking on System 1 problems. While closed-source LRMs show higher efficiency, they still consume over three times more tokens. Open-source LRMs are considerably less efficient, with s1.1 and Light-R1 family models exhibiting severe overthinking to achieve better performance on complex problems. (2) LRMs within the same family show similar efficiency without demonstrating clear scaling phenomena. Across DS, Qwen3, and Nemotron, response lengths cluster by model family rather than by size, showing no clear correlation between model scale and thinking length. This suggests that training methods and data compo-

###### S1-Bench-EN S1-Bench-ZH

RSN KNO IF ANA Avg RSN KNO IF ANA Avg Avg

Model ID size

Gemma2-9B 9B 74.8 29.4 5.3 52.4 45.9 51.6 19.8 7.5 35.1 31.0 38.8 Llama3.1-8B 8B 91.0 35.4 12.4 61.9 56.0 44.0 28.3 15.2 18.7 26.7 42.0 Qwen2.5-7B 7B 65.5 46.3 6.4 49.6 46.5 50.5 46.6 9.8 36.9 38.8 42.8 Mistral-8B 8B 67.2 55.5 8.6 50.1 49.6 47.3 56.1 14.8 29.7 38.7 44.4

Column Avg - 74.6 41.6 8.2 53.5 49.5 48.3 37.7 11.8 30.1 33.8 42.0

o4-mini – 180.5 114.2 63.1 181.1 147.2 148.1 105.9 33.5 122.0 111.2 130.0 claude-sonnet-4 – 188.5 127.5 66.4 225.9 168.2 180.5 149.5 65.6 204.0 164.3 166.3 claude-3.7-sonnet – 190.8 161.6 89.8 222.4 179.2 180.0 166.9 100.3 216.3 178.1 178.7 gemini-2.5-flash – 295.1 141.8 122.2 398.6 268.1 383.0 220.0 271.3 465.3 353.6 309.0 Nemotron-49B 49B 599.7 587.6 396.5 526.1 540.4 232.9 157.3 235.5 107.8 168.8 362.5 Nemotron-8B 8B 561.0 585.1 458.0 303.1 462.6 369.5 326.0 288.1 166.7 273.5 372.6 DS-R1-32B 32B 421.8 504.4 414.7 521.1 473.7 362.2 385.6 343.1 408.8 382.2 429.9 DS-R1-8B 8B 472.2 528.9 530.7 462.7 491.2 521.9 404.4 266.2 395.5 409.4 452.1 DS-R1-70B 70B 464.1 501.3 378.5 536.1 484.0 450.8 450.2 328.4 416.7 420.9 453.8 DS-R1-7B 7B 447.5 623.9 353.8 510.0 495.5 446.5 463.2 339.5 373.0 409.4 454.5 DS-R1-14B 14B 503.7 674.7 367.3 494.2 519.0 452.0 465.4 375.3 405.8 428.0 475.5 DS-R1-1.5B 1.5B 480.8 584.7 417.4 577.2 529.1 493.0 497.4 329.8 423.1 446.0 489.5 L-R1-7B-DS 7B 568.1 667.1 501.7 566.3 580.3 444.8 454.6 344.1 366.4 405.0 496.5 L-R1-32B-DS 32B 574.5 706.6 647.6 632.8 636.3 431.2 367.0 377.1 418.7 402.2 524.1 Hunyuan-T1 – 561.6 693.8 380.9 435.0 521.2 676.8 553.8 505.1 523.8 565.3 542.3 Qwen3-14B 14B 700.4 639.5 286.2 575.0 579.8 730.4 557.2 403.1 586.0 586.5 583.0 Qwen3-1.7B 1.7B 790.4 720.6 399.9 526.2 624.6 689.8 563.6 406.4 545.9 564.7 595.9 Qwen3-A3B 30B 745.0 729.3 328.1 594.8 625.7 773.7 655.8 453.7 648.6 652.2 638.4 DS-R1 671B 786.1 723.8 711.4 529.2 672.5 727.3 638.5 607.9 533.9 617.9 646.4 Qwen3-8B 8B 853.7 753.1 394.4 629.5 683.2 749.2 623.8 459.3 624.0 630.0 657.8 Qwen3-32B 32B 805.7 774.2 356.9 645.5 674.7 780.2 695.2 446.6 645.3 662.1 668.7 L-R1-14B-DS 14B 951.0 1026.0 829.8 653.5 848.2 594.7 610.1 442.2 451.7 525.7 693.2 Qwen3-A22B 235B 925.3 864.3 487.2 605.7 734.5 803.3 713.4 487.2 611.3 665.9 701.7 s1.1-7B 7B 1039.5 840.8 1923.2 529.4 929.9 489.6 351.3 1034.3 332.4 475.6 711.5 QwQ-32B 32B 873.3 808.1 520.8 634.7 722.4 866.9 707.3 613.3 667.7 717.6 720.1 s1.1-14B 14B 871.8 746.2 2233.1 708.1 960.2 654.6 546.0 1512.6 579.7 710.7 839.9 s1.1-32B 32B 1077.9 889.7 2055.4 781.7 1081.7 995.6 765.2 1634.6 666.5 906.5 998.0 L-R1-32B 32B 1614.0 1217.3 1996.9 930.1 1338.3 1035.6 737.7 1240.7 610.2 835.3 1095.4

Column Avg - 713.8 665.2 673.1 538.0 634.0 620.1 484.2 588.3 445.2 516.7 577.6 Improvement - ×9.6 ×16.0 ×82.1 ×10.1 ×12.8 ×12.8 ×12.8 ×49.9 ×14.8 ×15.3 ×13.8

- Table 5: Average response tokens in the top-p sampling setting on the S1-bench across two languages and four main categories. Bold teal marks best performance, teal second best, bold burgundy worst, and burgundy second worst. Bold represents the maximum Improvement value for each language.

Model ID

S1-Bench MATH500 AIME 24 Tok. p@1 RNT Tok. p@1 RNT Tok. p@1 RNT

1.5B Model DS-R1-1.5B 489 81.5 0.0 5534 82.1 0.0 12339 28.1 0.0 Shorterbetter-1.5B 143 86.7 0.0 1131 74.8 0.0 4328 18.9 0.0 Laser-D-1.5B 353 85.6 0.0 1872 84.2 0.0 5750 34.2 0.0 TLMRE-1.5B 185 87.2 0.0 2376 84.9 0.0 9459 31.6 0.0 Adaptthink-1.5B 365 84.8 23.8 1782 82.0 76.8 6670 31.0 40.4 Autothink-1.5B 111 60.1 91.4 2195 84.0 71.1 9514 34.6 6.4

7B Model DS-R1-7B 454 95.2 0.0 3593 92.0 0.0 10490 52.9 0.0 Shorterbetter-7B 163 97.2 0.0 1210 86.6 0.0 5288 53.3 0.0 Laser-D-7B 351 97.3 0.0 1836 92.2 0.0 5379 58.3 0.0 TLMRE-7B 270 97.4 0.0 2073 91.8 0.0 9410 52.3 0.0 Adaptthink-7B 317 97.0 26.1 1875 92.0 76.6 8051 55.6 6.3 Autothink-7B 66 91.8 93.3 2146 91.2 12.0 8599 54.8 3.0

- Table 6: Results of RL-based methods for think-token reduction. p@1 denotes pass@1 and RNT the ratio of no-thinking responses.

of LRMs exceeds that of LLMs by more than a factor of ten. Response lengths also increase significantly across all subcategories (see Appendix C.2). This suggests that while LRMs are primarily trained on reasoning data to produce long CoT style responses, this stylistic pattern generalizes well across a wide range of question types. (2) LRMs exhibit the most significant increase in tokens for instruction following questions and tend to over-explore when the solution space is vast. As shown in Table 5, although small LLMs provide the most concise responses to instruction following questions, LRMs generate dramatically longer outputs—82.1 × longer in English and 49.9 × longer in Chinese than small LLMs. To investigate the cause, we further analyze the subcategories of instruction following questions. As shown in Appendix C.2, average tokens is notably longer in the subcategories of length constraints, character constraints, and sentence constraints. These three question types share a similar characteristic: their correctness is easy to verify, but the solution space is vast. We find that, although the model quickly identifies a correct answer, it becomes trapped in the search space, continually exploring alternatives and failing to stop in time.

sition may be more critical factors in determining System 1 thinking effective.

##### Analysis across Question Types

We further present the average response tokens by fine-grained categories and languages, as shown in Table 5. The main findings are as follows: (1) LRMs exhibit a substantial increase in response length across all four major categories, 28 subcategories, and two languages. As shown in Table 5, for each of the four major categories, the average response length

##### Thinking Solution Analysis

To better understand the causes of inefficiency in LRMs on S1-Bench, we analyze the solution rounds of their thinking processes We first use DeepSeek-v36 to segment each thinking

6See Appendix C.8 for robustness experiments on automated analysis.

(a) First Round v.s. Additional Tokens

(b) Solution Round Distribution

|Round Tokens Additional Tokens Validator LLMs| |
|---|---|
| | |

First

4.1 3.8 4.0 4.2 4.1 3.8 4.5 4.3 4.8 4.2 5.4 5.1 5.1 5.2 5.3 5.6 5.8 6.1 6.3 9.7

DS-R1-32B

DS-R1-8B DS-R1-70B

DS-R1-7B DS-R1-1.5B L-R1-7B-DS

Nemotron-49B

DS-R1-14B Qwen3-14B

Nemotron-8B L-R1-32B-DS Qwen3-1.7B

Qwen3-A3B Qwen3-32B

Qwen3-8B Qwen3-A22B

DS-R1 QwQ-32B

L-R1-14B-DS L-R1-32B

0 200 400 600 800

0% 20% 40% 60% 80% 100%

Token Counts

Percentage

S=0 S=1 S=2 S=3 S=4 S=5 S=6 S=7 S=8 S>8

Figure 3: (a) Comparison of first round and additional token costs for each LRM. (b) Distribution of solution rounds for each LRM.

process into several solutions, each defined as a point at which LRMs explicitly arrives at a conclusion that matches the correct answer. We then compute the average token counts in the first solution. The detailed experimental setup is provided in Appendix C.3. Our analysis reveals the following: (1) The token consumed in the first solution of LRMs significantly exceeds that of validator LLMs. As shown in Figure 3 (a), this suggests that LRMs may involve unnecessary reasoning steps in first solution, which could be one of the reasons for their inefficiency. (2) The primary reason for efficiency gaps between LRMs lies in the number of redundant solution rounds they generate, rather than the token cost in the initial round. As shown in Figure 3 (a), although total thinking token counts vary widely across LRMs, their token counts in the initial round are similar and only account for a small fraction of the total. Figure 3 (b) further shows the distribution of solution rounds on S1-Bench, revealing that LRMs with longer thinking processes tend to generate more solution round, and this redundancy greatly increases computational cost. Furthermore, further experiments reveal that the redundancy in the reasoning process gradually increases over time. Furthermore, the similarity across solution rounds progressively increases as reasoning proceeds, suggesting that later rounds contribute increasingly redundant information. Detailed results can be found in Appendix C.4.

#### 4.3 Efficient Reasoning Algorithms Analysis

Recently, several methods have been proposed to mitigate overthinking in LRMs based on reinforcement learning (RL). We evaluate representative approaches on S1-Bench, including (1) RL with length-based penalties (e.g., Shorterbetter [Yi et al., 2025], Laser-D [Liu et al., 2025], and TLMRE [Arora and Zanette, 2025]), and (2) adaptive mode selection (e.g., Adaptthink [Zhang et al., 2025a] and Autothink [Tu et al., 2025]). Table 6 reports results for models fine-tuned from DSR1-1.5B/7B, leading to the following observations. (1) Token reduction on challenging benchmarks does not necessarily translate to similar gains on S1-Bench. For instance, LaserD-1.5B and AdaptThink-1.5B reduce reasoning tokens by nearly 50% on AIME2024, but achieve only about 25% reduction on S1-Bench, indicating limited generalization of these

easy-zh easy-en neutral-zh neutral-en difficult-zh difficult-en

120

100

80

Counts

60

40

20

0

Qwen3-A22BQwen3-A3BQwQ-32BQwen3-32BQwen3-14BQwen3-8BQwen3-1.7BHunyuan-T1DS-R1-70BDS-R1DS-R1-32BDS-R1-14BDS-R1-8BDS-R1-7BDS-R1-1.5BL-R1-32BL-R1-32B-DSL-R1-14B-DSL-R1-7B-DSs1.1-32Bs1.1-14BNemotron-49Bs1.1-7BNemotron-8BEXAONE-32BEXAONE-7.8BEXAONE-2.4BSky-T1-32B

RSN KNO

39%42%42%40%50%35%42%43%38%23%19%26%20%23%25%43%21%26%21%17% 6% 11%16%14%22%17%18%10% 42%28%42%30%33%31%28%27%36%18%21%27%19%17%12%33%25%29%10% 9% 7% 8% 10% 4% 5% 5% 13% 6% 13%22%19%17%16%13%19%19%21%22%22%33%24%33%13%14%22%25%35%22%16%13%10%10%17%14%13%14% 2% 3% 6% 3% 5% 3% 2% 4% 3% 18%18%19%18%21%13%21%10%10% 6% 1% 3% 1% 3% 1% 1% 2% 1% 3%

IF ANA

Qwen3-A22BQwen3-A3BQwQ-32BQwen3-32BQwen3-14BQwen3-8BQwen3-1.7BHunyuan-T1DS-R1-70BDS-R1DS-R1-32BDS-R1-14BDS-R1-8BDS-R1-7BDS-R1-1.5BL-R1-32BL-R1-32B-DSL-R1-14B-DSL-R1-7B-DSs1.1-32Bs1.1-14BNemotron-49Bs1.1-7BNemotron-8BEXAONE-32BEXAONE-7.8BEXAONE-2.4BSky-T1-32B

Figure 4: Top: Count of early difficulty awareness across models. Bottom: Probability of early difficulty awareness by question type.

methods to out-of-distribution problems. (2) Some methods, such as ShorterBetter and AutoThink, achieve substantial compression on S1-Bench at the cost of severe accuracy degradation. Specifically, ShorterBetter-1.5B attains only 18.9% pass@1 on AIME24, while AutoThink-1.5B achieves merely 60.1% accuracy on S1-Bench, suggesting difficulty in balancing reasoning efficiency with answer correctness.

### 5 Early Difficulty Awareness in LRMs

We observe that on S1-Bench, LRMs may form an early intuitive judgment of question difficulty before initiating formal reasoning. In some cases, models begin their responses with statements such as “This looks like a simple problem,” prior to any substantive reasoning. We refer to this phenomenon as early awareness of question difficulty. To better understand this behavior, we conduct the following analyses7.

Exploring Frequency and Stylistic Patterns of Difficulty Awareness. We employ GPT-4o to classify the initial portion of model responses—defined as the content before the first “\n\n”—based on whether and how the model comments on question difficulty. Each response is categorized into one of four types: easy, neutral, difficult, or no comment. Figure 4 presents the distribution and corresponding probabilities of these categories across four types of questions. From these results, we make several observations. First, all LRMs exhibit early difficulty awareness to varying degrees, with the phenomenon being particularly prominent in the Qwen, DeepSeek, and Light-R1 model families. Second, LRMs display clear stylistic differences in expressing such awareness. For instance, the Qwen family tends to characterize questions in S1Bench as easy, whereas DeepSeek-distilled models produce more diverse difficulty-related comments. Third, this early difficulty awareness is most evident in reasoning-oriented questions, while it appears less frequently in analytical questions. Exploring the Relationship Between Early Difficulty Awareness, Reasoning Length, and Entropy. To investigate whether the early sense of a question as “easy” leads to a corresponding reduction in response length, we compare

7See Appendix C.5, C.6, C.7 for experimental setup and details.

1600

600

|easy all easy - all|136<br><br>74<br><br>252256<br><br>145| |
|---|---|---|
| | | |
|126<br><br>74 92| | |
|26 14 14<br><br>51<br><br>18 5<br><br>0<br><br>22<br><br>-|9 76<br><br>-113<br><br>-10<br><br>16<br><br>-18<br><br>49| |

500

1200

400

Difference

300

Tokens

800

200

100

400

0

100

Qwen3-A22BQwen3-A3BQwQ-32BQwen3-32BQwen3-14BQwen3-8BQwen3-1.7BDS-R1-70BDS-R1DS-R1-32BDS-R1-14BDS-R1-8BDS-R1-7BDS-R1-1.5BL-R1-32BL-R1-32B-DSL-R1-14B-DSL-R1-7B-DSs1.1-32Bs1.1-14BNemotron-49Bs1.1-7BNemotron-8B

- Figure 5: Average response tokens in the easy category vs. all samples. Dots show difference: easy minus all.

| | | | | | |
|---|---|---|---|---|---|
| | | |e|asy| |
| | | |o e o<br><br>|thers asy avg thers a|. ent vg. ent|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0 250 500 750 1000 1250 1500

Token Count

0.5

1.0

1.5

2.0

AverageEntropy

DS-R1-1.5B

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | |e|asy| |
| | | | |o e<br><br>|thers asy avg|. ent|
| | | | |o|thers a|vg. ent|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0 250 500 750 1000 1250 1500

Token Count

0.2

0.4

0.6

0.8

1.0

1.2

1.4

1.6

AverageEntropy

DS-R1-7B

| | | | | | |
|---|---|---|---|---|---|
| | | |e|asy| |
| | | |o e<br><br>|thers asy avg|. ent|
| | | |o|thers a|vg. ent|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0 250 500 750 1000 1250 1500

Token Count

0.25

0.50

0.75

1.00

1.25

1.50

AverageEntropy

DS-R1-14B

- Figure 6: Average response entropy of easy and other responses across three LRMs.

the average response tokens for questions in the easy category versus all samples. The results are shown in Figure 5. Except for L-R1-32B, the remaining LRMs do not exhibit a noticeable decrease in response length when questions are perceived as easy. On the contrary, 18 out of 23 LRMs generate longer responses under this condition. We further analyze the average token entropy of responses associated with easy judgments, with the results presented in Figure 6. Interestingly, these responses exhibit higher average entropy, indicating stronger exploratory behavior during generation. One possible explanation is that models may exhibit self-doubt toward their own early difficulty judgments, leading them to engage in increased exploration and produce longer responses, despite initially perceiving the question as simple. This suggests a discrepancy between the LRM’s initial sense of difficulty and its generative behavior, the causes and improvements of which warrant further investigation.

Exploring Activation Pattern Differences Across Question Difficulty. In contrast to the preceding analyses based on explicit difficulty-related cues in model outputs, we further examine whether question difficulty is reflected at the representation level through systematic differences in model activation patterns. Specifically, using DS-R1-1.5B and DS-R1-7B as representative examples, we analyze whether the final-layer representation of the last token in the encoded question varies systematically with question difficulty. To this end, we conduct a probing-based analysis by training a single-layer MLP as a weak linear classifier with limited training data, which serves solely to assess whether hidden representations contain information predictive of question difficulty. The training set is constructed by selecting one Chinese and one English example from each subcategory of S1-Bench as easy questions, and sampling an equal number of bilingual questions from AIME25 as hard questions, where Chinese examples are obtained via translation from English. We then evaluate the trained probe on the MATH500 dataset, which naturally contains five difficulty levels. For each level, we compute

Model Level 1 Level 2 Level 3 Level 4 Level 5

DS-R1-1.5B 46.5 56.7 56.7 79.7 92.5 DS-R1-7B 27.9 43.3 60.9 81.3 91.1

Table 7: Proportion (%) of questions classified as difficult across five difficulty levels on the MATH500 dataset.

the proportion of questions classified as difficult. As shown in Table 7, this proportion increases consistently with question difficulty, indicating that model representations implicitly encode structured information aligned with problem difficulty.

### 6 Related work

- 6.1 Large Reasoning Models

Large Reasoning Models (LRMs), characterized by explicitly generating external thinking processes before final answers [Kumar et al., 2025b; Chen et al., 2025a], achieve a paradigm shift from intuitive system 1 thinking to deliberative system 2 reasoning compared to traditional LLMs [Li et al., 2025b; Qu et al., 2025a], thus achieving superior performance on complex tasks. The development of recent LRMs has largely followed two main approaches: large-scale reinforcement learning (RL) and model distillation. Models trained via RL [Ai et al., 2026; Team et al., 2025b; Nie et al., 2026] leverage reward-based optimization to gradually incentivize deliberative reasoning. In contrast, distillation-based LRMs [OpenAI, 2024; Min et al., 2024; Team, 2025; Ye et al., 2025; Muennighoff et al., 2025; Zhang et al., 2025c] acquire such abilities by transferring structured reasoning patterns from advanced teacher models.

- 6.2 Limitations of LRMs

While LRMs have shown significant performance gains through deliberate reasoning, rigid adherence to this overly cautious thinking can introduce new limitations. On one hand, intermediate reasoning steps can cause excessive token generation and unnecessary solving attempts [Chen et al., 2024b; Hashemi et al., 2025; Kumar et al., 2025a], even leading to redundancy in the hidden layers [Chen et al., 2025b]. On the other hand, LRMs’ performance can drop in specific contexts like safety scenarios [Jiang et al., 2025], agents [Zhang et al., 2026] and role-playing [Feng et al., 2025]. However, prior studies mainly evaluated LRMs on complex tasks that more suited for deliberative system 2 thinking. Our work examines how deliberative reasoning impacts extremely simple problems better matched to intuition-driven system 1 processing.

### 7 Conclusion

This paper extensively explores the system 1 thinking capabilities of LRMs. We construct S1-Bench, a simple, domaindiverse, and natural benchmark for evaluating system 1 thinking abilities. Our exploration reveals fundamental issues of LRMs on system 1 problems—under-accuracy and overreasoning—and exposes limitations of existing efficient training methods. We also demonstrate models’ inherent perception of system 1 problems from both linguistic and representational perspectives, providing insights for future research.

### Acknowledgments

We would like to thank the anonymous reviewers, the area chairs and program chairs for their valuable comments and efforts. This work is supported by the National Natural Science Foundation of China (Grant No. 62572465).

### References

[Aggarwal and Welleck, 2025] Pranjal Aggarwal and Sean Welleck. L1: Controlling how long a reasoning model thinks with reinforcement learning. arXiv preprint arXiv:2503.04697, 2025.

[Ai et al., 2026] Zhengyang Ai, Zikang Shan, Xiaodong Ai, Jingxian Tang, Hangkai Hu, and Pinyan Lu. Shape: Stageaware hierarchical advantage via potential estimation for llm reasoning, 2026.

[AI-MO, 2024] AI-MO. Amc 2023, 2024. [Arora and Zanette, 2025] Daman Arora and Andrea Zanette.

Training language models to reason efficiently. NeurIPS, 2025.

[Aytes et al., 2025] Simon A Aytes, Jinheon Baek, and Sung Ju Hwang. Sketch-of-thought: Efficient llm reasoning with adaptive cognitive-inspired sketching. arXiv preprint arXiv:2503.05179, 2025.

[Booch et al., 2021] Grady Booch, Francesco Fabiano, Lior Horesh, Kiran Kate, Jonathan Lenchner, Nick Linck, Andreas Loreggia, Keerthiram Murgesan, Nicholas Mattei, Francesca Rossi, et al. Thinking fast and slow in ai. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 15042–15046, 2021.

[Chang et al., 2024] Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, et al. A survey on evaluation of large language models. ACM transactions on intelligent systems and technology, 15(3):1–45, 2024.

- [Chen et al., 2024a] Xiaoshu Chen, Sihang Zhou, Ke Liang, and Xinwang Liu. Distilling reasoning ability from large language models with adaptive thinking. arXiv preprint arXiv:2404.09170, 2024.
- [Chen et al., 2024b] Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, et al. Do not think that much for 2 + 3 =? on the overthinking of o1-like llms. arXiv preprint arXiv:2412.21187, 2024.

- [Chen et al., 2025a] Qiguang Chen, Libo Qin, Jinhao Liu, Dengyun Peng, Jiannan Guan, Peng Wang, Mengkang Hu, Yuhang Zhou, Te Gao, and Wangxiang Che. Towards reasoning era: A survey of long chain-of-thought for reasoning large language models. arXiv preprint arXiv:2503.09567, 2025.
- [Chen et al., 2025b] Yilong Chen, Junyuan Shang, Zhenyu Zhang, Yanxi Xie, Jiawei Sheng, Tingwen Liu, Shuohuan Wang, Yu Sun, Hua Wu, and Haifeng Wang. Inner thinking transformer: Leveraging dynamic depth scaling to foster adaptive internal thinking. arXiv preprint arXiv:2502.13842, 2025.

[Chiang and Lee, 2024] Cheng-Han Chiang and Hung-yi Lee. Over-reasoning and redundant calculation of large language models. In Yvette Graham and Matthew Purver, editors, Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 2: Short Papers), pages 161–169, St. Julian’s, Malta, March 2024. Association for Computational Linguistics.

[Cobbe et al., 2021] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

[Cui et al., 2025] Yingqian Cui, Pengfei He, Jingying Zeng, Hui Liu, Xianfeng Tang, Zhenwei Dai, Yan Han, Chen Luo, Jing Huang, Zhen Li, et al. Stepwise perplexity-guided refinement for efficient chain-of-thought reasoning in large language models. arXiv preprint arXiv:2502.13260, 2025.

[Ding et al., 2025] Yifu Ding, Wentao Jiang, Shunyu Liu, Yongcheng Jing, Jinyang Guo, Yingjie Wang, Jing Zhang, Zengmao Wang, Ziwei Liu, Bo Du, et al. Dynamic parallel tree search for efficient llm reasoning. arXiv preprint arXiv:2502.16235, 2025.

[Feng et al., 2025] Xiachong Feng, Longxu Dou, and Lingpeng Kong. Reasoning does not necessarily improve roleplaying ability. arXiv preprint arXiv:2502.16940, 2025.

- [Fu et al., 2024] Yichao Fu, Junda Chen, Siqi Zhu, Zheyu Fu, Zhongdongming Dai, Aurick Qiao, and Hao Zhang. Efficiently serving llm reasoning programs with certaindex. arXiv preprint arXiv:2412.20993, 2024.
- [Fu et al., 2025] Yichao Fu, Junda Chen, Yonghao Zhuang, Zheyu Fu, Ion Stoica, and Hao Zhang. Reasoning without self-doubt: More efficient chain-of-thought through certainty probing. In ICLR 2025 Workshop on Foundation Models in the Wild, 2025.

[Gallegos et al., 2024] Isabel O Gallegos, Ryan A Rossi, Joe Barrow, Md Mehrab Tanjim, Sungchul Kim, Franck Dernoncourt, Tong Yu, Ruiyi Zhang, and Nesreen K Ahmed. Bias and fairness in large language models: A survey. Computational Linguistics, 50(3):1097–1179, 2024.

[Guo et al., 2025] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. DeepSeek-R1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

[Han et al., 2024] Tingxu Han, Zhenting Wang, Chunrong Fang, Shiyu Zhao, Shiqing Ma, and Zhenyu Chen. Token-budget-aware llm reasoning. arXiv preprint arXiv:2412.18547, 2024.

[Hashemi et al., 2025] Masoud Hashemi, Oluwanifemi Bamgbose, Sathwik Tejaswi Madhusudhan, Jishnu Sethumadhavan Nair, Aman Tiwari, and Vikas Yadav. Dna bench: When silence is smarter–benchmarking over-reasoning in reasoning llms. arXiv preprint arXiv:2503.15793, 2025.

[He et al., 2024] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu,

Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008, 2024.

- [Hendrycks et al., 2021a] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations, 2021.
- [Hendrycks et al., 2021b] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

[Hou et al., 2025] Bairu Hou, Yang Zhang, Jiabao Ji, Yujian Liu, Kaizhi Qian, Jacob Andreas, and Shiyu Chang. Thinkprune: Pruning long chain-of-thought of llms via reinforcement learning. arXiv preprint arXiv:2504.01296, 2025.

[Jain et al., 2025] Naman Jain, King Han, Alex Gu, WenDing Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. In The Thirteenth International Conference on Learning Representations, 2025.

[Jiang et al., 2025] Fengqing Jiang, Zhangchen Xu, Yuetai Li, Luyao Niu, Zhen Xiang, Bo Li, Bill Yuchen Lin, and Radha Poovendran. Safechain: Safety of language models with long chain-of-thought reasoning capabilities. arXiv preprint arXiv:2502.12025, 2025.

[Kahneman, 2011] Daniel Kahneman. Thinking, fast and slow. macmillan, 2011.

- [Kumar et al., 2025a] Abhinav Kumar, Jaechul Roh, Ali Naseh, Marzena Karpinska, Mohit Iyyer, Amir Houmansadr, and Eugene Bagdasarian. Overthink: Slowdown attacks on reasoning llms. arXiv e-prints, pages arXiv–2502, 2025.
- [Kumar et al., 2025b] Komal Kumar, Tajamul Ashraf, Omkar Thawakar, Rao Muhammad Anwer, Hisham Cholakkal, Mubarak Shah, Ming-Hsuan Yang, Phillip HS Torr, Salman Khan, and Fahad Shahbaz Khan. Llm post-training: A deep dive into reasoning large language models. arXiv preprint arXiv:2502.21321, 2025.

[Lee et al., 2025] Ayeong Lee, Ethan Che, and Tianyi Peng. How well do llms compress their own chain-ofthought? a token complexity approach. arXiv preprint arXiv:2503.01141, 2025.

- [Li et al., 2024] Yiwei Li, Peiwen Yuan, Shaoxiong Feng, Boyuan Pan, Xinglin Wang, Bin Sun, Heda Wang, and Kan Li. Escape sky-high cost: Early-stopping self-consistency for multi-step reasoning. arXiv preprint arXiv:2401.10480,

- 2024.

- [Li et al., 2025a] Dacheng Li, Shiyi Cao, Tyler Griggs, Shu Liu, Xiangxi Mo, Eric Tang, Sumanth Hegde, Kourosh Hakhamaneshi, Shishir G Patil, Matei Zaharia, et al.

Llms can easily learn to reason from demonstrations structure, not content, is what matters! arXiv preprint arXiv:2502.07374, 2025.

[Li et al., 2025b] Zhong-Zhi Li, Duzhen Zhang, Ming-Liang Zhang, Jiaxin Zhang, Zengyan Liu, Yuxuan Yao, Haotian Xu, Junhao Zheng, Pei-Jie Wang, Xiuyi Chen, et al. From system 1 to system 2: A survey of reasoning large language models. arXiv preprint arXiv:2502.17419, 2025.

[Liao et al., 2025] Baohao Liao, Yuhui Xu, Hanze Dong, Junnan Li, Christof Monz, Silvio Savarese, Doyen Sahoo, and Caiming Xiong. Reward-guided speculative decoding for efficient llm reasoning. arXiv preprint arXiv:2501.19324, 2025.

[Liu et al., 2025] Wei Liu, Ruochen Zhou, Yiyun Deng, Yuzhen Huang, Junteng Liu, Yuntian Deng, Yizhe Zhang, and Junxian He. Learn to reason efficiently with adaptive length-based reward shaping, 2025.

[Lu et al., 2025] Ximing Lu, Seungju Han, David Acuna, Hyunwoo Kim, Jaehun Jung, Shrimai Prabhumoye, Niklas Muennighoff, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, et al. Retro-search: Exploring untaken paths for deeper and efficient reasoning. arXiv preprint arXiv:2504.04383, 2025.

[Luo et al., 2025] Haotian Luo, Li Shen, Haiying He, Yibo Wang, Shiwei Liu, Wei Li, Naiqiang Tan, Xiaochun Cao, and Dacheng Tao. O1-pruner: Length-harmonizing finetuning for o1-like reasoning pruning. arXiv preprint arXiv:2501.12570, 2025.

- [Ma et al., 2025a] Wenjie Ma, Jingxuan He, Charlie Snell, Tyler Griggs, Sewon Min, and Matei Zaharia. Reasoning models can be effective without thinking. arXiv preprint arXiv:2504.09858, 2025.
- [Ma et al., 2025b] Xinyin Ma, Guangnian Wan, Runpeng Yu, Gongfan Fang, and Xinchao Wang. Cot-valve: Lengthcompressible chain-of-thought tuning. arXiv preprint arXiv:2502.09601, 2025.

[MAA Committees, ] MAA Committees. Aime problems and solutions. https://artofproblemsolving.com/wiki/i ndex.php/AIME Problems and Solutions.

[Miao et al., 2021] Shen-Yun Miao, Chao-Chun Liang, and Keh-Yih Su. A diverse corpus for evaluating and developing english math word problem solvers. arXiv preprint arXiv:2106.15772, 2021.

[Min et al., 2024] Yingqian Min, Zhipeng Chen, Jinhao Jiang, Jie Chen, Jia Deng, Yiwen Hu, Yiru Tang, Jiapeng Wang, Xiaoxue Cheng, Huatong Song, et al. Imitate, explore, and self-improve: A reproduction report on slow-thinking reasoning systems. arXiv preprint arXiv:2412.09413, 2024.

[Muennighoff et al., 2025] Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Cand`es, and Tatsunori Hashimoto. s1: Simple test-time scaling. arXiv preprint arXiv:2501.19393, 2025.

[Munkhbat et al., 2025] Tergel Munkhbat, Namgyu Ho, Seo Hyun Kim, Yongjin Yang, Yujin Kim, and Se-Young

Yun. Self-training elicits concise reasoning in large language models. URL https://arxiv. org/abs/2502.20122,

- 2025.

[Nie et al., 2026] Shuaiyi Nie, Siyu Ding, Wenyuan Zhang, Linhao Yu, Tianmeng Yang, Yao Chen, Weichong Yin, Yu Sun, Hua Wu, and Tingwen Liu. Attnpo: Attentionguided process supervision for efficient reasoning, 2026.

[OpenAI, 2024] OpenAI. Learning to reason with LLMs. https://openai.com/index/learning-to-reason-with-llms/, September 2024.

[Pan et al., 2025] Rui Pan, Yinwei Dai, Zhihao Zhang, Gabriele Oliaro, Zhihao Jia, and Ravi Netravali. Specreason: Fast and accurate inference-time compute via speculative reasoning. arXiv preprint arXiv:2504.07891, 2025.

[Patel et al., 2021] Arkil Patel, Satwik Bhattamishra, and Navin Goyal. Are nlp models really able to solve simple math word problems? In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2080–2094, 2021.

- [Qu et al., 2025a] Xiaoye Qu, Yafu Li, Zhaochen Su, Weigao Sun, Jianhao Yan, Dongrui Liu, Ganqu Cui, Daizong Liu, Shuxian Liang, Junxian He, et al. A survey of efficient reasoning for large reasoning models: Language, multimodality, and beyond. arXiv preprint arXiv:2503.21614, 2025.
- [Qu et al., 2025b] Yuxiao Qu, Matthew YR Yang, Amrith Setlur, Lewis Tunstall, Edward Emanuel Beeching, Ruslan Salakhutdinov, and Aviral Kumar. Optimizing test-time compute via meta reinforcement fine-tuning. arXiv preprint arXiv:2503.07572, 2025.

[Rein et al., 2024] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

[She et al., 2025] Jianshu She, Zhuohao Li, Zhemin Huang, Qi Li, Peiran Xu, Haonan Li, and Qirong Ho. Hawkeye: Efficient reasoning with model collaboration. arXiv preprint arXiv:2504.00424, 2025.

- [Shen et al., 2025a] Yi Shen, Jian Zhang, Jieyun Huang, Shuming Shi, Wenjing Zhang, Jiangze Yan, Ning Wang, Kai Wang, and Shiguo Lian. Dast: Difficulty-adaptive slow-thinking for large reasoning models. arXiv preprint arXiv:2503.04472, 2025.
- [Shen et al., 2025b] Zhenyi Shen, Hanqi Yan, Linhai Zhang, Zhanghao Hu, Yali Du, and Yulan He. Codi: Compressing chain-of-thought into continuous space via self-distillation. arXiv preprint arXiv:2502.21074, 2025.

[Singhal et al., 2025] Soumye Singhal, Jiaqi Zeng, Alexander Bukharin, Yian Zhang, Gerald Shen, Ameya Sunil Mahabaleshwarkar, Bilal Kartal, Yoshi Suhara, Akhiad Bercovich, Itay Levy, et al. Llama-nemotron: Efficient reasoning models. In The Exploration in AI Today Workshop at ICML 2025, 2025.

[Talmor et al., 2019] Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4149–4158, 2019.

[Taubenfeld et al., 2025] Amir Taubenfeld, Tom Sheffer, Eran Ofek, Amir Feder, Ariel Goldstein, Zorik Gekhman, and Gal Yona. Confidence improves self-consistency in llms. arXiv preprint arXiv:2502.06233, 2025.

- [Team et al., 2025a] Kimi Team, A Du, B Gao, B Xing, C Jiang, C Chen, C Li, C Xiao, C Du, C Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. URL https://arxiv. org/abs/2501.12599, 2025.
- [Team et al., 2025b] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.

[Team, 2025] NovaSky Team. Sky-t1: Train your own o1 preview model within $450. https://novasky-ai.github.io/ posts/sky-t1, 2025.

[Tencent, 2025] Tencent. Reasoning efficiency redefined! meet tencent’s ‘hunyuan-t1’—the first mamba-powered ultra-large model. https://llm.hunyuan.tencent.com/#/ Blog/hy-t1/, March 2025.

[Tu et al., 2025] Songjun Tu, Jiahao Lin, Qichao Zhang, Xiangyu Tian, Linjing Li, Xiangyuan Lan, and Dongbin Zhao. Learning when to think: Shaping adaptive reasoning in r1-style models via multi-stage rl. arXiv preprint arXiv:2505.10832, 2025.

[Wan et al., 2024] Guangya Wan, Yuqi Wu, Jie Chen, and Sheng Li. Dynamic self-consistency: Leveraging reasoning paths for efficient llm sampling. arXiv preprint arXiv:2408.17017, 2024.

- [Wang et al., 2024] Xinglin Wang, Shaoxiong Feng, Yiwei Li, Peiwen Yuan, Yueqi Zhang, Chuyi Tan, Boyuan Pan, Yao Hu, and Kan Li. Make every penny count: Difficultyadaptive self-consistency for cost-efficient reasoning. arXiv preprint arXiv:2408.13457, 2024.
- [Wang et al., 2025a] Junlin Wang, Shang Zhu, Jon SaadFalcon, Ben Athiwaratkun, Qingyang Wu, Jue Wang, Shuaiwen Leon Song, Ce Zhang, Bhuwan Dhingra, and James Zou. Think deep, think fast: Investigating efficiency of verifier-free inference-time-scaling methods. arXiv preprint arXiv:2504.14047, 2025.

[Wang et al., 2025b] Yiming Wang, Pei Zhang, Siyuan Huang, Baosong Yang, Zhuosheng Zhang, Fei Huang, and Rui Wang. Sampling-efficient test-time scaling: Selfestimating the best-of-n sampling in early decoding. arXiv preprint arXiv:2503.01422, 2025.

[Wei et al., 2022] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning

in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

[Wen et al., 2025] Liang Wen, Yunke Cai, Fenrui Xiao, Xin He, Qi An, Zhenyu Duan, Yimin Du, Junchen Liu, Lifu Tang, Xiaowei Lv, et al. Light-R1: Curriculum sft, dpo and rl for long cot from scratch and beyond. arXiv preprint arXiv:2503.10460, 2025.

[Xia et al., 2025] Heming Xia, Yongqi Li, Chak Tou Leong, Wenjie Wang, and Wenjie Li. Tokenskip: Controllable chain-of-thought compression in llms. arXiv preprint arXiv:2502.12067, 2025.

- [Xu et al., 2025a] Jingxian Xu, Mengyu Zhou, Weichang Liu, Hanbing Liu, Shi Han, and Dongmei Zhang. Twt: Thinking without tokens by habitual reasoning distillation with multi-teachers’ guidance. arXiv preprint arXiv:2503.24198, 2025.
- [Xu et al., 2025b] Yuhui Xu, Hanze Dong, Lei Wang, Doyen Sahoo, Junnan Li, and Caiming Xiong. Scalable chain of thoughts via elastic reasoning. arXiv preprint arXiv:2505.05315, 2025.

- [Yan et al., 2025a] Kai Yan, Yufei Xu, Zhengyin Du, Xuesong Yao, Zheyu Wang, Xiaowen Guo, and Jiecao Chen. Recitation over reasoning: How cutting-edge language models can fail on elementary school-level reasoning problems? arXiv preprint arXiv:2504.00509, 2025.
- [Yan et al., 2025b] Yuchen Yan, Yongliang Shen, Yang Liu, Jin Jiang, Mengdi Zhang, Jian Shao, and Yueting Zhuang. Inftythink: Breaking the length limits of long-context reasoning in large language models. arXiv preprint arXiv:2503.06692, 2025.

- [Yang et al., 2024] Jingfeng Yang, Hongye Jin, Ruixiang Tang, Xiaotian Han, Qizhang Feng, Haoming Jiang, Shaochen Zhong, Bing Yin, and Xia Hu. Harnessing the power of llms in practice: A survey on chatgpt and beyond. ACM Transactions on Knowledge Discovery from Data, 18(6):1–32, 2024.
- [Yang et al., 2025a] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

- [Yang et al., 2025b] Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Zheng Lin, Li Cao, and Weiping Wang. Dynamic early exit in reasoning models. arXiv preprint arXiv:2504.15895, 2025.
- [Yang et al., 2025c] Junjie Yang, Ke Lin, and Xing Yu. Think when you need: Self-adaptive chain-of-thought learning. arXiv preprint arXiv:2504.03234, 2025.
- [Yang et al., 2025d] Wang Yang, Xiang Yue, Vipin Chaudhary, and Xiaotian Han. Speculative thinking: Enhancing small-model reasoning with large model guidance at inference time. arXiv preprint arXiv:2504.12329, 2025.
- [Yang et al., 2025e] Wenkai Yang, Shuming Ma, Yankai Lin, and Furu Wei. Towards thinking-optimal scaling of test-time compute for llm reasoning. arXiv preprint arXiv:2502.18080, 2025.

[Ye et al., 2025] Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. LIMO: Less is more for reasoning. arXiv preprint arXiv:2502.03387, 2025.

[Yeo et al., 2025] Edward Yeo, Yuxuan Tong, Morry Niu, Graham Neubig, and Xiang Yue. Demystifying long chain-of-thought reasoning in llms. arXiv preprint arXiv:2502.03373, 2025.

[Yi et al., 2025] Jingyang Yi, Jiazheng Wang, and Sida Li. Shorterbetter: Guiding reasoning models to find optimal inference length for efficient reasoning. arXiv preprint

- arXiv:2504.21370, 2025.

- [Yu et al., 2025a] Bin Yu, Hang Yuan, Yuliang Wei, Bailing Wang, Weizhen Qi, and Kai Chen. Long-short chainof-thought mixture supervised fine-tuning eliciting efficient reasoning in large language models. arXiv preprint

arXiv:2505.03469, 2025.

- [Yu et al., 2025b] Zhaojian Yu, Yinghao Wu, Yilun Zhao, Arman Cohan, and Xiao-Ping Zhang. Z1: Efficient test-time scaling with code. arXiv preprint arXiv:2504.00810, 2025.

[Zhang et al., 2023] Xinghua Zhang, Bowen Yu, Haiyang Yu, Yangyu Lv, Tingwen Liu, Fei Huang, Hongbo Xu, and Yongbin Li. Wider and deeper llm networks are fairer llm evaluators, 2023.

- [Zhang et al., 2025a] Jiajie Zhang, Nianyi Lin, Lei Hou, Ling Feng, and Juanzi Li. Adaptthink: Reasoning models can learn when to think. arXiv preprint arXiv:2505.13417, 2025.
- [Zhang et al., 2025b] Jintian Zhang, Yuqi Zhu, Mengshu Sun, Yujie Luo, Shuofei Qiao, Lun Du, Da Zheng, Huajun Chen, and Ningyu Zhang. Lightthinker: Thinking step-by-step compression. arXiv preprint arXiv:2502.15589, 2025.
- [Zhang et al., 2025c] Wenyuan Zhang, Tianyun Liu, Mengxiao Song, Xiaodong Li, and Tingwen Liu. SOTOPIA-Ω: Dynamic strategy injection learning and social instruction following evaluation for social agents, 2025.

[Zhang et al., 2026] Wenyuan Zhang, Xinghua Zhang, Haiyang Yu, Shuaiyi Nie, Bingli Wu, Juwei Yue, Tingwen Liu, and Yongbin Li. Expseek: Self-triggered experience seeking for web agents, 2026.

[Zheng et al., 2023] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-ajudge with MT-bench and chatbot arena. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023.

### A More Information of S1-Bench Construction

coreference resolution grammar classification

temporal reasoning

spatial reasoning

- A.1 Benchmark Statistics

We survey studies on improving the efficiency of LRMs, as there is potential overlap between these studies and the technical approaches aimed at enhancing system 1 thinking in LRMs. Table 14 presents the results of our survey. We compile the benchmarks used in these studies for evaluation, that are typically used to verify whether models achieve efficiency improvements. Benchmarks that appear more than four times include: MATH500 [Hendrycks et al., 2021b], GSM8K [Cobbe et al., 2021], AIME24/25 [MAA Committees, ], GPQA [Rein et al., 2024], AMC23 [AI-MO, 2024], MMLU [Hendrycks et al., 2021a], Olympiad-Bench [He et al., 2024], SVAMP [Patel et al., 2021], LiveCodeBench [Jain et al., 2025], and CommonSenseQA [Talmor et al., 2019].

The accuracy shown in Table 1 of the main paper is the average result of the four models, Qwen2.5-7B, Llama3.1-8B, Mistral-8B, and Gemma2-9B, at temperature 0, using GPT-4o as the evaluator.

- A.2 Subcategories in S1-Bench

- Figure 7 shows the pie chart distribution of 28 subcategories in S1-Bench. For more details on the subcategories, please refer to Table 15, 16. We present the names of all subcategories along with a representative question, making every effort to ensure orthogonality across all categories to improve the diversity of S1-Bench.

- A.3 Prompt for S1-Bench construction

This section presents the prompts used in the construction of S1-Bench, including the Initial Generation prompt, the Discriminating Generation Quality prompt, and the Reduce Difficulty prompt. See Table 17 for details.

- A.4 Human-involved Construction

set reasoning

intent recognition

numericalreasoning

languageclassification

namedentityrecognition

naturallawreasoning

reasoning question

###### analysis question

codereasoning

sentimentclassification

causalreasoning

scientificnotation

syntaxclassification

knowledge question

measurementunits

instruction following

topicclassification

historicalfacts

characterconstraints countingconstraints length constraints repetition constraints sentence constraints

geometryfacts

geographic facts

transformation constraints

creative authorship

biographical facts

Figure 7: S1-Bench Category Display. The inner circle represents four major categories, and the outer circle includes 28 subcategories. Detailed information is presented in 15 and 16.

#### B.3 GPT-4o and Human Evaluation

We use GPT-4o as the evaluator to assess the correctness of the responses. If a final answer can be isolated, only the final answer is evaluated; otherwise, the entire response is assessed. The evaluation prompt is provided in Table 18.

To evaluate the consistency between the GPT-4o judge’s assessments and human judgments, we conduct a comprehensive human evaluation study involving three of the authors. All participating authors are proficient in both English and Chinese. Specifically, we randomly sample 20 question-answer pairs from each model’s greedy decoding results, resulting in a dataset of 640 pairs derived from 32 models (including 4 verifier LLMs and 28 LRMs). The questions, reference answers, and model responses are then presented to three annotators, who independently judge the correctness of each model response. The final human evaluation results are determined through majority voting. Ultimately, the Cohen’s Kappa between the human evaluators and the GPT-4o judge is calculated to be 0.83, indicating an exceptionally high level of agreement.

All annotators are native Chinese speakers with high English proficiency. The annotators are well-versed in the objectives of benchmark construction and received concentrated training to ensure that all questions have unique answers and are as distinct and non-repetitive as possible. After the benchmark was finalized, all three annotators re-examined every question to ensure data quality, removing all inconsistent questions.

### B Baseline Models and Evaluation Details B.1 Baseline Model Details

#### B.4 Accuracy Metrics Details

Table 12 presents the abbreviations, IDs, and URLs of LLMs used in this paper. Table 13 displays the abbreviations, IDs, URLs, organizations, training algorithms, and training data volumes of open-source LRMs evaluated in this study.

Pass@1: Followed DeepSeek-R1 [Guo et al., 2025], we calculate pass@1 to assess the percentage of correct responses among the k=5 generations. Specifically, it is defined as:

#### B.2 Efficient Reasoning Algorithms Settings

We evaluate all RL-based methods using publicly released checkpoints from the original authors. We sample 5 responses per prompt on S1-Bench (max length 10,000), 1 on Math500 (max length 16,384), and 16 on AIME2024 (max length 16,384). All models use top-p sampling with temperature 0.6 temperature = 0.6 and p=0.95.

1 k

pass@1 =

k

pi, (1)

i=1

where pi is the correctness of the i-th generation.

Acc@k: Since S1-Bench is composed of extremely simple questions, we calculate acc@k. Specifically, acc@k=1 if all k responses are correct and acc@k = 0 otherwise. It is defined

Model size acc f-acc ↑ tokens ↓

Qwen3-235B-A22B 235B 100.00 100.00 702.70 Qwen3-30B-A3B 30B 100.00 100.00 636.35 QwQ-32B 32B 100.00 100.00 750.41 Qwen3-32B 32B 99.76 99.76 673.62 Qwen3-14B 14B 99.76 100.00 597.06 Qwen3-8B 8B 99.76 100.00 649.45 Qwen3-1.7B 1.7B 99.53 99.76 579.01 DS-R1 671B 100.00 100.00 621.89 DS-R1-70B 70B 99.76 100.00 469.78 DS-R1-32B 32B 100.00 100.00 428.46 DS-R1-14B 14B 99.29 99.76 463.52 DS-R1-8B 8B 97.39 99.53 452.11 DS-R1-7B 7B 94.31 97.87 436.87 DS-R1-1.5B 1.5B 76.54 91.94 473.67 Nemotron-49B 49B 99.53 100.00 337.94 Nemotron-8B 8B 77.73 81.99 446.62 L-R1-32B 32B 85.78 85.78 996.36 L-R1-32B-DS 32B 99.29 99.29 528.45 L-R1-14B-DS 14B 98.82 98.82 664.28 L-R1-7B-DS 7B 92.65 98.82 514.60 s1.1-32B 32B 98.82 98.82 983.38 s1.1-14B 14B 95.50 95.97 786.30 s1.1-7B 7B 87.68 87.91 630.52

Table 8: Main results in the greedy decoding setting on the S1-Bench, sorted by model family. Bold teal marks best performance, teal second best, bold burgundy worst, and burgundy second worst.

as:

acc@k =

k

pi, (2)

i=1

#### B.5 Types and Analysis of Format Errors

This section introduces a comprehensive taxonomy of format errors and emphasizes the importance of addressing these issues in future research, particularly in the design of format rewards for reinforcement learning. Unlike conventional LLMs, LRMs frequently exhibit format errors. These errors are defined by failing to use a unique end thinking marker (e.g., </think>) to separate the thinking process from the final answer. Format errors increase the difficulty of distinguishing the thinking process from the final answer and reveal the vulnerability of LRMs in following predefined formats.

To illustrate this phenomenon, we identify 12 distinct types of response formats produced by LRMs, each assigned a unique ID, as shown in Table 9. These 12 types are further grouped into three major categories:

- • Standard-Conforming Responses: These responses meet the expected format by including exactly one end thinking marker (e.g., </think>) to delimit the thinking process from the final answer. Among these, type ID100 includes a thinking process, while ID-101 omits it. The proportion of such responses is measured using the S-Corr metric.
- • Unreadable Responses: These refer to generation failures, including cases where LRMs produce endlessly thinking content or solely produce end thinking markers. The proportion of all other (i.e., readable) responses is measured using the L-Corr metric.

• Readable but Malformed Responses: These responses deviate from the standard format yet still contain extractable information. In some cases, the final answer is missing (e.g., ID-200, ID-202, ID-205), and we instead evaluate the correctness of the thinking process. In other cases, multiple (e.g., ID-201, ID-203) or unmatched8 (e.g., ID204, ID-206) end thinking markers are generated. In such instances, we treat the content following the last end thinking marker as the final answer for evaluation.

Table 19 and Table 20 present the distributions of 12 format types under top-p sampling and greedy sampling, respectively. we find: (1) The infinite generation phenomenon is widespread across most LRMs, particularly concentrated in LRMs with fewer than 32B parameters. (2) The Nemotron family frequently produces correctly formatted responses without any explicit thinking processes. This behavior can be viewed as a mechanism for mitigating over-thinking. (3) None of the evaluated LRMs exhibited behaviors classified as ID-205/206.

Format ID marker marker marker thinking final

(standard) (unmatched) (number) process answer Standard

- 100 √ – 1 √ √

- 101 √ – 1 ×

√

- 200 √ – 1 √

×

- 201 √ – >1 √ √

- 202 √ – >1 √

×

- 203 √ – >1 ×

√

- 204 ×

√

≥1 √ √

- 205 ×

√

≥1 √

×

- 206 ×

√

≥1 ×

√

- 207 × × 0 – √

Readable but Malformed

- 300 √ √

≥1 × ×

- 301 × × 0 – ∞ Table 9: Twelve types of response format.

Unreadable

### C More Experimental Setups & Results

#### C.1 Greedy Sampling Results

To distinguish between the two settings, we refer to the main text sampling as top-p sampling. Table 8 presents the performance of LRMs on S1-Bench under greedy sampling. While overall accuracy improves compared to top-p sampling, issues of inefficiency and accuracy degradation on simple questions remain.

#### C.2 Efficiency Analysis across Subcategories.

Figure 9 illustrates the average response tokens across the 28 subcategories. In the heatmap, both models (rows) and subcategories (columns) are ordered in descending order according to their average number of response tokens.

#### C.3 Solution Analysis Details

For solution analysis, We only use well-formatted thinking processes with correct final answers, as incorrect answers

8This paper provides a reference collection of unmatched end thinking makers: </ think>, </th think>, </ reason>, \nanswer\n ,**Final Answer** and **答案**.

Reasoning Question

Knowledge Question

Instruction Following

Analysis Question

0.8

0.8

0.8

0.8

MaximumSimilarityScore

MaximumSimilarityScore

MaximumSimilarityScore

MaximumSimilarityScore

0.7

0.7

0.7

0.7

0.6

0.6

0.6

0.6

0.5

0.5

0.5

0.5

0.4

0.4

0.4

0.4

2 4 6 8 10 12 14

2 4 6 8 10 12 14

2 4 6 8 10 12 14

2 4 6 8 10 12 14

Segment Number

Segment Number

Segment Number

Segment Number

DS-R1

DS-R1-14B

DS-R1-1.5B L-R1-7B-DS L-R1-14B-DS

L-R1-32B-DS

s1.1-14B s1.1-32B Nemotron-49B

Nemotron-8B

Qwen3-8B

Qwen3-30B-A3B

DS-R1-70B DS-R1-32B

DS-R1-8B DS-R1-7B

L-R1-32B

QwQ-32B

Qwen3-14B Qwen3-32B

Qwen3-235B-A22B

s1.1-7B

Qwen3-1.7B

Figure 8: Maximum similarity between each segment and all preceding segments for open-source LRMs across four categories.

Type Sentence Count

easy-zh 这个问题看起来挺简单的 308 easy-en that seems straightforward 36

difficult-zh 这个问题看起来有点复杂 308 difficult-en percentages can sometimes be tricky 7 neutral-zh 这个问题看起来好像不难 24 neutral-en Hmm, interesting 3

Table 10: The most common sentence in each type of early difficulty awareness.

make it unclear whether LRMs are over-reasoning or underreasoning, and malformed thinking processes cannot be precisely extracted. The segmentation process is performed by DeepSeek-v3, with prompts detailed in Table 21. We compute the average token count in the first solution round; if no solution is found, we use the token count of the entire thinking process.

#### C.4 Thinking Redundancy Analysis

We conduct a similarity analysis to analyze how information redundancy in the thinking processes changes as reasoning sequences increase. Specifically, we first divide the complete thinking process into k equal-length segments9. Then, we encode each segment using the all-MiniLM-L6-v2 model10. For each segment, we calculate the cosine similarity with all its preceding segments and use the maximum similarity as a measure of its information redundancy. As shown in

- Figure 8, information redundancy increases across all four main categories as reasoning sequences increase.

#### C.5 Error Analysis Details

In error analysis, we only use well-formatted samples, as malformed thinking processes cannot be precisely extracted. For samples with correct final answers, we categorize them based on whether the thinking process contains explicit incorrect conclusions in intermediate steps. For samples with incorrect final answers, we categorize them based on whether the correct answer is mentioned at least once during reasoning. We

9We set k=15, changing its value does not affect the conclusions. 10https://huggingface.co/sentence-transformers/allMiniLM-L6-v2

use DeepSeek-v3 for categorization, with prompts provided in Table 22.

#### C.6 Difficulty Awareness Analysis Details

We prompt GPT-4o to classify the initial part of model responses (before the first ‘\n\n’) into four types based on its comment on difficulty: easy, neutral, difficult, and no comment. The prompts for english question can be seen in Table 23. For Chinese queries, we use the translated version of the prompt in Chinese. In Table 10, we show the most common sentence of all LRMs in each type of early difficulty awareness.

#### C.7 Linear Classifier Training Configuration

We train the linear prob for two epochs using a learning rate of 1e-3 and binary cross-entropy loss. Training completes in less than 10 seconds on a single RTX 3090 GPU.

#### C.8 Robustness of Automated Analysis

Manual analysis of 28 models across two settings totaling 11.8K responses with an average length of 650 tokens each is impractical, while manually analyzing random subsets introduces bias. We aim to analyze all data to obtain robust conclusions.

When conducting these two sets of experiments, we randomly sampled 200 responses for extensive pre-experiments (different prompts, analysis models, etc.) and conducted manual review (by three authors) on the accuracy of error analysis and difficulty awareness extraction. Specifically, error analysis was modeled as a four-class classification task, and difficulty awareness was modeled as a four-class classification task (easy, neutral, difficult, and no comment). We found that DeepSeek-V3, which excels at instruction following, has dual advantages in both accuracy and cost compared to Qwen2.572B-Instruction and GPT-4o. Additionally, we carefully designed different prompts for Chinese and English and tried multiple approaches, as the two languages differ in expression habits and extraction boundaries. The manual review results of our final approach are shown in Table 11.

avg Acc Kappa avg Acc Kappa Input Price Output Price (error ana) (error ana) (diff awareness) (diff awareness) (API) (API)

Model

Qwen2.5-72B-Instruction 90.0% 0.51 94.5% 0.66 $0.54 $1.62 GPT-4o 93.0% 0.56 98.0% 0.70 $2.50 $10.00 DeepSeek-V3-0324 90.5% 0.63 97.5% 0.68 $0.27 $1.10

Table 11: Manual review results.

Model Model ID URL Qwen2.5-7B Qwen2.5-7B-Instruct https://huggingface.co/Qwen/Qwen2.5-7B-Instruct Llama3.1-8B Llama-3.1-8B-Instruct https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct Mistral-8B Ministral-8B-Instruct-2410 https://huggingface.co/mistralai/Ministral-8B-Instruct-2410 Gemma2-9B gemma-2-9b-it https://huggingface.co/google/gemma-2-9b-it

Qwen2.5-14B Qwen2.5-14B-Instruct https://huggingface.co/Qwen/Qwen2.5-14B-Instruct Qwen2.5-32B Qwen2.5-32B-Instruct https://huggingface.co/Qwen/Qwen2.5-32B-Instruct Qwen2.5-72B Qwen2.5-72B-Instruct https://huggingface.co/Qwen/Qwen2.5-72B-Instruct Llama3.3-70B Llama-3.3-70B-Instruct https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct DeepSeek-v3 DeepSeek-V3-0324 https://huggingface.co/deepseek-ai/DeepSeek-V3-0324

Table 12: Mapping of LLM abbreviations and IDs used in this paper, with their open-source URLs.

Model ID Abbreviation Base Model Alg. Size DeepSeek

DeepSeek-R1-Distill-Qwen-1.5B DS-R1-1.5B Qwen2.5-Math-1.5B SFT 800K DeepSeek-R1-Distill-Qwen-7B DS-R1-7B Qwen2.5-Math-7B SFT 800K DeepSeek-R1-Distill-Llama-8B DS-R1-8B Llama-3.1-8B SFT 800K DeepSeek-R1-Distill-Qwen-14B DS-R1-14B Qwen2.5-14B SFT 800K DeepSeek-R1-Distill-Qwen-32B DS-R1-32B Qwen2.5-32B SFT 800K DeepSeek-R1-Distill-Llama-70B DS-R1-70B Llama-3.3-70B-Instruct SFT 800K DeepSeek-R1 DS-R1 DeepSeek-V3-0324 SFT&RL 800K&–

Qwen

QwQ-32B QwQ-32B Qwen2.5-32B – – Qwen3-235B-A22B Qwen3-A22B Qwen3-235B-A22B-Base SFT&RL –&– Qwen3-30B-A3B Qwen3-A3B Qwen3-30B-A3B-Base SFT&RL –&– Qwen3-32B Qwen3-32B Qwen3-32B-Base SFT&RL –&– Qwen3-14B Qwen3-14B Qwen3-14B-Base SFT&RL –&– Qwen3-8B Qwen3-8B Qwen3-8B-Base SFT&RL –&– Qwen3-1.7B Qwen3-1.7B Qwen3-1.7B-Base SFT&RL –&–

qihoo360

Light-R1-7B-DS L-R1-7B-DS DeepSeek-R1-Distill-Qwen-7B SFT 3K Light-R1-14B-DS L-R1-14B-DS DeepSeek-R1-Distill-Qwen-14B SFT&RL 3K&– Light-R1-32B-DS L-R1-32B-DS DeepSeek-R1-Distill-Qwen-32B SFT 3K Light-R1-32B L-R1-32B Qwen2.5-32B-Instruct SFT&DPO 73K&–

simplescaling

s1.1-7B s1.1-7B Qwen2.5-7B-Instruct SFT 1K s1.1-14B s1.1-14B Qwen2.5-14B-Instruct SFT 1K s1.1-32B s1.1-32B Qwen2.5-32B-Instruct SFT 1K

NVIDIA

Llama-3.1-Nemotron-Nano-8B-v1 Nemotron-8B Llama-3.1-8B-Instruct SFT&RL –&– Llama-3.3-Nemotron-Super-49B-v1 Nemotron-49B Llama-3.3-70B-Instruct SFT&RL –&–

Table 13: The open-source LRMs details evaluated for S1-Bench.

### D Error Cases

This section presents several error cases observed in LRMs. See Tables 24, 25, 26, 27, 28, and 29.

Olympiad-

LiveCode- CommonBench Bench SenseQA

Paper Abbreviation MATH GSM8K AIME GPQA AMC MMLU

SVAMP

Codi [Shen et al., 2025b] ✓ ✓ CISC [Taubenfeld et al., 2025] ✓ ✓ ✓ CoT-Valve [Ma et al., 2025b] ✓ ✓ Dast [Shen et al., 2025a] ✓ ✓ ATM [Chen et al., 2024a] ✓ ✓ ✓ DEER [Yang et al., 2025b] ✓ ✓ ✓ ✓ DPTS [Ding et al., 2025] ✓ ✓ Dynasor [Fu et al., 2024] ✓ ✓ ESC [Li et al., 2024] ✓ ✓ ✓ ✓ Hawkeye [She et al., 2025] ✓ ✓ ✓ token complexity [Lee et al., 2025] ✓ ✓ ✓ INFTYTHINK [Yan et al., 2025b] ✓ ✓ ✓ KIMI K1.5 [Team et al., 2025a] ✓ ✓ ✓ ✓ L1 [Aggarwal and Welleck, 2025] ✓ ✓ ✓ ✓ ✓ LightThinker [Zhang et al., 2025b] ✓ ✓ ✓ LS-Mixture SFT [Yu et al., 2025a] ✓ ✓ ✓ DSC [Wang et al., 2024] ✓ ✓ O1-Pruner [Luo et al., 2025] ✓ ✓ MRT [Qu et al., 2025b] ✓ ✓ ✓ Self-Doubt [Fu et al., 2025] ✓ ✓ ✓ RASC [Wan et al., 2024] ✓ ✓ ✓ NoThinking [Ma et al., 2025a] ✓ ✓ ✓ ✓ Retro-Search [Lu et al., 2025] ✓ ✓ ✓ ✓ ✓ RSD [Liao et al., 2025] ✓ ✓ ✓ ✓ ✓ ST-BoN [Wang et al., 2025b] ✓ ✓ Elastic Reasoning [Xu et al., 2025b] ✓ ✓ ✓ ✓ ✓ FS-BoN [Munkhbat et al., 2025] ✓ ✓ SoT [Aytes et al., 2025] ✓ ✓ ✓ SpecReason [Pan et al., 2025] ✓ ✓ ✓ Speculative Thinking [Yang et al., 2025d] ✓ ✓ ✓ ✓ SPIRIT [Cui et al., 2025] ✓ ITC Analysis [Wang et al., 2025a] ✓ ✓ ✓ Think when needed [Yang et al., 2025c] ✓ THINKPRUNE [Hou et al., 2025] ✓ ✓ ✓ ✓ TALE [Han et al., 2024] ✓ TokenSkip [Xia et al., 2025] ✓ ✓ TOPS [Yang et al., 2025e] ✓ ✓ ✓ efficient reasoning [Arora and Zanette, 2025] ✓ ✓ ✓ TWT [Xu et al., 2025a] ✓ Z1 [Yu et al., 2025b] ✓ ✓ ✓

Count 28 24 20 11 8 6 6 5 5 5

Table 14: A total of 40 studies on LRM efficiency before May 2025 were included. Benchmarks that appeared more than four times are listed.

Questions that require performing basic mathematical operations or solving simple algebraic

numerical reasoning

equations to arrive at a numerical answer.

Case: What’s two plus three?

Questions that require tracing through and executing simple code snippets to determine their

output or behavior when run in a specific programming environment.

code reasoning

Case: What is the output of the following code when run in Python 3 environment: word =

”hello”\nprint(len(word))

Questions that require applying simple syllogistic reasoning to determine whether elements

reasoningquestion

set reasoning

belong to sets based on clearly stated relationships.

Case: All squares are quadrilaterals. A shape is a square, is it a quadrilateral?

Questions that require calculating time durations, ages, or future dates by applying simple

temporal reasoning

arithmetic operations to temporal information. Case: How many minutes equal 120 seconds?

Questions that require determining relative positions, directions, or orientations of objects

spatial reasoning

in space based on simple spatial relationships.

Case: If a bird is flying above a tree, where is the tree in relation to the bird?

Questions that require determining outcomes by applying simple cause-and-effect

relationships based on given conditional statements.

causal reasoning

Case: If ferromagnetic material is placed in a magnetic field, it will become magnetized. An

iron nail was placed next to a strong magnet for some time. Has the nail been magnetized?

Questions that require applying basic knowledge of physical laws and natural phenomena to

natural law reasoning

predict simple observable outcomes in everyday scenarios.

Case: Which is faster, an airplane or the propagation of light?

Questions that require recalling simple and fundamental geometric properties about shapes,

geometry facts

angles, and basic geometric figures.

Case: How many angles does a trapezoid have?

Questions that require recalling simple factual information about locations, landmarks,

geographic facts

political divisions, celestial bodies, and other basic geographic knowledge.

Case: Which is the largest continent on Earth? historical facts

knowledgequestion

Questions that require recalling basic facts about historical events.

Case: Which country first invented paper?

Questions that require recalling basic facts about the identities, achievements, and

biographical facts

characteristics of historical figures.

Case: Who proposed the theory of universal gravitation?

Questions that require recalling simple conversion relationships between

measurement units

standard units of measurement.

Case: How many centimeters equal 1 meter?

Questions that require recalling basic scientific symbols, formulas, and standard units

scientific notation

used in scientific communication.

Case: What is the chemical symbol for oxygen?

Questions that require recalling the creators or originators of notable artistic, literary,

creative authorship

musical, and cultural works.

Case: Who is the author of Hamlet?

Table 15: The subcategory descriptions and cases of reasoning questions and knowledge questions.

Questions that require outputting specified characters, words, or phrases a specific number

repetition constraints

of times according to simple formatting instructions.

Case: Output the number ”7” four times, without using separators.

Questions that require generating outputs of a specific length or with a specific number

length constraints

of components based on simple counting constraints.

instructionfollowing

Case: Output a four-digit number.

Questions that require generating words or numbers that conform to simple specified

character constraints

character patterns or formatting rules.

Case: Output a number that begins with 8.

Questions that require counting specific characters or elements within

counting constraints

a given text or sequence.

Case: Output the number of letter ”y” in the word ”yes”.

Questions that require modifying text or numbers according to simple formatting or character

transformation constraints

substitution rules to produce a transformed output.

Case: Output the word ”good” with all letters capitalized directly.

Questions that require generating sentences that conform to simple specified content or

sentence constraints

structural requirements.

Case: Give a sentence that contains the phrase ”have lunch” directly.

Questions that require determining whether simple statements express positive or negative

sentiment classification

emotions based on the tone and word choice.

Case: Does the sentence ”I hate rainy days.” express a positive or negative emotion?

Questions that require identifying the correct category of named entities (such as people,

named entity recognition

places, organizations, or time expressions) within simple sentences.

Case: In the sentence ”Napoleon died in 1821”, is ”1821” a time or a place name?

Questions that require identifying the language of origin for simple words or phrases

language classification

based on their characteristic writing systems or common vocabulary.

Case: Is the word ”hello” English or Japanese?

analysisquestion

Questions that require identifying the primary subject matter or thematic category of

topic classification

simple sentences based on their content and context clues.

Case: Is the topic of the sentence ”The stock market rose 2% today” finance or technology?

Questions that require determining the communicative purpose behind simple utterances

intent recognition

or statements based on their phrasing and context.

Case: Is the intention of the sentence ”I’m sorry I’m late.” to apologize or to blame?

Questions that require identifying the correct grammatical structure or sentence type of

syntax classification

simple expressions based on their form, punctuation, and communicative function.

Case: Is ”Close the door!” an imperative sentence or an interrogative sentence?

Questions that require identifying simple grammatical properties (like tense, voice,

grammar classification

or polarity) of sentences based on their structure and verb forms. Case: Is ”The apple was eaten.” in active voice or passive voice?

Questions that require identifying which entity a pronoun or reference term refers to in

coreference resolution

simple sentences by tracking relationships between words in the text.

Case: In ”My computer is broken, and I need to fix it.” What does ”it” refer to?

Table 16: The subcategory descriptions and cases of instruction following questions and analysis questions.

Prompt for construction workflow for S1-Bench Data Generation Prompt

Generate 50 pairs of questions and answers in both Chinese and English based on the category’s name, definition, and specific simplicity criteria. The following conditions must be satisfied:

- 1. Questions must be naturally and clearly expressed, unambiguous, and free of intentional traps.
- 2. Answers must be unique or easily falsifiable, with no possibility of multiple correct answers.
- 3. Make the questions as diverse as possible.

# Category Name and Definition: {name and definition}

# Specific Simplicity Criteria: {criteria}

# Cases: ## English question: {question en} ## English Answer: {answer en}

## Chinese question: {question zh} ## Chinese Answer: {answer zh}

Please generate 50 pairs of Chinese and English questions and answers in the following format: [question]English-question[answer]English-answer[question]Chinese-question[answer]Chinese-answer...

Start generating: Quality Discrimination Prompt Given a question, its answer, and its category, please analyze from the following perspectives as comprehensively as possible:

- 1. Whether the question belongs to the specified category and meet the Specific Simplicity Criteria.
- 2. Whether the question is easy, clear, unambiguous, and has an absolutely unique answer.
- 3. Whether the answer is absolutely correct; if not, what the correct answer should be.
- 4. Whether the question is similar to other given questions, and if similar, whether more diverse questions can be generated.

# Category Name and Definition: {name and definition}

# Specific Simplicity Criteria: {criteria}

# Question and Answer: {question with answer}

# Other Questions: {questions list}

Begin your analysis, aiming to be as detailed and comprehensive as possible: Difficulty Reduction Prompt Given a question and answer that are too complex for the model to answer correctly, you need to further reduce their difficulty while trying to:

- - Ensure the question aligns with the Category Name and Definition.
- - Ensure the question meets the Specific Simplicity Criteria.

# Category Name and Definition: {name and definition}

# Specific Simplicity Criteria: {criteria}

# Question and Answer: {question with answer}

The new question and answer:

Table 17: “Category Name and Definition” refers to the subcategory name and its definition, while Specific Simplicity Criteria refers to the simplicity requirements specific to the main category.

Prompt for Correctness Evaluation Evaluation on S1-Bench

**Question:** {question}

**Model Answer:** {model answer}

**Ground Truth:** {ground truth}

Your task is to evaluate whether the model’s answer is correct. An answer is considered correct as long as it contains the ground truth (regardless of how complex or detailed the description is). If there are parenthetical notes after the ground truth, then there may be multiple correct answers. In this case, the given answer is just one example, and any answer that meets the requirements specified in the notes can be considered correct. Additionally, some reasonably uncertain supplementary information is also considered appropriate, including more details, possibilities, and expanded discussion. You should focus more on whether the reply contains the correct answer.

You need to output a standard JSON, providing your explanation of the evaluation in the ”explain” field, and giving the evaluation result in the ”result” field, where 1 means the answer is correct and 0 means it is incorrect. Your action should follow the given format: “explain”: “”, “result”: 0/1

Table 18: Prompt for Correctness Evaluation.

706 472 470 446 392 357 349 347 347 334 327 326 314 312 306 300 297 293 292 285 285 283 283 245 241 233 222 219

L-R1-32B

1776 1145 1555 2712 1554 1112 1854 1328 1213 749 920 1037 1010 1360 732 679 773 903 1722 1642 683 763 1157 1066 651 1128 584 615

1095

s1.1-32B

1935 1259 924 2858 1131 890 2573 878 809 978 1032 903 711 2564 809 798 653 719 1159 955 684 854 903 811 587 866 552 524

998

s1.1-14B

1782 855 690 2587 936 571 2360 733 920 866 948 655 644 2358 717 618 596 655 1809 627 729 556 632 762 405 1427 514 426

840

QwQ-32B

1555 992 1078 949 828 832 526 942 474 718 720 750 686 584 668 674 688 680 389 826 685 602 883 575 570 393 551 499

720

s1.1-7B

1568 983 687 1874 982 597 2513 767 806 750 682 714 504 1523 536 494 427 337 1497 588 384 567 866 460 283 797 258 202

711

[Figure 15]

Qwen3-A30B

1421 897 1240 721 1152 768 582 981 645 750 668 685 718 425 684 642 614 607 338 855 626 599 803 577 577 345 495 507

702

L-R1-14B-DS

1456 964 1420 1389 854 910 651 1315 600 484 694 705 617 461 646 498 548 480 607 643 540 638 599 450 545 419 424 434

693

2500

Qwen3-32B

1171 971 1027 441 858 728 388 852 586 711 670 693 701 353 736 682 664 719 331 749 695 567 703 572 618 386 504 501

669

Qwen3-8B

1320 961 1072 566 976 703 350 823 608 692 732 684 561 368 655 607 667 656 338 735 705 539 737 635 513 393 500 461

658

DS-R1

1131 817 1135 1365 737 729 742 804 473 637 566 573 587 533 573 573 561 566 413 811 549 531 682 626 501 374 427 458

646

2000

Qwen3-A3B

1414 896 1012 357 779 702 351 777 563 736 671 643 623 369 728 622 615 723 322 693 634 537 644 625 573 384 498 481

638

Qwen3-1.7B

1344 892 1050 520 1131 563 372 849 503 608 624 601 519 419 542 553 589 583 302 687 549 405 641 503 505 347 459 384

596

Qwen3-14B

1324 886 938 429 759 638 304 669 554 642 616 605 529 312 634 559 628 649 262 612 619 465 638 528 401 290 474 449

583

1500

L-R1-32B-DS

1214 801 735 956 570 782 630 425 520 479 557 651 634 467 610 522 526 587 428 169 489 531 230 318 474 320 394 424

524

L-R1-7B-DS

1662 839 840 608 539 606 420 283 557 512 467 519 684 269 518 600 507 512 600 167 461 631 211 249 463 386 413 364

496

DS-R1-1.5B

852 820 800 294 512 614 392 359 440 565 705 674 562 554 528 575 525 449 380 245 475 548 298 258 459 309 365 372

490

1000

DS-R1-14B

998 705 957 506 439 622 349 789 417 459 533 561 489 311 496 422 466 418 438 277 409 431 314 286 378 316 378 371

475

- DS-R1-7B

DS-R1-70B

- DS-R1-8B

1155 736 904 401 315 635 350 524 401 463 529 512 458 312 505 477 437 441 424 187 426 556 221 246 413 319 363 354

455

943 720 619 266 409 699 415 392 429 504 556 541 522 334 578 463 500 447 433 204 433 477 230 284 396 365 403 388

454

500

912 1065 749 1107 558 587 291 275 449 468 499 528 534 241 437 469 474 421 355 158 416 450 182 197 426 272 318 377

452

DS-R1-32B

870 650 627 541 386 534 404 318 470 427 489 590 501 361 539 465 452 442 353 155 460 470 171 283 345 329 362 395

430

Nemotron-8B

808 705 705 593 872 312 289 451 368 502 223 200 502 356 261 320 344 326 399 342 302 414 409 332 334 316 115 152

373

Nemotron-49B

784 467 574 455 614 420 291 461 302 272 396 379 365 217 346 297 392 322 385 394 300 306 384 360 284 260 252 223

363

Mistral-8B

132 85 34 11 20 50 10 12 136 74 36 43 83 15 36 89 46 51 9 27 36 65 32 18 51 8 39 36

44

Qwen2.5-7B

179 91 41 9 18 41 7 14 100 80 46 49 62 11 48 78 49 39 2 33 44 41 32 15 26 6 38 37

43

Llama3.1-8B

185 102 31 4 25 64 5 10 148 78 36 39 20 14 37 56 61 43 9 29 58 26 39 38 12 20 18 39

42

Gemma2-9B

172 86 28 3 23 50 2 14 150 56 29 28 21 12 30 29 65 61 3 34 53 17 30 19 18 3 46 42

39

causalreasoning naturallawreasoning

geometryfacts lengthconstraintsspatialreasoning

setreasoning

characterconstraintsmeasurementunits

codereasoninghistoricalfacts

coreferenceresolutionlanguageclassification

creativeauthorshipsentenceconstraints namedentityrecognition

biographicalfacts grammarclassificationsyntaxclassificationrepetitionconstraintsnumericalreasoningintentrecognitiongeographicfacts temporalreasoningcountingconstraintsscientificnotation transformationconstraints

topicclassification sentimentclassification

Figure 9: Average response token counts on the 28 subcategories, which is the average result of five generations under top-p sampling.

Model Standard Readable but Malformed Unreadable 100 101 200 201 202 203 204 205 206 207 300 301

Qwen3-A22B 100.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 Qwen3-A3B 100.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 QwQ-32B 100.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 Qwen3-32B 99.91 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.09

- Qwen3-14B 99.95 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.05

- Qwen3-8B 99.95 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.05 Qwen3-1.7B 99.81 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.19 DS-R1 100.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00

- DS-R1-70B 99.91 0.00 0.09 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 DS-R1-32B 100.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 DS-R1-14B 100.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 DS-R1-8B 99.53 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.24 0.00 0.24

- DS-R1-7B 99.24 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.76 DS-R1-1.5B 97.58 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 2.42 Nemotron-49B 66.07 33.93 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 Nemotron-8B 58.06 26.26 0.00 0.00 0.00 0.09 0.00 0.00 0.00 15.02 0.00 0.57 L-R1-32B 95.07 0.00 0.00 0.00 0.00 0.00 0.81 0.00 0.00 3.03 0.00 1.09 L-R1-32B-DS 99.81 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.19 L-R1-14B-DS 99.19 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.81 L-R1-7B-DS 99.67 0.00 0.05 0.05 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.24 s1.1-32B 99.53 0.00 0.00 0.05 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.43 s1.1-14B 97.39 0.00 0.00 0.14 0.00 0.00 0.24 0.00 0.00 0.00 0.00 2.23 s1.1-7B 88.96 0.00 0.00 7.96 0.09 0.00 0.00 0.00 0.00 0.09 0.00 2.89

Table 19: Format type rates under top-p sampling.

Model Standard Readable but Malformed Unreadable 100 101 200 201 202 203 204 205 206 207 300 301

Qwen3-A22B 100.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 Qwen3-A3B 100.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 QwQ-32B 100.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 Qwen3-32B 99.76 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.24 Qwen3-14B 100.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 Qwen3-8B 100.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 Qwen3-1.7B 99.76 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.24 DS-R1 100.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 DS-R1-70B 100.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 DS-R1-32B 100.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 DS-R1-14B 99.76 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.24

- DS-R1-8B 99.53 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.24 0.00 0.24 DS-R1-7B 97.87 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 2.13 DS-R1-1.5B 91.94 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 8.06 Nemotron-49B 60.90 39.10 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 Nemotron-8B 55.21 26.78 0.00 0.00 0.00 0.00 0.00 0.00 0.00 16.35 0.00 1.66 L-R1-32B 85.55 0.24 0.00 0.24 0.71 0.24 0.95 0.00 0.00 6.64 2.61 2.84 L-R1-32B-DS 99.29 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.71 L-R1-14B-DS 98.82 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.18 L-R1-7B-DS 98.82 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.18 s1.1-32B 98.82 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.18 s1.1-14B 95.97 0.00 0.00 0.24 0.00 0.00 0.24 0.00 0.00 0.00 0.00 3.55 s1.1-7B 87.91 0.00 0.00 6.64 0.00 0.00 0.00 0.00 0.00 0.00 0.00 5.45

Table 20: Format type rates under greedy decoding setting.

###### Prompts for Solution Segmentation

# Task Description: Your task is to segment the given Chain of Thought according to the following rules:

- 1. Segmentation positions:

- (1) Please identify and extract all sub-reasoning processes from the Chain of Thought that meet the following condition: They explicitly arrive at a conclusion (including cases phrased as questions, e.g., ”right?”) that is directly consistent with the Ground Truth. Reasoning processes that only indirectly support the Ground Truth or result in partially aligned conclusions should be excluded.
- (2) After clearly reaching the conclusion consistent with the Ground Truth, insert the segmentation marker <split> to distinctly separate each qualifying sub-reasoning process.

- 2. Output Restriction:

- (1) You should only directly output the segmentation result without adding any additional supplements.
- (2)Except for inserting the <split> separator, you must not make any other modifications to the original Chain of Thought, not even minor characterlevel changes such as punctuation, spacing, or capitalization. In other words, after removing all <split> separators, the output text must match the original Chain of Thought exactly.

# Some cases: ## Case 1: ### Question: What is the value of x if x plus 3 equals 8?

### Ground Truth:

- 5

### Chain of Thought: Okay, so I need to figure out what x is if x plus 3 equals 8. The problem says ”x plus 3 equals 8,” so in mathematical terms, that would be: x + 3 = 8 Alright, now I need to solve for x. From what I remember in algebra, the goal is to isolate x on one side of the equation. That means I need to get rid of that ”+ 3” that’s being added to x. To do that, I should perform the opposite operation. Since it’s adding 3, the opposite would be subtracting 3. So, if I subtract 3 from both sides, that should keep things equal. Let me try that: x + 3 - 3 = 8 - 3 Simplifying both sides, on the left side, the +3 and -3 cancel each other out, leaving just x. And on the right side, 8 minus 3 is 5. So that simplifies to: x

= 5 Wait, let me check if that makes sense. If I substitute x back into the original equation, does it hold true? Let’s see: Original equation: x + 3 = 8 Substitute x = 5: 5 + 3 = 8 5 plus 3 is indeed 8, so that works out. Another way might be to consider what number, when you add 3 to it, gives 8. So starting from 8, if I take away 3, I should get back to the original number. That’s the same as subtracting 3 from 8, which again gives 5. So that’s consistent with the first method. Wait a second, what if the original problem was written differently? The problem says ”x plus 3 equals 8.” In symbols, that’s unambiguous: x + 3 = 8. So no, I think that’s right.

**Final Answer** The value of x is

|5|
|---|

.

### Output: Okay, so I need to figure out what x is if x plus 3 equals 8. The problem says ”x plus 3 equals 8,” so in mathematical terms, that would be:x + 3 = 8 Alright, now I need to solve for x. From what I remember in algebra, the goal is to isolate x on one side of the equation. That means I need to get rid of that ”+ 3” that’s being added to x. To do that, I should perform the opposite operation. Since it’s adding 3, the opposite would be subtracting 3. So, if I subtract 3 from both sides, that should keep things equal. Let me try that: x + 3 - 3 = 8 - 3 Simplifying both sides, on the left side, the +3 and -3 cancel each other out, leaving just x. And on the right side, 8 minus 3 is 5. So that simplifies to: x

= 5<split> Wait, let me check if that makes sense. If I substitute x back into the original equation, does it hold true? Let’s see: Original equation: x + 3 = 8 Substitute x = 5: 5 + 3 = 8 5 plus 3 is indeed 8, so that works out.<split> Another way might be to consider what number, when you add 3 to it, gives 8. So starting from 8, if I take away 3, I should get back to the original number. That’s the same as subtracting 3 from 8, which again gives 5.<split> So that’s consistent with the first method. Wait a second, what if the original problem was written differently? The problem says ”x plus 3 equals 8.” In symbols, that’s unambiguous: x + 3 = 8. So no, I think that’s right.

**Final Answer** The value of x is

|5|
|---|

###### .<split>

...(Other examples are omitted.) # Eval Target: ## Question: {question} ## Ground Truth: {ground truth} ## Chain of Thought: {thinking process} ## Output:

Table 21: Prompts for Solution Segmentation.

Prompts for Error Analysis Prompts for samples whose final answer is correct

# Task Description: You will receive a Question, its corresponding Ground Truth, and a Chain of Thought(COT) generated by a LLM for that Question. Your task is to carefully analyze the CoT and assign it to one of the two predefined categories listed below.

# Categories:

- 1: The CoT ***includes explicit incorrect conclusions*** in intermediate reasoning steps.
- 2: The CoT ***doesn’t include any explicit incorrect conclusion*** in intermediate reasoning steps. Output your evaluation in the following format:

# TheReason: [note: Conduct a step-by-step analysis, stating if and where explicit incorrect conclusions occur in the COT.]

# ErrorType: [note: Summarize each incorrect conclusion into a specific error type using a phrase of less than 5 words, such as factual inaccuracies, logical fallacies, comprehension mistakes, calculation errors, formatting issues, and so forth, to better conduct further evaluation and analysis. Directly output a Python list, where each element represents the error type of a specific incorrect conclusion in the CoT. If there are no incorrect conclusions, return an empty list.]

# TheCategory: [note: Provide your classification based on your analysis using only the number ”1” or ”2”. Do not add any additional text.]

# Question: {question}

# Ground Truth: {gound truth}

# COT: {thinking process}

# TheReason: Prompts for samples whose final answer is incorrect

# Task Description: You will receive a Question, its corresponding Ground Truth, and a Chain of Thought(COT) generated by a LLM for that Question. Your task is to carefully analyze the CoT and assign it to one of the two predefined categories listed below.

# Categories:

- 1: Regardless of whether the CoT ultimately arrives at the correct final answer or not, ***the correct answer is explicitly mentioned at least once*** within the reasoning steps (even if it is not ultimately adopted).
- 2: ***The correct answer is never explicitly mentioned or referenced*** at any point within the reasoning steps. Output your evaluation in the following format:

# TheReason: [note: Conduct a step-by-step analysis, explicitly stating whether and where a correct answer is mentioned within the reasoning steps.]

# TheCategory: [note: Provide your classification based on your analysis using only the number ”1” or ”2”. Do not add any additional text.]

# Question: {question}

# Ground Truth: {answer}

# COT: {thinking part}

# TheReason:

Table 22: Prompts for Error Analysis.

###### Prompts for classify the “gut moment” for English questions

# Overall Task Description You will be given the beginning portion of a response written by a large language model when answering a question. Your task is to classify the response into one of the following four categories based on the initial comment about **the difficulty of the question**. Important: Only consider the **initial comment** on difficulty made in the response. If the model later changes its assessment, please ignore those later revisions—focus solely on the first difficulty-related comment. You must assign the response to exactly one of the four categories below:

- Category 1: Initial comment indicates the question is easy. Initial comment includes phrases like “simple,” “basic,” “straightforward,” “common,” etc., clearly stating the question is easy.
- Category 2: Initial comment indicates the question is difficult: Initial comment includes phrases like “a bit complex,” “somewhat difficult,” “challenging,” “tricky,” etc., clearly stating the question is hard.
- Category 3: Initial comment is neutral or suggests moderate difficulty: Initial comment includes phrases like “interesting,” “not hard,” “not complex,” “fun,” etc., without a clear indication of difficulty level.
- Category 4: No comment on difficulty: The response contains no mention of difficulty at all.

Please output your result in the following format: [Explanation of classification]<sep>[Category (just the number)]<sep>[Short phrase containing the comment] If it falls into Category 4 (no comment), write ”None” for the [Short phrase containing the comment] part.

# Examples for each category will be shown below.

- ## Category 1 Examples:

Question: What is two plus three?

Response: Hmm, the user is asking what two plus three equals. This question seems very simple, though it might be a bit confusing for someone just starting to learn math. First, I need to figure out what the user’s intent is. They might be testing my basic calculation ability or genuinely need help solving the problem.

Output: The initial comment describes the question as ”very simple”<sep>1<sep>This question seems very simple

...(Other examples are omitted.)

- ## Category 2 Examples:

Question: What’s the next floor after basement level two?

Response: Hmm, this question seems a bit complex, but I’ll try to understand it first. The question is “What’s the next floor after basement level two?” First, I need to clarify what “basement level two” means. Usually, the first basement is the deepest, and then going up you have ground floor, first floor, and so on.

Output: The initial comment says it’s “a bit complex”<sep>2<sep>This question seems a bit complex

...(Other examples are omitted.)

- ## Category 3 Examples: Question: “Doctors recommend drinking at least eight glasses of water a day.” Is this sentence about health or education?

Response: Hmm, this question seems not complicated, though if you think carefully it might be a bit tricky. First, I need to understand the meaning of the sentence “Doctors recommend drinking at least eight glasses of water a day.”

Output: The initial comment says it’s “not complicated,” even though it later mentions it might be tricky. Classification should follow the initial comment<sep>3<sep>This question doesn’t seem complicated

...(Other examples are omitted.)

- ## Category 4 Examples:

Question: All birds have feathers. A sparrow is a bird. Does a sparrow have feathers?

Response: Hmm, the user is asking a question about birds and feathers—specifically, that all birds have feathers, a sparrow is a bird, so does the sparrow have feathers? This looks like a logic reasoning question. The user might be learning the basics of logic or biology.

Output: Although the response mentions it’s a “logic reasoning question,” it contains no evaluation of difficulty<sep>4<sep>None

...(Other examples are omitted.) # Here is the one you need to classify. Question: question Response: response Output:

Table 23: Prompts for classifying the “gut moment” in English questions.

Categories: instruction following Model: DeepSeek-R1-Distill-Qwen-7B Question: Output the number ”7” four times, without using separators. Ground Truth: 7777 Thinking process:

To solve the problem of outputting the number ”7” four times without using separators, I need to find a way to repeat the digit 7 multiple times without relying on concatenation or other connecting characters. One approach is to use mathematical operations that inherently produce the number 7. Addition is a simple and effective operation that can be used to achieve this. By adding 7 to itself multiple times, I can create the desired sequence of numbers. Specifically, adding 7 four times will give me 7 + 7 + 7 + 7, which equals 28. This method ensures that the number 7 is used exactly four times without the need for any separators. Therefore, the solution is to use the addition operation to repeat the number 7 four times and output the result.

###### Final Answer:

To output the number ”7” four times without using separators, you can utilize mathematical operations that inherently produce the number 7 multiple times. One straightforward way is to use addition:

[7 + 7 + 7 + 7 = 28] This expression shows that the number ”7” is used four times in a row without any separators, resulting in the total of 28. [

|7 + 7 + 7 + 7|
|---|

]

Table 24: Error Case for LRM.

Categories: instruction following Model: hunyuan-T1 Question: 输出一个动物的英文单词，并且只有三个字母 Output an English word for an animal, and it must be only three letters long. Ground Truth: 任何一个由三个字母组成的动物名称都可以（例如：cat） Any animal name consisting of three letters is acceptable (e.g., cat). Thinking process:

嗯，用户让我输出一个动物的英文单词，而且必须是三个字母的。首先，我得确定有哪些常见的动物是三个字母的。可能想到的有猫 （Cat）、狗（Dog），不过这两个都是三个字母吗？Cat是C-A-T，确实是三个字母，Dog也是D-O-G，三个字母。那这两个应该都可 以。不过用户可能需要更多的例子，或者只需要一个？题目里说的是“输出一个”，所以可能只需要一个。

不过等等，可能还有其他选项，比如Ant（蚂蚁）、Ape（猿）、Bat（蝙蝠）、Bear（熊）是四个字母，所以不行。Fish是四个字 母，Fox是三个字母，F-O-X，对，Fox也是三个字母。还有Lion是四个字母，不行。Tiger也是五个字母。Rat是三个字母，R-A-T。还 有Sheep是五个字母，Cow是三个字母，C-O-W。所以可能的选择还挺多的。

不过用户可能想要的是最常见的，比如Cat、Dog、Fox、Ant、Rat、Cow这些。需要确认这些单词是否都是正确的动物名称。比如Ant是 蚂蚁，没错。Fox是狐狸，Rat是老鼠，Cow是奶牛。这些都是正确的。所以可能需要从这些里面选一个。 不过用户可能希望答案只有一个，所以可能需要随机选一个，或者选最典型的。比如Cat和Dog是最常见的宠物，可能更容易被想到。 但Fox也是一个不错的选择，可能更有趣一些。或者像Ape，虽然也是三个字母，但可能不如其他常见。 另外，用户有没有可能有其他需求？比如是否区分大小写？题目里说是英文单词，通常首字母不大写，除非是专有名词。所以应该用小 写，比如cat、dog。不过输出的时候可能需要大写首字母，但题目没有特别说明，可能保持小写即可。

再检查一下有没有拼写错误，比如Bat是B-A-T，没错。Rat是R-A-T，正确。Cow是C-O-W，没错。所以这些都可以作为候选答案。现在 需要从中选一个作为回答。可能用户希望得到多个例子，但题目要求输出一个，所以可能需要选最典型的。比如Cat或者Dog，因为这两 个是最常见的宠物，可能更容易被用户想到。不过也有可能用户希望有变化，比如Fox这样的动物，比较有趣。

或者，用户可能是在测试知识库，看看是否知道除了Cat和Dog之外的其他三个字母的动物。比如Ant，虽然蚂蚁很小，但确实是动物。 或者Rat，虽然老鼠可能不太受欢迎，但确实属于动物。Cow是家畜，可能也是合理的选择。 所以，综合考虑，可能需要给出多个例子，但题目要求输出一个。这时候可能需要选择其中一个，比如Cat，或者Dog，或者Fox。例 如，回答“Cat”或者“Dog”都是正确的。不过为了多样性，可能选Fox，因为不是所有人都会立刻想到它，而Cat和Dog比较基础。或者用 户可能希望得到不同的答案，所以可以选其中一个，比如“Fox”作为回答。

Well, the user asked me to output an English word for an animal, and it must be three letters long. First, I need to figure out which common animals have three letters. I can think of Cat and Dog, but are both of these three letters? Cat is C-A-T, and it is indeed three letters. Dog is D-O-G, also three letters. So both of these should be valid. But the user might need more examples, or just one? The task says ”output one,” so it might only require one. But wait, there could be other options, like Ant, Ape, Bat, and Bear, but Bear has four letters, so it doesn’t count. Fish has four letters, too. Fox is three letters, F-O-X, yes, Fox is three letters. Lion has four letters, so that doesn’t work. Tiger has five letters. Rat is three letters, R-A-T. Sheep has five letters. Cow is three letters, C-O-W. So there are quite a few possibilities. However, the user might be looking for the most common ones, like Cat, Dog, Fox, Ant, Rat, and Cow. I need to check if these words are all correct animal names. For example, Ant is an ant, that’s right. Fox is a fox, Rat is a rat, Cow is a cow. These are all correct. So we might need to choose one from these. But the user may want only one answer, so I might need to pick randomly or choose the most typical one. For instance, Cat and Dog are the most common pets, so they might be easier to think of. But Fox is also a good choice and might be more interesting. Or something like Ape, though it’s three letters, might not be as common as the others. Also, is there any other requirement from the user? For instance, should I differentiate between uppercase and lowercase? The task says it’s an English word, and usually, the first letter isn’t capitalized unless it’s a proper noun. So, it should be in lowercase, like cat, dog. However, when outputting, the first letter might need to be capitalized, but the task doesn’t specify this, so lowercase should be fine. Let’s double-check for spelling errors: Bat is B-A-T, correct. Rat is R-A-T, correct. Cow is C-O-W, correct. So these can all be candidate answers. Now, I need to choose one as the response. The user might want several examples, but the task asks for only one, so I might need to pick the most typical one. For example, Cat or Dog, since these are the most common pets and might be easier for the user to think of. But Fox is also a good choice, as it’s more interesting. Or the user might be testing the knowledge base to see if it knows other three-letter animals besides Cat and Dog. For example, Ant, although ants are small, they are indeed animals. Or Rat, although rats might not be as popular, they are still animals. Cow is livestock, so that could also be a reasonable choice. So, taking everything into account, I might need to give multiple examples, but the task requires one. In that case, I could choose one, like Cat, or Dog, or Fox. For instance, answering “Cat” or “Dog” would both be correct. However, for diversity, I might pick Fox, because not everyone will immediately think of it, whereas Cat and Dog are more basic. Or the user might want a different answer, so I could choose one of them, like “Fox.” Final Answer:

**Wolf**

- Table 25: Error Case for LRM.

Categories: knowledge question Model: DeepSeek-R1-Distill-Qwen-7B Question: In which city is the Sydney Opera House located? Ground Truth: Sydney. Thinking process:

Okay, so I need to figure out which city the Sydney Opera House is in. I know that Australia has a lot of cities, but I’m pretty sure the Opera House isn’t in Sydney. Wait, isn’t it in a different place? I think I’ve heard of a place called Petersham before, maybe that’s where it’s located. I remember seeing pictures of it in the news once, and they mentioned Petersham as the area where it’s situated. But I’m not entirely sure if it’s just a suburb or a major city. Maybe I should think about other major cities in Australia. There’s Melbourne, which is known as the ”Melbourne” city, and then there’s Brisbane, which is more on the northern side. Then there’s Sydney, which is the capital, but I don’t think the Opera House is there. I think it’s in a regional area instead of the capital city. So, putting it all together, I’m pretty confident that the Sydney Opera House is located in Petersham, which is a suburb in New South Wales, Australia. But I should double-check to make sure I’m not mixing up the names of places or cities. Maybe I can recall any famous landmarks in Petersham. I think there’s a big shopping center there called the Petersham Shopping Centre, and I’ve heard that the Opera House is nearby. Yeah, that makes sense. So, I’m going to go with Petersham as the location.

Final Answer: The Sydney Opera House is located in Petersham, a suburb in New South Wales, Australia.

- Table 26: Error Case for LRM.

Categories: knowledge question Model: DeepSeek-R1-Distill-Qwen-7B Question: Who composed *Symphony No. 9 in D minor, Op. 125*? Ground Truth: Ludwig van Beethoven Thinking process:

Alright, so I need to figure out who composed Symphony No. 9 in D minor, Op. 125. Let me start by recalling what I know about classical music composers. I know that there are several famous composers, but I’m trying to remember if it’s someone like Beethoven, Mozart, or perhaps someone from a later era. Wait, Beethoven wrote many symphonies, but I think he didn’t write Symphony No. 9 in D minor. Let me think. Beethoven’s 9th symphony is actually in E flat minor, Op. 67, right? So that’s not it. Then there’s Beethoven’s Moonlight Sonata, which is different. Mozart composed a lot of piano concertos, but I don’t think he wrote a symphony in D minor. His compositions are usually in various key signatures, but D minor isn’t one I associate with him. Then there’s Schubert, who wrote the famous symphonies, but I don’t recall him composing a symphony in D minor either. Wait, maybe it’s someone else. I’ve heard of Igor Stravinsky before, but he was more known for his operas like The Rite of Spring. Then there’s Paul Simon, but he’s a contemporary composer, so probably not. Oh, maybe it’s someone from the classical era. Let me think about other composers. There’s Shostakovich, but I don’t think he wrote that. What about someone like Tchaikovsky? No, he wrote symphonies in C minor and F major, but not D minor. Hmm, maybe it’s someone more obscure. I think I’ve heard of a composer named Igor Kovalsky. Wait, was that the composer of this symphony? I’m not entirely sure, but I think he’s associated with writing complex symphonies. Alternatively, could it be someone like Shoenfield? I’m not certain. Wait, I think I remember that Igor Kovalsky wrote this symphony. It’s sometimes referred to as ”The New World” because of its emotional depth and complexity. The symphony is in D minor, Op. 125, and it’s known for its intense, dramatic sound. I think that’s correct, but I should double-check to make sure it’s not another composer. Another thought: maybe it’s someone from the 19th century. I’m trying to recall if any composers from that era wrote such a symphony. I’m not sure, but I think Igor Kovalsky is the one. He was an American composer, so that makes sense as a modern touch in classical music. So, putting it all together, I believe the composer is Igor Kovalsky, and the symphony is known for its dramatic and intense sound. I think that’s the correct answer, but I should confirm if there’s any other possibility I might be forgetting.

###### Final Answer:

The composer of Symphony No. 9 in D minor, Op. 125, is Igor Kovalsky. This symphony is often recognized for its dramatic and intense sound, sometimes referred to as ”The New World.”

Table 27: Error Case for LRM.

Categories: reasoning question Model: s1.1-7B Question:

1.5加3.5等于几？ What is 1.5 plus 3.5? Ground Truth:

- 5 Thinking process:

好的，我需要计算1.5加3.5等于多少。首先，这两个数都是小数，处理起来相对简单。我可以在心里快速算一下。 首先，1.5加上3.5。可以将它们看成是1 + 0.5 + 3 + 0.5。这样就可以先将整数部分相加，再将小数部分相加。1+3=4，0.5+0.5=1。然后把 两部分加起来是4+1=5。所以答案是5。 另一种方法是将它们视为货币金额。比如，1.5美元加3.5美元。1.5美元加上3美元是4.5美元，再加上0.5美元就是5美元。这也得到同样的 结果。 当然，也可以用电子计算器来验证，但我认为手动计算更快。不过，为了确保准确，我再检查一遍。1.5加3.5。10加35等于45，然后小数 点一位，所以是4.5。对的，1.5+3.5=4.5。或者用竖式计算也可以： 1.5

+3.5

- - - - 4.5 对，这样一看也确实是4.5。所以答案应该是4.5。我之前得到的结果是正确的。

Alright, I need to calculate what 1.5 plus 3.5 equals. First, both of these numbers are decimals, so it’s relatively simple to handle. I can quickly calculate it in my head. First, 1.5 plus 3.5. You can think of them as 1 + 0.5 + 3 + 0.5. This way, I can add the integer parts first, then the decimal parts. 1 + 3 = 4, and 0.5 + 0.5

= 1. Then, adding the two parts together gives 4 + 1 = 5. So, the answer is 5. Another way is to treat them as currency amounts. For example, 1.5 dollars plus 3.5 dollars. 1.5 dollars plus 3 dollars equals 4.5 dollars, then adding

- 0.5 dollars gives 5 dollars. This also leads to the same result. Of course, you could use a calculator to verify, but I think doing it manually is quicker. However, to ensure accuracy, I will check again. 1.5 plus 3.5. 10 plus 35 equals 45, then move the decimal point one place, so it’s 4.5. Yes, 1.5 + 3.5 = 4.5. Or you could use column addition:
- 1.5

+3.5

- - - - 4.5 Yes, looking at it this way, it indeed adds up to 4.5. So the answer should be 4.5. The result I got earlier is correct.

Final Answer: 答案是 The answer is

|4.5|
|---|

|4.5|
|---|

Table 28: Error Case for LRM.

Categories: analysis question Model: DeepSeek-R1-Distill-Qwen-1.5B Question: In the sentence ”Napoleon died in 1821”, is ”1821” a time or a place name? Ground Truth: Time. Thinking process:

Okay, so I have this sentence: ”Napoleon died in 1821.” I need to figure out whether ”1821” is a time or a place name. Hmm, let me think about this step by step. First, I know that ”Napoleon” is a person, specifically a historical figure. So, if it’s the name of a person, then ”1821” must be related to their life. I’m pretty sure that ”1821” is a year. I remember that Napoleon was born in 1769 and died in 1821. So, 1821 is definitely a year. Wait, but what if I’m not sure? Let me consider other possibilities. Sometimes names can be place names or time periods. For example, if someone’s name were a city or a country, the number could be a place. But in this case, ”Napoleon” is a person, so it’s more likely a year. Also, thinking about the sentence structure, it’s written as ”Napoleon died in 1821.” The structure ”in” suggests that 1821 is the place. So, in this context, ”in” is an article that indicates the location, which would be a place name. That makes me think that ”1821” is indeed a place name, the year. I don’t think it’s a time because time is usually expressed as something like ”in the 19th century” or ”2023.” So, 1821 is specifically a year, not a time period. It’s more precise because it’s a specific year when Napoleon died. So, putting it all together, ”Napoleon” is a person, and ”in 1821” is indicating the year he died. Therefore, ”1821” is a place name, specifically a year.

Final Answer: In the sentence ”Napoleon died in 1821,” the number ”1821” is a place name, specifically a year. It is used to indicate the year Napoleon died, which is

1821. The use of ”in” confirms that it refers to a place, not a time period. Therefore, ”1821” is a year.

Table 29: Error Case for LRM.

