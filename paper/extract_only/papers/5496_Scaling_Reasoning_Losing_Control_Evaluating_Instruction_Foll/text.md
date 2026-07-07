arXiv:2505.14810v2[cs.CL]25May2025

# Scaling Reasoning, Losing Control: Evaluating Instruction Following in Large Reasoning Models

Tingchen Fu1, Jiawei Gu2, Yafu Li2∗, Xiaoye Qu2, Yu Cheng3 1 Renmin University of China, 2 Shanghai AI Laboratory, 3 The Chinese University of Hong Kong

## Abstract

Instruction-following is essential for aligning large language models (LLMs) with user intent. While recent reasoning-oriented models exhibit impressive performance on complex mathematical problems, their ability to adhere to natural language instructions remains underexplored. In this work, we introduce MathIF, a dedicated benchmark for evaluating instruction-following in mathematical reasoning tasks. Our empirical analysis reveals a consistent tension between scaling up reasoning capacity and maintaining controllability, as models that reason more effectively often struggle to comply with user directives. We find that models tuned on distilled long chains-of-thought or trained with reasoning-oriented reinforcement learning often degrade in instruction adherence, especially when generation length increases. Furthermore, we show that even simple interventions can partially recover obedience, though at the cost of reasoning performance. These findings highlight a fundamental tension in current LLM training paradigms and motivate the need for more instruction-aware reasoning models. We release the code and data at https://github.com/TingchenFu/MathIF.

## 1 Introduction

Recent advancements in Large Reasoning Models (LRMs) [1], such as o3 and o4-mini [2], DeepSeekR1 [3], and K1.5 [4], have demonstrated impressive capabilities in mathematical reasoning, including solving olympiad-level problems [5, 6, 7] and automating formal theorem proving [8]. These breakthroughs have sparked growing interest in scaling chain-of-thought (CoT) reasoning [9], where models produce explicit multi-step explanations to solve complex tasks. Typical approaches include imitation learning, e.g., supervised fine-tuning (SFT), and reinforcement learning with verifiable rewards [10], both of which aim to strengthen model intelligence across various tasks and scales.

Despite these advances, instruction following, i.e., the ability to accurately and reliably comply with user directives, has received comparatively little attention in the context of LRMs. Yet this ability is critical for real-world alignment and safety [11]. Our empirical evaluations on IFEval [12] and FollowBench [13] reveal a consistent pattern: although LRMs excel at mathematical reasoning, they often fail to follow even simple instructions. This raises an important question: As reasoning scales, do models become more intelligent yet less controllable? Unfortunately, existing instructionfollowing benchmarks are ill-suited for answering this question. Most are designed for generalpurpose language models and lack coverage of math-specific reasoning behaviors. In contrast, LRMs are typically trained on math-heavy datasets and optimized specifically for problem-solving capacity. This gap highlights the urgent need to evaluate whether increasing intelligence in specialized reasoning models inherently leads to diminishing control over their behavior, an issue at the heart of instruction alignment for advanced LRMs.

To address this, we propose MathIF, the first benchmark specifically designed to evaluate the instruction-following capabilities of LRMs in math domains. MathIF introduces 15 Python-verifiable

∗Corresponding author: yafuly@gmail.com Preprint.

constraints across 4 categories, which are programmatically combined to create 30 double-constraint and 15 triple-constraint prompts. These are applied to math problems of varying difficulty, resulting in a total of 420 high-quality evaluation samples. Using MathIF, we evaluate 23 LRMs across a wide range of scales and architectures. Surprisingly, most models fail to reliably follow instructions, and performance does not consistently improve with larger model sizes. Even the best-performing model, Qwen3-14B, achieves only 50.71% accuracy on strict instruction-following. Furthermore, performance deteriorates with increasing task difficulty and constraint complexity, revealing substantial headroom for improvement.

Our deeper analysis further uncovers a mutual interference between instruction-following and reasoning capabilities, observed at both training and inference stages. Common reasoning-oriented training strategies (e.g., SFT and RL) enhance reasoning ability but degrade instruction adherence. This degradation becomes more pronounced as the CoT length increases, likely because longer reasoning paths widen the contextual gap between the original instruction and the final answer, making it harder for the model to retain and execute directives. Conversely, enforcing brevity by limiting CoT length improves instruction-following performance, but at the cost of reasoning depth and accuracy.

These observations reveal a consistent pattern: improving reasoning capability often comes at the cost of instruction adherence, suggesting an inherent trade-off between the two abilities. This trade-off highlights a crucial challenge in LRM development: training for stronger reasoning alone may undermine alignment, and future methods must account for this tension to build models that are both capable and controllable. To summarize, our contributions are three-fold:

- • We introduce MathIF, the first benchmark for systematically measuring instruction-following performance in large reasoning models within math domains.
- • We evaluate 23 recent LRMs and uncover a widespread inability to follow user constraints, especially on harder problems and multi-constraint settings.
- • We identify and empirically validate a trade-off between reasoning performance and instruction-following ability, with mutual interference observed during both training and inference.

## 2 Related Work

#### 2.1 Large Reasoning Models (LRMs)

Recent advances in enhancing the reasoning ability of language models and reimplementing large reasoning models generally fall into two paradigms. The first paradigm constructs high-quality long CoT by distilling from more capable LRMs or combining primitive reasoning actions [14, 3], on which SFT and subsequent RL are conducted. For example, s1 [14] shows that even a small amount of CoT data could significantly promote the reasoning ability, and LIMO [15] further put forward the less-is-more reasoning hypothesis, stating that sophisticated reasoning capabilities can emerge through minimal demonstration of cognitive processes if relevant domain knowledge is already encoded in the language model during pre-training. Similarly, [16] suggests that injecting more cognitive behavior into the long CoT during SFT brings promotion in mathematical benchmarks.

On the other hand, cold-RL on base language models directly attracts more and more attention in the subfield because of the success of deepseek-R1-zero and the previous findings that the model tends to memorize training data during the SFT process [17]. In contrast with SFT, cold-RL does not rely on long CoT data and provides supervision signals by rewards on the final outcome [3] or the reasoning process [18]. To simplify and accelerate the RL process, various techniques have been proposed, such as dynamic sampling [19], process-reward [20], off-policy guidance [21], and CoT preference optimization [22]. To compare two different pathways and evaluate the capacity of LRMs [16], previous studies mostly focus on their complex reasoning performance on olympiad-level math problems, while their instruction-following ability when solving math problems is left unconsidered.

#### 2.2 Instruction-following benchmarks

As a crucial factor determining the practicality of a language model for real-world scenarios, the instruction-following ability is a core metric for language model evaluation, with numerous protocols

Table 1: Single constraints and sample dual-/triple-constraint compositions across four categories.

Category Sub-Category Example Length Length Answer with less than 500 words. Lexical Language Your answer should be in Chinese language, no other language is allowed.

Keyword Include keywords "condition" in your response.

Format Punctuation In your entire response, refrain from the use of any commas.

Case Your entire response should be in English, and in all lowercase letters. No capital letters are allowed. Highlight Your answer must contain exactly 3 bullet points. Use the markdown bullet points such as:

* This is point 1. * This is point 2.

Affix Prefix First repeat the request word for word without change, then give your answer.

Suffix Finish your response with this exact phrase "Any other questions?". No other words should follow this phrase. Both Wrap your entire response with double quotation marks.

and benchmarks being developed [23, 24]. Earlier benchmarks primarily focused on the completeness of user queries and depended on proprietary language models [23, 24] to measure the response quality by its win-rate over the baseline method, which is an oversimplification of real user queries. For a more comprehensive evaluation, sophisticated benchmarks have been developed to test the ability of a language model in following format constraints [12, 25, 26], multi-turn instruction [27, 28, 29, 30], long-context instruction [31], multi-lingual instruction [27, 32], compositional instruction [33, 34, 29] and refutation instructions [35, 36]. However, most instruction-following benchmarks concentrate on the general domain and relatively straightforward queries. The domain shift and the lack of long CoT become a deterrent for using the benchmark on LRMs.

## 3 MathIF

Overview As shown in Figure 1, the same base model exhibits a clear performance drop in both IFEval [12] and FollowBench [13] after transitioning from the Instruct variant to the LRM variant. The “Instruct” and “LRM” are abbreviations for Instruction LLMs and their corresponding large reasoning models, with details presented in the Appendix. This observation suggests that reasoning-oriented training, while beneficial for problem-solving, may compromise instruction-following ability. However, since both benchmarks are designed for generalpurpose tasks rather than mathematical reasoning, where LRMs are specifically optimized, it remains difficult to isolate instruction-following performance from confounding factors such as domain mismatch.

IFEval FollowBench

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

Llama-70B-Instruct

Llama-70B-LRM

Qwen-32B-Instruct

Qwen-32B-LRM

Qwen-Coder-32B-Instruct

Qwen-Coder-32B-LRM

0.0 0.2 0.4 0.6 0.8

Score

Figure 1: Performance of Instruct LLMs and LRMs on IFEval [12] and FollowBench [13].

In this study, we propose MathIF, the first benchmark for evaluating the instruction-following ability of LRMs. Specifically, in this benchmark, we consider four major types of Python-verifiable constraints and construct compositional instructions by combining two or three constraints together. The constraints are then incorporated into the math problems, which are collected from diverse sources and vary in levels of difficulty. Two metrics are proposed for fine-grained analysis of LRM’s instruction-following ability.

Constraint Type Inspired by IFEval [12] and ComplexBench [37], we mainly consider 15 constraints in four types when establishing our benchmark: (1) length constraint puts a limitation on the length of the response since an overlong response might lead to extra latency at inference; (2) lexical constraint requires the model to give the response in a specified language or contains pre-defined key words and phrases in response; (3) format constraint, being probably the most common type of constraint from real users of LRMs, it includes various forms of restriction including the number of sections, the number of highlighted bullet, usage of punctuation, case sensitivity and many other forms of instruction; and finally (4) affix constraint, refers to the prefix constraint or suffix constraint or a combination of both that requires the model to begin/end the response with specific symbols or

Table 2: Dataset statistics grouped by source and by constraint.

Group by source Group by constraint Total GSM8K MATH500 Minerva Olympiad AIME Single Double Triple

# samples 90 90 90 90 60 140 140 140 420 Avg. Len 86.73 57.24 88.09 80.42 87.25 64.89 83.84 89.54 79.43

sentences. To mitigate reliance on proprietary language model [38], we ensure that the compliance of constraints in our benchmark can be verified simply by a Python programme. A more detailed categorization for the type and subtype of constraints involved in our benchmark is listed in Table 1 together with an illustrating example. The entire list of constraints used in our benchmark can be found in the Appendix.

Compositional Constraint Queries with only a single constraint can hardly reflect the complex scenarios encountered by a downstream application of LRM, as the real user queries to LRMs typically contain more than one restrictive condition. Therefore, following previous work in complex instruction-following [37], we construct compositional constraints by combining two or three individual constraints. Specifically, given the set of individual constraints denoted as C, we enumerate all the elements in the Cartesian product C2 = {(c1,c2) | c1,c2 ∈ C} and C3 = {(c1,c2,c3) | c1,c2,c3 ∈ C}, from which we randomly sample several combinations after manually filtering out the ones in which the constraints are incompatible with each other and fall into the same subtype of constraint. Through this procedure, we harvest 30 dual-constraints and 15 triple-constraints. The detailed list of dual-constraints and triple-constraints is presented in Table 1.

Math Problem Collection With the constructed individual constraints and compositional constants, the next step is to incorporate these constraints into math problems to constitute a query. Different from recent benchmarks that mostly center around math problems from undergraduate level or even higher level [7], MathIF contains math problems of varying levels of difficulty, ranging from math word problems in primary school and math problems in high school to the latest math problems in world-level competition. Specifically, we randomly sample 90 problems from GSM8K [39], MATH-500 [6], Minerva [40], Olympiad [5] respectively. For AIME2024&2025 [7], we use all the 60 problems it contained. For each data source, we apply a single constraint, dual constraints, and triple constraints, resulting in three subsets of equivalent size. For a sanity check, we manually review the curated samples and double-check whether the added constraints are contradictory to the math problem itself. The statistics for the established dataset are shown in Table 2.

Evaluation Metric To systematically measure whether one or more constraints in the query are satisfied by the LRM while solving the math problems, we follow previous works [12, 13] and use two metrics of different granularity. Specifically, we employ hard accuracy (HAcc) and soft accuracy (SAcc) to measure whether the model response follows the constraints at the query level and constraint level, respectively. Formally, suppose a query has n constraints C1,C2,C3,...,Cn and we use I(Ci) to denote whether the i-th constraint is satisfied or not, with I(Ci) = 1 for satisfied constraint and I(Ci) = 0 for unsatisfied constraint. The hard accuracy (HAcc) and soft accuracy (SAcc) for a query is defined as:

n

n

1 n

I(Ci) (1)

I(Ci), SAcc =

HAcc =

i=1

i=1

Notably, for queries with only a single constraint, the two metrics are identical in number. The overall hard accuracy and soft accuracy on the benchmark are averaged among all the queries in the dataset. Apart from instruction-following ability, we also measure the correctness of the math problem solution on our proposed MathIF, defined as whether the final answer exactly matches the ground-truth, regardless of constraint satisfaction. By default, correctness refers to performance with constraints in the prompts unless specified (e.g., Table 3).

## 4 Experiment

To benchmark the instruction-following ability of LRMs, we evaluate a diverse set of models across three parameter scales. All models are decoded using nucleus sampling (T=1.0, p=0.95) with

Table 3: Experimental results of LRMs on MathIF. We report hard accuracy (HAcc) and soft accuracy (SAcc) for instruction-following, alongside math-solving correctness with and without constraints (w/o const. / w/ const.). The last column shows the relative change in correctness when constraints are included. Models are sorted in descending order of instruction-following performance. † indicates models trained by supervised fine-tuning only (no reasoning-oriented RL). Bold and underlined values denote the top-2 and bottom-2 entries in each column, respectively.

Instruction Following Correctness HAcc SAcc w/o const. w/ const. Diff.(%)

Model

Models with no more than 4B parameters

Qwen3-4B 44.05 61.43 68.10 58.57 -13.99 Qwen3-1.7B 30.24 50.24 62.38 51.19 -17.94 Qwen3-0.6B 27.86 50.44 40.95 32.14 -21.51 L1-Qwen-1.5B-Exact 19.76 39.60 53.81 42.86 -20.35 L1-Qwen-1.5B-Max 19.76 39.40 55.48 45.71 -17.61 DeepSeek-R1-Distill-Qwen-1.5B† 17.14 36.62 52.86 31.67 -40.09 DeepScaler-1.5B-Preview 14.52 34.52 58.10 36.19 -37.71 Qwen2.5-1.5B-SimpleRL-Zoo 9.05 24.33 27.14 22.38 -17.54 Qwen2.5-Math-1.5B-Instruct 7.62 21.39 44.05 44.29 +0.54

Models with approximately 7B–14B parameters

Qwen3-14B 50.71 67.06 71.43 64.29 -10.00 DeepSeek-R1-Distill-Qwen-14B† 39.28 60.55 67.14 50.95 -24.11 Qwen3-8B 37.86 57.34 69.52 66.43 -4.44 DeepSeek-R1-Distill-Qwen-7B† 26.43 44.96 65.24 48.57 -25.55 DeepSeek-R1-Distill-Llama-8B† 22.14 44.04 59.76 36.43 -39.04 Open-Reasoner-Zero-7B 13.57 32.26 52.86 51.90 -1.82 Qwen2.5-Math-7B-Instruct 9.05 25.60 46.90 37.14 -20.81

Models with 32B or more parameters

Qwen3-32B 43.81 62.82 72.62 70.00 -3.61 DeepSeek-R1-Distill-Qwen-32B† 42.62 60.91 71.43 57.62 -19.33 DeepSeek-R1-Distill-Llama-70B† 41.43 61.07 71.19 54.05 -24.08 QwQ-32B 40.24 59.99 70.95 68.81 -3.02 OlympicCoder-32B 35.95 57.97 59.29 54.52 -8.05 s1-32B† 20.95 41.78 62.86 60.95 -3.04 Open-Reasoner-Zero-32B 15.47 35.52 65.48 67.62 +3.27

a maximum generation length of 16,384 tokens, powered by the vLLM [41] engine for efficient inference.

- • Small-scale models (≤ 4B parameters): Qwen3-0.6B [42], Qwen2.5-1.5B-SimpleRLZoo [43], Qwen2.5-Math-1.5B-Instruct [44], DeepSeek-R1-Distill-Qwen-1.5B [3], DeepScaler-

- 1.5B-Preview [45], L1-Qwen-1.5B-Max [46], L1-Qwen-1.5B-Exact [46], Qwen3-1.7B [42], Qwen3-

- 4B [42].

- • Medium-scale models (7B∼14B parameters): Qwen2.5-Math-7B-Instruct [44], DeepSeek-R1Distill-Qwen-7B [3], Open-Reasoner-Zero-7B [47], DeepSeek-R1-Distill-Llama-8B [3], Qwen38B [42], DeepSeek-R1-Distill-Qwen-14B [3], Qwen3-14B [42].
- • Large-scale models (≥ 32B parameters): s1-32B [14], OlympicCoder-32B [48], DeepSeek-R1Distill-Qwen-32B [3], QwQ-32B [49], Open-Reasoner-Zero-32B [50], Qwen3-32B [42], DeepSeekR1-Distill-Llama-70B [3].

- 4.1 Experimental Results The experimental results, as summarized in Table 3, reveal several key factors that influence the instruction-following performance of LRMs, including model scale, architecture design, and reasoning format.

All LRMs fail to obey most user instructions. All LRMs evaluated on MathIF exhibit poor instruction-following performance. Even the best-performing model, Qwen3-14B, achieves only 50.71% hard accuracy, barely surpassing the halfway mark. The majority of models, including

large-scale variants such as DeepSeek-R1-Distill-Llama-70B and Open-Reasoner-Zero-32B, fail to meet even minimal expectations for faithfully executing user-specified constraints.

The Qwen3 series consistently demonstrates strong instruction-following capabilities across model scales. Qwen3-14B achieves the highest hard and soft accuracy among all LRMs. Qwen3-4B leads in the sub-4B category, while Qwen3-32B outperforms all models with 32B or more parameters. Overall, the Qwen3 series shows remarkable performance across sizes.

Model scale alone does not determine instruction-following performance. While larger models often perform better within the same series (e.g., Qwen2.5-Math and Open-Reasoner-Zero), scaling up does not guarantee improvement across different architectures. For instance, DeepSeek-R1-DistillLlama-70B underperforms Qwen3-4B despite being more than 15× larger. Notably, Qwen3-8B and Qwen3-32B deviate from the within-series scaling trend, highlighting that instruction-following ability depends on both model size and design.

Explicit reasoning separation may enhance instruction-following. Most LRMs in our study adopt special tokens (e.g., <think> and </think>) to isolate reasoning from the final answer, with only four exceptions: Qwen2.5-Math-1.5B, 7B-Instruct and Qwen2.5-1.5B, 7B-SimpleRL-Zoo. These models generally underperform their peers of similar size, suggesting that the lack of explicit separation may blur the boundary between reasoning and answer, leading to poorer adherence to constraints.

There exists a trade-off between instruction-following and mathematical reasoning. As shown in the “Diff” column of Table 3, most models experience a drop in problem-solving correctness when additional constraints are introduced, with margins ranging from 0.96 to 23.33. This suggests that stronger adherence to external constraints may compromise core mathematical reasoning. The only exceptions are Qwen2.5-Math-1.5B-Instruct and Open-Reasoner-Zero-32B, which maintain or slightly improve their performance under constrained conditions.

#### 4.2 A Closer Look at the Results

80

Correcness HAcc SAcc

| |
|---|

Accuracy(%)

60

We first scrutinize the model performance on each subset and visualize the average accuracy of 23 LRMs in Figure 2. From the figures, we can observe a performance difference among different subsets for both hard accuracy and soft accuracy. And the performance on GSM8K, the easiest benchmark among the five, is obviously higher than the other four subsets, especially the challenging AIME subset. Therefore, we can conclude that whether an LRM follows the constraints is correlated with the difficulty level of the math problem and constraints on easier math problems are more likely to be followed. More experimental results could be found in Appendix.

40

20

0 GSM8K MATH500Olympiad Minerva AIME

Figure 2: The accuracy on each subset MathIF averaged over 23 LRMs.

Next, we investigate the impact of the constraint number and plot the instruction-following accuracy of three LRMs in Figure 3. Unsurprisingly, more constraints are associated with the drop in hard accuracy. To be more specific, for all the 23 models involved in our experiment, we can observe an obvious deterioration in hard accuracy when increasing the number of constraints to 2 (double-constraint) and 3 (triple-constraint). However, it is worth noting that the same pattern is not applicable to the soft accuracy,

DeepSeek-R1-Distill-Qwen-1.5B

Open-Reasoner-Zero-7B

QwQ-32B

60

Accuracy(%)

40

20

- as the soft accuracy remains unchanged or even exhibits growth when more constraints are applied. It seems that the model’s ability to follow every individual constraint can be enhanced by the existence of more constraints.

0

Single Double Triple

Figure 3: The HAcc (solid line) and SAcc (dashed line) on the single/double/tripleconstraint subset.

100

100

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

80

80

Percentage(%)

Percentage(%)

23.33% 8.81%

24.52% 7.14%

Correct

Correct

60

60

40

40

48.81% 19.05%

58.33% 10.00%

Incorrect

Incorrect

20

20

Unfollowed Followed

Unfollowed Followed

0

0

100

100

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

80

80

Percentage(%)

Percentage(%)

45.24% 7.38%

27.62% 8.81%

Correct

Correct

60

60

40

40

41.19% 6.19%

50.24% 13.33%

Incorrect

Incorrect

20

20

Unfollowed Followed

Unfollowed Followed

0

0

Figure 4: Error set analysis for Qwen3-0.6B, DeepSeek-R1-Distill-Qwen-1.5B, Open-Reasoner-Zero-7B, and DeepSeek-R1-Distill-Llama-8B (from left to right).

## 5 When Scaling Reasoning Meets Losing Control

As discussed in Section 4.1, there may exist a trade-off between the instruction-following ability and the mathematical reasoning capability of LRMs. In this section, we further investigate this trade-off through a fine-grained error analysis (Section 5.1), examine the effects of different reasoning-oriented training paradigms (Section 5.2), and explore how CoT length impacts reasoning and instructionfollowing by applying both inference-time and training-aware interventions (Section 5.3).

#### 5.1 The Intelligence–Obedience Trade-off

Dilemma between Reasoning and Instruction Following. We begin by analyzing the relationship between reasoning and instruction-following through an error-based categorization. Each sample is grouped into one of four categories based on two criteria: (1) whether the math problem was solved correctly, and (2) whether all user-specified constraints were satisfied. The proportions of these four categories are shown in Figure 4. We observe that LRMs often struggle to fulfill both objectives simultaneously, as evidenced by the particularly low proportion of (Correct, Followed) cases. Interestingly, the proportion of (Correct, Followed) is even smaller than that of either (Correct, Unfollowed) or (Incorrect, Followed), suggesting that LRMs frequently sacrifice one objective to achieve the other. In other words, they are more likely to violate constraints when solving problems correctly, and more likely to fail in problem-solving when attempting to follow constraints. This observation is consistent with the trend in Table 3.

Table 3 (last column) shows a noticeable degradation in math problem correctness when constraints are introduced. Figure 5 further breaks down this effect by dataset. Surprisingly, we find that the drop rate on GSM8K (the easiest subset) is even higher than that on AIME (the hardest), a significantly more challenging benchmark. This suggests that the impact of constraints on reasoning performance is not necessarily correlated with problem difficulty. In conclusion, the tradeoff between instruction-following and reasoning appears to be a general phenomenon across difficulty levels. Notably, LRMs fine-tuned on long CoT traces (e.g., DeepSeek-R1 variants) tend to exhibit more severe performance degradation than RL-trained models like Qwen3-32B and QwQ-32B, possibly due to the inherent limitations of SFT [17].

DeepSeek-R1-Distill-Qwen-1.5B

DropRate

DeepSeek-R1-Distill-Llama-8B Qwen3-32b

60

| |
|---|

QwQ-32B

40

20

0 GSM8k MATH500 Minerva Olympiad AIME

Figure 5: Relative correctness drop of four LRMs across five subsets.

DeepSeek-R1-Distill-Llama-8B

Qwen3-32B

Qwen3-0.6B

Longer CoTs Impair Instruction Following. We further analyze the impact of CoT length on instructionfollowing performance. Specifically, for each LRM, we divide its benchmark outputs into six bins based on the number of tokens between the <think> and </think> delimiters. The resulting trends are shown in Figure 6. Across all three models—DeepSeek-R1-Distill-Llama-8B, Qwen3-0.6B, and Qwen3-32B—we observe a consistent decline in both hard accuracy and soft accuracy as CoT length increases, suggesting a negative correlation between generation length and instruction compliance. One possible explanation is that longer CoTs, while beneficial for

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

Accuracy(%)

0.8

0.6

0.4

0.2

1 2 3 4 5 6

Length (Bin Index)

Figure 6: HAcc (solid line) and SAcc (dashed line) across six CoT length bins; higher indices correspond to longer CoT generations.

reasoning, increase the distance between the user-specified constraint and the final answer. This may dilute the model’s attention to the constraint, making accurate instruction-following more difficult (see Section 5.3).

#### 5.2 How Does Reasoning-Oriented Training Affect Instruction-Following?

Motivated by the patterns observed in Figure 5, we further investigate how different reasoningoriented training paradigms affect a model’s instruction-following behavior. Specifically, we examine three representative strategies: (1) SFT-only, (2) SFT followed by RL (SFT+RL), and (3) cold-start RL, which bypasses SFT entirely.

Training Setup. We base our experiments on the DeepScaler dataset [45], which contains approximately 40k math reasoning samples. All training is conducted using 16 NVIDIA H100 GPUs. For SFT-only and SFT+RL settings, we first distill long CoT reasoning traces from QwQ-32B [49], filtering out samples where QwQ-32B fails to generate a correct answer or the CoT exceeds 8192 tokens. This results in 18k high-quality examples. We use models from the Qwen-2.5 and Qwen2.5-Math series as our base. Since some models are limited to 4096 position embeddings, we extend the RoPE [51] scaling factor θ from 10,000 to 20,000 to accommodate longer sequences, following prior work [21]. For reinforcement learning, we adopt the GRPO [52] framework and use verifiable outcome-based rewards. In addition to standard correctness rewards, we design a format-aware reward variant that grants 0.1 if the model includes special reasoning tokens (e.g., <think> and </think>) and 1.0 for a correct solution. Details can be referred to in the Appendix.

Table 4: Comparison of reasoning-oriented training strategies. Avg. Acc. denotes the average performance across all benchmarks. Cells shaded in green and red indicate increased and decreased performance, respectively, relative to the base model.

The Double-Edged Sword of Reasoning-Oriented Training. Table 4 presents the results for different training pathways. Avg. Acc. denotes the average benchmark performance with details shown in the Appendix. While both SFT and RL contribute to improved reasoning performance, neither of the training strategies enhances instruction-following. On the contrary, we observe a slight but consistent drop in both HAcc and SAcc across the board. The format-aware reward provides a minor improvement on Qwen-2.5-1.5B, 7B, but has a negligible effect on the Qwen-2.5-Math series. These findings reveal a double-edged nature of reasoning-oriented training: while it sharpens the model’s problemsolving ability, it simultaneously dulls its capacity to follow user instructions. This underscores a central dilemma in current training paradigms: enhancing intelligence often comes at the expense of obedience.

Model HAcc SAcc Avg. Acc. Qwen2.5-1.5B 10.00 27.26 1.21

+SFT 7.86 22.70 4.20

+SFT+RL 7.86 20.44 12.54 +cold-RL 9.52 23.97 14.58

w/ format reward 10.95 28.49 11.17

Qwen-2.5-7B 15.95 33.13 13.59 +SFT 7.86 21.03 23.10 +SFT+RL 7.62 21.07 32.82 +cold-RL 10.48 27.26 28.39

w/ format reward 14.52 32.50 24.80

Qwen2.5-Math-1.5B 9.28 23.33 18.91 +SFT 7.86 21.03 14.39 +SFT+RL 7.14 20.56 24.71 +cold-RL 8.33 21.31 24.88

w/ format reward 7.62 20.08 23.95

Qwen2.5-Math-7B 9.76 23.53 20.68 +SFT 8.09 22.06 29.11 +SFT+RL 8.57 21.03 40.65 +cold-RL 7.85 22.62 32.61

w/ format reward 7.86 21.79 32.66

#### 5.3 How does the CoT Length Affect Instruction Following?

Prior work has shown that generating longer chains of thought (CoT) can significantly improve a model’s ability to solve complex tasks [53]. However, as illustrated in Figure 6, we observe that scaling up CoT length can come at the cost of instruction-following accuracy. To better understand this phenomenon, we systematically investigate how CoT length influences instruction adherence.

The More Thinking, the Less Following. We begin by artificially increasing the CoT length using budget forcing [14], which appends the token "Wait" each time the model attempts to terminate the reasoning process. This encourages the model to continue generating longer CoTs. The experiment is performed on DeepSeek-R1-DistillQwen-1.5B, and Figure 7 shows the instruction-following performance as the number of budget-forcing steps N increases from 2 to 8. The results reveal a clear trend: SAcc steadily declines as CoT length increases, suggesting that excessively long CoTs may impair the model’s ability to follow instructions. This degradation likely stems from the increasing distance between the instruction and the final output, which may dilute the model’s attention to user-specified constraints.

Single

60

Double

50

Triple

SAcc

40

30

2 4 6 8

# Thinking Times

Figure 7: The trend of SAcc variation on GSM8K subset as the number of budget forcing increases from 2 to 8.

Controlling CoT Length During RL Training. Beyond inference-time manipulation, we investigate whether controlling the length of CoT during reinforcement learning has a similar impact on instruction-following. Specifically, we continue RL training on DeepSeek-R1-Distill-Qwen-1.5B using the DeepScaler dataset [45], varying the maximum response length during rollouts.

In this setup, overlong responses are truncated and receive no outcome reward, encouraging the model to learn more concise reasoning traces. We adopt a pure outcome-based reward function and conduct RL training for three epochs, varying the maximum rollout length from 1k to 8k tokens. The results, shown in Table 5, reveal a clear trend: as the maximum rollout length increases, math reasoning performance (averaged across AIME2024, AIME2025, AMC2023, Minerva, and Olympiad) improves, while both hard accuracy and soft accuracy consistently decline. This observation further reinforces our conclusion: reasoning-oriented training that favors longer CoTs can inadvertently harm instructionfollowing fidelity, highlighting a persistent trade-off between reasoning strength and obedience to user constraints.

Table 5: Impact of the maximum response length during RL. Cells shaded in red denote lower performance relative to the base (Original), with intensity proportional to the drop magnitude.

Model HAcc SAcc Avg. Acc. Original 17.14 36.62 36.13

- +cold-RL (1k) 19.05 39.88 28.73

- +cold-RL (2k) 16.43 36.75 36.32

+cold-RL (4k) 16.91 35.87 40.03 +cold-RL (8k) 14.29 34.13 39.82

Bringing Instructions Closer Improves Obedience at the Cost of Intelligence. One possible explanation for the negative impact of lengthy CoTs on instruction-following is that extended reasoning increases the distance between the user query and the final answer, making it more likely for the model to overlook the original constraint. To preliminarily verify this hypothesis, we propose a simple yet effective remedy: repeating the constraint at the end of the CoT. Concretely, we manually append the token "Wait" to prolong the CoT and then reintroduce the original constraint immediately afterward. As a result, the constraint appears twice in the input, i.e., once before the CoT begins and once again

Table 6: Effect of +repeat on model performance. Model HAcc SAcc Correctness DeepSeek-R1-Distill-Qwen-1.5B 17.14 36.62 31.67

+repeat 21.66 42.58 22.38 Open-Reasoner-Zero-7B 13.57 32.26 51.90 +repeat 14.53 33.14 30.00 Qwen3-32B 43.81 62.82 70.00 +repeat 59.29 68.34 63.81

- at the end, thereby shortening its contextual distance from the final answer. Experimental results on DeepSeek-R1-Distill-Qwen-1.5B, Open-Reasoner-Zero-7B, and Qwen3-32B are shown in Table 6. This straightforward intervention leads to clear improvements in instruction-following (SAcc and HAcc), albeit at a modest cost to problem-solving accuracy. These findings further confirm the inherent trade-off between reasoning depth and obedience during inference: enhancing one often comes at the expense of the other.

## 6 Conclusion

While large reasoning models continue to demonstrate impressive progress in mathematical problemsolving, our study reveals a persistent and underexplored trade-off between reasoning strength and instruction-following fidelity.

Through MathIF, a benchmark tailored for evaluating instruction adherence in math reasoning tasks, we show that reasoning scaling does not guarantee control. Empirical results reveal that longer chains of thought and reasoning-oriented training methods (e.g., SFT and RL) often impair a model’s ability to comply with user-specified constraints. These findings highlight a core tension in the development of LRMs: as models become more intelligent, they often become less controllable. This dilemma is central to the alignment problem in reasoning-centric systems. Addressing it requires rethinking current training paradigms to build models that can reason effectively without drifting from user intent. We hope that our benchmark and findings serve as a foundation for future research that bridges the growing gap between intelligence and obedience in large reasoning models.

## Bibliography

- [1] Xiaoye Qu, Yafu Li, Zhaochen Su, Weigao Sun, Jianhao Yan, Dongrui Liu, Ganqu Cui, Daizong Liu, Shuxian Liang, Junxian He, et al. A survey of efficient reasoning for large reasoning models: Language, multimodality, and beyond. arXiv preprint arXiv:2503.21614, 2025.
- [2] OpenAI. Introducing openai o3 and o4-mini. https://openai.com/index/ introducing-o3-and-o4-mini/.
- [3] DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025.
- [4] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, Chuning Tang, Congcong Wang, Dehao Zhang, Enming Yuan, Enzhe Lu, Fengxiang Tang, Flood Sung, Guangda Wei, Guokun Lai, Haiqing Guo, Han Zhu, Hao Ding, Hao Hu, Hao Yang, Hao Zhang, Haotian Yao, Haotian Zhao, Haoyu Lu, Haoze Li, Haozhen Yu, Hongcheng Gao, Huabin Zheng, Huan Yuan, Jia Chen, Jianhang Guo, Jianlin Su, Jianzhou Wang, Jie Zhao, Jin Zhang, Jingyuan Liu, Junjie Yan, Junyan Wu, Lidong Shi, Ling Ye, Longhui Yu, Mengnan Dong, Neo Zhang, Ningchen Ma, Qiwei Pan, Qucheng Gong, Shaowei Liu, Shengling Ma, Shupeng Wei, Sihan Cao, Siying Huang, Tao Jiang, Weihao Gao, Weimin Xiong, Weiran He, Weixiao Huang, Wenhao Wu, Wenyang He, Xianghui Wei, Xianqing Jia, Xingzhe Wu, Xinran Xu, Xinxing Zu, Xinyu Zhou, Xuehai Pan, Y. Charles, Yang Li, Yangyang Hu, Yangyang Liu, Yanru Chen, Yejie Wang, Yibo Liu, Yidao Qin, Yifeng Liu, Ying Yang, Yiping Bao, Yulun Du, Yuxin Wu, Yuzhi Wang, Zaida Zhou, Zhaoji Wang, Zhaowei Li, Zhen Zhu, Zheng Zhang, Zhexu Wang, Zhilin Yang, Zhiqi Huang, Zihao Huang, Ziyao Xu, and Zonghan Yang. Kimi k1.5: Scaling reinforcement learning with llms, 2025.
- [5] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems, 2024.
- [6] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.
- [7] Hemish Veeraboina. Aime problem set 1983-2024, 2023.
- [8] Z. Z. Ren, Zhihong Shao, Junxiao Song, Huajian Xin, Haocheng Wang, Wanjia Zhao, Liyue Zhang, Zhe Fu, Qihao Zhu, Dejian Yang, Z. F. Wu, Zhibin Gou, Shirong Ma, Hongxuan Tang, Yuxuan Liu, Wenjun Gao, Daya Guo, and Chong Ruan. Deepseek-prover-v2: Advancing formal mathematical reasoning via reinforcement learning for subgoal decomposition, 2025.
- [9] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.
- [10] Yi Su, Dian Yu, Linfeng Song, Juntao Li, Haitao Mi, Zhaopeng Tu, Min Zhang, and Dong Yu. Crossing the reward bridge: Expanding rl with verifiable rewards across diverse domains, 2025.
- [11] Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, Saizhuo Wang, Kun Zhang, Yuanzhuo Wang, Wen Gao, Lionel Ni, and Jian Guo. A survey on llm-as-a-judge, 2025.
- [12] Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023.
- [13] Yuxin Jiang, Yufei Wang, Xingshan Zeng, Wanjun Zhong, Liangyou Li, Fei Mi, Lifeng Shang, Xin Jiang, Qun Liu, and Wei Wang. Followbench: A multi-level fine-grained constraints following benchmark for large language models. arXiv preprint arXiv:2310.20410, 2023.

- [14] Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling. arXiv preprint arXiv:2501.19393, 2025.
- [15] Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. Limo: Less is more for reasoning, 2025.
- [16] Edward Yeo, Yuxuan Tong, Morry Niu, Graham Neubig, and Xiang Yue. Demystifying long chain-of-thought reasoning in llms, 2025.
- [17] Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Sergey Levine, and Yi Ma. SFT memorizes, RL generalizes: A comparative study of foundation model post-training. In The Second Conference on Parsimony and Learning (Recent Spotlight Track), 2025.
- [18] Zichen Liu, Changyu Chen, Wenjun Li, Tianyu Pang, Chao Du, and Min Lin. There may not be aha moment in r1-zero-like training — a pilot study. https://oatllm.notion.site/ oat-zero, 2025. Notion Blog.
- [19] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.
- [20] Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, et al. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456, 2025.
- [21] Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, and Yue Zhang. Learning to reason under off-policy guidance, 2025.
- [22] Wang Yang, Hongye Jin, Jingfeng Yang, Vipin Chaudhary, and Xiaotian Han. Thinking preference optimization. arXiv preprint arXiv:2502.13173, 2025.
- [23] Yann Dubois, Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Alpacafarm: A simulation framework for methods that learn from human feedback, 2023.
- [24] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023.
- [25] Congying Xia, Chen Xing, Jiangshu Du, Xinyi Yang, Yihao Feng, Ran Xu, Wenpeng Yin, and Caiming Xiong. FOFO: A benchmark to evaluate LLMs’ format-following capability. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 680–699, Bangkok, Thailand, August 2024. Association for Computational Linguistics.
- [26] Xiangru Tang, Yiming Zong, Jason Phang, Yilun Zhao, Wangchunshu Zhou, Arman Cohan, and Mark Gerstein. Struc-bench: Are large language models good at generating complex structured tabular data? In Kevin Duh, Helena Gomez, and Steven Bethard, editors, Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), pages 12–34, Mexico City, Mexico, June 2024. Association for Computational Linguistics.
- [27] Yun He, Di Jin, Chaoqi Wang, Chloe Bi, Karishma Mandyam, Hejia Zhang, Chen Zhu, Ning Li, Tengyu Xu, Hongjiang Lv, et al. Multi-if: Benchmarking llms on multi-turn and multilingual instructions following. arXiv preprint arXiv:2410.15553, 2024.
- [28] Jinnan Li, Jinzhe Li, Yue Wang, Yi Chang, and Yuan Wu. Structflowbench: A structured flow benchmark for multi-turn instruction following. arXiv preprint arXiv:2502.14494, 2025.
- [29] Chi Han. Can language models follow multiple turns of entangled instructions? arXiv preprint arXiv:2503.13222, 2025.

- [30] Ved Sirdeshmukh, Kaustubh Deshpande, Johannes Mols, Lifeng Jin, Ed-Yeremai Cardona, Dean Lee, Jeremy Kritz, Willow Primack, Summer Yue, and Chen Xing. Multichallenge: A realistic multi-turn conversation evaluation benchmark challenging to frontier llms. arXiv preprint arXiv:2501.17399, 2025.
- [31] Xiaodong Wu, Minhao Wang, Yichen Liu, Xiaoming Shi, He Yan, Xiangju Lu, Junmin Zhu, and Wei Zhang. Lifbench: Evaluating the instruction following performance and stability of large language models in long-context scenarios. arXiv preprint arXiv:2411.07037, 2024.
- [32] Zhenyu Li, Kehai Chen, Yunfei Long, Xuefeng Bai, Yaoyin Zhang, Xuchen Wei, Juntao Li, and Min Zhang. Xifbench: Evaluating large language models on multilingual instruction following. arXiv preprint arXiv:2503.07539, 2025.
- [33] Zhihan Zhang, Shiyang Li, Zixuan Zhang, Xin Liu, Haoming Jiang, Xianfeng Tang, Yifan Gao, Zheng Li, Haodong Wang, Zhaoxuan Tan, Yichuan Li, Qingyu Yin, Bing Yin, and Meng Jiang. IHEval: Evaluating language models on following the instruction hierarchy. In Luis Chiruzzo, Alan Ritter, and Lu Wang, editors, Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 8374–8398, Albuquerque, New Mexico, April

2025. Association for Computational Linguistics.

- [34] Shirley Anugrah Hayati, Taehee Jung, Tristan Bodding-Long, Sudipta Kar, Abhinav Sethy, Joo-Kyung Kim, and Dongyeop Kang. Chain-of-instructions: Compositional instruction tuning on large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 24005–24013, 2025.
- [35] Jianhao Yan, Yun Luo, and Yue Zhang. RefuteBench: Evaluating refuting instruction-following for large language models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Findings of the Association for Computational Linguistics: ACL 2024, pages 13775–13791, Bangkok, Thailand, August 2024. Association for Computational Linguistics.
- [36] Jianhao Yan, Yun Luo, and Yue Zhang. Refutebench 2.0 – agentic benchmark for dynamic evaluation of llm responses to refutation instruction, 2025.
- [37] Bosi Wen, Pei Ke, Xiaotao Gu, Lindong Wu, Hao Huang, Jinfeng Zhou, Wenchuang Li, Binxin Hu, Wendy Gao, Jiaxing Xu, et al. Benchmarking complex instruction-following with multiple constraints composition. Advances in Neural Information Processing Systems, 37:137610– 137645, 2024.
- [38] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [39] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.
- [40] Arup De, Maya Gokhale, Rajesh Gupta, and Steven Swanson. Minerva: Accelerating data analysis in next-generation ssds. In 2013 IEEE 21st Annual International Symposium on Field-Programmable Custom Computing Machines, pages 9–16. IEEE, 2013.
- [41] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.
- [42] Qwen Team. Qwen3, April 2025.
- [43] Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild. arXiv preprint arXiv:2503.18892, 2025.

- [44] An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu, Xingzhang Ren, and Zhenru Zhang. Qwen2.5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122, 2024.
- [45] Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl, 2025. Notion Blog.
- [46] Pranjal Aggarwal and Sean Welleck. L1: Controlling how long a reasoning model thinks with reinforcement learning. arXiv preprint arXiv:2503.04697, 2025.
- [47] Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model. arXiv preprint arXiv:2503.24290, 2025.
- [48] Hugging Face. Open r1: A fully open reproduction of deepseek-r1, January 2025.
- [49] Qwen Team. Qwq-32b: Embracing the power of reinforcement learning, March 2025.
- [50] Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model, 2025.
- [51] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.
- [52] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [53] Mingyu Jin, Qinkai Yu, Dong Shu, Haiyan Zhao, Wenyue Hua, Yanda Meng, Yongfeng Zhang, and Mengnan Du. The impact of reasoning step length on large language models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Findings of the Association for Computational Linguistics: ACL 2024, pages 1830–1842, Bangkok, Thailand, August 2024. Association for Computational Linguistics.

## A Overview of the Appendix

This Appendix is organized as follows:

- • Section B and Section C discussed the limitation and ethic considerations of our study, respectively.
- • Section D elaborate on the hyper-parameters used for our reasoning-oriented training in Section 5.2.
- • Section E provides more detailed results on our benchmark to facilitate analysis on the difficulty of math problems and the number of constraints.
- • Section F contains detailed reasoning performance for LRMs trained in Section 5.
- • Section G provides more details for our preliminary observation on existing instruction-following benchmarks.
- • Section H lists the constraints used in our proposed MathIF benchmark.

## B Limitations

The limitations of this study can be summarized as below:

- • In this study, we evaluate 23 recently released LRMs in text modality, and we plan to leave the benchmarking of large vision reasoning models for future work.
- • When investigating how reasoning-oriented training affects instruction-following, we mainly use GRPO [52] for RL training because of its simplicity, stability, and widespread practical adoption. Experimenting with other RL training algorithms is left for future work.

## C Ethics Considerations

Our proposed MathIF evaluates the instruction-following ability of publicly released LRMs, adhering strictly to the Neurips Code of Ethics. The math problems used for our benchmark are collected from free public datasets, and the construction of our benchmark does not involve recruiting crowdsource workers or human annotators. Our benchmark should only be used for research, not for any malicious purpose.

## D Hyper-parameter Setting

Our experiments on different reasoning-oriented training strategies in Section 5.2 are conducted on a cloud Linux server with Ubuntu 16.04 operating system. The codes are written in Python 3.10 with the huggingface libraries2. We run our experiments on 16 Nvidia H100 with 80GiB GPU memory. The detailed hyper-parameter settings for supervised fine-tuning and reinforcement learning are shown in Table 7, which mostly follow the default setting in VeRL framework 3.

- 2https://github.com/huggingface/transformers
- 3https://github.com/volcengine/verl

- Table 7: The value of the hyper-parameters in our reasoning-oriented training experiment (Section 5.2) for SFT (left) and RL (right).

Hyper-parameter Value

batch_size 256 micro_batch_size 1 max_length 8192 rope_theta 20000 lr 1e-6 betas (0.9, 0.95) weight_decay 0.01 warmup_ratio 0.1 schedule cosine clip_grad 1 epoch 3 truncation right sliding_window none

Hyper-parameter Value

max_prompt_length 1024 max_response_length 3072 lr 1e-6 batch_size 128 mini_batch_size 64 grad_clip 1 clip_ratio 0.2 entropy_coeff 0.001 kl_loss_coef 0.001 rl_epoch 1 warmup_ratio 0 schedule constant rollout_n 8 rollout_temperature 1

E More Benchmark Results

- In Section 4.3, we visualize the model performance grouped by the source of math problems and the number of constraints. In this section, we supplement with more detailed benchmark results for LRMs involved in our experiments. The fine-grained instruction-following performance across different source of math problems are presented in Table 8, while the hard accuracy (HAcc) and soft accuracy (SAcc) for different number of constraints are shown in Table 9 and Table 10, respectively.

- Table 8: Experimental results of LRMs on MathIF. We report hard accuracy (HAcc) and soft accuracy (SAcc) for instruction-following. † indicates models trained by supervised fine-tuning only (no reasoning-oriented RL).

single double triple

Model

Acc HAcc SAcc HAcc SAcc Models with no more than 4B parameters

Qwen3-4B 53.57 38.57 57.86 40.00 72.86 Qwen3-1.7B 42.14 22.86 46.43 25.71 62.14 Qwen3-0.6B 48.57 22.86 48.93 12.14 53.81 L1-Qwen-1.5B-Exact 33.57 18.57 43.57 7.14 41.66 L1-Qwen-1.5B-Max 37.14 16.43 43.93 5.71 37.14 DeepSeek-R1-Distill-Qwen-1.5B† 33.57 14.29 38.21 3.57 38.09 DeepScaler-1.5B-Preview 30.71 10.00 35.00 2.86 37.85 Qwen2.5-1.5B-SimpleRL-Zoo 21.43 2.86 21.07 2.86 30.48 Qwen2.5-Math-1.5B-Instruct 19.29 2.14 19.64 1.43 25.24

Models with approximately 7B–14B parameters

Qwen3-14B 63.57 40.71 60.71 47.86 76.90 DeepSeek-R1-Distill-Qwen-14B† 57.14 35.71 62.86 25.00 61.66 Qwen3-8B 51.43 31.43 54.64 30.71 65.95 DeepSeek-R1-Distill-Qwen-7B† 39.29 27.14 50.36 12.86 45.23 DeepSeek-R1-Distill-Llama-8B† 34.29 22.14 47.14 10.00 50.7 Open-Reasoner-Zero-7B 25.71 13.57 39.64 1.43 31.42 Qwen2.5-Math-7B-Instruct 22.86 2.86 24.64 1.43 29.29

Models with 32B or more parameters

Qwen3-32B 61.43 35.00 57.50 35.00 69.52 DeepSeek-R1-Distill-Qwen-32B† 57.14 37.14 60.36 33.57 65.23 DeepSeek-R1-Distill-Llama-70B† 54.29 39.29 61.07 30.71 67.85 QwQ-32B 55.71 35.71 58.57 29.29 65.69 OlympicCoder-32B† 55.71 31.43 60.36 20.71 57.85 s1-32B† 37.14 13.57 38.93 12.14 49.27 Open-Reasoner-Zero-32B 30.71 13.57 41.79 2.14 34.05

- Table 9: Experimental results of LRMs on MathIF. We report hard accuracy (HAcc) for instruction-following on five subsets of our MathIF. † indicates models trained by supervised fine-tuning only (no reasoning-oriented RL).

Model GSM8K MATH500 Minerva Olympiad AIME Models with no more than 4B parameters

Qwen3-4B 66.67 40.00 53.33 31.11 21.67 Qwen3-1.7B 44.44 25.56 41.11 24.44 8.33 Qwen3-0.6B 36.67 25.56 34.44 24.44 13.33 L1-Qwen-1.5B-Exact 27.78 15.56 21.11 17.78 15.00 L1-Qwen-1.5B-Max 24.44 18.89 22.22 16.67 15.00 DeepSeek-R1-Distill-Qwen-1.5B† 32.22 12.22 15.56 12.22 11.67 DeepScaler-1.5B-Preview 26.67 10.00 15.56 7.78 11.67 Qwen2.5-1.5B-SimplRL-Zoo 11.11 10.00 11.11 4.44 8.33 Qwen2.5-Math-1.5B-Instruct 8.89 5.56 8.89 6.67 8.33

Models with approximately 7B–14B parameters

Qwen3-14B 71.11 53.33 63.33 35.56 20.00 DeepSeek-R1-Distill-Qwen-14B† 55.56 35.56 44.44 31.11 25.00 Qwen3-8B 56.67 37.78 44.44 24.44 20.00 DeepSeek-R1-Distill-Qwen-7B† 46.67 22.22 31.11 14.44 13.33 DeepSeek-R1-Distill-Llama-8B† 41.11 18.89 20.00 13.33 15.00 Open-Reasoner-Zero-7B 13.33 14.44 11.11 13.33 16.67 Qwen2.5-Math-7B-Instruct 12.22 5.56 10.00 8.89 8.33

Models with 32B or more parameters

Qwen3-32B 73.33 40.00 52.22 26.67 18.33 DeepSeek-R1-Distill-Qwen-32B† 57.78 38.89 52.22 32.22 26.67 DeepSeek-R1-Distill-Llama-70B† 55.56 42.22 53.33 28.89 20.00 QwQ-32B 60.00 38.89 45.56 32.22 16.67 OlympicCoder-32B† 36.67 36.67 37.78 31.11 38.33 s1-32B† 33.33 20.00 22.22 13.33 13.33 Open-Reasoner-Zero-32B 15.56 14.44 15.56 14.44 18.33

- Table 10: Experimental results of LRMs on MathIF. We report soft accuracy (SAcc) for instruction-following on five subsets of our MathIF. † indicates models trained by supervised fine-tuning only (no reasoning-oriented RL).

Model GSM8K MATH500 Minerva Olympiad AIME Models with no more than 4B parameters

Qwen3-4B 80.19 57.41 70.37 50.19 42.78 Qwen3-1.7B 65.74 44.81 61.85 45.19 25.28 Qwen3-0.6B 61.3 47.04 59.07 45.37 33.89 L1-Qwen-1.5B-Exact 50.56 37.59 39.62 33.7 35 L1-Qwen-1.5B-Max 45.37 40.56 42.78 34.44 31.1 DeepSeek-R1-Distill-Qwen-1.5B† 54.26 32.59 37.03 28.7 27.5 DeepScaler-1.5B-Preview 49.44 32.96 33.89 25.56 28.88 Qwen2.5-1.5B-SimplRL-Zoo 25.93 25 27.96 18.7 23.89 Qwen2.5-Math-1.5B-Instruct 22.41 19.07 23.33 20.37 21.94

Models with approximately 7B–14B parameters

Qwen3-14B 83.33 68.52 77.96 55.56 41.39 DeepSeek-R1-Distill-Qwen-14B† 76.84 58.14 62.22 55.56 44.72 Qwen3-8B 74.44 55.74 64.07 45 42.5 DeepSeek-R1-Distill-Qwen-7B† 67.96 41.67 52.59 29.44 27.22 DeepSeek-R1-Distill-Llama-8B† 62.59 42.22 43.51 35.93 31.93 Open-Reasoner-Zero-7B 32.22 32.78 31.67 29.62 36.38 Qwen2.5-Math-7B-Instruct 29.63 20.93 27.41 25.19 24.44

Models with 32B or more parameters

Qwen3-32B 86.11 59.26 70.74 48.89 42.22 DeepSeek-R1-Distill-Qwen-32B† 75.73 60.37 67.78 50.73 44.44 DeepSeek-R1-Distill-Llama-70B† 75.73 60.93 70.56 48.89 43.33 QwQ-32B 78.14 57.03 66.67 51.11 40.50 OlympicCoder-32B† 58.89 55.92 64.26 54.26 55.83 s1-32B† 54.81 43.51 45.56 31.48 29.43 Open-Reasoner-Zero-32B 36.85 33.52 37.04 33.15 37.78

## F More Results on Math Benchmarks

- In Section 5.3, we control the CoT length during RL training and report the averaged math reasoning performance among five benchmarks (AIME2024, AIME2025, AMC2023, Minerva, and Olympiad) in Table 5. For fine-grained analysis, we report more detailed results on five benchmarks in Table 11.

- Table 11: Reasoning performance for LRMs when trained with varying maximum response length (the number in the bracket) during RL.

Model AIME2024 AIME2025 AMC2023 Minerva Olympiad Average Original 28.33 21.15 67.73 23.16 40.30 36.13

- +cold-RL (1k) 14.27 11.67 58.20 23.53 36.00 28.73
- +cold-RL (2k) 24.06 19.58 70.39 26.10 41.48 36.32

+cold-RL (4k) 28.65 24.17 75.39 26.47 45.48 40.03 +cold-RL (8k) 30.73 24.06 73.05 26.84 44.44 39.82

- Table 12: Reasoning performance for LRMs when trained with different reasoning-oriented training strategies. AIME2024 AIME2025 AMC2023 Minerva Olympiad Average

Qwen2.5-1.5B 0.21 0.00 2.89 1.47 1.48 1.21 +SFT 0.10 0.10 10.70 4.04 6.07 4.20

+SFT+RL 4.48 2.08 28.36 9.56 18.22 12.54 +cold-RL 4.48 2.19 30.47 16.18 19.56 14.58

w/ format reward 2.60 0.31 26.80 9.56 16.59 11.17

Qwen2.5-7B 4.90 1.98 27.81 13.24 20.00 13.59 +SFT 10.00 10.52 40.78 25.00 29.19 23.10 +SFT+RL 18.65 18.23 57.34 27.94 41.93 32.82 +cold-RL 15.21 8.75 53.98 29.78 34.22 28.39

w/ format reward 10.52 8.13 46.56 27.21 31.56 24.80

Qwen2.5-Math-1.5B 7.92 4.27 42.89 14.71 24.74 18.91 +SFT 5.94 3.65 30.08 13.60 18.67 14.39 +SFT+RL 10.94 9.27 48.75 23.16 31.41 24.71 +cold-RL 13.30 7.70 52.00 20.58 30.81 24.88

w/ format reward 12.81 6.46 51.95 20.22 28.30 23.95

Qwen2.5-Math-7B 16.45 8.13 45.63 7.72 25.48 20.68 +SFT 16.88 15.94 53.36 25.00 34.37 29.11 +SFT+RL 30.21 23.96 70.55 31.25 47.26 40.65 +cold-RL 27.50 13.60 59.84 25.36 36.74 32.61

w/ format reward 28.75 11.15 62.50 26.10 34.81 32.66

G More Results on Previous Instruction-following Benchmarks.

As shown in Figure 1 and Table 13, we evaluate the instruction-following performance of three LRMs together with the Instruction models from which they are trained, and report the results on IFEval [12] and FollowBench [13]. From the table, we can observe that reasoning-oriented training seems to be detrimental to instruction-following.

- Table 13: Instruction-following performance for three Instruction models and LRMs trained on them (highlighted in gray). We report the prompt-level strict accuracy for IFEval and use GPT-4o-mini as evaluator in FollowBench.

###### IFEval FollowBench

Llama-3.3-70B-Instruct 90.38 61.82 DeepSeek-R1-Distill-Llama-70B 78.74 49.26

Qwen2.5-32B-Instruct 80.96 60.12 s1-32B 58.04 51.46

Qwen2.5-Coder-32B-instruct 80.03 59.14 OlympicCoder-32B 57.11 46.70

### H List of Constraints In this section we provide a detailed list of the 15 constraints used in our benchmark in Table 14.

Table 14: The list of 15 constraints used in our proposed MathIF.

Category Constraint length • Answer with at least/around/most {N} words. lexical • Include keywords {keyword1}, {keyword2} in your response.

- • In your response, the word word should appear {N} times.
- • Do not include keywords {forbidden words} in the response.
- • Your ENTIRE response should be in {language}, no other language is allowed.

format • Your answer must contain exactly {N} bullet points. Use the markdown bullet points

such as: * This is a point.

- • Highlight at least {N} sections in your answer with markdown, i.e. highlighted section.
- • Your response must have {N} sections. Mark the beginning of each section with {section_splitter} X.
- • Your entire response should be in English, capital letters only.
- • Your entire response should be in English, and in all lowercase letters. No capital letters are allowed.
- • In your response, words with all capital letters should appear at least / around / at most {N} times.
- • In your entire response, refrain from the use of any commas.

affix • Finish your response with this exact phrase {end_phrase}. No other words should follow

this phrase.

- • Wrap your entire response with double quotation marks.
- • First, repeat the request without change, then give your answer.

