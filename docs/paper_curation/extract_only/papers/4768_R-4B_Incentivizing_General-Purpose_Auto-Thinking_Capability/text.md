# arXiv:2508.21113v2[cs.CV]2Sep2025

[Figure 1]

## -4B: INCENTIVIZING GENERAL-PURPOSE AUTOTHINKING CAPABILITY IN MLLMS VIA BI-MODE ANNEALING AND REINFORCE LEARNING

1Tencent Hunyuan Team 2Institute of Automation, CAS

https://github.com/yannqi/R-4B

https://huggingface.co/YannQi/R-4B

ABSTRACT

Multimodal Large Language Models (MLLMs) equipped with step-by-step thinking capabilities have demonstrated remarkable performance on complex reasoning problems. However, this thinking process is redundant for simple problems solvable without complex reasoning. To address this inefficiency, we propose R-4B, an auto-thinking MLLM, which can adaptively decide when to think based on problem complexity. The central idea of R-4B is to empower the model with both thinking and non-thinking capabilities using bi-mode annealing, and apply Bimode Policy Optimization (BPO) to improve the model’s accuracy in determining whether to activate the thinking process. Specifically, we first train the model on a carefully curated dataset spanning various topics, which contains samples from both thinking and non-thinking modes. Then it undergoes a second phase of training under an improved GRPO framework, where the policy model is forced to generate responses from both modes for each input query. Experimental results show that R-4B achieves state-of-the-art performance across 25 challenging benchmarks. It outperforms Qwen2.5-VL-7B in most tasks and achieves performance comparable to larger models such as Kimi-VL-A3B-Thinking-2506 (16B) on reasoning-intensive benchmarks with lower computational cost.

R-4B [AutoThink] Keye-VL-8B [AutoThink] InternVL3.5-4B Kimi-VL-A3B-Thinking-2506 InternVL3-8B Qwen2.5-VL-7B

80

73.1

72.8

70.4

68.7

68.1

70

66.8

66.6

64.9

65.0

64.1

64.0

62.2

61.7

59.1

Accuracy(%)

60

58.0

57.4

56.8

56.4

51.0

50.6

50

47.7

44.5

43.6

42.5

41.2

40.0 40.8

39.5

39.6

40

37.6

35.7

35.3

32.4

30

27.1

23.9

20.1

20

10

MMMU MMStar CharXiV(RQ) MathVerse-Vision DynaMath LogicVista

Figure 1: Comparison between R-4B-RL and frontier open-source MLLMs, including non-thinking MLLMs (e.g., InternVL3-8B, Qwen2.5-VL 7B), thinking MLLMs (e.g., Kimi-VL-A3B-Thinking2506) and auto-thinking MLLMs (e.g., Keye-VL-8B), on different benchmarks.

- 1 INTRODUCTION

Multimodal Large Language Models (MLLMs) have made significant progress in recent years, particularly through the integration of explicit, step-by-step thinking processes (Guo et al., 2025b; Yue et al., 2025; Yang et al., 2025b). These models employ structured token blocks to distinguish between the exploratory thinking process and the precise answer generation. Specifically, exploratory thinking is incorporated into the <think> </think> block. It involves detailed step-by-step deduction and leverages reflection to explore alternatives or correct past reasoning. In contrast, precise answer generation is dedicated to producing concise and clear results. As shown in Figure 2, this thinking capability has substantially improved performance on complex problems such as mathematical reasoning and scientific diagram interpretation (Yue et al., 2023; Lu et al., 2024; Guo et al., 2025c). However, the default always-thinking behavior leads to unnecessary computational overhead on simple problems, such as a query like “What is the name of the dish?”. This motivates a more intelligent auto-thinking paradigm, i.e., the model can automatically decide whether to enable thinking based on the complexity of the problem.

Thinking Preference Thinking Response

[Figure 2]

###### Non-thinking Preference

|[Figure 3]<br><br>74.7%<br><br>73.9%<br><br>70.2%<br><br>32.2%<br><br>27.1%<br><br>1.6%<br><br>25.3%<br>26.1%<br><br><br>29.8%<br><br>67.8%<br><br>72.9%<br><br>98.4% Thinking Rate<br><br>[Figure 4]<br><br>Non-Thinking Rate<br><br>[Figure 5]| | | | |
|---|---|---|---|---|
| | | | | |

OCRBench

Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end. Question: Complete the matrix.

What is the name of this dish? How does it taste?

<think> Got it, let's try to figure out this matrix problem. First, I need to analyze the patterns in the rows and columns. Let's look at the shapes and their positions.

e

MMVet

[Figure 6]

[Figure 7]

HallusionBench

Let's look at the first row:

- - First cell: outer square, inner hexagon
- - Second cell: outer hexagon, inner circle
- - Third cell: outer circle, inner square Second row:

MathVista

Non-Thinking Response

[Figure 8]

AI2D

This dish is called 'mapo tofu' and is a spicy and flavorful Chinese dish. It has a spicy, savory taste with a mix of bold and mild flavors.

... </think> Answer:

WeMath

D

0% 20% 40% 60% 80% 100%

Thinking Triggering Rate (%)

Figure 2: Non-thinking and thinking mode response examples (left); Auto-thinking triggering rates across multiple benchmarks (right).

Previous explorations into auto-thinking have shown promise but come with their own limitations. For instance, some models like Qwen3 (Yang et al., 2025a) require users to manually enable the thinking mode within one model. To further automate the thinking process, other methods (Lou et al., 2025; Zhan et al., 2025) achieve auto-thinking by relying on manually curated data or complex reward functions during RL. However, training for these methods is dependent on carefully tuned reward strategies, and they are restricted to text-only modalities. More recent work such as Keye-VL (Yang et al., 2025b) attempts multimodal auto-thinking for the first time by constructing data with explicit complexity analysis to trigger thinking. While effective, it requires manually constructing training data with complex analysis, which makes it imprecise. Moreover, it introduces an additional token cost during inference due to the additional complexity analysis. These challenges demand smarter, more computationally efficient solutions for auto-thinking.

In this study, we introduce R-4B, a MLLM designed for general-purpose auto-thinking. It can autonomously switch between complex reasoning and direct responses directly based on the user’s query. This capability is enabled by a novel training paradigm for content-aware auto-thinking.

Considering that the foundation for auto-thinking is built upon a model mastering two distinct modes for general-purpose application, i.e., thinking and non-thinking, we first propose bi-mode annealing, which is designed to train a model that is inherently capable of both thinking and non-thinking modes in general domains. To support this annealing, we develop a bi-mode data curation strategy to carefully construct general reasoning-intensive and direct-answer datasets. This strategy separates: (a) reasoning-intensive examples requiring reasoning (e.g., diagram analysis, logical deduction), and (b) non-reasoning examples demanding direct factual responses. Both types are formatted in a unified instruction-following structure without extra complexity analysis. Then, we perform bimode annealing by mixing these datasets and obtain R-4B-Base. This lays a solid foundation for the model’s subsequent auto-thinking training in general-purpose domains.

After the annealing stage, R-4B possesses both thinking and non-thinking abilities. However, with regard to auto-thinking, it exhibits a preference for non-thinking even for complex queries, due to the uneven distribution of reasoning and non-reasoning data. This highlights a lack of judgment in

###### Adaptive Reasoning Data Selection

Objective Reasoning Response

[Figure 9]

[Figure 10]

Easy Samples

<think> Got it, let's see. I need to figure out ... <\think> These features match the parent logarithmic function y = log(x). The answer is D.

[Figure 11]

[Figure 12]

Offline Hard Mining

Reasoning Data

[Figure 13]

Hard Samples

Multiple Choice Math

[Figure 14]

[Figure 15]

Subjective

Non-Reasoning Response

[Figure 16]

Need Reasoning

[Figure 17]

[Figure 18]

Non-Reasoning Data

[Figure 19]

Prompt Engineering

<think> <\think> In the picture, there is a cat sleeping, and behind it is a tree.

No Need Reasoning

Caption Creation

Figure 3: Framework of heuristic-driven strategy for bi-mode data curation.

mode selection. Therefore, to further incentivize auto-thinking, we propose Bi-mode Policy Optimization (BPO), a reinforcement learning algorithm tailored for auto-thinking. Unlike existing RL methods (Zhang et al., 2025; Lou et al., 2025; Yang et al., 2025b) that require complex reward functions, extensive data dependency, or are prone to hyperparameter sensitivity, BPO leverages a simple, rule-based mathematical reward. We find that the proposed approach possesses promising generalization, achieving auto-thinking in other topics as well. Specifically, the core of BPO lies in the proposed bi-mode rollouts, which incorporate both thinking and non-thinking response trajectories simultaneously. By forcing the inclusion of thinking and non-thinking modes, the model is prevented from favoring a certain mode during RL training. This mechanism enables the model to learn an adaptive policy for optimal thinking strategy selection, leading to R-4B-RL with enhanced adaptive thinking and improved performance across modes.

[Figure 20]

Experiments demonstrate that R-4B-RL achieves excellent performance in auto-thinking mode, achieving an optimal balance between maximizing performance and minimizing computational overhead. In comprehensive evaluations across multiple public benchmarks, R-4B-RL outperforms Qwen2.5-VL-7B on nearly all tasks. Furthermore, as shown in Figure 1, R-4B-RL surpasses the performance of the significantly larger Kimi-VL-Thinking-2506 (3B activated, 16B total parameters) on reasoning-intensive benchmarks. To promote further advancement in MLLMs, we open-source R-4B-RL, which sets a new state of the art among models of comparable scale.

- 2 THE DESIGN OF BI-MODE ANNEALING

- 2.1 A HEURISTIC-DRIVEN STRATEGY FOR BI-MODE DATA CURATION

The goal of bi-mode annealing is to develop a model that can master two distinct response modes: thinking and non-thinking. To achieve this, we propose a general-purpose bi-mode data curation strategy, which systematically partitions data into reasoning and non-reasoning data without laborious manual annotation. This strategy leverages a powerful existing MLLM, Qwen2.5-32B-VL (Bai et al., 2025), to serve as a consistent annotator, partitioning data into two distinct categories: reasoning data and non-reasoning data.

As illustrated in Figure 3, our methodology employs two distinct heuristics tailored to the type of the query: (i) Difficulty-based heuristic (for subjective queries): For queries where correctness is not easily verifiable (e.g., creative or open-ended questions), we leverage prompt engineering with an existing MLLM to assess whether they require a reasoning process based on their inherent difficulty. Queries deemed complex are labeled as reasoning-intensive. (ii) Performance-based heuristic (for objective queries): For queries with verifiable answers (e.g., math or multiple choice questions), we introduce a model-based offline hard mining strategy to systematically identify difficult samples. Specifically, we generate multiple responses for each query (N = 8); if all attempts fail (hard samples), the query is classified as reasoning-intensive. Conversely, if the model can answer correctly (easy samples), the query is marked as suitable for a direct answer.

Subsequently, for the identified reasoning samples, we first employ the multimodal reasoning model (Guo et al., 2025b) to extract the reasoning context. Then, to ensure the quality of the generated thinking processes, we filter out invalid samples through consistency verification, keyword filtering and duplicate detection. The resulting data proportions are shown in Figure 4. This unified data curation strategy ensures consistent annotation, eliminating inconsistencies inherent in manual labeling while adaptively addressing both question types through automated decision protocols.

Topic Category Non-reasoning Items Reasoning Items Total Proportion General 1,351,060 1,365,693 2,716,753 16% Math/K12 1,908,486 1,821,412 4,088,776 23% Code 643,323 161,085 804,408 5% Chart 1,351,060 1,088,858 2,439,918 15% OCR 1,366,849 225,146 1,591,995 10% Grounding 280,740 148,710 429,450 3% Caption 1,166,676 133,741 1,300,417 8% Knowledge 928,190 359,778 1,287,967 8% Text-Only 1,875,174 190,787 2,065,961 12% Total 10,871,558 5,495,209 16,366,767 100%

Table 1: Data distribution across different topics in bi-mode annealing stage.

- 2.2 DATA FORMULATION AND TRAINING PROTOCOL

With the curated bi-mode data, we reorganize the annealing dataset into the following domains: General, Math/K12, Code, Chart, OCR, Grounding, Caption, Knowledge, and Text-Only, as detailed in Table 1. This structured categorization enhances the richness of the data, enabling R-4B-Base to generalize its thinking and non-thinking capabilities across diverse scenarios.

[Figure 21]

Annealing Data 16.3M

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

General

Text

Knowledge /Caption

Math/Code

Chart/OCR

Figure 4: Distributions of bi-mode data. Darker regions represent items with thinking mode, while lighter correspond to items without thinking.

During bi-mode annealing, the training instances are formatted according to their designated mode:

- • For queries benefiting from reasoning patterns, responses include the complete reasoning process, formatted as: <think>reasoning steps</think>answer
- • For queries suitable for direct answers, the response format maintains structural consistency but omits the thinking content, using the format: <think> </think>answer

Notably, consistent use of the <think> tags ensures structural uniformity. In addition, increasing the proportion of reasoning data strengthens both thinking and non-thinking capabilities in general-purpose applications. This lays the foundation for the subsequent policy optimization stage, designed to refine its auto-thinking judgment.

- 3 AUTO-THINKING INCENTIVIZATION VIA BI-MODE POLICY OPTIMIZATION

The bi-mode annealing stage successfully endows R-4B-Base with the dual capabilities of reasoning (thinking) and direct response (non-thinking). However, as illustrated in Figure 5, when operating in an auto-thinking inference setting in R-4B-Base, the model exhibits a performance degradation. This phenomenon, which we term "thinking atrophy", manifests as a tendency towards non-thinking mode, even for complex queries where reasoning is essential. This instruction-following failure indicates that while the model possesses the necessary skills (e.g., thinking and non-thinking capabilities), it lacks the judgment to deploy them appropriately. To bridge this gap, we transition from merely enabling these modes to actively incentivizing the optimal selection between them.

Fortunately, reinforcement learning (RL) (Guo et al., 2025a; Yu et al., 2025) presents a natural paradigm for this challenge, as it can optimize a policy based on outcome-driven rewards, thereby teaching the model to select the most effective thinking strategy. However, using vanilla RL directly will cause the model to develop thinking preferences during training, as mentioned in Section 5. Some methods (Lou et al., 2025; Zhang et al., 2025) aim to train auto-thinking models by proposing

Question: How many bars have value below 40?

[Figure 35]

[Figure 36]

[Figure 37]

(78, 57.0) Auto-thinking (RL)

[Figure 38]

###### Auto-thinking (RL)

###### Auto-thinking (Base)

<think> So, let's look at the bar graph. We need to count how many bars have a value below 40. First, let's list all the bars and their values:

[Figure 39]

The Answer is 5.

[Figure 40]

Auto-thinking (RL)

Auto-thinking (Base)

- - 2014 Banknotes: 63 (above 40)
- - 2014 Coins: 46 (above 40)

The bars with values <40 are 2016 Coins, 2017 Coins, and 2018 Coins.

... </think>

[Figure 41]

[Figure 42]

Final Answer: Thinking Process

- Figure 5: Comparison of R-4B-Base and R-4B-RL in auto-thinking mode on OpenCompass Multimodal Reasoning Benchmarks. The R-4B-RL achieves better auto-thinking inference performance.

new reward mechanisms in RL to modulate thinking preferences. Despite their effectiveness, these approaches encounter two principal limitations:

- 1. Complex reward engineering and data dependency: Existing methods often rely on intricate reward functions or manually annotated data that specifies query complexity to balance exploration between thinking and non-thinking modes. This dependency introduces subjectivity, is difficult to scale, and adds significant overhead to the training pipeline.
- 2. Hyperparameter sensitivity and training instability: The balance between encouraging thinking and non-thinking modes in refined reward-based RL methods is typically controlled by sensitive hyperparameters. Misconfiguration may easily lead to "mode collapse," where the policy converges to a suboptimal, single-mode strategy (either always thinking or never thinking), defeating the objective of auto-thinking.

To overcome these challenges, we introduce Bi-mode Policy Optimization (BPO), a novel yet elegant reinforcement learning framework designed to cultivate robust auto-thinking. Its core principle is to learn an adaptive policy by explicitly contrasting the utility of thinking versus non-thinking pathways for the same input query. Notably, BPO achieves a "a little goes a long way" effect. Instead of requiring complex, hand-crafted reward mechanisms or RL datasets for general domains, it leverages a simple, rule-based reward signal derived solely from the mathematical topic. We discovered that this highly specific reward mechanism possesses a remarkable universality, effectively promoting auto-thinking in diverse, non-mathematical topics as well.

- 3.1 THE DESIGN OF BI-MODE POLICY OPTIMIZATION

The BPO algorithm is designed to directly optimize the model’s decision-making process, training it to select the most appropriate mode on a per-query basis. As illustrated in Figure 6, BPO builds upon the Group Relative Policy Optimization (GRPO) framework, but introduces a critical modification: a bi-mode sampling strategy that forces the model to generate and contrast the outcomes from both thinking modes.

Specifically, to achieve this, we deterministically generate two distinct response groups for each input prompt: a set of thinking responses, denoted as {o1,...,og}, and a set of non-thinking responses, denoted as {o˜1,...,o˜g}. To ensure the generation of these distinct response types, we control the sampling process by conditioning on special tokens. For example, the special token sequence <thinking token> is appended to the input prompt to trigger thinking responses, whereas the special token sequence <non-thinking token> is used to prompt direct (nonthinking) responses. This design promotes balanced exploration across both modes by ensuring the two groups are of equal size, i.e., |Groupthinking| = |Groupnon-thinking| = g. Formally, we propose BPO, which optimizes the policy model πθ by maximizing the following objective function:

J BPO(θ) = Eq∼P(Q)

2g

- 1

- 2g

′ k)

k A(o

min R(o

k=1

′ k)

′ k)

′ k)

k − βDKL (πθ(·|q)∥πref(·|q)) , (1)

k ,1−ϵ,1+ϵ A(o

k , clip R(o

###### Bi-mode Policy Optimization (BPO)

[Figure 43]

Mixed Advantage

User: Please explain why this picture is so interesting? Assistant: <thinking token>

KL

[Figure 44]

[Figure 45]

Reference Model

. . .

. . .

Thinking Mode Sample

[Figure 46]

[Figure 47]

Bi-mode Rollouts

Policy Model

Reward Rules

Mixed-Group Computation

[Figure 48]

[Figure 49]

[Figure 50]

. . .

Non-thinking Mode Sample

. . .

Special Token

[Figure 51]

User: Please explain why this picture is so interesting? Assistant: <non-thinking token>

[Figure 52]

[Figure 53]

- 1.<thinking token>
- 2.<non-thinking token>

- Figure 6: The Bi-mode Policy Optimization (BPO) framework. For each input query, the policy model is conditioned to generate two distinct groups of responses (thinking and non-thinking).

ok if 0 < k ≤ g o˜k if k > g

represents the thinking and non-thinking rollouts respectively. ϵ

where o′k =

and β are hyperparameters controlling the clipping range and KL penalty strength, respectively. The policy ratio Rk for each token in the generated responses and the advantage values Ak are computed as in GRPO (Shao et al., 2024).

This deterministic grouping mechanism ensures balanced exploration between reasoning and direct answering behaviors. Through this simple yet effective reinforcement learning framework, the R4B-RL model achieves enhanced adaptive thinking capabilities in general-purpose domains, while simultaneously improving performance in both reasoning and non-reasoning generation.

- 4 EVALUATION

This section presents a comprehensive evaluation of R-4B, across a diverse set of multimodal benchmarks. Specifically, we first detail the evaluation protocols in Section 4.1, ensuring reproducible comparisons. Sections 4.2 and 4.3 explore the model’s general visual understanding and advanced reasoning skills, respectively. Furthermore, Section 4.4 compares auto-thinking token costs with other modes. We also provide visualization cases in the supplementary material.

- 4.1 EVALUATION SETTINGS

For all benchmark evaluations, we employ greedy decoding with the temperature set to 0 and a maximum generation length of 8,192 tokens for R-4B. To ensure fair and consistent comparisons with existing MLLMs, we adopt the widely-used VLM-EvalKit repository for evaluation. For tasks that require scoring-based evaluation, such as open-ended reasoning and complex visual understanding, we utilize Qwen3-32B as the judge model to streamline the process. As indicated in Table 2, we evaluate in three distinct modes: non-thinking (N-T) with extra token <think>\n\n</think>, thinking (T) with extra token <think>\n, and auto-thinking (A-T) with extra token <think>. R-4B-Base and R-4B-RL are evaluated in the thinking and auto-thinking modes, respectively.

- 4.2 GENERAL CAPABILITIES

General visual question answering. As shown in Table 2, R-4B demonstrates exceptional performance on a wide array of general visual question answering benchmarks. On MMMUval (Yue

- et al., 2023), a challenging multi-disciplinary benchmark, R-4B-RL achieves a state-of-the-art score of 68.1%, outperforming all other models, including Keye-VL-8B (66.8%). While Kimi-VL-A3BThinking leads on MMMU-Pro (Yue et al., 2024), our models remain highly competitive, with R4B-Base (46.7%) and R-4B-RL (46.5%) surpassing several larger counterparts. On MMStar (Chen
- et al., 2024a), R-4B-RL secures the second-best position with a score of 72.1%, closely following the top-performing Keye-VL-8B (72.8%). This strong performance extends to standard MMBench (Liu

et al., 2023a) evaluations. On both MMBenchV1.1-ENdev and MMBenchV1.1-CNdev, R-4BRL consistently ranks second with scores of 84.9% and 84.7%, respectively, reaffirming its robust cross-lingual understanding. Notably, on MMVet (Yu et al., 2023), our base model R-4B-Base achieves a remarkable score of 85.9%, significantly surpassing all competitors. Furthermore, R-

Qwen2.5-VL InternVL3 InternVL3.5 Kimi-VL-A3B Keye-VL

R-4B-Base R-4B-RL

Capability Benchmark

-7B -8B -4B -Thinking -8B

(N-T) (N-T) (T) (T) (A-T) (T) (A-T)

MMMUval 58.6 62.7 66.6 64.0 66.8 63.2 68.1 MMMU-Pro 34.7 45.6 - 49.2 47.5 46.7 46.5 MMStar 64.1 68.7 65.0 70.4 72.8 70.8 73.1 MMBenchV1.1-ENdev 82.1 84.7 - 82.6 89.7 81.9 84.9 MMBenchV1.1-CNdev 81.3 83.6 - 80.7 89.8 83.2 84.7 MMVet 69.7 82.8 76.6 81.9 65.5 85.9 81.9 HallusionBench 55.7 49.4 44.8 57.2 57.3 53.9 58.9 VLMs are Blind 37.4 36.8 - 60.8 61.0 47.0 52.3 MMVP 73.3 79.3 - 80.3 79.0 79.3 80.7 VisuLogic 20.0 26.1 - 25.0 21.1 22.5 25.1 RealWorldQA 68.2 70.6 66.3 66.1 66.3 70.5 69.1

General Visual QA

AI2D 83.9 85.2 83.9 82.7 85.8 84.8 86.2 CharXiv (DQ) 73.9 73.6 71.1 75.4 74.5 82.8 82.9 CharXiv (RQ) 42.5 37.6 39.6 47.7 40.0 55.4 56.8 DocVQAval 95.5 89.4 92.4 69.0 86.3 89.6 91.0

Table & Chart & OCR

OCRBench 89.7 88.0 81.5 86.2 85.3 82.8 83.6 BLINKval 56.4 55.5 58.1 56.2 52.5 54.8 56.3 CountBench 74.1 80.0 - 91.4 75.4 92.6 90.2

Visual Perception & Counting

MathVision 26.2 28.8 26.2 56.8 42.4 45.7 47.8 MathVistaMINI 66.8 70.7 77.1 80.1 75.2 76.8 78.0 MathVerse-vision 41.2 32.4 61.7 57.4 40.8 65.0 64.9 OlympiadBench 19.4 25.9 - 33.9 45.2 47.0 49.6 WeMath 37.7 38.5 50.1 47.0 58.6 54.1 52.8 LogicVista 44.5 43.6 56.4 51.0 50.6 58.8 59.1 DynaMath 20.1 23.9 35.7 27.1 35.3 36.3 39.5

Math & Reasoning

- Table 2: Performance comparison of multimodal large language models across diverse benchmarks. The best and second-best results are highlighted in bold and underlined, respectively. We denote thinking, non-thinking, and auto-thinking modes as T, N-T, and A-T. Notably, Keye-VL-8B and R-4B-RL are evaluated in A-T mode.

4B is good at tasks with visual illusions and logical fallacies. It achieves the top score of 58.9 on HallusionBench (Guan et al., 2023) and 80.7% on MMVP (Tong et al., 2024), setting a new standard on both benchmarks. On the logic-intensive VisuLogic (Xu et al., 2025) benchmark, R-4B-RL attains a competitive second-place score of 25.1%. Similarly, on RealWorldQA, R-4BBase is the second-best performer with 70.5%. These results collectively underscore R-4B’s superior general visual understanding and reasoning capabilities.

Document and chart understanding. R-4B demonstrates outstanding proficiency in interpreting structured visual content like documents, diagrams, and charts. On AI2D (Kembhavi et al., 2016), a diagram understanding benchmark, R-4B-RL achieves the highest score of 86.2%, showcasing its advanced spatial and semantic comprehension. The model’s dominance is particularly evident in chart analysis. On CharXiv (Wang et al., 2024b), R-4B-RL not only leads in Descriptive Questions (DQ) with a score of 82.9% but also establishes a significant lead in Reasoning Questions (RQ) with a score of 56.8%. This is a substantial improvement of over 9 percentage points compared to the Kimi-VL-A3B-Thinking (47.7%), highlighting its superior ability to reason over complex chart semantics. On DocVQAval (Mathew et al., 2021), R-4B-RL obtains a strong score of 91.0%. This performance confirms R-4B’s exceptionally robust capabilities in both precise text extraction and document structure interpretation.

Visual perception and counting. In tasks requiring fine-grained visual perception, R-4B delivers highly competitive results. On BLINKval (Fu et al., 2024), a benchmark for referring expression comprehension, R-4B-RL scores 56.3%, effectively tying for the top position with Qwen2.5-VL (56.4%) and outperforming other models. The most impressive result in this category comes from CountBench (Paiss et al., 2023), where our base model, R-4B-Base, sets the highest score with 92.6%, demonstrating exceptional object counting abilities. R-4B-RL also performs strongly with a score of 90.2%. On OCRBench (Liu et al., 2023b), R-4B-RL achieves a robust score of 83.6%, proving its solid text recognition capabilities in diverse scenes.

[Figure 54]

AverageOutputTokens/Query

Non-thinking Mode 1382

1278

|57 66<br><br>394<br><br>Auto-thinking Thinking Mod|82<br><br>331<br><br>596<br><br>Mode e|78<br><br>327<br><br>532|100<br><br>466 520|270<br><br>996<br><br>1136|408|
|---|---|---|---|---|---|
|OCRBench|MMVet|HallusionBench|AI2D|MathVista|WeMath|

- Figure 7: Comparison of average output tokens per query across non-thinking, auto-thinking, and thinking modes on different benchmarks. The auto-thinking mode achieves a trade-off between efficiency and performance.

- 4.3 COMPLEX REASONING CAPABILITIES

R-4B exhibits exceptional strength in complex mathematical and logical reasoning tasks. Notably, our models secure the top two positions on several challenging math-focused evaluations. On MathVerse-vision (Zhang et al., 2024), R-4B-Base (65.0%) and R-4B-RL (64.9%) dramatically outperform all prior models, with the next-best competitor lagging at 57.4%. A similar dominance is observed on OlympiadBench (He et al., 2024), where R-4B-RL (49.6%) and R-4B-Base (47.0%) take the lead. The trend continues on LogicVista (Xiao et al., 2024) (R-4B-RL: 59.1%, R-4B-Base: 58.8%) and DynaMath (Zou et al., 2024) (R-4B-RL: 39.5%, R-4B-Base: 36.3%), where our models again claim the top two spots, underscoring their advanced logical deduction skills. On other prominent math benchmarks, R-4B remains a top contender. It achieves the secondbest scores on MathVision (Wang et al., 2024a) (47.8%) and MathVistaMINI (Lu et al., 2024) (78.0%), trailing only Kimi-VL-A3B-Thinking but significantly surpassing other baselines. These results demonstrate that our advanced training strategies empower R-4B to compete with larger MLLMs and R-4B-RL sets a new standard for 4B MLLMs.

- 4.4 TOKEN CONSUMPTION ACROSS DIFFERENT MODES

As illustrated in Figure 7, we analyze the average output tokens per query of R-4B-RL, to verify the effectiveness of our BPO method. This analysis provides empirical evidence for the generalization capability of the auto-thinking learned via BPO. Although trained with a reward signal exclusively from the mathematical domain, R-4B-RL demonstrates an understanding of task complexity across a diverse set of general-purpose benchmarks. For simpler tasks such as those in OCRBench, the auto-thinking mode generates only 66 tokens, a volume comparable to the non-thinking mode (57 tokens) and significantly lower than the full thinking mode (394 tokens). Besides, its performance of 83.6% matches the non-thinking mode (83.6%) and even surpasses the thinking mode (82.6%). This indicates that for straightforward queries, our model conserves computational resources while maintaining performance. Meanwhile, when confronted with reasoning-intensive benchmarks like MathVista and WeMath, the auto-thinking mode dynamically increases its token output to 996 and 1278, respectively. These figures closely approach the token counts of the thinking mode (1136 and 1382 tokens). At the same time, this enables our model to achieve performance of 78.0% on MathVista and 52.8% on WeMath, which are substantially higher than the non-thinking mode’s scores (71.5% and 46.6%) and competitive with the thinking mode (79.7% and 55.8%). In conclusion, these findings validate that our BPO is not merely a thinking switch but a truly intelligent and generalizable policy. It effectively discerns task complexity across varied domains, striking an optimal trade-off between performance and efficiency.

- 5 ANALYSIS

Training data ablation in annealing stage. We performed an ablation study to determine the optimal bi-mode annealing strategy, by focusing on different data compositions. As detailed in Table 3, results demonstrate that a mixed-data approach combined with a thinking mode is markedly superior. This strategy achieves an average performance of 69.5%, surpassing models trained on only reasoning data (+4.1%) and a two-stage curriculum (+2.6%). In addition, while the Only-R

###### Strategy Data Mode MMMU MMStar AI2D OCRBench MathVista MathVision LogicVista Average

Non-R 16.3M N-T 60.4 70.4 83.2 83.3 71.1 33.2 49.0 64.4 Only-R 5.5M T 62.8 65.7 78.9 82.1 73.6 41.9 52.6 65.4 Non-R → R 10.8M → 5.5M T 63.8 68.7 80.4 82.7 74.9 43.7 54.4 66.9 Mixed-R 16.3M N-T 56.3 67.5 84.5 82.7 65.5 29.9 47.7 62.0 Mixed-R 16.3M T 64.6 73.1 84.5 82.8 76.8 45.7 58.8 69.5

*Note: Non-R → Reasoning strategy resulting model possesses reasoning capabilities only.

- Table 3: Ablation study on training strategies during the annealing stage. We compare four data strategies: using only non-reasoning data (Non-R), only reasoning data (Only-R), a two-stage curriculum (Non-R → R), and a mix of both reasoning data and non-reasoning data (Mixed-R). Performance is evaluated in either non-thinking (N-T) or thinking (T) inference mode.

[Figure 55]

[Figure 56]

75.0 75.7 76.4 76.4 76.7 76.4 76.6

84.4

AverageThinkingTriggeringRate(%)

| |[Figure 57]<br><br>5.9<br><br>19.4<br><br>32.8<br><br>39.7<br><br>42.9 43.1 43.8 45.7<br><br>17.8<br><br>54.1<br><br>71.6<br><br>78.7<br><br>81.5 81.8 81.7<br><br>[Figure 58]<br><br>Non-Reasoning Benchmarks Reasoning Benchmarks<br><br>[Figure 59]| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

73.9

75

80

AverageAccuracy(%)

70

60

65

59.4

56.6 57.5 58.2 58.7 57.9

60

40

54.0

55

[Figure 60]

20

49.1

Non-Reasoning Benchmarks Reasoning Benchmarks

50

45

0

20 40 60 80 100 120 140 160 180 200 220 240 260 280 300 320

20 40 60 80 100 120 140 160 180 200 220 240 260 280 300 320

Training Steps

Training Steps

- Figure 8: Average thinking triggering rate (%) across training steps for non-reasoning benchmarks, and reasoning benchmarks.

Figure 9: Average accuracy (%) across RL training steps for non-reasoning benchmarks, and reasoning benchmarks.

strategy enhances performance on specific complex tasks (e.g., +8.7% on MathVision), it fails to generalize tasks, leading to a lower overall score. The Mixed-R strategy, however, successfully balances specialized reasoning with general capabilities. This suggests that co-training on both data types prevents catastrophic forgetting of general skills while effectively instilling complex reasoning abilities.

Thinking triggering rate during reinforcement learning. To dissect the learning process of BPO, we tracked the evolution of the thinking triggering rate across benchmarks with distinct complexity profiles. As illustrated in Figure 8, on reasoning-intensive benchmarks (e.g., MathVision, MathVista), the model rapidly learns to activate its thinking mode, with the trigger rate showing a steep initial climb before stabilizing at a high level. Conversely, on non-reasoning benchmarks (e.g., OCRBench, HallusionBench), the trigger rate shows only a marginal and slow increase. This differential learning processing is the direct consequence of our BPO mechanism. The model quickly discovers that applying the thinking mode to reasoning tasks yields a high reward signal, thus reinforcing this behavior. Simultaneously, it learns that ‘thinking’ on simpler, non-reasoning tasks offers few rewards. The benefit of our RL policy is reflected in the performance gains. As shown in Figure 9, BPO achieves a substantial +10.3% improvement on reasoning tasks, a gain that far surpasses its impact on non-reasoning tasks. This directly validates BPO’s effectiveness in mitigating "thinking atrophy" by incentivizing the appropriate auto-thinking.

Comparison between R-4B-Base and R-4B-RL. Table 4 presents a comprehensive comparison between R-4B-Base and R-4B-RL across multiple reasoning benchmarks under non-thinking, thinking, and auto-thinking modes. The results demonstrate that RL consistently enhances performance in both non-thinking and thinking capabilities. In non-thinking mode, where models generate answers directly without explicit reasoning, R-4B-RL achieves a substantial improvement over R-4B-Base, raising the average score from 42.0% to 49.9%, outperforming the base model on all benchmarks. This significant gain underscores the effectiveness of RL for direct response generation, thereby strengthening non-reasoning capabilities. Besides, in thinking mode, R-4B-RL retains its edge, pushing the average from 56.1% to 58.1%, showing greater benefits from reasoning. Overall, R4B-RL consistently outperforms R-4B-Base in both modes, confirming our RL method boosts both thinking and non-thinking performance.

##### Model Mode MathVista MathVision MathVerse DynaMath WeMath LogicVista Average

R-4B-Base N-T 65.5 29.9 48.4 28.5 32.0 47.7 42.0 R-4B-RL N-T 71.5 35.0 61.5 31.7 46.6 52.8 49.9 R-4B-Base A-T 65.7 30.8 48.9 30.0 34.3 49.4 43.2 R-4B-RL A-T 78.0 47.8 64.9 39.5 52.8 59.1 57.0 R-4B-Base T 76.8 45.7 65.0 36.3 54.2 58.8 56.1 R-4B-RL T 79.7 47.9 67.1 39.5 55.8 58.6 58.1

- Table 4: Performance comparison of R-4B-Base and R-4B-RL models across reasoning benchmarks.

Thinking preference dilemma of vanilla GRPO. As shown in Figure 10, we compare the average items per rollout for vanilla GRPO, considering both thinking and nonthinking scenarios, across the reinforcement learning training steps. For GRPO, the nonthinking items increase initially but then drop and remain at a low average correct items per rollout, which falls into the thinking preference dilemma. Overall, BPO demonstrates better performance in both thinking and non-thinking modes across the training steps, indicating that BPO is more effective in handling adaptive reasoning in the RL training stage compared to GRPO and avoid the preference dilemma.

16

14

###### AverageItems/Rollout

12

10

vanilla-GRPO Non-thinking Mode Items

8

vanilla-GRPO Thinking Mode Items

6

4

2

40 80 120 160 200 240 280

Training Steps

Figure 10: The average thinking and non-thinking items across training steps on vanilla GRPO.

- 6 RELATED WORK

Recently, existing methods to enhance the efficiency of LLMs have centered on manually enabling the thinking mode in model responses (Yang et al., 2025a). To automate this process, some methods incorporate RL with extra thinking preference rewards to encourage models to think on their own (Tu et al., 2025; Zhang et al., 2025). Other approaches further fine-tune models on datasets containing both thinking and non-thinking responses, which are obtained from prompt engineering and existing reasoning models (Lou et al., 2025; Zhan et al., 2025). In the multimodal domain, Keye-VL (Yang et al., 2025b) addresses auto-thinking using Mix-Mode RL. Inspired by the above methods, by introducing bi-mode annealing and BPO strategy, R-4B enables efficient, content-dependent autothinking by a single model, filling a crucial gap towards efficient and auto-thinking MLLMs.

- 7 CONCLUSION

In this work, we introduce R-4B, a multimodal large language model designed to address the critical trade-off between complex reasoning and inference efficiency. R-4B features an "auto-thinking" mechanism that dynamically switches between thinking and direct answering modes. Our method first equips R-4B-Base with thinking and direct answering capabilities through a bi-mode annealing stage. Subsequently, we employ bi-mode policy optimization, a reinforcement learning approach that utilizes hybrid mixed-policy rollouts to mitigate "thinking collapse" and learn an optimal policy for mode selection in general domains. R-4B-RL achieves state-of-the-art performance on reasoning-intensive benchmarks, outperforming comparable models and matching the larger KimiVL-A3B-Thinking-2506. This research demonstrates a practical and effective pathway toward developing more intelligent and resource-efficient MLLMs.

REFERENCES

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv:2403.20330, 2024a.

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024b.

Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. In European Conference on Computer Vision, pp. 148–166. Springer, 2024.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusionbench: An advanced diagnostic suite for entangled language hallucination & visual illusion in large visionlanguage models. arXiv:2310.14566, 2023.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025a.

Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025b.

Meng-Hao Guo, Xuanyu Chu, Qianrui Yang, Zhe-Han Mo, Yiqing Shen, Pei-lin Li, Xinjie Lin, Jinnian Zhang, Xin-Sheng Chen, Yi Zhang, et al. Rbench-v: A primary assessment for visual reasoning models with multi-modal outputs. arXiv preprint arXiv:2505.16770, 2025c.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008, 2024.

Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In European conference on computer vision, pp. 235–251. Springer, 2016.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 26296–26306, 2024.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023a.

Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xucheng Yin, Cheng lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: On the hidden mystery of ocr in large multimodal models. arXiv:2305.07895, 2023b.

Zuyan Liu, Yuhao Dong, Jiahui Wang, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Ola: Pushing the frontiers of omni-modal language model. arXiv preprint arXiv:2502.04328, 2025.

Chenwei Lou, Zewei Sun, Xinnian Liang, Meng Qu, Wei Shen, Wenqi Wang, Yuntao Li, Qingping Yang, and Shuangzhi Wu. Adacot: Pareto-optimal adaptive chain-of-thought triggering via reinforcement learning. arXiv preprint arXiv:2505.11896, 2025.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, KaiWei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In ICLR, 2024.

Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pp. 2200–2209, 2021.

Roni Paiss, Ariel Ephrat, Omer Tov, Shiran Zada, Inbar Mosseri, Michal Irani, and Tali Dekel. Teaching clip to count to ten. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 3170–3180, 2023.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9568–9578, 2024.

Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

Songjun Tu, Jiahao Lin, Qichao Zhang, Xiangyu Tian, Linjing Li, Xiangyuan Lan, and Dongbin Zhao. Learning when to think: Shaping adaptive reasoning in r1-style models via multi-stage rl. arXiv preprint arXiv:2505.10832, 2025.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. arXiv:2402.14804, 2024a.

Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, et al. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. Advances in Neural Information Processing Systems, 37:113569–113697, 2024b.

Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts. arXiv preprint arXiv:2407.04973, 2024.

Weiye Xu, Jiahao Wang, Weiyun Wang, Zhe Chen, Wengang Zhou, Aijun Yang, Lewei Lu, Houqiang Li, Xiaohua Wang, Xizhou Zhu, et al. Visulogic: A benchmark for evaluating visual reasoning in multi-modal large language models. arXiv preprint arXiv:2504.15279, 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Biao Yang, Bin Wen, Changyi Liu, Chenglong Chu, Chengru Song, Chongling Rao, Chuan Yi, Da Li, Dunju Zang, et al. Kwai keye-vl technical report. arXiv preprint arXiv:2507.01949, 2025b.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. arXiv:2311.16502, 2023.

Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Ming Yin, Botao Yu, Ge Zhang, et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024.

Zihao Yue, Zhenru Lin, Yifan Song, Weikun Wang, Shuhuai Ren, Shuhao Gu, Shicheng Li, Peidian Li, Liang Zhao, Lei Li, Kainan Bao, Hao Tian, Hailin Zhang, Gang Wang, Dawei Zhu, Cici, Chenhong He, Bowen Ye, Bowen Shen, Zihan Zhang, Zihan Jiang, Zhixian Zheng, Zhichao Song, Zhenbo Luo, Yue Yu, Yudong Wang, Yuanyuan Tian, Yu Tu, Yihan Yan, Yi Huang, Xu Wang, Xinzhe Xu, Xingchen Song, Xing Zhang, Xing Yong, Xin Zhang, Xiangwei Deng, Wenyu Yang, Wenhan Ma, Weiwei Lv, Weiji Zhuang, Wei Liu, Sirui Deng, Shuo Liu, Shimao Chen, Shihua Yu, Shaohui Liu, Shande Wang, Rui Ma, Qiantong Wang, Peng Wang, Nuo Chen, Menghang Zhu, Kangyang Zhou, Kang Zhou, Kai Fang, Jun Shi, Jinhao Dong, Jiebao Xiao, Jiaming Xu, Huaqiu Liu, Hongshen Xu, Heng Qu, Haochen Zhao, Hanglong Lv, Guoan Wang, Duo Zhang, Dong Zhang, Di Zhang, Chong Ma, Chang Liu, Can Cai, and Bingquan Xia. Mimo-vl technical report, 2025. URL https://arxiv.org/abs/2506.03569.

Zizheng Zhan, Ken Deng, Huaixi Tang, Wen Xiang, Kun Wu, Weihao Li, Wenqiang Zhu, Jingxuan Xu, Lecheng Huang, Zongxian Feng, et al. Kat-v1: Kwai-autothink technical report. arXiv preprint arXiv:2507.08297, 2025.

Jiajie Zhang, Nianyi Lin, Lei Hou, Ling Feng, and Juanzi Li. Adaptthink: Reasoning models can learn when to think. arXiv preprint arXiv:2505.13417, 2025.

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pp. 169–186. Springer, 2024.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

Chengke Zou, Xingang Guo, Rui Yang, Junyu Zhang, Bin Hu, and Huan Zhang. Dynamath: A dynamic visual benchmark for evaluating mathematical reasoning robustness of vision language models. arXiv preprint arXiv:2411.00836, 2024.

APPENDIX

- A CONTRIBUTIONS

Project Lead Jie Jiang1

Core Contributors Qi Yang1,2, Bolin Ni1

Supervisors Shiming Xiang2, Han Hu1, Houwen Peng1

- B PRE-TRAINING STAGE

- B.1 R-4B MODEL ARCHITECTURE

Similar to other MLLMs (Li et al., 2024; Zhu et al., 2025; Bai et al., 2025), R-4B consists of three core components for foundational multimodal understanding. The visual encoder is initialized with SigLIP2-So400m (Tschannen et al., 2025) enhanced with the AnyRes strategy (Liu et al., 2024), enabling it to flexibly process images of any resolution. For language comprehension and generation, we utilize the Qwen3-4B Large Language Model (LLM) (Yang et al., 2025a), chosen for its strong reasoning capabilities and computational efficiency. To bridge these two modalities, a randomly initialized Multi-Layer Perceptron (MLP) projector maps the visual features into the LLM’s latent space, ensuring a coherent multimodal interaction. Furthermore, to establish robust non-thinking multimodal understanding capabilities, we employ a similar three-stage pre-training strategy (Li et al., 2024; Chen et al., 2024b) with 1. MLP warmup, 2. vision-language alignment, and 3. joint multimodal pre-training, producing a base vision-language model with comprehensive non-thinking multimodal understanding. More details are provided in supplementary material.

- B.2 PRE-TRAINING SETTINGS Our framework implements a three-stage pre-training paradigm, summarized in Table I.

Stages Stage 1 Stage 2 Stage 3 Stage 1 Purpose

MLP Vision-Language Joint Multimodal Bi-Mode

Warmup Alignment Pre-training Annealing

Batch Size 512 2048 1024 1024 Scheduler Type Cosine Cosine Constant Cosine MLP Learning Rate 1e-3 4e-5, 4e-6 4e-6 4e-6 ViT Learning Rate - 4e-5, 4e-6 4e-6 4e-6 LLM Learning Rate - - 4e-5 4e-5,4e-6 Packed Sequence Length 8192 16384 16384 16384 Trainable Components MLP ViT, MLP ViT, MLP, LLM ViT, MLP, LLM

Table I: Overview of R-4B-Base pre-training and annealing stages, including MLP warmup, visionlanguage alignment, joint multimodal pre-training, and bi-mode annealing.

- Stage 1: MLP warmup We begin by freezing the parameters of both the ViT and the LLM, while initializing a randomly-initialized two-layer MLP projection module. This projector is trained using image-caption pairs to establish initial cross-modal alignment. This stage enables stable gradient propagation in subsequent stages and mitigates instability caused by poorly aligned representations.
- Stage 2: Vision-language alignment In this stage, the ViT backbone is unfrozen while the LLM remains frozen, and training proceeds using interleaved multimodal data. The inclusion of diverse visual content in these batches systematically improves the visual encoder’s ability to handle different visual domains.

- Stage 3: Joint multimodal pre-training This stage enables full-parameter optimization across the entire architecture. We expand the training regimen to incorporate 145 billion tokens spanning diverse modalities and tasks, including OCR interpretation, visual grounding, mathematical reasoning, and structured data (tables/charts). Additionally, we implement a non-thinking loss masking strategy. In this strategy, <think> </think> tags are appended before response generation, and their corresponding loss contributions are masked. This strategy effectively preserves Qwen3’s (Yang

- et al., 2025a) specialized reasoning capabilities during joint multimodal pre-training.

- B.3 NON-REASONING DATA DISTRIBUTION FOR PRE-TRAINING STAGE

To enhance R-4B-Base’s capabilities, we employed diverse categories of data across distinct training stages. The pre-training stages primarily utilized non-reasoning data to improve multimodal understanding and visual perceptual abilities.

- Stage 1: We trained the model using 808K image captioning samples sourced from the LAION datasets (Liu et al., 2025). The primary objective was to establish vision-language connections.
- Stage 2: This stage focuses on refining the visual module and enhancing image understanding. We utilized a large corpus (25 million items) comprising Visual OCR (31.8%), Knowledge (33.7%), Captioning (26.3%), and Math (8.2%) data.
- Stage 3: To boost overall multimodal understanding, we introduced 13.3% text data alongside higher proportions of quality Math and K-12 data (22.9%). This stage included approximately 37 million items.

- C CASE STUDY

### Case: Object Counting

[Figure 61]

User Question: How many tomatoes are there?

Thinking: (Non-thinking)

### Model Answer:

There are five tomatoes visible in the image. They are arranged in a cluster, with four of them lying flat and one slightly elevated. The tomatoes appear to be fresh and ripe, with a vibrant red color. They are positioned in the foreground of the image, likely on a wooden surface, and are surrounded by other fruits and jars of preserved produce.

Figure I: An example demonstrating R-4B-RL solves object counting problem.

### Case: Image Understanding

[Figure 62]

User Question: What is funny about this image?

Thinking: (Non-thinking)

### Model Answer:

The humor in this image comes from the rhinoceros painting pictures that all prominently feature its horn, reflecting the caption "You see the world as you are!" The rhino’s unique perspective in art mirrors our personal views, suggesting that our individual experiences shape how we see the world.

Figure II: An example demonstrating R-4B-RL solves image understanding problem.

### Case: Optical Character Recognition

[Figure 63]

User Question: Who is the author of this book?

Thinking: (Non-thinking)

### Model Answer:

The author of this book is Steven N. Meyers. This information is clearly stated on the cover of the book, which features the title "flower spirits" and the subtitle "Radiographs of Nature by Steven N. Meyers." The book appears to be a 16-month weekly engagement planner covering the period from September 2014 to December 2015.

Figure III: An example demonstrating R-4B-RL solves OCR tasks.

Case: Sphere Path Counting

[Figure 64]

User Question:

There are three great circles on a sphere that intersect each other in a right angle. Starting in point S a little bug moves along the great circles in the direction indicated. At crossings it turns alternately to the right or left. How many quarter circles does it crawl along until it is back in point S?

### Thinking (Part):

Got it, let’s try to figure out this problem step by step. First, let’s recall what a great circle is: a circle on a sphere whose center is the same as the sphere’s center, so it’s the largest possible circle on the sphere. The problem says there are three great circles intersecting at right angles. So, imagine a sphere with three great circles, like the x, y, z axes in a coordinate system, each pair intersecting at 90 degrees. Now, the bug starts at point S and moves along these great circles, turning right or left at each intersection. We need to find how many quarter circles it crawls before returning to S. Let’s visualize the path. Let’s ...

### Model Answer:

To solve the problem, we analyze the bug’s movement along the great circles and the geometric properties of the sphere:

- Step 1: Understand Great Circles and Intersections Three great circles intersecting at right angles form a coordinate - like system (analogous to the x, y, and z axes of a 3D coordinate system). Each great circle is the largest circle on the sphere, and their intersections create a closed - loop path for the bug.
- Step 2: Track the Bug’s Movement The bug starts at point S and moves along great circles, turning right or left at each intersection. Each segment of the bug’s path is a **quarter circle** (since the great circles intersect at right angles, and the bug’s turns alternate, each segment corresponds to a 90° arc, or a quarter circle). To return to S, the bug’s path must form a closed loop. By visualizing the path (and using the diagram’s structure, where the bug’s path forms a closed loop with 6 quarter - circle segments), we count the number of quarter circles in this loop.
- Step 3: Count the Quarter Circles The bug’s path consists of 6 quarter - circle segments. Each segment is a 90◦ arc, and the total rotation after 6 segments is 6 × 90◦ = 540◦, but this is not necessary for the count—we only need the number of quarter circles in the closed loop.

|6|
|---|

.

Thus, the number of quarter circles the bug crawls along until it returns to S is

Figure IV: An example demonstrating R-4B-RL solves complex geometry problem.

### Case: Abstract Reasoning

[Figure 65]

User Question: What diagram replaces the question mark?

### Thinking (Part):

Got it, let’s try to figure out this problem step by step. First, we need to identify which operation the green circle (the "m" symbol) represents. Looking at the top part of the image, there are four examples:

- 1. First row left: "Each symbol moves one place to the right", so that’s a shift right.
- 2. First row right: "Changes shading on 1st and 3rd symbol", so shading the first and third.
- 3. Second row left: "Reflects all symbols in X axis", reflection over X-axis.
- 4. Second row right: "Replaces all squares with stars", replace squares with stars. Now, the green circle in the bottom part is the "m" symbol, which we need to match to one of these operations. Let’s check each example: The first example (top left) has the "m" symbol, and the operation is ...

### Model Answer:

To solve this problem, we first identify the operation associated with the green “m” symbol by analyzing the top - row examples:

### 1. Identify the operation for the “m” symbol:

- - The first example in the top - left shows the “m” symbol, and the description is “Each symbol moves one place to the right”.
- - Let’s verify with the input [L,+,|]: After the operation, it becomes [|,L,+]. This means each symbol shifts one position to the right (the rightmost symbol moves to the leftmost position). 2. Apply the operation to the input diagram [□,□,⋆]:
- - For the input [□,□,⋆], we perform the right - shift: the rightmost symbol (⋆) moves to the leftmost position, the first symbol (□) moves to the second position, and the second symbol (□) moves to the third position.
- - After the shift, the new order is [⋆,□,□]. 3. Match with the options:
- - Option A is [⋆,□,□], which matches our result.
- - Options B, C, and D do not match the right - shift operation (B involves shading, C involves replacing stars, and D has an incorrect star position).

|A|
|---|

.

Thus, the diagram that replaces the question mark is

Figure V: An example demonstrating R-4B-RL solves logic reasoning tasks.

Case: Pie Chart Description

[Figure 66]

User Question:

The pie charts below show units of electricity production by fuel source in Australia and France in 1980 and 2000. Summarise the information by selecting and reporting the main features, and make comparisons where relevant. You should write at least 150 words.

### Thinking (Part):

Got it, let’s tackle this task. First, I need to analyze the pie charts for Australia and France in 1980 and 2000. Let’s start by identifying the fuel sources: Natural Gas, Oil, Coal, Nuclear Power, Hydro Power. First, Australia’s 1980: Total 100 units. Coal is 50 (half), ...

### Model Answer:

The pie charts depict the units of electricity produced by different fuel sources in Australia and France during 1980 and 2000. Overall, coal was the primary fuel in Australia, while nuclear power dominated electricity generation in France. Over the two decades, coal and nuclear power saw significant increases, whereas natural gas and oil production declined, and hydro power remained a minor contributor in both countries. In Australia, coal was the largest source of electricity in 1980, accounting for 50 units (50% of total production). Hydro power and natural gas each contributed 20 units (20% each), while oil provided 10 units (10%). By 2000, coal’s share surged to 130 units (76% of the total 170 units), reflecting a substantial increase. Hydro power also rose to 36 units (21%), but natural gas and oil production plummeted to just 2 units each (1% of the total). In France, the fuel mix in 1980 was more balanced, with natural gas, oil, and coal each contributing 25 units (28% each), nuclear power at 15 units (17%), and hydro power at 5 units (5%). By 2000, nuclear power became the dominant source, supplying 126 units (70% of the total 180 units). In contrast, natural gas and oil production dropped to 2 units each (1%), while coal and hydro power remained relatively low at 25 units (14%) and 2 units (1%) respectively. In summary, Australia’s electricity generation became increasingly coal - reliant, while France shifted towards nuclear power. Both countries experienced a decline in natural gas and oil usage, with hydro power playing a negligible role throughout the period.

Figure VI: An example demonstrating R-4B-RL solves pie chart description problem.

