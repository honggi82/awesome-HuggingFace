## arXiv:2506.02397v3[cs.AI]6Jan2026

### OThink-R1: Intrinsic Fast/Slow Thinking Mode Switching for Over-Reasoning Mitigation

#### Shengjia Zhang1✌, Junjie Wu2✌, Jiawei Chen1✉, Changwang Zhang2✉, Zhe Li1, Xingyu Lou2, Wangchunshu Zhou2, Sheng Zhou1, Can Wang1, Jun Wang2✉ 1 Zhejiang University, 2 OPPO Research Institute

shengjia.zhang@zju.edu.cn, wujunjie1@oppo.com, sleepyhunt@zju.edu.cn, changwangzhang@foxmail.com zheli03@zju.edu.cn, louxingyu@oppo.com, zhouwangchunshu@oppo.com, zhousheng_zju@zju.edu.cn wcan@zju.edu.cn, junwang.lu@gmail.com

#### Abstract

Human cognition operates through two complementary modes: fast intuitive thinking and slow deliberate thinking. Vanilla large language models (LLMs) predominantly follow the fast-thinking paradigm, producing immediate responses; while recent large reasoning models (LRMs) adopt slow-thinking strategies, generating detailed reasoning chains before arriving at answers. While LRMs often achieve higher accuracy, this comes at the cost of substantially increased token usage. To address this efficiency–accuracy trade-off, we propose OThink-R1, a hybrid reasoning framework that integrates both modes within a single LRM and enables automatic mode switching based on problem characteristics. We first identifies three major patterns of essential and redundant reasoning trajectories in LRMs, which guide the design of an auxiliary LLM-based judge that adaptively determines when slow thinking is necessary. Leveraging the judge’s decisions, we construct a a hybrid fine-tuning dataset by pruning redundant reasoning to produce fast-thinking samples and retaining complete reasoning for slow-thinking samples. This dataset is then used to fine-tune LRMs, equipping them with inherent autonomous mode-selection capabilities. Extensive experiments on mathematical and question-answering benchmarks show that OThink-R1 reduces reasoning token usage significantly while maintaining competitive accuracy. The code is available at https: //github.com/AgenticIR-Lab/OThink-R1.

#### 1 Introduction

Human cognitive processes are often divided into two complementary modes (Kahneman, 2011; Li et al., 2025; Wason and Evans, 1974; Evans, 2008;

✌Equal contribution. ✉Corresponding author.

Stanovich and West, 2000): 1) Fast intuitive thinking (System 1), which is is instinctive and effortless, allowing for quick decisions but often leading to cognitive biases in complex situations; 2) Slow deliberate thinking (System 2), which is thoughtful and analytical, employing logical reasoning to achieve more accurate problem-solving. Traditional large language models (LLMs) primarily mimic fast thinking, relying on heuristics from vast data patterns, which can limit their effectiveness in handling intricate tasks. Recent advancements in language reasoning models (LRMs), such as OpenAI o1 (Jaech et al., 2024) and DeepSeek-R1 (Guo et al., 2025), emulate slow thinking by generating explicit reasoning chains (CoT) (Wei et al., 2022), transforming raw predictions into more structured, step-by-step solutions. This approach helps to mitigate the biases of fast thinking, significantly enhancing their reasoning capabilities for tackling complex problems.

Despite the impressive performance on complex tasks, LRMs have been associated with considerably higher computational overhead than LLMs. For instance, across commonly solved questions by both model types, typical LRMs generate on average 7.32 times more tokens than non-reasoning LLMs (Table 1). While several studies have explored token compression techniques to control the length of reasoning for LRMs (Su et al., 2025a; Hou et al., 2025)2, these approaches still uniformly employ a slow-thinking paradigm across all tasks, irrespective of variations in problem difficulty. As a result, relatively straightforward tasks still require excessive reasoning, while complex tasks are constrained by token limits that undermine the completeness of reasoning. This inflexibility introduces an inherent trade-off between computational effi-

2While DualFormer (Su et al., 2025a) claims to support adaptively switching between fast and slow thinking, this capability was demonstrated exclusively on maze navigation and Sokoban game tasks rather than LLM reasoning tasks.

###### 1 Thinking Paradigm Identification

- 2 Fine-tune with Hybrid Thinking Dataset

- 3 Adaptive Switch Between Fast and Slow Thinking

[Figure 1]

|SFT with Dual KL Divergence<br><br>[Figure 2]|
|---|

|OThink-R1|
|---|

|Reasoning|
|---|

|Answer|
|---|

|Reasoning|
|---|

|Answer|
|---|

[Figure 3]

###### Your job is to evaluate whether the model's thinking is necessary and Instruction classify it accordingly

###### Redundant Thinking Essential Thinking

|Multi-Solution Exploration|
|---|

|Key-Word Identification|
|---|

|<think>\n\n</think> The answer and the step by step demonstration|
|---|

[Figure 4]

[Figure 5]

Fast Thinking

|Repeated Self-Validation|
|---|

|Misunderstanding Prevention|
|---|

LLM-Judge

|Defensive Assumptions|
|---|

|Premise Omission Avoidance|
|---|

Question

[Figure 6]

|<think> Detailed<br><br>reasoning </think>|The answer and the step by step demonstration|
|---|---|

||Reasoning|
|---|
<br><br>[Figure 7]<br><br>|Answer|
|---|
<br><br>|Reasoning|
|---|
<br><br>|Answer|
|---|
|
|---|

Slow Thinking

Prune Redundant Reasoning

OThink-R1

- Figure 1: Illustration of the proposed OThink-R1 that equips LRMs with the adaptive hybrid reasoning ability. The pipeline consists of two main steps: ❶ Thinking Paradigm Identification. Distinctive patterns differentiating essential from redundant reasoning are extracted from LRM trajectories and organized as prompts to guide an special LLM to act as a judge in classifying reasoning trajectories. ❷ Fine-tune with Hybrid Thinking Dataset. The hybrid dataset is constructed by removing redundant reasoning trajectories to form fast-thinking samples and preserving essential ones as slow-thinking samples. The model is then fine-tuned on this dataset with dual KL-divergence constraint.

ciency and reasoning completeness, where reductions in token length are frequently accompanied by a decline in accuracy (Hou et al., 2025).

soning mode and lack automatic adaptation.

In this work, we propose OThink-R1, a novel framework that enables LRMs to automatically switch between fast and slow thinking modes. To identify instances where fast thinking is sufficient, we systematically analyze LRM reasoning trajectories and extract distinctive patterns that differentiate essential from redundant reasoning. These patterns are then organized as prompts that guide a auxiliary LLM (e.g., GPT-4o) to act as a judge, classifying reasoning trajectories as either essential or redundant. This classification serves as the basis for training LRMs to perform adaptive reasoning.

To address this issue, we advocate for a new hybrid paradigm that integrates both fast-thinking and slow-thinking modes within a single model, enabling autonomous selection of the appropriate mode for different problems. This concept is inspired by cognitive theories (Kahneman, 2011), which postulate that humans possess both fast and slow thinking capacities and can flexibly switch between them as needed. Typically, fast thinking is used for a multitude of simple tasks, providing responses swiftly; while slow thinking is reserved for complex problems that demand detailed logical reasoning. We seek to transfer this paradigm to LRMs, allowing them to dynamically adapt their reasoning strategies. The rationality behind this approach is supported by our empirical results (Table 1), which indicate that slow thinking is not always necessary—vanilla non-reasoning LLMs can solve a substantial portion of tasks with significantly fewer tokens. However, implementing adaptive hybrid reasoning in LRMs presents two key challenges: 1) How to identify which scenarios that fast thinking is sufficient? 2) How to empower LRMs to autonomously select the appropriate reasoning mode? While a few pioneering works, such as Dualformer (Su et al., 2025a) and Qwen3 (Yang et al., 2025a), have explored hybrid modes, their methods still require manual selection of the rea-

To further empower LRMs with adaptive hybrid reasoning capabilities, we explicitly construct a supervised fine-tuning (SFT) dataset. For tasks in which fast-thinking LLMs produce correct answers and the LRM’s reasoning trajectories are judged redundant, we remove the redundant steps and retain only the immediate responses, forming the fast-thinking samples. Conversely, for tasks judged as essential, or where only LRMs produce correct answers, the full reasoning processes are preserved as slow-thinking samples. Fine-tuning LRMs on this hybrid dataset enables the model to adaptively select fast or slow thinking modes based on the nature of the problem. Furthermore, we introduce a dual KL-divergence constraint to prevent thinking mode collapse by anchoring the model to both the original LRM and a non-reasoning LLM. Extensive experiments across multiple benchmarks demon-

strate that OThink-R1 reduces computational overhead while preserving competitive accuracy. Our main contributions are summarized as follows:

- • We propose OThink-R1, a hybrid reasoning framework that enables LRMs adaptively switch between fast and slow thinking modes, which is more aligned with human cognitive process.
- • We uncover three major patterns of essential and redundant reasoning trajectories in LRMs, which inform the design of an LLM-based judge for adaptively assessing reasoning necessity.
- • We conduct extensive experiments to validate the effectiveness of the proposed OThink-R1, which significantly reduces token cost while maintaining accuracy.

#### 2 Preliminaries

In this section, we present the two reasoning modes in human cognition and introduce LLMs and LRMs that correspond to these modes.

Large Language Models (LLMs) and Fast Thinking. Human fast thinking (Kahneman, 2011; Evans, 2008) is characterized by rapid, intuitive responses without deliberate analysis. Traditional LLMs like Qwen-2.5 (Qwen et al., 2025), exhibit analogous fast-thinking behavior by directly generating answers without producing intermediate reasoning steps. Formally, given an input x, these models produce output tokens y = [y1,...,ym] based on the token-wise generative probability:

πθ(y|x) =

m

πθ(yj|x,y<j). (1)

j=1

This paradigm would direct generate the response as the fast-thinking procedure. However, the lack of explicit reasoning steps leads to suboptimal performance on complex problems (Guo et al., 2025). Large Reasoning Models (LRMs) and Slow Thinking. Human slow thinking (Kahneman, 2011; Evans, 2008) engages deliberate analysis through step-by-step reasoning and careful reflection on intermediate results. LRMs replicate this sophisticated reasoning mode by generating explicit reasoning trajectories before drawing final answers. Representative models include OpenAI o1 (Jaech et al., 2024) and DeepSeek-R1 (Guo et al., 2025). These models delineate reasoning processes using special tokens (e.g., <think> and

</think>). Within these delimiters, the generated sequences systematically decompose problems into sub-components, explore alternative solution strategies, perform necessary computations, and validate intermediate results. This explicit and sophisticated reasoning paradigm enhances performance on complex tasks, such as mathematical problem solving and question answering. Nonetheless, the production of such reasoning trajectories substantially increases inference latency and computational overhead (Yang et al., 2025a).

Toward Adaptive Hybrid Reasoning. Both vanillia LLMs and LRMs are typically restricted to a single reasoning paradigm, each with its respective limitations: (1) In LLMs, reliance solely on fast thinking is insufficient for effectively addressing complex problems. (2) In LRMs, the uniform application of slow thinking to simple tasks leads to unnecessary computational overhead. This inflexibility contrasts with human cognition, where individuals autonomously adjust their reasoning mode according to task difficulty (Kahneman, 2011; Evans, 2008). Therefore, a more promising approach is to integrate both slow and fast thinking and enable adaptive switching between them according to the problem characteristics.

While recently has seen several concurrent explorations of hybrid reasoning, these approaches exhibit notable limitations. Qwen3 (Yang et al., 2025a) requires users to manually pre-specify the reasoning mode, and therefore fail to achieve adaptive mode selection. In contrast, GPT-5 and Claude3.7-Sonnet (Anthropic, 2025) claim to enable adaptive switching, but they are closed-source, whose underlying mechanisms and procedural workflows remain poorly understood.

#### 3 Methodology

In this section, we propose OThink-R1 that equips LRMs with adaptive hybrid reasoning capabilities. OThink-R1 comprises two main steps:

###### 3.1 Thinking Paradigm Identification

While slow-thinking LRMs exhibit strong reasoning capabilities, our analysis in Table 1 shows that both fast-thinking LLMs (e.g., Qwen2.5 series (Qwen et al., 2025)) and slow-thinking LRMs produce correct solutions for a substantial portion of problems—over 50% across four representative datasets. Notably, fast-thinking LLMs achieve these results using significantly fewer tokens. This

7B 14B

Dataset

Qwen Tokens R1 Tokens Overlap Ratio Qwen Tokens R1 Tokens Overlap Ratio OpenBookQA 36 638 60.92% (3020/4957) 89 559 73.01% (3619/4957) CommonsenseQA 50 741 51.20% (3990/7793) 65 572 61.62% (4802/7793) ASDIV 104 311 90.10% (865/960) 104 316 88.75% (852/960) GSM8K 276 691 87.36% (5223/5979) 274 649 89.31% (5340/5979)

- Table 1: Comparison experiments between typical non-reasoning LLMs (Qwen2.5-Instruct) and LRMs (DeepSeekR1-Distill-Qwen). In this table, we report: 1) the average number of tokens generated by LLMs and LRMs; 2) the overlap ratio, defined as the proportion of problems correctly solved by both models with respect to all problems.

substantial overlap indicates that in many cases, the reasoning processes generated by LRMs do not facilitate concluding correct solutions. This motivates us to identify patterns that characterize reasoning facilitating problem-solving.

To address this, we examine problems where LRMs succeed but LLMs fail: in these cases, reasoning demonstrably enables correct solutions that are unattainable without it. To extract characteristics of such facilitative reasoning, we invited over 10 senior NLP researchers to analyze 100 reasoning trajectories from these problems. Researchers identified common patterns across these trajectories and provided explanations for why these patterns characterize essential reasoning (examples are provided in Section B). Synthesizing these expert analyses, we identify trajectories exhibiting these patterns as essential reasoning, characterized by the following properties:

- 1. Key-word Identification: Extracting and emphasizing critical problem components forms the basis for solution derivation.
- 2. Misunderstanding Prevention: Eliminating misunderstandings in the problem statement prevents errors from incorrect assumptions.
- 3. Premise Omission Avoidance: Comprehensive coverage of all given premises and conditions ensures solution validity.

Conversely, we also invited the same senior NLP researchers to examine 100 reasoning trajectories from problems where LLMs succeed but LRMs fail—in these cases, reasoning leads to incorrect solutions while direct generation succeeds. Common patterns identified across these trajectories characterize redundant reasoning:

- 1. Multi-Solution Exploration: Persistent explo-

ration of alternative solutions despite determining a correct answer.

- 2. Repeated Self-Validation: Excessive revalidation of every intermediate step in final solutions.
- 3. Defensive Assumptions: Being overly cautious, taking extraneous hypotheses into consideration based on internal knowledge, rather than problem-specific constraints.

###### 3.2 Fine-tune with Hybrid Thinking Dataset

Having identified characteristics of essential and redundant reasoning, the next question is how to equip LRMs with the ability to autonomously select the appropriate reasoning mode. A straightforward idea is to construct a supervised fine-tuning (SFT) dataset that teaches LRMs to choose the proper reasoning mode (fast or slow) for a given problem. Constructing Hybrid SFT Dataset. To construct a dataset that enables adaptive reasoning, the identified essential and redundant patterns are organized as prompts that guide an LLM (GPT-4o) to act as a judge, classifying LRM reasoning trajectories as essential or redundant (cf. Section C). Based on this classification, we construct a dataset consisting of two components:

- • Fast-Thinking Sub-Dataset: We curate this dataset from problems where both LRMs and LLMs produce correct solutions and reasoning is classified as redundant. For these problems, the reasoning content is removed while the immediate conclusions are retained, creating fastthinking samples.
- • Slow-Thinking Sub-Dataset: We curate this dataset from two sources: (1) problems where both models succeed but reasoning is classified as essential, and (2) problems where only LRMs

OpenBookQA CommonsenseQA ASDIV GSM8K

Model

Tokens ↓ ACC ↑ Tokens ↓ ACC ↑ Tokens ↓ ACC ↑ Tokens ↓ ACC ↑ DeepSeek-R1-Distill-Qwen-7B 783 76.40% 730 64.70% 352 97.00% 719 86.10% NoThinking-R1-7B (Ma et al., 2025a) 130 56.60% 106 54.30% 138 88.00% 258 77.50% DualFormer-R1-7B (Su et al., 2025a) 723 75.20% 701 66.70% 223 96.00% 460 81.30% OThink-R1-7B 667 76.80% 634 66.90% 270 98.00% 488 86.70% DeepSeek-R1-Distill-Qwen-14B 522 92.80% 569 81.70% 319 97.00% 657 91.20% NoThinking-R1-14B (Ma et al., 2025a) 296 87.80% 373 78.90% 197 94.40% 458 88.60% DualFormer-R1-14B (Su et al., 2025a) 1688 91.40% 2003 79.80% 223 95.70% 482 86.40% OThink-R1-14B 421 93.40% 435 81.80% 412 98.00% 791 89.80%

- Table 2: Performance comparison across multiple datasets (↓: fewer tokens, better efficiency; ↑: higher accuracy, better performance). The bolded results indicate that OThink-R1 achieve the state-of-the-art performance.

(1) Prune Ratio (Training).

Dataset

Prune Ratio (Training) 7B 14B

OpenBookQA

75.18 % 67.36 % (2845/3784) (3021/4485) CommonsenseQA

74.89% 62.70% (3684/4919) (3870/6172) ASDIV

32.51% 31.45% (303/932) (295/938) GSM8K

32.11% 31.65% (1734/5400) (1738/5492)

(2) Fast-thinking Ratio (Test).

Dataset

Fast-thinking Ratio (Test)

7B 14B OpenBookQA 6.40% 36.80% CommonsenseQA 8.76% 35.70% ASDIV 13.62% 7.97% GSM8K 9.93% 8.56%

- Table 3: Prune and fast-thinking ratios across model scales and datasets. In this table, we report: (1) Prune Ratio (Training): the proportion of training examples where redundant reasoning trajectories within <think> were pruned; (2) Fast-thinking Ratio (Test): the proportion of test examples where the model activates fast-thinking.

succeed. In both cases, complete reasoning trajectories are preserved.

The constructed dataset Dtrain contains both fastthinking (immediate responses) and slow-thinking (detailed reasoning) samples. By exposing the LRM to both types during fine-tuning, it learns to adaptively engage in step-by-step reasoning, or directly drawing conclusions based on problem characteristics.

Fine-tune with Dual KL-Divergence Loss. Direct fine-tuning on Dtrain using standard maximum likelihood estimation poses a challenge: the dataset contains two fundamentally different generation patterns. Fast-thinking samples resemble nonreasoning LLMs’ direct response style, while slowthinking samples follow LRMs’ deliberate reasoning style. This diversity risks the model collapsing to a single pattern during training, as the model tends to prioritize one generation pattern over the other. Empirical results (Section 4.3) show that standard training leads to performance degradation.

To address this issue, we incorporate dual KLdivergence constraints that anchor the fine-tuned

model to both the reference LRM and LLM:

L = E(x,y)∈Dtrain β1DKL(πθ∥πLRM;x,y) + β2DKL(πθ∥πLLM;x,y) − log πθ(y|x)

(2)

where πLRM and πLLM denote the reference LRM and LLM respectively. πθ(y|x) is defined in Eq. (1). The regularization term β1 preserves reasoning capability while β2 maintains efficient generation, enabling the model to balance between fast and slow thinking. Let V denotes the token vocabulary, the KL term DKL(π1∥π2;x,y) is:

|y|

- π1(v|x,y<j)

- π2(v|x,y<j)

1 |y|

(3)

π1(v|x,y<j)log

j=1 v∈V

#### 4 Experiments

###### 4.1 Experimental Setting

Datasets. We conduct experiments on four widelyused datasets: OpenBookQA (Mihaylov et al., 2018) and CommonsenseQA (Talmor et al., 2019), two commonly used datasets for question answering evaluation; ASDIV (Miao et al., 2021) and

OpenBookQA CommonsenseQA ASDIV GSM8K Tokens ↓ ACC ↑ Tokens ↓ ACC ↑ Tokens ↓ ACC ↑ Tokens ↓ ACC ↑

Model

w/o prune (7B) 997 75.40% 775 66.10% 249 98.00% 432 86.40% w/o LLM-Judge (7B) 997 69.40% 776 65.70% 171 97.00% 279 84.20% w/o KL-constraint (7B) 328 68.80% 268 64.00% 292 97.30% 588 86.40% w/o reference LRM (7B) 4364 72.20% 4320 69.40% 2144 97.70% 2073 84.20% w/o reference LLM (7B) 649 76.40% 617 64.20% 267 97.30% 480 85.50% OThink-R1-7B 667 76.80% 634 66.90% 270 98.00% 488 86.70%

w/o prune (14B) 1676 94.60% 1455 80.70% 305 96.70% 629 91.70% w/o LLM-Judge (14B) 4353 93.20% 5038 81.50% 196 98.00% 300 88.70% w/o KL-constraint (14B) 3731 94.80% 3001 81.60% 281 97.00% 338 91.10% w/o reference LRM (14B) 10599 89.00% 11516 80.80% 8566 89.70% 4807 82.70% w/o reference LLM (14B) 253 91.40% 409 81.20% 282 96.30% 475 88.30% OThink-R1-14B 421 93.40% 435 81.80% 412 98.00% 791 89.80%

Table 4: Ablation study across multiple datasets. (↓: lower is better, ↑: higher is better)

GSM8K (Cobbe et al., 2021), two commonly used datasets for mathematical reasoning evaluation. We refer readers to Section D.1 for more details.

Models. We fine-tune the DeepSeek-R1-DistillQwen series (i.e., 7B/14B variants) on the constructed hybrid thinking dataset. To align with the fine-tuned model, we use the same DeepSeek-R1Distill-Qwen series as the reference LRM (πLRM in Eq. (2)); Additionally, we select Qwen2.5-Instruct series, the non-reasoning model that directly generates immediate responses, as the reference LLM (πLLM in Eq. (2)). Both reference models match the size of the fine-tuned model.

Baselines. We compare two baselines: 1) NoThinking (Ma et al., 2025a), a training-free method which directly bypass the reasoning process with explicit prompting; 2) DualFormer (Su et al., 2025a) proposed dropping reasoning steps during training, which enables thinking mode switch on structured tasks (e.g., maze navigation and Sokoban). However, in LLM reasoning tasks, the DualFormer fails to achieve the adaptive hybrid reasoning. In this paper, we follow its LLM setting by randomly dropping intermediate reasoning steps in large reasoning model outputs to construct the fine-tuning dataset, and then conduct SFT.

Hyperparameter Tuning. A grid search is adopted to find optimal hyperparameters. For all compared methods, we closely follow configurations in their respective publications to ensure the optimal performance. For more details, we refer readers to Section D.3.

###### 4.2 Main Results

Performance comparison on QA tasks. As shown in Table 2, OThink-R1 achieves superior accuracy while generating fewer tokens on Open-

BookQA and CommonsenseQA at both parameter scales. NoThinking (Ma et al., 2025a) instructs the model to skip reasoning by incorporating the special token (</think>) into the prompt, which generates the fewest tokens but suffers severe performance degradation compared to the base LRM. This premature termination disrupts the model’s reasoning structure and leads to substantial accuracy drop. DualFormer (Su et al., 2025a) fails to achieve adaptive hybrid reasoning, exhibiting unstable performance across benchmarks. Notably, at the 14B scale, DualFormer substantially increases token counts on QA tasks while decreasing performance relative to the base LRM. In contrast, OThink-R1 successfully achieves adaptive hybrid reasoning (cf. Table 3), and also improves both accuracy and efficiency on QA tasks.

Performance comparison on mathematical reasoning tasks. On mathematical reasoning tasks, OThink-R1 maintains competitive or superior accuracy compared to the base LRM and baseline methods. At the 7B scale, OThink-R1 achieves the highest accuracy on both tasks while generating fewer tokens compared with the base LRM. In addition, OThink-R1 also demonstrates adaptive hybrid reasoning capability (cf. Table 3).

###### 4.3 Ablation Study

In this section, five variants of OThink-R1 were evaluated: 1) w/o prune. OThink-R1 selectively removes redundant reasoning trajectories while preserving essential ones. In contrast, this variant retains reasoning trajectories for all training samples without any pruning; 2) w/o LLM-Judge. OThink-R1 employs an auxiliary LLM to classify reasoning as essential or redundant, removing only redundant cases. This variant removes the LLM-

[Figure 8]

If someone was making breakfast, they'd probably put two slices of bread in the what?

🤯🤨 (A)plasticbag (B)pantry✗ (C)supermarket (D)toaster✓ (E)prison 🫠🧐🤔

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

# 🫨

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

###### The Response of DeepSeek-R1 The Response of OThink-R1

<think>\n\n</think> The correct answer is: (D) Toaster. Explanation:

<think> Option B is the pantry. ... That makes sense because it's a common place for storing and accessing breakfast items. … Option D is the toaster. ... But the question is about where they put the bread when making breakfast, not using it in the toaster. So the bread would go into the toaster for that specific use, but not where you store it. … I think I'm confident with that. So the answer should be (B) pantry. </think> The answer is (B) pantry

✓

When someone is making breakfast, they typically toast bread using a toaster.

###### ✗ ✓

The toaster is the device that provides the heat needed to toast the bread slices, making it the most appropriate choice among the given options.

675 Tokens on Average 57 Tokens on Average

##### ✗ 675 Tokens on Average 57 Tokens on Average

- Figure 2: Comparison of DeepSeek-R1-Distill-Qwen-7B/OThink-R1-7B generated responses on CommonsenseQA.

Judge component and treats all cases where both LRM and LLM answers are correct as redundant, removing their reasoning trajectories; 3) w/o KLconstraint, which employed vanilla SFT loss without any KL divergence constraint; 4) w/o reference LRM (β1 = 0), which removed the KL constraint on the reference LRM while retaining the LLM constraint; 5) w/o reference LLM (β2 = 0), which removed the KL constraint on the reference LRM while retaining the LRM constraint.

The benefits of hybrid reasoning. OThink-R1 outperforms the w/o prune variant across most benchmarks. Since w/o prune retains all reasoning trajectories and relies solely on slow thinking, the superior performance validates the effectiveness of the hybrid reasoning.

Reasoning classification matters. The w/o LLMJudge variant underperformed OThink-R1 across all datasets. This demonstrates that answer correctness alone is insufficient—even when both models produce correct answers, some reasoning trajectories remain essential and should be preserved through fine-grained classification. This further validates the necessity of distinguishing essential from redundant reasoning.

The necessity of Dual KL constraints. OThinkR1 generally outperforms the w/o KL-constraint variant, validating the importance of the dual constraint mechanism. Moreover, the w/o reference LRM variant (β1 = 0) leads to severe overthinking, with token counts significantly increasing (e.g.,

421 to 10,599 in OpenBookQA on 14B model). In addition, the w/o reference LLM variant (β2 = 0) exhibits performance degradation. Thus, both constraints are essential for balancing efficiency and accuracy.

###### 4.4 Case Study

We present a case study in Figure 2. OThink-R1 employs fast-thinking mode and solves the problem correctly with an immediate response. In contrast, the baseline LRM exhibits overthinking: despite deriving the correct answer during reasoning, it continues the reasoning process and ultimately conclude the incorrect answer. This demonstrates that OThink-R1 can dynamically switch between fast and slow thinking modes. Moreover, this case illustrates that extensive reasoning is not always beneficial and may even degrade model performance.

#### 5 Related Work

Large Reasoning Models. Recent advances in large reasoning models (LRMs) have demonstrated remarkable capabilities in complex problemsolving tasks. LRMs such as OpenAI o1 (Jaech et al., 2024) and DeepSeek-R1 (Guo et al., 2025) generate extended reasoning trajectories that involve step-by-step analysis and reflection on intermediate results before drawing final conclusions. Unlike traditional large language models (LLMs) such as Qwen-2.5 series (Qwen et al., 2025) and GPT-4 (Achiam et al., 2023), which

directly produce answers without explicit reasoning steps, LRMs actively construct longer chains of thought to tackle challenging problems that require multi-step inference and logical deduction.

However, existing LRMs lack the capability to adaptively select appropriate thinking modes. LRMs like OpenAI o1 (Jaech et al., 2024) and DeepSeek-R1 (Guo et al., 2025) uniformly apply slow-thinking mode across all tasks, generating reasoning trajectories even for simple questions where direct answers suffice. This uniform application introduces unnecessary computational overhead and latency. While some LRMs attempt to address this limitation, they remain inadequate. Qwen3 (Yang et al., 2025a) requires users to manually specify the reasoning mode before inference, failing to achieve adaptive mode selection. Conversely, closed-source LRMs such as GPT-5 and Claude-3.7-Sonnet (Anthropic, 2025) claim to support adaptive mode switching, but their mechanisms remain unclear and inaccessible for systematic investigation.

Efficient Reasoning. Recent work reduces computational cost by uniformly shortening reasoning trajectories across all inputs. A common approach applies reinforcement learning with length penalties that discourage longer reasoning chains (Arora and Zanette, 2025; Yi et al., 2025; Hou et al., 2025). To enable more flexible control, L1 (Aggarwal and Welleck, 2025) trains models to match reasoning length with user pre-specified reasoning length, enabling length adjustment for efficient reasoning based on user needs. Alternatively, DAST (Shen et al., 2025b) employs direct preference optimization on length-preference pairs, where shorter correct responses are preferred over longer ones. Other approaches learn compressed representations (Hao

- et al., 2024; Cheng and Van Durme, 2024; Shen
- et al., 2025a; Xu et al., 2025b; Saunshi et al., 2025), distill answers from reasoning models to non-reasoning models (Kang et al., 2025), merge reasoning and non-reasoning models to control output length (Ma et al., 2025b), use prompt engineering (Han et al., 2024; Xu et al., 2025a; Renze and Guven, 2024; Wu et al., 2025), or prune reasoning steps by semantic importance (Xia et al., 2025). Beyond these length-control approaches, some methods dynamically terminate reasoning by detecting failure signals during generation (Yang et al., 2025b).

However, these methods apply uniform strategies across all problems and cannot adaptively

switch between fast and slow-thinking modes. his limitation creates a fundamental trade-off: excessive reduction in reasoning length degrades accuracy on complex tasks (Arora and Zanette, 2025; Hou et al., 2025), while insufficient reduction still incurs unnecessary reasoning costs on simple tasks that could be solved directly. In contrast, OThinkR1enables adaptive hybrid reasoning by identifying when reasoning is necessary, preserving full reasoning capacity on complex problems while bypassing reasoning entirely on simple ones.

Hybrid Reasoning. Recent works have explored various strategies to equip language models with hybrid reasoning capabilities. Several studies have proposed routing methods that rely on a base language model (Chuang et al., 2025; Mahaut et al.,

- 2024) to select between fast and slow-thinking models, either by training it as a router (Ong et al.,

- 2024) or by analyzing its output distributions (Su et al., 2025b). However, this reliance on an external language model incurs substantial parameter overhead. In contrast, OThink-R1 equips the reasoning model itself with the capability to adaptively switch reasoning modes without introducing additional parameters. Other work like NoThinking (Ma et al.,
- 2025a) instructs the model to skip reasoning by incorporating the special token (</think>) into the prompt. DualFormer (Su et al., 2025a) claims to achieve adaptive hybrid reasoning by dropping portions of reasoning steps during training. However, it only exhibits this capability on maze navigation and Sokoban tasks, failing on LLM reasoning tasks. Moreover, premature termination disrupts reasoning structure while random deletion removes critical content, degrading performance (cf. Table 2). In contrast, OThink-R1 preserves essential reasoning while removing redundant components, enabling adaptive reasoning without disrupting the reasoning chain. Additionally, ACPO (Cheng et al.,

- 2025) employs reinforcement learning to optimize reasoning mode selection. However, this approach remains closed-source without available implementation. Moreover, it can only sequentially apply slow-thinking first then fast-thinking, rather than adaptively switching between the two modes. 6 Conclusion

We propose OThink-R1, a hybrid reasoning framework that integrates fast and slow thinking within a single large reasoning model. Specifically, OThinkR1 employs an auxiliary LLM-based judge to dis-

tinguish essential from redundant reasoning, constructing a hybrid fine-tuning dataset. This dataset is then used to fine-tune LRMs, equipping them with adaptive hybrid reasoning capabilities. Experiments on mathematical reasoning and questionanswering tasks demonstrate that OThink-R1 significantly reduces reasoning tokens while maintaining competitive accuracy, validating the effectiveness of hybrid reasoning for efficient AI systems.

#### 7 Ethical Considerations

All experiments in this work are conducted using open-source datasets and models, which does not involve any sensitive data and commercial use3 4 5 6. 10 senior NLP researchers were invited to voluntarily identify reasoning patterns, with full understanding of the research purpose. We refer readers to Section G for more details

#### 8 Limitations

While OThink-R1demonstrates effective performance, it currently employs an auxiliary LLMbased judge to classify reasoning trajectories based on identified patterns. Future work could explore more end-to-end approaches where the model directly learns to distinguish essential from redundant reasoning through its internal representations, eliminating the need for external judging mechanisms.

#### References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, and 1 others. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Pranjal Aggarwal and Sean Welleck. 2025. L1: Controlling how long a reasoning model thinks with reinforcement learning. arXiv preprint arXiv:2503.04697.

Anthropic. 2025. Claude 3.7 sonnet and claude code.

Daman Arora and Andrea Zanette. 2025. Training language models to reason efficiently. In The Thirtyninth Annual Conference on Neural Information Processing Systems.

- 3https://huggingface.co/datasets/tau/

commonsense_qa (MIT license)

- 4https://github.com/allenai/OpenBookQA (Apache

2.0 license)

- 5https://huggingface.co/datasets/EleutherAI/

asdiv (CC-BY-4.0 license)

- 6https://huggingface.co/datasets/openai/gsm8k

(MIT license)

Jeffrey Cheng and Benjamin Van Durme. 2024. Compressed chain of thought: Efficient reasoning through dense representations. arXiv preprint arXiv:2412.13171.

Xiaoxue Cheng, Junyi Li, Zhenduo Zhang, Xinyu Tang, Xin Zhao, XinYu KONG, and Zhiqiang Zhang. 2025. Incentivizing dual process thinking for efficient large language model reasoning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Yu-Neng Chuang, Leisheng Yu, Guanchu Wang, Lizhe Zhang, Zirui Liu, Xuanting Cai, Yang Sui, Vladimir Braverman, and Xia Hu. 2025. Confident or seek stronger: Exploring uncertainty-based on-device llm routing from benchmarking to generalization. arXiv preprint arXiv:2502.04428.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. Preprint, arXiv:2110.14168.

Jonathan St BT Evans. 2008. Dual-processing accounts of reasoning, judgment, and social cognition. Annu. Rev. Psychol., 59(1):255–278.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Tingxu Han, Zhenting Wang, Chunrong Fang, Shiyu Zhao, Shiqing Ma, and Zhenyu Chen. 2024. Token-budget-aware llm reasoning. arXiv preprint arXiv:2412.18547.

Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. 2024. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769.

Bairu Hou, Yang Zhang, Jiabao Ji, Yujian Liu, Kaizhi Qian, Jacob Andreas, and Shiyu Chang. 2025. Thinkprune: Pruning long chain-of-thought of llms via reinforcement learning. arXiv preprint arXiv:2504.01296.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, and 1 others. 2024. Openai o1 system card. arXiv preprint arXiv:2412.16720.

Daniel Kahneman. 2011. Thinking, fast and slow. macmillan.

Yu Kang, Xianghui Sun, Liangyu Chen, and Wei Zou. 2025. C3ot: Generating shorter chain-of-thought without compromising effectiveness. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 24312–24320.

Zhong-Zhi Li, Duzhen Zhang, Ming-Liang Zhang, Jiaxin Zhang, Zengyan Liu, Yuxuan Yao, Haotian Xu, Junhao Zheng, Pei-Jie Wang, Xiuyi Chen, and 1 others. 2025. From system 1 to system 2: A survey of reasoning large language models. arXiv preprint arXiv:2502.17419.

Wenjie Ma, Jingxuan He, Charlie Snell, Tyler Griggs, Sewon Min, and Matei Zaharia. 2025a. Reasoning models can be effective without thinking. arXiv preprint arXiv:2504.09858.

Xinyin Ma, Guangnian Wan, Runpeng Yu, Gongfan Fang, and Xinchao Wang. 2025b. Cot-valve: Lengthcompressible chain-of-thought tuning. arXiv preprint arXiv:2502.09601.

Matéo Mahaut, Laura Aina, Paula Czarnowska, Momchil Hardalov, Thomas Müller, and Lluís Màrquez. 2024. Factual confidence of llms: on reliability and robustness of current estimators. arXiv preprint arXiv:2406.13415.

Shen-Yun Miao, Chao-Chun Liang, and Keh-Yih Su. 2021. A diverse corpus for evaluating and developing english math word problem solvers. arXiv preprint arXiv:2106.15772.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a suit of armor conduct electricity? a new dataset for open book question answering. In EMNLP.

Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E Gonzalez, M Waleed Kadous, and Ion Stoica. 2024. Routellm: Learning to route llms from preference data. In The Thirteenth International Conference on Learning Representations.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, and 25 others. 2025. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Matthew Renze and Erhan Guven. 2024. The benefits of a concise chain of thought on problem-solving in large language models. In 2024 2nd International Conference on Foundation and Large Language Models (FLLM), pages 476–483. IEEE.

Nikunj Saunshi, Nishanth Dikkala, Zhiyuan Li, Sanjiv Kumar, and Sashank J Reddi. 2025. Reasoning with latent thoughts: On the power of looped transformers. arXiv preprint arXiv:2502.17416.

Xuan Shen, Yizhou Wang, Xiangxi Shi, Yanzhi Wang, Pu Zhao, and Jiuxiang Gu. 2025a. Efficient reasoning with hidden thinking. arXiv preprint arXiv:2501.19201.

Yi Shen, Jian Zhang, Jieyun Huang, Shuming Shi, Wenjing Zhang, Jiangze Yan, Ning Wang, Kai Wang, and Shiguo Lian. 2025b. Dast: Difficulty-adaptive slowthinking for large reasoning models. arXiv preprint arXiv:2503.04472.

Keith E Stanovich and Richard F West. 2000. Advancing the rationality debate. Behavioral and brain sciences, 23(5):701–717.

DiJia Su, Sainbayar Sukhbaatar, Michael Rabbat, Yuandong Tian, and Qinqing Zheng. 2025a. Dualformer: Controllable fast and slow thinking by learning with randomized reasoning traces. In The Thirteenth International Conference on Learning Representations.

Jiayuan Su, Fulin Lin, Zhaopeng Feng, Han Zheng, Teng Wang, Zhenyu Xiao, Xinlong Zhao, Zuozhu Liu, Lu Cheng, and Hongwei Wang. 2025b. Cprouter: An uncertainty-aware router between llm and lrm. arXiv preprint arXiv:2505.19970.

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. 2019. CommonsenseQA: A question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4149–4158, Minneapolis, Minnesota. Association for Computational Linguistics.

P.C. Wason and J.ST.B.T. Evans. 1974. Dual processes in reasoning? Cognition, 3(2):141–154.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, and 1 others. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824– 24837.

Tong Wu, Chong Xiang, Jiachen T Wang, and Prateek Mittal. 2025. Effectively controlling reasoning models through thinking intervention. arXiv preprint arXiv:2503.24370.

Heming Xia, Yongqi Li, Chak Tou Leong, Wenjie Wang, and Wenjie Li. 2025. Tokenskip: Controllable chain-of-thought compression in llms. arXiv preprint arXiv:2502.12067.

Silei Xu, Wenhao Xie, Lingxiao Zhao, and Pengcheng He. 2025a. Chain of draft: Thinking faster by writing less. arXiv preprint arXiv:2502.18600.

Yige Xu, Xu Guo, Zhiwei Zeng, and Chunyan Miao. 2025b. Softcot: Soft chain-of-thought for efficient reasoning with llms. arXiv preprint arXiv:2502.12134.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41

others. 2025a. Qwen3 technical report. Preprint, arXiv:2505.09388.

Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Zheng Lin, Li Cao, and Weiping Wang. 2025b. Dynamic early exit in reasoning models. arXiv preprint arXiv:2504.15895.

Jingyang Yi, Justin Wang, and Sida Li. 2025. Shorterbetter: Guiding reasoning models to find optimal inference length for efficient reasoning. In The Thirtyninth Annual Conference on Neural Information Processing Systems.

#### A Case Study on Redundant Reasoning

In this section, we present three redundant reasoning cases as follows:

###### A.1 Repeated Self-Validation

Question

Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?

DeepSeek-R1-Distill-Qwen-7B (Generated Reasoning Tokens: 1416)

<think>...I think I’ve verified it enough ways. So, it’s 5 hours. But just to recap ...I double-checked it multiple ways, which is helpful because sometimes when you do math ...Just to make sure, let me recap one more time...</think>

Figure 3: Repeated Self-Validation on GSM8K.

- As shown in Figure 3, the LRM double-checks

its own reasoning steps. It does this by going back over its previous conclusions, using phrases like “I double-checked it multiple ways” and “let me recap one more time”. While this repeated checking helps ensure answers stay consistent, it also uses extra time and computational resources. Furthermore, once the model has already derived the correct answer, these extra checks are unnecessary.

A.2 Defensive Assumptions

- As shown in Figure 4, the LRM acts too cautiously by exploring multiple hypotheses. Specifically, it questions its initial understanding by expressing “maybe she is asking” or “if there’s another interpretation”. Since the problem directly asks, “How many ounces of tea does she need?”—a straightforward question requiring the total number, these additional hypotheses are unnecessary. It wastes additional computational resources, reducing the inference efficiency.

###### A.3 Multi-Solution Exploration

As illustrated in Figure 5, the model continues to explore alternative solutions even after deriving the correct answer. This behavior is evidenced by recurring phrases such as ‘another way’ ‘Alternatively’ and ‘Another approach’. Since the model has de-

###### Question

Artemis is making tea for a party. She knows her mom drinks an 8-ounce cup of tea and uses one ounce of tea. She will use this same ratio for the party. The party has 12 people there and each of them wants a 6-ounce cup of tea. How many ounces of tea does she need?

###### DeepSeek-R1-Distill-Qwen-7B (Generated Reasoning Tokens: 1316)

<think>...Alternatively, maybe she is asking how much tea she needs to add, not how much total tea is made? ...Wait, no, the question is, how many ounces of tea does she need? ...But let me think if there’s another interpretation. Maybe the 8-ounce cup is the total tea, including the tea itself and water?...</think>

- Figure 4: Defensive Assumptions on GSM8K.

Question

Tim has 30 less apples than Martha, and Harry has half as many apples as Tim. If Martha has 68 apples, how many apples does Harry have?

DeepSeek-R1-Distill-Qwen-7B (Generated Reasoning Tokens: 818)

<think>...Is there another way to approach this problem? ...Alternatively, maybe I can visualize this with a diagram ...Another way to think about it is by using equations ...</think>

- Figure 5: Multi-Solution Exploration on GSM8K.

rived the correct answer, exploring other solutions is redundant and wasting computational resources.

#### B Case Study on Essential Reasoning

In this section, we present three essential reasoning examples as follows:

###### Question

Tony’s dad is very strict about the washing machine and family members are only allowed to wash 50 total ounces of clothing at a time. Tony doesn’t want to break the rules, so he weighs his clothes and finds that a pair of socks weighs 2 ounces, underwear weighs 4 ounces, a shirt weighs 5 ounces, shorts weigh 8 ounces, and pants weigh 10 ounces. Tony is washing a pair of pants, 2 shirts, a pair of shorts, and 3 pairs of socks. How many more pairs of underwear can he add to the wash and not break the rule?

###### DeepSeek-R1-Distill-Qwen-7B (Generated Reasoning Tokens: 214)

<think>...A pair of pants weighs 10 ounces, each shirt weighs 5 ounces, shorts weigh 8 ounces, and each pair of socks weighs 2 ounces ...Tony is washing one pair of pants, two shirts, one pair of shorts, and three pairs of socks ...</think>

Figure 6: Premise Omission Avoidance on GSM8K.

Premise Omission Avoidance: The problem contains critical premises, and omitting any of these may lead to incorrect conclusions. As demonstrated in Figure 6, the model systematically identifies and incorporates each premise during reasoning. Specifically, it explicitly catches: 1) the weight of each item, and 2) What Tony’s washing items includes.

###### Question

One barnyard owl makes 5 hoot sounds per minute. If 5 less than 20 hoots per minute are heard coming out of the barn, how many Barnyard owls would this noise be coming from?

###### DeepSeek-R1-Distill-Qwen-7B (Generated Reasoning Tokens: 154)

<think>...The problem states that this number is ‘5 less than 20.’ ...Next, I know that one barnyard owl makes 5 hoots per minute ...</think>

Key-word Identification: The model performs reasoning to identify and analyze critical keywords within the problem statement. In Figure 7, the model explicitly recognizes “5 less than 20” and “one barnyard owl makes 5 hoots per minute”, which enables it to derive the correct solution.

Misunderstanding Prevention: The model explicitly verifies task requirements to avoid misunderstanding. As shown in Figure 8, when asked to calculate temperature decrease, the model recognizes this, rather than simply report the final temperature. It specifically highlights the phrases “lose one-fourth of its initial temperature” and “find the decrease”, demonstrating awareness that the task requires computing the temperature change rather than deriving the resultant Addison mountain’s temperature.

###### Question

In one hour, Addison mountain’s temperature will decrease to 3/4 of its temperature. If the current temperature of the mountain is 84 degrees, what will the temperature decrease by?

###### DeepSeek-R1-Distill-Qwen-7B (Generated Reasoning Tokens: 94)

<think>...To determine the temperature decrease, I first recognize that the temperature decreases to three-fourths of its current value after one hour ...This means the temperature will lose one-fourth of its initial temperature ...Given that the current temperature is 84 degrees, I calculate one-fourth of this value to find the decrease ...</think>

Figure 8: Misunderstanding Prevention on GSM8K.

Figure 7: Key-word Identification on GSM8K.

[Figure 18]

Figure 9: System Prompt

#### C System Prompt

OThink-R1 employs an auxiliary LLM (GPT-4o) to distinguish essential from redundant reasoning. Based on the observed reasoning patterns, we construct prompts that guide the judge to classify reasoning trajectories, enabling systematic categorization of reasoning paradigms for hybrid dataset construction.

#### D Experimental Detials

###### D.1 Datasets

Datasets. We evaluated our approach using the following four datasets:

- • OpenBookQA (Mihaylov et al., 2018): OpenBookQA provides the additional open-book knowledge for solving the multiple-choice questions. It consists of 4.96K training pairs with 500 validation and test examples. In this study, we intentionally exclude this open-book information during training and test, using only questionanswer pairs. This design choice isolates the evaluation of OThink-R1, as incorporating external knowledge risks conflating performance gains between retrieved knowledge and the method itself.
- • CommonsenseQA (Talmor et al., 2019): CommonsenseQA consists of multiple-choice questions designed to evaluate models’ ability to apply diverse forms of commonsense knowledge in reasoning tasks. The dataset is split into 9.4K training, 1.22K validation and 1.14K test sets. As there is no answer key in the test set, the validation set is utilized to evaluate the performance of our proposed OThink-R1.
- • ASDIV (Miao et al., 2021): A dataset of elementary math word problems comprising 1,200 training and 301 test problems. The test set includes diverse problem types designed to assess basic arithmetic operations, such as aggregation, addition, and others.
- • GSM8K (Cobbe et al., 2021): GSM8K is a linguistically diverse dataset of 8.5K grade school math word problems—comprising 7.5K training and 1K test examples—that is designed to evaluate multi-step reasoning.

###### D.2 SFT Dataset Construction

We constructed two SFT datasets for training: a QA dataset combining OpenBookQA and CommonsenseQA and a MATH dataset merging ASDIV and GSM8K examples, with sample pruning ratios fully specified in Table 3. All original training instances were processed through DeepSeek-R1Distill-Qwen (τ = 0.9, top-p = 0.95) to generate responses. The validation sets comprised 20% heldout samples from each dataset except for OpenBookQA which has a predefined 500-sample validation set.

###### D.3 Optimal Hyperparameters for Tuning

Hyperparameter Setting. We use the open-source TRL library 7 to train our models. The learning rate is searched in the range lr ∈ {3 × 10−5,4 × 10−5,5×10−5}, β1,β2 ∈ {10−3,10−4}, gas (gradient accumulation steps) is searched in the range {1,4}, batch (batch size in each GPU device) is set as 2. The maximum padding length for SFT is set as 3000. For all experiments, we train for 4 epochs, saving a checkpoint after each epoch. Final reported performance corresponds to the model checkpoint (selected by epoch) and hyperparameters that achieve the highest validation accuracy. Training is conducted on 8 × 80GB NVIDIA A100 GPUs. The optimal hyperparameters for each method, across all datasets and model scales (7B, 14B), are listed in Tables 6 and 7, following the order of hyperparameters specified in Table 5. We report the model results (cf. Table 2) with the best validation performance.

Inference Setting. For inference, we utilize the setting τ = 0.9, top-p = 0.95 for 7B/14B models. The LLM inference is conducted via the opensource vLLM library 8.

Table 5: Hyperparameters to be searched for each method. gas denotes the gradient accumulation steps; batch denotes the batch size in each GPU device.

|Method|Hyperparameters<br><br>|
|---|---|
|DualFormer OThink-R1|lr, gas, batch<br><br>lr, gas, batch, β1, β2|

#### E Potential Risks

The training process was conducted on 8 × 80GB NVIDIA A100 GPUs, which involves environmental costs associated with energy consumption and carbon emissions. However, the proposed framework is designed to reduce the computational overhead of reasoning models during inference by decreasing token usage, which may reduce energy consumption across numerous inference calls.

#### F Details on Scientific Artifacts

Citation of artifact creators. The creators of all scientific artifacts used in this work were properly cited. The experiments were conducted on

- 7https://github.com/huggingface/trl
- 8https://github.com/vllm-project/vllm

- Table 6: Optimal hyperparameters and best-performing epoch checkpoint (selected by validation accuracy) on QA.

|Model Size<br><br>|Method|Hyperparameters Checkpoint Epoch<br><br>|
|---|---|---|
|7B|DualFormer<br><br>OThink-R1<br><br>|3 × 10−5 4 2 2 3 × 10−5 4 2 10−3 10−3 1|
|14B<br><br>|DualFormer OThink-R1<br><br>|3 × 10−5 4 2 2 3 × 10−5 4 2 10−4 10−4 4|

- Table 7: Optimal hyperparameters and best-performing epoch checkpoint (selected by validation accuracy) on MATH.

|Model Size|Method<br><br>|Hyperparameters Checkpoint Epoch|
|---|---|---|
|7B<br><br>|DualFormer OThink-R1<br><br>|4 × 10−5 4 2 2 3 × 10−5 1 2 10−3 10−4 2|
|14B|DualFormer<br><br>OThink-R1|3 × 10−5 4 2 4 3 × 10−5 1 2 10−4 10−4 3<br><br>|

four widely-used datasets: OpenBookQA (Mihaylov et al., 2018), CommonsenseQA (Talmor et al., 2019), ASDIV (Miao et al., 2021) , and GSM8K (Cobbe et al., 2021). Each dataset was cited with reference to its original publication, and additional details regarding these datasets are provided in the appendix.

License and terms of use. The license and terms of use for the artifacts were considered in Section 7. All four datasets employed in this work—OpenBookQA, CommonsenseQA, ASDIV, and GSM8K—are publicly available and released under open-source licenses that permit research use.

Artifact usage consistency. The use of existing artifacts was consistent with their intended purposes. All four datasets—OpenBookQA, CommonsenseQA, ASDIV, and GSM8K—are standard benchmarks designed for research evaluation of question answering and mathematical reasoning capabilities, which aligns directly with their application in this work. These datasets were used solely for experimental evaluation purposes without modification. The proposed framework and any associated artifacts are intended exclusively for research purposes and academic evaluation, consistent with the original access conditions of the underlying datasets.

Privacy and content verification. The datasets used in this work—OpenBookQA, CommonsenseQA, ASDIV, and GSM8K—are established

benchmarks that have been curated and vetted by their original creators for research purposes. These datasets consist of question-answering and mathematical reasoning tasks that do not contain personal identifiable information such as names, addresses, phone numbers, or other sensitive data.

Documentation of artifacts. Detailed descriptions of all four datasets used in this work—including their domains, sizes, linguistic characteristics, and intended evaluation purposes—are presented in Section D.1.

#### G Human Subjects

Instructions given to participants. The analysis involved senior NLP researchers examining reasoning trajectories to identify patterns of essential reasoning. No formal experimental protocol with associated risks was required, as the task consisted solely of analytical review of model-generated text outputs without exposure to offensive content, personal identifying information, or other potentially harmful material. Researchers were asked to analyze reasoning trajectories, identify common patterns, and provide explanations for why these patterns characterize essential reasoning. Examples of the identified patterns are provided in Sections A and B.

Recruitment and payment. Participants were recruited from senior NLP researchers within the authors’ academic network who volunteered to contribute their expertise to this analysis. No monetary

compensation was provided, as participation was voluntary and based on academic collaboration.

Data consent. Prior to the involvement, participants were informed about the purpose of the study, specifically that their analytical expertise would be used to identify patterns in reasoning trajectories to guide the development of the proposed framework. Researchers were made aware that their analyses would contribute to characterizing essential versus redundant reasoning patterns in large reasoning models. All participants voluntarily agreed to contribute their expertise with full understanding of how their input would be utilized in the research.

Ethics review. The data collection protocol did not require ethics review board approval. The task involved technical analysis of model-generated text outputs by expert researchers and did not constitute human subjects research as defined by standard ethics review guidelines. No personal data, sensitive information, or experimental interventions involving human subjects were involved in this study.

Characteristics of annotators. This work did not involve human data annotation. The participating senior NLP researchers contributed analytical expertise to identify patterns in model-generated reasoning trajectories rather than performing data labeling tasks. No protected personal information or demographic data was collected, as the contribution was purely technical and analytical in nature.

#### H Acknowledgement on AI Assistant.

AI assistants were used in this research in the following capacities:

- • Research Methodology: GPT-4o was employed as an LLM-based judge to classify reasoning trajectories as essential or redundant based on identified patterns (see Section C for details). This use is a core component of the proposed methodology and is fully documented in the paper.
- • Writing Assistance: AI tools were used to assist with language refinement and editing of the manuscript, including grammar checking and stylistic improvements. All intellectual contributions, research design, experimental work, analysis, and core content generation were performed by the authors. The use of AI assistants was limited to editorial support and did not involve generation of scientific claims, results, or substantive content, in accordance with ACL publication ethics policies.

