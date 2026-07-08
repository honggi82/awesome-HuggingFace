# arXiv:2508.18773v1[cs.CL]26Aug2025

[Figure 1]

## ThinkDial: An Open Recipe for Controlling Reasoning Effort in Large Language Models

### Qianyu He1,2∗, Siyu Yuan1,2∗, Xuefeng Li1,3, Mingxuan Wang1,4†, Jiangjie Chen1,4†

1ByteDance Seed 2Fudan University 3Shanghai Jiao Tong University 4SIA-Lab of Tsinghua AIR and ByteDance Seed

∗Equal Contribution, Alphabetically Ordered, †Supervisors

### Abstract

Large language models (LLMs) with chain-of-thought reasoning have demonstrated remarkable problem-solving capabilities, but controlling their computational effort remains a significant challenge for practical deployment. Recent proprietary systems like OpenAI’s gpt-oss series have introduced discrete operational modes for intuitive reasoning control, but the open-source community has largely failed to achieve such capabilities. In this paper, we introduce ThinkDial, the first open-recipe end-to-end framework that successfully implements gpt-oss-style controllable reasoning through discrete operational modes. Our system enables seamless switching between three distinct reasoning regimes: High mode (full reasoning capability), Medium mode (50% token reduction with <10% performance degradation), and Low mode (75% token reduction with <15% performance degradation). We achieve this through an end-to-end training paradigm that integrates budget-mode control throughout the entire pipeline: budget-mode supervised fine-tuning that embeds controllable reasoning capabilities directly into the learning process, and two-phase budget-aware reinforcement learning with adaptive reward shaping. Extensive experiments demonstrate that ThinkDial achieves target compression-performance trade-offs with clear response length reductions while maintaining performance thresholds. The framework also exhibits strong generalization capabilities on out-of-distribution tasks.

Correspondence: Jiangjie Chen at jiangjiec@bytedance.com

### 1 Introduction

The advancement of large language models (LLMs) has led to remarkable capabilities in complex reasoning tasks through extended reasoning chains [1, 2]. However, these models often generate unnecessarily lengthy reasoning processes with redundant steps and circular reasoning, leading to increased computational costs, potential quality degradation through error propagation, and reduced interpretability [1, 2]. This poses a significant challenge for practical deployment, where different scenarios may require different levels of reasoning depth and computational budget.

Recent breakthrough systems, notably OpenAI’s gpt-oss series [3], have demonstrated remarkable reasoning capabilities while introducing an innovative paradigm for controlling computational effort through discrete

o3-mini gpt-oss-120b Original Performance Peak Ours

###### AIME 2024

###### AIME 2025

###### GSM8K

GPQA

0.98

0.9

0.9

0.8

0.8

0.97

0.8

Accuracy

Accuracy

Accuracy

Accuracy

0.7

0.7

0.96

0.7

0.6

0.95

0.6

0.6

0.5

0.5

0.94

0.4

5 10

5 10 15

0.0 0.5 1.0

0.0 2.5 5.0 7.5

Thinking Token costs (k)

Thinking Token costs (k)

Thinking Token costs (k)

Thinking Token costs (k)

Figure 1 Comparison of our ThinkDial and gpt-oss-style model in controllable reasoning. The red star indicates the performance ceiling achievable by the Qwen2.5-32B-Instruct model after RL training. Circles, squares, and triangles represent Low, Medium, and High modes, respectively.

operational modes (e.g., "Low", "Medium", "High"). Unlike explicit token budget methods that require users to specify exact computational constraints [4], the gpt-oss paradigm provides: (1) User accessibility: users can specify their preference without needing technical knowledge of token economics; (2) Dynamic allocation: the system can dynamically allocate computational resources based on problem complexity rather than rigid constraints; Unlike adaptive CoT approaches that are limited to binary switching between thinking and non-thinking modes [5], the gpt-oss framework enables: (3) Fine-grained control: multiple efficiency priorities (Low, Medium, High) can be applied to the same problem based on user requirements.

Despite the clear superiority of this mode-based paradigm, the open-source community has largely failed to achieve such capabilities. Existing controllable generation approaches predominantly require explicit specification of token budgets [4, 6–8] or rely on adaptive switching between thinking and non-thinking modes [5, 9, 10], both of which lack the intuitive three-mode control paradigm. This gap represents a significant limitation in democratizing advanced reasoning capabilities and has created an urgent need for open-recipe solutions that can match the sophistication of proprietary systems.

In this paper, we introduce ThinkDial, the first open-recipe end-to-end framework to successfully implement gpt-oss-style controllable reasoning through discrete operational modes. As shown in Figure 1, our system enables seamless switching between three distinct reasoning regimes: High mode (full reasoning capability), Medium mode (50% token reduction with <10% performance degradation), and Low mode (75% token reduction with <15% performance degradation).

The technical foundation of our approach rests on an end-to-end training paradigm that integrates budget-mode control throughout the entire pipeline: (1) Budget-Mode Supervised Fine-tuning: Rather than retrofitting existing models with RL compression techniques, we embed controllable reasoning capabilities directly into the SFT learning process. The key insights are: (a) models must learn to naturally associate different mode specifications with appropriate reasoning patterns, and (b) we must first establish stable output distributions for each mode to prevent interference between different modes during RL training. (2) BudgetAware Reinforcement Learning: To enable the model to seamlessly switch between three budget modes while preserving its performance ceiling, we employ a two-phase RL training strategy: first conducting warm-up RL training to reach optimal performance, then implementing budget-aware reward shaping with different length rewards for each mode, allowing the model to learn distinct reasoning capabilities with varying response lengths across different modes without compromising its peak reasoning ability. Additionally, we discover that models increasingly exhibit reasoning leakage from thinking sections to answer sections during RL training, a phenomenon we term "Reasoning Length Hacking". This behavior stems from aggressive compression in Low mode SFT data that caused reasoning overflow, which RL training inadvertently reinforces. To address this issue, we incorporate Leak Penalty in our reward shaping strategy.

Extensive experiments across multiple mathematical reasoning benchmarks demonstrate that ThinkDial successfully achieves the target compression-performance trade-offs with remarkable consistency. Our results show clear step-wise thinking token reductions (High → Medium → Low) while maintaining the specified

performance thresholds across diverse problem types. The framework also exhibits strong generalization capabilities, maintaining controllable behavior even on out-of-distribution tasks, despite being trained primarily on mathematical reasoning data.

The main contributions of this work include:

- • The first open-recipe implementation of reasoning control, moving beyond token budget specification.
- • An end-to-end training framework that integrates budget-mode control from supervised fine-tuning through reinforcement learning, featuring adaptive reward shaping.
- • Comprehensive experimental validation demonstrating robust controllable reasoning across multiple benchmarks with strong out-of-distribution generalization.

### 2 Related Work

CoT Compression Large reasoning models suffer from overthinking, where models generate excessively lengthy chains with redundant steps, circular reasoning, or unnecessary elaboration [1, 2], leading to increased computational costs, potential quality degradation through error propagation, reduced interpretability, and inefficient resource utilization. Many existing works have addressed the overthinking problem through CoT compression techniques, including: (1) constructing supervised fine-tuning (SFT) datasets by selecting short yet correct reasoning chains [11–13] and learning summary tokens for training [14], (2) reward shaping during reinforcement learning that rewards short and correct answers [15–17], provides dense rewards for higher-quality reasoning processes [18], or rewards early-exit rollouts that stop sufficiently early yet can be resumed to the correct answer [19], and (3) inference-time scaling methods that explore higher-quality intermediate reasoning processes to achieve CoT compression [20]. However, these approaches focus solely on CoT compression without providing controllability—the ability for users to dynamically adjust the reasoning depth based on their specific needs and priorities.

Control Reasoning Effort Several works have investigated how to enable models to achieve controllable prioritization between efficiency and performance. (1) Explicit Reasoning Token Budget: Some methods use parameters to control reasoning length by setting explicit token budgets [4, 6–8]. However, it is challenging to determine appropriate budgets for problems of varying difficulty, such as comparing GSM8K and AIME problems. (2) Adaptive CoT: These approaches allow models to autonomously decide between thinking and non-thinking modes based on the query [5, 9, 10]. However, they are limited in their ability to provide multiple efficiency priorities for the same problem. (3) Three-Mode Systems offer a structured compromise by providing discrete reasoning levels (low, medium, high) that enable users to explicitly specify their preference for the efficiency-performance trade-off [3]: prioritizing speed for time-sensitive applications, emphasizing accuracy for critical decisions, or balancing both for general use cases. This approach provides intuitive control without requiring users to understand token budgets. However, existing three-mode implementations remain proprietary, with no open-recipe solutions available to the research community. Our work is the first to provide an open-recipe solution for three-mode system, enabling users to achieve controllable efficiency-performance trade-offs without requiring token specification.

### 3 Method

We propose an end-to-end training paradigm for models to learn controllable reasoning capabilities through three sequential steps: (1) Budget-Mode Supervised Fine-tuning to embed mode-specific reasoning patterns directly into the model’s base capabilities; (2) Stage-1 RL training to establish peak performance foundation, thus the following RL training will not compromise the model’s peak reasoning ability; (3) Stage-2 RL training to implement budget-aware reward shaping that enables seamless switching between reasoning modes.

#### 3.1 Budget-Mode Supervised Fine-tuning

Previous approaches to controllable reasoning typically focus on the RL training phases. However, we argue that the SFT stage is crucial for establishing the foundation of controllable reasoning capabilities for two

reasons: (a) models must learn to naturally associate different mode specifications with appropriate reasoning patterns, and (b) we must first establish stable output distributions for each mode to prevent interference between different modes during RL training.

We construct specialized training data that demonstrates how the same problem can be solved with different levels of reasoning depth while maintaining correctness. Starting with high-quality complete reasoning chains for High mode, we systematically derive compressed variants for Medium and Low modes through targeted truncation at approximately rmed and rlow of the original thinking token length. After truncation, we add mode-specific connective text at the end of the thinking section, right before the end-of-think token (⟨/think⟩), to ensure smooth transitions and logical flow. Then we regenerate the answer section for each truncated sample to ensure the model learns to generate correct answers even when the reasoning is truncated. Only samples that maintain both logical coherence and accuracy are retained. Each reasoning mode is associated with distinct system prompts that guide mode-specific behavior, establishing the semantic relationship between mode specifications and expected reasoning patterns. The detailed construction details, statistics, prompts, and corresponding examples of the SFT data can be found in Appendix A and B.

The SFT training objective minimizes negative log-likelihood loss across all modes:

N

1 N

LSFT = −

i=1

Ti

log πθ(oi,t|oi,<t,mi,qi) (1)

t=1

where N is the total number of training samples, oi,t represents the t-th token in the output sequence of the i-th training sample, Ti denotes the length of the output sequence for the i-th sample, oi,<t represents all tokens before position t in the i-th output sequence, mi ∈ {High,Medium,Low} is the mode indicator for the i-th sample, qi is the input query for the i-th sample, and πθ is the model’s policy parameterized by θ. The training data is balanced across the three modes to ensure the model learns appropriate reasoning patterns for each mode while preserving its original capabilities.

#### 3.2 Budget-Aware Reinforcement Learning

Our reinforcement learning strategy follows a carefully designed two-phase approach to ensure that controllable reasoning capabilities are built upon a strong performance foundation rather than compromising the model’s peak abilities.

##### 3.2.1 DAPO Framework

Our reinforcement learning approach builds upon the Decouple Clip and Dynamic sAmpling Policy Optimization (DAPO) framework [21]. DAPO optimizes the policy by sampling a group of outputs {oi}Gi=1 for each input query q (which includes both the mode specification and the problem statement) and corresponding answer a. The optimization objective is formulated as:

JDAPO(θ) = E(q,a)∼D,{o

i}Gi=1∼πθold(·|q)

|oi|

G

1

min ri,t(θ)Aˆi,t, clip ri,t(θ),1 − εlow,1 + εhigh A ˆi,t

G i=1 |oi|

t=1

i=1

(2)

where the importance sampling ratio is ri,t(θ) = ππθ(oi,t|q,oi,<t)

θold(oi,t|q,oi,<t) and the advantage estimate is Aˆi,t = Ri−mean({Ri}Gi=1)

std({Ri}Gi=1) .

##### 3.2.2 Phase 1: Warm-up RL Training

In the first phase, we focus exclusively on maximizing model performance without any compression constraints. The model is trained using standard RL objectives to reach its peak reasoning capability, establishing the

performance ceiling that will serve as the reference point for subsequent compression. This warm-up phase is critical because it ensures that the model starts from its optimal state before learning to compress reasoning chains.

##### 3.2.3 Phase 2: RL with Budget-Aware Reward Shaping

The second phase introduces controllable reasoning through budget-aware reward shaping. We implement different response length rewards for each mode, allowing the model to learn distinct reasoning capabilities with varying response lengths across different modes. However, we discovered that models tend to exploit compression objectives through "Reasoning Length Hacking"—reducing thinking tokens within ⟨think⟩ tags while compensating with extended reasoning in the answer section, rather than genuinely reducing reasoning depth. To address this issue, we incorporate Leak Penalty in our reward shaping strategy, effectively preventing reasoning spillover into answer sections.

Adaptive Reward Shaping Strategies To enable controllable reasoning, we design a composite reward function that balances task performance with response length efficiency. Our approach builds upon the length reward framework from Kimi k1.5 [16] and extends it with mode-specific adaptive mechanisms. The total reward for output oi in mode m is defined as:

Ri(m) = Rtask(oi) + α(m) · Rlength(oi) + Rleak(oi) (3)

where Rtask(oi) ∈ {0,1} evaluates task correctness through exact answer matching, Rlength(oi) enforces response length constraints, Rleak(oi) ∈ {−0.5,+0.5} ensures proper reasoning-answer separation, and α(m) is the mode-specific scaling coefficient that controls response compression intensity.

Mode-Specific Response Length Reward The response length reward component implements mode-specific compression that allows different reasoning capabilities with varying response lengths across different modes:

λi if Rtask(oi) = 1 min(0,λi) if Rtask(oi) = 0

Rlength(oi) =

(4)

len(oi) − lenmin lenmax − lenmin

where λi = 0.5 −

Here, len(oi) denotes the total response length of output oi (including both thinking tokens within ⟨think⟩ tags and answer tokens after ⟨/think⟩), lenmin and lenmax are the minimum and maximum response lengths within the current group, respectively. The normalized response length penalty λi ensures scale invariance across different problem complexities by mapping response lengths to a standardized [−0.5,0.5] range.

Leak Penalty During RL training, we observed that models increasingly exhibit Reasoning Length Hacking behavior, where reasoning content leaks from the thinking section (within ⟨think⟩ tags) to the answer section (after ⟨/think⟩). This phenomenon stems from pre-existing patterns in our Budget-Mode SFT data, particularly in Low mode samples, where reasoning often overflows into answer sections due to aggressive compression during data construction. These pre-existing leakage patterns become reinforced during RL training, as they provide a path for models to maintain task correctness while achieving shorter thinking sections. However, this reinforcement of data artifacts defeats our compression objective, as reasoning effort is merely redistributed rather than genuinely reduced. To counteract this undesired pattern reinforcement, we implement:

+0.5 if no transition keywords in answer section −0.5 if transition keywords detected in answer section

(5)

Rleak(oi) =

where "transition keywords" refer to reasoning-related tokens such as "Wait", "Let me think", "Actually", "Alternatively", "However", and similar metacognitive expressions that typically indicate ongoing reasoning

processes. This binary reward mechanism penalizes the appearance of such keywords in the answer section (content after ⟨/think⟩), discouraging models from reinforcing the leakage patterns inherited from SFT data and ensuring that genuine reasoning compression occurs within the thinking section rather than pattern-based content redistribution. Illustrative cases are shown in Appendix C.

- 3.3 Inference Usage

During inference, users control reasoning modes by using the corresponding mode-specific prompts that were established during training. The model supports three operational modes: High mode for maximum accuracy with full reasoning capability, Medium mode for balanced performance with 50% compression, and Low mode for rapid responses with 75% compression. Users activate each mode by incorporating the respective budget-mode prompts (detailed in Appendix A) into their system prompts, ensuring consistent mode activation without requiring manual parameter tuning.

- 4 Experiments

- 4.1 Experimental Setup

Datasets and Benchmarks We evaluate ThinkDial across mathematical reasoning benchmarks spanning different difficulty levels: AIME 2025 (hard), AIME 2024 (medium), and GSM8K (easy). Additionally, we use GPQA diamond for out-of-distribution evaluation to assess generalization beyond mathematical domains. For evaluation reliability, we use different sampling strategies: AIME problems are evaluated 32 times each, GSM8K uses 500 randomly sampled problems evaluated 4 times each, and GPQA uses 198 samples evaluated 8 times each.

Implementation Details We use Qwen-2.5-Instruct-32B [10] as our foundation model. We first perform supervised fine-tuning (SFT) including 12K original reasoning data and 6K Budget-Mode SFT data, with details provided in the Appendix B. For RL training, both the warm-up phase and the length reward shaping phase utilize the same set of 20K in-house mathematical problems. We set truncation ratios rmed = 0.5 and rlow = 0.25 for Medium and Low mode, respectively, and of course, rhigh = 1. The mode-specific scaling coefficients follow a progressive strategy: α(high) = 0.0, α(med) = 0.5, and α(low) = 1.0. Training follows our two-phase strategy: 95 steps on High mode data to establish peak performance, then 40 additional steps with length rewards for controllable reasoning capabilities.

Baselines We evaluate ThinkDial against several key baselines to validate the effectiveness of our approach: (1) Peak-Performance Checkpoint: The capability peak checkpoint of the Qwen-2.5-Instruct-32B model after undergoing training with 12K original reasoning data for SFT and 20K in-house mathematical problems for RL. (2) w/o Budget-Mode SFT: Our framework without specialized mode-conditioned SFT but only original reasoning data to assess controllable reasoning pretraining necessity; (3) Only Budget-Mode SFT: Exclusive mode-specific SFT without RL optimization; (4) w/o Warm-up: Direct compression training without establishing peak performance; (5) Peak Truncation: Simple truncation-based compression at performance peaks; (6) gpt-oss-120b and o3-mini: OpenAI’s proprietary controllable reasoning models for state-of-the-art mode-based comparison.

Evaluation Metrics To quantify the overall effectiveness of controllable reasoning, we define a composite metric Accuracy-Cost Trade-off (ACT) Score that balances accuracy retention and compression efficiency. For mode m ∈ {High,Medium,Low}, we compute accuracy retention ratio Am = Acc

Accbase and compression rate Cm = 1 − Cost

m

Costbase , where base values represent the corresponding performance of the peak-performance checkpoint. The ACT score is:

m

Table 1 Overall ACT Score performance. L, M, and H represent Low, Medium, and High modes, respectively. "w/o BM SFT" refers to the model trained without Budget Mode SFT data. The best result in each column is highlighted in bold.

AIME 2024 AIME 2025 GSM8K L M H Avg. L M H Avg. L M H Avg.

Model

Ours 63.4 68.3 108.4 80.0 57.1 70.2 107.6 78.3 99.3 98.7 100.8 99.6 w/o BM SFT 59.5 68.1 84.7 70.8 57.7 62.8 85.7 68.8 93.9 94.6 99.0 95.8 Only BM SFT 66.0 60.7 93.1 73.3 63.8 63.2 97.2 74.8 61.1 73.9 99.8 78.3 w/o Warmup 48.6 64.6 95.5 69.6 47.1 61.7 87.2 65.3 98.4 95.6 100.5 98.2 Peak Truncation 47.1 40.5 106.5 64.7 44.3 38.2 108.8 63.8 76.6 82.9 100.1 86.5

(β(m) · Am + (1 − β(m)) · Cm) |M|

SACT = m∈M

1 if m = High 0.5 if m ∈ {Medium,Low}

β(m) =

(6)

(7)

Since High mode prioritizes performance maintenance while Medium and Low modes balance accuracy and efficiency equally, we set different weighting coefficients βm to reflect these distinct operational objectives.

Direct Performance Visualization We analyze raw accuracy and thinking token length across modes by comparing with gpt-oss-120b and o3-mini to evaluate whether our approach can replicate the controllable reasoning patterns of proprietary systems. This visualization approach reveals whether our step-wise degradation curves match those of state-of-the-art mode-based reasoning models.

#### 4.2 Overall Performance Analysis

Table 1 presents comprehensive ACT score evaluation results across different benchmarks, validating our core hypothesis that models can learn efficient reasoning without sacrificing problem-solving capabilities. Notably, High mode performance matches or even exceeds the original model baseline, addressing a fundamental concern in compression research—that controllability training might compromise peak performance. The clear stepwise degradation pattern (High → Medium → Low) demonstrates precise control over the accuracy-efficiency trade-off, achieving the target compression rates while maintaining specified performance thresholds.

Beyond raw performance metrics, our framework demonstrates intelligent adaptation to problem complexity. The model naturally allocates more reasoning effort to challenging problems (AIME series) while maintaining efficiency on simpler tasks (GSM8K), suggesting learned rather than mechanical compression behavior. Most importantly, direct performance visualizations confirm successful replication of gpt-oss-style patterns—our accuracy-token curves closely match those of gpt-oss-120b and o3-mini, proving that sophisticated modebased control can be achieved through our approach. Finally, the framework’s robustness extends beyond mathematical reasoning, with GPQA evaluation showing effective transfer to other domains despite training primarily on mathematical data.

#### 4.3 Detailed Analysis

We conduct systematic ablations to validate the necessity of each component in our end-to-end training paradigm, using direct performance visualization for comprehensive analysis.

Impact of Budget-Mode Supervised Fine-tuning Figure 2 demonstrates the critical role of Budget-Mode SFT in preventing mode interference during RL training. Without specialized SFT, RL training with length rewards alone causes the three operational modes to interfere with each other, leading to performance collapse in High mode that falls significantly below the original performance peak. This interference pattern completely

o3-mini gpt-oss-120b Original Performance Peak Ours w/o BM SFT only BM SFT

###### AIME 2024

###### AIME 2025

###### GSM8K

GPQA

0.98

0.9

0.8

0.8

0.8

0.96

Accuracy

Accuracy

Accuracy

Accuracy

0.7

0.7

0.6

0.94

0.6

0.6

0.92

0.5

0.4

5 10

5 10 15

0.0 0.5 1.0

0.0 2.5 5.0 7.5

Thinking Token costs (k)

Thinking Token costs (k)

Thinking Token costs (k)

Thinking Token costs (k)

###### Figure 2 The impact of Budget Mode SFT across different datasets.

o3-mini gpt-oss-120b Original Performance Peak Ours w/o Warmup

###### AIME 2024

###### AIME 2025

###### GSM8K

GPQA

0.98

0.9

0.9

0.8

0.8

0.96

0.8

Accuracy

Accuracy

Accuracy

Accuracy

0.7

0.7

0.7

0.94

0.6

0.6

0.6

0.92

0.5

0.5

0.4

0.90

5 10

5 10 15

0.0 0.5 1.0

0.0 2.5 5.0 7.5

Thinking Token costs (k)

Thinking Token costs (k)

Thinking Token costs (k)

Thinking Token costs (k)

Figure 3 The impact of warm-up RL training on controllable reasoning performance across different modes.

undermines the controllable reasoning objectives. With proper SFT initialization, the RL exploration phase cannot cause modes to interfere with each other. Consequently, High mode maintains performance at or above the original peak level, while Medium and Low modes achieve effective compression with controlled degradation. However, while Budget-Mode SFT establishes mode awareness, relying exclusively on SFT without RL optimization leads to significant accuracy degradation in High and Medium modes.

This finding validates our end-to-end training paradigm: Budget-Mode SFT provides the essential semantic foundation, while RL optimization fine-tunes the accuracy-efficiency balance. Neither component alone can achieve the sophisticated controllable reasoning demonstrated by our complete framework.

The Importance of Two-Phase RL Training Strategy Our two-phase RL training strategy proves essential for effective controllable reasoning. Figure 3 demonstrates that without the warm-up phase (Phase 1) to first establish peak performance, the model struggles to maintain quality in High and Medium modes when budgetaware reward shaping (Phase 2) is introduced. The warm-up phase ensures that compression capabilities are built upon a strong performance foundation rather than compromising the model’s peak abilities.

More importantly, Figure 4 compares our approach with the Peak Truncation Method that cuts reasoning chains at peak performance to target token budgets, then asks the model to generate summaries and answers. The visualization reveals that such mechanical truncation completely fails to achieve gpt-oss-style controllable reasoning patterns. While our learned compression maintains smooth degradation curves similar to proprietary systems, the Peak Truncation Method shows catastrophic performance collapse, demonstrating that sophisticated training is essential for effective mode-based reasoning control.

Addressing Reasoning Length Hacking Figure 5 presents token statistics comparing models with and without Leak Penalty, revealing the critical importance of addressing Reasoning Length Hacking. The bar chart demonstrates a counterintuitive phenomenon: without Leak Penalty, although thinking tokens decrease as intended, answer tokens significantly increase, resulting in higher total token consumption—defeating our compression objectives.

o3-mini gpt-oss-120b Original Performance Peak Ours Peak Truncation

###### AIME 2024

###### AIME 2025

###### GSM8K

GPQA

0.98

0.9

0.9

0.8

0.97

0.8

0.8

Accuracy

Accuracy

Accuracy

Accuracy

0.96

0.7

0.7

0.7

0.95

0.6

0.6

0.6

0.94

0.5

0.5

0.93

0.4

5 10 15

5 10 15

0.0 0.5 1.0 1.5

0 2 4 6 8

Token costs (k)

Token costs (k)

Token costs (k)

Token costs (k)

###### Figure 4 Performance comparison between our approach and the Peak Truncation Method.

AIME2024

AIME2025

GSM8K

GPQA

15000

15000

Thinking Tokens

15000

w/oLeakPenaltyw/LeakPenalty

TokenCount(k)

TokenCount(k)

TokenCount(k)

TokenCount(k)

1200

Total Tokens

10000

10000

10000

800

5000

5000

5000

400

0

0

0

0

low medium high

low medium high

low medium high

low medium high

AIME2024

AIME2025

GSM8K

GPQA

15000

15000

Thinking Tokens

15000

1200

TokenCount(k)

TokenCount(k)

TokenCount(k)

TokenCount(k)

Total Tokens

10000

10000

10000

800

5000

5000

5000

400

0

0

0

0

low medium high

low medium high

low medium high

low medium high

- Figure 5 The impact of Leak Penalty on model response length. Total Tokens include both Thinking Tokens and Summary Tokens.

However, with Leak Penalty in place, the model not only reduces thinking tokens but also maintains concise answer sections, achieving genuine overall token reduction. This analysis validates that effective controllable reasoning requires preventing models from circumventing compression constraints through reasoning spillover into answer sections.

Analysis of BM SFT Data Amount

As shown in Figure 6, the relationship between Budget-Mode (BM) SFT data amount and model performance reveals a critical balance in training data composition. We compare two configurations: the balanced setup (6K BM SFT + 12K original reasoning data) versus the BM-heavy setup (12K BM SFT + 12K original reasoning data).

First, adding an appropriate amount of BM data does not compromise the model’s performance ceiling and can even provide modest improvements. The balanced configuration maintains the model’s peak reasoning capabilities while establishing effective mode differentiation. Importantly, moderate BM data inclusion does not suppress the model’s output length in High mode, preserving its ability to generate comprehensive reasoning when needed.

However, excessive BM SFT data creates significant performance degradation. When BM SFT data increases substantially, the model’s performance ceiling drops noticeably across Medium and Hard questions. More critically, this BM-heavy configuration severely suppresses reasoning length across all operational modes, indicating that the model becomes overly constrained in its reasoning capacity. This length suppression is particularly problematic as it limits the model’s ability to engage in thorough reasoning even when explicitly

+ 6K BM SFT + 12K BM SFT Only Original

AIME2024

###### AIME2025

###### GSM8K

Response Length

ResponseLength

18000

0.72

0.975

| |
|---|

0.60

| |
|---|

Performance

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

16500

| |
|---|

| |
|---|

| |
|---|

0.69

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.56

| |
|---|

0.960

| |
|---|

| |
|---|

| |
|---|

15000

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.66

| |
|---|

| |
|---|

| |
|---|

0.52

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.945

13500

| |
|---|

| |
|---|

| |
|---|

0.63

| |
|---|

| |
|---|

| |
|---|

0.48

| |
|---|

| |
|---|

| |
|---|

| |
|---|

12000

| |
|---|

| |
|---|

0.60

0.930

| |
|---|

| |
|---|

| |
|---|

0 50 100

0 50 100

0 50 100

0 50 100

Step

Step

Step

Step

- Figure 6 Impact of budget mode (BM) SFT data amount on warm-up phase RL training performance and response length. Here, "+ BM SFT" represents the amount of BM SFT data mixed with original reasoning data.

instructed to operate in High mode. These findings highlight the importance of balanced training data composition in our Budget-Mode SFT approach.

### 5 Conclusion

We present ThinkDial, the first open-recipe framework to successfully implement gpt-oss-style controllable reasoning through discrete operational modes. Our end-to-end training paradigm, combining Budget-Mode Supervised Fine-tuning and Budget-Aware Reinforcement Learning with reward shaping, achieves target compression-performance trade-offs while preserving peak capabilities. Extensive experiments demonstrate that our approach closely matches proprietary systems’ controllable reasoning patterns, significantly outperforming naive truncation methods. By democratizing sophisticated mode-based reasoning control, this work enables broader research and application development in controllable AI reasoning, establishing a foundation for future advances in adaptive computational allocation.

### References

- [1] Yang Sui, Yu-Neng Chuang, Guanchu Wang, Jiamu Zhang, Tianyi Zhang, Jiayi Yuan, Hongyi Liu, Andrew Wen, Shaochen Zhong, Hanjie Chen, et al. Stop overthinking: A survey on efficient reasoning for large language models. arXiv preprint arXiv:2503.16419, 2025.

- [2] Sicheng Feng, Gongfan Fang, Xinyin Ma, and Xinchao Wang. Efficient reasoning models: A survey. arXiv preprint arXiv:2504.10903, 2025.

- [3] OpenAI. gpt-oss-120b & gpt-oss-20b model card, 2025.
- [4] Anthropic. Building with extended thinking, 2025.
- [5] Chenwei Lou, Zewei Sun, Xinnian Liang, Meng Qu, Wei Shen, Wenqi Wang, Yuntao Li, Qingping Yang, and Shuangzhi Wu. Adacot: Pareto-optimal adaptive chain-of-thought triggering via reinforcement learning. arXiv preprint arXiv:2505.11896, 2025.

- [6] Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling. arXiv preprint arXiv:2501.19393, 2025.

- [7] Yi Sun, Han Wang, Jiaqiang Li, Jiacheng Liu, Xiangyu Li, Hao Wen, Yizhen Yuan, Huiwen Zheng, Yan Liang, Yuanchun Li, et al. An empirical study of llm reasoning ability under strict output length constraint. arXiv preprint arXiv:2504.14350, 2025.

- [8] Pranjal Aggarwal and Sean Welleck. L1: Controlling how long a reasoning model thinks with reinforcement learning. arXiv preprint arXiv:2503.04697, 2025.

- [9] Jiajie Zhang, Nianyi Lin, Lei Hou, Ling Feng, and Juanzi Li. Adaptthink: Reasoning models can learn when to think. arXiv preprint arXiv:2505.13417, 2025.

- [10] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.

- [11] Heming Xia, Chak Tou Leong, Wenjie Wang, Yongqi Li, and Wenjie Li. Tokenskip: Controllable chain-of-thought compression in llms. arXiv preprint arXiv:2502.12067, 2025.

- [12] Xinyin Ma, Guangnian Wan, Runpeng Yu, Gongfan Fang, and Xinchao Wang. Cot-valve: Length-compressible chain-of-thought tuning. arXiv preprint arXiv:2502.09601, 2025.

- [13] Yu Kang, Xianghui Sun, Liangyu Chen, and Wei Zou. C3oT: Generating Shorter Chain-of-Thought Without Compromising Effectiveness. Proceedings of the AAAI Conference on Artificial Intelligence, 39(23):24312–24320, 2025.

- [14] Jintian Zhang, Yuqi Zhu, Mengshu Sun, Yujie Luo, Shuofei Qiao, Lun Du, Da Zheng, Huajun Chen, and Ningyu Zhang. Lightthinker: Thinking step-by-step compression. ArXiv, abs/2502.15589, 2025.

- [15] Yi Shen, Jian Zhang, Jieyun Huang, Shuming Shi, Wenjing Zhang, Jiangze Yan, Ning Wang, Kai Wang, Zhaoxiang Liu, and Shiguo Lian. Dast: Difficulty-adaptive slow-thinking for large reasoning models. arXiv preprint arXiv:2503.04472, 2025.

- [16] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.

- [17] Haotian Luo, Li Shen, Haiying He, Yibo Wang, Shiwei Liu, Wei Li, Naiqiang Tan, Xiaochun Cao, and Dacheng Tao. O1-pruner: Length-harmonizing fine-tuning for o1-like reasoning pruning. arXiv preprint arXiv:2501.12570, 2025.

- [18] Yuxiao Qu, Matthew YR Yang, Amrith Setlur, Lewis Tunstall, Edward Emanuel Beeching, Ruslan Salakhutdinov, and Aviral Kumar. Optimizing test-time compute via meta reinforcement fine-tuning. arXiv preprint arXiv:2503.07572, 2025.

- [19] Muzhi Dai, Chenxu Yang, and Qingyi Si. S-grpo: Early exit via reinforcement learning in reasoning models. arXiv preprint arXiv:2505.07686, 2025.

- [20] Ximing Lu, Seungju Han, David Acuna, Hyunwoo Kim, Jaehun Jung, Shrimai Prabhumoye, Niklas Muennighoff, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, et al. Retro-search: Exploring untaken paths for deeper and efficient reasoning. arXiv preprint arXiv:2504.04383, 2025.

- [21] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

- [22] Wenkai Yang, Shuming Ma, Yankai Lin, and Furu Wei. Towards thinking-optimal scaling of test-time compute for llm reasoning. arXiv preprint arXiv:2502.18080, 2025.

- [23] Liang Wen, Yunke Cai, Fenrui Xiao, Xin He, Qi An, Zhenyu Duan, Yimin Du, Junchen Liu, Lifu Tang, Xiaowei Lv, et al. Light-r1: Curriculum sft, dpo and rl for long cot from scratch and beyond. arXiv preprint arXiv:2503.10460, 2025.

/* Low Mode */

You have extremely limited time to think and respond to the users query. Every additional second of processing and reasoning incurs a significant resource cost, which could affect efficiency and effectiveness. Your task is to prioritize speed without sacrificing essential clarity or accuracy. Provide the most direct and concise answer possible. Avoid unnecessary steps, reflections, verification, or refinements UNLESS ABSOLUTELY NECESSARY. Your primary goal is to deliver a quick, clear and correct response.

/* Medium Mode */

You have sufficient time to think and respond to the user’s query, allowing for a more thoughtful and in-depth answer. However, be aware that the longer you take to reason and process, the greater the associated resource costs and potential consequences. While you should not rush, aim to balance the depth of your reasoning with efficiency. Prioritize providing a well-thought-out response, but do not overextend your thinking if the answer can be provided with a reasonable level of analysis. Use your reasoning time wisely, focusing on what is essential for delivering an accurate response without unnecessary delays and overthinking.

/* High Mode */

You have unlimited time to think and respond to the user’s question. There is no need to worry about reasoning time or associated costs. Your only goal is to arrive at a reliable, correct final answer. Feel free to explore the problem from multiple angles, and try various methods in your reasoning. This includes reflecting on reasoning by trying different approaches, verifying steps from different aspects, and rethinking your conclusions as needed. You are encouraged to take the time to analyze the problem thoroughly, reflect on your reasoning promptly and test all possible solutions. Only after a deep, comprehensive thought process should you provide the final answer, ensuring it is correct and well-supported by your reasoning.

- Table 2 System prompts used for High, Medium, and Low modes during data construction.

Budget Mode Quantity Average Length

High 2197 9440.0 Medium 2197 5125.3 Low 2132 1302.7

- Table 3 Data composition and average response lengths for budget mode supervised fine-tuning.

## Appendix

### A Mode-Specific System Prompts

To enable the switching of model output distributions across different modes, we design distinct system prompts for each reasoning mode using the same data [22]. As shown in Table 2, these mode-specific prompts establish clear semantic relationships between budget constraints and expected reasoning behaviors, guiding the model to adopt appropriate reasoning strategies for each mode.

### B Budget-Mode Supervised Fine-tuning Data

#### B.1 Data Construction

To construct high-quality training data for budget-mode supervised fine-tuning, we build a dataset of 6K budget-mode SFT samples based on our in-house training data. The dataset maintains a balanced distribution across reasoning effort levels with a ratio of high:medium:low = 1:1:1, ensuring equal representation of each budget mode during training.

For the medium and low budget modes, we set the reasoning effort ratios rmedium and rlow to 50% and 25%, respectively. Additionally, we incorporate 800 GSM8K samples with empty thinking into the low mode to further enhance the model’s ability to provide concise responses under strict budget constraints. Table 3 summarizes the data composition and characteristics for each budget mode.

To preserve the model’s general problem-solving capabilities during budget-aware fine-tuning, we incorporate 12K samples of light-R1 long chain-of-thought reasoning data from DeepSeek-R1 [23]. This auxiliary dataset consists of 3K samples from stage 2 training and 9k samples randomly selected from stage 1, providing diverse reasoning patterns that help maintain the model’s foundational reasoning abilities while adapting to budget constraints.

#### B.2 Training Data Examples

We present training data examples demonstrating how the same problem is solved with different levels of reasoning depth across the three budget modes. Starting with high-quality complete reasoning chains for High mode, we systematically derive compressed variants for Medium and Low modes through targeted truncation at approximately rmedium (50%) and rlow (25%) of the original thinking token length.

Tables 4, 5, and 6 showcase the key differences: (1) the thinking section length varies significantly across modes, with High mode containing the most detailed reasoning process; (2) after truncation, we add mode-specific connective text before the end-of-think marker (highlighted in brown) to ensure smooth logical transitions; and (3) the answer sections are regenerated for each truncated sample to maintain correctness despite the compressed reasoning. Only samples that preserve both logical coherence and accuracy after this construction process are retained in the training data.

### C Case Study of Leak Penalty

To assess the effectiveness of the Leak Penalty, we present two contrasting cases. Without the penalty (Table 7), the model keeps the thinking section brief but moves extensive reasoning—detailed calculations, repeated attempts, and lengthy explanations—into the answer, which is precisely the leak we aim to prevent. With the penalty (Table 8), the model conducts its reasoning within the ⟨think⟩ tags and returns a clean, direct answer; the answer section contains only the final summarized solution, achieving genuine compression without loss of quality.

### D Training Details

During RL training, we set the maximum prompt length to 2,048 tokens and the maximum response length to 32,768 tokens. DAPO training is conducted for 135 steps (95 steps High mode + 40 steps budget-aware) with a batch size of 256 and a mini-batch size of 256. The actor is optimized using Adam, with learning rates of 1 × 10−6, and a linear warm-up schedule over 20 steps. Our implementation uses grouped sampling with group size G = 16. For clipping parameters, we set the clip ratio low to 0.2 and clip ratio high to 0.28.

During the 95-step warmup phase of RL training, we employ Dynamic Sampling. However, in the RL with length rewards phase, we disable Dynamic Sampling because even when all responses in a group are either correct or incorrect, the presence of length rewards still provides effective gradients.

Our implementation is based on the VeRL framework. Rollouts are generated using temperature sampling (τ = 1.0), with enforced end-of-sequence tokens. We leverage vLLM for efficient batched decoding with 256 rollout slots and paged attention.

##### Prompt for High Mode

/* System Prompt */ You have unlimited time to think and respond to the user’s question. There is no need to worry about reasoning time or associated costs. Your only goal is to arrive at a reliable, correct final answer. Feel free to explore the problem from multiple angles, and try various methods in your reasoning. This includes reflecting on reasoning by trying different approaches, verifying steps from different aspects, and rethinking your conclusions as needed. You are encouraged to take the time to analyze the problem thoroughly, reflect on your reasoning promptly and test all possible solutions. Only after a deep, comprehensive thought process should you provide the final answer, ensuring it is correct and well-supported by your reasoning.

/* User Prompt */ One of the external common tangent lines of the two externally tangent circles with centers O1 and O2 touches the circles at B and C, respectively. Let A be the common point of the circles. The line BA meets the circle with center O2 at A and D. If |BA| = 5 and |AD| = 4, then what is |CD|?

##### Assistant Response

<think>

Um, let’s try to solve this geometry problem step by step. First, let’s understand the problem statement. We have two externally tangent circles with centers O1 and O2. They touch at point A, which is their common point. There’s an external common tangent that touches the first circle at B and the second at C. The line BA intersects the second circle again at D, with |BA| = 5 and |AD| = 4. We need to find |CD|.

( ... 20874 tokens omitted for brevity ) Instead, use the fact that the power of A with respect to the tangent BC is zero, and the ratio |BA|/|AD| = 5/4 = r1/r2, so by similar triangles, |CD| = |BC| · (r2/(r1 + r2)) = 3√5 · (4/9) = 4√5/3, no. I think the correct answer is 6, and the earlier steps had a miscalculation, but given the time I’ve spent, I’ll go with 6.

</think>

The problem involves two externally tangent circles with a common external tangent touching them at points B and C. The line BA intersects the second circle again at D. Using the power of a point theorem, the length of the tangent BC squared is equal to the product of the segments of the secant from B through A to D. Given |BA| = 5 and |AD| = 4, the power of point B with respect to the second circle gives |BC|2 = |BA| × |BD| = 5 × (5 + 4) = 5 × 9 = 45. Recognizing the geometric relationship and using the power of point C with respect to the first circle, we find that |CD| can be determined by similar triangles or the properties of the tangent and secant. The key insight is using the power of a point theorem correctly, leading to the conclusion that |CD| = 6.

|6|
|---|

Answer:

- Table 4 A case study of supervised fine-tuning data, presenting the prompt and corresponding response for the high

##### Prompt for Medium Mode

/* System Prompt */ You have sufficient time to think and respond to the user’s query, allowing for a more thoughtful and in-depth answer. However, be aware that the longer you take to reason and process, the greater the associated resource costs and potential consequences. While you should not rush, aim to balance the depth of your reasoning with efficiency. Prioritize providing a well-thought-out response, but do not overextend your thinking if the answer can be provided with a reasonable level of analysis. Use your reasoning time wisely, focusing on what is essential for delivering an accurate response without unnecessary delays and overthinking.

/* User Prompt */ One of the external common tangent lines of the two externally tangent circles with centers O1 and O2 touches the circles at B and C, respectively. Let A be the common point of the circles. The line BA meets the circle with center O2 at A and D. If |BA| = 5 and |AD| = 4, then what is |CD|?

##### Assistant Response

<think>

Um, let’s try to solve this geometry problem step by step. First, let’s understand the problem statement. ( ... 10863 tokens omitted for brevity )

- ∆x = 2s cos2 θ − s cos θ = s cos θ(2 cos θ − 1)
- ∆y = 2s cos θ sin θ − (−s sin θ) = s sin θ(2 cos θ + 1) Length CD = (∆x)2 + (∆y)2 = s| sin θ| cos2 θ(2 cos θ − 1)2 + (2 cos θ + 1)2 Simplify inside the square root: 4 cos4 θ − 4 cos3 θ + cos2 θ + 4 cos2 θ + 4 cos θ + 1 I should balance depth of reasoning with efficiency, so now I’ll stop thinking and deliver a well-considered response. </think>

To solve the problem, we start by noting that the two circles are externally tangent at point A, so the distance between their centers O1 and O2 is r1 + r2, where r1 and r2 are their radii.

Key Observations:

- 1. Power of a Point: For point B on the first circle, the power with respect to the second circle is given by BA · BD = 5 · 9 = 45. This power is also equal to |BO2|2 − r22.
- 2. Common Tangent: The length of the common tangent BC can be related to the radii and the distance between centers using the Pythagorean theorem for the external tangent: BC2 = (O1O2)2 − (r1 − r2)2.
- 3. Coordinate Geometry: Placing A at the origin, O1 at (−r1, 0), and O2 at (r2, 0), we use the coordinates of B and C on the tangent line to relate the radii and solve for r1 and r2.

Calculations:

- • Using the power of point B, we find r1 = 5 and solve for r2 using the length AD = 4.
- • The common tangent length BC is found using the radii and the distance between centers.

|6|
|---|

Result: After detailed calculations, the length of CD is found to be

.

- Table 5 A case study of supervised fine-tuning data, presenting the prompt and corresponding response for the medium mode.

##### Prompt for Low Mode

/* System Prompt */ You have extremely limited time to think and respond to the users query. Every additional second of processing and reasoning incurs a significant resource cost, which could affect efficiency and effectiveness. Your task is to prioritize speed without sacrificing essential clarity or accuracy. Provide the most direct and concise answer possible. Avoid unnecessary steps, reflections, verification, or refinements UNLESS ABSOLUTELY NECESSARY. Your primary goal is to deliver a quick, clear and correct response.

/* User Prompt */ One of the external common tangent lines of the two externally tangent circles with centers O1 and O2 touches the circles at B and C, respectively. Let A be the common point of the circles. The line BA meets the circle with center O2 at A and D. If |BA| = 5 and |AD| = 4, then what is |CD|?

##### Assistant Response

<think>

Um, let’s try to solve this geometry problem step by step. First, let’s understand the problem statement. ( ... 5633 tokens omitted for brevity )

Since both are radii, |O2C| = r2, so k = r2/r1 (since |O1B| = r1). Therefore:

r2 r1

x2 = r1 + r2 +

x1

r2 r1

y1

y2 =

I need to prioritize speed, so I should stop thinking now and provide the most direct answer possible. </think>

To solve the problem, we use the power of a point theorem and properties of tangent lines and circles.

- 1. Power of Point B: The power of point B with respect to the circle centered at O2 is equal to the product of the lengths of the segments from B to the points of intersection with the circle. Here, |BA| = 5 and |AD| = 4, so |BD| = |BA| + |AD| = 9. Thus, the power of B is |BA| · |BD| = 5 · 9 = 45.
- 2. Distance from B to O2: Let r1 and r2 be the radii of the circles centered at O1 and O2, respectively. Since BC is a common tangent, O1B ⊥ BC and O2C ⊥ BC, so O1B ∥ O2C. The distance between O1 and O2 is r1 + r2.
- 3. Using Coordinates and Similar Triangles: By setting up coordinates and using the fact that BA intersects the second circle at D, we find that the power of B gives |BO2|2 − r22 = 45.
- 4. Length of CD: Since CD is a tangent segment from C to D on the circle, and using the properties of the tangent and the circle, we find that CD can be determined by the power of point C or by recognizing similar triangles. However, using the power of point B and the fact that BC is a common tangent, we find that |CD| = 6. Answer:

|6|
|---|

- Table 6 A case study of supervised fine-tuning data, presenting the prompt and corresponding response for the low

<think>

Um, let’s try to solve this problem. We need to find the number of nonnegative integer triples (a, b, c) such that a + b + c = 300 and the sum of all pairwise products of the form a2b, a2c, etc., equals 6,000,000. First, let’s denote the sum S = a + b + c = 300 and the other sum as T = a2b + a2c + b2a + b2c + c2a + c2b = 6,000,000.

I should balance depth of reasoning with efficiency, so now I’ll stop thinking and deliver a wellconsidered response. </think>

Let’s denote S = a + b + c = 300. We can factor the given sum T = a2b + a2c + b2a + b2c + c2a + c2b = ab(a + b) + bc(b + c) + ca(c + a). Since a + b = S − c, b + c = S − a, and c + a = S − b, we can rewrite T as:

T = ab(S − c) + bc(S − a) + ca(S − b) = S(ab + bc + ca) − 3abc Given T = 6, 000, 000 and S = 300, we have:

300(ab + bc + ca) − 3abc = 6, 000, 000

( ... solutions omitted for brevity )

But wait, let’s check d = 0 to d = 100: that’s 101 values of d. For d = 0, 1 ordered triple. For d = 1 to 99 (99 values), 6 each: 99 × 6 = 594. For d = 100, 6. Total 1 + 594 + 6 = 601.

But earlier, when we thought only specific ratios worked, we were wrong; the algebra shows any arithmetic sequence centered at 100 works, which is a broader set.

( ... solutions omitted for brevity ) Therefore, every solution is a permutation of (100 − d, 100, 100 + d) for some nonnegative integer d ≤ 100. Now, count the ordered triples:

- - When d = 0: (100,100,100) → 1 way.
- - When d ≥ 1 and d ≤ 99: Each triple has three distinct numbers, so 3! = 6 permutations per d. There are 99 such d values (1 through 99).
- - When d = 100: The triple includes a zero: (0,100,200). This has one zero and two distinct non-zeros, so permutations: choose position for 0 (3 choices), then arrange 100 and 200 in the remaining two positions (2 ways): 3 × 2 = 6. Total permutations: 1 + 99 × 6 + 6 = 1 + 594 + 6 = 601.

Yes, this makes sense because for any nonnegative integer d from 0 to 100, the triple (100 − d, 100, 100 + d) is valid, and each such triple (except all 100s) can be permuted in 6 ways if all distinct, or 6 ways if one is zero.

The earlier mistake was not recognizing the general arithmetic sequence form, but the algebra confirms T is constant for this family, which exhausts all solutions due to the symmetric dependency on the average term 100.

|601|
|---|

Final Answer

###### Table 7 A case where the leak penalty was not applied, resulting in the think process leaking into the answer section.

<think>

Um, let’s try to solve this problem. We need to find the number of triples of nonnegative integers (a, b, c) such that a + b + c = 300 and the expression a2b + a2c + b2a + b2c + c2a + c2b = 6,000,000. Hmm, that’s a symmetric equation in a, b, c. Maybe we can use some symmetry or substitution to simplify it.

First, let’s denote S = a + b + c = 300. The given expression is the sum of all terms like a2b, which is symmetric. Let’s recall that in symmetric problems, often assuming an order, like a ≥ b ≥ c, can help reduce the cases we need to check, and then we can multiply by the number of permutations at the end.

Let’s assume a ≥ b ≥ c. Then c can range from 0 up to some value. Let’s try to express the given sum in terms of S and other symmetric sums. Let’s denote the given sum as T = a2b + a2c + b2a + b2c + c2a + c2b.

( ... solutions omitted for brevity )

Total solutions: 1 + 594 + 6 = 601. Wait, but hold on, when k = 0, that’s the all 100s, which is case 1, and k = 100 is case 3. Now, let’s check if the zero case is valid: when c = 0, the original equation has no terms with c2*0 or 02*something, so it’s just a2b + b2a = ab(a + b) = 20,000*300 = 6,000,000, which works, as we saw.

I should balance depth of reasoning with efficiency, so now I’ll stop thinking and deliver a wellconsidered response. </think>

The solutions are all triples where the variables are permutations of 100 + k, 100, and 100 − k for nonnegative integers k such that 100 − k ≥ 0. This includes the all-100s triple and permutations with one zero.

- k = 0: 1 solution (100,100,100). - k = 1 to 99: Each k gives 6 permutations, totaling 99 × 6 = 594. - k = 100: 6 permutations (with one zero).

Total: 1 + 594 + 6 = 601.

|601|
|---|

Final Answer

###### Table 8 A case where the leak penalty was applied, preventing the think process from leaking into the answer section.

