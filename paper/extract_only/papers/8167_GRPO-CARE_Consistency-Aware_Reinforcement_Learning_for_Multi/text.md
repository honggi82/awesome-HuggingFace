arXiv:2506.16141v1[cs.CV]19Jun2025

# GRPO-CARE: Consistency-Aware Reinforcement Learning for Multimodal Reasoning

Yi Chen1,2, Yuying Ge2∗, Rui Wang3, Yixiao Ge2∗, Junhao Cheng2, Ying Shan2, Xihui Liu1∗ 1The University of Hong Kong, 2ARC Lab, Tencent PCG, 3The Chinese University of Hong Kong

https://github.com/TencentARC/GRPO-CARE

Level-2

Level-1

Level-1: In-distribution | Task Domain: Daily Life | Environment: Seen kitchens Question: Considering the progress shown in the video and my current observation shown in the image, what action should I take next in order to clean the sink (Task Goal)?

60

60

+4.7% +3.8%

50

50

40

40

[Figure 1]

30

30

[Figure 2]

[Figure 3]

[Figure 4]

Consistency Rate

Level-3

85

60

+6.7% +24.5%

70

50

55

40

[Figure 5]

[Figure 6]

###### Task Progress 12Current Observation

30

40

###### Candidate Actions:

[Figure 7]

GRPO GRPO-CARE

A. pick up cloth B. wring cloth

[Figure 8]

[Figure 9]

C. close tap D. rinse cloth

###### GRPO

[Figure 10]

<think>The person has just washed the cloth under running water … The next logical step would be to wring out the cloth to remove excess water …. This aligns with the typical cleaning process where the cloth is used to wipe down surfaces after being rinsed. </think>

Level-2: Cross-environment

Level-3: Cross-task & Cross-environment

###### Task Domain: Daily life

Task Domain: Hobbies, Daily life, Recreation, Work

[Figure 11]

Environment: Unseen kitchens

Environment: Unseen in-door and out-door scenes

<answer> D. rinse cloth </answer>

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

GRPO-CARE (ours)

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

<think> Let me think through this step by step. I've just washed the cloth under running water. The image shows me standing at the sink with the tap running. The task is to ensure the cloth is thoroughly cleaned. The natural next step is to close the tap after rinsing. </think> <answer> C . close tap </answer>

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Hierarchy of SEED-Bench-R1 (SB-R1)’s Validation Set Model Performance

- Figure 1: (a) SEED-Bench-R1 (SB-R1) provides a systematic, three-level evaluation of post-training methods for MLLMs in video understanding, encompassing tasks that require both perception and reasoning to tackle complex real-world scenarios. (b) Our analysis identifies a key limitation of standard outcome-supervised GRPO: while it improves answer accuracy, it often compromises logical consistency between reasoning and answers. By introducing an adaptive, group-relative consistency bonus via reference-likelihood calibration, our GRPO-CARE achieves higher answer accuracy across all difficulty levels and improves interpretability, as reflected by increased consistency rates.

## Abstract

Recent reinforcement learning (RL) approaches, such as outcome-supervised GRPO, have advanced Chain-of-Thought reasoning in large language models (LLMs), yet their adaptation to multimodal LLMs (MLLMs) remains unexplored. To address the lack of rigorous evaluation for MLLM post-training methods—especially on tasks requiring balanced perception and reasoning—we present SEED-Bench-R1, a benchmark featuring complex real-world videos that demand intricate visual understanding and commonsense planning. SEED-Bench-R1 uniquely provides a large-scale training set and evaluates generalization across three escalating challenges: in-distribution, cross-environment, and cross-environmenttask scenarios. Using SEED-Bench-R1, we identify a key limitation of standard outcome-supervised GRPO: while it improves answer accuracy, it often degrades the logical coherence between reasoning steps and final answers, achieving only a 57.9% consistency rate. We attribute this to (1) reward signals focused solely on final answers, which encourage shortcut solutions at the expense of reasoning quality,

∗Corresponding Authors.

Preprint. Under review.

and (2) strict KL divergence penalties, which overly constrain model exploration and hinder adaptive reasoning. To overcome these issues, we propose GRPOCARE, a novel consistency-aware RL framework that jointly optimizes for both answer correctness and reasoning coherence, without requiring explicit process supervision. GRPO-CARE introduces a two-tiered reward: (1) a base reward for answer correctness, and (2) an adaptive consistency bonus, computed by comparing the model’s reasoning-to-answer likelihood (via a slowly-evolving reference model) against group peers. This dual mechanism amplifies rewards for reasoning paths that are both correct and logically consistent. By replacing the KL penalty with an adaptive, group-relative consistency bonus, GRPO-CARE consistently outperforms standard GRPO on SEED-Bench-R1, achieving a 6.7% performance gain on the most challenging evaluation level and a 24.5% improvement in consistency rate. Furthermore, GRPO-CARE demonstrates strong transferability, improving model performance across diverse video understanding benchmarks. Our work contributes a systematically designed benchmark and a generalizable post-training framework, advancing the development of more interpretable and robust MLLMs.

## 1 Introduction

Recent advances in the reasoning capabilities of Large Language Models (LLMs) [1, 2, 3] have been largely driven by improvements in long Chain of Thought (CoT) generation. Among various enhancement strategies, reinforcement learning (RL) [4, 5, 6] has emerged as a powerful posttraining technique, enabling LLMs to refine their CoT reasoning through self-improvement guided by verifiable rewards. This leads to models that excel at solving complex problems and generalize well to out-of-distribution (OOD) tasks. MLLMs extend LLMs by integrating modules for processing multimodal inputs, inheriting strong reasoning abilities while tackling richer, more complex data [7, 8, 9]. However, existing evaluations for RL-like post-training methods for MLLMs tend to focus narrowly on either perception tasks (e.g., detection, grounding) [8] or reasoning tasks (e.g., multimodal math problem solving) [10], or rely on broad general-domain training datasets without structured generalization assessment [11].

We argue that an ideal benchmark for post-training in multimodal understanding must balance perception and logical reasoning, while enabling rigorous evaluation of generalization. Such a benchmark would foster models that integrate sophisticated perception and reasoning to achieve accurate, interpretable multimodal understanding, and robust performance in real-world scenarios. To address this, we introduce SEED-Bench-R1, a challenging benchmark designed for systematic evaluation of post-training methods on video understanding. Built on prior benchmarks [12, 13] using realistic egocentric videos capturing everyday human activities [14, 15], SEED-Bench-R1 requires models to comprehend open-form task goals, track long-horizon visual progress, perceive complex environmental cues, and reason about next actions using world knowledge, as shown in Fig. 1. Crucially, it features a three-level hierarchy for generalization assessment: Level-1 (in-distribution), Level-2 (OOD cross-environment), and Level-3 (OOD cross-environment-task), supported by largescale training data and verifiable ground-truth answers suitable for RL.

Using SEED-Bench-R1, we conduct a comprehensive study comparing representative post-training methods. Our experiments confirm that RL—specifically GRPO based on outcome supervision [4]—is highly data-efficient and significantly outperforms supervised fine-tuning (SFT) on both in-distribution and OOD questions. However, we identify a key limitation: while outcomesupervised GRPO improves perception and answer accuracy for MLLMs, it often sacrifices logical coherence between reasoning chains and final answers, with a consistency rate of only 57.9%. This restricts interpretability and limits the potential performance ceiling. It originates from that optimizing solely for the final answer rewards creates a shortcut, where models prioritize answer correctness over maintaining logical coherence in reasoning steps. At the same time, strict KL divergence penalties excessively constrain the model’s exploration, preventing adaptive adjustment of causal relationships between reasoning paths and answers, and further amplifying logical inconsistencies.

To overcome this, we propose GRPO-CARE, a novel RL framework with Consistency-Aware Reward Enhancement that jointly optimizes answer correctness and logical consistency without relying on explicit process supervision. As illustrated in Fig. 3, in addition to the base reward for answer correctness, we introduce a consistency bonus derived from a slowly-updated reference model

through likelihood calibration. This bonus incentivizes the model to produce reasoning traces that are not only accurate but also logically coherent with the final answer. Specifically, GRPO-CARE maintains a reference model by updating its parameters through an exponential moving average (EMA) of the online model’s parameters. This model calibrates reasoning-to-answer consistency likelihoods for high-accuracy samples generated by the online model. To evaluate alignment, we measure the likelihood that the reference model reproduces the same answer when given the reasoning trace and multimodal question inputs. Based on this likelihood, we assign a sparse bonus to samples that demonstrate both high accuracy and strong consistency within their group. By removing the KL divergence penalty and replacing it with an adaptive, group-relative consistency bonus, GRPO-CARE encourages more effective exploration of coherent reasoning paths that lead to accurate answers.

Extensive evaluation on SEED-Bench-R1 demonstrates that GRPO-CARE consistently outperforms standard GRPO across all difficulty levels, especially in challenging OOD scenarios, improving performance by 6.7% on the most difficult Level-3 evaluation and increasing the consistency rate by 24.5%. Ablation studies confirm that the consistency-aware reward is critical for balancing overall performance and reasoning interpretability, surpassing alternative KL-based and reward-based baselines. Furthermore, GRPO-CARE-trained models show strong transferability to diverse general video understanding benchmarks, validating the robustness and generality of our approach.

In summary, our main contributions include:

- • We introduce SEED-Bench-R1, a novel benchmark that balances perception and reasoning, with rigorous hierarchical generalization evaluation for multimodal video understanding.
- • We conduct a systematic experimental analysis of post-training methods for MLLMs, revealing the limitations of current outcome-supervised GRPO in maintaining logical coherence.
- • We propose GRPO-CARE, a novel RL framework with a consistency-aware reward that significantly improves reasoning coherence and overall performance without explicit process supervision.

## 2 Related Work

RL for LLMs/MLLMs. RL from human feedback (RLHF) aligns LLM outputs with human preferences via reward models trained on human preference data [5, 16]. To enhance complex reasoning, generating long CoT is effective [1, 3, 2]. RL methods like GRPO [4] and its variants DAPO [17] and Dr.GRPO [18] optimize CoT generation using outcome-based rewards. However, outcome-only supervision can yield inconsistent reasoning despite correct answers. Addressing this, some works train additional process supervision reward models with costly step-wise annotations [19, 20, 21, 22, 23], incorporate LLM judges [24, 25, 26], or adaptive regularization via EMA-updated reference models [27]. In MLLMs, outcome-based RL may cause “Thought Collapse,” mitigated by stronger correctors [28] or step-wise reward matching [7]. Our GRPO-CARE employs a slowly updated reference model to provide bonus feedback on logically consistent and accurate responses, improving reasoning and accuracy without extra annotations or stronger correctors.

Benchmarks for MLLM Post-training. Recent RL-based post-training methods for MLLMs have primarily targeted image tasks—from perception (e.g., classification) to reasoning (e.g., visual math) [10, 8, 7, 29]. In contrast, video understanding, a more complex and general scenario, remains underexplored. Early RL-based efforts on video benchmarks [30, 31] are limited by narrow tasks (e.g., emotion recognition) [32, 33] or scarce training data [34], hindering scalable analysis. Existing benchmarks [35, 36, 37] mostly evaluate models post-trained on diverse general-domain data (e.g., Video-R1 [11]) but lack rigorous generalization assessment. To date, no comprehensive benchmark provides (1) large-scale training data for robust post-training, (2) structured validation sets across multiple generalization levels, and (3) multimodal questions balancing perception and reasoning in real-world scenarios. To address this, we propose SEED-Bench-R1, a video understanding benchmark with large-scale training data and a validation set partitioned into three generalization tiers, enabling comprehensive evaluation of MLLM post-training methods.

## 3 Pilot Study with SEED-Bench-R1

### 3.1 SEED-Bench-R1

Benchmark Overview. As shown in Fig. 1, SEED-Bench-R1 is a benchmark designed for systematically studying the impact of post-training methods for MLLMs on video understanding. Based on

Table 1: Data statistics of SEED-Bench-R1, which consists of a training set and a hierarchical threelevel validation set for in-distribution, cross-environment, and cross-environment-task evaluations.

Split # Samples Domain Cross-Environment Cross-Task Video Source Benchmark Source Train 50,269 Daily life - - Epic-Kitchens EgoPlan-Bench

Val-L1 2,432 Daily life × × Epic-Kitchens EgoPlan-Bench Val-L2 923 Daily life √

× Ego4D EgoPlan-Bench Val-L3 1,321

Hobbies, Daily life, Recreation, Work

√ √ Ego4D EgoPlan-Bench2

previous work, EgoPlan-Bench [12] and EgoPlan-Bench2 [13], SEED-Bench-R1 features 1) intricate visual input from the real world, 2) diverse questions requiring logical inference with common sense to solve practical tasks, 3) rigorous partition of validation sets to assess the robustness and generalization abilities of MLLMs across different levels, and 4) large-scale automatically constructed training questions with easily verifiable ground truth answers.

Visual Inputs and Question Design. As shown in Fig. 1, the visual inputs and questions of SEEDBench-R1 are grounded from realistic egocentric videos [14, 15] capturing everyday human activities. To answer questions in SEED-Bench-R1 correctly, the model must be capable of understanding open-form task goals, tracking long-horizon task progress, perceiving real-time environment state from an egocentric view, and utilizing inherent world knowledge to reason about the next action plan. The ground-truth answer comes from the actual next action occurring right after the current observation in the original uncropped video, with the negative options sampled from the same video. This challenging setting of candidate options demands a deep understanding of the environment state from dynamic visual input and world knowledge, such as action order dependency, rather than just the semantic meanings of task goals and actions, to discern the correct action plan. Moreover, the derivation of golden answers is traceable and easy to verify.

Dataset Composition and Validation Levels. As listed in Tab. 1, we provide both training and validation datasets to benefit community research. The training dataset is automatically constructed using Epic-Kitchens [14] videos recording daily life household tasks in kitchen environments. The validation dataset has undergone strict human verification to ensure correctness and is divided into three levels. The Level-1 (L1) questions are created using the same video source as the training data, representing in-distribution evaluation scenarios where the visual environments and task goals share overlaps with the training data. The Level-2 (L2) questions cover similar task goals as L1, but the visual observations are recorded in unseen kitchen environments by new participants from the Ego4D [15] team. The Level-3 (L3) validation subset utilizes the full set of Ego4D videos beyond the kitchen-specific subset. It contains general-domain questions spanning hobbies, recreation, and work, in addition to daily life. The visual inputs come from a variety of indoor and outdoor environments, posing greater challenges for testing the models’ generalization abilities.

### 3.2 Experiment Setup

We use Qwen2.5-VL-Instruct-7B [38] as the backbone to study post-training methods in advancing the model performance on SEED-Bench-R1. We adopt outcome-supervised GRPO [4] as a representative RL method and compare it with SFT. Both RL and SFT utilize 6k out of the 50k training samples from SEED-Bench-R1 for a pilot study. To enhance training efficiency, we limit each video to 16 frames at resolution 128×28×28, and append a frame indicating the current observation as an additional input. For SFT, training data is augmented with CoT reasoning distilled from Qwen2.5-VL-Instruct-72B and 7B via rejection sampling. GRPO uses outcome supervision with rule-based rewards, eliminating the need for explicit CoT annotations. Following DeepSeek-R1 [1], the model outputs reasoning within <think> </think> tags and the final answer within <answer> </answer> tags.

Given a multimodal question x ∼ D, GRPO samples G responses {og = (τg,ag)}Gg=1 from the policy πθ

, where τg and ag are the reasoning process and the corresponding final answer, respectively. Unlike SFT, GRPO does not rely on predefined responses. The policy is optimized by maximizing:

old

1 G

JGRPO(θ) = Ex,{og}

G

g=1

|og|

1 |og|

min

i=1

πθ(og,i|x, og,<i) πθold(og,i|x, og,<i)

Aˆg,i,

πθ(og,i|x, og,<i) πθold(og,i|x, og,<i)

, 1 − ε, 1 + ε A ˆg,i − βDKL[πθ||πref]

clip

Table 2: Performance comparison on SEED-Bench-R1’s hierarchical validation set.

L1 (In-Distribution) L2 (Cross-Env) L3 (Cross-Task, Cross-Env) Daily Life Daily Life Daily Life Hobbies Recreation Work Overall

Models

Qwen2.5-VL-7B 38.4 40.1 35.8 31.2 26.8 28.5 31.3 SFT 46.2 46.3 46.7 41.7 44.3 38.4 42.7 GRPO 52.3 53.2 51.9 43.7 55.2 39.4 46.7 GRPO-CARE (ours) 57.0 57.0 57.6 51.2 57.4 48.5 53.4

Here, ε and β are hyperparameters, and DKL is the KL divergence between the trained policy πθ and a reference policy πref. The per-token advantage Aˆg,i is set to the normalized reward rg, computed from rule-based rewards rg (e.g., rg = 1 if the extracted answer matches ground truth, else 0) across the group Aˆg,i = rg = rg−std(mean({r {r1,...,rG})

1,...,rG}) .

### 3.3 Result Analysis

Tab. 2 summarizes the performance of MLLMs trained with various methods on SEED-Bench-R1. Notably, compared to SFT, reinforcement learning with GRPO significantly improves data efficiency and boosts MLLM performance on both in-distribution (L1) and OOD (L2, L3) questions, despite relying only on a simple outcome-based reward without specialized CoT annotations.

Our analysis shows that GRPO mainly enhances perceptual abilities rather than reasoning. As shown in Fig. 2, the SFT-trained model is more prone to perceptual hallucinations, such as describing “a ball being hit from a tee” when this event does not occur. Attention map analysis reveals that GRPOtrained models generate CoT tokens that act as dynamic queries, attending to visual content more thoroughly—especially in OOD scenarios. For example, the GRPO model better highlights key visual observations and allocates more attention to critical objects (e.g., the ball on the tee), even if these are not explicitly referenced in the reasoning. We hypothesize that RL methods like GRPO encourage broader visual exploration via CoT, while SFT tends to produce superficial, pattern-memorized CoT with limited visual grounding. This likely underpins GRPO’s superior generalization.

However, outcome-supervised GRPO training for MLLMs has key limitations: unlike LLMs, MLLM reasoning does not improve proportionally during RL, often resulting in logical inconsistencies. While the GRPO-trained model frequently reaches correct answers, its CoT reasoning often lacks coherence. For instance, as shown in Fig. 2, initial reasoning steps mirror those of the base model (Qwen2.5-VL-7B), but later steps diverge and may contradict each other—e.g., suggesting “move the ball to the golf tee” but ultimately answering “hit ball with club.” Such inconsistencies, though sometimes yielding correct answers, undermine transparency.

Limited reasoning also constrains overall performance, as effective reasoning is crucial for integrating world knowledge with perception. For example, in Fig. 1, the GRPO model correctly identifies “running water” but fails to infer that the next logical step after cleaning is “turning off the faucet.” These reasoning-answer mismatches further complicate interpretability.

## 4 Consistency-Aware Reward-Enhanced GRPO for MLLMs (GRPO-CARE)

While outcome-supervised GRPO enhances visual perception in MLLMs, our analysis on SEEDBench-R1 uncovers a critical trade-off: it often produces less logically coherent reasoning chains, thereby limiting interpretability and performance. This issue arises from two main limitations. First, the standard reward focuses exclusively on final-answer accuracy, overlooking the quality of intermediate reasoning steps. This can incentivize shortcut solutions—correct answers reached via inconsistent reasoning. Second, the KL penalty disproportionately constrains reasoning traces, typically longer than answers, thereby stifling exploration of diverse and coherent reasoning paths.

To address these challenges, we propose GRPO-CARE (Consistency-Aware Reward Enhancement), a method that jointly optimizes for both answer correctness and logical consistency, without requiring explicit supervision of the reasoning process. As shown in Fig. 3, GRPO-CARE introduces a twotiered reward system: a base reward for answer correctness, and an adaptive consistency bonus. The consistency bonus is calculated by comparing the likelihood that a reasoning trace leads to the correct answer, as estimated by a slowly evolving reference model. For each high-accuracy sample generated

Question: Considering the progress shown in the video and my current observation shown in the image, what action should I take next in order to play golf (Task Goal)?

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

###### Candidate Actions:

- A. move ball to golf tee
- B. place club in cart bag
- C. hit ball with club
- D. move towards golf balls

[Figure 31]

###### Qwen2.5-VL-7B

<think>The video shows a golf driving range with a golf cart and a golf ball dispenser. The golf ball dispenser is full of golf balls, and the golf cart is positioned in front of it. This suggests that the next logical step is to move towards the golf balls to start playing golf. </think>

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

<answer> D. move towards golf balls </answer>

[Figure 36]

###### SFT

<think>The video shows a golf driving range where a ball is being hit from a tee. The next logical step after hitting the ball would be to move towards the golf balls to retrieve another one for the next shot. </think>

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

<answer> D. move towards golf balls

</answer>

[Figure 41]

###### GRPO

<think>Let me think. The video shows a golf driving range with a golf cart and a golf ball dispenser. The ball dispenser is full of golf balls, and the cart is positioned on the driving range. …, which is typically done by picking up a golf ball from the dispenser and placing it on the tee. Therefore, the most appropriate action to take next is to move the ball to the golf tee. </think>

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

<answer> C. hit ball with club </answer>

[Figure 46]

###### GRPO-CARE

<think>Let me think through this step by step. The video shows me at a driving range, and I've just selected a golf ball from the cart.

The image confirms that I‘m standing at the tee, ready to hit. The question asks what to do next. The options provided suggest I should

either move the ball to the tee, place the club in the cart bag, hit the ball with the club, or move towards the golf balls. The image clearly shows the golf ball on the tee, and the question asks what to do next. The answer is straightforward.</think>

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

<answer> C. hit ball with club

</answer>

[Figure 51]

- Figure 2: Case study of an L3 question from SEED-Bench-R1, showing a video of task progress, a final observation image, and attention maps (output-to-visual tokens). The SFT model tends to memorize reasoning patterns and exhibits perceptual hallucinations. The GRPO model attends more comprehensively to the highlighted key visual observation while lacking logical consistency in the generated content. The GRPO-CARE model further balances visual perception and logical reasoning.

by the online model, this likelihood is compared with those of its peers within the same group, encouraging the exploration of reasoning traces that are logically consistent with correct answers.

The training process, detailed in Algorithm 4, involves two-stage filtering. (1) First, we generate multiple reasoning traces per input and retain only those that exceed an accuracy baseline. (2) For these high-accuracy candidates, we assess how well each reasoning trace supports the final answer by calibrating its likelihood using a slowly evolving reference model.

Reference Model and Likelihood Calibration. The key insight is that a stable reference model—when conditioned on the online model’s reasoning trace—should assign a higher likelihood to the correct answer if the reasoning is logically grounded in the multimodal input. Specifically, the reference model is initialized from the same pretrained weights as the online model and updated via exponential moving average (EMA) to ensure stable likelihood estimation and self-adaptation. To avoid reinforcing “consistent-but-wrong” reasoning, we compute this likelihood only for trajectories with correct answers. Additionally, we cap the likelihood at a maximum threshold to prevent over-optimization toward artificially high values.

thinking 𝜏 + answer 𝑎

advantages

[Figure 52]

[Figure 53]

|𝐴1|
|---|

𝑟1𝑏

###### consistency bonus calibrate reference-likelihood

- 𝜏1 𝑎1

[Figure 54]

- 𝜏2 𝑎2

[Figure 55]

- 𝜏3 𝑎3

[Figure 56]

- 𝜏4 𝑎4

[Figure 57]

[Figure 58]

question 𝑥 online model 𝜋𝜃

[Figure 59]

|𝑟2𝑏| |
|---|---|
| |[Figure 60]|

|𝐴2|
|---|

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

|𝑟3𝑏| |
|---|---|
| | |

|𝐴3|
|---|

GroupComputation

[Figure 65]

[Figure 66]

|𝑟4𝑏| |
|---|---|
| | |

|𝐴4|
|---|

|𝜋𝜃|
|---|

[Figure 67]

##### …

|𝑝1|
|---|

|𝑟1𝑐|
|---|

[Figure 68]

- 𝜏1 𝑎1

[Figure 69]

- 𝜏2 𝑎2 𝑝3

[Figure 70]

EMA

consistency

[Figure 71]

[Figure 72]

update

bonus

|𝑝2|
|---|

|𝑟2𝑐|
|---|

[Figure 73]

[Figure 74]

[Figure 75]

|𝜋𝜙|
|---|

|[Figure 76]<br><br>𝜏3| |𝑎3|
|---|---|---|

[Figure 77]

reference model 𝜋𝜙

likelihood 𝑝

- Figure 3: GRPO-CARE uses a two-tier reward system: a base reward for answer correctness (r∗b) and

an adaptive consistency bonus (r∗c). The consistency bonus is given to high-accuracy samples whose reasoning-to-answer likelihood—estimated by a slowly updated (EMA) reference model—is higher than that of their group peers, conditioned on the multimodal question. The total reward, the sum of base and consistency rewards, is then used to compute advantages for updating the online model.

Consistency Bonus Calculation. Based on the clipped reference likelihoods, we compute a grouprelative consistency baseline as the mean clipped likelihood (minus a small margin to avoid penalizing near-average samples). Trajectories that exceed this baseline receive a sparse consistency bonus, weighted by their accuracy, ensuring that rewards prioritize both correctness and logical coherence. Model Update. To promote exploration of diverse reasoning paths, we remove the KL penalty from the GRPO training objective. Instead, we rely on the consistency bonus—added to the base reward to form the total reward—to guide online model updates toward higher-quality outputs. The reference model is updated via EMA every few steps, allowing it to gradually inherit improvements from the online model (e.g., better visual grounding or more complex reasoning) while maintaining stability against sampling noise. This balanced optimization process enhances multimodal understanding without sacrificing logical consistency, ultimately improving both performance and interpretability.

### 4.1 Evaluation on SEED-Bench-R1

We first evaluate our method on SEED-Bench-R1. As shown in Tab. 2, GRPO-CARE significantly outperforms GRPO across all three difficulty levels, with a particularly notable improvement of nearly 10% on the most challenging L3 evaluation in domains such as Hobbies and Work, which exhibit substantial distributional divergence from the training data.

To thoroughly assess the effectiveness of GRPO-CARE, we compare it against two families of baseline methods: KL-oriented baselines, which modify the application of divergence constraints, and reward-based alternatives, which replace KL penalties with consistency-aware rewards.

- • KL-Oriented Baselines. 1) KL-EMA introduces an EMA-updated reference model for adaptive constraints. 2) KL-EMA-HA selectively applies KL penalty only to high-accuracy samples, applying regularization on where alignment matters most. 3) SepKL-EMA-HA further decomposes KL into separate terms for reasoning and answer tokens to alleviate disproportionately penalizing lengthy reasoning tokens while potentially overlooking answer-reasoning inconsistencies. 4) NoKL removes the KL penalty, demonstrating the raw optimization potential absent any regularization.
- • Reward-Based Alternatives. 5) DenseCons applies continuous likelihood weighting to derive

dense consistency rewards: rcons = λcons·racc·pϕ(a|x,τ). 6) RefGen takes a more explicit approach by having the reference model regenerate answers from sampled reasoning paths, using the regen-

erated answer’s accuracy as the consistency signal: rcons = λcons · accuracy(a′ ∼ πϕ(·|x,τ),y∗).

- As shown in Tab. 3, we report both benchmark performance and the consistency rate between generated reasoning and final answers, where consistency is evaluated by GPT-4.1 to assess whether the reasoning sufficiently supports the answer. Our analysis shows that while the EMA-updated reference model improves both accuracy and consistency, restricting KL penalties to high-accuracy samples

Algorithm 1 Consistency-Aware Reward Enhanced GRPO

Require: πθ: Online policy model (initialized from pretrained weights) πϕ: Reference model with EMA updates (ϕ ← θ initially) D: Multimodal training dataset {(x, y∗)}

λcons: Consistency reward coefficient (e.g., 0.5) γacc: Minimum accuracy threshold (e.g. 0.1) γp: Maximum likelihood threshold (e.g. 0.95) ϵp: Consistency margin (e.g. 0.01)

- 1: procedure TRAINING(πθ, D, T)
- 2: for t ← 1 to T do
- 3: for each multimodal input x in batch D do
- 4: Phase 1: Trajectory Generation & Reward Computation
- 5: Generate G reasoning traces + answers: {τg, ag}Gg=1 ∼ πθ(·|x)
- 6: Compute accuracy rewards: racc,g = accuracy_score(ag, y∗)
- 7: Compute format rewards: rfmt,g = format_score(τg, ag)
- 8: Phase 2: Relative High-Accuracy Trajectory Selection
- 9: Calculate relative accuracy baseline: rˆacc = max(Eg[racc,g], γacc)
- 10: Select trajectories where racc,g ≥ rˆacc
- 11: Phase 3: Relative Consistency Evaluation
- 12: for selected trajectories (τg, ag) do
- 13: Compute reference likelihood: pg = |a1

g|

|ag| i=1 πϕ(ag,i | x, τg, ag,<i)

- 14: Clip likelihood: p˜g = min(pg, γp)
- 15: end for
- 16: Calculate relative consistency baseline: µˆp = Eg[˜pg] − ϵp
- 17: Select consistent trajectories where p˜g ≥ µˆp
- 18: Phase 4: Enhanced Reward Calculation
- 19: for each trajectory g do
- 20: Rg = racc,g + rfmt,g base reward

+ λcons · racc,g · I[consistent]

consistency bonus

- 21: Normalize advantage: Aˆg = (Rg − µR)/σR
- 22: end for
- 23: end for
- 24: Phase 5: Model Update
- 25: Update πθ via GRPO policy gradient (without KL penalty)
- 26: if t mod k = 0 then ▷ EMA update every k = 10 steps
- 27: ϕ ← αϕ + (1 − α)θ ▷ α=0.995 typical
- 28: end if
- 29: end for
- 30: return optimized policy πθ
- 31: end procedure

(KL-EMA-HA) boosts in-domain (L1) results but slightly reduces OOD (L2/L3) generalization. Decomposing KL penalties (SepKL-EMA-HA) mitigates reasoning-answer inconsistency, yielding minor gains on L2 but limited impact on L3. Notably, none of the KL-based variants outperform NoKL, indicating that standard KL regularization may hinder the performance ceiling in this context.

Among reward-based methods, DenseCons surpasses NoKL on L1 and L2 with improved consistency, but slightly underperforms on L3, likely due to overreliance on reference model calibration. RefGen greatly increases consistency but introduces instability from samplingbased answer regeneration, ultimately reducing overall performance.

Table 3: Ablation studies on SEED-Bench-R1.

Models L1 L2 L3 Consistency GRPO 52.3 53.2 46.7 57.9 KL-EMA 54.7 54.1 49.4 60.0 KL-EMA-HA 55.1 53.8 49.2 61.7 SepKL-EMA-HA 54.8 54.9 47.5 76.8 noKL 55.4 54.4 51.3 70.0 DenseCons 56.6 55.5 50.6 80.3 RefGen 55.2 54.2 49.4 86.4 CARE (ours) 57.0 57.0 53.4 82.4

Our proposed GRPO-CARE uses sparse consistency rewards to achieve robust improvements across all levels. Its two-stage filtering—leveraging adaptive

Table 4: Performance of different models on general video understanding benchmarks

Models Frames VSI-Bench VideoMMMU MMVU MVBench TempCompass VideoMME GPT-4o [39] - 34.0 61.2 75.4 - - 71.9 LLaMA-VID [40] - - - - 41.9 45.6 VideoLLaMA2 [41] - - - 44.8 54.6 - 47.9 LongVA-7B [42] - 29.2 23.9 - - 56.9 52.6 VILA-1.5-8B [43] - 28.9 20.8 - - 58.8 Video-UTR-7B [44] - - - - 61.1 62.5 56.0 LLaVA-OneVision-7B [45] - 32.4 33.8 49.2 56.7 - 58.2 Kangeroo-8B [46] - - - 61.1 62.5 69.9 55.4 Video-R1-7B [11] 32 35.8 52.3 63.8 63.9 73.2 59.3

Qwen2.5-VL-7B 32 30.1 48.1 60.0 59.0 72.6 56.6 CARE-7B (SB-R1) 32 34.3 51.6 66.2 63.2 74.3 58.1 CARE-7B (Video-R1) 32 35.8 50.4 65.8 65.1 73.5 59.6

EMA-updated reference likelihoods to provide relative, sparse feedback for high-accuracy samples—effectively enhances logical consistency and answer accuracy. This demonstrates that grouprelative sparse rewards deliver more reliable learning signals, avoiding overfitting to imperfect likelihoods (as in DenseCons) or sampling noise (as in RefGen).

### 4.2 Generalization to General Video Understanding Benchmarks

To comprehensively evaluate our model’s capabilities, we conduct extensive experiments on six challenging benchmarks spanning diverse aspects of video understanding: spatial reasoning (VSIBench [47]), knowledge-intensive QA (VideoMMMU [48] and MMVU [49]), and general video understanding (MVBench [35], TempCompass [36], and VideoMME [50]). For MMVU, we employ multiple-choice questions to ensure evaluation stability, while for VideoMME, we adopt the subtitlefree setting to focus on visual understanding.

- As shown in Tab. 4, our CARE-7B (SB-R1) achieves significant performance improvements over the base model across all benchmarks after training on SEED-Bench-R1. These consistent gains validate the quality of our benchmark’s training data, the robustness of our methodology, and the comprehensiveness of our evaluation protocol.

To further verify the effectiveness of our approach, we conduct additional experiments following Video-R1 [11], training our model using GRPO-CARE with 16-frame video inputs on general-domain data (Video-R1-260k) for 1k RL steps and testing with 32-frame inputs. The comparative results from other methods shown in Tab. 4 are taken from the Video-R1 paper. Notably, even when trained solely with RL, our model achieves competitive or superior performance compared to Video-R1-7B on most benchmarks. This is particularly remarkable given that Video-R1-7B benefits from explicit temporal order grounding constraints via GRPO rewards and supplementary supervised fine-tuning with additional data. Our model’s ability to match or outperform this strong baseline with a more streamlined training pipeline underscores the efficiency of our method.

Notably, our results suggest that improving reasoning-answer consistency can effectively encourage the model to align its responses with visual grounding results. This implicit approach demonstrates comparable efficacy to explicit visual perception constraints, presenting a promising alternative pathway for enhancing MLLM performance.

- 5 Conclusion

In this paper, we present SEED-Bench-R1, a comprehensive benchmark for evaluating post-training methods in MLLMs, and GRPO-CARE, a consistency-aware RL framework. SEED-Bench-R1 assesses model performance on tasks requiring balanced perception and reasoning through a novel three-level hierarchical generalization. Our detailed analysis shows that while standard RL methods like GRPO improve answer accuracy, they often reduce reasoning coherence. GRPO-CARE addresses this by jointly optimizing correctness and logical consistency using likelihood calibration with a slow-evolving reference model, leading to better performance and interpretability. We envision SEED-Bench-R1 and GRPO-CARE as valuable tools for advancing robust post-training methods, driving the development of more powerful MLLMs.

## References

- [1] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [2] Learning to reason with LLMs, 2024.
- [3] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.
- [4] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [5] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.
- [6] Edward Yeo, Yuxuan Tong, Morry Niu, Graham Neubig, and Xiang Yue. Demystifying long chain-of-thought reasoning in llms. arXiv preprint arXiv:2502.03373, 2025.
- [7] Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, and Dacheng Tao. R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization. arXiv preprint arXiv:2503.12937, 2025.
- [8] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual-rft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025.
- [9] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, et al. Mm-eureka: Exploring visual aha moment with rule-based large-scale reinforcement learning. arXiv preprint arXiv:2503.07365, 2025.
- [10] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025.
- [11] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms, 2025.
- [12] Yi Chen, Yuying Ge, Yixiao Ge, Mingyu Ding, Bohao Li, Rui Wang, Ruifeng Xu, Ying Shan, and Xihui Liu. Egoplan-bench: Benchmarking multimodal large language models for human-level planning. arXiv preprint arXiv:2312.06722, 2023.
- [13] Lu Qiu, Yuying Ge, Yi Chen, Yixiao Ge, Ying Shan, and Xihui Liu. Egoplan-bench2: A benchmark for multimodal large language model planning in real-world scenarios. arXiv preprint arXiv:2412.04447, 2024.
- [14] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Antonino Furnari, Jian Ma, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, and Michael Wray. Rescaling egocentric vision: Collection, pipeline and challenges for epic-kitchens-100. International Journal of Computer Vision (IJCV), 130:33–55, 2022.
- [15] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18995–19012, 2022.
- [16] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017.

- [17] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Weinan Dai, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. Dapo: An open-source llm reinforcement learning system at scale, 2025.
- [18] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.
- [19] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step, 2023.
- [20] Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process-and outcome-based feedback. arXiv preprint arXiv:2211.14275, 2022.
- [21] Guoxin Chen, Minpeng Liao, Chengxi Li, and Kai Fan. Alphamath almost zero: Process supervision without process. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.
- [22] Liangchen Luo, Yinxiao Liu, Rosanne Liu, Samrat Phatale, Meiqi Guo, Harsh Lara, Yunxuan Li, Lei Shu, Yun Zhu, Lei Meng, et al. Improve mathematical reasoning in language models by automated process supervision. arXiv preprint arXiv:2406.06592, 2024.
- [23] Peiyi Wang, Lei Li, Zhihong Shao, RX Xu, Damai Dai, Yifei Li, Deli Chen, Y Wu, and Zhifang Sui. Math-shepherd: A label-free step-by-step verifier for llms in mathematical reasoning. arXiv preprint arXiv:2312.08935, 2023.
- [24] Bofei Gao, Zefan Cai, Runxin Xu, Peiyi Wang, Ce Zheng, Runji Lin, Keming Lu, Dayiheng Liu, Chang Zhou, Wen Xiao, et al. Llm critics help catch bugs in mathematics: Towards a better mathematical verifier with natural language feedback. arXiv preprint arXiv:2406.14024, 2024.
- [25] Shijie Xia, Xuefeng Li, Yixin Liu, Tongshuang Wu, and Pengfei Liu. Evaluating mathematical reasoning beyond accuracy. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 27723–27730, 2025.
- [26] Lunjun Zhang, Arian Hosseini, Hritik Bansal, Mehran Kazemi, Aviral Kumar, and Rishabh Agarwal. Generative verifiers: Reward modeling as next-token prediction. arXiv preprint arXiv:2408.15240, 2024.
- [27] Alexandre Ramé, Johan Ferret, Nino Vieillard, Robert Dadashi, Léonard Hussenot, Pierre-Louis Cedoz, Pier Giuseppe Sessa, Sertan Girgin, Arthur Douillard, and Olivier Bachem. Warp: On the benefits of weight averaged rewarded policies, 2024.
- [28] Tong Wei, Yijun Yang, Junliang Xing, Yuanchun Shi, Zongqing Lu, and Deheng Ye. Gtr: Guided thought reinforcement prevents thought collapse in rl-based vlm agent training, 2025.
- [29] Kai Sun, Yushi Bai, Ji Qi, Lei Hou, and Juanzi Li. Mm-math: Advancing multimodal math evaluation with process evaluation and fine-grained classification. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 1358–1375, 2024.
- [30] Xiaodong Wang and Peixi Peng. Open-r1-video. https://github.com/ Wang-Xiaodong1899/Open-R1-Video, 2025.
- [31] Jiaxing Zhao, Xihan Wei, and Liefeng Bo. R1-omni: Explainable omni-multimodal emotion recognition with reinforcing learning. arXiv preprint arXiv:2503.05379, 2025.
- [32] Yuanyuan Liu, Wei Dai, Chuanxu Feng, Wenbin Wang, Guanghao Yin, Jiabei Zeng, and Shiguang Shan. Mafw: A large-scale, multi-modal, compound affective database for dynamic facial expression recognition in the wild. In Proceedings of the 30th ACM international conference on multimedia, pages 24–32, 2022.

- [33] Xingxun Jiang, Yuan Zong, Wenming Zheng, Chuangao Tang, Wanchuang Xia, Cheng Lu, and Jiateng Liu. Dfew: A large-scale database for recognizing dynamic facial expressions in the wild. In Proceedings of the 28th ACM international conference on multimedia, pages 2881–2889, 2020.
- [34] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for longcontext interleaved video-language understanding. Advances in Neural Information Processing Systems, 37:28828–28857, 2024.
- [35] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195–22206, 2024.
- [36] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? In Findings of the Association for Computational Linguistics ACL 2024, pages 8731–8772, 2024.
- [37] Xinyu Fang, Kangrui Mao, Haodong Duan, Xiangyu Zhao, Yining Li, Dahua Lin, and Kai Chen. Mmbench-video: A long-form multi-shot benchmark for holistic video understanding. Advances in Neural Information Processing Systems, 37:89098–89124, 2024.
- [38] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [39] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [40] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. In European Conference on Computer Vision, pages 323–340. Springer, 2024.
- [41] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024.
- [42] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024.
- [43] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26689–26699, 2024.
- [44] En Yu, Kangheng Lin, Liang Zhao, Yana Wei, Zining Zhu, Haoran Wei, Jianjian Sun, Zheng Ge, Xiangyu Zhang, Jingyu Wang, et al. Unhackable temporal rewarding for scalable video mllms. arXiv preprint arXiv:2502.12081, 2025.
- [45] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [46] Jiajun Liu, Yibing Wang, Hanghang Ma, Xiaoping Wu, Xiaoqi Ma, Xiaoming Wei, Jianbin Jiao, Enhua Wu, and Jie Hu. Kangaroo: A powerful video-language model supporting long-context video input. arXiv preprint arXiv:2408.15542, 2024.
- [47] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. arXiv preprint arXiv:2412.14171, 2024.

- [48] Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Yuanhan Zhang, Xiang Yue, Bo Li, and Ziwei Liu. Video-mmmu: Evaluating knowledge acquisition from multi-discipline professional videos. arXiv preprint arXiv:2501.13826, 2025.
- [49] Yilun Zhao, Lujing Xie, Haowei Zhang, Guo Gan, Yitao Long, Zhiyuan Hu, Tongyan Hu, Weiyuan Chen, Chuhan Li, Junyang Song, et al. Mmvu: Measuring expert-level multi-discipline video understanding. arXiv preprint arXiv:2501.12380, 2025.
- [50] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024.

