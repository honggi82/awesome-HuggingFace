# arXiv:2508.14460v1[cs.LG]20Aug2025

[Figure 1]

[Figure 2]

## DuPO: Enabling Reliable LLM Self-Verification via Dual Preference Optimization

#### Shuaijie She♡♠∗, Yu Bao♠, Yu Lu♠, Lu Xu♠, Tao Li♠, Wenhao Zhu♠, Shujian Huang♡( ), Shanbo Cheng♠( ), Lu Lu♠, Yuxuan Wang♠

♠ByteDance Seed, ♡Nanjing University ∗Work done during an internship at ByteDance Seed, Corresponding authors

### Abstract

We present DuPO, a dual learning-based preference optimization framework that generates annotation-free feedback via a generalized duality. DuPO addresses two key limitations: Reinforcement Learning with Verifiable Rewards (RLVR)’s reliance on costly labels and applicability restricted to verifiable tasks, and traditional dual learning’s restriction to strictly dual task pairs (e.g., translation and back-translation). Specifically, DuPO decomposes a primal task’s input into known and unknown components, then constructs its dual task to reconstruct the unknown part using the primal output and known information (e.g., reversing math solutions to recover hidden variables), broadening applicability to non-invertible tasks. The quality of this reconstruction serves as a self-supervised reward to optimize the primal task, synergizing with LLMs’ ability to instantiate both tasks via a single model. Empirically, DuPO achieves substantial gains across diverse tasks: it enhances the average translation quality by 2.13 COMET over 756 directions, boosts the mathematical reasoning accuracy by an average of 6.4 points on three challenge benchmarks, and enhances performance by 9.3 points as an inference-time reranker (trading computation for accuracy). These results position DuPO as a scalable, general, and annotation-free paradigm for LLM optimization.

Date: August 21, 2025 Correspondence: chengshanbo@bytedance.com( ), huangsj@nju.edu.cn( )

### 1 Introduction

Large Language Models (LLMs) [1, 10, 13, 29, 34, 51] have shown remarkable progress in tasks like mathematical reasoning [2, 8, 42, 50] and multilingual translation [9, 26, 27, 57]. To further enhance their capabilities, reinforcement learning (RL) paradigms like Reinforcement Learning from Human Feedback (RLHF) [13, 29, 51] and Reinforcement Learning with Verifiable Rewards (RLVR) [11, 17, 20, 43, 52] have gained traction. Specifically, RLHF aligns models with human preferences but relies on costly, inconsistent human annotations [23, 54]. RLVR addresses this for objective tasks (e.g., math, code) via binary rewards from verifiable answers, reducing annotation burdens. However, RLVR still depends on external supervision: acquiring verifiable answers remains a bottleneck, limiting scalability. Moreover, it struggles with open-ended tasks (e.g., multilingual translation), where single references cannot capture diverse high-quality outputs [6, 22]. Recent attempts (e.g., AI-Feedback/RLAIF [23], Constitutional AI [3]) merely swap dependencies (human labels → teacher models/rules), failing to resolve the core bottleneck.

Dual learning [15] offers a self-supervised alternative by leveraging task duality to generate intrinsic feedback: through paired “primal” and “dual” tasks (e.g., translation and back-translation [41]), models validate outputs via cycle consistency, eliminating reliance on external labels. Given that LLMs possess diverse capabilities from extensive pretraining, they could be trained across various tasks. However, applying this framework to LLMs is non-trivial, which faces two critical challenges:

- 1. Limited Duality in Irreversible Tasks: Most real-world LLM tasks (e.g., creative writing, math reasoning) lack strict invertibility. LLM’s output (e.g., a math solution) rarely contains enough information to reconstruct its input (e.g., the original problem), breaking the duality cycle and invalidating selfsupervision.
- 2. Bidirectional Competence Asymmetry: LLMs often exhibit uneven performance across primal/dual tasks (e.g., strong at solving math problems but weak at generating problems from solutions). Noisy self-signals from asymmetric tasks hinder optimization, reducing the framework’s utility.

These mismatches render traditional dual learning ill-suited for general LLM optimization, leaving it an open challenge.

In this paper, we propose DuPO (Dual Learning-based Preference Optimization), a framework that aligns LLM generalization with a (relaxed) duality applicable to general tasks. At its core lies a generalized duality framework (§3.3) built on complementary relationships: it decomposes each input x into disjoint known (xk) and unknown (xu) components, then designs the dual objective to reconstruct only xu from the primal output y and the known input xk, rather than inverting the full input. This framework resolves two asymmetries: it restores sufficient information flow between the primal and dual tasks (task asymmetry) and reduces the complexity burden on the dual task side (capability asymmetry). The formulation naturally synergizes with LLMs: their broad foundational capabilities allow a single model to instantiate both primal and dual tasks without specific architectures, while the dual task converts the model’s outputs into self-supervised reward signals, enabling continual improvement without external annotations. This bidirectional benefit addresses a critical challenge in LLM development: obtaining high-quality feedback for capability enhancement.

We empirically validate DuPO on two diverse and representative tasks: mathematical reasoning and multilingual translation, demonstrating significant and consistent improvements. By applying DuPO to one of the strongest translation LLM, Seed-X-7B-Instruct [9], we demonstrate a significant further performance gain of 2.13 COMET points on the multilingual translation benchmark, bringing the 7B model to performance comparable to ultra large state-of-the-art systems. In mathematical reasoning, our method yields robust gains across models of varying scales, from 1.5B to 7B parameters; notably, DuPO improves the Qwen3-4B model’s score on three challenging mathematical benchmark by 6.4 percentage points. Our comprehensive ablation studies confirm that our design, the generalized duality, is crucial for achieving these results. Beyond training, DuPO acts as a reranking mechanism at inference, boosting performance by 9.3 points without finetuning—enabling smaller models to outperform stronger ultra-large LLM like DeepSeek-R1 even without training. In summary, DuPO reimagines task duality for non-invertible LLM tasks. It eliminates external annotation reliance, scales across tasks/domains, and enhances both training and inference—offering a scalable path to align LLMs with diverse goals using self-supervised feedback.

### 2 Related Work

#### 2.1 Preference Optimization for LLMs

Preference optimization has driven significant advancements in large language models (LLMs) by aligning outputs with feedback signals, with three dominant paradigms shaping the field: (1) Reinforcement Learning from Human Feedback (RLHF) [35] has become a cornerstone for aligning LLMs with human preferences. Its workflow typically involves training a reward model [30, 48] on human-annotated preference pairs, then using reinforcement learning (e.g., PPO [39], GRPO [42]) to optimize the policy model [13, 29, 51]. While effective for subjective tasks, RLHF faces critical bottlenecks: human annotation is costly to scale [23], and consistency across annotators degrades for complex tasks [54], limiting its applicability to large-scale or nuanced scenarios. (2) Recent work [3, 23, 56] has leveraged LLM-as-a-Judge to evaluate outputs and provide

optimization signals, advancing capabilities in complex tasks. However, the reliability of this paradigm heavily hinges on the judge model’s own capabilities and its susceptibility to systematic biases, where evaluations are confounded by various factors such as the presentation order of responses or a preference for certain linguistic styles [14, 24, 44]. (3) In response, research has shifted towards exploring Reinforcement Learning from Verifiable Rewards (RLVR)—a paradigm designed to enhance a model’s complex reasoning capabilities in domains like mathematics [11, 43, 51]. By leveraging ground-truth answers as reward signals, RLVR avoids human annotation, but its reliance on verifiable outcomes restricts it to tasks with definitive solutions. This makes it ill-suited for open-ended tasks such as multilingual translation, where multiple valid outputs exist and no single ground truth can capture all high-quality responses.

Notably, both paradigms share a fundamental limitation: dependence on external supervision—whether human annotations or pre-defined verifiable answers. This reliance constrains LLMs’ adaptability and scalability across diverse tasks, highlighting the need for self-supervised preference optimization mechanisms.

#### 2.2 Dual Learning

Dual learning enhances model performance by leveraging intrinsic task symmetry, where a primal task and its complementary dual task mutually provide supervision. He et al. [15] first introduced dual learning for machine translation, which uses bidirectional tasks (e.g., En→Zh and Zh→En) to generate pseudo-labels via back-translation [41], reducing reliance on parallel corpora—a breakthrough for low-resource language pairs.

This framework has since expanded to diverse domains:

- • Cross-modal tasks: DualGAN [25] frames image-to-text and text-to-image generation as dual tasks, enforcing cycle consistency to align visual and linguistic representations. Ren et al. [38] apply a similar principle to text-to-speech (TTS) and automatic speech recognition (ASR), enabling joint training with minimal paired data.
- • Knowledge reasoning: DualTKB [12] treats knowledge base path generation and natural language query parsing as symmetric tasks, improving factual consistency via bidirectional validation.
- • Reinforcement learning integration: Zhang et al. [55] designed policy gradient algorithms that transfer rewards between dual tasks, mitigating reward sparsity in low-supervision scenarios.

For LLMs, dual learning has enabled capability enhancement. Trans-Zero [58] uses back-translation to verify semantic preservation in multilingual generation. DualReflect [7] employs dual tasks (e.g., translation and back-translation) as structured feedback to refine output quality.

However, a critical limitation persists: existing methods require strict task duality where primal and dual tasks are mutually invertible (e.g., translation pairs). This restricts application to tasks with ambiguous or non-invertible dual counterparts (e.g., open-ended reasoning, creative writing). Our work addresses this by reframing dual learning as a preference optimization framework. Instead of relying on explicit task symmetry, we decompose inputs into known/unknown components to construct flexible dual tasks, enabling generalization across diverse tasks without rigid invertibility constraints.

### 3 Dual Learning-based Preference Optimization

In this section, we propose Dual Learning-based Preference Optimization (DuPO). Its core objective is to leverage the intrinsic relationships between tasks and their dual counterparts to generate self-supervised rewards, enabling LLMs to improve performance without relying on expensive human annotations or complex handcrafted rules.

- 3.1 Task Duality We begin by formalizing the task duality between a primal task and its dual counterpart.

- Definition 1. Let X be the input space and Y the output space. A primal task is a mapping Tp : X → Y, and a dual task is a mapping Td : Y → X. The pair (Tp,Td) is said to form a dual pair if they satisfy the

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

(a) Non-unique reconstruction of x from y

(b) Failure to reconstruct x given y

(c) Predicting subspaces B to preserve uniqueness

Figure 1 Challenges in Dual Learning and Solutions via Relaxed Duality Constraints. Dilemmas in applying dual learning — (a) Non-unique reconstruction of x from y breaks the closed-loop; (b) Failure to reconstruct x from y due to competence asymmetry. Resolutions by relaxing duality restrictions — (c) Predicting subspaces B to preserve uniqueness.

consistency principle:

∀x ∈ X, d x, Td(Tp(x)) ≤ ϵX,

where d(·) : X × X → R+ is a domain-specific distance metric, and ϵX ≥ 0 is a tolerance threshold that quantifies acceptable reconstruction errors in each space.

Leveraging this duality, we can construct a self-supervised reward to quantify the quality of a primal-task output. Given an input x ∈ X and its corresponding output y = Tp(x), we could define reward as

r(y) ∝ exp −λ · d x, Td(y) , (1)

where λ > 0 controls the sensitivity of the reward to reconstruction error. High-quality outputs maximize the expected reward E[r(y)] by preserving information that is recoverable through the duality cycle. This principle has been successfully applied in various domains, including machine translation [15, 58].

#### 3.2 Challenges in Dual Learning-based Optimization

While task duality offers a promising self-supervised paradigm, its application to LLM optimization is non-trivial, as it confronts two critical challenges that disrupt the closed-loop information flow.

- Challenge I: Limited Duality in Non-Mutually Implicative Tasks. The natural utility of task duality hinges on Tp and Td being mutually implicative — specifically, the output y of Tp contains sufficient information to reconstruct x via Td, and vice versa. This property holds for canonical tasks like machine translation, where y (a translation) preserves the semantic content of x (the source sentence), enabling Td (back-translation) to recover xˆ ≈ x.

However, most real-world tasks lack this mutual implicativity (Fig. 1a). Consider mathematical reasoning, where Tp solves a problem x (e.g., “A box contains 3 red and 5 blue balls; what is the total?”) to produce y = 8. Here, y (the total count) is insufficient to uniquely reconstruct x via Td, as 8 could answer infinitely many disparate questions, such as “What is 10 − 2?” or “What is the atomic number of Oxygen?”. This underdetermined relationship breaks the duality loop: Td cannot reliably recover x from y, making the self-supervised reward (based on xˆ ≈ x) untrustworthy. Such tasks thus require a redefinition of duality beyond direct input-output reversal.

- Challenge II: Bidirectional Competence Asymmetry. Even for mutually implicative tasks, duality optimization is sensitive to the bidirectional competence of the LLM — the performance gap between Tp and Td. If Tp is strong but Td is weak, Td may produce noisy xˆ that distorts the supervision signal (Fig. 1b). This asymmetry is particularly pronounced in LLMs, where extensive pretraining creates diverse but uneven capabilities across tasks, even within the same domain.

For instance, in machine translation optimization, let x = “The quick brown fox jumps over the lazy dog” (English) and y = “Der schnelle braune Fuchs springt über den faulen Hund” (correct German translation). A high-quality y should enable Td to back-translate to xˆ ≈ x. However, if Td struggles with nuanced vocabulary (e.g., “schnelle” → “fast” instead of “quick”), xˆ might be “The fast brown fox jumps over the lazy dog” — a divergence from x that erroneously penalizes y despite its correctness.

Using separate models for Tp and Td, as was common in traditional dual learning, merely sidesteps the challenge of intrinsic competence asymmetry [45, 46]. This imbalance is still arising from the distinct natures and complexities of the primal-dual tasks, destabilizing the self-supervised feedback loop.

#### 3.3 Generalized Duality Framework via Complementary Task

- To address the two-fold challenges of limited duality in non-mutually implicative tasks and bidirectional competence asymmetry, we propose a generalized duality that redefines task duality through complementary dependencies. It transcends traditional duality’s strict input-output reversal requirement by leveraging partial and stable dependencies between task components, enabling robust self-supervised rewarding even for tasks lacking inherent mutual implicativity.

Definition 2. Let the input space X of a primal task Tp be decomposed into two disjoint subspaces: Xk (known components) and Xu (unknown components), such that X = Xk ∪ Xu. The primal task Tp is a mapping

- Tp : X → Y that maps x ∈ X to an output space y ∈ Y. Its complementary dual task Tcd is a mapping that leverages y and the known component xk to reconstruct the unknown component xˆu ∈ Xu:

Tcd : (y,xk)  → xˆu. The pair (Tp,Tcd) is said to form a generalized dual pair if they satisfy the complementary consistency principle: ∀x ∈ X, y = Tp(x) : d xu,Tcd(y,xk) ≤ ϵ,

where d(·) : X × X → R+ is a domain-specific distance metric, and ϵ ≥ 0 is a tolerance threshold.

Leveraging this generalized duality, we can construct a self-supervised reward to quantify the preference of a primal-task output analogously to Def. 1. Given an input x ∈ X with decomposition x = (xk,xu) and its corresponding output y = Tp(x), the reward is defined as

r(y) ∝ exp(−λ · d(xu,Tcd(y,xk))), (2) where λ > 0 controls reward sensitivity.

Thanks to the generalized duality, we can explicitly use xk (known components) to constrain Tcd, enabling stable optimization even when y alone is insufficient. To highlight how generalized duality resolves the limitations (§3.2) of classic duality (§3.1), we examine a simple two-sum example:

Example 1: Generalized Duality Feedback for a Two-Sum Task: A + B The primal task Tp : y ← xu + xk is to compute the sum of two numbers, with its input and output as:

- – The input x is decomposed as x ← (A,B), where xk = A (a known number) and xu = B (an unknown number, without loss of generality).
- – The output y is the result of sum: C = A + B.

The complementary dual task Tcd : xu ← y − xk is designed to reconstruct the unknown component xu, using the primal output y (i.e. C) and the known xk (i.e. A):

xˆu ← B′ = C − A

Then, we can directly quantifies whether B (original unknown) and B′ (reconstructed unknown) are consistent as reward signal:

r(y) ∝ exp(−λ · I(B ̸= B′)).

Here, I(·) is an indicator function: it equals 0 if B = B′ (consistent) and 1 otherwise (inconsistent). This ensures the reward is maximized when B and B′ match, and reduced otherwise.

Remark 1. Compared to traditional dual learning, which suffers from strict mutual implicativity (i.e., y must fully encode x) and bidirectional competence asymmetry, our generalized duality framework offers three fundamental advantages:

- 1. Overcomes the Invertibility Constraint. By redesigning the dual objective from reconstructing the full

input x to only a selected unknown component xu, our framework fundamentally bypasses the stringent requirement of task symmetry. This relaxation is the key to unlocking dual learning for tasks that are inherently non-invertible, where the primal output does not contain sufficient information to recover the entire input.

- 2. Mitigates the Competence Asymmetry. The difficulty of the dual task is significantly reduced in two ways. First, the known component xk acts as a strong contextual anchor, constraining the solution space for reconstruction. Second, we can simply yet effectively select an xu that is not only feasibly reconstructible but also act as a faithful reward signal for the primal task’s solution quality (Appendix A). This directly addresses the “weak dual” pitfall and ensures the self-supervised reward is reliable and informative.
- 3. Enables Broad Applicability. It unlocks dual learning for a broad class of tasks previously considered unsuitable, including complex domains such as mathematical reasoning, code generation, and dialogue systems where input-output relationships are partial or conditional.

This generalized duality, therefore, provides a systematic way to overcome the traditional barriers of noninvertibility and competence asymmetry. Our case studies in Appendix D present concrete examples that illustrate how this process is applied in the multilingual translation and mathematical reasoning scenarios.

- 3.4 Preference Optimization

The core of our Dual Learning-based Preference Optimization (DuPO) framework is to optimize LLMs using duality-derived self-supervised rewards r(y), without external annotations. The objective is to maximize the expected reward based on the (complementary) dual task:

θ(y|x) [r(y)], (3)

J (θ) = Ey∼π

where πθ(y|x) denotes the LLM’s policy (parameterized by θ) for generating output y given input x = (xu,xk). The distance metric d(·) design is highly flexible and compatible with various rule-based metrics, enabling application across diverse tasks. For example, we could employ BLEU scores for multilingual translation which provide scores from 0 to 1, while for mathematical reasoning, we evaluate variable equality, yielding binary rewards.

Notably, DuPO is compatible with various reinforcement learning algorithms (e.g., PPO [39], ReMAX [28], REINFORCE++ [19]), we adopt Group Relative Policy Optimization (GRPO) [42] in our experiments—for its stability in high-dimensional parameter spaces (critical for LLMs) and compatibility with rule-based rewards.

### 4 Experiment

We validate the efficacy of DuPO on two representative tasks: multilingual translation and mathematical reasoning. Below, we detail the experimental setup, datasets, and evaluation metrics for each task, followed by key results.

#### 4.1 Experiment Setup

Base Model. We evaluate DuPO on a diverse set of strong and popular base models to demonstrate its effectiveness and robustness. For translation tasks, we employ Seed-X-7B-Instruct [9], one of the strongest open-source translation models. For mathematical reasoning, we select models of varying scales and capabilities, including small-scale yet powerful DeepSeek-R1-Distill-Qwen-1.5B [11] and its larger counterpart DeepSeekR1-Distill-Qwen-7B, both distilled from the state-of-the-art DeepSeek-R1. We also include Qwen3-4B [51],

Seed-X-7B-Instruct w/ DuPO

GPT-4o

DeepSeek-R1-0120

Google Translate

| |
|---|

| |
|---|

| |
|---|

Model BLEU COMET BLEURT Avg.

3.7

3.67

3.67 3.66 3.66 3.65 3.67 3.66 3.57

3.64 3.60

Qwen3-8B 21.69 84.82 65.81 57.44 Doubao-1.5-Thinking 26.19 87.87 71.66 61.91 Qwen3-235B-22B 28.37 88.76 73.91 63.68 DeepSeek-R1-0528 30.21 89.16 75.03 64.80

3.6

Score

3.5

3.44

3.4

3.28

3.3

Seed-X-7B-Instruct 28.76 86.96 72.62 62.78 w/ DuPO (ours) 30.31 89.09 74.57 64.66

3.2

En2XX Zh2XX Avg

Figure 2 Human Evaluation Scores (0-4) on the SeedX-Challenge for 14 Language Directions. DuPO achieves performance comparable to or even surpassing ultra-large models such as GPT-4o and DeepSeek-R1-0120, while significantly outperforming Google Translate.

- Table 1 Multilingual Translation Performance Across 756 Translation Directions in 28 Languages. DuPO significantly improves all metrics and performs comparably to its strong counterparts (DeepSeek models).

the latest strong small LLM, and the most capable open-source reasoning model, OpenReasoning-Nemotron7B [32]. These models represent strong and representative baselines within their respective model scales, ensuring comprehensive evaluation. Additionally, we also include some SOTA and impressive ultra-large models like Doubao-1.5/1.6-Thinking [5], Claude-Sonnet4-Thinking, and DeepSeek-R1 [11] for comparison.

Dataset. For translation tasks, we focus on 28 languages that are aligned with the language coverage of Seed-X, selecting 1,000 prompts for each language from a multilingual pre-training dataset to create our training prompt set. Additionally, we collect 7,000 parallel data entries across these specified languages to support our experiments from the dev set of Flores-200 [33]. For mathematical reasoning tasks, we utilize a mixture of publicly available mathematics question datasets1. These datasets encompass diverse sources and are commonly used for synthesizing supervised fine-tuning data with ultra-large LLMs or conducting reinforcement learning with oracle labels, covering various subjects of competition-level mathematical problems, logic puzzles, and other reasoning tasks.

Benchmarks. To comprehensively evaluate the effectiveness of DuPO, we conduct extensive experiments using the following test sets:

- • Multilingual Translation: For multilingual translation, we construct our test set by randomly selecting 50 samples for each of the 756 translation directions (among 28 languages) from the testset of Flores2, resulting in a total of 37,800 samples. We will release this dataset for convenient comparison. We employ BLEU [36], COMET [37], and BLEURT [40] as evaluation metrics. Additionally, we conduct human evaluation on Seed-X-Challenge [9]3, a challenging benchmark designed to test the boundaries of LLMs’ translation capabilities with diverse linguistic elements across multiple domains. Human experts assess accuracy, fluency, and idiomaticity, scoring translations from Chinese or English to seven languages on a 0-4 scale (higher score denotes better translation quality).
- • Mathematical Reasoning: We evaluate our approach on multiple benchmarks, including AMC23 [31], AIME24 [4], and AIME25 [4], to assess performance in standardized contest environments. For each problem, we sample 32 responses using a temperature of 0.8 and a maximum reasoning budget of 32,000 tokens, then report the average accuracy (Avg@32).

Ultra-large models like DeepSeek-R1 and Doubao-1.6-thinking are accessed via their official APIs. More details about training are provided in Appendix C.

#### 4.2 Main Results

##### 4.2.1 DuPO Boosts LLM’s Performance on Various Tasks

DuPO achieves strong performance on diverse tasks, including multilingual translation and mathematical reasoning. On multilingual translation, DuPO elevates the base model to a state-of-the-art performance level, rivaling and even surpassing significantly ultra large LLM. As detailed in Table 1, applying DuPO

1More details on math data preparation can be found in Appendix B.

- 2https://huggingface.co/datasets/openlanguagedata/flores_plus
- 3https://github.com/ByteDance-Seed/Seed-X-7B/tree/main/challenge_set

Model AMC23 AIME24 AIME25 Average

DeepSeek-R1-0120 97.7 79.8 70.0 82.5 Claude-Sonnet4-Thinking 97.5 82.5 70.0 83.3

- Doubao-1.5-Thinking 99.4 86.3 73.3 86.3
- Doubao-1.6-Thinking 98.8 88.4 83.4 90.2 DeepSeek-R1-0528 99.4 91.4 87.5 92.8

DeepSeek-R1-Distill-Qwen-1.5B 67.5 20.0 20.0 35.8 w/ DuPO (ours) 72.5 30.0 26.7 39.7 (+3.9)

DeepSeek-R1-Distill-Qwen-7B 85.0 56.7 36.7 59.5 w/ DuPO (ours) 90.0 63.3 40.0 64.4 (+4.9)

Qwen3-4B 95.0 70.0 66.7 77.2 w/ DuPO (ours) 97.5 83.3 70.0 83.6 (+6.4)

OpenReasoning-Nemotron-7B 95.0 83.3 73.3 83.9 w/ DuPO (ours) 97.5 83.3 90.0 90.3 (+6.4)

- Table 2 Mathematical Reasoning Performances (%) on Representative Benchmarks. DuPO significantly improves the performances across models with varying base capabilities, enabling Qwen3-4B to outperform DeepSeek-R1-0120 and OpenReasoning-Nemotron-7B to achieve SOTA performance.

to the Seed-X-7B-Instruct model boosts its performance by 1.55, 2.13, and 1.95 across three automatic evaluation metrics, reaching an average score of 64.66. This performance even surpasses that of current SOTA closed-source ultra-large language models, such as Doubao1.5-thinking (+2.75) and Qwen3-235B-22B (+0.98), and is on par with the performance of the latest DeepSeek-R1. As shown in Figure 2, DuPO demonstrates remarkable performance, achieving results comparable to state-of-the-art ultra-large models such as GPT-4o and DeepSeek-R1. Moreover, DuPO substantially outperforms widely-used commercial closed-source systems like Google Translate, showcasing a clear advantage in translation quality as perceived by human evaluators.

On mathematical reasoning, the results in Table 2 clearly demonstrate that DuPO yields consistent and significant performance improvements across all models at different scales and baseline reasoning ability. On the most powerful OpenReasoning-Nemotron-7B model, applying DuPO increased the average score from 83.9% to 90.3%, achieving impressive performance. This trend of significant gains continues on the mid-sized Qwen3-4B model, which saw its average score boosted by 6.4 points from 77.2% to 83.6%, even surpassing the ultra-large model DeepSeek-R1-0120. The approach remains remarkably effective on DeepSeek’s distilled models as well. Even on DeepSeek-R1-Distilled-Qwen-1.5B, the least reasoning capability among the strong baselines, we still achieved a 3.9-point increase in average accuracy. This directly demonstrates that DuPO is sufficiently robust and stable to enhance the mathematical reasoning capabilities of models consistently. Our framework’s robust performance is further validated by concrete examples in multilingual translation and mathematical reasoning (see case studies in Appendix D).

##### 4.2.2 DuPO Scales to Various Backbones Effectively

To validate the robustness and generalization of our proposed DuPO framework, we extend our evaluation to the LlaMA architectural family. Our experiments are conducted on two LlaMA architectural models: LlaMA-3.1-8B [13] and OctoThinker-8B-Hybrid-Base [47], the latter of which has undergone middle training on mathematical reasoning knowledge. Considering the significant difference of model ability, we select two benchmarks of moderate difficulty, AMC23 [31] and MATH500 [18]. For a fair comparison, all models are finetuned using identical training data and settings. Results are listed in Table 3.

As seen, DuPO’s effectiveness is not tied to a specific model architecture; it serves as a robust and generalizable enhancement, delivering significant improvements to diverse backbones regardless of their initial reasoning proficiency. DuPO lifts the average score of LlaMA-3.1-8B to 32.1%, a +24.0 percentage-point gain over the vanilla model, and surpasses SimpleRL-Zoo [53] (which relies on oracle-labeled answers during training) by 13.1%. When applied to the OctoThinker-8B-Hybrid-Base [47], our DuPO approach yields even more

Model AMC23 MATH500 Average

LlaMA-3.1-8B 2.5 13.6 8.1

w/ SimpleRL-Zoo 15.0 23.0 19.0 w/ DuPO (ours) 20.0 44.2 32.1

OctoThinker-8B-Hybrid-Base 5.0 42.6 23.8 w/ DuPO (ours) 55.0 70.0 62.5

- Table 3 Performances (%) of DuPO on Different Backbone Models. DuPO even surpasses SimpleRLZoo, which utilizes labeled answers as reward. DuPO’s potential is further exemplified by OctoThinker, which underwent additional middle training.

AMC AIME24 AIME25 Forward Acc

60

Accuracy(%)

40

| |
|---|

20

| |
|---|

| |
|---|

0

50 100 150 200 250 300

Steps

Figure 3 Training Progress of DuPO on Qwen34B-Base. The performance consistently improves on the primal task and the benchmarks.

impressive performance improvements of +50.0 on AMC23 and +27.4 on MATH500, achieving an average performance of 62.5.

##### 4.2.3 DuPO Incentivizes Reasoning Capability on Base Model

We further demonstrate that our DuPO framework can directly elicit and enhance complex reasoning capabilities from a base model. We apply DuPO directly to a base model, without preliminary supervised fine-tuning (SFT) stage activating the reasoing ability. We track the learning dynamics by simultaneously collecting the primal task accuracy (“Forward Acc”) on the training set and its generalization performance on three distinct, unseen challenge test set: AMC23, AIME24, and AIME25.

We can see from Figure 3 that DuPO provides a stable and effective pathway to awaken and generalize the latent reasoning abilities of a base model, validating its utility as a powerful training methodology. Specifically, the training dynamics reveal a clear and substantial improvement on the primal task, with the “Forward Acc” soaring from a nascent 15.2% to 56.5%. This upward trajectory provides direct evidence that the reward signal derived from our dual-task serves as an effective guide for enhancing the model’s reasoning. More importantly, this acquired skill demonstrates robust generalization. Performance on the unseen test set AMC23 leaped from 20% to 70%, with similarly significant gains observed on the AIME24 and AIME25 datasets.

##### 4.2.4 DuPO Scales Reasoning during Inference without Training

Beyond serving as a reward signal for RL training, the DuPO mechanism can be naturally applied as a training-free, inference-time reranking strategy to improve the reasoning capabilities of any LLM. The process unfolds in three stages: 1) Similar to the rollout stage during RL process, we could prompt any given policy model to generate diverse reasoning trajectories. 2) For each candidate trajectory, we use its final answer to ask the policy model to solve the corresponding dual question automatic constructed without accessing labeled answer. We could apply more computation by performing K (K = 8 in our experiments) sampling runs on each dual question for a more reliable reward estimate, a practice distinct from RL training. 3) Finally, for each test set question, we select the trajectory with the highest backward accuracy on its dual questions as the final output.

As presented in Table 4, the experimental results demonstrate that DuPO provides accurate reward signals, effectively guiding models towards correct reasoning, serving as an efficient approach for scaling reasoning capabilities even without training. On the two challenging AIME benchmarks, applying DuPO as a reranking method improves the average performance of Qwen3-4B by 9.3 points, elevating its accuracy from 68.4% to 77.7% without any additional training. Notably, the DuPO-enhanced Qwen3-4B surpasses DeepSeek-R1 and Claude-Sonnet4-Thinking (77.7% vs. 74.9%/76.3% on average). The impact on DeepSeek-R1-Distill-Qwen1.5B s even more pronounced, with an 18.7 point increase (20.0% to 38.7%).

Model AIME24 AIME25 Average

DeepSeek-R1-0120 79.8 70.0 74.9 Claude-Sonnet4-Thinking 82.5 70.0 76.3

DeepSeek-R1-Distill-Qwen-1.5B 20.0 20.0 20.0 w/ DuPO rewarding 53.3 24.1 38.7 (+18.7)

Qwen3-4B 70.0 66.7 68.4 w/ DuPO rewarding 86.6 68.9 77.7 (+9.3)

- Table 4 Inference-Time Scaling on Mathematical Reasoning Using DuPO Rewarding (Backward Acc) for Reranking. Our method improves the performance of policy models with varying base ability, without requiring additional training.

AMC23 AIME24 AIME25 Avg

20

30

40

50

60

70

Acc(%)

67.5

20.0 20.0

35.8

72.5

30.0

26.7

43.1

65.0

26.7 26.7

39.5

1.5B Baseline

DuPO DuPO w/o filter

| |
|---|

AMC23 AIME24 AIME25 Avg

65

70

75

80

85

90

95

100

Acc(%)

95.0

70.0

66.7

77.2

97.5

83.3

73.3

84.7

95.0

73.3

70.0

79.3

4B Baseline DuPO DuPO w/o filter

| |
|---|

| |
|---|

Figure 4 Performance Ablation of DeepSeek-R1-Distill-Qwen-1.5B/Qwen3-4B on Mathematical Reasoning. Our unknown component selection strategy reduces training noise and improves these models’ performance across three benchmarks.

- 4.3 Effects of Task Duality

To thoroughly investigate the effectiveness of our proposed framework and validate how our unknown component selection strategy contributes to achieving better task duality, we conduct an ablation study by maintaining identical experimental settings while removing the unknown component selection mechanism from our dual framework.

As illustrated in Figure 4, the results showcase the efficacy of our approach in resolving duality issues. For the 1.5B model, DuPO achieves a remarkable 7.3 percentage point improvement over the baseline. Notably, when we remove data filtering, thereby introducing poorer duality, we observe a significant 3.6 percentage point drop in performance. This pattern not only persists with stronger models. In the case of the 4B model, the benefits of our method become even more pronounced, outperforming the poorer duality variant by an impressive 5.4 points on average. These consistent and substantial improvements across various model sizes provide strong empirical evidence that our component selection strategy is a crucial component of the dual framework, effectively ensuring high-quality task duality and thereby enabling the framework to achieve superior performance.

- 5 Conclusion

We introduce DuPO, a dual learning-based preference optimization framework that eliminates the need for costly human annotations and handcrafted rewards in LLM training. At its core, DuPO’s innovation lies in a generalized duality framework that decomposes and reconstructs input spaces into known and unknown components, addressing critical limitations of traditional dual learning and preference optimization paradigms. Empirical validation across two diverse, high-stakes tasks confirms DuPO’s versatility and effectiveness. In mathematical reasoning, DuPO consistently improves performance across model scales from 1.5B to 7B

parameters, with notable gains of 6.4% average accuracy improvement of three benchmarks. For multilingual translation, DuPO elevates the 7B-parameter Seed-X model to performance levels comparable to much larger state-of-the-art models, boosting COMET scores by up to 2.13 points across 28 languages and 756 translation directions. Additionally, DuPO serves as an effective training-free reranking mechanism, enabling smaller models to outperform larger counterparts with up to 9.3 points improvement, bypassing the need for expensive parameter scaling.

DuPO’s model-agnostic design and broad task applicability position it as a scalable solution for annotationefficient LLM development. By harnessing intrinsic task structure to generate self-supervised feedback, it moves beyond the constraints of human supervision and rigid reward engineering—paving the way for more autonomous, adaptable, and cost-effective language model optimization.

### 6 Limitations

Despite the promising results, we acknowledge several limitations of our work that present avenues for future research. First, unknown components selection for mathematical reasoning introduces the computational overhead. While this step is crucial for ensuring the quality of the self-supervised reward signal, developing more efficient or even learnable filtering mechanisms could enhance the scalability and practical applicability of DuPO. Second, our empirical validation is primarily conducted on models of moderate scale. Although DuPO demonstrates consistent improvements across various model sizes, its scalability and effects on significantly larger models remain an open question. Finally, while we demonstrate DuPO’s efficacy on various tasks like mathematical reasoning and multilingual translation, its application to more open-ended and creatively demanding tasks, such as open-ended instruction-following, requires further exploration.

### Acknowledgments

We extend our sincere gratitude to our colleagues at ByteDance, including Qian Cao4, Zhichao Huang, Liyan Kang, Ningxin Peng, Xinghua Qu, Ming Tu, Xiangpeng Wei, Rong Ye, Runsheng Yu, and Zaixiang Zheng, for their valuable advice and insightful discussions, and to Meng Yang and Evaluation Team for their help with the translation evaluation.

4Qian has already left ByteDance.

### References

- [1] Mistral AI. Mistral large. https://mistral.ai/news/mistral-large-2407, 2024.
- [2] Alon Albalak, Duy Phung, Nathan Lile, Rafael Rafailov, Kanishk Gandhi, Louis Castricato, Anikait Singh, Chase Blagden, Violet Xiang, Dakota Mahan, and Nick Haber. Big-Math: A large-scale, high-quality math dataset for reinforcement learning in language models, 2025. URL https://arxiv.org/abs/2502.17387.
- [3] Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, and Jared Kaplan. Constitutional ai: Harmlessness from ai feedback, 2022. URL https://arxiv.org/abs/2212.08073.
- [4] Mislav Balunović, Jasper Dekoninck, Ivo Petrov, Nikola Jovanović, and Martin Vechev. MathArena: Evaluating llms on uncontaminated math competitions, 2025.
- [5] ByteDance Seed Team. Seed1.5-Thinking: Advancing superb reasoning models with reinforcement learning, 2025. URL https://arxiv.org/abs/2504.13914.
- [6] Chris Callison-Burch, Miles Osborne, and Philipp Koehn. Re-evaluating the role of Bleu in machine translation research. In Diana McCarthy and Shuly Wintner, editors, 11th Conference of the European Chapter of the Association for Computational Linguistics, pages 249–256, Trento, Italy, April 2006. Association for Computational Linguistics. URL https://aclanthology.org/E06-1032/.

- [7] Andong Chen, Lianzhang Lou, Kehai Chen, Xuefeng Bai, Yang Xiang, Muyun Yang, Tiejun Zhao, and Min Zhang. DUAL-REFLECT: Enhancing large language models for reflective translation through dual learning feedback mechanisms, 2024. URL https://arxiv.org/abs/2406.07232.
- [8] Yang Chen, Zhuolin Yang, Zihan Liu, Chankyu Lee, Peng Xu, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. AceReason-Nemotron: Advancing math and code reasoning through reinforcement learning. arXiv preprint arXiv:2505.16400, 2025.

- [9] Shanbo Cheng, Yu Bao, Qian Cao, Luyang Huang, Liyan Kang, Zhicheng Liu, Yu Lu, Wenhao Zhu, Jingwen Chen, Zhichao Huang, Tao Li, Yifu Li, Huiying Lin, Sitong Liu, Ningxin Peng, Shuaijie She, Lu Xu, Nuo Xu, Sen Yang, Runsheng Yu, Yiming Yu, Liehao Zou, Hang Li, Lu Lu, Yuxuan Wang, and Yonghui Wu. Seed-X: Building strong multilingual translation llm with 7b parameters, 2025. URL https://arxiv.org/abs/2507.13618.
- [10] DeepMind. Gemini 2.5. https://deepmind.google/technologies/gemini/, 2025. Accessed: 2025-04-18.
- [11] DeepSeek-AI. DeepSeek-R1: Incentivizing reasoning capability in llms via reinforcement learning, 2025.
- [12] Pierre L. Dognin, Igor Melnyk, Inkit Padhi, Cícero Nogueira dos Santos, and Payel Das. Dualtkb: A dual learning bridge between text and knowledge base. In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu, editors, Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, pages 8605–8616. Association for Computational Linguistics, 2020. doi: 10.18653/V1/2020.EMNLP-MAIN.694. URL https://doi.org/10.18653/v1/2020.emnlp-main.694.

- [13] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. ArXiv preprint, abs/2407.21783, 2024. URL https://arxiv.org/abs/2407.21783.

- [14] Arnav Gudibande, Eric Wallace, Charlie Snell, Xinyang Geng, Hao Liu, Pieter Abbeel, Sergey Levine, and Dawn Song. The false promise of imitating proprietary llms. CoRR, abs/2305.15717, 2023. doi: 10.48550/ARXIV.2305.

15717. URL https://doi.org/10.48550/arXiv.2305.15717.

- [15] Di He, Yingce Xia, Tao Qin, Liwei Wang, Nenghai Yu, Tie-Yan Liu, and Wei-Ying Ma. Dual learning for machine translation. In Daniel D. Lee, Masashi Sugiyama, Ulrike von Luxburg, Isabelle Guyon, and Roman Garnett, editors, Advances in Neural Information Processing Systems 29: Annual Conference on Neural Information Processing Systems 2016, December 5-10, 2016, Barcelona, Spain, pages 820–828, 2016. URL https://proceedings.neurips. cc/paper/2016/hash/5b69b9cb83065d403869739ae7f0995e-Abstract.html.

- [16] Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, Siyuan Li, Liang Zeng, Tianwen Wei, Cheng Cheng, Yang Liu, and Yahui Zhou. Skywork open reasoner series. https://capricious-hydrogen-41c.notion.site/ Skywork-Open-Reaonser-Series-1d0bc9ae823a80459b46c149e4f51680, 2025. Notion Blog.
- [17] Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, Siyuan Li, Liang Zeng, Tianwen Wei, Cheng Cheng, Yang Liu, and Yahui Zhou. Skywork open reasoner series, 2025. Notion Blog.
- [18] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. ArXiv preprint, abs/2103.03874,

2021. URL https://arxiv.org/abs/2103.03874.

- [19] Jian Hu, Jason Klein Liu, Haotian Xu, and Wei Shen. REINFORCE++: An efficient rlhf algorithm with robustness to both prompt and reward models, 2025. URL https://arxiv.org/abs/2501.03262.
- [20] Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Open-reasonerzero: An open source approach to scaling up reinforcement learning on the base model, 2025. URL https: //arxiv.org/abs/2503.24290.
- [21] Yunjie Ji, Sitong Zhao, Xiaoyu Tian, Haotian Wang, Shuaiting Chen, Yiping Peng, Han Zhao, and Xiangang Li. How difficulty-aware staged reinforcement learning enhances llms’ reasoning capabilities: A preliminary experimental study, 2025. URL https://arxiv.org/abs/2504.00829.
- [22] Ruipeng Jia, Yunyi Yang, Yongbo Gai, Kai Luo, Shihao Huang, Jianhe Lin, Xiaoxi Jiang, and Guanjun Jiang. Writing-zero: Bridge the gap between non-verifiable tasks and verifiable rewards, 2025. URL https: //arxiv.org/abs/2506.00103.
- [23] Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Lu, Thomas Mesnard, Colton Bishop, Victor Carbune, and Abhinav Rastogi. RLAIF vs. RLHF: Scaling reinforcement learning from human feedback with ai feedback. In International Conference on Machine Learning, 2023. URL https://api.semanticscholar.org/CorpusID: 261493811.

- [24] Dawei Li, Bohan Jiang, Liangjie Huang, Alimohammad Beigi, Chengshuai Zhao, Zhen Tan, Amrita Bhattacharjee, Yuxuan Jiang, Canyu Chen, Tianhao Wu, Kai Shu, Lu Cheng, and Huan Liu. From generation to judgment: Opportunities and challenges of llm-as-a-judge. CoRR, abs/2411.16594, 2024. doi: 10.48550/ARXIV.2411.16594. URL https://doi.org/10.48550/arXiv.2411.16594.

- [25] Hongsheng Li, Xiaogang Wang, and Jianchao Yang. DualGAN: Unsupervised dual learning for image-to-image translation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017.

- [26] Jiahuan Li, Shanbo Cheng, Shujian Huang, and Jiajun Chen. Mt-patcher: Selective and extendable knowledge distillation from large language models for machine translation, 2024. URL https://arxiv.org/abs/2403.09522.
- [27] Jiahuan Li, Hao Zhou, Shujian Huang, Shanbo Cheng, and Jiajun Chen. Eliciting the translation ability of large language models via multilingual finetuning with translation instructions. Transactions of the Association for Computational Linguistics, 12:576–592, 2024. doi: 10.1162/tacl_a_00655. URL https://aclanthology.org/ 2024.tacl-1.32/.

- [28] Ziniu Li, Tian Xu, Yushun Zhang, Zhihang Lin, Yang Yu, Ruoyu Sun, and Zhi-Quan Luo. Remax: A simple, effective, and efficient reinforcement learning method for aligning large language models. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=Stn8hXkpe6.

- [29] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. ArXiv preprint, abs/2412.19437, 2024. URL https://arxiv.org/abs/2412.19437.

- [30] Chris Yuhao Liu, Liang Zeng, Yuzhen Xiao, Jujie He, Jiacai Liu, Chaojie Wang, Rui Yan, Wei Shen, Fuxiang Zhang, Jiacheng Xu, Yang Liu, and Yahui Zhou. Skywork-Reward-V2: Scaling preference data curation via human-ai synergy. arXiv preprint arXiv:2507.01352, 2025.

- [31] MAA. Amc 2023 problems, 2023. URL https://artofproblemsolving.com/wiki/index.php/2023_AMC_12A_ Problems. Accessed: 2025-05-11.

- [32] Ivan Moshkov, Darragh Hanley, Ivan Sorokin, Shubham Toshniwal, Christof Henkel, Benedikt Schifferer, Wei Du, and Igor Gitman. AIMO-2 Winning Solution: Building state-of-the-art mathematical reasoning models with openmathreasoning dataset, 2025. URL https://arxiv.org/abs/2504.16891.
- [33] NLLB Team. Scaling neural machine translation to 200 languages. Nature, 630(8018):841–846, 2024. ISSN 1476-4687. doi: 10.1038/s41586-024-07335-x. URL https://doi.org/10.1038/s41586-024-07335-x.

- [34] OpenAI. Hello gpt-4o. https://openai.com/index/hello-gpt-4o/, 2023. Accessed: 2025-04-18.
- [35] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback, 2022. URL https://arxiv.org/abs/2203.02155.
- [36] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Pierre Isabelle, Eugene Charniak, and Dekang Lin, editors, Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA, July 2002. Association for Computational Linguistics. doi: 10.3115/1073083.1073135. URL https: //aclanthology.org/P02-1040/.

- [37] Ricardo Rei, Craig Stewart, Ana C Farinha, and Alon Lavie. COMET: A neural framework for mt evaluation,

2020. URL https://arxiv.org/abs/2009.09025.

- [38] Yi Ren, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, and Tie-Yan Liu. Almost unsupervised text to speech and automatic speech recognition, 2020. URL https://arxiv.org/abs/1905.06791.
- [39] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. CoRR, abs/1707.06347, 2017. URL http://arxiv.org/abs/1707.06347.

- [40] Thibault Sellam, Dipanjan Das, and Ankur Parikh. BLEURT: Learning robust metrics for text generation. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault, editors, Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7881–7892, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.704. URL https://aclanthology.org/2020.acl-main.704/.

- [41] Rico Sennrich, Barry Haddow, and Alexandra Birch. Improving neural machine translation models with monolingual data. CoRR, abs/1511.06709, 2015. URL http://arxiv.org/abs/1511.06709.

- [42] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. DeepseekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

- [43] Kimi Team. Kimi k1.5: Scaling reinforcement learning with llms. ArXiv, abs/2501.12599, 2025. URL https: //api.semanticscholar.org/CorpusID:275789974.

- [44] Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Lingpeng Kong, Qi Liu, Tianyu Liu, and Zhifang Sui. Large language models are not fair evaluators. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9440–9450, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.511. URL https://aclanthology.org/2024.acl-long.511/.

- [45] Yijun Wang, Yingce Xia, Li Zhao, Jiang Bian, Tao Qin, Guiquan Liu, and Tie-Yan Liu. Dual transfer learning for neural machine translation with marginal distribution regularization. In Sheila A. McIlraith and Kilian Q. Weinberger, editors, Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, (AAAI-18), the 30th innovative Applications of Artificial Intelligence (IAAI-18), and the 8th AAAI Symposium on Educational Advances in Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February 2-7, 2018, pages 5553–5560. AAAI Press, 2018. doi: 10.1609/AAAI.V32I1.11999. URL https://doi.org/10.1609/aaai.v32i1.11999.

- [46] Yisen Wang, Xingjun Ma, James Bailey, Jinfeng Yi, Bowen Zhou, and Quanquan Gu. On the convergence and robustness of adversarial training, 2022. URL https://arxiv.org/abs/2112.08304.
- [47] Zengzhi Wang, Fan Zhou, Xuefeng Li, and Pengfei Liu. Octothinker: Mid-training incentivizes reinforcement learning scaling, 2025. URL https://arxiv.org/abs/2506.20512.
- [48] Zhilin Wang, Jiaqi Zeng, Olivier Delalleau, Daniel Egert, Ellie Evans, Hoo-Chang Shin, Felipe Soares, Yi Dong, and Oleksii Kuchaiev. HelpSteer3: Human-annotated feedback and edit data to empower inference-time scaling in open-ended general-domain tasks. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher

- Pilehvar, editors, Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 25640–25662, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.1246. URL https://aclanthology.org/ 2025.acl-long.1246/.
- [49] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. ArXiv preprint, abs/2412.15115, 2024. URL https://arxiv.org/abs/2412.15115.

- [50] An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu, Xingzhang Ren, and Zhenru Zhang. Qwen2.5-Math Technical Report: Toward mathematical expert model via self-improvement, 2024. URL https://arxiv.org/abs/2409.12122.
- [51] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.
- [52] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Weinan Dai, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. DAPO: An open-source llm reinforcement learning system at scale, 2025.
- [53] Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild, 2025. URL https: //arxiv.org/abs/2503.18892.
- [54] Michael JQ Zhang, Zhilin Wang, Jena D. Hwang, Yi Dong, Olivier Delalleau, Yejin Choi, Eunsol Choi, Xiang Ren, and Valentina Pyatkin. Diverging preferences: When do annotators disagree and do models know?, 2024. URL https://arxiv.org/abs/2410.14632.
- [55] Saizheng Zhang, Xiujun Li, Le Song, Tao Wang, Dongyan Yang, and Ming Zhou. Deep reinforcement learning with dual learning for dialogue generation. In AAAI Conference on Artificial Intelligence, 2018.

- [56] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llmas-a-judge with mt-bench and chatbot arena. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/ 91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html.

- [57] Wenhao Zhu, Hongyi Liu, Qingxiu Dong, Jingjing Xu, Shujian Huang, Lingpeng Kong, Jiajun Chen, and Lei Li. Multilingual machine translation with large language models: Empirical results and analysis, 2024. URL https://arxiv.org/abs/2304.04675.
- [58] Wei Zou, Sen Yang, Yu Bao, Shujian Huang, Jiajun Chen, and Shanbo Cheng. TRANS-ZERO: Self-play incentivizes large language models for multilingual translation without parallel data. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Findings of the Association for Computational Linguistics: ACL 2025, pages 12337–12347, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.637. URL https://aclanthology.org/2025.findings-acl.637/.

## Appendix

### A Construction of Dual Question in Math Reasoning

We propose a simple approach for construction of dual question of mathematical reasoning. The algorithm operates on mathematical expressions and performs the following key steps:

- 1. Pattern Recognition and Exclusion: The algorithm first identifies numerical candidates within the

expression while excluding numbers in specific contexts: subscripts (x1,x2), inequality constraints (x ≤ 5), common exponential bases (2n,10k), and function arguments (f(3)).

- 2. Variable Generation and Replacement: For each valid numerical candidate, the system generates a

unique variable identifier of the form Variable{str} where str is a randomly generated lowercase string. The original number is then substituted with this variable.

- 3. Question Generation of Dual Task: Using the transformed expression and the original answer, the algorithm constructs inverse problems following templates such as: “Given that the correct answer is {answer}, determine the value of {variable}.”

This methodology enables systematic generation of problem variants while preserving mathematical validity and semantic coherence. From a single primal question, multiple dual questions can be derived. To ensure that these dual questions robustly satisfy the properties of duality, we filter the candidates using the following two principles:

- 1. Answerability of the Dual Question: For the set of sampled answers collected for a given primal question, at least one answer must be capable of correctly solving the corresponding dual question.
- 2. Uniqueness of the Correct Answer: Among the same set of sampled answers, at most one should correctly answer the dual question.

Taken together, these two principles ensure that for any selected dual question, there is one and only one correct answer within the pool of candidate solutions for the primal task. This establishes the one-to-one correspondence necessary for generating a reliable self-supervised reward signal.

### B Math RL Dataset Preparation

Our dataset preparation process began with the collection of 1,815,942 prompts from various publicly available datasets [2, 8, 16, 21]. After deduplication, we obtained 318,649 primal questions and generated 1,059,671 dual questions through our designed steps as discussed above. After that, we employed Qwen2.5-7B-Instruct [49] to sample 32 candidate answers for each primal question and then prompted it to answer the corresponding dual question based on these candidates. Subsequently, we rigorously filtered out all dual questions that failed to meet our predefined principles above. We repeated this sampling and filtering process with Qwen3-4B [51], this time with 8 candidate answers per question. The resulting collection of high-quality, diverse mathematical questions along with corresponding dual questions formed our final RL training set, providing a robust foundation for our reinforcement learning tasks in the mathematical domain.

### C Experiment Details

We presents more details about our training as follows: for the training process, we use a train batch size of 512, mini batch size of 32, sampling temperature of 1.0, and 16 rollouts per prompt, with a learning rate of 1e-6 and gradient clipping set to 1.0. For translation tasks, we set the maximum input length to 2,048 tokens and output length to 4,096 tokens. For mathematical tasks, we use the same input length but extend the maximum output length to 30,000 tokens.

Scenario 1: DuPO on Mathematical Reasoning Primal Task Let △ABC have circumcenter O and incenter I with IA ⊥ OI, circumradius 13, and

inradius 6. Find AB · AC. (Correct Answer: 468)

- Dual Task #1 Let △ABC have circumcenter O and incenter I with IA ⊥ OI, circumradius Vsk, and inradius 6. Find AB · AC. Check your work: If the solution for above question is

|boxed answer|
|---|

, what must Vsk have been?

- Dual Task #2 Let’s examine: Let △ABC have circumcenter O and incenter I with IA ⊥ OI, circumradius 13, and inradius Vrj. Find AB · AC. When the solution for above

|boxed answer|
|---|

question is

, what’s the corresponding Vrj?

Candidates Answer: 468 Backward Accuracy: 69.1%

Answer: 108 Backward Accuracy: 0% Answer: 312 Backward Accuracy: 0%

Scenario 2: DuPO on Machine Translation (MT) Primal Task Translate to Chinese: As knowledge of Greek declined, the West found itself cut off

from its Greek philosophical and scientific roots.

Reference 随着希腊知识的衰落，西方脱离了其希腊哲学和科学根源。

- Primal MT #1 随着希腊语知识的衰落，西方发现自己与希腊的哲学和科学根源失去了联系。(BLEU: 45.85)

- Dual MT #1 As knowledge of Greek declined, the West found itself cut off from its philosophical and scientific roots in Greece.(BLEU: 82.07)

Primal MT #2 随着对希腊语的了解逐渐消失，西方发现自己与希腊哲学和科学根源隔绝开

来。(BLEU: 28.65)

- Dual MT #2 As understanding of the Greek language gradually fades, the West finds itself cut off from the roots of Greek philosophy and science.(BLEU: 16.11)

Table 5 Case Studies of DuPO on Mathematical Reasoning and Machine Translation. DuPO validates each candidate’s quality through a corresponding dual task, reliably identifies the superior solution over inferior ones.

### D Case Study

To illustrate the efficacy of our DuPO approach, we present two representative scenarios in Table 5 that demonstrate how DuPO provides a reliable reward signal across diverse domains.

- Scenario 1: Mathematical Reasoning Validation. In mathematical reasoning, DuPO derives dual task questions from the primal task question where key numerical parameters are replaced with variables, and the model tries to work backwards conditioned on candidate answers. When given a geometry problem about triangle properties, three candidate answers are sampled: 468, 108, and 312. DuPO automatically derives two dual questions by replacing the circumradius (13) and inradius (6) with variables, asking the model to deduce these values from the proposed answer. The candidate answer 468 achieves 69.1% accuracy on dual task, while the incorrect answers (108 and 312) totally fail to answer the dual task.
- Scenario 2: Machine Translation Quality Assessment. For translation tasks, DuPO leverages reverse direction translation as the dual task to evaluate translation quality. Given an English sentence about Greek philosophical decline, two Chinese translation candidates are generated and subsequently back-translated to English. The first translation achieves a BLEU score of 45.85 in the forward direction and 82.07 in the back-translation, demonstrating semantic preservation and translation fidelity. In contrast, the second candidate shows degraded performance with BLEU scores of 28.65 and 16.11, respectively, indicating semantic drift and poor translation quality. These case studies validate DuPO’s core hypothesis: high-quality solutions maintain consistent information

###### across dual task formulations, while inferior solutions exhibit significant degradation. This dual validation mechanism provides a robust framework for automatic quality assessment without requiring ground truth labels.

