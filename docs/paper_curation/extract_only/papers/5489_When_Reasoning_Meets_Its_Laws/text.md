# arXiv:2512.17901v1[cs.AI]19Dec2025

## WHEN REASONING MEETS ITS LAWS

Junyu Zhang 1∗‡ Yifan Sun 1∗ Tianang Leng 3∗ Jingyan Shen 4∗ Liu Ziyin 25† Paul Pu Liang 2† Huan Zhang 1†

1University of Illinois Urbana-Champaign 2Massachusetts Institute of Technology 3University of Pennsylvania 4New York University 5NTT Research

ABSTRACT

Despite the superior performance of Large Reasoning Models (LRMs), their reasoning behaviors are often counterintuitive, leading to suboptimal reasoning capabilities. To theoretically formalize the desired reasoning behaviors, this paper presents the Laws of Reasoning (LORE), a unified framework that characterizes intrinsic reasoning patterns in LRMs. We first propose compute law with the hypothesis that the reasoning compute should scale linearly with question complexity. Beyond compute, we extend LORE with a supplementary accuracy law. Since the question complexity is difficult to quantify in practice, we examine these hypotheses by two properties of the laws, monotonicity and compositionality. We therefore introduce LORE-BENCH, a benchmark that systematically measures these two tractable properties for large reasoning models. Evaluation shows that most reasoning models exhibit reasonable monotonicity but lack compositionality. In response, we develop an effective finetuning approach that enforces compute-law compositionality. Extensive empirical studies demonstrate that better compliance with compute laws yields consistently improved reasoning performance on multiple benchmarks, and uncovers synergistic effects across properties and laws. Project page: https://lore-project.github.io/.

1 INTRODUCTION

Large Reasoning Models (LRMs) such as OpenAI o1 (Jaech et al., 2024) have demonstrated unprecedented progress in approaching human-like reasoning capabilities. Despite their strong performance on solving complex problems, even powerful LRMs exhibit abnormal behaviors that deviate from typical human reasoning patterns. Human generally adapt their thinking based on problem complexity (Newell et al., 1972). In contrast, as illustrated in Fig. 1, DeepSeek-R1 (Guo et al., 2025) tends to generate longer reasoning but with a lower accuracy on a simpler sub-problem1.

We also identify this unexpected phenomenon across a wide range of reasoning models, as shown in Fig. 5. This is primarily because researchers generally overlook the high variability of Chain-ofThought (CoT) (Wei et al., 2022) data during the training phase. These CoT data are heuristically curated by human annotators or generated through online rollout (Schulman et al., 2017; Shao et al., 2024), rarely constrained by explicit rules, e.g., how much thinking budget to allocate for a given problem (Wu et al., 2025). Hence, the current training paradigm fails to guide models toward an optimal thinking strategy. It will lead to inefficient allocation of computation—either overthinking (Chen et al., 2024b; Sui et al., 2025) or underthinking (Su et al., 2025; Yang et al., 2025; Wang et al., 2025), which in turn harms the performance (Stechly et al., 2024; Zhou et al., 2025a).

To overcome this limitation, one line of work focuses on adaptive post-training techniques, including supervised fine-tuning with variable-length CoT (Aggarwal & Welleck, 2025; Team et al., 2025). Another line of work modulates reasoning at test time (Muennighoff et al., 2025; Fan et al., 2025; Zhang et al., 2025b). While many attempts have been made to control reasoning, existing approaches primarily rely on ad-hoc heuristics and still behave undesirably in our studies. Therefore, beyond

∗Equal contribution. †Equal mentorship. ‡Work partially done during the internship at MIT.

Correspondence to: junyuz6@illinois.edu, ppliang@mit.edu, huan@huan-zhang.com. 1For each problem, we generated multiple samples to account for randomness.

Question 1+2

###### Question 1

Question 2

[Figure 1]

[Figure 2]

“Compute the sum of 1 through 10, then square that sum.” Reasoning

“Compute the sum of 1 through 10.”

“Square the number 55 .”

Reasoning

Answer of Question 1 525 tokens

Reasoning tokens 525 1197

Accuracy

1197 tokens

Expected: <38% Actual Accuracy:

Evaluation Evaluation

Expected: Actual Reasoning: 817

Accuracy Accuracy

50%

38%

75%

- Figure 1: Illustrative example with DeepSeek-R1 on (a) a summation question, (b) a squaring question, and (c) their composition (“sum, then square”). The model allocates ~300 more reasoning tokens to solve the squaring question than to the composite question, with a 12.5% accuracy drop. The mismatch with human reasoning reveals an abnormal reasoning pattern present in current LRMs.

empirical methods, several key challenges remain: (Q1) Can we theoretically formalize model reasoning to ensure desirable behavior? (Section 2) (Q2) How can we evaluate whether popular LRMs follow these proposed principles? (Section 3) (Q3) Does enforcing these principles further improve general reasoning capabilities? (Section 4, 5)

To fill this gap, we introduce the Laws of Reasoning (LORE), which systematically formalize the relationship between complexity and model reasoning behaviors in LRMs. The LORE framework comprises a core compute law and a complementary accuracy law. Given the practical challenges of measuring these hypotheses, the two fundamental laws are approximated via two tractable properties of optimal reasoning models, monotonicity and compositionality.

We then evaluate whether current LRMs follow the laws by developing LORE-BENCH, a comprehensive benchmark that examines monotonicity and compositionality in LRMs. While LORE-MONO is a curated benchmark across diverse domains for monotonicity, LORE-COMPO is constructed from MATH500 (Lightman et al., 2023) to measure compositionality. Our evaluation shows that current models exhibit reasonable monotonicity but lack compositionality, even for competitive baselines.

In response, we propose a simple yet effective fine-tuning approach to enforce the compute-law compositionality. From validation experiments, we present three key insights: (1) the compositionality of reasoning compute can be greatly improved with simple fine-tuning approach; (2) Enforcing compositionality generally leads to better reasoning capability; (3) Synergistic effects emerge, yielding broader improvements across different properties and laws.

2 THE LAWS OF REASONING

We introduce the Laws of Reasoning (LORE), a unified framework that formalizes the relationship between question complexity and model reasoning behaviors. Specifically, we focus on two key aspects, reasoning compute and accuracy, which are fundamental to understanding how models scale, generalize, and allocate computation budget when solving complex problems. Section 2.1 formulates the key concepts of reasoning. In Section 2.2, we present the central compute law, with a hypothesis that the reasoning budget should scale proportionally with question complexity. In Section 2.3, we introduce the complementary accuracy law, which posits that overall accuracy should decay exponentially with increasing complexity. See Fig. 2 for an illustration of the overall framework.

- 2.1 PROBLEM FORMULATION

Notation. Let x ∈ X ⊆ V∗ denote a question, where V∗ is the space of finite-length sequences over a vocabulary V. Let Mθ ∈ M denote an autoregressive large reasoning model. LRMs adopts the thinking-then-answering paradigm (Guo et al., 2025; Abdin et al., 2025; Comanici et al., 2025), where the model Mθ first generates a reasoning chain r ∈ R ⊆ V∗ with probability pθ(r | x) and then an answer y ∈ Y ⊆ V∗ with probability pθ(y | x,r). We assume a fixed decoding strategy by default and denote the model’s output by o=(r,y) ∈ O ⊆ V∗. We define the composition of two questions x1 and x2 as their concatenation with a connector prompt c ∈ V∗2, i.e., x1 ⊕ x2 = concat(x1,c,x2).

2One example of c can be “Answer the following questions in order: Q1. {Q1}\nQ2. {Q2}”.

###### Compute Law

###### Accuracy Law

Monotonicity Property

Compositionality Property

Compute

Accuracy 1

For independent questions, Additive Compute

[Figure 3]

[Figure 4]

Compute Law

Complexity Compute

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

For independent questions, Multiplicative Accuracy

[Figure 10]

Accuracy Law

[Figure 11]

Complexity Accuracy

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Complexity

Complexity

- Figure 2: Overview of the LORE Framework. We present the compute law with the complementary accuracy law. These laws posit that compute scales linearly and accuracy decays exponentially with question complexity. Our framework approximates these laws using two properties: monotonicity and compositionality. Specifically, for the compute law, monotonicity captures that more complex questions require more compute, while compositionality indicates that for two independent questions, the compute for their composition is the sum of solving each individually.

- Definition 1 (Complexity). Let a unit-cost primitive step denote a single valid transition of a fixed deterministic Turing machine (Turing et al., 1936), and let τ be any finite sequence of primitive steps with length ℓ(τ) ∈ N. Let v(x,τ) ∈ {0,1} be a binary verifier that accepts (x,τ) if and only if τ is a valid solution sequence for x. The complexity of x ∈ X is

κ(x) ≜ min{ℓ(τ) : v(x,τ) = 1} ∈ N ∪ {∞}, with κ(x) = ∞ if no valid solution sequence exists.

Here the complexity refers to the minimal number of unit-cost primitive steps. Conceptually, κ(x) can be well-defined via a binary verifier and a fixed deterministic Turing machine. However, computing κ(x) is generally intractable, as verifying the minimal solution length requires a global search over a potentially exponential space.

For a given model, its test-time reasoning compute is directly proportional to the number of reasoning tokens generated. We therefore quantify reasoning compute as follows.

- Definition 2 (Reasoning Compute). The reasoning compute on question x is defined as the expected number of reasoning tokens generated by the model

Cθ(x) ≜ Er∼p

θ(·|x)[ℓ(r)],

where ℓ(r) denotes the length (in tokens) of the reasoning chain r.

- Definition 3 (Reasoning Accuracy). The reasoning accuracy is defined as the probability that the model, when generating a reasoning chain and an answer given input x, produces a final answer that matches the ground truth. Formally,

Aθ(x) ≜ E(r,y)∼p

θ(·|x) [1{ans(y) = a⋆(x)}] ∈ [0,1].

where a⋆(x) denotes the correct answer to x, and ans(y) extracts the final answer from y.

- 2.2 COMPUTE LAW

We hypothesize that, if a reasoning model allocates its reasoning compute efficiently, the amount of compute is expected to scale proportionally with complexity in approximation, i.e., Cθ(x) ∝ κ(x):

- Hypothesis 1 (Compute Law). For an optimal reasoning model Mθ and a question x with complexity κ(x), there exist αθ > 0 with,

Cθ(x) = αθ κ(x) + o(κ(x)),

for some αθ > 0 that depends only on Mθ and the decoding strategy. o(κ(x)) denotes a small systematic overhead that is sublinear, i.e., o(κ)/κ → 0 when κ → ∞.

Specifically, the o(κ(x)) term captures the introductory and transition tokens during the reasoning process. These tokens generally constitute a very small portion of the overall reasoning and can therefore be ignored in practice.

Two Tractable Alternative Properties as Proxies. As discussed in Definition 1, the complexity κ(x) is difficult to measure in practice. Consequently, empirically validating the linear relationship is nontrivial, as it would require known complexity values for individual questions. To address this, we adopt two tractable properties as empirical proxies for studying the laws: monotonicity and compositionality. These properties offer two key advantages: (i) they are tractable to verify without access to the exact value of κ(x). Monotonicity relies only on relative comparisons between questions, while compositionality tests whether compute is additive over independent question pairs; (ii) they are theoretically sufficient to imply the proposed compute law (Proposition 1).

- Property 1 (Compute-Complexity Monotonicity). For x1,x2 ∈ X, the reasoning compute is monotonically non-decreasing with complexity:

κ(x1) ≤ κ(x2) =⇒ Cθ(x1) ≤ Cθ(x2). Definition 4 (Independence). For x1,x2 ∈ X, x1 and x2 are independent if the complexity of their composition is additive, i.e., κ(x1 ⊕ x2) = κ(x1) + κ(x2). In practice, since the exact complexity values are difficult to obtain, we define independence operationally. Suppose each question x ∈ X is associated with a set of mathematical concepts3 S(x) ⊆ S relevant to solving it. We consider two questions x1 and x2 to be independent if their concept sets are disjoint, i.e., S(x1) ∩ S(x2) = ∅.

- Property 2 (Compute-Complexity Compositionality). For x1,x2 ∈ X, if x1 and x2 are independent, their composite x1 ⊕ x2 exhibits additive compute:

Cθ(x1 ⊕ x2) = Cθ(x1) + Cθ(x2) + o(κ(x1) + κ(x2)),

where the sublinear terms accounts for systematic overhead in the reasoning process (as assumed in Hypothesis 1). Therefore, the reasoning compute is approximately additive:

Cθ(x1 ⊕ x2) ≈ Cθ(x1) + Cθ(x2).

Discussion. Intuitively, these properties are motivated by two basic principles: (i) more complex questions naturally require more reasoning; (ii) two independent sub-questions involve no overlapping reasoning, so the total compute is the sum of solving each one individually. In the next proposition, we state informally that these properties imply the compute law (Hypothesis 1); a formal proof is provided in Appendix D. These tractable properties thus offer a practical means to evaluate whether current LRMs follow the compute law.

- Proposition 1. Under certain conditions, if a reasoning model Mθ satisfies compute-complexity monotonicity and compositionality, then its reasoning compute Cθ(x) ∝ κ(x) for x ∈ X.

2.3 BEYOND COMPUTE: ACCURACY LAW

Following Definition 1, suppose a question requires solving κ(x) unit-cost primitive steps. If each step succeeds independently with a fixed probability and all steps must succeed for the final answer to be correct, then the overall accuracy is expected to decrease exponentially with κ(x). This intuition motivates the following formulation of the accuracy law:

Hypothesis 2 (Accuracy Law). For an optimal reasoning model Mθ and a question x with complexity κ(x), when 0 < Aθ(x) ≤ 1, there exists λθ ≥ 0 with,

Aθ(x) = exp − λθ κ(x) . Equivalently, log Aθ(x) ∝ −κ(x), where λθ ≥ 0 is the decay rate. Similar to the compute law, we assume that the reasoning accuracy for the optimal reasoning model Mθ also satisfies two fundamental properties: monotonicity and compositionality.

- Property 3 (Accuracy-Complexity Monotonicity). For x1,x2 ∈ X, the reasoning accuracy is monotonically non-increasing with complexity:

#### κ(x1) ≤ κ(x2) =⇒ Aθ(x1) ≥ Aθ(x2).

3For example, concepts may come from Calculus (e.g., derivatives), Algebra (e.g., group theory), or Discrete mathematics (e.g., logic).

###### Seed question: Given integerN, update the initial matrix iteratively with the update rules … Return as the final answer.

[Figure 22]

the number of update steps

[Figure 23]

Variant N2: = 2 Variant 30:N= 30

###### VariantN1: = 1

Reasoning:

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

. . . Update

Update Update

Update

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

- Figure 3: Question Generation of LORE-MONO. For each seed question, we generate 30 variants with increasing complexity. Specifically, variant N applies the update rules N times to compute the answer, so the question complexity increases monotonically with N.

- Property 4 (Accuracy-Complexity Compositionality). For x1,x2 ∈ X, if x1 and x2 are independent, their composite x1 ⊕ x2 exhibits multiplicative accuracy:

Aθ(x1 ⊕ x2) = Aθ(x1) · Aθ(x2).

Discussion. These properties are motivated by two basic principles: (i) more complex questions tend to have lower accuracy; (ii) for two independent questions with accuracies p1 and p2 (e.g., p1 = 0.8, p2 = 0.7), the probability of correctly answering both should be p1 · p2 (e.g., 0.56). We state below that these properties imply the accuracy law, with a formal proof provided in Appendix D.

- Proposition 2. Under certain conditions, if a reasoning model Mθ satisfies accuracy-complexity monotonicity and compositionality, then its reasoning accuracy log Aθ(x) ∝ −κ(x) for x ∈ X.
- 3 DO CURRENT LRMS FOLLOW THE LAWS?

In this section, we systematically evaluate whether current LRMs follow our proposed reasoning laws. Specifically, we introduce LORE-BENCH, a two-fold benchmark that leverages two tractable properties, monotonicity and compositionality, to examine LRMs.

- 3.1 LORE-MONO

Evaluating the monotonicity property in Property 1 or Property 3 requires comparing the complexity of arbitrary question pairs. However, due to its definition via minimal solution length, complexity is inherently difficult to quantify in practice. As a result, existing benchmarks are not suited for such analysis. To address this challenge, we construct LORE-MONO, a synthetic benchmark where questions are carefully curated and validated to follow known complexity orderings, allowing us to systematically assess the monotonicity of reasoning compute and accuracy.

- (1) Seed Question Curation. We select four domains that require extensive reasoning—math, science, language, and code—and curate 10 diverse seed questions for each. A seed question defines a problem template shared across its variants.
- (2) From Seed Questions to Variants. As shown in Fig. 3, for each seed question, we create a series of variants (30 in total) that become increasingly complex by requiring more steps to reach the final answer. For example, variant 1 requires one matrix operation, variant 2 requires two, and variant 30 requires thirty, with the identical operation applied repeatedly. By design, a larger number of steps directly corresponds to higher complexity. Note that this construction does not assume or require each operation to use the same compute.
- (3) Program-based Generation and Manual Verification. All variants are generated through Python scripts to ensure correctness and scalability. To prevent unintended shortcuts such as periodic patterns, we manually verify each seed question and review sampled variants. We provide detailed seed questions and variants for each domain in Appendix E.

We use the Spearman correlation coefficients ρ ∈ [−1,1] to measure how the variant index, which directly determines the constructed question’s complexity, relates to two quantities: reasoning compute and log accuracy. A high correlation with reasoning compute indicates that compute grows

monotonically with complexity (Property 1), while a negative correlation with log accuracy indicates that accuracy tends to degrade as complexity increases (Property 3).

- 3.2 LORE-COMPO

In contrast, assessing compositionality is more straightforward: it only requires taking any two independent questions as sub-questions and constructing their composition. We build LORE-COMPO from MATH500 (Lightman et al., 2023), where each question is labeled by subject (e.g., Algebra, Geometry). Specifically, we randomly sample a pair of questions (x1,x2) from distinct pre-defined subjects to ensure independence, and concatenate them into a composite question x12. Each original question is used at most once, yielding 250 triplets, each with two sub-questions and their composition:

DLoRe-Compo = {(x(1i),x(2i),x(12i))}250i=1. Recall that for a function fθ(·) (either Cθ(·) or log Aθ(·)), compositionality implies that fθ(x12) ≈ fθ(x1) + fθ(x2). We therefore quantify the degree to which a model follows this property using the mean absolute deviation (MAD):

MADf =

(x1,x2,x12)∈DLoRe-Compo

fθ(x12) − fθ(x1) + fθ(x2)

A smaller MAD indicates stronger adherence to the compositionality property. However, MAD is scale-dependent. To address this, we adopt the Normalized MAD (nMAD):

nMADf =

MADf Sf

, Sf =

(x1,x2,x12)∈DLoRe-Compo

|fθ(x1) + fθ(x2)|.

- 3.3 FINDINGS AND ANALYSIS

Evaluation Setups. We examine 10 LRMs on LORE-MONO and LORE-COMPO: eight standard models— DeepSeek-R1-Distill (Qwen-1.5B, Qwen-7B, Llama-8B, Qwen-14B) (Guo et al., 2025), Phi-4-mini-reasoning (Xu et al., 2025a), OpenReasoning-Nemotron-14B (Ahmad et al., 2025), SkyT1-32B-Preview (Team, 2025a) and Qwen3-Next-80B-A3B-Thinking (Team, 2025b)—and two models that apply reasoning length control, Thinkless-1.5B-RL-DeepScaleR (Fang et al., 2025) and AdaptThink-7B-delta0.05 (Zhang et al., 2025a). For each question, we sample 8 outputs per model with a fixed decoding temperature (0.6 for the DeepSeek family and 0.8 for the Phi-4 family from their technical reports) and a maximum length of 20480 tokens. For LORE-MONO, at each variant index we first average reasoning compute4 and log accuracy across the 40 questions, and then compute the Spearman correlation.

- Table 1: Monotonicity Results on LORE-MONO. We examine whether reasoning compute and log accuracy of 6 popular LRMs satisfy the monotonicity property across four domains. Spearman correlations are reported for reasoning compute and log accuracy. Lang. stands for Language.

Reasoning Compute ↑ Log Accuracy ↓

Size Models

Math Science Lang. Code All Math Science Lang. Code All 1.5B

DeepSeek-R1-1.5B 0.861 0.910 -0.346 0.151 0.875 -0.795 -0.864 -0.210 -0.487 -0.868

Thinkless-1.5B 0.943 0.961 0.648 0.794 0.976 -0.951 -0.934 -0.556 -0.539 -0.960 3.8B Phi-4-mini 0.980 0.973 0.936 0.922 0.988 -0.965 -0.802 -0.911 -0.822 -0.954

- 7B

DeepSeek-R1-7B 0.956 0.975 0.901 0.970 0.991 -0.946 -0.876 -0.899 -0.818 -0.978 AdaptThink-7B 0.984 0.995 0.950 0.984 0.995 -0.963 -0.949 -0.904 -0.888 -0.972

- 8B DeepSeek-R1-8B 0.982 0.962 0.864 0.963 0.988 -0.944 -0.796 -0.924 -0.843 -0.947

Nemontron-14B 0.964 0.976 0.917 0.970 0.993 -0.778 -0.751 -0.793 -0.911 -0.913

14B

DeepSeek-R1-14B 0.978 0.973 0.903 0.980 0.990 -0.888 -0.888 -0.803 -0.933 -0.981 32B Sky-T1-32B 0.988 0.982 0.711 0.874 0.967 -0.939 -0.876 -0.783 -0.860 -0.963 80B Qwen3-80B-Next 0.977 0.984 0.794 0.993 0.992 -0.403 -0.551 -0.774 -0.907 -0.973

4We apply max–min normalization to the reasoning compute of each question to prevent any single item from dominating the results.

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

𝐿𝑜𝑔(𝐴 (𝑥))

𝐶 (𝑥)

- Figure 4: Visualizations of Monotonicity Results on DeepSeek-R1-1.5B. For each domain, we plot reasoning compute and log accuracy as a function of variant index. The curves report the mean accuracy across 10 questions series, and the shaded regions denote the standard deviation.

- Table 2: Compositionality Results on LORECOMPO. We calculate nMAD for reasoning compute (Cθ) and log accuracy (log Aθ).

Models nMADCθ ↓ nMADlogAθ ↓ DeepSeek-R1-1.5B 0.528 2.368

Thinkless-1.5B 0.339 0.694 Phi-4-mini 0.322 0.732

- DeepSeek-R1-7B 0.337 1.170 AdaptThink-7B 0.327 0.791
- DeepSeek-R1-8B 0.423 0.818 DeepSeek-R1-14B 0.368 1.310

Nemotron-14B 0.354 1.188 Qwen3-80B-Next 0.411 1.487

Sky-T1-32B 0.392 1.900

DeepSeek-R1-1.5B

DeepSeek-R1-8B

×104

×104

| |nMAD =|0.528| |
|---|---|---|---|
| | | | |
| | | |y = x|
| | | | |

| |nMAD = 0.|423| |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | |y =|x|
| | | | |

2.0

- 0
- 1
- 2

1.5

()Cx12

()Cx12

1.0

0.5

0.0

0 1 2 C (x1) + C (x2) ×10

0 1 2 C (x1) + C (x2) ×10

4

4

Figure 5: Visualizations of Compositionality Results on Reasoning Compute. We plot Cθ(x1 ⊕ x2) against Cθ(x1) + Cθ(x2). Further results are provided in Appendix E.4.

Current LRMs Largely Satisfy Monotonicity. On LORE-MONO, all LRMs exhibit a strong positive correlation between reasoning compute and the variant index, which directly reflects question complexity, with most overall Spearman correlations close to 1, as shown in Tab. 1. The only exception is DeepSeek-R1-Distill-Qwen-1.5B, which has the weakest reasoning ability among the six models and yields a lower overall correlation (0.875). As illustrated in Fig. 4, notably, in the language domain its correlation between reasoning compute and complexity is negative (−0.346), while in the code domain, it is near zero (0.151). This indicates that in some domains, the reasoning compute for this model does not systematically increase with complexity, and may even decrease. We provide a case study as additional analysis along with visualization results for other models in Appendix E.2. Meanwhile, most LRMs exhibit a negative correlation between log accuracy and the variant index, as expected. For DeepSeek-R1-Distill-Qwen-1.5B, however, this trend appears noticeably weaker.

Current LRMs Fail to Exhibit Compositionality. The nMAD is large for both reasoning compute and log accuracy (Tab. 2), indicating that current LRMs do not satisfy compositionality. Fig. 5 further plots Cθ(x1 ⊕ x2) against Cθ(x1) + Cθ(x2) for two representative LRMs. If an LRM adhered to the compositionality law, most points would align closely with the y = x line. In practice, however, the majority of points deviate substantially. Notably, even models equipped with reasoning length control mechanisms (Thinkless-1.5B and AdaptThink-7B) exhibit considerable deviations, suggesting that such techniques do not inherently promote compositional behavior.

- 4 IMPROVING REASONING VIA ENFORCING COMPOSITIONALITY

In Section 3, we showed that while most LRMs generally satisfy monotonicity, they often fail to satisfy compositionality. Based on Hypothesis 1, this observation motivates a natural question: can enforcing compositionality lead to stronger reasoning capacity? In response, we propose a simple yet effective supervised fine-tuning (SFT) method to promote compositional behavior in LRMs. Importantly, we focus on enforcing compositionality specifically with respect to reasoning compute, as it provides a more direct and actionable criterion for selecting supervision examples.5

5Accuracy compositionality is not easy to enforce directly, as it does not specify which reasoning path should be selected for supervision.

Proposed Method: SFT-Compo Specifically, let Mθ be an LRM and Dtrain a training dataset. Following the construction in Section 3.2, we select question pairs (x1,x2) ∈ Dtrain from distinct categories and form composite questions x12 = x1 ⊕ x2. For each triplet (x1,x2,x12), we sample K model outputs o = (r,y) ∈ O from an LRM (either the current model Mθ or a stronger teacher model), where r ∈ R is a reasoning path and y ∈ Y is the corresponding final answer:

{o(1k) = (r1(k),y1(k))}Kk=1 for x1, {o(2k) = (r2(k),y2(k))}Kk=1 for x2, {o(12k) = (r12(k),y12(k))}Kk=1 for x12.

Since compositionality is defined over reasoning paths, among the K3 combinations (o1,o2,o12), we consider only those where all three reasoning paths r1,r2,r12 lead to correct answers, and select the combination that best satisfies the compositionality condition:

(r1∗,r2∗,r12∗ ) = arg min r1,r2,r12

|ℓ(r1) + ℓ(r2) − ℓ(r12)|

s.t. r1,r2,r12 each yielding a correct final answer. (1)

Each triplet thus yields three supervised examples: (x1,o∗1), (x2,o∗2), and (x12,o∗12), where o∗i = (ri∗,yi∗) with yi∗ the final answer paired with ri∗ in the sampled outputs. Aggregating across all triplets, we construct the compositional supervision dataset Dcomp = {(x1,o∗1), (x2,o∗2), (x12,o∗12) | (x1,x2) ∈ Dtrain}. We then perform SFT on Dcomp to encourage Mθ to internalize compositional reasoning behavior.

- 5 EXPERIMENTS

We now empirically evaluate SFT-Compo, addressing two research questions: (1) whether it effectively enforces compositionality, and (2) whether it further improves the reasoning capacity of LRMs. We also provide additional insightful findings in our analysis.

- 5.1 EXPERIMENTAL SETUP Model, Dataset and SFT Recipe. We evaluate four LRMs: DeepSeek-R1-Distill (Qwen-1.5B, Qwen-

- 7B, Llama-8B) (Guo et al., 2025) and Phi-4-mini-reasoning (Xu et al., 2025a). We construct a dataset of sub-question and composite-question triplets using a subset of DeepScaler (Luo et al., 2025b). For each question (either sub-question or composite), we use DeepSeek-R1-Distill-Qwen-14B as a stronger teacher model to sample K = 8 model outputs. We then construct the compositionalityenforced dataset Dcomp as described in Eqn. 1, which contains 3.9K question-output pairs. We fine-tune each LRM on Dcomp for 5 epochs with a batch size of 16. Details are provided in Appendix F.

Evaluation. To evaluate compositionality, we use LORE-COMPO. For general reasoning capacity, we consider six benchmarks: GSM8K (Cobbe et al., 2021), MATH500 (Lightman et al., 2023), AIME 2024, AIME 2025 (Mathematical Association of America, 2025), AMC 2023 (AI-MO, 2024), and OlympiadBench (He et al., 2024). We set the maximum generation length to 10240 tokens.

- 5.2 MAIN RESULTS

DeepSeek-R1-1.5B

SFT-Compo-1.5B (Ours)

×104

×104

| |nMAD =|0.528| |
|---|---|---|---|
| | | | |
| | | |y = x|
| | | | |

| |nMAD = 0.|314| |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | |y =|x|
| | | | |

Base SFT-Compo

2.0

0.528

- 0
- 1
- 2

1.5

0.423

()Cx12

()Cx12

0.337

0.328

0.317

0.314

1.0

0.5

0.0

0 1 2 C (x1) + C (x2) ×10

0 1 2 C (x1) + C (x2) ×10

1.5B 7B 8B

4

4

(a) nMADCθ on LORE-COMPO.

(b) Visualizations of Reasoning Compute Compositionality.

### Figure 6: Comparison of Reasoning Compute Compositionality on LORE-COMPO for Base and

SFT-Compo models. (a) SFT-Compo consistently achieves a lower nMADC

across 1.5B, 7B, and 8B

θ

models compared to the base model. (b) We visualize Cθ(x1 ⊕ x2) against Cθ(x1) + Cθ(x2) for 1.5B models. SFT-Compo aligns more closely with the y=x line than the base model.

Does SFT-Compo Effectively Enforce Compositionality Compared to the Base Model? We compare LRMs before and after SFT using the nMAD of reasoning compute on LORE-COMPO. As shown in Fig. 6a, SFT-Compo consistently reduces nMAD compared to the base model. On the 1.5B model, SFT-Compo achieves a reduction from 0.528 to 0.314 (a 40.5% reduction), and on the

- 8B model, from 0.423 to 0.328 (a 22.5% reduction). We further visualize the results on the 1.5B model in Fig. 6b, where SFT-Compo aligns much more closely with the y=x line. Therefore, the compositionality of reasoning compute can be effectively enforced in a simple manner via SFT-Compo.

Does Enforcing Compositionality Lead to Stronger Reasoning Capabilities? As shown in Tab. 3, SFT-Compo consistently improves performance across all six benchmarks and all four model sizes. For instance, on the 8B model, it yields a notable gain of +5.0 in average Pass@1. To rule out the possibility that performance gains stem solely from leveraging outputs generated by a stronger teacher model, we introduce a control baseline, SFT, which constructs the training dataset by uniformly sampling one correct reasoning path for each question in the triplet:

(r1∗,r2∗,r12∗ ) ∼ Unif({(r1,r2,r12) | r1,r2,r12 each yield a correct final answer}).

Notably, SFT-Compo outperforms SFT in all cases, showing that the gains are not just from distilling a stronger model but from better compliance with compositionality. This supports our Hypothesis 1—that stronger models better follow reasoning laws—and demonstrate that encouraging compositionality further enhances the reasoning capabilities of LRMs.

- Table 3: General Reasoning Evaluation Results. We evaluate Base (pre-SFT), SFT, and SFT-Compo (Ours) on mathematical and science reasoning benchmarks. All numbers report Pass@1 accuracy (%) computed over 8 sampled outputs. Pass@1 denotes the average across the six benchmarks. Numbers in orange indicate improvements relative to the base model.

MATH SCIENCE

Base Model Method

Pass@1 AIME24 AIME25 AMC23 MATH500 GSM8K Olympiad

Base 18.8 20.4 59.7 71.6 81.2 33.8 47.6

DeepSeek-R1-1.5B

SFT 20.4+1.6 21.5+1.1 59.6-0.1 76.4+4.8 81.7+0.5 36.1+2.3 49.3+1.7 SFT-Compo (Ours) 26.2+7.4 21.7+1.3 65.0+5.3 77.6+6.0 85.1+3.9 38.7+4.9 52.4+4.8

Base 36.3 27.5 79.0 86.8 91.0 48.1 61.5

###### DeepSeek-R1-7B

SFT 40.0+3.7 32.5+5.0 80.4+1.4 88.0+1.2 91.6+0.6 48.4+0.3 63.5+2.0 SFT-Compo (Ours) 43.3+7.0 33.2+5.7 80.6+1.6 88.8+2.0 91.6+0.6 50.5+2.4 64.7+3.2

Base 28.3 22.9 71.9 76.4 86.5 40.9 54.5

###### DeepSeek-R1-8B

SFT 30.4+2.1 24.2+1.3 75.2+3.3 82.6+6.2 88.0+1.5 44.7+3.8 57.5+3.0 SFT-Compo (Ours) 31.3+3.0 29.2+6.3 76.9+5.0 83.0+6.6 89.5+3.0 46.8+5.9 59.5+5.0

Base 32.5 24.6 71.9 86.4 91.4 50.0 59.5

Phi-4-mini

SFT 40.0+7.5 26.2+1.6 73.4+1.5 87.0+0.6 91.6+0.2 49.5-0.5 61.3+1.8 SFT-Compo (Ours) 43.7+11.2 31.7+7.1 76.9+5.0 87.0+0.6 92.8+1.4 51.2+1.2 63.9+4.4

- 5.3 SYNERGISTIC EFFECT ANALYSIS

LoRe-Mono

LoRe-Mono-Code

ReasoningCompute

ReasoningCompute

Base SFT-Compo

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.8

0.6

2.368

0.6

0.4

0.4

1.170

0.2

0.2

0.756 0.818

0.685

0.0

0.677

1 5 10 15 20 25 30

1 5 10 15 20 25 30

Variant Index

Variant Index

SFT-Compo-1.5B DeepSeek-R1-1.5B

1.5B 7B 8B

(a) Visualizations of monoticity results on LORE-MONO and the code domain for 1.5B models.

(b) nMADlogAθ on LORE-COMPO.

- Figure 7: Synergistic Effects Among Different Reasoning Properties and Laws. (a) Enforcing compositionality in reasoning compute improves its monotonicity. (b) Enforcing compositionality in

reasoning compute also improves the compositionality of log accuracy, measured by nMADlogA

.

θ

Enforcing Compositionality in Reasoning Compute Improves Its Monotonicity. Recall from Section 3.3 that DeepSeek-R1-Distill-Qwen-1.5B initially exhibits relatively weak monotonicity in reasoning compute. Fig. 7a shows SFT-Compo significantly improves this property, increasing the overall Spearman correlation from 0.875 to 0.977. Specifically, in the code domain, it rises from 0.151 to 0.914. This indicates that enforcing compositionality can implicitly enhance monotonicity.

Enforcing Compositionality in Reasoning Compute Improves Compositionality in Accuracy. Interestingly, though SFT-Compo is designed to enhance compositionality in reasoning compute, it improves the compositionality of log accuracy. Fig. 7b shows that the nMAD of log accuracy drops from 2.368 to 0.685 on the 1.5B model (a 71.1% reduction), and from 1.170 to 0.756 on the 7B model (a 35.4% reduction). This suggests a possible interplay among different reasoning laws.

- 6 RELATED WORK

LRMs have emerged as a family of foundation models (Wiggins & Tejani, 2022). Since the advent of OpenAI o1 (Jaech et al., 2024), the “thinking-then-answering” paradigm has been widely adopted, with notable follow-ups such as DeepSeek-R1 and Phi-4-Reasoning (Abdin et al., 2024; Guo et al., 2025; Qwen Team, 2025). Our framework builds upon the contemporary paradigm of adaptive reasoning, wherein the model’s reasoning budget is dynamically controlled either through posttraining interventions (Luo et al., 2025a; Zhou et al., 2025b) or at test time (Muennighoff et al., 2025; Xu et al., 2025b; Zhang et al., 2025b). Specifically, one line of work explores post-training techniques that modulate when and how long a model should reason (Chen et al., 2024a; Yong et al., 2025), while another frontier focuses on dynamically adjusting reasoning behavior during inference (Qiao et al., 2025; Liu & Wang, 2025). Refer to Appendix B for additional related work.

- 7 CONCLUSIONS

As a comprehensive study from theoretical hypotheses to empirical validation, we advance a theoretical perspective grounded in human reasoning for improving reasoning in LRMs. We hope LORE can inspire more potential strategies that guide models toward their optimal paradigms of thinking.

ACKNOWLEDGMENTS

The authors thank Heng Dong, Runpei Dong and Chanakya Ekbote for their insightful feedback and valuable suggestions on this project.

ETHICS STATEMENT

This work does not raise any known ethical concerns.

REPRODUCIBILITY STATEMENT

Our code repository is available at https://github.com/ASTRAL-Group/LoRe. In Section 3.3 and Section 5.1, we provide a detailed description of the experimental setup, including dataset, models, training and evaluation procedures. Additional implementation details can be found in Appendix F.1.

REFERENCES

Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J Hewett, Mojan Javaheripi, Piero Kauffmann, et al. Phi-4 technical report. arXiv preprint arXiv:2412.08905, 2024.

Marah Abdin, Sahaj Agarwal, Ahmed Awadallah, Vidhisha Balachandran, Harkirat Behl, Lingjiao Chen, Gustavo de Rosa, Suriya Gunasekar, Mojan Javaheripi, Neel Joshi, et al. Phi-4-reasoning technical report. arXiv preprint arXiv:2504.21318, 2025.

Pranjal Aggarwal and Sean Welleck. L1: Controlling how long a reasoning model thinks with reinforcement learning, 2025. URL https://arxiv. org/abs/2503.04697, 2025.

Wasi Uddin Ahmad, Somshubra Majumdar, Aleksander Ficek, Sean Narenthiran, Mehrzad Samadi, Jocelyn Huang, Siddhartha Jain, Vahid Noroozi, and Boris Ginsburg. OpenCodeReasoning-II: A Simple Test Time Scaling Approach via Self-Critique, 2025. URL https://arxiv.org/abs/ 2507.09075.

AI-MO. AIMO Validation Dataset - AMC. https://huggingface.co/datasets/ AI-MO/aimo-validation-amc, 2024. URL https://huggingface.co/datasets/AI-MO/ aimo-validation-amc. Accessed: 2025-05-19.

Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, et al. Graph of thoughts: Solving elaborate problems with large language models. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pp. 17682–17690, 2024.

Qiguang Chen, Libo Qin, Jiaqi Wang, Jingxuan Zhou, and Wanxiang Che. Unlocking the capabilities of thought: A reasoning boundary framework to quantify and optimize chain-of-thought. Advances in Neural Information Processing Systems, 37:54872–54904, 2024a.

Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, et al. Do not think that much for 2+ 3=? on the overthinking of o1-like llms. arXiv preprint arXiv:2412.21187, 2024b.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Chongyu Fan, Yihua Zhang, Jinghan Jia, Alfred Hero, and Sijia Liu. Cyclicreflex: Improving large

reasoning models via cyclical reflection token scheduling. arXiv preprint arXiv:2506.11077, 2025. Gongfan Fang, Xinyin Ma, and Xinchao Wang. Thinkless: Llm learns when to think. arXiv preprint

arXiv:2505.13379, 2025.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3828–3850, 2024.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Junyan Li, Wenshuo Zhao, Yang Zhang, and Chuang Gan. Steering llm thinking with budget guidance. arXiv preprint arXiv:2506.13752, 2025.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023.

Wei Liu, Ruochen Zhou, Yiyun Deng, Yuzhen Huang, Junteng Liu, Yuntian Deng, Yizhe Zhang, and Junxian He. Learn to reason efficiently with adaptive length-based reward shaping. arXiv preprint

- arXiv:2505.15612, 2025.

Xin Liu and Lu Wang. Answer convergence as a signal for early stopping in reasoning. arXiv preprint

- arXiv:2506.02536, 2025.

Haotian Luo, Li Shen, Haiying He, Yibo Wang, Shiwei Liu, Wei Li, Naiqiang Tan, Xiaochun Cao, and Dacheng Tao. O1-pruner: Length-harmonizing fine-tuning for o1-like reasoning pruning. arXiv preprint arXiv:2501.12570, 2025a.

Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Tianjun Zhang, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl, 2025b. Notion Blog.

Mathematical Association of America. American Invitational Mathematics Examination – AIME. American Invitational Mathematics Examination – AIME 2025, February 2025. URL https://maa. org/math-competitions/american-invitational-mathematics-examination-aime. Accessed: 2025-09-21.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling. arXiv preprint arXiv:2501.19393, 2025.

Allen Newell, Herbert Alexander Simon, et al. Human problem solving, volume 104. Prentice-hall Englewood Cliffs, NJ, 1972.

Ziqing Qiao, Yongheng Deng, Jiali Zeng, Dong Wang, Lai Wei, Fandong Meng, Jie Zhou, Ju Ren, and Yaoxue Zhang. Concise: Confidence-guided compression in step-by-step efficient reasoning. arXiv preprint arXiv:2505.04881, 2025.

Qwen Team. Preview of qwen qwen1.5-32b. https://qwenlm.github.io/blog/ qwq-32b-preview/, 2025. Accessed: 2025-03-20.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Parshin Shojaee, Iman Mirzadeh, Keivan Alizadeh, Maxwell Horton, Samy Bengio, and Mehrdad Farajtabar. The illusion of thinking: Understanding the strengths and limitations of reasoning models via the lens of problem complexity. arXiv preprint arXiv:2506.06941, 2025.

Kaya Stechly, Karthik Valmeekam, and Subbarao Kambhampati. Chain of thoughtlessness? an analysis of cot in planning, 2024. URL https://arxiv. org/abs/2405.04776, 2024.

Jinyan Su, Jennifer Healey, Preslav Nakov, and Claire Cardie. Between underthinking and overthinking: An empirical study of reasoning length and correctness in llms. arXiv preprint arXiv:2505.00127, 2025.

Yang Sui, Yu-Neng Chuang, Guanchu Wang, Jiamu Zhang, Tianyi Zhang, Jiayi Yuan, Hongyi Liu, Andrew Wen, Shaochen Zhong, Na Zou, et al. Stop overthinking: A survey on efficient reasoning for large language models. arXiv preprint arXiv:2503.16419, 2025.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, C Chen, C Li, C Xiao, C Du, C Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms, 2025. URL https://arxiv. org/abs/2501.12599, 2025.

NovaSky Team. Sky-t1: Fully open-source reasoning model with o1-preview performance in 450

budget, 2025a. Qwen Team. Qwen3 technical report, 2025b. URL https://arxiv.org/abs/2505.09388. Alan Mathison Turing et al. On computable numbers, with an application to the entscheidungsproblem.

J. of Math, 58(345-363):5, 1936.

Yue Wang, Qiuzhi Liu, Jiahao Xu, Tian Liang, Xingyu Chen, Zhiwei He, Linfeng Song, Dian Yu, Juntao Li, Zhuosheng Zhang, et al. Thoughts are all over the place: On the underthinking of o1-like llms. arXiv preprint arXiv:2501.18585, 2025.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Walter F Wiggins and Ali S Tejani. On the opportunities and risks of foundation models for natural language processing in radiology. Radiology: Artificial Intelligence, 4(4):e220119, 2022.

Yuyang Wu, Yifei Wang, Ziyu Ye, Tianqi Du, Stefanie Jegelka, and Yisen Wang. When more is less: Understanding chain-of-thought length in llms. arXiv preprint arXiv:2502.07266, 2025.

Haoran Xu, Baolin Peng, Hany Awadalla, Dongdong Chen, Yen-Chun Chen, Mei Gao, Young Jin Kim, Yunsheng Li, Liliang Ren, Yelong Shen, et al. Phi-4-mini-reasoning: Exploring the limits of small reasoning language models in math. arXiv preprint arXiv:2504.21233, 2025a.

Silei Xu, Wenhao Xie, Lingxiao Zhao, and Pengcheng He. Chain of draft: Thinking faster by writing less. arXiv preprint arXiv:2502.18600, 2025b.

Wenkai Yang, Shuming Ma, Yankai Lin, and Furu Wei. Towards thinking-optimal scaling of test-time compute for llm reasoning. arXiv preprint arXiv:2502.18080, 2025.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36:11809–11822, 2023.

Xixian Yong, Xiao Zhou, Yingying Zhang, Jinlin Li, Yefeng Zheng, and Xian Wu. Think or not? exploring thinking efficiency in large reasoning models via an information-theoretic lens. arXiv preprint arXiv:2505.18237, 2025.

Jiajie Zhang, Nianyi Lin, Lei Hou, Ling Feng, and Juanzi Li. Adaptthink: Reasoning models can learn when to think. arXiv preprint arXiv:2505.13417, 2025a.

Junyu Zhang, Runpei Dong, Han Wang, Xuying Ning, Haoran Geng, Peihao Li, Xialin He, Yutong Bai, Jitendra Malik, Saurabh Gupta, and Huan Zhang. Alphaone: Reasoning models thinking slow and fast at test time. arXiv preprint arXiv:2505.24863, 2025b.

Kaiwen Zhou, Chengzhi Liu, Xuandong Zhao, Shreedhar Jangam, Jayanth Srinivasa, Gaowen Liu, Dawn Song, and Xin Eric Wang. The hidden risks of large reasoning models: A safety assessment of r1. arXiv preprint arXiv:2502.12659, 2025a.

Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, Bryan Kian Hsiang Low, and Paul Pu Liang. Mem1: Learning to synergize memory and reasoning for efficient long-horizon agents. arXiv preprint arXiv:2506.15841, 2025b.

- A LLM USAGE LLMs were used solely for language polishing.
- B ADDITIONAL RELATED WORK

Large Reasoning Models. Large Reasoning Models (LRMs) have emerged as a family of foundation models (Wiggins & Tejani, 2022). Since the advent of OpenAI o1 (Jaech et al., 2024), this “thinking-then-answering” paradigm has been widely adopted. Notably, o1-like Reasoning Models can solve increasingly complex reasoning problems through elaborate reasoning chains (Wei et al., 2022; Yao et al., 2023; Besta et al., 2024). Numerous efforts replicating o1’s success include DeepSeek-R1 and Phi-4-Reasoning (Abdin et al., 2024; Guo et al., 2025; Qwen Team, 2025). Despite impressive progress, the internal mechanisms and behavioral patterns of reasoning in LRMs remain underexplored. Shojaee et al. (2025) take a step in this direction by examining reasoning through the lens of problem complexity, though their analysis is limited to a constrained puzzle-solving setting.

Reasoning Length Control. Our framework builds upon the contemporary paradigm of adaptive reasoning, in which the reasoning budget of the model is controlled either during post-training (Luo et al., 2025a; Zhou et al., 2025b) or at test time (Muennighoff et al., 2025; Xu et al., 2025b; Zhang et al., 2025b). One line of work develops post-training techniques that modulate when and how long a model should reason (Chen et al., 2024a; Yong et al., 2025). This is achieved through two primary strategies: one involves supervised fine-tuning on variable-length CoT with concise yet sufficient reasoning (Aggarwal & Welleck, 2025; Team et al., 2025); the other utilizes RL through length penalty (Zhang et al., 2025a; Fang et al., 2025; Liu et al., 2025). Beyond these, another frontier involves implementing dynamic control over reasoning during inference. For example, some approaches allocate inference budget via confidence (Qiao et al., 2025; Liu & Wang, 2025), while others employ a secondary controller to modulate (Li et al., 2025).

- C LIMITATIONS

We acknowledge several limitations. First, our LORE-MONO currently includes only 40 seed questions in total. Expanding its topic diversity and coverage is an important direction for future work. Second, we operationalize independence through disjoint sets of mathematical concepts. Although this proxy is not rigorous, it is motivated by the practical difficulty of formalizing independence between questions in an actionable and general way. We leave more refined treatments of independence to future work. Finally, due to budget constraints, we focus on strong open-source LRMs, as evaluating closed-source models would require substantial additional cost.

- D PROOFS AND COROLLARIES

We first restate Proposition 1 and Proposition 2 formally and provide a complete proof, along with corresponding corollaries.

- Proposition 1 (Formal Version). Fix a question space X, a complexity map κ : X → N ∪ {∞}, and a reasoning compute map Cθ : X → R≥0. Let ⊕ be a binary composition operator. For m ≥ 3 and jointly independent x1,...,xm, define x1 ⊕ ··· ⊕ xm by a fixed bracketing (e.g. right-associated). Define

Xfin := {x ∈ X : κ(x) < ∞}, K := κ(Xfin) ⊆ N. All assumptions below are imposed on Xfin.

- (A1) Monotonicity. If κ(x) ≤ κ(y) then Cθ(x) ≤ Cθ(y).
- (A2) Additivity under composition of independent questions. If x,y are independent, then

- κ(x ⊕ y) = κ(x) + κ(y), Cθ(x ⊕ y) = Cθ(x) + Cθ(y).

(A3) For every u ∈ K and every m ∈ N, there exist x1,...,xm ∈ Xfin such that

- κ(xi) = u for all i, {x1,...,xm} is jointly independent.

Consequently, x1 ⊕ ··· ⊕ xm is valid and

m

κ(x1 ⊕ ··· ⊕ xm) = mu ∈ K, Cθ(x1 ⊕ ··· ⊕ xm) =

Cθ(xi).

i=1

Then there exists a constant αθ ≥ 0 such that

Cθ(x) = αθ κ(x) for all x ∈ Xfin.

Proof. Define an equivalence relation x ∼ y ⇐⇒ κ(x) = κ(y). By (A1), κ(x) ≤ κ(y) and κ(y) ≤ κ(x) imply Cθ(x) ≤ Cθ(y) and Cθ(y) ≤ Cθ(x), hence Cθ(x) = Cθ(y) whenever x ∼ y. Thus there is a well-defined f : K → R≥0 with f(n) = Cθ(x) for any x such that κ(x) = n.

Fix u ∈ K and m ∈ N. By (A3) choose jointly independent x1,...,xm with κ(xi) = u. By (A2) and the fixed bracketing,

m

κ(x1 ⊕ ··· ⊕ xm) = mu, Cθ(x1 ⊕ ··· ⊕ xm) =

Cθ(xi) = mf(u), so

i=1

f(mu) = mf(u) (∀u ∈ K, ∀m ∈ N).

If K = {0} then Cθ ≡ 0 and the claim holds with αθ = 0. Otherwise take u,v ∈ K with u,v > 0 and let ℓ = lcm(u,v). Then

ℓ u

ℓ u

ℓ v

ℓ v

f(u) and f(ℓ) = f v ·

f(ℓ) = f u ·

=

=

f(v),

hence f(u)/u = f(v)/v, independent of u,v. Write this common ratio as αθ ≥ 0. Therefore f(n) = αθn for all n ∈ K, and Cθ(x) = f(κ(x)) = αθ κ(x) for all x ∈ Xfin.

| |
|---|

- Corollary D.1 (Asymptotic version with sublinear overhead). If the compositional compute holds up to a sublinear overhead, i.e., for independent x,y,

Cθ(x ⊕ y) = Cθ(x) + Cθ(y) + o(κ(x) + κ(y)), and the same (A3) assumption holds, then the above proof yields

#### Cθ(x) = αθ κ(x) + o κ(x) (κ(x) → ∞).

- Proposition 2 (Formal Version). Let Xfin = {x ∈ X : κ(x) < ∞}. Assume the setting and independence notion of Property 3, Property 4, and Assumption (A3). Then there exists λθ ≥ 0 such that for all x ∈ Xfin with 0 < Aθ(x) ≤ 1,

#### Aθ(x) = exp − λθ κ(x) .

Proof. Define an equivalence relation x ∼ y iff κ(x) = κ(y). By (A1), if x ∼ y then both κ(x) ≤ κ(y) and κ(y) ≤ κ(x) hold, hence Aθ(x) ≥ Aθ(y) and Aθ(y) ≥ Aθ(x), so Aθ(x) = Aθ(y). Therefore there exists a well-defined map

f : K → (0,1], f(n) := Aθ(x) for any x ∈ Xfin with κ(x) = n. Let g : K → R≥0 be g(n) := −log f(n). Fix u ∈ K and m ∈ N. By (A3), choose jointly independent x1,...,xm with κ(xi) = u. By (A2) and the fixed bracketing,

Hence

κ(x1 ⊕ ··· ⊕ xm) = mu, Aθ(x1 ⊕ ··· ⊕ xm) =

m

Aθ(xi) = f(u) m.

i=1

#### g(mu) = −log Aθ(x1 ⊕ ··· ⊕ xm) = mg(u) (∀u ∈ K, ∀m ∈ N). (2)

If K = {0} then Aθ ≡ 1 and the claim holds with λθ = 0. Otherwise, let u,v ∈ K with u,v > 0 and set ℓ = lcm(u,v). Applying Eqn. 2 twice gives

ℓ u

ℓ u

ℓ v

ℓ v

g(u) and g(ℓ) = g v ·

g(ℓ) = g u ·

=

=

g(v),

so g(u)/u = g(v)/v. This ratio is independent of u,v > 0 in K; denote it by λθ ≥ 0.

For any n ∈ K, if n = 0 then g(n) = 0 = λθn; if n > 0 pick any u ∈ K \ {0} and write n = nu u to get from Eqn. 2 that g(n) = nug(u) = λθn. Therefore g(n) = λθn for all n ∈ K, i.e. f(n) = exp(−λθn), and for any x ∈ Xfin,

Aθ(x) = f κ(x) = exp − λθ κ(x) .

| |
|---|

- Corollary D.2 (Asymptotic version with sublinear coupling). If for independent x,y the multiplicativity holds up to a sublinear deviation in the exponent,

log Aθ(x ⊕ y) = log Aθ(x) + log Aθ(y) + o κ(x) + κ(y) , and (A3) holds, then

log Aθ(x) = −λθ κ(x) + o κ(x) (κ(x) → ∞), equivalently Aθ(x) = exp − λθκ(x) + o(κ(x)) .

- E ADDITIONAL DETAILS AND RESULTS OF LORE-BENCH

- E.1 ADDITIONAL DETAILS OF LORE-MONO

- E.1.1 EXAMPLE SEED QUESTIONS OF LORE-MONO Here we provide one representative seed question example for each domain.

### Math - Example seed question

Given an integer n = {N}, consider the order-2 recurrence over integers modulo M with an alternating update rule and a mild nonlinear term. You are given the initial values

#### x0 = x0, x1 = x1.

We update the sequence one step at a time. Let t = 1,2,3,...,n denote the update index, where t = 1 is the update that produces x2 from (x1,x0). At each update t, compute xk+1 from (xk,xk−1) using the parity of t:

- - Define the nonlinear map φ(z) = (z + 1)2. (You may reduce intermediate values modulo M at any time.)
- - Odd step (t odd): xk+1 ≡ Axk + Bxk−1 + C φ(xk) (mod M).
- - Even step (t even): xk+1 ≡ Axk − Bxk−1 + C φ(xk−1) (mod M).

For clarity, the first two updates are:

- t = 1 : x2 ≡ Ax1 + Bx0 + C φ(x1) (mod M),
- t = 2 : x3 ≡ Ax2 − Bx1 + C φ(x1) (mod M).

Apply exactly n − 1 updates starting from x0,x1 to reach xn(n = n), and **return xn** as a single non-negative integer in [0,M − 1]. Conventions: - All modular reductions are taken modulo M and return a non-negative remainder.

- The alternating rule depends on the **update index** t. - Output only the integer value of xn (no extra text).

### Science - Example seed question

You are modeling a **batch bioreactor** where an enzyme E converts substrate A to product B, but each catalytic event requires a recyclable **cofactor token** C (e.g., NAD∗/NADH). Let At,Bt,Ct be the nonnegative integer counts of A, B, and C **after** completing tick t. You are given fixed initial counts and a regeneration period: - A0 = args.A0, B0 = args.B0, C0 = args.C0 - Regeneration period k = k

For **each discrete tick** t = 1,2,...,n (with n = {N}), apply the following **biochemical rule order**:

- 1) **Reaction (consumes cofactor)** — if both substrate and cofactor are available: - If At−1 > 0 **and** Ct−1 > 0, then one catalytic turnover occurs:

At = At−1 − 1, Bt = Bt−1 + 1, Ct = Ct−1 − 1.

- Otherwise, no reaction this tick:

At = At−1, Bt = Bt−1, Ct = Ct−1.

- 2) **Cofactor regeneration (post-reaction)** — models an external respiratory/oxidative cycle returning the cofactor to its usable form at fixed intervals: - If t mod k = 0, then **after** the reaction stage:

Ct ← Ct + 1. All updates are integer and at most ±1 per tick (“min/+=1” granularity). **Output** the product count Bn after completing exactly n = {N} ticks (i.e., after applying the regeneration rule at tick n).

### Language - Example seed question

You are given a letter maze and a number of moves n={N}. The maze is a rectangular grid of letters G with h={H} rows and w={W} columns:

{grid_block} Start at the cell (r0, c0) = ({R0}, {C0}). Build a string S as you move:

- 1) First, write down the starting letter G[r0][c0] into S. (This is done before any moves.)
- 2) Then repeat the following exactly n={step} times (t = 1..n):

- • Let (r, c) be your current cell BEFORE moving, and let ch = G[r][c].
- • Move one step based on ch (case-insensitive):

- – If ch ∈ {a, e, i, o, u} (a vowel): move RIGHT → c ← (c + 1) mod {W}
- – Otherwise (a consonant): move DOWN → r ← (r + 1) mod {H}

- • After moving to the destination cell (r, c), append its letter G[r][c] to S.
- • Now mutate the grid based on ch (the letter you moved FROM):

- – If ch is a vowel: cyclically rotate COLUMN c upward by 1.

- (Formally, for all i: G[i][c] ← old G[(i + 1) mod H][c].)

– Otherwise (ch is a consonant): cyclically rotate ROW r left by 1.

- (Formally, for all j: G[r][j] ← old G[r][(j + 1) mod W].)

Important: The mutation happens AFTER appending G[r][c] to S, and it affects the grid used for the NEXT iteration. Indices are 0-based and the maze wraps around like a torus. Thus, after n moves, S has length n + 1 (because the starting letter was included).

Let k = {K}. Your task is to return the word W made by the LAST k letters of S (in order). Output W as a plain string.

### Code - Example seed question

You are given runnable Python 3.10 code. Execute it exactly as-is in a clean environment (no extra imports). This is a Code Execution task: run the program, do not rewrite it. The loop counter i is 0-based. Return only the value of ANSWER (no other text, no formatting). Code:

N = {N}; s = {init_state!r} def f(s, i):

if len(s) == 0:

return s L = len(s); r = (i % L) + 1; s1 = s[r:] + s[:r] trans = str.maketrans({'a':'e','e':'i','i':'o','o':'u','u':'a'}) return s1.translate(trans)

for i in range(N):

s = f(s, i) ANSWER = s

- E.1.2 POTENTIAL SHORTCUT FAILURES

A basic requirement of LORE-MONO is that the complexity of question variants increases monotonically with the variant index. However, certain seed questions may violate this requirement. For instance, if answers follow a periodic pattern (e.g., when all even-indexed variants have the answer 1 and all odd-indexed variants have the answer 0), a model could exploit prior patterns to guess the correct answer without performing the intended computation. To ensure benchmark reliability, we manually reviewed all variants and excluded those exhibiting periodic answer patterns.

- E.2 ADDITIONAL RESULTS OF LORE-MONO

- E.2.1 ADDITIONAL VISUALIZATION RESULTS Visualizations of monotonicity results for the remaining 5 LRMs are provided in Fig. 8.

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Thinkless-1.5B

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

AdaptThink-7B

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

DeepSeek-R1-Distill-Llama-8B

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

DeepSeek-R1-Distill-Qwen-7B

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

𝐿𝑜𝑔(𝐴 (𝑥))

𝐶 (𝑥)

Phi-4-mini-reasoning

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

DeepSeek-R1-Distill-Qwen-14B

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Nemotron-14B

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Qwen3-Next-80B-A3B-Thinking

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

Sky-T1-32B-Preview

### Figure 8: Visualizations of Monotonicity Results on Remaining LRMs.

- E.2.2 "FAIL TO THINK" CASE STUDY

We observe that the weakest model, DeepSeek-R1-Distill-Qwen-1.5B, occasionally produces no reasoning tokens on certain code questions. For example, on variant 1 of a seed code question, the generated output is simply <think>\n</think>, corresponding to a reasoning length of 0.

"Fail to think" - Example model output

<think>\n</think>\n\n To solve this problem, we need to simulate the execution of a given Python code and determine the final value assigned to the variable ‘ANSWER‘. The code involves string manipulation and state transitions, which we will simulate step by step.\n\n### Approach\n1. **Initialization**:\n - We start with ‘N = 1‘ iteration, ‘s = \"01938\"‘, and a function ‘f‘ that processes the string ‘s‘.\n\n2. **Function Definition**:\n - The function ‘f‘ takes a string ‘s‘ and processes each character based on its position in the string. The position determines whether the count of ’0’, ’1’, ’2’, or ’3’ is incremented modulo 5.\n\n3. **Processing Each Character**:\n - For each character in ‘s‘, we count the occurrences of ’0’, ’1’, ’2’, and ’3’..... (omitted ...)

- E.3 ADDITIONAL DETAILS OF LORE-COMPO

For all composite questions, a fixed connector links the sub-questions, ensuring they are answered in sequential order. An illustrative example is provided below.

Example composite question Answer the following questions in order:

- Q1. Jim and Martha are standing together at the corner of a rectangular field. Jim walks diagonally across the field. Martha gets to the same location by walking along its length and width. The field is 300 feet wide and 400 feet long. How many feet less than Martha does Jim walk?
- Q2. Find all values of x that satisfy the equation x =√11 − 2x + 4.

- E.4 ADDITIONAL RESULTS OF LORE-COMPO Visualizations of compositionality results for the remaining LRMs are provided in Fig. 9.

- F EXPERIMENTAL DETAILS AND ADDITIONAL RESULTS

- F.1 IMPLEMENTATION DETAILS

Since DeepScaler does not come with predefined categories, we first annotate each question using GPT-4.1-mini to assign it to one of the following categories: Algebra (Prealgebra), Counting & Probability, Geometry, Number Theory, or Calculus (Precalculus). Based on these annotations, we construct sub-question and composite-question triplets by pairing questions from different categories.

For SFT, we perform a grid search over learning rates in {1e-6, 5e-6, 5e-5}, using a batch size of 8, gradient accumulation of 2, and a warmup ratio of 0.

- F.2 ADDITIONAL EXPERIMENTAL RESULTS

In Fig. 10, we further compare the reasoning compute compositionality of DeepSeek-R1-DistillQwen-7B and DeepSeek-R1-Distill-Llama-8B before and after SFT-Compo. With SFT-Compo, the nMAD decreases, and the results align more closely with the y = x line compared to their base counterparts.

###### ×104 Thinkless-1.5B

###### ×104 Phi-4-mini

###### ×104 DeepSeek-R1-7B

###### ×104 AdaptThink-7B

2.5

2.5

| | |y = x| | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | |nM|AD =|0.3|39|
| | | | | | |

| | | | |
|---|---|---|---|
| |y = x| | |
| | | | |
| | | | |
| | | | |
| |n|MAD = 0|.322|
| | | | |

| | | | |
|---|---|---|---|
| |y = x| | |
| | | | |
| | | | |
| | | | |
| |n|MAD = 0.|337|
| | | | |

| |y|= x| | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | |nMA|D = 0|.327|
| | | | | |

2.0

2.0

1.5

2.0

1.5

1.5

1.5

()Cx12

()Cx12

()Cx12

()Cx12

1.0

1.0

1.0

1.0

0.5

0.5

0.5

0.5

0.0

0.0

0.0

0.0

0.0 0.5 1.0 1.5 2.0

0 1 2

0 1 2

0.0 0.5 1.0 1.5

C (x1) + C (x2) ×104

C (x1) + C (x2) ×104

C (x1) + C (x2) ×104

C (x1) + C (x2) ×104

DeepSeek-R1-Distill-Qwen-14B

Nemotron-14B

Sky-T1-32B

Qwen3-80B-Next

×104

×104

×104

×104

2.0

| |y|= x| | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | |nMAD|= 0.36|8|
| | | | | |

| |y|= x| | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | |nMAD|= 0|.393|
| | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | |y = x| |
| | | | |
| |nMA|D = 0.392| |
| | | | |

| |y = x| | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| |nM|AD = 0.4|11|
| | | | |

1.5

2.0

- 0
- 1
- 2
- 3

1.5

1.5

()Cx12

()Cx12

()Cx12

()Cx12

1.0

1.0

1.0

0.5

0.5

0.5

0.0

0.0

0.0

0.0 0.5 1.0 1.5

0 1 2 3

0 1 2

0 1 2

C (x1) + C (x2) ×104

C (x1) + C (x2) ×104

C (x1) + C (x2) ×104

C (x1) + C (x2) ×104

### Figure 9: Visualizations of Reasoning Compute Compositionality on More LRMs.

DeepSeek-R1-7B

SFT-Compo-7B (Ours)

DeepSeek-R1-8B

SFT-Compo-8B (Ours)

×104

×104

×104

×104

2.5

- 0
- 1
- 2
- 3

| | | | |
|---|---|---|---|
| |y = x| | |
| | | | |
| | | | |
| | | | |
| |nM|AD = 0.|337|
| | | | |

| |y = x| | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| |nMA|D = 0.31|7|
| | | | |

| | | | | |
|---|---|---|---|---|
| |y =|x| | |
| | | | | |
| |n|MAD =|0.423| |
| | | | | |

| |y = x| | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| |nM|AD = 0.3|28|
| | | | |

2.0

2.0

2.0

1.5

1.5

1.5

()Cx12

()Cx12

()Cx12

()Cx12

1.0

1.0

1.0

0.5

0.5

0.5

0.0

0.0

0.0

0 1 2

0 1 2

0 1 2 3

0 1 2

C (x1) + C (x2) ×104

C (x1) + C (x2) ×104

C (x1) + C (x2) ×104

C (x1) + C (x2) ×104

### Figure 10: Visualizations of Reasoning Compute Compositionality on DeepSeek-R1-DistillQwen-7B and DeepSeek-R1-Distill-Llama-8B.

- G QUALITATIVE EXAMPLES OF LORE-COMPO

We present an example generated of DeepSeek-Distill-1.5B and after SFT-Compo. For DeepSeekDistill-1.5B, we report metrics for the question pair and the composite question with accuracies Aθ(x1) = 1.0, Aθ(x2) = 1.0, Aθ(x12) = 0.25. Reasoning compute is Cθ(x1) = 2177.25, Cθ(x2) = 3546.50, and Cθ(x12) = 732.13, showing that the model severely underthinks on the composite question and compromises the accuracy. After SFT-Compo, the composite accuracy improves to Aθ˜(x12) = 0.875 with reasoning compute as Cθ˜(x12) = 4300.00. We provide detailed reasoning processes of the two models below.

Example composite question Q: Answer the following questions in order:

- Q1. Pat is to select six cookies from a tray containing only chocolate chip, oatmeal, and peanut butter cookies. There are at least six of each of these three kinds of cookies on the tray. How many different assortments of six cookies can be selected? (Note that cookies of the same type are not distinguishable.)
- Q2. Let f(x) = x10 + 5x9 − 8x8 + 7x7 − x6 − 12x5 + 4x4 − 8x3 + 12x2 − 5x − 5. Without using long division (which would be horribly nasty!), find the remainder when f(x) is divided by x2 − 1.

### Example reasoning process (DeepSeek-R1-1.5B)

- Q1. To determine the number of different assortments of six cookies, I need to consider the number of ways to select six cookies from three types: chocolate chip, oatmeal, and peanut butter. Since the cookies of the same type are indistinguishable, the order of selection doesn’t matter.

I can model this problem using combinations with repetition. The formula for the number of ways to choose k items from n types is given by:

n + k − 1 k

In this case, n = 3 (types of cookies) and k = 6 (cookies to select). Plugging in the values:

3 + 6 − 1 6

=

8 6

Calculating 86 gives:

8 6

=

8! 6!(8 − 6)!

= 28 [Correct Answer]

Therefore, there are 28 different assortments of six cookies that can be selected.

- Q2. To find the remainder when f(x) is divided by x2 − 1 without using long division, I can utilize the Remainder Theorem. The Remainder Theorem states that the remainder of a polynomial f(x) divided by x − a is f(a). However, since the divisor here is x2 − 1, which factors into (x−1)(x+1), I need to evaluate f(x) at the roots of x2−1, which are x = 1 and x = −1.

The remainder will be a linear polynomial of the form R(x) = ax + b. To find a and b, I can set up two equations by plugging in x = 1 and x = −1 into f(x): 1. f(1) = a(1) + b 2. f(−1) = a(−1) + b

Calculating f(1): f(1) = 110 + 5(1)9 − 8(1)8 + 7(1)7 − 1(1)6 − 12(1)5 + 4(1)4 − 8(1)3 + 12(1)2 − 5(1) − 5 f(1) = 1 + 5 − 8 + 7 − 1 − 12 + 4 − 8 + 12 − 5 − 5 f(1) = −1 Calculating f(−1):

f(−1) = (−1)10 + 5(−1)9 − 8(−1)8 + 7(−1)7 − 1(−1)6 − 12(−1)5 + 4(−1)4

−8(−1)3 + 12(−1)2 − 5(−1) − 5 f(−1) = 1 − 5 − 8 − 7 − 1 + 12 + 4 + 8 + 12 + 5 − 5 f(−1) = 13

Now, setting up the equations: 1. a + b = −1 2. −a + b = 13 Solving these equations: Adding both equations: 2b = 12 => b = 6 Substituting b = 6 into the first equation: a + 6 = −1 => a = −7

Thus, the remainder R(x) is:

R(x) = −7x + 6 [Incorrect Answer] </think>

Example reasoning process (DeepSeek-R1-1.5B SFT-Compo) Okay, so I have these two questions to solve. Let me start with the first one.

- **Q1. Pat is to select six cookies from a tray containing only chocolate chip, oatmeal, and peanut butter cookies. There are at least six of each of these three kinds of cookies on the tray. How many different assortments of six cookies can be selected? (Note that cookies of the same type are not distinguishable.)**

Hmm, okay. So Pat is selecting six cookies, and the tray has three types: chocolate chip, oatmeal, and peanut butter. Each type has at least six cookies, so there’s no shortage. Since the cookies of the same type are indistinct, we’re dealing with combinations with repetition, I think.

I remember that the formula for combinations with repetition is C(n + k - 1, k), where n is the number of types and k is the number of items to choose. In this case, n is 3 (since there are three types of cookies) and k is 6 (since Pat is selecting six cookies). So plugging into the formula, it should be C(3 + 6 - 1, 6) = C(8, 6).

Wait, let me double-check. So, combinations with repetition: if order doesn’t matter and items are indistinct except for type, yes, that’s the case here. So, the formula is indeed C(n + k 1, k). So, n=3, k=6, so 3 + 6 -1 = 8, so C(8,6). Alternatively, C(8,2) since combinations are symmetric, which is 28. So is that the answer?

But wait, let me think again. Since each type has at least six cookies, and we’re selecting six, does that affect anything? Because if each type has at least six, then when we select six, the maximum number we can take from any one type is six, but since we’re only selecting six, the maximum from any type is six, but since we’re selecting six in total, the maximum from any type is six, but the minimum is zero.

But in this case, since we have at least six of each, so when we choose six, the number of ways is the same as the number of non-negative integer solutions to the equation:

x1 + x2 + x3 = 6, where x1, x2, x3 are the number of chocolate chip, oatmeal, and peanut butter cookies respectively.

Which is exactly the stars and bars problem, so C(6 + 3 -1, 3 -1) = C(8,2) = 28. So yeah, the answer is 28. That seems right.

- **Q2. Let f(x) = x10 + 5x9 − 8x8 + 7x7 − x6 − 12x5 + 4x4 − 8x3 + 12x2 − 5x − 5.**

Without using long division (which would be horribly nasty!), find the remainder when f(x) is divided by x2 − 1. Okay, so polynomial division. But f(x) is a degree 10 polynomial, and we’re dividing by x2 − 1, which is also degree 2. So the remainder should be of degree less than 2, so it should be a linear polynomial, right? So it can be written as ax + b, where a and b are constants.

To find a and b, we can use the Remainder Theorem. Since x2 − 1 factors as (x − 1)(x + 1), the remainder when f(x) is divided by x2 − 1 is the same as the remainder when f(1) and f(-1) are computed, and then we can set up equations to solve for a and b.

Wait, let me recall: if we have f(x) divided by (x − c)(x − d), the remainder is R(x) = e x + f. So, to find R(x), we can evaluate f(c) and f(d), which gives us two equations: f(c) = ec + f f(d) = ed + f

Then, solving these two equations gives us e and f. In our case, the divisor is x2 − 1, which is (x − 1)(x + 1). So, c =1 and d = -1.

Therefore, we can compute f(1) and f(-1), set up the equations: f(1) = a(1) + b = a + b f(−1) = a(−1) + b = −a + b Then, solving these two equations will give us a and b. So, let me compute f(1) and f(-1). First, f(1): f(1) = (1)10 + 5 ∗ (1)9 − 8 ∗ (1)8 + 7 ∗ (1)7 − (1)6 − 12 ∗ (1)5 + 4 ∗ (1)4 − 8 ∗ (1)3 + 12 ∗

(1)2 − 5 ∗ (1) − 5 Simplify each term: 1 + 5 − 8 + 7 − 1 − 12 + 4 − 8 + 12 − 5 − 5 Let me compute step by step: Start with 1. 1 + 5 = 6 6 − 8 = −2 − 2 + 7 = 5 5 − 1 = 4 4 − 12 = −8 − 8 + 4 = −4 −4 − 8 = −12 − 12 + 12 = 0 0 − 5 = −5 − 5 − 5 = −10 So f(1) = −10. Similarly, compute f(−1): f(−1) = (−1)10 + 5 ∗ (−1)9 − 8 ∗ (−1)8 + 7 ∗ (−1)7 − (−1)6 − 12 ∗ (−1)5 + 4 ∗ (−1)4 − 8 ∗ (−1)3 + 12 ∗ (−1)2 − 5 ∗ (−1) − 5 Simplify each term: (−1)10 = 1 5 ∗ (−1)9 = 5 ∗ (−1) = −5 − 8 ∗ (−1)8 = −8 ∗ (1) = −8 7 ∗ (−1)7 = 7 ∗ (−1) = −7 − (−1)6 = −1 − 12 ∗ (−1)5 = −12 ∗ (−1) = 12 4 ∗ (−1)4 = 4 ∗ (1) = 4 − 8 ∗ (−1)3 = −8 ∗ (−1) = 8 12 ∗ (−1)2 = 12 ∗ (1) = 12 − 5 ∗ (−1) = 5

-5 remains as is. So let’s write all terms: 1 − 5 − 8 − 7 − 1 + 12 + 4 + 8 + 12 + 5 − 5 Let me compute step by step: Start with 1. 1 − 5 = −4 − 4 − 8 = −12 − 12 − 7 = −19 = −19 − 1 = −20 −20+12 = −8 −8+4 = −4 −4+8 = 4 4+12 = 16 16+5 = 21 21−5 = 16 So f(−1) = 16. Therefore, we have: f(1) = a + b = −10 f(−1) = −a + b = 16 So now, we have the system of equations: 1) a + b = −10 2) −a + b = 16. Let me solve this. Add equations 1 and 2: (a + b) + (−a + b) = (−10) + 16 Simplify: 0a + 2b = 6 So 2b = 6 => b = 3. Then, plug b=3 into equation 1: a + 3 = −10 => a = −13. So the remainder is ax + b = −13x + 3

Therefore, the remainder when f(x) is divided by x2 − 1 is −13x + 3. Let me double-check my computations for f(1) and f(-1), just to make sure I didn’t make any arithmetic mistakes.

**Recomputing f(1):** 1 + 5 = 6 6 − 8 = −2 − 2 + 7 = 5 5 − 1 = 4 4 − 12 = −8 − 8 + 4 = −4 −4 − 8 = −12 − 12 + 12 = 0 0 − 5 = −5 − 5 − 5 = −10

Yes, that’s correct. **Recomputing f(−1):** 1−5 = −4 −4−8 = −12 −12−7 = −19 −19−1 = −20 −20+12 = −8 −8+4 = −4 −4+8 = 4 4+12 = 16 16 + 5 = 21 21 − 5 = 16 Yes, that’s correct.

So the equations are: a + b = −10 − a + b = 16

Adding them gives 2b = 6 => b = 3, then a= -13. So the remainder is -13x +3. I think that’s correct.

**Final Answer**

- Q1:

|28|
|---|

[Correct Answer]

- Q2:

|−13x + 3|
|---|

[Correct Answer] </think>

